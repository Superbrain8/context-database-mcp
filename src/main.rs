//! Context Database MCP server.
//!
//! One process runs per MCP client. Isolation (`client_id`, `namespace`) comes
//! from this process's environment, never from tool arguments -- the model
//! cannot address another client's memory even if it tries.

mod db;
mod embed;

use std::sync::Arc;

use rmcp::{
    handler::server::wrapper::Parameters, schemars, tool, tool_handler, tool_router,
    transport::stdio, ServiceExt,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use embed::Embedder;

#[derive(Debug, Clone)]
pub struct Scope {
    pub client_id: String,
    pub namespace: String,
    pub session_id: String,
}

#[derive(Clone)]
struct ContextDb {
    pool: PgPool,
    embedder: Arc<Embedder>,
    scope: Arc<Scope>,
    /// Cosine-distance cutoff for the dense ranker. Measured with bge-m3 on this
    /// project's own notes: a paraphrase of a stored note sits near 0.33, an
    /// unrelated note near 0.66. 0.55 keeps paraphrases and drops strangers.
    max_distance: f64,
}

// ---------------------------------------------------------------- tool params

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SaveParams {
    /// Short, searchable headline. Write it the way you would later search for
    /// it, e.g. "auth middleware rejects tokens issued in the same second".
    title: String,
    /// The content to store. Self-contained: it will be read months later
    /// without the surrounding conversation.
    body: String,
    /// One of: note, decision, error, snippet, file-summary, todo, fact.
    #[serde(default)]
    kind: Option<String>,
    /// Lowercase keywords for filtering, e.g. ["auth", "postgres"].
    #[serde(default)]
    tags: Option<Vec<String>>,
    /// Id of a memory this one corrects or replaces. The old memory stops
    /// appearing in search. Use this instead of trying to edit -- memories are
    /// append-only.
    #[serde(default)]
    supersedes: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SearchParams {
    /// Natural language question or keywords. Both are handled -- retrieval
    /// fuses semantic and exact-token matching.
    query: String,
    /// Number of results. Default 8.
    #[serde(default)]
    limit: Option<i64>,
    /// Restrict to a single kind.
    #[serde(default)]
    kind: Option<String>,
    /// Only return memories carrying all of these tags.
    #[serde(default)]
    tags: Option<Vec<String>>,
    /// Widen the search to every project this client has memories for, instead
    /// of only the current one. Use when the question is about a technique,
    /// tool or mistake that is not specific to this codebase -- e.g. "have I hit
    /// this error before?". Results from elsewhere are marked with their origin.
    #[serde(default)]
    cross_project: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct GetParams {
    /// Memory id from a context_search result.
    id: i64,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ForgetParams {
    /// Memory id to forget.
    id: i64,
    /// Why it is being forgotten. Kept for audit.
    #[serde(default)]
    reason: Option<String>,
}

// ---------------------------------------------------------------------- tools

#[tool_router]
impl ContextDb {
    /// Descriptions are written as triggers ("call this WHEN...") rather than
    /// as nouns. A tool the model never thinks to call is a tool that does not
    /// exist, and offloading is worthless without reliable recall.
    #[tool(
        description = "Store information in long-term memory so it can be dropped from the \
                       current context and retrieved later. Call this WHEN you learn something \
                       durable: a decision and its reasoning, a non-obvious constraint, the root \
                       cause of a bug, a user preference, or a summary of a large file you just \
                       read. Do NOT store things trivially re-derivable from the code. Returns \
                       the new memory id."
    )]
    async fn context_save(&self, Parameters(p): Parameters<SaveParams>) -> String {
        let kind = p.kind.as_deref().unwrap_or("note");
        let tags = p.tags.unwrap_or_default();

        // Long bodies are split so each vector covers one coherent passage; a
        // short body yields a single chunk. The title is prepended to every
        // chunk, so a chunk from the middle of a long note still carries what
        // the note is about.
        let pieces = embed::chunk_text(&p.body);
        if pieces.is_empty() {
            return "ERROR: body is empty".to_string();
        }
        let inputs: Vec<String> = pieces
            .iter()
            .map(|c| format!("{}\n\n{}", p.title, c))
            .collect();

        // One request for all chunks -- the embedding server batches far better
        // than a round trip per chunk.
        let embeddings = match self.embedder.embed(&inputs).await {
            Ok(v) => v,
            Err(e) => return format!("ERROR: embedding failed: {e:#}"),
        };
        if embeddings.len() != pieces.len() {
            return format!(
                "ERROR: embedding server returned {} vectors for {} chunks",
                embeddings.len(),
                pieces.len()
            );
        }
        let chunks: Vec<(String, Vec<f32>)> = pieces.into_iter().zip(embeddings).collect();
        let chunk_count = chunks.len();

        match db::save(
            &self.pool,
            &self.scope,
            &p.title,
            &p.body,
            kind,
            &tags,
            chunks,
            self.embedder.model(),
            p.supersedes,
        )
        .await
        {
            Ok(id) => {
                let split = if chunk_count > 1 {
                    format!(" in {chunk_count} chunks")
                } else {
                    String::new()
                };
                match p.supersedes {
                    Some(old) => format!(
                        "saved id={id}{split} (supersedes id={old}, which is now retired)"
                    ),
                    None => format!("saved id={id}{split}"),
                }
            }
            Err(e) => format!("ERROR: save failed: {e:#}"),
        }
    }

    #[tool(
        description = "Search long-term memory. Call this BEFORE answering any question that \
                       refers to earlier work, past decisions, previous errors, or anything the \
                       user implies you should already know -- and before concluding you lack \
                       information. Returns ranked snippets with ids; call context_get for the \
                       full text of the ones that look relevant."
    )]
    async fn context_search(&self, Parameters(p): Parameters<SearchParams>) -> String {
        let limit = p.limit.unwrap_or(8).clamp(1, 50);
        let tags = p.tags.unwrap_or_default();

        let embedding = match self.embedder.embed_one(&p.query).await {
            Ok(v) => v,
            Err(e) => return format!("ERROR: embedding failed: {e:#}"),
        };

        let hits = match db::search(
            &self.pool,
            &self.scope,
            &p.query,
            embedding,
            limit,
            p.kind.as_deref(),
            &tags,
            self.max_distance,
            p.cross_project.unwrap_or(false),
        )
        .await
        {
            Ok(h) => h,
            Err(e) => return format!("ERROR: search failed: {e:#}"),
        };

        if hits.is_empty() {
            return "no matching memories".to_string();
        }

        // Snippets only. Returning full bodies here would re-inflate the context
        // window and defeat the point of offloading in the first place.
        let mut out = String::new();
        for h in hits {
            // Only label the origin when it is not the current project, so the
            // common case stays quiet and a foreign memory stands out.
            let origin = if h.namespace == self.scope.namespace {
                String::new()
            } else {
                format!(" | from project {}", h.namespace)
            };
            out.push_str(&format!(
                "[id={}] ({}) {} | tags={:?} | {}{} | score={:.4}\n  {}\n",
                h.id,
                h.kind,
                h.title,
                h.tags,
                h.created_at.format("%Y-%m-%d"),
                origin,
                h.score,
                h.snippet.replace('\n', " "),
            ));
        }
        out
    }

    #[tool(
        description = "Retrieve the full body of one memory by id, after context_search showed \
                       its snippet was relevant."
    )]
    async fn context_get(&self, Parameters(p): Parameters<GetParams>) -> String {
        match db::get(&self.pool, &self.scope, p.id).await {
            Ok(Some(m)) => {
                let supersedes = m
                    .supersedes_id
                    .map(|s| format!(" (supersedes id={s})"))
                    .unwrap_or_default();
                let origin = if m.namespace == self.scope.namespace {
                    String::new()
                } else {
                    format!("  from project {}", m.namespace)
                };
                format!(
                    "id={} ({}) {}{}\ntags={:?}  created={}{}\n\n{}",
                    m.id,
                    m.kind,
                    m.title,
                    supersedes,
                    m.tags,
                    m.created_at.to_rfc3339(),
                    origin,
                    m.body
                )
            }
            Ok(None) => format!("no memory with id={} in this scope", p.id),
            Err(e) => format!("ERROR: get failed: {e:#}"),
        }
    }

    #[tool(
        description = "Forget a memory that is wrong or obsolete. Prefer context_save with \
                       `supersedes` when you have a corrected version -- use forget only when \
                       nothing should replace it. This is a soft delete and is recoverable."
    )]
    async fn context_forget(&self, Parameters(p): Parameters<ForgetParams>) -> String {
        match db::forget(&self.pool, &self.scope, p.id, p.reason.as_deref()).await {
            Ok(true) => format!("forgot id={}", p.id),
            Ok(false) => format!("no live memory with id={} in this scope", p.id),
            Err(e) => format!("ERROR: forget failed: {e:#}"),
        }
    }
}

/// Hand-written rather than using `#[tool_router(server_handler)]`: that
/// shortcut reports the SDK's own identity ("rmcp" / "3.1.0") in the initialize
/// handshake, and it gives no way to set `instructions`.
///
/// The instructions are load-bearing. Tool descriptions are only consulted once
/// the model is already considering a tool; this text is in front of it the
/// whole session, and is the main lever on whether offloading actually happens.
#[tool_handler]
impl rmcp::ServerHandler for ContextDb {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        // ServerInfo is #[non_exhaustive], so it has to be built by mutating a
        // Default rather than with a struct literal.
        let mut info = rmcp::model::ServerInfo::default();
        // NOT Implementation::from_build_env(): that reads CARGO_PKG_* as of
        // rmcp's own compilation, so it reports "rmcp"/"3.1.0". env!() here
        // expands against this crate.
        info.server_info.name = env!("CARGO_PKG_NAME").to_string();
        info.server_info.version = env!("CARGO_PKG_VERSION").to_string();
        info.capabilities = rmcp::model::ServerCapabilities::builder()
            .enable_tools()
            .build();
        info.instructions = Some(
            "Long-term memory for this project, backed by Postgres. It persists across \
             sessions, so it holds things you cannot otherwise recall.\n\n\
             Search it before answering anything that refers to earlier work, past \
             decisions, previous bugs, or that the user speaks about as already settled -- \
             and before saying you lack information.\n\n\
             Save to it when you learn something durable: a decision and why, a constraint \
             that is not obvious from the code, the root cause of a bug, a user preference. \
             Do not save what is trivially re-derivable by reading the code.\n\n\
             Memories are append-only. To correct one, save a new memory with `supersedes` \
             set to the old id rather than trying to edit it."
                .to_string(),
        );
        info
    }
}

// ----------------------------------------------------------------------- main

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Body of `--recent`. A session-start hook must never break the session, so
/// every failure here is swallowed: the process prints nothing and exits 0
/// rather than surfacing a database error into the transcript.
async fn print_recent(database_url: &str, scope: &Scope, limit: i64) -> anyhow::Result<()> {
    let pool = match db::connect(database_url).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("recent: database unreachable: {e:#}");
            return Ok(());
        }
    };

    let hits = match db::recent(&pool, scope, limit).await {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("recent: query failed: {e:#}");
            return Ok(());
        }
    };

    if hits.is_empty() {
        return Ok(());
    }

    // Deliberately titles + ids only, no bodies. The point is to make the model
    // aware these memories exist so it can pull the ones it wants with
    // context_get; dumping bodies here would spend the context this is meant to
    // save, on every single session start.
    println!(
        "Long-term memory for this project holds {} recent item(s). \
         Use context_get with an id to read one, or context_search to look for others.",
        hits.len()
    );
    for h in hits {
        println!(
            "  [id={}] ({}) {} | tags={:?} | {}",
            h.id,
            h.kind,
            h.title,
            h.tags,
            h.created_at.format("%Y-%m-%d")
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // stdout is the MCP transport. Any stray byte written there corrupts the
    // protocol framing, so all logging goes to stderr.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("CTXDB_LOG")
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let scope = Scope {
        client_id: env_or("CTXDB_CLIENT_ID", "unknown-client"),
        namespace: env_or("CTXDB_NAMESPACE", "default"),
        session_id: uuid::Uuid::new_v4().to_string(),
    };

    let database_url = env_or(
        "CTXDB_DATABASE_URL",
        "postgres://ctx:ctx@127.0.0.1:5433/ctxdb",
    );
    let embed_url = env_or("CTXDB_EMBED_URL", "http://127.0.0.1:8085");
    let embed_model = env_or("CTXDB_EMBED_MODEL", "BAAI/bge-m3");

    // `--recent [N]`: print pinned + newest memories and exit, for the
    // SessionStart hook. Pull, on its own, is not enough -- if the model never
    // thinks to search, nothing is ever recalled. This is the push half.
    //
    // No embedder is touched on this path, so the hook still works when TEI is
    // down or still loading its weights.
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--recent") {
        let limit = args
            .get(pos + 1)
            .and_then(|n| n.parse::<i64>().ok())
            .unwrap_or(5)
            .clamp(1, 50);
        return print_recent(&database_url, &scope, limit).await;
    }

    tracing::info!(
        client_id = %scope.client_id,
        namespace = %scope.namespace,
        "starting context-database mcp server"
    );

    let pool = db::connect(&database_url).await?;
    let embedder = Embedder::new(embed_url, embed_model);

    // Fail loudly at startup rather than on the first tool call, where the model
    // would see the error and probably just give up on the tool.
    embedder.ping().await?;
    tracing::info!("postgres and embedder reachable");

    let max_distance = env_or("CTXDB_MAX_DISTANCE", "0.55")
        .parse::<f64>()
        .unwrap_or(0.55);

    let server = ContextDb {
        pool,
        embedder: Arc::new(embedder),
        scope: Arc::new(scope),
        max_distance,
    };

    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
