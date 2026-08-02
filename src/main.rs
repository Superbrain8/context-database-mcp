//! Context Database MCP server.
//!
//! One process runs per MCP client. Isolation (`client_id`, `namespace`) comes
//! from this process's environment, never from tool arguments -- the model
//! cannot address another client's memory even if it tries.

mod admin;
mod consolidate;
mod db;
mod embed;
mod ingest;
mod reindex;

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
    /// Id, or list of ids, that this memory corrects or replaces. Those memories
    /// stop appearing in search. Use this instead of trying to edit -- memories
    /// are append-only. Pass several ids to consolidate overlapping memories
    /// into one merged memory.
    #[serde(default)]
    supersedes: Option<Supersedes>,
}

/// One id or several.
///
/// Untagged rather than plain `Vec<i64>`: correcting a single memory is by far
/// the common call, `supersedes: 42` is what a model writes unprompted, and
/// rejecting it as a type error would cost a retry on the most frequent path.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum Supersedes {
    One(i64),
    Many(Vec<i64>),
}

impl Supersedes {
    /// Deduplicated, because `[7, 7]` would otherwise report one retired id out
    /// of two requested and read as a failure.
    fn ids(&self) -> Vec<i64> {
        let mut ids = match self {
            Supersedes::One(id) => vec![*id],
            Supersedes::Many(v) => v.clone(),
        };
        ids.sort_unstable();
        ids.dedup();
        ids
    }
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
        // short body yields a single chunk. Sized by the embedding model's own
        // tokenizer, so this costs a round trip before the embedding one.
        let (pieces, inputs) = match self.embedder.chunk_inputs(&p.title, &p.body).await {
            Ok(v) => v,
            Err(e) => return format!("ERROR: chunking failed: {e:#}"),
        };
        if pieces.is_empty() {
            return "ERROR: body is empty".to_string();
        }

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

        let asked: Vec<i64> = p
            .supersedes
            .as_ref()
            .map(Supersedes::ids)
            .unwrap_or_default();

        match db::save(
            &self.pool,
            &self.scope,
            &p.title,
            &p.body,
            kind,
            &tags,
            chunks,
            self.embedder.model(),
            &asked,
        )
        .await
        {
            Ok(saved) => {
                let id = saved.id;
                let split = if chunk_count > 1 {
                    format!(" in {chunk_count} chunks")
                } else {
                    String::new()
                };
                if asked.is_empty() {
                    return format!("saved id={id}{split}");
                }

                // A requested id that was not retired is reported, never
                // swallowed: it means the id belongs to another project or was
                // already superseded, and a merge that silently leaves one of
                // its originals in search looks like it worked.
                let missed: Vec<String> = asked
                    .iter()
                    .filter(|id| !saved.retired.contains(id))
                    .map(i64::to_string)
                    .collect();
                let retired = saved
                    .retired
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let mut out = if saved.retired.is_empty() {
                    format!("saved id={id}{split} (nothing was retired)")
                } else {
                    format!("saved id={id}{split} (retired id={retired})")
                };
                if !missed.is_empty() {
                    out.push_str(&format!(
                        " -- id={} left alone: not in this project, or already superseded",
                        missed.join(", ")
                    ));
                }
                out
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
                let supersedes = if m.supersedes.is_empty() {
                    String::new()
                } else {
                    format!(
                        " (supersedes id={})",
                        m.supersedes
                            .iter()
                            .map(i64::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                };
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
             set to the old id rather than trying to edit it, or to several ids to replace a \
             group of overlapping memories with one merged memory."
                .to_string(),
        );
        info
    }
}

// ----------------------------------------------------------------------- main

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Which project's memories this process may touch.
///
/// `CTXDB_NAMESPACE` wins when set. Otherwise the working directory's name is
/// used, which is what makes a single user-scope MCP registration usable across
/// every project: clients launch the server with the project as its working
/// directory, so the namespace follows the project without per-project config.
///
/// The trade-off is that the namespace is the *folder name*, not the full path,
/// so two checkouts named `api` in different parents share one namespace. Set
/// `CTXDB_NAMESPACE` explicitly to separate them.
fn namespace_from_dir(dir: &std::path::Path) -> Option<String> {
    dir.file_name()
        .map(|n| n.to_string_lossy().trim().to_string())
        .filter(|n| !n.is_empty())
}

fn resolve_namespace() -> String {
    if let Ok(ns) = std::env::var("CTXDB_NAMESPACE") {
        let ns = ns.trim();
        if !ns.is_empty() {
            return ns.to_string();
        }
    }

    let derived = std::env::current_dir()
        .ok()
        .and_then(|p| namespace_from_dir(&p))
        .unwrap_or_else(|| "default".to_string());

    // Logged loudly: a silently wrong namespace is the worst failure this thing
    // has, because saves still succeed and simply become invisible.
    tracing::info!(
        namespace = %derived,
        "CTXDB_NAMESPACE not set, derived namespace from working directory"
    );
    derived
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

    // Session summaries are kept out of the ranked list and offered as a single
    // separate line -- see db::recent for why. A failure here must not cost the
    // list that did load.
    let summary = match db::latest_session_summary(&pool, scope).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("recent: session summary lookup failed: {e:#}");
            None
        }
    };

    if hits.is_empty() && summary.is_none() {
        return Ok(());
    }

    // Deliberately titles + ids only, no bodies. The point is to make the model
    // aware these memories exist so it can pull the ones it wants with
    // context_get; dumping bodies here would spend the context this is meant to
    // save, on every single session start.
    if !hits.is_empty() {
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
    }

    // Phrased as a condition rather than an instruction: after /clear this is
    // the only trace of the session that was discarded, but on a session that
    // starts fresh work it is last month's news and reading it would waste the
    // context this whole mechanism exists to protect.
    if let Some(s) = summary {
        println!(
            "The previous session in this project was compacted on {}; its summary is [id={}]. \
             Read it with context_get only if this session continues that work.",
            // Local, matching the timestamp --ingest puts in the title. The
            // column is timestamptz, so formatting it directly prints UTC and
            // the same summary would show two different times.
            s.created_at
                .with_timezone(&chrono::Local)
                .format("%Y-%m-%d %H:%M"),
            s.id
        );
    }
    Ok(())
}

/// The value following `flag`, if there is one and it is not itself a flag.
///
/// Rejecting a `--`-prefixed value matters: `--restore --detach` would otherwise
/// read "--detach" as the id, fail to parse, and report a confusing error about
/// the wrong argument.
fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let pos = args.iter().position(|a| a == flag)?;
    args.get(pos + 1)
        .map(String::as_str)
        .filter(|v| !v.starts_with("--"))
}

fn id_after(args: &[String], flag: &str) -> anyhow::Result<i64> {
    let raw = arg_after(args, flag)
        .ok_or_else(|| anyhow::anyhow!("{flag} needs a memory id, e.g. `{flag} 42`"))?;
    raw.parse()
        .map_err(|_| anyhow::anyhow!("{flag} takes a memory id, got {raw:?}"))
}

fn limit_after(args: &[String], flag: &str, default: i64) -> i64 {
    arg_after(args, flag)
        .and_then(|n| n.parse::<i64>().ok())
        .unwrap_or(default)
        .clamp(1, 200)
}

/// A cosine distance from the command line, clamped to the range the metric can
/// actually produce for these embeddings.
///
/// Silently clamping rather than erroring, because both ends are harmless: 0
/// finds only identical text and 1 finds everything, and the report says which
/// cutoff it used.
fn threshold_after(args: &[String], flag: &str, default: f64) -> f64 {
    arg_after(args, flag)
        .and_then(|n| n.parse::<f64>().ok())
        .filter(|d| d.is_finite())
        .unwrap_or(default)
        .clamp(0.0, 1.0)
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
        namespace: resolve_namespace(),
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

    // `--ingest`: read a PostCompact hook payload on stdin and store the
    // compaction summary. Stdout stays empty on this path -- a compaction hook
    // that prints goes straight into the freshly compacted context, which is
    // exactly the cost this mode exists to avoid.
    if args.iter().any(|a| a == "--ingest") {
        return ingest::run(&database_url, embed_url, embed_model, scope).await;
    }

    // `--reindex [--dry-run]`: recompute chunks and embeddings for every stored
    // memory. An operator command, run by hand after a chunker or model change --
    // no hook calls it, and unlike the hook paths it reports failure instead of
    // swallowing it.
    if args.iter().any(|a| a == "--reindex") {
        let dry_run = args.iter().any(|a| a == "--dry-run");
        return reindex::run(&database_url, embed_url, embed_model, &scope, dry_run).await;
    }

    // Operator modes. Kept off the MCP surface on purpose -- every tool costs
    // schema tokens in every session whether the model uses it or not, and these
    // are occasional human decisions, not conversational ones.
    if args.iter().any(|a| a == "--stale") {
        return admin::stale(&database_url, &scope, limit_after(&args, "--stale", 20)).await;
    }
    // `--consolidate [N] [--threshold D]`: clusters of overlapping memories. A
    // report like `--stale`: it cannot write the merged text -- an embedding
    // server makes vectors, not sentences -- so it hands the model the ids and
    // the tool call to make.
    if args.iter().any(|a| a == "--consolidate") {
        return consolidate::run(
            &database_url,
            &scope,
            threshold_after(&args, "--threshold", consolidate::DEFAULT_THRESHOLD),
            limit_after(&args, "--consolidate", 10),
        )
        .await;
    }
    if args.iter().any(|a| a == "--history") {
        return admin::history(&database_url, &scope, limit_after(&args, "--history", 20)).await;
    }
    if args.iter().any(|a| a == "--pin") {
        return admin::pin(&database_url, &scope, id_after(&args, "--pin")?, true).await;
    }
    if args.iter().any(|a| a == "--unpin") {
        return admin::pin(&database_url, &scope, id_after(&args, "--unpin")?, false).await;
    }
    if args.iter().any(|a| a == "--restore") {
        let detach = args.iter().any(|a| a == "--detach");
        return admin::restore(&database_url, &scope, id_after(&args, "--restore")?, detach).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn reads_the_value_after_a_flag() {
        let a = args(&["bin", "--pin", "42"]);
        assert_eq!(arg_after(&a, "--pin"), Some("42"));
        assert_eq!(id_after(&a, "--pin").unwrap(), 42);
    }

    #[test]
    fn a_following_flag_is_not_a_value() {
        // `--restore 7 --detach` and `--restore --detach` must not be confused:
        // the second is a mistake, and reading "--detach" as the id would report
        // an error about the wrong argument.
        let a = args(&["bin", "--restore", "--detach"]);
        assert_eq!(arg_after(&a, "--restore"), None);
        assert!(id_after(&a, "--restore").is_err());

        let b = args(&["bin", "--restore", "7", "--detach"]);
        assert_eq!(id_after(&b, "--restore").unwrap(), 7);
    }

    #[test]
    fn a_flag_at_the_end_has_no_value() {
        let a = args(&["bin", "--stale"]);
        assert_eq!(arg_after(&a, "--stale"), None);
        assert_eq!(limit_after(&a, "--stale", 20), 20);
    }

    #[test]
    fn limits_are_clamped_rather_than_rejected() {
        // A hand-typed `--stale 100000` should print a big report, not fail.
        let a = args(&["bin", "--stale", "100000"]);
        assert_eq!(limit_after(&a, "--stale", 20), 200);
    }

    #[test]
    fn a_threshold_is_clamped_to_the_metrics_range() {
        let at = |v: &str| threshold_after(&args(&["bin", "--threshold", v]), "--threshold", 0.22);
        assert_eq!(at("0.3"), 0.3);
        assert_eq!(at("9"), 1.0);
        // Garbage falls back to the default rather than to 0, which would report
        // no clusters and look like a clean corpus.
        assert_eq!(at("x"), 0.22);
    }

    #[test]
    fn supersedes_accepts_one_id_or_many() {
        let one: Supersedes = serde_json::from_str("42").unwrap();
        assert_eq!(one.ids(), vec![42]);
        let many: Supersedes = serde_json::from_str("[7, 3, 7]").unwrap();
        // Sorted and deduplicated: a repeated id would otherwise be reported as
        // one retired out of two asked for, which reads as a partial failure.
        assert_eq!(many.ids(), vec![3, 7]);
    }

    #[test]
    fn namespace_comes_from_the_folder_name() {
        let ns = namespace_from_dir(std::path::Path::new("/home/x/Projects/api"));
        assert_eq!(ns.as_deref(), Some("api"));
    }
}
