//! Operator modes: `--stale`, `--pin`, `--unpin`, `--history`, `--restore`,
//! `--save`.
//!
//! None of these are MCP tools, and that is the design rather than an omission.
//! Every tool exposed to the model costs schema tokens in every session's
//! context whether it is used or not, and these are decisions a person makes
//! occasionally, not decisions a model makes mid-conversation. Pinning is a
//! standing judgement about what future sessions should be told exists;
//! restoring is an undo for a mistake the model itself made; `--save` writes
//! into whatever namespace this process resolves to, which is exactly the
//! power a tool argument must never have (see the crate doc comment).
//!
//! Like `--reindex`, these fail loudly. They are run by hand, so an error is
//! read by someone who can act on it.

use anyhow::{Context, Result};
use chrono::Utc;

use crate::{db, embed::Embedder, Scope};

/// `--stale [N]`: the least-read live memories, oldest access first.
///
/// A report and nothing else. Nothing in this system deletes a memory on a
/// timer -- for a store whose whole promise is not forgetting, an automatic
/// eviction that guesses wrong fails silently and is discovered months later by
/// a search that finds nothing. This surfaces the candidates and leaves the
/// decision, and the `context_forget` call, to a person.
pub async fn stale(database_url: &str, scope: &Scope, limit: i64) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let rows = db::stale(&pool, scope, limit).await?;

    if rows.is_empty() {
        println!("no memories in namespace {}", scope.namespace);
        return Ok(());
    }

    println!(
        "{} least-read memories in {} (reads count context_get only, not search hits)",
        rows.len(),
        scope.namespace
    );

    let now = Utc::now();
    let mut never_read = 0usize;
    let mut chars_never_read = 0i64;

    for r in &rows {
        if r.access_count == 0 {
            never_read += 1;
            chars_never_read += i64::from(r.body_chars);
        }
        let age = (now - r.created_at).num_days();
        // "Never read" is the honest label for access_count 0: accessed_at is
        // set to now() at insert, so its value on an unread row is the save
        // time and reads as a recent access if printed unqualified.
        let last = if r.access_count == 0 {
            "never read".to_string()
        } else {
            format!("{}d ago", (now - r.accessed_at).num_days())
        };
        println!(
            "  [id={}] {}{} reads={} {} | {}d old | {} chars | {}",
            r.id,
            if r.pinned { "PINNED " } else { "" },
            format_args!("({})", r.kind),
            r.access_count,
            last,
            age,
            r.body_chars,
            r.title
        );
    }

    println!(
        "{never_read} of {} never read, {chars_never_read} chars. Forget one with the \
         context_forget tool, or pin what should always be pushed at session start.",
        rows.len()
    );
    Ok(())
}

/// `--pin <id>` / `--unpin <id>`: control what `--recent` sorts to the top.
pub async fn pin(database_url: &str, scope: &Scope, id: i64, pinned: bool) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let verb = if pinned { "pinned" } else { "unpinned" };

    if db::set_pinned(&pool, scope, id, pinned).await? {
        println!("{verb} id={id}");
    } else {
        // Scoped like every other write: same client, same namespace, not
        // forgotten. A live row in another project is deliberately unreachable.
        println!("no live memory with id={id} in namespace {}", scope.namespace);
    }
    Ok(())
}

/// `--history [N]`: what the read path cannot see -- forgotten, superseded,
/// expired.
pub async fn history(database_url: &str, scope: &Scope, limit: i64) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let rows = db::history(&pool, scope, limit).await?;

    if rows.is_empty() {
        println!("nothing hidden in namespace {}", scope.namespace);
        return Ok(());
    }

    println!(
        "{} hidden memories in {} (rows are never reaped; these are still on disk)",
        rows.len(),
        scope.namespace
    );

    let now = Utc::now();
    for r in &rows {
        let mut why = Vec::new();
        if let Some(at) = r.forgotten_at {
            why.push(match &r.forget_reason {
                Some(reason) => format!("forgotten {} ({reason})", at.format("%Y-%m-%d")),
                None => format!("forgotten {}", at.format("%Y-%m-%d")),
            });
        }
        if let Some(by) = r.superseded_by {
            why.push(format!("superseded by id={by}"));
        }
        if r.expires_at.is_some_and(|e| e <= now) {
            why.push("expired".to_string());
        }
        println!(
            "  [id={}] ({}) {} | {} | saved {}",
            r.id,
            r.kind,
            r.title,
            why.join(", "),
            r.created_at.format("%Y-%m-%d")
        );
    }

    println!("Bring one back with --restore <id> (add --detach if it was superseded).");
    Ok(())
}

/// `--restore <id> [--detach]`: undo a forget, optionally cutting the link to
/// whatever superseded the row.
pub async fn restore(database_url: &str, scope: &Scope, id: i64, detach: bool) -> Result<()> {
    let pool = db::connect(database_url).await?;

    let Some(r) = db::restore(&pool, scope, id, detach).await? else {
        println!("no memory with id={id} in namespace {}", scope.namespace);
        return Ok(());
    };

    let mut did = Vec::new();
    if r.unforgotten {
        did.push("un-forgotten".to_string());
    }
    if let Some(from) = r.detached_from {
        did.push(format!("detached from id={from}"));
    }
    // Not "already visible": a row can have nothing to un-forget and still be
    // hidden behind a superseding row, so the reasons below are printed either
    // way and this line only reports what changed.
    if did.is_empty() {
        println!("nothing to undo for id={id}");
    } else {
        println!("restored id={id}: {}", did.join(", "));
    }

    // Saying "restored" about a row that is still invisible is the one way this
    // command can mislead, so the remaining reasons are always spelled out.
    if let Some(by) = r.still_hidden_by {
        println!(
            "  still hidden: superseded by id={by}. Rerun with --detach to put both back in search."
        );
    }
    if r.still_expired {
        println!("  still hidden: expires_at is in the past.");
    }
    Ok(())
}

/// `--save --title T --body "..."|--body-file F [--kind K] [--tags a,b] \
/// [--supersedes 1,2,3]`: write a memory in this process's own namespace.
///
/// Exists for one reason: `context_save` (the MCP tool) can only ever write to
/// the namespace this process resolved at launch, on purpose -- a model must
/// never be able to name another project's namespace in a tool argument. That
/// leaves no way to merge memories that live in a project *other* than the one
/// the current session is running in. This is that path: point `CTXDB_NAMESPACE`
/// (or cwd) at the target project and run it by hand, same trust boundary as
/// every other operator mode here, just exercised by a person instead of a
/// model mid-conversation.
#[allow(clippy::too_many_arguments)]
pub async fn save(
    database_url: &str,
    embed_url: String,
    embed_model: String,
    scope: &Scope,
    title: String,
    body: String,
    kind: String,
    tags: Vec<String>,
    supersedes: Vec<i64>,
) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let embedder = Embedder::new(embed_url, embed_model);
    embedder
        .ping()
        .await
        .context("embedding server unreachable")?;

    let (pieces, inputs) = embedder
        .chunk_inputs(&title, &body)
        .await
        .context("chunking failed")?;
    anyhow::ensure!(!pieces.is_empty(), "body is empty");

    let embeddings = embedder.embed(&inputs).await.context("embedding failed")?;
    anyhow::ensure!(
        embeddings.len() == pieces.len(),
        "embedding server returned {} vectors for {} chunks",
        embeddings.len(),
        pieces.len()
    );
    let chunks: Vec<(String, Vec<f32>)> = pieces.into_iter().zip(embeddings).collect();
    let chunk_count = chunks.len();

    let saved = db::save(
        &pool,
        scope,
        &title,
        &body,
        &kind,
        &tags,
        chunks,
        embedder.model(),
        &supersedes,
    )
    .await
    .context("save failed")?;

    let split = if chunk_count > 1 {
        format!(" in {chunk_count} chunks")
    } else {
        String::new()
    };
    if supersedes.is_empty() {
        println!("saved id={} in namespace {}{split}", saved.id, scope.namespace);
        return Ok(());
    }

    // Same asked-vs-retired distinction as the MCP tool: an id that was not
    // retired means it belongs to another namespace or was already superseded,
    // and swallowing that silently is how a merge looks done while an original
    // is still sitting in search.
    let missed: Vec<String> = supersedes
        .iter()
        .filter(|id| !saved.retired.contains(id))
        .map(i64::to_string)
        .collect();
    if saved.retired.is_empty() {
        println!(
            "saved id={} in namespace {}{split} (nothing was retired)",
            saved.id, scope.namespace
        );
    } else {
        let retired = saved
            .retired
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "saved id={} in namespace {}{split} (retired id={retired})",
            saved.id, scope.namespace
        );
    }
    if !missed.is_empty() {
        println!(
            "  id={} left alone: not in namespace {}, or already superseded",
            missed.join(", "),
            scope.namespace
        );
    }
    Ok(())
}
