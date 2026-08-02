//! Operator modes: `--stale`, `--pin`, `--unpin`, `--history`, `--restore`.
//!
//! None of these are MCP tools, and that is the design rather than an omission.
//! Every tool exposed to the model costs schema tokens in every session's
//! context whether it is used or not, and these four are decisions a person
//! makes occasionally, not decisions a model makes mid-conversation. Pinning is
//! a standing judgement about what future sessions should be told exists;
//! restoring is an undo for a mistake the model itself made.
//!
//! Like `--reindex`, these fail loudly. They are run by hand, so an error is
//! read by someone who can act on it.

use anyhow::Result;
use chrono::Utc;

use crate::{db, Scope};

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
