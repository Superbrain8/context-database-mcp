//! `--ingest`: turn a compaction summary into a memory.
//!
//! Wired to the **PostCompact** hook, not PreCompact. At PreCompact no summary
//! exists yet, so there is nothing distilled to store; and compaction events
//! only accept `command` hooks (the `prompt`/`agent` types are restricted to
//! tool events), so the hook cannot summarise anything itself. The model also
//! gets no turn during compaction in which it could call `context_save`.
//!
//! The point is to make `/clear` cheap. This store only saves tokens if it
//! *replaces* context rather than adding to it: with the session's own summary
//! already in Postgres, clearing loses much less than it used to.
//!
//! Every failure here is swallowed and the process exits 0. A hook that breaks
//! compaction is far worse than a missed memory.

use std::path::Path;

use anyhow::Result;
use serde_json::Value;
use sqlx::PgPool;

use crate::{db, embed, Scope};

/// Refuse absurd inputs rather than embedding a whole transcript by accident.
/// A real compaction summary from a 290k-token session measured ~19 KB.
const MAX_SUMMARY_CHARS: usize = 200_000;

/// Below this a "summary" is a stub or an error string, not worth a row.
const MIN_SUMMARY_CHARS: usize = 200;

/// Read the hook payload from stdin and store the compaction summary.
pub async fn run(
    database_url: &str,
    embed_url: String,
    embed_model: String,
    mut scope: Scope,
) -> Result<()> {
    let payload = match std::io::read_to_string(std::io::stdin()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("ingest: reading stdin: {e:#}");
            return Ok(());
        }
    };

    let hook: Value = match serde_json::from_str(&payload) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("ingest: payload is not JSON: {e:#}");
            return Ok(());
        }
    };

    // The hook runs in the project directory, so the process-derived namespace
    // is normally already right. Preferring the payload's `cwd` covers the case
    // where it is not -- a silently wrong namespace is this system's worst
    // failure, because the write succeeds and simply becomes invisible.
    if std::env::var("CTXDB_NAMESPACE").is_err() {
        if let Some(ns) = hook
            .get("cwd")
            .and_then(Value::as_str)
            .and_then(|c| crate::namespace_from_dir(Path::new(c)))
        {
            scope.namespace = ns;
        }
    }
    if let Some(sid) = hook.get("session_id").and_then(Value::as_str) {
        scope.session_id = sid.to_string();
    }

    let trigger = hook
        .get("trigger")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let Some(summary) = summary_text(&hook) else {
        tracing::warn!("ingest: no compaction summary found in payload or transcript");
        return Ok(());
    };
    let summary = strip_continuation_preamble(&summary);

    if summary.chars().count() < MIN_SUMMARY_CHARS {
        tracing::warn!("ingest: summary too short to be real, skipping");
        return Ok(());
    }
    let summary: String = summary.chars().take(MAX_SUMMARY_CHARS).collect();

    let pool = match db::connect(database_url).await {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("ingest: database unreachable: {e:#}");
            return Ok(());
        }
    };

    if let Err(e) = store(&pool, &scope, embed_url, embed_model, &summary, trigger).await {
        tracing::warn!("ingest: {e:#}");
    }
    Ok(())
}

async fn store(
    pool: &PgPool,
    scope: &Scope,
    embed_url: String,
    embed_model: String,
    summary: &str,
    trigger: &str,
) -> Result<()> {
    // Compaction can fire several times in one session, and a hook can be
    // re-run by hand while testing. Identical text means the same compaction,
    // so exact-body matching is enough to stay idempotent without a schema
    // change; a later compaction produces different text and still gets stored.
    if let Some(existing) = db::find_by_body(pool, scope, summary).await? {
        tracing::info!("ingest: summary already stored as id={existing}, skipping");
        return Ok(());
    }

    let title = format!(
        "session summary {} ({} compaction)",
        chrono::Local::now().format("%Y-%m-%d %H:%M"),
        trigger
    );

    let pieces = embed::chunk_text(summary);
    if pieces.is_empty() {
        return Ok(());
    }
    let inputs: Vec<String> = pieces
        .iter()
        .map(|c| format!("{title}\n\n{c}"))
        .collect();

    let embedder = embed::Embedder::new(embed_url, embed_model);
    let embeddings = embedder.embed(&inputs).await?;
    if embeddings.len() != pieces.len() {
        anyhow::bail!(
            "embedding server returned {} vectors for {} chunks",
            embeddings.len(),
            pieces.len()
        );
    }

    let chunk_count = pieces.len();
    let chunks: Vec<(String, Vec<f32>)> = pieces.into_iter().zip(embeddings).collect();

    let id = db::save(
        pool,
        scope,
        &title,
        summary,
        "session-summary",
        &[
            "session-summary".to_string(),
            "auto-ingest".to_string(),
            trigger.to_string(),
        ],
        chunks,
        embedder.model(),
        None,
    )
    .await?;

    tracing::info!(
        namespace = %scope.namespace,
        "ingest: stored compaction summary as id={id} in {chunk_count} chunks"
    );
    Ok(())
}

/// The summary, from the payload if it carries one, otherwise from the
/// transcript on disk.
///
/// The transcript is the reliable source: the hook payload's documented fields
/// are the common ones (`session_id`, `transcript_path`, `cwd`, ...), so the
/// direct-field lookups are opportunistic and the file read is the real path.
fn summary_text(hook: &Value) -> Option<String> {
    for key in ["summary", "compact_summary", "compactSummary"] {
        if let Some(s) = hook.get(key).and_then(Value::as_str) {
            if !s.trim().is_empty() {
                return Some(s.to_string());
            }
        }
    }

    let path = hook.get("transcript_path").and_then(Value::as_str)?;
    match std::fs::read_to_string(path) {
        Ok(text) => last_compact_summary(&text),
        Err(e) => {
            tracing::warn!("ingest: reading transcript {path}: {e:#}");
            None
        }
    }
}

/// Pull the newest compaction summary out of a transcript.
///
/// The transcript is JSONL, one entry per line, and the summary is a `user`
/// entry flagged `isCompactSummary`. Lines are cheaply prefiltered on that
/// substring because a transcript runs to megabytes and parsing every line as
/// JSON to find one of them is wasted work. The *last* match wins: a session
/// that compacted twice must ingest the summary just written, not the first one.
fn last_compact_summary(transcript: &str) -> Option<String> {
    transcript
        .lines()
        .rev()
        .filter(|l| l.contains("isCompactSummary"))
        .find_map(|line| {
            let entry: Value = serde_json::from_str(line).ok()?;
            if !entry
                .get("isCompactSummary")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                return None;
            }
            message_text(&entry)
        })
}

/// Message content is a bare string in some transcript versions and a list of
/// typed blocks in others, so both are handled.
fn message_text(entry: &Value) -> Option<String> {
    match entry.get("message")?.get("content")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let joined = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n");
            (!joined.trim().is_empty()).then_some(joined)
        }
        _ => None,
    }
}

/// Drop the "This session is being continued..." framing that Claude Code wraps
/// around every summary.
///
/// It is identical on every compaction, so keeping it would put the same
/// boilerplate in the first chunk of every session-summary memory and pull
/// unrelated summaries toward each other in vector space.
fn strip_continuation_preamble(text: &str) -> &str {
    let t = text.trim();
    if !t.starts_with("This session is being continued") {
        return t;
    }
    match t.find("Summary:") {
        Some(i) => t[i..].trim_start(),
        None => t,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_last_summary_in_a_transcript() {
        // Realistic shape: a compact_boundary system line, then the summary as a
        // user entry, and ordinary traffic in between.
        let transcript = concat!(
            r#"{"type":"user","isCompactSummary":true,"message":{"role":"user","content":"FIRST summary"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":"noise"}}"#,
            "\n",
            r#"{"type":"system","subtype":"compact_boundary","content":"Conversation compacted"}"#,
            "\n",
            r#"{"type":"user","isCompactSummary":true,"message":{"role":"user","content":"SECOND summary"}}"#,
            "\n",
            r#"{"type":"user","message":{"role":"user","content":"later turn"}}"#,
        );
        assert_eq!(
            last_compact_summary(transcript).as_deref(),
            Some("SECOND summary")
        );
    }

    #[test]
    fn ignores_entries_that_only_mention_the_flag() {
        // A transcript records tool results verbatim, so the substring
        // prefilter will hit lines that merely quote the field name.
        let transcript =
            r#"{"type":"assistant","message":{"role":"assistant","content":"isCompactSummary is a field"}}"#;
        assert!(last_compact_summary(transcript).is_none());
    }

    #[test]
    fn reads_block_style_content() {
        let transcript = r#"{"isCompactSummary":true,"message":{"role":"user","content":[{"type":"text","text":"block summary"}]}}"#;
        assert_eq!(
            last_compact_summary(transcript).as_deref(),
            Some("block summary")
        );
    }

    #[test]
    fn no_summary_in_an_ordinary_transcript() {
        let transcript = r#"{"type":"user","message":{"role":"user","content":"hello"}}"#;
        assert!(last_compact_summary(transcript).is_none());
    }

    #[test]
    fn preamble_is_stripped_but_body_kept() {
        let text = "This session is being continued from a previous conversation that ran out \
                    of context. The summary below covers the earlier portion.\n\n\
                    Summary:\n1. Primary Request: build the thing";
        let out = strip_continuation_preamble(text);
        assert!(out.starts_with("Summary:"));
        assert!(out.contains("build the thing"));
    }

    #[test]
    fn text_without_the_preamble_is_untouched() {
        let text = "1. Primary Request: build the thing";
        assert_eq!(strip_continuation_preamble(text), text);
    }
}
