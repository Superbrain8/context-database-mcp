//! `--reindex`: re-chunk and re-embed stored memories in place.
//!
//! Chunk boundaries and embeddings are derived from the body by code that
//! changes -- the chunker has already been fixed once, and rows saved before the
//! fix still carry the old boundaries. Bodies are untouched, so the only way to
//! carry a fix backwards is to recompute the derived half.
//!
//! Two things it deliberately cannot do. It cannot repair a *body*: re-chunking
//! polluted text just re-chunks polluted text, and a bad body needs a re-ingest
//! plus a supersede. And it cannot reindex a subset -- see `db::embed_models_in_use`.
//!
//! Unlike the hook paths (`--recent`, `--ingest`), this one fails loudly. A hook
//! that breaks a session is worse than a missed memory; a reindex that quietly
//! half-finishes is worse than one that stops and says so.

use anyhow::{bail, Result};

use crate::{db, embed, Scope};

/// Body of `--reindex`. Covers the whole corpus of this `client_id`, every
/// namespace -- the HNSW index is shared across namespaces, so "reindex what
/// this project can see" is exactly the partial run that mixes vector spaces.
pub async fn run(
    database_url: &str,
    embed_url: String,
    embed_model: String,
    scope: &Scope,
    dry_run: bool,
) -> Result<()> {
    let pool = db::connect(database_url).await?;

    let in_use = db::embed_models_in_use(&pool, &scope.client_id).await?;
    let foreign: Vec<&str> = in_use
        .iter()
        .map(String::as_str)
        .filter(|m| *m != embed_model)
        .collect();
    if !foreign.is_empty() {
        bail!(
            "refusing to reindex: this corpus already holds vectors from {}, and CTXDB_EMBED_MODEL \
             is {embed_model}. Reindexing with a second model would mix incomparable vectors in \
             one index and degrade ranking silently. Reindex the whole corpus with one model, or \
             not at all.",
            foreign.join(", ")
        );
    }

    let rows = db::rows_to_reindex(&pool, &scope.client_id).await?;
    if rows.is_empty() {
        println!("nothing to reindex for client {}", scope.client_id);
        return Ok(());
    }

    let chunks_before: i64 = rows.iter().map(|r| r.chunks).sum();
    println!(
        "reindex: {} memories, {} chunks, model {} (client {})",
        rows.len(),
        chunks_before,
        embed_model,
        scope.client_id
    );

    if dry_run {
        let mut planned = 0usize;
        let mut stale = 0usize;
        for row in &rows {
            let (chunks, _) = embed::chunk_inputs(&row.title, &row.body);
            planned += chunks.len();

            // Compared as text, not by count: the chunker fix this mode exists
            // for moves boundaries inside rows whose chunk count never changes.
            let stored = db::chunk_texts(&pool, row.id).await?;
            if stored != chunks {
                stale += 1;
                println!(
                    "  id={} [{}] {} chunks -> {}  {}",
                    row.id,
                    row.namespace,
                    row.chunks,
                    chunks.len(),
                    row.title
                );
            }
        }
        println!(
            "dry run: {stale} of {} memories would change, {chunks_before} chunks would become \
             {planned}. Every row is re-embedded regardless -- an unchanged chunk still gets a \
             fresh vector. Nothing was written.",
            rows.len()
        );
        return Ok(());
    }

    let embedder = embed::Embedder::new(embed_url, embed_model);
    // Before a single chunk is deleted. A dead embedder discovered on row 30
    // leaves the corpus half-converted for no reason.
    embedder.ping().await?;

    let mut done = 0usize;
    let mut skipped = 0usize;
    let mut chunks_after = 0usize;

    for row in &rows {
        let (chunks, inputs) = embed::chunk_inputs(&row.title, &row.body);
        if chunks.is_empty() {
            // A body that chunks to nothing is whitespace. Deleting its chunks
            // would strip it from search; leaving them costs nothing.
            tracing::warn!("reindex: id={} has an empty body, left alone", row.id);
            skipped += 1;
            continue;
        }

        let embeddings = embedder.embed(&inputs).await.map_err(|e| {
            e.context(format!(
                "embedding id={} ({} memories already reindexed and committed; \
                 rerun to finish -- reindex is idempotent)",
                row.id, done
            ))
        })?;
        if embeddings.len() != chunks.len() {
            bail!(
                "embedding server returned {} vectors for {} chunks of id={} \
                 ({done} memories already reindexed and committed)",
                embeddings.len(),
                chunks.len(),
                row.id
            );
        }

        chunks_after += chunks.len();
        let paired: Vec<(String, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
        db::replace_chunks(&pool, row.id, paired, embedder.model()).await?;

        done += 1;
        // Progress on stdout, not just in the log: this runs for minutes on a
        // CPU embedder and a silent process looks hung.
        if done.is_multiple_of(10) {
            println!("  {done}/{} memories", rows.len());
        }
    }

    let skip_note = if skipped > 0 {
        format!(", {skipped} left alone (empty body)")
    } else {
        String::new()
    };
    println!("reindexed {done} memories into {chunks_after} chunks{skip_note}");
    Ok(())
}
