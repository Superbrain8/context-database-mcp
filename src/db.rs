//! Postgres access layer.
//!
//! All queries are runtime-checked (`sqlx::query`) rather than compile-time
//! checked (`sqlx::query!`) so the crate builds without a live database.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use pgvector::Vector;
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

use crate::Scope;

/// Reciprocal-rank-fusion constant. 60 is the value from the original RRF paper
/// and is deliberately large so no single ranker can dominate the other.
const RRF_K: f64 = 60.0;

/// How deep each individual ranker goes before fusion.
const CANDIDATES: i64 = 50;

#[derive(Debug)]
pub struct SearchHit {
    pub id: i64,
    pub title: String,
    pub snippet: String,
    pub kind: String,
    pub tags: Vec<String>,
    pub namespace: String,
    pub created_at: DateTime<Utc>,
    pub score: f64,
}

#[derive(Debug)]
pub struct MemoryRow {
    pub id: i64,
    pub title: String,
    pub body: String,
    pub kind: String,
    pub tags: Vec<String>,
    pub namespace: String,
    pub created_at: DateTime<Utc>,
    pub supersedes_id: Option<i64>,
}

pub async fn connect(url: &str) -> Result<PgPool> {
    PgPoolOptions::new()
        .max_connections(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .connect(url)
        .await
        .with_context(|| format!("connecting to postgres at {url}"))
}

/// `chunks` pairs each chunk's text with its embedding, in order. The memory row
/// itself carries no vector -- a memory is found through its best-matching
/// chunk (see `search`).
#[allow(clippy::too_many_arguments)]
pub async fn save(
    pool: &PgPool,
    scope: &Scope,
    title: &str,
    body: &str,
    kind: &str,
    tags: &[String],
    chunks: Vec<(String, Vec<f32>)>,
    embed_model: &str,
    supersedes: Option<i64>,
) -> Result<i64> {
    let mut tx = pool.begin().await?;

    let id: i64 = sqlx::query(
        r#"
        INSERT INTO memory
            (client_id, namespace, session_id, kind, title, body, tags,
             embed_model, supersedes_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        RETURNING id
        "#,
    )
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(&scope.session_id)
    .bind(kind)
    .bind(title)
    .bind(body)
    .bind(tags)
    .bind(embed_model)
    .bind(supersedes)
    .fetch_one(&mut *tx)
    .await
    .context("inserting memory")?
    .get("id");

    for (ord, (text, embedding)) in chunks.into_iter().enumerate() {
        sqlx::query(
            r#"
            INSERT INTO memory_chunk (memory_id, ord, text, embedding)
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(id)
        .bind(ord as i32)
        .bind(&text)
        .bind(Vector::from(embedding))
        .execute(&mut *tx)
        .await
        .context("inserting memory chunk")?;
    }

    // Close the loop on the superseded row so it drops out of search. Scoped to
    // the same client+namespace so one client cannot retire another's memory.
    if let Some(old) = supersedes {
        sqlx::query(
            r#"
            UPDATE memory SET superseded_by = $1
            WHERE id = $2 AND client_id = $3 AND namespace = $4
            "#,
        )
        .bind(id)
        .bind(old)
        .bind(&scope.client_id)
        .bind(&scope.namespace)
        .execute(&mut *tx)
        .await
        .context("marking superseded row")?;
    }

    tx.commit().await?;
    Ok(id)
}

/// Hybrid retrieval: dense vector search fused with BM25-ish full-text search.
///
/// Vector search alone is weak on exactly the content this database holds --
/// identifiers, error codes, file paths -- because embeddings blur rare tokens.
/// Full-text alone misses paraphrase. RRF over both needs no tuned weights and
/// no cross-encoder reranker.
///
/// `max_distance` caps how far the dense ranker may reach. Without it, vector
/// search always returns its k nearest rows no matter how unrelated they are,
/// so an empty-handed query still comes back looking like a hit -- which teaches
/// the model to treat noise as recall. The lexical ranker is deliberately not
/// capped: an exact token match is its own evidence.
#[allow(clippy::too_many_arguments)]
pub async fn search(
    pool: &PgPool,
    scope: &Scope,
    query_text: &str,
    query_embedding: Vec<f32>,
    limit: i64,
    kind: Option<&str>,
    tags: &[String],
    max_distance: f64,
    cross_project: bool,
) -> Result<Vec<SearchHit>> {
    let rows = sqlx::query(
        r#"
        WITH filtered AS (
            SELECT id, title, body, kind, tags, namespace, created_at, tsv
            FROM memory_live
            WHERE client_id = $1
              AND ($11 OR namespace = $2)
              AND ($5::text IS NULL OR kind = $5)
              AND ($6::text[] = '{}' OR tags @> $6)
        ),
        -- A memory scores by its single best-matching chunk. Without the
        -- collapse, one long memory would occupy several of the top slots with
        -- its own chunks and crowd everything else out.
        -- DISTINCT ON keeps the closest chunk per memory *and* its text, so the
        -- snippet can show the passage that actually matched rather than
        -- whatever happens to be at the top of a long body.
        vec AS (
            SELECT DISTINCT ON (c.memory_id)
                   c.memory_id            AS id,
                   (c.embedding <=> $3)   AS dist,
                   c.text                 AS chunk_text
            FROM memory_chunk c
            JOIN filtered f ON f.id = c.memory_id
            WHERE c.embedding IS NOT NULL
              AND (c.embedding <=> $3) <= $10
            ORDER BY c.memory_id, (c.embedding <=> $3)
        ),
        vec_ranked AS (
            SELECT id, chunk_text, ROW_NUMBER() OVER (ORDER BY dist) AS rank
            FROM vec
            ORDER BY dist
            LIMIT $7
        ),
        kw AS (
            SELECT id, ROW_NUMBER() OVER (
                       ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $4)) DESC
                   ) AS rank
            FROM filtered
            WHERE tsv @@ websearch_to_tsquery('english', $4)
            LIMIT $7
        )
        SELECT f.id,
               f.title,
               -- The matching chunk is shown WHOLE: truncating it can cut off
               -- the exact sentence that caused the match, which is what makes a
               -- correct hit look irrelevant. Chunk size is what bounds the cost
               -- here (see embed::CHUNK_CHARS). Lexical-only hits have no chunk,
               -- so they fall back to the body's opening.
               COALESCE(vec_ranked.chunk_text, left(f.body, 300)) AS snippet,
               f.kind,
               f.tags,
               f.namespace,
               f.created_at,
               COALESCE(1.0 / ($8 + vec_ranked.rank), 0.0)
             + COALESCE(1.0 / ($8 + kw.rank),         0.0) AS score
        FROM filtered f
        LEFT JOIN vec_ranked ON vec_ranked.id = f.id
        LEFT JOIN kw         ON kw.id         = f.id
        WHERE vec_ranked.id IS NOT NULL OR kw.id IS NOT NULL
        ORDER BY score DESC
        LIMIT $9
        "#,
    )
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(Vector::from(query_embedding))
    .bind(query_text)
    .bind(kind)
    .bind(tags)
    .bind(CANDIDATES)
    .bind(RRF_K)
    .bind(limit)
    .bind(max_distance)
    .bind(cross_project)
    .fetch_all(pool)
    .await
    .context("hybrid search")?;

    Ok(rows
        .into_iter()
        .map(|r| SearchHit {
            id: r.get("id"),
            title: r.get("title"),
            snippet: r.get("snippet"),
            kind: r.get("kind"),
            tags: r.get("tags"),
            namespace: r.get("namespace"),
            created_at: r.get("created_at"),
            score: r.get("score"),
        })
        .collect())
}

/// Scoped to `client_id` only, deliberately not to `namespace`: a cross-project
/// search can surface an id from another namespace, and refusing to fetch it
/// here would leave the model holding an id it cannot read. Reads widen with
/// search; writes (`save`, `forget`) stay pinned to the process's own namespace,
/// so one project can never retire another's memory.
pub async fn get(pool: &PgPool, scope: &Scope, id: i64) -> Result<Option<MemoryRow>> {
    // Bumping access stats here is what makes "what did I look at recently"
    // answerable later; it is the one in-place update the system performs.
    let row = sqlx::query(
        r#"
        UPDATE memory
        SET accessed_at = now(), access_count = access_count + 1
        WHERE id = $1 AND client_id = $2 AND forgotten_at IS NULL
        RETURNING id, title, body, kind, tags, namespace, created_at, supersedes_id
        "#,
    )
    .bind(id)
    .bind(&scope.client_id)
    .fetch_optional(pool)
    .await
    .context("fetching memory")?;

    Ok(row.map(|r| MemoryRow {
        id: r.get("id"),
        title: r.get("title"),
        body: r.get("body"),
        kind: r.get("kind"),
        tags: r.get("tags"),
        namespace: r.get("namespace"),
        created_at: r.get("created_at"),
        supersedes_id: r.get("supersedes_id"),
    }))
}

/// Exact-body lookup, used by `--ingest` to stay idempotent.
///
/// Scoped like a write (client + namespace), not like a read: it exists to stop
/// this process from inserting a duplicate of its own, and a matching body in
/// another project is not that.
pub async fn find_by_body(pool: &PgPool, scope: &Scope, body: &str) -> Result<Option<i64>> {
    let row = sqlx::query(
        r#"
        SELECT id FROM memory
        WHERE client_id = $1 AND namespace = $2 AND body = $3 AND forgotten_at IS NULL
        LIMIT 1
        "#,
    )
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(body)
    .fetch_optional(pool)
    .await
    .context("looking for an identical body")?;

    Ok(row.map(|r| r.get("id")))
}

#[derive(Debug)]
pub struct ReindexRow {
    pub id: i64,
    pub namespace: String,
    pub title: String,
    pub body: String,
    /// How many chunks the row carries right now, for the before/after report.
    pub chunks: i64,
}

/// Every embedding model present in this client's corpus.
///
/// `--reindex` refuses to run when this holds anything but the model it is
/// configured with. Re-embedding part of a corpus with a different model mixes
/// incomparable vectors inside one HNSW index, and the damage is silent: nothing
/// errors, ranking simply gets worse.
pub async fn embed_models_in_use(pool: &PgPool, client_id: &str) -> Result<Vec<String>> {
    let rows = sqlx::query(
        r#"
        SELECT DISTINCT embed_model FROM memory
        WHERE client_id = $1 AND forgotten_at IS NULL
        ORDER BY embed_model
        "#,
    )
    .bind(client_id)
    .fetch_all(pool)
    .await
    .context("listing embedding models in use")?;

    Ok(rows.into_iter().map(|r| r.get("embed_model")).collect())
}

/// Everything `--reindex` will rewrite: the whole client corpus, every
/// namespace, forgotten rows excluded.
///
/// Not restricted to `memory_live`. A superseded or expired row is invisible to
/// search today but still on disk and still restorable, and leaving it on stale
/// chunk boundaries would mean restoring it later restores something that
/// retrieves badly.
pub async fn rows_to_reindex(pool: &PgPool, client_id: &str) -> Result<Vec<ReindexRow>> {
    let rows = sqlx::query(
        r#"
        SELECT m.id, m.namespace, m.title, m.body,
               (SELECT count(*) FROM memory_chunk c WHERE c.memory_id = m.id) AS chunks
        FROM memory m
        WHERE m.client_id = $1 AND m.forgotten_at IS NULL
        ORDER BY m.id
        "#,
    )
    .bind(client_id)
    .fetch_all(pool)
    .await
    .context("listing rows to reindex")?;

    Ok(rows
        .into_iter()
        .map(|r| ReindexRow {
            id: r.get("id"),
            namespace: r.get("namespace"),
            title: r.get("title"),
            body: r.get("body"),
            chunks: r.get("chunks"),
        })
        .collect())
}

/// One memory's stored chunk text, in order.
///
/// Used by `--reindex --dry-run` to say whether a row is actually stale. Chunk
/// *count* is the wrong test on its own: a chunker fix that moves boundaries
/// without adding or removing a chunk changes every embedding in the row and
/// leaves the count untouched.
pub async fn chunk_texts(pool: &PgPool, memory_id: i64) -> Result<Vec<String>> {
    let rows = sqlx::query("SELECT text FROM memory_chunk WHERE memory_id = $1 ORDER BY ord")
        .bind(memory_id)
        .fetch_all(pool)
        .await
        .context("reading stored chunk text")?;

    Ok(rows.into_iter().map(|r| r.get("text")).collect())
}

/// Swap one memory's chunks for freshly computed ones.
///
/// `memory.body` is never touched: chunks are derived data, the body is the
/// source of truth. That is also why deleting and reinserting chunk rows does
/// not violate the append-only rule -- that rule protects memories, not
/// derivatives.
///
/// Delete and insert share one transaction because the failure to design against
/// is a row left with zero chunks: it disappears from dense search with no error
/// at all, and only surfaces months later as a search that stops finding
/// something. The empty-input guard covers the same hole from the other side.
pub async fn replace_chunks(
    pool: &PgPool,
    memory_id: i64,
    chunks: Vec<(String, Vec<f32>)>,
    embed_model: &str,
) -> Result<()> {
    if chunks.is_empty() {
        anyhow::bail!("refusing to leave memory id={memory_id} with no chunks");
    }

    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM memory_chunk WHERE memory_id = $1")
        .bind(memory_id)
        .execute(&mut *tx)
        .await
        .context("deleting old chunks")?;

    for (ord, (text, embedding)) in chunks.into_iter().enumerate() {
        sqlx::query(
            r#"
            INSERT INTO memory_chunk (memory_id, ord, text, embedding)
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(memory_id)
        .bind(ord as i32)
        .bind(&text)
        .bind(Vector::from(embedding))
        .execute(&mut *tx)
        .await
        .context("inserting reindexed chunk")?;
    }

    // The row now records which model its vectors actually came from, so a
    // half-finished run is visible in the data rather than only in a log line.
    sqlx::query("UPDATE memory SET embed_model = $2 WHERE id = $1")
        .bind(memory_id)
        .bind(embed_model)
        .execute(&mut *tx)
        .await
        .context("recording the embedding model")?;

    tx.commit().await?;
    Ok(())
}

/// Soft delete. The row stays on disk so a wrong forget can be undone by hand;
/// nothing in the read path can see it again.
pub async fn forget(pool: &PgPool, scope: &Scope, id: i64, reason: Option<&str>) -> Result<bool> {
    let done = sqlx::query(
        r#"
        UPDATE memory
        SET forgotten_at = now(), forget_reason = $4
        WHERE id = $1 AND client_id = $2 AND namespace = $3 AND forgotten_at IS NULL
        "#,
    )
    .bind(id)
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(reason)
    .execute(pool)
    .await
    .context("forgetting memory")?;

    Ok(done.rows_affected() > 0)
}

/// The kind `--ingest` writes. Held apart from the rest of the push list; see
/// `recent`.
pub const SESSION_SUMMARY: &str = "session-summary";

/// Pinned + most recent, for the SessionStart push path (`--recent`).
///
/// Session summaries are excluded deliberately. The push sends titles only, so
/// the model can judge relevance without paying for bodies -- which makes the
/// title the entire value of a slot. "session summary 2026-08-01 18:41" says
/// nothing about whether it is worth fetching, while "sqlx must track whatever
/// version pgvector resolves to" sells itself. And summaries are exactly the
/// rows that would monopolise the list: one per compaction, always newest.
/// `latest_session_summary` surfaces one of them separately instead.
pub async fn recent(pool: &PgPool, scope: &Scope, limit: i64) -> Result<Vec<SearchHit>> {
    let rows = sqlx::query(
        r#"
        SELECT id, title, left(body, 300) AS snippet, kind, tags, namespace,
               created_at, 0.0::float8 AS score
        FROM memory_live
        WHERE client_id = $1 AND namespace = $2 AND kind <> $4
        ORDER BY pinned DESC, created_at DESC
        LIMIT $3
        "#,
    )
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(limit)
    .bind(SESSION_SUMMARY)
    .fetch_all(pool)
    .await
    .context("listing recent memories")?;

    Ok(rows
        .into_iter()
        .map(|r| SearchHit {
            id: r.get("id"),
            title: r.get("title"),
            snippet: r.get("snippet"),
            kind: r.get("kind"),
            tags: r.get("tags"),
            namespace: r.get("namespace"),
            created_at: r.get("created_at"),
            score: r.get("score"),
        })
        .collect())
}

/// The newest compaction summary, or None if this project has never compacted.
///
/// Exactly one, and never more: straight after `/clear` the last summary is the
/// only thing standing in for the session that was thrown away, but the one
/// before it is just an older session -- no more deserving of a slot than any
/// other memory, and reachable through search like everything else.
pub async fn latest_session_summary(pool: &PgPool, scope: &Scope) -> Result<Option<SearchHit>> {
    let row = sqlx::query(
        r#"
        SELECT id, title, left(body, 300) AS snippet, kind, tags, namespace,
               created_at, 0.0::float8 AS score
        FROM memory_live
        WHERE client_id = $1 AND namespace = $2 AND kind = $3
        ORDER BY created_at DESC
        LIMIT 1
        "#,
    )
    .bind(&scope.client_id)
    .bind(&scope.namespace)
    .bind(SESSION_SUMMARY)
    .fetch_optional(pool)
    .await
    .context("fetching the latest session summary")?;

    Ok(row.map(|r| SearchHit {
        id: r.get("id"),
        title: r.get("title"),
        snippet: r.get("snippet"),
        kind: r.get("kind"),
        tags: r.get("tags"),
        namespace: r.get("namespace"),
        created_at: r.get("created_at"),
        score: r.get("score"),
    }))
}
