-- Move embeddings from the memory row onto per-chunk rows.
--
-- A single embedding for a whole memory is fine for a two-sentence note and bad
-- for a long one: averaging a 3000-word file summary into one 1024-d vector
-- blurs every specific thing it contains, which is exactly what retrieval needs.
-- Chunks are embedded separately and a memory scores by its best-matching chunk.
--
-- Short bodies simply produce one chunk, so there is no special case.
--
-- NOTE: files in migrations/ only run automatically on a fresh pgdata volume.
-- Apply this to an existing database by hand:
--   docker exec -i ctxdb-postgres psql -U ctx -d ctxdb < migrations/002_chunks.sql

CREATE TABLE memory_chunk (
    id        BIGSERIAL PRIMARY KEY,
    memory_id BIGINT  NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    ord       INTEGER NOT NULL,
    text      TEXT    NOT NULL,
    embedding VECTOR(1024),
    UNIQUE (memory_id, ord)
);

CREATE INDEX memory_chunk_embedding_hnsw
    ON memory_chunk USING hnsw (embedding vector_cosine_ops);

CREATE INDEX memory_chunk_memory_id ON memory_chunk (memory_id);

-- Carry existing rows over as a single chunk each, so nothing already stored
-- becomes unsearchable.
INSERT INTO memory_chunk (memory_id, ord, text, embedding)
SELECT id, 0, body, embedding
FROM memory
WHERE embedding IS NOT NULL;

-- The per-memory embedding is now dead weight; the view is rebuilt without it.
DROP VIEW memory_live;

DROP INDEX IF EXISTS memory_embedding_hnsw;
ALTER TABLE memory DROP COLUMN embedding;

CREATE VIEW memory_live AS
    SELECT * FROM memory
    WHERE forgotten_at IS NULL
      AND superseded_by IS NULL
      AND (expires_at IS NULL OR expires_at > now());
