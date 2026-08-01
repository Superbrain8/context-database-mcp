-- Context Database schema.
--
-- Model is append + forget: rows are never UPDATEd in place by the LLM.
-- A correction is a new row pointing at the old one via supersedes_id; the old
-- row gets superseded_by set and drops out of search. A forget is a soft delete
-- (forgotten_at), so an over-eager forget is recoverable.
--
-- EMBED_DIM is 1024 to match BAAI/bge-m3. Changing the embedding model means
-- changing this dimension and re-embedding every row.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE memory (
    id            BIGSERIAL PRIMARY KEY,

    -- isolation: set from server env, never from LLM tool arguments
    client_id     TEXT        NOT NULL,
    namespace     TEXT        NOT NULL DEFAULT 'default',
    session_id    TEXT,

    kind          TEXT        NOT NULL DEFAULT 'note',
    title         TEXT        NOT NULL,
    body          TEXT        NOT NULL,
    tags          TEXT[]      NOT NULL DEFAULT '{}',
    meta          JSONB       NOT NULL DEFAULT '{}',

    embedding     VECTOR(1024),
    embed_model   TEXT        NOT NULL DEFAULT 'BAAI/bge-m3',

    tsv           TSVECTOR GENERATED ALWAYS AS (
                      setweight(to_tsvector('english', title), 'A') ||
                      setweight(to_tsvector('english', body),  'B')
                  ) STORED,

    -- append + forget lifecycle
    supersedes_id BIGINT      REFERENCES memory(id) ON DELETE SET NULL,
    superseded_by BIGINT      REFERENCES memory(id) ON DELETE SET NULL,
    forgotten_at  TIMESTAMPTZ,
    forget_reason TEXT,
    expires_at    TIMESTAMPTZ,

    pinned        BOOLEAN     NOT NULL DEFAULT FALSE,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    access_count  INTEGER     NOT NULL DEFAULT 0
);

-- Partial index: only live rows are ever searched, so only live rows are indexed.
CREATE INDEX memory_embedding_hnsw
    ON memory USING hnsw (embedding vector_cosine_ops)
    WHERE forgotten_at IS NULL AND superseded_by IS NULL;

CREATE INDEX memory_tsv_gin
    ON memory USING gin (tsv)
    WHERE forgotten_at IS NULL AND superseded_by IS NULL;

CREATE INDEX memory_scope
    ON memory (client_id, namespace, created_at DESC)
    WHERE forgotten_at IS NULL AND superseded_by IS NULL;

CREATE INDEX memory_tags_gin ON memory USING gin (tags);

-- trigram index on title, fallback for fuzzy identifier lookup
CREATE INDEX memory_title_trgm ON memory USING gin (title gin_trgm_ops);

-- Convenience view of what search is allowed to see.
CREATE VIEW memory_live AS
    SELECT * FROM memory
    WHERE forgotten_at IS NULL
      AND superseded_by IS NULL
      AND (expires_at IS NULL OR expires_at > now());
