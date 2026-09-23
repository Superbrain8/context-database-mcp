-- Typed edges between memories: the memory graph.
--
-- Before this, the only relation between two memories was supersede. Edges let
-- a memory point at the facts, decisions and dead ends it relates to, so a
-- caller can follow a relation instead of hoping a search surfaces both ends.
--
-- Append + forget, like `memory`: an edge is never updated in place except to
-- set `forgotten_at`. An edge to a row that is later superseded is NOT
-- rewritten; readers follow `superseded_by` to the live head (see graph.rs).
--
-- `rel` has no CHECK constraint on purpose. The allowed values live in one Rust
-- enum (graph::Rel), so adding a relation type needs no migration.
--
-- Safe to run twice.
--
-- NOTE: files in migrations/ only run automatically on a fresh pgdata volume.
-- Apply this to an existing database by hand:
--   docker exec -i ctxdb-postgres psql -U ctx -d ctxdb < migrations/003_edges.sql

-- Target for the composite foreign keys below. `id` is already unique, so this
-- adds no restriction on `memory`; it exists so the database itself refuses an
-- edge whose two ends belong to different clients.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'memory_id_client'
    ) THEN
        ALTER TABLE memory ADD CONSTRAINT memory_id_client UNIQUE (id, client_id);
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS memory_edge (
    id            BIGSERIAL PRIMARY KEY,
    -- isolation: set from server env, never from LLM tool arguments
    client_id     TEXT        NOT NULL,
    src_id        BIGINT      NOT NULL,
    dst_id        BIGINT      NOT NULL,
    rel           TEXT        NOT NULL,
    note          TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    forgotten_at  TIMESTAMPTZ,
    forget_reason TEXT,
    FOREIGN KEY (src_id, client_id) REFERENCES memory (id, client_id) ON DELETE CASCADE,
    FOREIGN KEY (dst_id, client_id) REFERENCES memory (id, client_id) ON DELETE CASCADE,
    CHECK (src_id <> dst_id)
);

-- Symmetric relations are stored with src_id < dst_id (enforced in Rust), so
-- this one index also stops `relates_to A-B` next to `relates_to B-A`.
CREATE UNIQUE INDEX IF NOT EXISTS memory_edge_unique_live
    ON memory_edge (src_id, dst_id, rel) WHERE forgotten_at IS NULL;
CREATE INDEX IF NOT EXISTS memory_edge_src ON memory_edge (src_id) WHERE forgotten_at IS NULL;
CREATE INDEX IF NOT EXISTS memory_edge_dst ON memory_edge (dst_id) WHERE forgotten_at IS NULL;
CREATE INDEX IF NOT EXISTS memory_edge_client ON memory_edge (client_id) WHERE forgotten_at IS NULL;
