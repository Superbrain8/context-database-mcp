# Context Database MCP

A local, offline memory store for LLMs. The model offloads context into Postgres and searches it
back later through four MCP tools, instead of carrying everything in its context window.

Nothing leaves the machine: Postgres and the embedding model both run in Docker on localhost, and
the server uses no TLS and no external API.

## How it works

```
Claude Code    ──stdio──▶  context-database-mcp (process, CTXDB_CLIENT_ID=claude-code)    ─┐
Claude Desktop ──stdio──▶  context-database-mcp (process, CTXDB_CLIENT_ID=claude-desktop) ─┤
                                                                                           │
                                          ┌────────────────────────────────────────────────┘
                                          ▼
                        Postgres 17 + pgvector  :5433      TEI + BAAI/bge-m3  :8085
```

One server process per MCP client. Isolation comes from `CTXDB_CLIENT_ID` and `CTXDB_NAMESPACE` in
that process's environment — never from tool arguments — so a Claude Desktop session cannot read or
retire a Claude Code memory.

### Reads can widen across projects; writes cannot

`CTXDB_NAMESPACE` is normally the project. Hard-isolating it would waste the most valuable thing
here: a lesson learned in one repo is usually worth having in the next one.

So the two directions are deliberately asymmetric:

- `context_search` takes `cross_project: true`, which drops the namespace filter but keeps the
  `client_id` one. Foreign hits are labelled `from project <namespace>` so the model knows the
  advice comes from elsewhere. `context_get` follows, otherwise search would hand back ids that
  cannot be read.
- `context_save` and `context_forget` are always pinned to the process's own namespace. One project
  can never write into, or retire a memory belonging to, another.

A different `CTXDB_CLIENT_ID` sees nothing either way.

The embedding model runs in its own container rather than inside the server, because one process
runs per client and loading ~2 GB of weights into each of them would multiply that cost.

### Retrieval is hybrid, not pure vector

This database mostly holds identifiers, error strings and file paths, and embeddings blur exactly
those rare tokens. Every search fuses two rankers with Reciprocal Rank Fusion (k=60):

- dense: pgvector HNSW, cosine distance
- lexical: Postgres `tsvector`, title weighted `A`, body `B`

RRF needs no tuned weights and no cross-encoder reranker.

The dense ranker is capped by `CTXDB_MAX_DISTANCE` (default 0.55 cosine). Vector search otherwise
returns its k nearest rows no matter how unrelated, so a query with no real answer still comes back
looking like a hit — which teaches the model to treat noise as recall. The lexical ranker is not
capped: an exact token match is its own evidence.

### Bodies are chunked before embedding

One vector per memory works for a two-sentence note and fails for a long one — averaging a
3,000-word document into a single 1024-d vector blurs every specific thing in it. Bodies are split
into ~700-character overlapping chunks (on paragraph, then sentence, then whitespace boundaries) and
each chunk is embedded separately. A short body produces exactly one chunk, so there is no special
case.

A memory scores by its **single best-matching chunk**, collapsed with `DISTINCT ON`. Without that
collapse, one long memory would fill several of the top slots with its own chunks and crowd
everything else out.

A hit's snippet is its whole matching chunk rather than the head of the body. Two reasons: the head
of a long document is usually irrelevant to why it matched, and truncating the chunk can cut off the
exact sentence that caused the match — making a correct hit look like a miss. Chunk size is
therefore the real cost knob for search output.

### Memories are append + forget, never edited

Closer to how human memory behaves, and it keeps an audit trail:

- **correct** something → `context_save` with `supersedes`. The new row is written, the old one gets
  `superseded_by` set and disappears from search. Both stay on disk.
- **drop** something → `context_forget`. Soft delete via `forgotten_at`; recoverable by hand.

The only in-place update the system performs is the access-stats bump in `context_get`.

## Tools

| Tool | Purpose |
|---|---|
| `context_save` | Store a durable fact, decision, root cause or file summary. Returns an id. |
| `context_search` | Hybrid search. Returns ranked **snippets** (300 chars) + ids. `cross_project: true` widens beyond the current project. |
| `context_get` | Full body of one memory, by id. |
| `context_forget` | Soft-delete a memory. |

`context_search` deliberately returns snippets only. Returning full bodies would re-inflate the
context window and defeat the point of offloading.

## Setup

### 1. Start the stack

```bash
docker compose up -d
```

First start downloads bge-m3 (~2.2 GB) into the `hfcache` volume. Watch it with
`docker logs -f ctxdb-embed`; the server is ready when `/health` returns 200.

Files in `migrations/` are applied automatically, but **only on a fresh `pgdata` volume**. A
database that already holds memories must have new migrations applied by hand — dropping the volume
to change schema is not an option for a store whose entire job is to not forget things:

```bash
docker exec -i ctxdb-postgres psql -U ctx -d ctxdb -v ON_ERROR_STOP=1 < migrations/002_chunks.sql
```

`002_chunks.sql` carries existing rows forward as single chunks, so applying it to a populated
database loses nothing.

### 2. Build the server

Requires a Rust toolchain with a working linker. On Windows that means the MSVC linker — install
"Desktop development with C++" via the Visual Studio Installer, then `winget install Rustlang.Rustup`.

If `cargo` is still "not recognized" after installing, the shell inherited a stale environment.
Restarting the editor is not always enough: it inherits from whatever launched it, which may itself
predate the install. Close every editor window, open a fresh terminal, confirm `cargo --version`
there, and launch the editor from that terminal — or just reboot.

```bash
cargo build --release
# -> target/release/context-database-mcp
```

### 3. Register with your MCP clients

Give each client its own `CTXDB_CLIENT_ID`.

**Claude Code** (`.mcp.json` in the project root):

```json
{
  "mcpServers": {
    "context-db": {
      "command": "d:/Projects/Context_Database_MCP/target/release/context-database-mcp.exe",
      "env": {
        "CTXDB_CLIENT_ID": "claude-code",
        "CTXDB_NAMESPACE": "context-database-mcp"
      }
    }
  }
}
```

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json`): same block with
`"CTXDB_CLIENT_ID": "claude-desktop"`.

Unset variables fall back to the defaults in `.env.example`.

The committed `.mcp.json` uses an absolute path to the binary, so it is machine-specific — adjust it
if the repo lives elsewhere.

### 4. Optional: push memories at session start

Search alone means recall only happens when the model thinks to ask for it. `.claude/settings.json`
registers a `SessionStart` hook that runs the binary's `--recent` mode and lists pinned and recent
memories (titles and ids only, never bodies) so the model knows they exist:

```bash
./target/release/context-database-mcp.exe --recent 5
```

The hook swallows every failure and exits 0 — a memory store being down must never break a session.
It touches no embedder, so it also works while the model weights are still loading.

## Operational notes

- **stdout is the MCP transport.** All logging goes to stderr. A stray `println!` corrupts the
  protocol framing.
- **Embedding dimension is hardcoded in two places** that must agree: `VECTOR(1024)` in the
  migration and `EMBED_DIM` in `src/embed.rs`. Changing the model means re-embedding every row;
  `embed_model` is stored per row so a migration can find the stale ones.
- The server pings Postgres and the embedder at startup and exits loudly if either is down, rather
  than failing on the first tool call where the model would just give up on the tool.

## Status

Working end to end. The built binary was driven over real stdio JSON-RPC against the live stack:
handshake, `tools/list`, all four tools, supersede, soft delete, the distance cutoff, the
missing-id path, and cross-project reads with narrow writes across three concurrent server
processes. `cargo test` covers the chunker, including multi-byte input that byte-offset slicing
would panic on.

Retrieval behaves as intended on both halves of the hybrid: a paraphrase query with no shared
keywords is answered by the dense ranker alone, and an exact identifier (`ivfflat`) ranks first in
both rankers. Chunking was verified against a 4,000-character document with one specific instruction
buried mid-way through: a paraphrased query sharing no keywords with that sentence retrieved it, and
returned the containing chunk as the snippet.

## Not done yet

- No eviction or summarisation. Memories accumulate forever; nothing ages out or gets consolidated.
- `pinned` exists in the schema and is honoured by `--recent`, but no tool sets it.
- Superseded and forgotten rows are never reaped, by design — but there is no tool to review or
  restore them either.
- Chunk boundaries are character-based, not token-based, so a chunk can land slightly over or under
  what the model would consider a natural passage.
