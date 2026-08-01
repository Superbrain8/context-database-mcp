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

#### Once for every project (user scope)

`CTXDB_NAMESPACE` defaults to the name of the server's working directory, and clients launch the
server with the project as its working directory. So one user-scope registration covers every
project and each still gets its own namespace — register it once in `~/.claude.json` and leave
`CTXDB_NAMESPACE` unset:

```json
{
  "mcpServers": {
    "context-db": {
      "command": "/ABSOLUTE/PATH/TO/context-database-mcp/target/release/context-database-mcp",
      "env": { "CTXDB_CLIENT_ID": "claude-code" }
    }
  }
}
```

Do **not** pin `CTXDB_NAMESPACE` here: a user-scope value applies to every project at once, so all of
them would share a single namespace and the isolation would be gone. The server logs the namespace
it resolved on startup, because a silently wrong one is the worst failure mode this has — saves
still succeed, they just become invisible.

The derived namespace is the folder name, not the full path, so two checkouts named `api` under
different parents share one namespace. Override per project when that matters.

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json`): same block with
`"CTXDB_CLIENT_ID": "claude-desktop"`.

Unset variables fall back to the defaults in `.env.example`.

#### Per project, only when you need to override

A project-level `.mcp.json` is worth adding only when the folder name is not the namespace you want
— two checkouts sharing a name, or a folder you would rather not have as the identifier. Copy
`.mcp.json.example`, set the absolute binary path, and pin `CTXDB_NAMESPACE`. Keep `.mcp.json` out of
version control: the server is launched by absolute path, and a committed entry pointing at a binary
that does not exist on someone else's disk errors in their client every session.

This repository deliberately has no project-level config, so it exercises the same path everyone
else gets.

### 4. Optional: push memories at session start

Search alone means recall only happens when the model thinks to ask for it. A `SessionStart` hook
closes that gap by running the binary's `--recent` mode, which lists pinned and recent memories —
titles and ids only, never bodies — so the model knows they exist:

```bash
./target/release/context-database-mcp --recent 5
```

Hooks have the same two scopes as the MCP registration above, and for the same reason the user scope
is the right default: `~/.claude/settings.json` covers every project at once, and the namespace still
resolves per project because hooks run with the project as their working directory. Leave
`CTXDB_NAMESPACE` unset here too:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "BIN=\"/ABSOLUTE/PATH/TO/context-database-mcp/target/release/context-database-mcp\"; [ -x \"$BIN\" ] || exit 0; CTXDB_CLIENT_ID=claude-code \"$BIN\" --recent 5 2>/dev/null || true",
        "timeout": 15
      }]
    }]
  }
}
```

The guard matters: this hook runs in *every* project, including ones that have never used the
memory store. It swallows every failure and exits 0, skips silently when the binary is missing, and
touches no embedder — so a stopped database, a still-loading model, or an unrelated repo all produce
nothing rather than an error. On Windows the binary is `context-database-mcp.exe`.

The same block works per project in `<project>/.claude/settings.json` (committed, shared with anyone
who clones) or `<project>/.claude/settings.local.json` (git-ignored, yours only). Both merge with the
user-scope config rather than replacing it, so a hook defined in both places runs twice — pick one.
Project scope is worth it only when a single repo needs different behaviour, e.g. a larger
`--recent` count. Prefer the local variant for anything holding an absolute path: a committed hook
pointing at a binary that does not exist on someone else's disk fires on every session start of
theirs.

### 5. Optional: ingest compaction summaries

A memory store only saves tokens if it *replaces* context instead of adding to it. `/compact` leaves
a fat summary behind and charges for producing it; `/clear` is the real reset, and it is only cheap
if what the session learned is already stored. `--ingest` closes that: it reads a **PostCompact**
hook payload on stdin and saves the compaction summary as a `session-summary` memory.

PostCompact, not PreCompact — at PreCompact no summary exists yet, and compaction events only accept
`command` hooks, so the hook cannot summarise anything itself. Same scopes as above; user scope
(`~/.claude/settings.json`) is the sane default:

```json
{
  "hooks": {
    "PostCompact": [{
      "hooks": [{
        "type": "command",
        "command": "BIN=\"/ABSOLUTE/PATH/TO/context-database-mcp/target/release/context-database-mcp\"; [ -x \"$BIN\" ] || exit 0; CTXDB_CLIENT_ID=claude-code \"$BIN\" --ingest 2>/dev/null || true",
        "timeout": 60
      }]
    }]
  }
}
```

The summary is read from the transcript (`transcript_path` in the payload), where it is the last
entry flagged `isCompactSummary`; if a future payload carries the text directly, that is preferred.
The namespace comes from the payload's `cwd`, so the hook stays correct even when it runs from
somewhere else. Re-running the hook on the same summary is a no-op — an identical body in the same
namespace is treated as already stored. Like `--recent`, this mode prints nothing to stdout and
exits 0 on every failure: anything it printed would land straight in the freshly compacted context.

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
- Chunk boundaries are fixed at save time, so improvements to the chunker never reach rows already
  stored. There is no `--reindex` mode to re-chunk and re-embed them.

## CI

`.github/workflows/build.yml` runs clippy (`-D warnings`), `cargo test`, and a release build for
Windows and Linux on every push and PR. CI reaches neither Postgres nor the embedder, so it covers
the chunker and transcript parsing only — everything else is verified by hand against the local
stack.

Ordinary builds only upload the binaries as run artifacts, which expire. Binaries reach the
**Releases** page only on a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Assets are named `context-database-mcp-<target-triple>` (`.exe` on Windows), and the publish step is
a separate job that waits for every target, so a tag publishes all binaries or none.

## License

[GNU AGPL-3.0-only](LICENSE). Use it, change it, run it — but a modified version offered to others
over a network has to come with its source, which is the point of the network clause. This is a
memory store people will run as a service for themselves; AGPL keeps any hosted fork of it open.
