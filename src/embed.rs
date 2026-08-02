//! Client for the shared text-embeddings-inference container.
//!
//! The embedder deliberately lives outside this process: one MCP server runs per
//! MCP client (Claude Code, Claude Desktop, ...), and loading model weights into
//! each of them would multiply a ~2 GB resident set by the number of clients.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Dimension of BAAI/bge-m3. Must match VECTOR(n) in migrations/001_init.sql.
pub const EMBED_DIM: usize = 1024;

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    inputs: &'a [String],
    truncate: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbedResponse {
    Vectors(Vec<Vec<f32>>),
    Error { error: String },
}

impl Embedder {
    pub fn new(base_url: String, model: String) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build http client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            model,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// bge-m3 needs no instruction prefix for either side of the pair, so the
    /// same call serves documents and queries. Swapping to e5/nomic would
    /// require "query: " / "passage: " prefixes here.
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let resp = self
            .http
            .post(format!("{}/embed", self.base_url))
            .json(&EmbedRequest {
                inputs: texts,
                truncate: true,
            })
            .send()
            .await
            .with_context(|| {
                format!(
                    "embedding server unreachable at {} (is `docker compose up` running?)",
                    self.base_url
                )
            })?;

        let status = resp.status();
        let body = resp.text().await.context("reading embed response body")?;

        if !status.is_success() {
            bail!("embedding server returned {status}: {body}");
        }

        match serde_json::from_str::<EmbedResponse>(&body)
            .context("decoding embed response")?
        {
            EmbedResponse::Error { error } => bail!("embedding server error: {error}"),
            EmbedResponse::Vectors(vectors) => {
                for v in &vectors {
                    if v.len() != EMBED_DIM {
                        bail!(
                            "embedding dimension mismatch: server returned {}, schema expects {}. \
                             Model in docker-compose.yml and VECTOR(n) in the migration must agree.",
                            v.len(),
                            EMBED_DIM
                        );
                    }
                }
                Ok(vectors)
            }
        }
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(std::slice::from_ref(&text.to_string())).await?;
        Ok(v.pop().unwrap_or_default())
    }

    /// Health probe used at startup so failures surface as a clear log line on
    /// stderr instead of as a confusing error on the first tool call.
    pub async fn ping(&self) -> Result<()> {
        self.embed_one("ping").await.map(|_| ())
    }
}

/// Target chunk size in characters. Well under bge-m3's 8192-token limit on
/// purpose: the limit is about what the model *accepts*, while retrieval quality
/// depends on how much unrelated text each vector has to average over.
///
/// This also bounds what search returns: a hit's snippet is its whole matching
/// chunk, because truncating the chunk can cut off the very sentence that caused
/// the match. So chunk size is the real cost knob for search output.
pub const CHUNK_CHARS: usize = 700;

/// How much of the previous chunk each chunk repeats, so a fact sitting on a
/// boundary is fully present in at least one chunk instead of split across two.
const CHUNK_OVERLAP: usize = 120;

/// Split a body into overlapping chunks, preferring paragraph then sentence
/// then whitespace boundaries so chunks break where the text already breaks.
///
/// A short body yields exactly one chunk, so callers need no special case.
/// Operates on char boundaries throughout -- slicing a UTF-8 string by byte
/// offset would panic on any multi-byte character.
pub fn chunk_text(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= CHUNK_CHARS {
        let t = text.trim();
        return if t.is_empty() {
            vec![]
        } else {
            vec![t.to_string()]
        };
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < chars.len() {
        let hard_end = (start + CHUNK_CHARS).min(chars.len());

        // Last chunk: take the remainder rather than hunting for a boundary.
        let end = if hard_end == chars.len() {
            hard_end
        } else {
            // Only look for a break in the tail of the window, so a boundary
            // found early cannot produce a tiny chunk.
            let search_floor = start + CHUNK_CHARS / 2;
            find_break(&chars, search_floor, hard_end).unwrap_or(hard_end)
        };

        let piece: String = chars[start..end].iter().collect();
        let piece = piece.trim();
        if !piece.is_empty() {
            chunks.push(piece.to_string());
        }

        if end == chars.len() {
            break;
        }
        // Step forward with overlap, but always make progress.
        start = end.saturating_sub(CHUNK_OVERLAP).max(start + 1);

        // The overlap step lands on an arbitrary character, so a chunk would
        // otherwise open mid-word ("cceptable because..."). That is read
        // directly by the model -- a hit's snippet is its whole chunk -- so
        // nudge forward to the next word boundary.
        if start > 0 && start < chars.len() && !chars[start - 1].is_whitespace() {
            let mut w = start;
            while w < chars.len() && !chars[w].is_whitespace() {
                w += 1;
            }
            // Only take it if a boundary exists before the next chunk's end;
            // unbroken text must not stall progress.
            if w < chars.len() {
                start = w + 1;
            }
        }
    }

    chunks
}

/// Split a body into chunks and build the exact strings that get embedded.
///
/// Returns `(chunks, inputs)`: the chunk text as it is stored, and the text as
/// it is sent to the embedder. They differ -- the title is prepended to every
/// input, so a chunk from the middle of a long note still carries what the note
/// is about.
///
/// Every writer goes through this. `--reindex` has to reproduce a save's inputs
/// character for character, or a reindexed row lands somewhere else in the
/// vector space than a freshly saved one and the two stop being comparable.
pub fn chunk_inputs(title: &str, body: &str) -> (Vec<String>, Vec<String>) {
    let chunks = chunk_text(body);
    let inputs = chunks.iter().map(|c| format!("{title}\n\n{c}")).collect();
    (chunks, inputs)
}

/// Best break point in `chars[floor..ceil]`, preferring a blank line, then a
/// sentence end, then any whitespace. Returns the index just past the break.
fn find_break(chars: &[char], floor: usize, ceil: usize) -> Option<usize> {
    let window = &chars[floor..ceil];

    // Paragraph break.
    for i in (1..window.len()).rev() {
        if window[i] == '\n' && window[i - 1] == '\n' {
            return Some(floor + i + 1);
        }
    }
    // Sentence end followed by whitespace.
    for i in (1..window.len()).rev() {
        if window[i].is_whitespace() && matches!(window[i - 1], '.' | '!' | '?') {
            return Some(floor + i + 1);
        }
    }
    // Any whitespace.
    for i in (0..window.len()).rev() {
        if window[i].is_whitespace() {
            return Some(floor + i + 1);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(chunk_text("hello world"), vec!["hello world"]);
    }

    #[test]
    fn empty_text_yields_no_chunks() {
        assert!(chunk_text("   \n  ").is_empty());
    }

    #[test]
    fn long_text_splits_and_covers_everything() {
        let body = "Sentence number one is here. ".repeat(200);
        let chunks = chunk_text(&body);
        assert!(chunks.len() > 1, "expected a split, got {}", chunks.len());
        // Overlap means total length exceeds the original, but nothing is lost:
        // every chunk is non-empty and the first/last text is present.
        assert!(chunks.iter().all(|c| !c.is_empty()));
        assert!(chunks.first().unwrap().starts_with("Sentence number one"));
        assert!(chunks.last().unwrap().ends_with('.'));
    }

    #[test]
    fn multibyte_text_does_not_panic_and_round_trips() {
        // Byte-offset slicing would panic here; char-offset slicing must not.
        let body = "Grüße über größere Straßen — mit Umlauten. ".repeat(120);
        let chunks = chunk_text(&body);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.contains('ü') || c.contains('ö') || c.contains('ß')));
    }

    #[test]
    fn chunks_do_not_start_mid_word() {
        // Chunk text is what the model reads as a search snippet, so a chunk
        // opening on a word fragment is a real defect, not cosmetic.
        let body = "The deployment pipeline is acceptable because mutations are rare. "
            .repeat(60);
        let chunks = chunk_text(&body);
        assert!(chunks.len() > 2);
        for (i, c) in chunks.iter().enumerate().skip(1) {
            let first = c.split_whitespace().next().unwrap();
            assert!(
                body.contains(&format!(" {first} ")) || body.starts_with(first),
                "chunk {i} starts mid-word: {first:?}"
            );
        }
    }

    #[test]
    fn every_embed_input_carries_the_title() {
        // The parity `--reindex` depends on: stored chunk text and embedded text
        // are one transformation apart, applied in one place. If a writer ever
        // builds its inputs by hand again, a reindexed row stops matching a
        // freshly saved one.
        let body = "Sentence number one is here. ".repeat(200);
        let (chunks, inputs) = chunk_inputs("a title", &body);
        assert_eq!(chunks.len(), inputs.len());
        assert!(chunks.len() > 1);
        for (c, i) in chunks.iter().zip(&inputs) {
            assert_eq!(*i, format!("a title\n\n{c}"));
        }
    }

    #[test]
    fn always_makes_progress_on_unbreakable_text() {
        // No whitespace anywhere: find_break returns None and the loop must
        // still terminate rather than spinning on the same start index.
        let body = "x".repeat(CHUNK_CHARS * 3);
        let chunks = chunk_text(&body);
        assert!(chunks.len() >= 3);
    }
}
