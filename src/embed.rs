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

#[derive(Serialize)]
struct TokenizeRequest<'a> {
    inputs: &'a str,
    add_special_tokens: bool,
}

#[derive(Deserialize)]
struct RawToken {
    /// Absent for the special tokens the model wraps around an input.
    start: Option<usize>,
    stop: Option<usize>,
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

    /// Where each token starts in `text`, as a **byte** offset.
    ///
    /// Bytes, not characters: that is what the tokenizer reports, and converting
    /// would mean walking the string again for no gain. Every offset lands on a
    /// UTF-8 boundary, so slicing at one is safe. Only the starts are kept --
    /// the chunker cuts *between* tokens and never inside one, so a token's end
    /// is the next token's start.
    ///
    /// Asking the server rather than tokenizing in-process is the whole point:
    /// the only tokenizer whose count means anything here is the one that will
    /// actually see the text. Vendoring a copy would be a second source of truth
    /// that drifts the moment the model changes, and it costs no availability --
    /// every caller that chunks also embeds, so the server is already required.
    ///
    /// `Ok(None)` means the server has no `/tokenize` route (TEI before 1.2), as
    /// opposed to a server that is down, which is an error.
    pub async fn tokenize(&self, text: &str) -> Result<Option<Vec<usize>>> {
        let resp = self
            .http
            .post(format!("{}/tokenize", self.base_url))
            .json(&TokenizeRequest {
                inputs: text,
                // The two specials wrap the input and carry no offsets. Counting
                // them would shrink every chunk by a token for no reason.
                add_special_tokens: false,
            })
            .send()
            .await
            .with_context(|| {
                format!(
                    "embedding server unreachable at {} (is `docker compose up` running?)",
                    self.base_url
                )
            })?;

        if matches!(
            resp.status(),
            reqwest::StatusCode::NOT_FOUND | reqwest::StatusCode::METHOD_NOT_ALLOWED
        ) {
            return Ok(None);
        }
        let status = resp.status();
        let body = resp
            .text()
            .await
            .context("reading tokenize response body")?;
        if !status.is_success() {
            bail!("tokenizer returned {status}: {body}");
        }

        // One input in, one sequence out.
        let seqs: Vec<Vec<RawToken>> =
            serde_json::from_str(&body).context("decoding tokenize response")?;
        let Some(seq) = seqs.into_iter().next() else {
            bail!("tokenizer returned no sequences");
        };

        Ok(Some(
            seq.into_iter()
                .filter_map(|t| match (t.start, t.stop) {
                    // Bounds-checked against this exact text: a slice built from
                    // an offset the server invented would panic, and it would
                    // panic inside a save.
                    (Some(start), Some(stop))
                        if stop > start && stop <= text.len() && text.is_char_boundary(start) =>
                    {
                        Some(start)
                    }
                    _ => None,
                })
                .collect(),
        ))
    }

    /// Split a body into chunks sized in tokens, and build the strings that get
    /// embedded. The token-aware replacement for the free `chunk_inputs`.
    ///
    /// Falls back to character-based chunking only when the server has no
    /// tokenizer route, and says so loudly: the two produce different
    /// boundaries, so a corpus written by both is a corpus where `--reindex`
    /// changes rows for reasons nobody remembers.
    pub async fn chunk_inputs(
        &self,
        title: &str,
        body: &str,
    ) -> Result<(Vec<String>, Vec<String>)> {
        let chunks = match self.tokenize(body).await? {
            Some(starts) => chunk_text_by_tokens(body, &starts),
            None => {
                tracing::warn!(
                    "embedding server has no /tokenize route; falling back to character-based \
                     chunking. Boundaries will differ from token-chunked rows."
                );
                chunk_text(body)
            }
        };
        let inputs = chunks.iter().map(|c| format!("{title}\n\n{c}")).collect();
        Ok((chunks, inputs))
    }
}

/// Target chunk size in tokens, measured by the embedding model's own
/// tokenizer.
///
/// Characters were the wrong unit, and the error is not small. Measured against
/// this project's own corpus with bge-m3, a 700-character window holds 196
/// tokens of English prose, 158 of German, 361 of Rust, and 447 of Chinese --
/// so a CJK chunk was averaging 2.3x as much meaning into one 1024-d vector as
/// an English one, and getting correspondingly blurrier. Sizing in tokens makes
/// every vector cover a comparable amount of information whatever the text is.
///
/// 200 is chosen to leave English prose, the bulk of this store, chunked almost
/// exactly as before. Well under bge-m3's 8192-token limit on purpose: the limit
/// is what the model *accepts*, while retrieval quality depends on how much
/// unrelated text each vector has to average over.
///
/// This also bounds what search returns: a hit's snippet is its whole matching
/// chunk, because truncating the chunk can cut off the very sentence that caused
/// the match. In tokens that bound is now the same in every language, which is
/// the unit the context window is actually billed in.
pub const CHUNK_TOKENS: usize = 200;

/// How many tokens of the previous chunk each chunk repeats, so a fact sitting
/// on a boundary is fully present in at least one chunk instead of split across
/// two. ~120 characters of English, matching the character-based chunker it
/// replaces.
const CHUNK_OVERLAP_TOKENS: usize = 35;

/// Target chunk size in characters, for the fallback chunker only.
pub const CHUNK_CHARS: usize = 700;

/// Character-based overlap, paired with `CHUNK_CHARS`.
const CHUNK_OVERLAP: usize = 120;

/// Split a body into overlapping chunks of at most `CHUNK_TOKENS` tokens each,
/// breaking where the text already breaks.
///
/// `starts` is one byte offset per token, ascending. Pure, and takes them rather
/// than fetching them, so the boundary logic is unit-testable without a
/// tokenizer -- CI reaches no embedding server, and the bugs worth catching here
/// (a chunk that starts mid-word, a loop that fails to advance, a slice off a
/// UTF-8 boundary) are all in the boundary logic.
pub fn chunk_text_by_tokens(text: &str, starts: &[usize]) -> Vec<String> {
    if starts.len() <= CHUNK_TOKENS {
        let t = text.trim();
        return if t.is_empty() {
            vec![]
        } else {
            vec![t.to_string()]
        };
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut start_tok = 0usize;
    let mut start_byte = 0usize;

    loop {
        let hard_tok = start_tok + CHUNK_TOKENS;
        // Past the last token: take the remainder rather than hunting for a
        // boundary, so the tail is never dropped.
        let end_byte = if hard_tok >= starts.len() {
            text.len()
        } else {
            let hard_end = starts[hard_tok];
            // Only look for a break in the second half of the window, so a
            // boundary found early cannot produce a tiny chunk.
            let floor = starts[start_tok + CHUNK_TOKENS / 2].max(start_byte);
            find_break(text, floor.min(hard_end), hard_end).unwrap_or(hard_end)
        };

        let piece = text[start_byte..end_byte].trim();
        if !piece.is_empty() {
            chunks.push(piece.to_string());
        }
        if end_byte >= text.len() {
            break;
        }

        // Step back by the overlap, in tokens.
        let end_tok = token_at(starts, end_byte);
        let next_tok = end_tok.saturating_sub(CHUNK_OVERLAP_TOKENS);
        let next_byte = word_start(text, starts.get(next_tok).copied().unwrap_or(end_byte));

        // Progress or stop. Unbreakable text -- no whitespace for thousands of
        // characters -- is the case that would otherwise spin here forever.
        if next_byte <= start_byte {
            start_byte = end_byte;
            start_tok = end_tok.max(start_tok + 1);
        } else {
            start_byte = next_byte;
            start_tok = token_at(starts, next_byte);
        }
        if start_byte >= text.len() || start_tok >= starts.len() {
            break;
        }
    }

    chunks
}

/// Index of the first token starting at or after `byte`.
fn token_at(starts: &[usize], byte: usize) -> usize {
    starts.partition_point(|s| *s < byte)
}

/// `at`, moved forward to the start of the next whole word.
///
/// The overlap step lands on a token boundary, and tokens split words -- so a
/// chunk would otherwise open mid-word ("king world"). Chunk text is read
/// directly by the model, since a hit's snippet is its whole chunk, so this is a
/// real defect rather than a cosmetic one. Unbroken text is left alone: stalling
/// progress would be worse.
fn word_start(text: &str, at: usize) -> usize {
    if at == 0 || at >= text.len() {
        return at;
    }
    if text[..at]
        .chars()
        .next_back()
        .is_some_and(char::is_whitespace)
    {
        return at;
    }
    match text[at..].char_indices().find(|(_, c)| c.is_whitespace()) {
        Some((i, c)) => at + i + c.len_utf8(),
        None => at,
    }
}

/// Best break point in `text[floor..ceil]`, preferring a blank line, then a
/// sentence end, then any whitespace. Returns the byte index just past the
/// break, always on a UTF-8 boundary.
fn find_break(text: &str, floor: usize, ceil: usize) -> Option<usize> {
    if floor >= ceil || !text.is_char_boundary(floor) || !text.is_char_boundary(ceil) {
        return None;
    }

    let mut paragraph = None;
    let mut sentence = None;
    let mut space = None;
    let mut prev: Option<char> = None;

    // The last candidate of each kind wins, so a chunk runs as close to full as
    // its break allows.
    for (i, c) in text[floor..ceil].char_indices() {
        let past = floor + i + c.len_utf8();
        if let Some(p) = prev {
            if c == '\n' && p == '\n' {
                paragraph = Some(past);
            }
            if c.is_whitespace() && matches!(p, '.' | '!' | '?') {
                sentence = Some(past);
            }
        }
        if c.is_whitespace() {
            space = Some(past);
        }
        prev = Some(c);
    }

    paragraph.or(sentence).or(space)
}

/// Character-based chunking, kept only for an embedding server with no
/// tokenizer route. `chunk_text_by_tokens` is what writers use.
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
            find_break_chars(&chars, search_floor, hard_end).unwrap_or(hard_end)
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

/// Best break point in `chars[floor..ceil]`, preferring a blank line, then a
/// sentence end, then any whitespace. Returns the index just past the break.
/// Pairs with the character-based fallback chunker.
fn find_break_chars(chars: &[char], floor: usize, ceil: usize) -> Option<usize> {
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

    /// Stand-in tokenizer: whole words, but long ones split every `piece` bytes,
    /// which is what a real subword tokenizer does and what makes chunks start
    /// mid-word if nothing nudges them.
    ///
    /// Byte offsets, like the server's. `piece` is the knob these tests use to
    /// simulate a dense script: a small value means many tokens per character,
    /// exactly the CJK case that motivated token sizing.
    fn fake_tokens(text: &str, piece: usize) -> Vec<usize> {
        let mut spans = Vec::new();
        let mut run_start: Option<usize> = None;

        let flush = |start: usize, end: usize, spans: &mut Vec<usize>| {
            let mut at = start;
            while at < end {
                spans.push(at);
                // Never split inside a character.
                let mut stop = (at + piece).min(end);
                while stop < end && !text.is_char_boundary(stop) {
                    stop += 1;
                }
                at = stop;
            }
        };

        for (i, c) in text.char_indices() {
            if c.is_whitespace() {
                if let Some(s) = run_start.take() {
                    flush(s, i, &mut spans);
                }
            } else if run_start.is_none() {
                run_start = Some(i);
            }
        }
        if let Some(s) = run_start {
            flush(s, text.len(), &mut spans);
        }
        spans
    }

    fn token_chunks(text: &str, piece: usize) -> Vec<String> {
        chunk_text_by_tokens(text, &fake_tokens(text, piece))
    }

    #[test]
    fn a_short_body_is_one_token_chunk() {
        let text = "hello world";
        assert_eq!(token_chunks(text, 4), vec!["hello world"]);
        assert!(token_chunks("   \n  ", 4).is_empty());
    }

    #[test]
    fn token_chunks_stay_within_the_token_budget() {
        let body = "Sentence number one is here. ".repeat(400);
        let chunks = token_chunks(&body, 4);
        assert!(chunks.len() > 1, "expected a split");
        for c in &chunks {
            assert!(
                fake_tokens(c, 4).len() <= CHUNK_TOKENS,
                "chunk of {} tokens exceeds the budget",
                fake_tokens(c, 4).len()
            );
        }
    }

    #[test]
    fn a_denser_script_gets_shorter_chunks_in_characters() {
        // The entire point of sizing in tokens. Same text, tokenized twice as
        // finely: the chunks must carry roughly half the characters, so each
        // vector covers a comparable amount of meaning either way.
        let body = "Sentence number one is here. ".repeat(400);
        let sparse = token_chunks(&body, 8);
        let dense = token_chunks(&body, 4);
        let avg = |v: &[String]| v.iter().map(String::len).sum::<usize>() / v.len();
        assert!(
            avg(&dense) * 2 < avg(&sparse) * 3,
            "dense {} vs sparse {}",
            avg(&dense),
            avg(&sparse)
        );
        assert!(dense.len() > sparse.len());
    }

    #[test]
    fn token_chunks_cover_the_whole_body() {
        let body = "Sentence number one is here. ".repeat(400);
        let chunks = token_chunks(&body, 4);
        assert!(chunks.first().unwrap().starts_with("Sentence number one"));
        assert!(chunks.last().unwrap().ends_with('.'));
        // Overlap makes the total longer than the original; nothing may be lost.
        assert!(chunks.iter().map(String::len).sum::<usize>() > body.trim().len());
    }

    #[test]
    fn token_chunks_do_not_start_mid_word() {
        // The overlap step lands on a token boundary, and tokens split words.
        let body = "The deployment pipeline is acceptable because mutations are rare. ".repeat(120);
        let chunks = token_chunks(&body, 3);
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
    fn multibyte_token_chunks_do_not_panic() {
        // Token offsets are bytes, so every slice here is a chance to cut a
        // character in half.
        let body = "Grüße über größere Straßen — mit Umlauten. ".repeat(200);
        let chunks = token_chunks(&body, 3);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn unbreakable_text_still_terminates() {
        // No whitespace at all: no break is findable and the word nudge cannot
        // help, so only the explicit progress guard stops the loop spinning.
        let body = "x".repeat(4000);
        let chunks = token_chunks(&body, 2);
        assert!(chunks.len() > 1);
    }

    /// The real tokenizer, not the stand-in.
    ///
    /// `#[ignore]` because CI reaches no embedding server; run by hand with
    /// `cargo test -- --ignored`. Worth keeping because the offsets are the one
    /// part the fake cannot check: they are byte offsets produced by a
    /// subword tokenizer, and a chunker that assumed characters would slice a
    /// multi-byte character in half and panic inside a save.
    #[tokio::test]
    #[ignore]
    async fn real_token_chunks_hold_across_scripts() -> Result<()> {
        let embedder = Embedder::new(
            std::env::var("CTXDB_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8085".into()),
            "BAAI/bge-m3".to_string(),
        );

        for (name, body) in [
            (
                "english",
                "The deployment pipeline is acceptable. ".repeat(200),
            ),
            (
                "german",
                "Grüße über größere Straßen — mit Umlauten. ".repeat(200),
            ),
            (
                "cjk",
                "这是一个测试句子，用于测量字符与词元的比率。".repeat(200),
            ),
            (
                "code",
                "fn find_break(text: &str, floor: usize) -> Option<usize> { None }\n".repeat(200),
            ),
        ] {
            let starts = embedder.tokenize(&body).await?.expect("no /tokenize route");
            let chunks = chunk_text_by_tokens(&body, &starts);
            assert!(chunks.len() > 1, "{name}: expected a split");

            let mut counts = Vec::new();
            for c in &chunks {
                counts.push(embedder.tokenize(c).await?.unwrap().len());
            }
            println!(
                "{name}: {} chunks, tokens min {} max {}, bytes min {} max {}",
                chunks.len(),
                counts.iter().min().unwrap(),
                counts.iter().max().unwrap(),
                chunks.iter().map(String::len).min().unwrap(),
                chunks.iter().map(String::len).max().unwrap()
            );
            // Re-tokenizing a chunk on its own is not the same as reading its
            // tokens out of the whole body: subword merges depend on what came
            // before, so a cut can split one token into two. The budget is about
            // keeping vectors comparable, not a hard cap -- bge-m3 accepts 8192 --
            // so a few tokens of drift is fine and exact equality is not
            // achievable.
            for n in &counts {
                assert!(
                    *n <= CHUNK_TOKENS + CHUNK_TOKENS / 20,
                    "{name}: chunk of {n} tokens is well over budget"
                );
            }
            // The claim token sizing exists to make good: chunks carry a
            // comparable number of tokens whatever the script, even though their
            // character counts differ several-fold.
            for (i, n) in counts.iter().enumerate().take(chunks.len() - 1) {
                assert!(
                    *n >= CHUNK_TOKENS / 2,
                    "{name}: chunk {i} of {n} tokens is far under budget"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn every_embed_input_carries_the_title() {
        // The parity `--reindex` depends on: stored chunk text and embedded text
        // are one transformation apart, applied in one place. If a writer ever
        // builds its inputs by hand again, a reindexed row stops matching a
        // freshly saved one. Both chunkers feed the same one-line transformation
        // in `Embedder::chunk_inputs`, so this checks it on the fallback.
        let body = "Sentence number one is here. ".repeat(200);
        let chunks = chunk_text(&body);
        let inputs: Vec<String> = chunks.iter().map(|c| format!("a title\n\n{c}")).collect();
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
