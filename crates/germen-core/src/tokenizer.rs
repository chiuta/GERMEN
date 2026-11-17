//! Tokenizers: turning bytes of text into the integers a model works with, and back.
//!
//! Two kinds exist:
//!
//! - **Byte** — every byte is its own token (vocabulary of 256). Needs no training, has no
//!   unknown tokens and round-trips any input exactly. The price is long sequences: a Romanian
//!   diacritic costs two tokens and a word about six.
//! - **BPE** — a vocabulary learned from the training text ([`crate::bpe`]), typically 2 000 to
//!   32 000 tokens. Common words become single tokens, so the same context window covers three
//!   to four times more text and the model learns faster per unit of compute. It is still built
//!   on bytes, so it never meets an unknown token either.
//!
//! A trained tokenizer is stored next to the weights (`tokenizer.json`), because a model is
//! useless without the exact vocabulary it was trained with.

use crate::bpe::Bpe;
use crate::fsutil::Staged;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// One token id. Vocabularies are limited to [`MAX_VOCAB`] entries so that ids fit 16 bits.
pub type TokenId = u16;

/// Vocabulary size of the byte tokenizer.
pub const BYTE_VOCAB: usize = 256;

/// Largest supported vocabulary.
pub const MAX_VOCAB: usize = 65_536;

/// File name of the tokenizer inside a checkpoint directory.
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The 256 single-byte strings, so [`Tokenizer::token_bytes`] can hand out slices.
static BYTES: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t
};

/// A tokenizer: either raw bytes or a learned BPE vocabulary.
#[derive(Clone, Debug, PartialEq)]
pub enum Tokenizer {
    /// One token per byte.
    Byte,
    /// Learned byte-pair encoding.
    Bpe(Bpe),
}

/// On-disk form of a tokenizer.
#[derive(Serialize, Deserialize)]
struct Stored {
    kind: String,
    vocab_size: usize,
    #[serde(default)]
    merges: Vec<(TokenId, TokenId)>,
}

impl Tokenizer {
    /// Number of distinct tokens.
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Byte => BYTE_VOCAB,
            Tokenizer::Bpe(b) => b.vocab_size(),
        }
    }

    /// Short name: `"byte"` or `"bpe"`.
    pub fn name(&self) -> &'static str {
        match self {
            Tokenizer::Byte => "byte",
            Tokenizer::Bpe(_) => "bpe",
        }
    }

    /// Text (any bytes) → tokens.
    pub fn encode(&self, data: &[u8]) -> Vec<TokenId> {
        match self {
            Tokenizer::Byte => data.iter().map(|&b| b as TokenId).collect(),
            Tokenizer::Bpe(b) => b.encode(data),
        }
    }

    /// The bytes a token stands for (empty for an id outside the vocabulary).
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        match self {
            Tokenizer::Byte => BYTES.get(id as usize..id as usize + 1).unwrap_or(&[]),
            Tokenizer::Bpe(b) => b.token_bytes(id),
        }
    }

    /// Tokens → bytes. Exactly the inverse of [`Tokenizer::encode`].
    pub fn decode_bytes(&self, ids: &[TokenId]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            out.extend_from_slice(self.token_bytes(id));
        }
        out
    }

    /// Tokens → text. Invalid UTF-8 (a model can emit it) becomes U+FFFD.
    pub fn decode(&self, ids: &[TokenId]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Like [`Tokenizer::decode`], but silently drops a character that was cut off at the very
    /// end (a generation limit can stop in the middle of `ș`). Use it for text shown to people.
    pub fn decode_complete(&self, ids: &[TokenId]) -> String {
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for &b in &self.decode_bytes(ids) {
            out.push_str(&stream.push(b));
        }
        out
    }

    /// The content of `tokenizer.json` for this tokenizer.
    pub fn to_json(&self) -> Result<String, String> {
        let stored = match self {
            Tokenizer::Byte => Stored {
                kind: "byte".into(),
                vocab_size: BYTE_VOCAB,
                merges: Vec::new(),
            },
            Tokenizer::Bpe(b) => Stored {
                kind: "bpe".into(),
                vocab_size: b.vocab_size(),
                merges: b.merges().to_vec(),
            },
        };
        serde_json::to_string(&stored).map_err(|e| e.to_string())
    }

    /// Writes `tokenizer.json` into `dir` (created if missing), atomically and durably: a
    /// crash leaves either the old file or the new one, never a half-written one.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let mut staged = Staged::new(dir)?;
        staged.bytes(TOKENIZER_FILE, self.to_json()?.as_bytes())?;
        staged.commit()
    }

    /// Reads the tokenizer stored in `dir`. A directory without `tokenizer.json` (every
    /// checkpoint from before GERMEN 0.2) means the byte tokenizer.
    pub fn load(dir: &Path) -> Result<Tokenizer, String> {
        let path = dir.join(TOKENIZER_FILE);
        if !path.exists() {
            return Ok(Tokenizer::Byte);
        }
        let text = fs::read_to_string(&path)
            .map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
        let stored: Stored = serde_json::from_str(&text)
            .map_err(|e| format!("{} e invalid: {e}", path.display()))?;
        match stored.kind.as_str() {
            "byte" => Ok(Tokenizer::Byte),
            "bpe" => {
                let bpe = Bpe::from_merges(stored.merges)
                    .map_err(|e| format!("{} e invalid: {e}", path.display()))?;
                if bpe.vocab_size() != stored.vocab_size {
                    return Err(format!(
                        "{}: vocabularul declarat ({}) nu se potrivește cu îmbinările ({})",
                        path.display(),
                        stored.vocab_size,
                        bpe.vocab_size()
                    ));
                }
                Ok(Tokenizer::Bpe(bpe))
            }
            other => Err(format!(
                "{}: tip de tokenizator necunoscut „{other}”",
                path.display()
            )),
        }
    }
}

/// Turns a stream of bytes into text without ever splitting a multi-byte character.
///
/// Feed bytes one by one with [`Utf8Stream::push`]; it returns whatever text became complete
/// (often an empty string while a character such as `ș` is still half-received). Bytes that can
/// never form valid UTF-8 become U+FFFD.
#[derive(Clone, Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Creates an empty stream.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one byte; returns the text completed by it (possibly empty).
    pub fn push(&mut self, byte: u8) -> String {
        self.pending.push(byte);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_owned();
                self.pending.clear();
                out
            }
            Err(e) if e.error_len().is_none() => {
                // Valid so far, but the last character is incomplete: emit the valid part only.
                let valid = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
            Err(_) => {
                let out = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                out
            }
        }
    }

    /// Adds all bytes of a token; returns the text completed by them.
    pub fn push_token(&mut self, tokenizer: &Tokenizer, id: TokenId) -> String {
        let mut out = String::new();
        for &b in tokenizer.token_bytes(id) {
            out.push_str(&self.push(b));
        }
        out
    }

    /// Flushes an unfinished character (as U+FFFD). Call once at the end of the stream.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::demo_corpus;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-tok-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn utf8_stream_never_splits_characters() {
        let text = "Ștefan învață ț și î în Brăila!";
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for b in text.bytes() {
            let piece = stream.push(b);
            // Every emitted piece must itself be valid, whole characters.
            assert!(
                !piece.contains('\u{FFFD}'),
                "split inside a character: {piece:?}"
            );
            out.push_str(&piece);
        }
        out.push_str(&stream.finish());
        assert_eq!(out, text);
    }

    #[test]
    fn decode_complete_drops_only_a_truncated_last_character() {
        let t = Tokenizer::Byte;
        let bytes = t.encode("repară".as_bytes()); // ends with the two bytes of 'ă'
        let cut = &bytes[..bytes.len() - 1];
        assert_eq!(t.decode_complete(cut), "repar");
        assert!(
            t.decode(cut).ends_with('\u{FFFD}'),
            "plain decode shows the cut"
        );
        assert_eq!(t.decode_complete(&bytes), "repară");
        assert_eq!(t.decode_complete(&[97, 0xff, 98]), "a\u{FFFD}b");
    }

    #[test]
    fn utf8_stream_replaces_invalid_bytes_and_flushes_truncated_tail() {
        let mut stream = Utf8Stream::new();
        assert_eq!(stream.push(0xff), "\u{FFFD}");
        assert_eq!(stream.push(b'a'), "a");
        assert_eq!(stream.push(0xc8), ""); // first half of 'ș' (C8 99)
        assert_eq!(stream.finish(), "\u{FFFD}");
    }

    #[test]
    fn byte_roundtrip_preserves_romanian_diacritics() {
        let t = Tokenizer::Byte;
        let s = "Ștefan și țăranul din Brăila învață limba română: ă â î ș ț Ă Â Î Ș Ț.";
        let enc = t.encode(s.as_bytes());
        assert!(enc.len() > s.chars().count(), "diacritics take two bytes");
        assert_eq!(t.decode(&enc), s);
        assert_eq!(t.vocab_size(), 256);
    }

    #[test]
    fn decode_never_panics_on_garbage() {
        let t = Tokenizer::Byte;
        let out = t.decode(&[0xff, 0xfe, 97, 0xc3, 60000]);
        assert!(out.contains('a'));
    }

    #[test]
    fn bpe_tokenizer_roundtrips_and_compresses_romanian_text() {
        let text = demo_corpus(120_000, 11);
        let bpe = Bpe::train(text.as_bytes(), 500, &mut |_, _| {}).unwrap();
        let t = Tokenizer::Bpe(bpe);
        let sample = "Ștefan și țăranul învață limba română în Brăila.";
        let ids = t.encode(sample.as_bytes());
        assert_eq!(t.decode(&ids), sample);
        assert!(t.vocab_size() > 256);
        let corpus_ids = t.encode(text.as_bytes());
        assert!(corpus_ids.len() < text.len() / 2);
    }

    #[test]
    fn streaming_a_bpe_token_by_token_equals_decoding_it_whole() {
        let text = demo_corpus(60_000, 12);
        let t = Tokenizer::Bpe(Bpe::train(text.as_bytes(), 400, &mut |_, _| {}).unwrap());
        let sample = "Doctorița învață țărănci și băieți: ș ț ă â î!";
        let ids = t.encode(sample.as_bytes());
        let mut stream = Utf8Stream::new();
        let mut out = String::new();
        for &id in &ids {
            let piece = stream.push_token(&t, id);
            assert!(!piece.contains('\u{FFFD}'), "split a character: {piece:?}");
            out.push_str(&piece);
        }
        out.push_str(&stream.finish());
        assert_eq!(out, sample);
    }

    #[test]
    fn save_and_load_roundtrip_for_both_kinds() {
        let dir = scratch("roundtrip");
        // No file at all means the byte tokenizer (checkpoints from before 0.2).
        assert_eq!(Tokenizer::load(&dir).unwrap(), Tokenizer::Byte);
        Tokenizer::Byte.save(&dir).unwrap();
        assert_eq!(Tokenizer::load(&dir).unwrap(), Tokenizer::Byte);

        let text = demo_corpus(60_000, 13);
        let t = Tokenizer::Bpe(Bpe::train(text.as_bytes(), 450, &mut |_, _| {}).unwrap());
        t.save(&dir).unwrap();
        assert!(!dir.join("tokenizer.json.tmp").exists());
        let back = Tokenizer::load(&dir).unwrap();
        assert_eq!(back, t);
        assert_eq!(
            back.encode("Ana citește.".as_bytes()),
            t.encode("Ana citește.".as_bytes())
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_tokenizer_files_give_clear_errors() {
        let dir = scratch("corrupt");
        fs::create_dir_all(&dir).unwrap();
        let write = |s: &str| fs::write(dir.join(TOKENIZER_FILE), s).unwrap();
        write("not json");
        assert!(Tokenizer::load(&dir).is_err());
        write(r#"{"kind":"wordpiece","vocab_size":10}"#);
        assert!(Tokenizer::load(&dir).unwrap_err().contains("necunoscut"));
        write(r#"{"kind":"bpe","vocab_size":258,"merges":[[300,1]]}"#);
        assert!(Tokenizer::load(&dir).is_err());
        // 60 merges, each doubling the longest token: a few hundred bytes of JSON that used to
        // ask for an exabyte.
        let doubling: Vec<String> = std::iter::once("[97,97]".to_string())
            .chain((0..60).map(|k| format!("[{},{}]", 256 + k, 256 + k)))
            .collect();
        write(&format!(
            r#"{{"kind":"bpe","vocab_size":317,"merges":[{}]}}"#,
            doubling.join(",")
        ));
        assert!(Tokenizer::load(&dir).unwrap_err().contains("octeți"));
        write(r#"{"kind":"bpe","vocab_size":999,"merges":[[97,98]]}"#);
        assert!(
            Tokenizer::load(&dir)
                .unwrap_err()
                .contains("nu se potrivește")
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
