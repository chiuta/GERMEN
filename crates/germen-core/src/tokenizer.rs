//! Byte-level tokenizer.
//!
//! Every byte is its own token (vocabulary of 256). It needs no training, never
//! produces an unknown token and round-trips any UTF-8 text exactly. The price is
//! longer sequences than a learned subword vocabulary (a Romanian diacritic costs
//! two tokens); a BPE tokenizer is planned as the next step.

/// Size of the byte-level vocabulary.
pub const VOCAB_SIZE: usize = 256;

/// Stateless byte-level tokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    /// Vocabulary size (always 256).
    pub fn vocab_size(&self) -> usize {
        VOCAB_SIZE
    }

    /// Text → tokens.
    pub fn encode(&self, text: &str) -> Vec<u8> {
        text.as_bytes().to_vec()
    }

    /// Tokens → text. Invalid UTF-8 sequences (a model can emit them) become U+FFFD.
    pub fn decode(&self, tokens: &[u8]) -> String {
        String::from_utf8_lossy(tokens).into_owned()
    }

    /// Like [`ByteTokenizer::decode`], but silently drops a character that was cut off at the very
    /// end (a generation limit can stop in the middle of `ș`). Use it for text shown to people.
    pub fn decode_complete(&self, tokens: &[u8]) -> String {
        let mut stream = Utf8Stream::new();
        let mut out = String::with_capacity(tokens.len());
        for &b in tokens {
            out.push_str(&stream.push(b));
        }
        out
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
        let t = ByteTokenizer;
        let bytes = "repară".as_bytes(); // ends with the two bytes of 'ă'
        let cut = &bytes[..bytes.len() - 1];
        assert_eq!(t.decode_complete(cut), "repar");
        assert!(
            t.decode(cut).ends_with('\u{FFFD}'),
            "plain decode shows the cut"
        );
        assert_eq!(t.decode_complete(bytes), "repară");
        assert_eq!(t.decode_complete(&[b'a', 0xff, b'b']), "a\u{FFFD}b");
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
    fn roundtrip_preserves_romanian_diacritics() {
        let t = ByteTokenizer;
        let s = "Ștefan și țăranul din Brăila învață limba română: ă â î ș ț Ă Â Î Ș Ț.";
        let enc = t.encode(s);
        assert!(enc.len() > s.chars().count(), "diacritics take two bytes");
        assert_eq!(t.decode(&enc), s);
    }

    #[test]
    fn decode_never_panics_on_garbage() {
        let t = ByteTokenizer;
        let out = t.decode(&[0xff, 0xfe, b'a', 0xc3]);
        assert!(out.contains('a'));
    }
}
