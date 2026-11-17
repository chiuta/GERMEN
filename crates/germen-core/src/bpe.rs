//! Byte-pair encoding (BPE) learned from a text corpus.
//!
//! The vocabulary starts with the 256 byte values; training then repeatedly merges the most
//! frequent pair of adjacent tokens into a new token until the requested vocabulary size is
//! reached. Because the base alphabet is *bytes*, every input is representable (no unknown
//! tokens) and decoding is exact for any bytes, valid UTF-8 or not.
//!
//! Text is first cut into "pre-tokens" (a word with its leading space, a run of digits, a run of
//! punctuation, a run of line breaks, …) and merges never cross those boundaries, so tokens stay
//! word-sized and the vocabulary is not wasted on pieces such as `. The`.

use crate::tokenizer::{MAX_VOCAB, TokenId};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

/// Longest pre-token in bytes; longer runs are cut. Keeps the cost of encoding a word bounded.
const MAX_PRETOKEN: usize = 64;

/// Training stops when the best pair occurs fewer times than this.
const MIN_PAIR_COUNT: i64 = 2;

/// Inputs larger than this are encoded on several threads.
const PARALLEL_ENCODE_BYTES: usize = 4 << 20;

/// Cheap multiplicative hasher (the same idea as rustc's FxHash). The maps below are keyed by
/// small integers and short byte strings of our own data, where SipHash's DoS resistance is
/// wasted effort.
#[derive(Default, Clone, Copy)]
struct Fx(u64);

impl Fx {
    #[inline]
    fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut v = 0u64;
            for (i, &b) in chunk.iter().enumerate() {
                v |= (b as u64) << (8 * i);
            }
            self.add(v);
        }
        self.add(bytes.len() as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

#[inline]
fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t' | 0x0B | 0x0C)
}

/// 0 = letter-like (ASCII letters and every byte of a multi-byte UTF-8 character), 1 = digit,
/// 2 = other ASCII (punctuation and control bytes).
#[inline]
fn class(b: u8) -> u8 {
    match b {
        b'a'..=b'z' | b'A'..=b'Z' | 0x80..=0xFF => 0,
        b'0'..=b'9' => 1,
        _ => 2,
    }
}

/// End of the pre-token that starts at `i` (`i < data.len()`).
fn pretoken_end(data: &[u8], i: usize) -> usize {
    let n = data.len();
    let mut j = i;
    // A single space glues to the word that follows it (" word").
    if data[j] == b' ' && j + 1 < n && !is_whitespace(data[j + 1]) {
        j += 1;
    }
    if is_whitespace(data[j]) {
        while j < n && is_whitespace(data[j]) {
            j += 1;
        }
        // In "  word" the last space belongs to the word, the rest is a pre-token of its own.
        if j < n && j - i > 1 && data[j - 1] == b' ' {
            j -= 1;
        }
    } else {
        let c = class(data[j]);
        while j < n && !is_whitespace(data[j]) && class(data[j]) == c {
            j += 1;
        }
    }
    j.min(i + MAX_PRETOKEN).max(i + 1)
}

/// Iterator over the byte ranges of the pre-tokens of a text. The ranges are contiguous and
/// cover the whole input.
pub struct PreTokens<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> PreTokens<'a> {
    /// Splits `data` into pre-tokens.
    pub fn new(data: &'a [u8]) -> Self {
        PreTokens { data, pos: 0 }
    }
}

impl Iterator for PreTokens<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        if self.pos >= self.data.len() {
            return None;
        }
        let start = self.pos;
        self.pos = pretoken_end(self.data, start);
        Some(start..self.pos)
    }
}

/// A trained BPE vocabulary: the list of merges, in the order they were learned.
///
/// Token ids `0..256` are the single bytes; merge number `k` creates token `256 + k`.
#[derive(Clone, Debug)]
pub struct Bpe {
    merges: Vec<(TokenId, TokenId)>,
    /// Bytes of every token, indexed by id.
    tokens: Vec<Vec<u8>>,
    /// `(left, right)` → `(rank, merged id)`.
    ranks: FxMap<(TokenId, TokenId), (u32, TokenId)>,
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.merges == other.merges
    }
}

struct Word {
    syms: Vec<TokenId>,
    count: u64,
}

/// Replaces every non-overlapping occurrence of `pair` (scanning left to right) by `new_id`.
fn merge_pair(syms: &mut Vec<TokenId>, pair: (TokenId, TokenId), new_id: TokenId) {
    let (mut r, mut w) = (0, 0);
    while r < syms.len() {
        if r + 1 < syms.len() && syms[r] == pair.0 && syms[r + 1] == pair.1 {
            syms[w] = new_id;
            r += 2;
        } else {
            syms[w] = syms[r];
            r += 1;
        }
        w += 1;
    }
    syms.truncate(w);
}

impl Bpe {
    /// Rebuilds a vocabulary from its merge list (as stored in `tokenizer.json`).
    pub fn from_merges(merges: Vec<(TokenId, TokenId)>) -> Result<Bpe, String> {
        if 256 + merges.len() > MAX_VOCAB {
            return Err(format!(
                "vocabularul ar avea {} de tokeni; maximul e {MAX_VOCAB}",
                256 + merges.len()
            ));
        }
        let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut ranks: FxMap<(TokenId, TokenId), (u32, TokenId)> = FxMap::default();
        for (k, &(a, b)) in merges.iter().enumerate() {
            let known = 256 + k;
            if a as usize >= known || b as usize >= known {
                return Err(format!(
                    "îmbinarea {k} folosește tokenul {}, care nu există încă",
                    a.max(b)
                ));
            }
            // Training never builds a token longer than a pre-token, and a file that asks for one
            // is corrupt or hostile: each merge may double the longest token, so a few hundred
            // bytes of JSON could otherwise demand terabytes. Check the length before copying.
            let len = tokens[a as usize].len() + tokens[b as usize].len();
            if len > MAX_PRETOKEN {
                return Err(format!(
                    "îmbinarea {k} ar produce un token de {len} octeți; maximul este {MAX_PRETOKEN}"
                ));
            }
            let mut bytes = Vec::with_capacity(len);
            bytes.extend_from_slice(&tokens[a as usize]);
            bytes.extend_from_slice(&tokens[b as usize]);
            tokens.push(bytes);
            if ranks
                .insert((a, b), (k as u32, (256 + k) as TokenId))
                .is_some()
            {
                return Err(format!("îmbinarea {k} apare de două ori"));
            }
        }
        Ok(Bpe {
            merges,
            tokens,
            ranks,
        })
    }

    /// The merges in the order they were learned.
    pub fn merges(&self) -> &[(TokenId, TokenId)] {
        &self.merges
    }

    /// Number of tokens (256 bytes + merges).
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Bytes of token `id` (empty for an id outside the vocabulary).
    pub fn token_bytes(&self, id: TokenId) -> &[u8] {
        self.tokens.get(id as usize).map_or(&[], |t| t.as_slice())
    }

    /// Learns a vocabulary of (at most) `vocab_size` tokens from `data`.
    ///
    /// Training stops early when no pair of tokens occurs at least twice. `progress` is called
    /// with `(merges done, merges wanted)` every few merges. The result depends only on `data`
    /// and `vocab_size`.
    pub fn train(
        data: &[u8],
        vocab_size: usize,
        progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Bpe, String> {
        if !(257..=MAX_VOCAB).contains(&vocab_size) {
            return Err(format!(
                "--vocab trebuie să fie între 257 și {MAX_VOCAB} (primit {vocab_size})"
            ));
        }
        let wanted = vocab_size - 256;

        // Count distinct pre-tokens: BPE then works on (word, count) pairs, not on the text.
        let mut counts: FxMap<&[u8], u64> = FxMap::default();
        for r in PreTokens::new(data) {
            if r.len() > 1 {
                *counts.entry(&data[r]).or_insert(0) += 1;
            }
        }
        let mut words: Vec<Word> = counts
            .into_iter()
            .map(|(w, count)| Word {
                syms: w.iter().map(|&b| b as TokenId).collect(),
                count,
            })
            .collect();
        // HashMap order is arbitrary; sort so that everything below is reproducible.
        words.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.syms.cmp(&b.syms)));

        let mut pair_counts: FxMap<(TokenId, TokenId), i64> = FxMap::default();
        let mut pair_words: FxMap<(TokenId, TokenId), Vec<u32>> = FxMap::default();
        for (wi, w) in words.iter().enumerate() {
            for p in w.syms.windows(2) {
                let key = (p[0], p[1]);
                *pair_counts.entry(key).or_insert(0) += w.count as i64;
                pair_words.entry(key).or_default().push(wi as u32);
            }
        }
        // Largest count first; among equal counts the smaller pair, so ties are deterministic.
        let mut heap: BinaryHeap<(i64, Reverse<(TokenId, TokenId)>)> =
            pair_counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

        let mut merges: Vec<(TokenId, TokenId)> = Vec::with_capacity(wanted);
        while merges.len() < wanted {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            let current = pair_counts.get(&pair).copied().unwrap_or(0);
            if current != count {
                // Stale entry: counts only ever shrink for old pairs, so re-queue the real one.
                if current >= MIN_PAIR_COUNT {
                    heap.push((current, Reverse(pair)));
                }
                continue;
            }
            if count < MIN_PAIR_COUNT {
                break;
            }

            let new_id = (256 + merges.len()) as TokenId;
            merges.push(pair);

            let mut affected = pair_words.remove(&pair).unwrap_or_default();
            affected.sort_unstable();
            affected.dedup();
            let mut created: Vec<(TokenId, TokenId)> = Vec::new();
            for wi in affected {
                let w = &mut words[wi as usize];
                if !w.syms.windows(2).any(|p| (p[0], p[1]) == pair) {
                    continue; // registered earlier, merged away since
                }
                for p in w.syms.windows(2) {
                    *pair_counts.get_mut(&(p[0], p[1])).expect("counted") -= w.count as i64;
                }
                merge_pair(&mut w.syms, pair, new_id);
                for p in w.syms.windows(2) {
                    let key = (p[0], p[1]);
                    *pair_counts.entry(key).or_insert(0) += w.count as i64;
                    if p[0] == new_id || p[1] == new_id {
                        pair_words.entry(key).or_default().push(wi);
                        created.push(key);
                    }
                }
            }
            pair_counts.remove(&pair);
            created.sort_unstable();
            created.dedup();
            for key in created {
                heap.push((pair_counts[&key], Reverse(key)));
            }
            if merges.len().is_multiple_of(16) {
                progress(merges.len(), wanted);
            }
        }
        progress(merges.len(), wanted);
        Bpe::from_merges(merges)
    }

    /// Encodes one pre-token into `out` (cleared first), applying merges in the order they were
    /// learned.
    fn encode_word(&self, word: &[u8], out: &mut Vec<TokenId>) {
        out.clear();
        out.extend(word.iter().map(|&b| b as TokenId));
        loop {
            let mut best: Option<u32> = None;
            for p in out.windows(2) {
                if let Some(&(rank, _)) = self.ranks.get(&(p[0], p[1]))
                    && best.is_none_or(|b| rank < b)
                {
                    best = Some(rank);
                }
            }
            let Some(rank) = best else { break };
            merge_pair(
                out,
                self.merges[rank as usize],
                (256 + rank as usize) as TokenId,
            );
        }
    }

    fn encode_chunk(&self, data: &[u8], out: &mut Vec<TokenId>) {
        // Real text repeats its words a lot, so remember how each one was encoded.
        const CACHE_LIMIT: usize = 1 << 20;
        let mut cache: FxMap<&[u8], Range<u32>> = FxMap::default();
        let mut arena: Vec<TokenId> = Vec::new();
        let mut tmp: Vec<TokenId> = Vec::with_capacity(MAX_PRETOKEN);
        for r in PreTokens::new(data) {
            let word = &data[r];
            if word.len() == 1 {
                out.push(word[0] as TokenId);
            } else if let Some(span) = cache.get(word) {
                out.extend_from_slice(&arena[span.start as usize..span.end as usize]);
            } else {
                self.encode_word(word, &mut tmp);
                if cache.len() < CACHE_LIMIT {
                    let start = arena.len() as u32;
                    arena.extend_from_slice(&tmp);
                    cache.insert(word, start..arena.len() as u32);
                }
                out.extend_from_slice(&tmp);
            }
        }
    }

    /// Text → tokens. Large inputs are split at line breaks and encoded on all worker threads.
    pub fn encode(&self, data: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(data.len() / 3 + 16);
        if data.len() < PARALLEL_ENCODE_BYTES {
            self.encode_chunk(data, &mut out);
            return out;
        }
        // Cut right after a line break that is followed by a non-blank byte: that is always a
        // pre-token boundary, so cutting there changes nothing about the result.
        let target = (data.len() / (rayon::current_num_threads() * 4).max(1)).max(1 << 20);
        let mut cuts = vec![0usize];
        let mut from = target;
        while from < data.len() {
            let found = (from..data.len().min(from + (1 << 20)))
                .find(|&i| i > 0 && data[i - 1] == b'\n' && !is_whitespace(data[i]));
            match found {
                Some(i) => {
                    cuts.push(i);
                    from = i + target;
                }
                None => break, // no usable boundary nearby: keep the rest in one piece
            }
        }
        cuts.push(data.len());
        let parts: Vec<Vec<TokenId>> = cuts
            .windows(2)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|w| {
                let mut v = Vec::with_capacity((w[1] - w[0]) / 3 + 16);
                self.encode_chunk(&data[w[0]..w[1]], &mut v);
                v
            })
            .collect();
        for p in parts {
            out.extend(p);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::demo_corpus;

    fn pieces(s: &str) -> Vec<&str> {
        PreTokens::new(s.as_bytes())
            .map(|r| &s[r])
            .collect::<Vec<_>>()
    }

    #[test]
    fn pretokens_split_words_numbers_punctuation_and_blanks() {
        assert_eq!(
            pieces("Ana are 12 mere, și pere."),
            ["Ana", " are", " 12", " mere", ",", " și", " pere", "."]
        );
        assert_eq!(pieces("a  b"), ["a", " ", " b"]);
        assert_eq!(pieces("x\n\ny"), ["x", "\n\n", "y"]);
        assert_eq!(pieces("end "), ["end", " "]);
        assert_eq!(pieces(""), Vec::<&str>::new());
    }

    #[test]
    fn pretokens_always_cover_the_input_exactly() {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        };
        for len in [0usize, 1, 2, 7, 100, 1000, 5000] {
            let data: Vec<u8> = (0..len)
                .map(|_| match next() % 4 {
                    0 => b' ',
                    1 => b'\n',
                    2 => b'a' + next() % 26,
                    _ => next(),
                })
                .collect();
            let mut at = 0;
            for r in PreTokens::new(&data) {
                assert_eq!(r.start, at);
                assert!((1..=MAX_PRETOKEN).contains(&r.len()));
                at = r.end;
            }
            assert_eq!(at, data.len());
        }
        // A very long run is cut, not turned into one enormous pre-token.
        let long = vec![b'a'; 1000];
        assert!(PreTokens::new(&long).all(|r| r.len() <= MAX_PRETOKEN));
    }

    #[test]
    fn training_learns_frequent_words_and_roundtrips_exactly() {
        let text = demo_corpus(200_000, 3);
        let bpe = Bpe::train(text.as_bytes(), 600, &mut |_, _| {}).unwrap();
        assert!(bpe.vocab_size() > 300 && bpe.vocab_size() <= 600);

        let ids = bpe.encode(text.as_bytes());
        assert!(
            ids.len() * 3 < text.len(),
            "expected at least 3 bytes per token, got {} tokens for {} bytes",
            ids.len(),
            text.len()
        );
        let back: Vec<u8> = ids
            .iter()
            .flat_map(|&t| bpe.token_bytes(t).iter().copied())
            .collect();
        assert_eq!(back, text.as_bytes());
        assert!(ids.iter().all(|&t| (t as usize) < bpe.vocab_size()));
    }

    #[test]
    fn any_bytes_roundtrip_even_invalid_utf8() {
        let bpe = Bpe::train(demo_corpus(50_000, 1).as_bytes(), 400, &mut |_, _| {}).unwrap();
        let mut state = 99u64;
        let data: Vec<u8> = (0..4000)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 40) as u8
            })
            .collect();
        let ids = bpe.encode(&data);
        let back: Vec<u8> = ids
            .iter()
            .flat_map(|&t| bpe.token_bytes(t).iter().copied())
            .collect();
        assert_eq!(back, data);
    }

    #[test]
    fn training_is_deterministic_and_merges_are_valid() {
        let text = demo_corpus(80_000, 2);
        let a = Bpe::train(text.as_bytes(), 500, &mut |_, _| {}).unwrap();
        let b = Bpe::train(text.as_bytes(), 500, &mut |_, _| {}).unwrap();
        assert_eq!(a, b);
        // Rebuilding from the stored merges gives the same vocabulary.
        let c = Bpe::from_merges(a.merges().to_vec()).unwrap();
        assert_eq!(a, c);
        assert_eq!(a.token_bytes(300), c.token_bytes(300));
    }

    #[test]
    fn training_stops_when_nothing_repeats() {
        let bpe = Bpe::train(b"abcdefg", 300, &mut |_, _| {}).unwrap();
        assert_eq!(bpe.vocab_size(), 256, "no pair occurs twice");
        let bpe = Bpe::train(b"ab ab ab ab", 300, &mut |_, _| {}).unwrap();
        assert!(bpe.vocab_size() >= 257);
        assert_eq!(bpe.encode(b"ab").len(), 1, "\"ab\" became a single token");
    }

    #[test]
    fn hostile_merge_lists_cannot_exhaust_memory() {
        // Every merge below doubles the longest token, so 60 of them would ask for an exabyte
        // from a file of a few hundred bytes. A vocabulary must be refused before it is built.
        let mut doubling: Vec<(TokenId, TokenId)> = vec![(97, 97)];
        doubling.extend((0..60).map(|k| (256 + k, 256 + k)));
        let err = Bpe::from_merges(doubling).unwrap_err();
        assert!(err.contains("octeți"), "{err}");

        // 64 bytes is the longest token that training can produce; it must still load.
        let mut at_limit: Vec<(TokenId, TokenId)> = vec![(97, 97)]; // 2 bytes
        at_limit.extend((0..5).map(|k| (256 + k, 256 + k))); // 4, 8, 16, 32 and 64 bytes
        let bpe = Bpe::from_merges(at_limit.clone()).unwrap();
        assert_eq!(bpe.token_bytes(261).len(), 64);
        at_limit.push((261, 261)); // 128 bytes
        assert!(Bpe::from_merges(at_limit).is_err());
    }

    #[test]
    fn invalid_merge_lists_are_rejected() {
        assert!(Bpe::from_merges(vec![(300, 1)]).is_err(), "undefined token");
        assert!(Bpe::from_merges(vec![(1, 2), (1, 2)]).is_err(), "duplicate");
        assert!(Bpe::train(b"hello", 100, &mut |_, _| {}).is_err());
    }

    #[test]
    fn parallel_encoding_equals_sequential_encoding() {
        let bpe = Bpe::train(demo_corpus(100_000, 4).as_bytes(), 700, &mut |_, _| {}).unwrap();
        // Larger than the parallel threshold.
        let big = demo_corpus(PARALLEL_ENCODE_BYTES + 500_000, 6);
        let mut sequential = Vec::new();
        bpe.encode_chunk(big.as_bytes(), &mut sequential);
        let parallel = bpe.encode(big.as_bytes());
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn repeated_letters_follow_the_left_to_right_rule() {
        // "aaa aaa aaa": the pair (a, a) is merged left to right, leaving a trailing "a".
        let bpe = Bpe::train(b"aaa aaa aaa aaa", 260, &mut |_, _| {}).unwrap();
        let ids = bpe.encode(b"aaa");
        let back: Vec<u8> = ids
            .iter()
            .flat_map(|&t| bpe.token_bytes(t).iter().copied())
            .collect();
        assert_eq!(back, b"aaa");
        assert!(ids.len() <= 2);
    }
}
