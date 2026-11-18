//! Corpus handling, random batches and a built-in synthetic demo corpus.

use crate::tokenizer::{TokenId, Tokenizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use std::path::Path;

/// Small, fast, dependency-free PRNG (xorshift64*). Not cryptographic.
#[derive(Clone, Debug)]
pub struct Rng(u64);

/// One round of SplitMix64: a bijection on `u64` that spreads every bit of its input over the whole
/// output, so that seeds which differ in a single bit start unrelated sequences.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Creates a generator. Different seeds give different sequences (all but one pair of the 2⁶⁴
    /// seeds, because the all-zero state, which xorshift cannot leave, is remapped).
    ///
    /// Up to 0.2.0 the seed was only XOR-ed with a constant and its lowest bit forced to 1, so
    /// seeds `2k` and `2k + 1` produced the same numbers.
    pub fn new(seed: u64) -> Self {
        let state = splitmix64(seed);
        Rng(if state == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            state
        })
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform integer in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((((self.next_u64() >> 11) as u128) * (n as u128)) >> 53) as usize
    }

    /// Uniform float in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Which part of the corpus to draw from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// Training data.
    Train,
    /// Held-out validation data.
    Val,
}

/// A text corpus, already turned into tokens and split into a training and a validation part.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// Training tokens.
    pub train: Vec<TokenId>,
    /// Validation tokens (the tail of the text).
    pub val: Vec<TokenId>,
    /// The tokenizer that produced the tokens; it is stored with every checkpoint.
    pub tokenizer: Tokenizer,
    /// Size of the original text in bytes.
    pub bytes: usize,
}

/// Up to `budget` bytes of `data`, taken as evenly spaced 1 MiB blocks so that a very large
/// corpus is represented by all of its parts, not only by its beginning.
pub fn sample_blocks(data: &[u8], budget: usize) -> Vec<u8> {
    const BLOCK: usize = 1 << 20;
    if data.len() <= budget {
        return data.to_vec();
    }
    let blocks = (budget / BLOCK).max(1);
    let stride = data.len() / blocks;
    let mut out = Vec::with_capacity(blocks * BLOCK);
    for i in 0..blocks {
        let start = i * stride;
        out.extend_from_slice(&data[start..(start + BLOCK).min(data.len())]);
    }
    out
}

/// Where the validation tail starts in a text of `len` bytes (everything before it is training
/// data). A tokenizer should be trained on `text[..split_point(..)]` only, so that the
/// validation text stays unseen.
///
/// Never panics: a fraction that is NaN or outside `[0, 1]` is clamped here, and it is
/// [`Corpus::build`] that rejects it with a message.
pub fn split_point(len: usize, val_fraction: f64) -> usize {
    let fraction = if val_fraction.is_nan() {
        0.0
    } else {
        val_fraction.clamp(0.0, 1.0)
    };
    len - (((len as f64) * fraction) as usize).min(len)
}

impl Corpus {
    /// Tokenizes `bytes` with `tokenizer` and splits off the last `val_fraction` as validation
    /// data (split on the text first, then each part is tokenized on its own).
    pub fn build(bytes: &[u8], val_fraction: f64, tokenizer: Tokenizer) -> Result<Corpus, String> {
        if !(0.0..0.5).contains(&val_fraction) {
            return Err("val_fraction trebuie să fie în [0, 0.5)".into());
        }
        if bytes.len() < 64 {
            return Err(format!(
                "corpusul e prea mic ({} octeți; minim 64)",
                bytes.len()
            ));
        }
        let split = split_point(bytes.len(), val_fraction);
        let train = tokenizer.encode(&bytes[..split]);
        let val = tokenizer.encode(&bytes[split..]);
        Ok(Corpus {
            train,
            val,
            tokenizer,
            bytes: bytes.len(),
        })
    }

    /// Byte-level corpus: splits `bytes`, the last `val_fraction` becoming validation data.
    pub fn from_bytes(bytes: Vec<u8>, val_fraction: f64) -> Result<Corpus, String> {
        Corpus::build(&bytes, val_fraction, Tokenizer::Byte)
    }

    /// Reads a file as raw bytes and splits it (byte-level).
    pub fn from_file(path: &Path, val_fraction: f64) -> Result<Corpus, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
        Corpus::from_bytes(bytes, val_fraction)
    }

    /// Total number of tokens.
    pub fn len(&self) -> usize {
        self.train.len() + self.val.len()
    }

    /// Whether the corpus holds no tokens at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Average bytes of text per token (1.0 for the byte tokenizer; 3–4 for a good BPE).
    pub fn bytes_per_token(&self) -> f64 {
        self.bytes as f64 / self.len().max(1) as f64
    }

    /// Checks that a window of `seq_len + 1` tokens fits in the training part
    /// (and in the validation part when there is one).
    pub fn check_seq(&self, seq_len: usize) -> Result<(), String> {
        if self.train.len() <= seq_len + 1 {
            return Err(format!(
                "partea de antrenare ({} tokeni) e prea mică pentru contextul {seq_len}; \
                 reduceți --seq sau folosiți un corpus mai mare",
                self.train.len()
            ));
        }
        if !self.val.is_empty() && self.val.len() <= seq_len + 1 {
            return Err(format!(
                "partea de validare ({} tokeni) e prea mică pentru contextul {seq_len}; \
                 reduceți --seq sau --val-fraction",
                self.val.len()
            ));
        }
        Ok(())
    }

    fn part(&self, split: Split) -> &[TokenId] {
        match split {
            Split::Train => &self.train,
            Split::Val => {
                if self.val.is_empty() {
                    &self.train
                } else {
                    &self.val
                }
            }
        }
    }

    /// Draws `batch` random windows of `seq` tokens; targets are the inputs shifted by one.
    ///
    /// Panics if the chosen part holds fewer than `seq + 2` tokens (see [`Corpus::check_seq`]).
    pub fn sample_batch(&self, split: Split, batch: usize, seq: usize, rng: &mut Rng) -> Batch {
        let data = self.part(split);
        assert!(data.len() > seq + 1, "corpus part too small for seq={seq}");
        let max_start = data.len() - seq - 1;
        let mut inputs = Vec::with_capacity(batch * seq);
        let mut targets = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            let o = rng.below(max_start + 1);
            inputs.extend(data[o..o + seq].iter().map(|&b| b as i32));
            targets.extend(data[o + 1..o + seq + 1].iter().map(|&b| b as i32));
        }
        Batch {
            inputs,
            targets,
            batch,
            seq,
        }
    }
}

/// A batch of token windows in row-major `[batch, seq]` layout.
#[derive(Clone, Debug)]
pub struct Batch {
    /// Input tokens.
    pub inputs: Vec<i32>,
    /// Next-token targets (inputs shifted by one position).
    pub targets: Vec<i32>,
    /// Number of windows.
    pub batch: usize,
    /// Tokens per window.
    pub seq: usize,
}

impl Batch {
    /// Uploads the batch to `device` as `(inputs, targets)`, both `[batch, seq]`.
    pub fn to_tensors(&self, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let shape = [self.batch, self.seq];
        let x = Tensor::<2, Int>::from_data(TensorData::new(self.inputs.clone(), shape), device);
        let y = Tensor::<2, Int>::from_data(TensorData::new(self.targets.clone(), shape), device);
        (x, y)
    }

    /// Splits the windows into `parts` consecutive batches whose sizes differ by at most one
    /// (the larger ones first). `parts` is clamped to `1..=batch`, so no part is ever empty.
    pub fn split(&self, parts: usize) -> Vec<Batch> {
        let parts = parts.clamp(1, self.batch.max(1));
        let (each, extra) = (self.batch / parts, self.batch % parts);
        let mut start = 0;
        (0..parts)
            .map(|i| {
                let rows = each + usize::from(i < extra);
                let (from, to) = (start * self.seq, (start + rows) * self.seq);
                start += rows;
                Batch {
                    inputs: self.inputs[from..to].to_vec(),
                    targets: self.targets[from..to].to_vec(),
                    batch: rows,
                    seq: self.seq,
                }
            })
            .collect()
    }

    /// A batch of uniformly random tokens, for benchmarks (content is irrelevant to speed).
    pub fn random(batch: usize, seq: usize, vocab: usize, rng: &mut Rng) -> Batch {
        let n = batch * seq;
        let inputs: Vec<i32> = (0..n).map(|_| rng.below(vocab) as i32).collect();
        let targets: Vec<i32> = (0..n).map(|_| rng.below(vocab) as i32).collect();
        Batch {
            inputs,
            targets,
            batch,
            seq,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Synthetic demo corpus
// ---------------------------------------------------------------------------------------------

const SUBJECTS_SG: &[&str] = &[
    "Ana",
    "Mihai",
    "profesorul",
    "copilul",
    "bunica",
    "pisica",
    "vulpea",
    "inginerul",
    "doctorița",
    "pescarul",
    "croitoreasa",
    "astronomul",
];
const SUBJECTS_PL: &[&str] = &[
    "copiii",
    "elevii",
    "vecinii",
    "păsările",
    "cercetătorii",
    "călătorii",
    "meșterii",
];
/// (singular, plural) present-tense forms.
const VERBS: &[(&str, &str)] = &[
    ("citește", "citesc"),
    ("vede", "văd"),
    ("caută", "caută"),
    ("găsește", "găsesc"),
    ("desenează", "desenează"),
    ("ascultă", "ascultă"),
    ("construiește", "construiesc"),
    ("repară", "repară"),
    ("poartă", "poartă"),
    ("așteaptă", "așteaptă"),
];
const OBJECTS: &[&str] = &[
    "o carte veche",
    "un cântec frumos",
    "harta orașului",
    "luna plină",
    "un copac înalt",
    "o poveste lungă",
    "un telescop mic",
    "o scrisoare",
    "pădurea întunecată",
    "un pod de piatră",
    "o lampă albastră",
    "vechea fântână",
];
const PLACES: &[&str] = &[
    "dimineața",
    "seara",
    "în parc",
    "la școală",
    "acasă",
    "pe malul râului",
    "în bibliotecă",
    "pe munte",
    "lângă fereastră",
    "în fața casei",
];
const LINKS: &[&str] = &[", apoi ", ", dar ", ", iar apoi ", ", și "];

fn pick<'a>(rng: &mut Rng, list: &[&'a str]) -> &'a str {
    list[rng.below(list.len())]
}

fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Generates roughly `target_bytes` of synthetic Romanian prose from a tiny grammar.
///
/// The text is invented for this tool (no third-party content, no rights to clear). It has
/// real structure — agreement between subject and verb, diacritics, punctuation, paragraphs —
/// so a model visibly learns it within minutes, which makes it a good smoke test and a
/// first thing to train on. It is NOT meant to produce a useful language model.
pub fn demo_corpus(target_bytes: usize, seed: u64) -> String {
    let mut rng = Rng::new(seed);
    let mut out = String::with_capacity(target_bytes + 256);
    let mut sentences_in_paragraph = 0;
    while out.len() < target_bytes {
        let plural = rng.below(4) == 0;
        let subject = if plural {
            pick(&mut rng, SUBJECTS_PL)
        } else {
            pick(&mut rng, SUBJECTS_SG)
        };
        let verb = |rng: &mut Rng| {
            let (s, p) = VERBS[rng.below(VERBS.len())];
            if plural { p } else { s }
        };
        let v1 = verb(&mut rng);
        let o1 = pick(&mut rng, OBJECTS);
        let mut s = format!("{} {} {}", capitalise(subject), v1, o1);
        if rng.below(3) == 0 {
            let link = pick(&mut rng, LINKS);
            let v2 = verb(&mut rng);
            let o2 = pick(&mut rng, OBJECTS);
            s.push_str(&format!("{link}{v2} {o2}"));
        }
        if rng.below(2) == 0 {
            s.push(' ');
            s.push_str(pick(&mut rng, PLACES));
        }
        s.push_str(". ");
        out.push_str(&s);
        sentences_in_paragraph += 1;
        if sentences_in_paragraph >= 3 + rng.below(4) {
            out.push('\n');
            sentences_in_paragraph = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitting_a_batch_keeps_every_window_once_and_in_order() {
        let mut rng = Rng::new(3);
        let batch = Batch::random(7, 5, 256, &mut rng);
        for parts in [1, 2, 3, 7, 50] {
            let pieces = batch.split(parts);
            assert_eq!(pieces.len(), parts.min(7));
            assert!(pieces.iter().all(|p| p.batch >= 1 && p.seq == 5));
            let (max, min) = (
                pieces.iter().map(|p| p.batch).max().unwrap(),
                pieces.iter().map(|p| p.batch).min().unwrap(),
            );
            assert!(max - min <= 1, "{parts}: sizes {max}/{min}");
            let inputs: Vec<i32> = pieces.iter().flat_map(|p| p.inputs.clone()).collect();
            let targets: Vec<i32> = pieces.iter().flat_map(|p| p.targets.clone()).collect();
            assert_eq!(inputs, batch.inputs);
            assert_eq!(targets, batch.targets);
        }
        assert_eq!(batch.split(0).len(), 1, "zero parts means one");
    }

    #[test]
    fn every_seed_gives_its_own_stream() {
        // Up to 0.2.0 the seeds 2k and 2k+1 started the very same sequence (`(seed ^ C) | 1`
        // throws the lowest bit away), so "--seed 2" and "--seed 3" trained on identical data.
        let mut seen = std::collections::HashSet::new();
        for seed in 0..2000u64 {
            let mut rng = Rng::new(seed);
            let stream: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
            assert!(
                seen.insert(stream),
                "seed {seed} repeats the stream of an earlier seed"
            );
        }
    }

    #[test]
    fn neighbouring_seeds_are_not_correlated() {
        // The first draw of seeds 0, 1, 2, … must look unrelated: about half of the bits differ
        // between neighbours (xorshift on a barely different state would not manage that).
        let first = |seed: u64| Rng::new(seed).next_u64();
        let mut differing = 0u32;
        let pairs = 500u32;
        for seed in 0..pairs as u64 {
            differing += (first(seed) ^ first(seed + 1)).count_ones();
        }
        let mean = differing as f64 / pairs as f64;
        assert!((28.0..36.0).contains(&mean), "mean differing bits: {mean}");
    }

    #[test]
    fn split_point_survives_nonsense_fractions() {
        assert_eq!(split_point(1000, 0.05), 950);
        assert_eq!(split_point(1000, 0.0), 1000);
        for fraction in [f64::NAN, -1.0, 7.0, f64::INFINITY, f64::NEG_INFINITY, 1e300] {
            let at = split_point(1000, fraction);
            assert!(at <= 1000, "split_point(1000, {fraction}) = {at}");
        }
        // The corpus builder is what rejects them, with a message.
        for fraction in [f64::NAN, -0.1, 0.5, 7.0] {
            assert!(Corpus::from_bytes(vec![b'a'; 1000], fraction).is_err());
        }
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let x = a.below(10);
            assert_eq!(x, b.below(10));
            assert!(x < 10);
        }
        let f = a.next_f32();
        assert!((0.0..1.0).contains(&f));
        // Seed 0 must not get stuck at zero.
        let mut z = Rng::new(0);
        assert_ne!(z.next_u64(), 0);
    }

    #[test]
    fn rng_covers_all_values() {
        let mut r = Rng::new(1);
        let mut seen = [false; 7];
        for _ in 0..500 {
            seen[r.below(7)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn corpus_split_and_batches_are_consistent() {
        let text: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let c = Corpus::from_bytes(text, 0.1).unwrap();
        assert_eq!(c.val.len(), 100);
        assert_eq!(c.train.len(), 900);
        c.check_seq(32).unwrap();
        let mut rng = Rng::new(3);
        let b = c.sample_batch(Split::Train, 4, 32, &mut rng);
        assert_eq!(b.inputs.len(), 4 * 32);
        for row in 0..4 {
            for t in 0..31 {
                // The target at position t is the input at position t + 1.
                assert_eq!(b.targets[row * 32 + t], b.inputs[row * 32 + t + 1]);
            }
        }
        let v = c.sample_batch(Split::Val, 2, 16, &mut rng);
        assert_eq!(v.inputs.len(), 32);
    }

    #[test]
    fn corpus_rejects_tiny_input_and_long_windows() {
        assert!(Corpus::from_bytes(vec![1; 10], 0.1).is_err());
        let c = Corpus::from_bytes(vec![7; 200], 0.1).unwrap();
        assert!(c.check_seq(500).is_err());
    }

    #[test]
    fn demo_corpus_is_deterministic_valid_utf8_with_diacritics() {
        let a = demo_corpus(5000, 11);
        let b = demo_corpus(5000, 11);
        assert_eq!(a, b);
        assert_ne!(a, demo_corpus(5000, 12));
        assert!(a.len() >= 5000);
        assert!(a.chars().any(|c| "ăâîșț".contains(c)));
        assert!(a.contains('\n'));
    }

    #[test]
    fn a_sample_is_the_whole_text_when_it_fits_and_evenly_spaced_blocks_when_not() {
        let small = vec![7u8; 1000];
        assert_eq!(sample_blocks(&small, 2000), small);
        assert_eq!(sample_blocks(&small, 1000), small);

        // Three MiB of text, each MiB a different byte: a 2 MiB budget takes two blocks, one from
        // the beginning and one from the middle (stride = len / blocks), not only the beginning.
        let mut big = Vec::new();
        for fill in [1u8, 2, 3] {
            big.extend(std::iter::repeat_n(fill, 1 << 20));
        }
        let sample = sample_blocks(&big, 2 << 20);
        assert_eq!(sample.len(), 2 << 20);
        assert_eq!(sample[0], 1);
        assert_eq!(
            sample[(1 << 20) + 1],
            2,
            "the second block starts in the second MiB"
        );
        // A budget below one block still takes one whole block.
        assert_eq!(sample_blocks(&big, 10).len(), 1 << 20);
    }
}
