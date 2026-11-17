//! SHA-256 (FIPS 180-4) for the manifest of a checkpoint.
//!
//! A checkpoint is several files that are only meaningful together. `meta.json` records the size
//! and the SHA-256 of each of the others, so that a damaged file (a bad sector, a half-finished
//! copy) and a file that belongs to another save (the weights of step 900 next to the optimizer of
//! step 600) are noticed instead of being trained on or sampled from.
//!
//! The sum protects against accidents, not against someone who can also rewrite `meta.json`.
//! SHA-256 is used rather than something faster because every operating system can recompute it
//! without GERMEN (`sha256sum`, `shasum -a 256`, PowerShell's `Get-FileHash`), which makes a
//! published checkpoint checkable by anyone.
//!
//! There is no dependency. The portable implementation is in safe Rust; on x86-64 processors that
//! have the SHA extensions (every AMD Zen, Intel since Ice Lake and Alder Lake) the compression
//! function runs on those instructions instead, found at run time, which is several times faster
//! (a one-gigabyte checkpoint is summed in about a second instead of five). Both are checked
//! against the NIST test vectors, against every message length around the block boundaries, and
//! against each other and a textbook version on random blocks.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// One round of the compression function. The eight working variables are not shuffled after each
/// round; instead the next round is written with the names rotated by one place (see
/// [`compress`]), which saves seven register moves per round.
macro_rules! round {
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $kw:expr) => {{
        let t1 = $h
            .wrapping_add($e.rotate_right(6) ^ $e.rotate_right(11) ^ $e.rotate_right(25))
            .wrapping_add($g ^ ($e & ($f ^ $g))) // Ch(e, f, g)
            .wrapping_add($kw);
        let t2 = ($a.rotate_right(2) ^ $a.rotate_right(13) ^ $a.rotate_right(22))
            .wrapping_add(($a & $b) | ($c & ($a | $b))); // Maj(a, b, c)
        $d = $d.wrapping_add(t1);
        $h = t1.wrapping_add(t2);
    }};
}

/// Processes one 64-byte block.
///
/// The message schedule is kept in a window of 16 words that is rewritten in place for each group
/// of 16 rounds, and the rounds are unrolled eight at a time. The textbook form of the same
/// function (a 64-word schedule and a loop that shuffles the variables) is kept in the tests as
/// the reference this one is compared with; this one is about twice as fast, which matters
/// because every load of a checkpoint hashes all of its files.
fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 16];
    for (word, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (group, k) in K.chunks_exact(16).enumerate() {
        if group > 0 {
            // W[t] = W[t-16] + s0(W[t-15]) + W[t-7] + s1(W[t-2]), in place: the slot of W[t-16] is
            // the slot of W[t], and the slots the other three words live in, `(t+1)`, `(t+9)` and
            // `(t+14)` modulo 16, hold them at the moment they are read.
            for i in 0..16 {
                let x = w[(i + 1) & 15];
                let y = w[(i + 14) & 15];
                let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
                let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
                w[i] = w[i]
                    .wrapping_add(s0)
                    .wrapping_add(w[(i + 9) & 15])
                    .wrapping_add(s1);
            }
        }
        for (k, w) in k.chunks_exact(8).zip(w.chunks_exact(8)) {
            round!(a, b, c, d, e, f, g, h, k[0].wrapping_add(w[0]));
            round!(h, a, b, c, d, e, f, g, k[1].wrapping_add(w[1]));
            round!(g, h, a, b, c, d, e, f, k[2].wrapping_add(w[2]));
            round!(f, g, h, a, b, c, d, e, k[3].wrapping_add(w[3]));
            round!(e, f, g, h, a, b, c, d, k[4].wrapping_add(w[4]));
            round!(d, e, f, g, h, a, b, c, k[5].wrapping_add(w[5]));
            round!(c, d, e, f, g, h, a, b, k[6].wrapping_add(w[6]));
            round!(b, c, d, e, f, g, h, a, k[7].wrapping_add(w[7]));
        }
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

/// Which implementation of the compression function a hasher uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    /// Safe Rust, runs everywhere.
    Portable,
    /// The SHA extensions of x86-64 processors.
    #[cfg(target_arch = "x86_64")]
    ShaNi,
}

impl Backend {
    /// The fastest backend this processor can run.
    fn detect() -> Backend {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("sha")
            && std::arch::is_x86_feature_detected!("sse2")
            && std::arch::is_x86_feature_detected!("ssse3")
            && std::arch::is_x86_feature_detected!("sse4.1")
        {
            return Backend::ShaNi;
        }
        Backend::Portable
    }

    /// Compresses every 64-byte block of `blocks` into `state`; `blocks.len()` is a multiple of 64.
    // The one `unsafe` call of this module: see the comment on it.
    #[allow(unsafe_code)]
    fn compress(self, state: &mut [u32; 8], blocks: &[u8]) {
        debug_assert_eq!(blocks.len() % 64, 0);
        match self {
            Backend::Portable => {
                for block in blocks.chunks_exact(64) {
                    compress(state, block.try_into().expect("chunks of 64 bytes"));
                }
            }
            #[cfg(target_arch = "x86_64")]
            Backend::ShaNi => {
                // SAFETY: `Backend::ShaNi` is only produced by `Backend::detect`, after the
                // processor reported every feature `shani::compress_blocks` is compiled for.
                unsafe { shani::compress_blocks(state, blocks) }
            }
        }
    }
}

/// The compression function on the SHA extensions (`sha256rnds2`, `sha256msg1`, `sha256msg2`).
///
/// Written with the byte loads and word stores done through plain integers, so that no raw pointer
/// is dereferenced; the only reason the function is not callable from safe code is that the CPU
/// might lack the instructions, which `Backend::detect` rules out.
#[cfg(target_arch = "x86_64")]
mod shani {
    use super::K;
    use core::arch::x86_64::*;

    /// Four rounds, with `$w` the four words of the message schedule for group `$i` (rounds
    /// `4 * $i` to `4 * $i + 3`). The instruction does two rounds at a time on the pair of
    /// registers (a, b, e, f) and (c, d, g, h).
    macro_rules! rounds4 {
        ($abef:ident, $cdgh:ident, $w:expr, $i:literal) => {{
            let k = _mm_set_epi32(
                K[4 * $i + 3] as i32,
                K[4 * $i + 2] as i32,
                K[4 * $i + 1] as i32,
                K[4 * $i] as i32,
            );
            let t1 = _mm_add_epi32($w, k);
            $cdgh = _mm_sha256rnds2_epu32($cdgh, $abef, t1);
            let t2 = _mm_shuffle_epi32::<0x0E>(t1);
            $abef = _mm_sha256rnds2_epu32($abef, $cdgh, t2);
        }};
    }

    /// The next four words of the schedule from the previous sixteen.
    macro_rules! schedule {
        ($v0:expr, $v1:expr, $v2:expr, $v3:expr) => {{
            let t = _mm_add_epi32(
                _mm_sha256msg1_epu32($v0, $v1),
                _mm_alignr_epi8::<4>($v3, $v2),
            );
            _mm_sha256msg2_epu32(t, $v3)
        }};
    }

    /// Compresses every 64-byte block of `blocks` into `state`.
    ///
    /// The processor must support the SHA extensions, SSE2, SSSE3 and SSE4.1.
    #[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
    pub(super) fn compress_blocks(state: &mut [u32; 8], blocks: &[u8]) {
        // Reverses the bytes of each 32-bit word: the message is big-endian.
        let swap = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b, 0x0405_0607_0001_0203);

        // The instructions want the state as (a, b, e, f) and (c, d, g, h), in reversed lanes.
        let dcba = _mm_set_epi32(
            state[3] as i32,
            state[2] as i32,
            state[1] as i32,
            state[0] as i32,
        );
        let hgfe = _mm_set_epi32(
            state[7] as i32,
            state[6] as i32,
            state[5] as i32,
            state[4] as i32,
        );
        let cdab = _mm_shuffle_epi32::<0xB1>(dcba);
        let efgh = _mm_shuffle_epi32::<0x1B>(hgfe);
        let mut abef = _mm_alignr_epi8::<8>(cdab, efgh);
        let mut cdgh = _mm_blend_epi16::<0xF0>(efgh, cdab);

        for block in blocks.chunks_exact(64) {
            let (abef_before, cdgh_before) = (abef, cdgh);
            let half = |i: usize| u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
            let mut w0 = _mm_shuffle_epi8(_mm_set_epi64x(half(1) as i64, half(0) as i64), swap);
            let mut w1 = _mm_shuffle_epi8(_mm_set_epi64x(half(3) as i64, half(2) as i64), swap);
            let mut w2 = _mm_shuffle_epi8(_mm_set_epi64x(half(5) as i64, half(4) as i64), swap);
            let mut w3 = _mm_shuffle_epi8(_mm_set_epi64x(half(7) as i64, half(6) as i64), swap);
            let mut w4;

            rounds4!(abef, cdgh, w0, 0);
            rounds4!(abef, cdgh, w1, 1);
            rounds4!(abef, cdgh, w2, 2);
            rounds4!(abef, cdgh, w3, 3);
            w4 = schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w4, 4);
            w0 = schedule!(w1, w2, w3, w4);
            rounds4!(abef, cdgh, w0, 5);
            w1 = schedule!(w2, w3, w4, w0);
            rounds4!(abef, cdgh, w1, 6);
            w2 = schedule!(w3, w4, w0, w1);
            rounds4!(abef, cdgh, w2, 7);
            w3 = schedule!(w4, w0, w1, w2);
            rounds4!(abef, cdgh, w3, 8);
            w4 = schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w4, 9);
            w0 = schedule!(w1, w2, w3, w4);
            rounds4!(abef, cdgh, w0, 10);
            w1 = schedule!(w2, w3, w4, w0);
            rounds4!(abef, cdgh, w1, 11);
            w2 = schedule!(w3, w4, w0, w1);
            rounds4!(abef, cdgh, w2, 12);
            w3 = schedule!(w4, w0, w1, w2);
            rounds4!(abef, cdgh, w3, 13);
            w4 = schedule!(w0, w1, w2, w3);
            rounds4!(abef, cdgh, w4, 14);
            w0 = schedule!(w1, w2, w3, w4);
            rounds4!(abef, cdgh, w0, 15);

            abef = _mm_add_epi32(abef, abef_before);
            cdgh = _mm_add_epi32(cdgh, cdgh_before);
        }

        let feba = _mm_shuffle_epi32::<0x1B>(abef);
        let dchg = _mm_shuffle_epi32::<0xB1>(cdgh);
        let dcba = _mm_blend_epi16::<0xF0>(feba, dchg);
        let hgef = _mm_alignr_epi8::<8>(dchg, feba);
        state[0] = _mm_extract_epi32::<0>(dcba) as u32;
        state[1] = _mm_extract_epi32::<1>(dcba) as u32;
        state[2] = _mm_extract_epi32::<2>(dcba) as u32;
        state[3] = _mm_extract_epi32::<3>(dcba) as u32;
        state[4] = _mm_extract_epi32::<0>(hgef) as u32;
        state[5] = _mm_extract_epi32::<1>(hgef) as u32;
        state[6] = _mm_extract_epi32::<2>(hgef) as u32;
        state[7] = _mm_extract_epi32::<3>(hgef) as u32;
    }
}

/// An incremental SHA-256.
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    length: u64,
    backend: Backend,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// A hasher that has seen nothing yet.
    pub fn new() -> Self {
        Self::with_backend(Backend::detect())
    }

    fn with_backend(backend: Backend) -> Self {
        Sha256 {
            state: INITIAL,
            block: [0; 64],
            filled: 0,
            length: 0,
            backend,
        }
    }

    /// Feeds `data` in.
    pub fn update(&mut self, mut data: &[u8]) {
        self.length = self.length.wrapping_add(data.len() as u64);
        if self.filled > 0 {
            let take = (64 - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled < 64 {
                return;
            }
            let block = self.block;
            self.backend.compress(&mut self.state, &block);
            self.filled = 0;
        }
        let whole = data.len() - data.len() % 64;
        self.backend.compress(&mut self.state, &data[..whole]);
        let rest = &data[whole..];
        self.block[..rest.len()].copy_from_slice(rest);
        self.filled = rest.len();
    }

    /// The digest of everything fed in.
    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.length.wrapping_mul(8);
        let mut padding = [0u8; 72];
        padding[0] = 0x80;
        // Pad with zeros to 56 bytes modulo 64, then the length in bits as a big-endian u64.
        let zeros = (119 - self.filled) % 64;
        padding[1 + zeros..1 + zeros + 8].copy_from_slice(&bits.to_be_bytes());
        self.update(&padding[..1 + zeros + 8]);
        debug_assert_eq!(self.filled, 0);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// Lowercase hexadecimal.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// SHA-256 of `data` as lowercase hexadecimal.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex(&hasher.finish())
}

/// Size in bytes and SHA-256 (lowercase hexadecimal) of the file at `path`.
pub fn file_sum(path: &Path) -> io::Result<(u64, String)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = match file.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        hasher.update(&buffer[..n]);
        total += n as u64;
    }
    Ok((total, hex(&hasher.finish())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every backend this processor can run: the portable one, and the fast one if there is one.
    fn backends() -> Vec<Backend> {
        let mut all = vec![Backend::Portable];
        if Backend::detect() != Backend::Portable {
            all.push(Backend::detect());
        }
        all
    }

    fn hex_with(backend: Backend, data: &[u8]) -> String {
        let mut hasher = Sha256::with_backend(backend);
        hasher.update(data);
        hex(&hasher.finish())
    }

    /// The textbook compression function: a 64-word schedule and a loop that shuffles the
    /// variables. It is what `compress` was before it was unrolled, kept as the oracle.
    fn compress_reference(state: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (word, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }

    #[test]
    fn every_backend_equals_the_textbook_compression() {
        // Random states and inputs of one to six blocks, and the extremes: every byte 0 and every
        // byte 0xFF, with the all-zero, all-ones and initial states.
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut cases: Vec<([u32; 8], Vec<u8>)> = Vec::new();
        for state in [[0; 8], [u32::MAX; 8], INITIAL] {
            for byte in [0u8, 0xFF] {
                cases.push((state, vec![byte; 64 * 3]));
            }
        }
        for _ in 0..10_000 {
            let mut state = [0u32; 8];
            for word in &mut state {
                *word = next() as u32;
            }
            let blocks = 1 + (next() % 6) as usize;
            let mut data = vec![0u8; 64 * blocks];
            for chunk in data.chunks_exact_mut(8) {
                chunk.copy_from_slice(&next().to_le_bytes());
            }
            cases.push((state, data));
        }
        for (state, data) in cases {
            let mut expected = state;
            for block in data.chunks_exact(64) {
                compress_reference(&mut expected, block.try_into().unwrap());
            }
            for backend in backends() {
                let mut got = state;
                backend.compress(&mut got, &data);
                assert_eq!(
                    got, expected,
                    "{backend:?}, state {state:x?}, data {data:x?}"
                );
            }
        }
    }

    /// A megabyte of pseudo-random bytes fed to every backend through `update` in pieces of
    /// awkward sizes (so that blocks straddle two calls) gives the digest of the whole.
    #[test]
    fn the_backends_agree_on_a_large_input_cut_into_awkward_pieces() {
        let mut seed = 42u64;
        let data: Vec<u8> = (0..(1 << 20) + 13)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 56) as u8
            })
            .collect();
        let reference = hex_with(Backend::Portable, &data);
        for backend in backends() {
            for piece in [1usize, 7, 63, 64, 65, 1000, 4096 + 3, 1 << 17] {
                let mut hasher = Sha256::with_backend(backend);
                for chunk in data.chunks(piece) {
                    hasher.update(chunk);
                }
                assert_eq!(
                    hex(&hasher.finish()),
                    reference,
                    "{backend:?}, pieces of {piece}"
                );
            }
        }
    }

    #[test]
    fn the_fast_backend_is_chosen_when_the_processor_has_it() {
        #[cfg(target_arch = "x86_64")]
        {
            let has = std::arch::is_x86_feature_detected!("sha")
                && std::arch::is_x86_feature_detected!("sse2")
                && std::arch::is_x86_feature_detected!("ssse3")
                && std::arch::is_x86_feature_detected!("sse4.1");
            assert_eq!(Backend::detect() == Backend::ShaNi, has);
            if !has {
                eprintln!("this processor has no SHA extensions: only the portable backend ran");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(Backend::detect(), Backend::Portable);
    }

    /// `cargo test --release -p germen-core sha256_throughput -- --ignored --nocapture`: the speed
    /// of the textbook version and of every backend on 64 MiB, taken in turns so that a noisy
    /// moment hits all of them; the best of four rounds is reported.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn sha256_throughput() {
        let data = vec![0x5Au8; 64 << 20];
        let mut best_textbook = f64::MAX;
        let mut best = vec![f64::MAX; backends().len()];
        for _ in 0..4 {
            let mut state = INITIAL;
            let started = std::time::Instant::now();
            for block in data.chunks_exact(64) {
                compress_reference(&mut state, block.try_into().unwrap());
            }
            best_textbook = best_textbook.min(started.elapsed().as_secs_f64());
            for (i, backend) in backends().into_iter().enumerate() {
                let mut other = INITIAL;
                let started = std::time::Instant::now();
                backend.compress(&mut other, &data);
                best[i] = best[i].min(started.elapsed().as_secs_f64());
                assert_eq!(state, other, "{backend:?}");
            }
        }
        let mb = 64.0;
        println!("textbook {:.0} MB/s", mb / best_textbook);
        for (backend, seconds) in backends().into_iter().zip(best) {
            println!(
                "{backend:?} {:.0} MB/s ({:.2}x the textbook)",
                mb / seconds,
                best_textbook / seconds
            );
        }
    }

    #[test]
    fn matches_the_nist_test_vectors() {
        for backend in backends() {
            let sum = |data: &[u8]| hex_with(backend, data);
            assert_eq!(
                sum(b""),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "{backend:?}"
            );
            assert_eq!(
                sum(b"abc"),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "{backend:?}"
            );
            assert_eq!(
                sum(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
                "{backend:?}"
            );
            assert_eq!(
                sum(
                    b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
                ),
                "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
                "{backend:?}"
            );
            let mut million = Sha256::with_backend(backend);
            for _ in 0..10_000 {
                million.update(&[b'a'; 100]);
            }
            assert_eq!(
                hex(&million.finish()),
                "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
                "{backend:?}"
            );
        }
    }

    /// Where the padding changes shape (55, 56 and 63 to 65 bytes, and the same again one block
    /// later) the digest is compared with Python's `hashlib` for
    /// `bytes(i % 251 for i in range(n))`; every length from 0 to 300 is also fed in whole, split in
    /// two at several places and one byte at a time, and all three must agree.
    #[test]
    fn digests_agree_with_hashlib_and_do_not_depend_on_how_the_input_is_split() {
        let data: Vec<u8> = (0..300).map(|i| (i % 251) as u8).collect();
        for (len, expected) in [
            (
                1,
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ),
            (
                55,
                "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
            ),
            (
                56,
                "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
            ),
            (
                63,
                "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488",
            ),
            (
                64,
                "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
            ),
            (
                65,
                "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781",
            ),
            (
                119,
                "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6",
            ),
            (
                120,
                "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c",
            ),
            (
                128,
                "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5",
            ),
            (
                300,
                "43f9b5d59eb108817176c6f65c2c6203a22f2ae8bc28b7a1dde45947678c5042",
            ),
        ] {
            for backend in backends() {
                assert_eq!(
                    hex_with(backend, &data[..len]),
                    expected,
                    "length {len}, {backend:?}"
                );
            }
        }
        for backend in backends() {
            for len in 0..=300 {
                let whole = hex_with(backend, &data[..len]);
                for split in [0, 1, len / 3, len / 2, len.saturating_sub(1), len] {
                    let split = split.min(len);
                    let mut h = Sha256::with_backend(backend);
                    h.update(&data[..split]);
                    h.update(&data[split..len]);
                    assert_eq!(
                        hex(&h.finish()),
                        whole,
                        "length {len}, split at {split}, {backend:?}"
                    );
                }
                let mut byte_by_byte = Sha256::with_backend(backend);
                for b in &data[..len] {
                    byte_by_byte.update(std::slice::from_ref(b));
                }
                assert_eq!(
                    hex(&byte_by_byte.finish()),
                    whole,
                    "length {len}, {backend:?}"
                );
            }
        }
    }

    #[test]
    fn a_file_is_summed_as_its_bytes_are() {
        let path = std::env::temp_dir().join(format!("germen-sum-{}", std::process::id()));
        let data: Vec<u8> = (0..(3 << 20) + 17).map(|i| (i * 7 % 256) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let (bytes, sum) = file_sum(&path).unwrap();
        assert_eq!(bytes, data.len() as u64);
        // From Python's hashlib, so that the one-megabyte read buffer is compared with something
        // other than this very implementation.
        assert_eq!(
            sum,
            "b6ce020bba55a0c6a8ab6f9fb9ff1f8fa46aaf7703a9e903cb19eb83e9ebebd8"
        );
        let _ = std::fs::remove_file(&path);
        assert!(file_sum(&path).is_err());
    }
}
