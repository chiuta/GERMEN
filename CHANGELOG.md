# Changelog

All notable changes to GERMEN are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-11-17 (prototype)

First working version, built to answer one question: *how fast can a small language model be
trained from scratch on an ordinary CPU, in pure Rust?*

### Added
- GPT-style decoder (RMSNorm, rotary positions, squared-ReLU MLP, no biases) on a byte-level vocabulary,
  written on Burn 0.22 with the pure-Rust `flex` CPU backend. No Python, CUDA, LLVM or C/C++
  toolchain required.
- Five size presets: `nano` (0.85 M parameters), `micro` (2.8 M), `mini` (10.8 M),
  `small` (25.4 M), `base` (85.4 M).
- Training loop: AdamW, linear warm-up with cosine decay, per-tensor gradient clipping,
  validation with fixed windows (reported in nats and bits per byte), periodic checkpoints,
  text samples, `--resume`, clean stop on Ctrl-C.
- Checkpoints: `config.json`, `weights.bpk` (Burn burnpack, f32), `meta.json`; written atomically.
- Text generation with temperature and top-k sampling; UTF-8-safe streaming (Romanian
  diacritics are never split).
- `germen bench`: times real training steps on the current machine and extrapolates the time
  needed for 100 M and 1 B tokens; `--json` saves the measurements.
- `germen demo-corpus` / `train --demo`: a synthetic Romanian corpus generated from a small
  grammar, for smoke tests and a first run that finishes in minutes.
- `germen info`: detected processors, SIMD level, model sizes and memory for weights.
- Memory estimate shown before training; a request that cannot fit (for example a typo such as
  `--seq 99999`) is refused with a clear message instead of failing inside the allocator.
- `examples/profile_ops.rs`: times single tensor operations with a chosen number of threads, to
  find which ones do not scale (run it on the target machine).
- 39 unit tests, including a causality test of the attention, a checkpoint round-trip,
  a loss-decreases test and a resume test.
- GitHub Actions workflow (Ubuntu 26.04, Windows, macOS) and a publishing script.

### Known limitations
- Byte-level tokenizer only (BPE is planned); sequences are longer than with subwords.
- Resuming restores the weights and the step counter but not AdamW's moment estimates.
- No key/value cache: generation re-reads the whole context window for every new byte.
- No gradient accumulation, no mixed precision, no GPU backend.
- Gradient clipping works per parameter tensor (Burn's built-in behaviour), not on the global norm.
- The command-line messages are in Romanian only.
- Not yet measured on Windows or macOS; see the README for how to run `germen bench` and share
  the numbers.
