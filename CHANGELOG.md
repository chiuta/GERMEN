# Changelog

All notable changes to GERMEN are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-11-17

Closes the limitations of 0.1 that mattered most (tokenizer, resume, clipping, generation speed,
batch size) and makes the training step faster. Measured on the same 2-core AVX-512 virtual
machine as in 0.1, batch 16 × 256: `nano` 6,040 → 8,261 tokens/s, `micro` 2,362 → 3,180,
`mini` 1,051 → 1,260, `small` 522 → 545.

### Added
- **BPE tokenizer** (`--tokenizer bpe`, `--vocab N` from 257 to 65,536, `--bpe-sample-mb`): a
  byte-level BPE with a GPT-2-like pre-tokenizer (a space sticks to the following word; merges
  never cross a word), deterministic training with incremental pair counts, parallel encoding.
  It is learned from the training part of the text only, stored in the checkpoint
  (`tokenizer.json`) and used by `generate`; every byte sequence round-trips exactly.
  `--resume` insists on the tokenizer the model was trained with.
- **AdamW state on `--resume`** (`optimizer.bpk`): training continues without the loss bump of a
  cold restart. Checkpoints from 0.1 still load; the optimizer then restarts from zero and the
  command says so.
- **Global-norm gradient clipping** (`--grad-clip`, default 1, 0 = off) instead of Burn's
  per-tensor clipping; the norm is shown in the training log.
- **Gradient accumulation** (`--accum`): the effective batch is `--batch × --accum`, the memory
  is that of `--batch`.
- **Key/value cache for generation** (`Gpt::forward_cached`, `KvCache`). 400 tokens from `mini`:
  15.3 s → 1.0 s; 300 bytes from `nano`: 1.23 s → 0.10 s. When the window is full, generation
  continues from its most recent half.
- **`germen bench --scaling [LIST]`**: the benchmark repeated on 1, 2, 4, … threads (or the list
  you give), with speed-up, efficiency and the thread count beyond which nothing is gained;
  `--json` includes it. `bench --vocab` measures a BPE-sized output head.
- README in eight languages: English, Romanian, French, Italian, Spanish, Portuguese, German and
  Hungarian.
- Development examples `profile_step`, `profile_mlp` and `profile_attn` (where the time of a
  training step goes, and which parts scale with the number of threads).
- 31 new unit tests (70 in total): global-norm clipping, accumulation equals a large batch,
  restored AdamW state reproduces the next step, block attention equals Burn's attention in values
  and gradients, cached generation equals the full forward pass, BPE properties, tokenizer files.

### Changed
- **Training attention** is computed block-wise (blocks of twice the head dimension) and never
  evaluates the part above the diagonal: whole steps are about 8 % (`mini`), 22 % (`micro`) and
  28 % (`nano`) faster on two cores, with the same values and gradients as Burn's attention
  (tested to about 1e-5).
- **The whole command runs as a worker of its own thread pool** instead of driving the global
  pool from outside: parallel loops start without a hand-off between threads. About + 7 % for
  `mini`, + 17 % for `micro` and + 25 % for `nano` on two cores.
- Validation is reported in nats per token and in bits per byte, so byte and BPE runs can be
  compared; the "loss of an untrained model" line is no longer printed when resuming.
- Library API: `tokenizer::Tokenizer` (byte or BPE) replaces `ByteTokenizer`; `generate` takes a
  tokenizer and returns token ids; `Event::Start` and `Event::Step` carry more fields; `Meta`
  records the number of accumulated micro-batches.

### Tried and rejected
- Spreading micro-batches over several OS threads that all call Burn: with four workers the
  process deadlocks (every thread waits in a futex, 0 % CPU), with two it gained 3.5 %. Details in
  [docs/PLAN.md](docs/PLAN.md).

### Known limitations
- The whole corpus is read into memory as 16-bit token ids.
- RMSNorm, rotary embeddings and the AdamW update do not use extra cores well yet.
- No mixed precision, no GPU backend.
- The command-line messages are in Romanian only.
- Not yet measured on Windows, macOS or a machine with many cores; the GitHub Actions workflow has
  not run yet.

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
