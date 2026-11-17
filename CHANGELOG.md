# Changelog

All notable changes to GERMEN are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-11-17

An adversarial audit of 0.2 and the optimizations it justified. Every finding, how it was
reproduced on 0.2.0, and what was tried and rejected by measurement is in
[docs/AUDIT.md](docs/AUDIT.md). Training is 23–44 % faster on two cores (median of five
interleaved runs against the ordinary step, batch 16 × 256): `nano` 6,807 → 8,366 tokens/s,
`micro` 2,769 → 3,981, `mini` 1,090 → 1,464, `small` 475 → 647. (The virtual machine was 9–14 %
slower in that session than when the 0.2 table was measured — the 0.2.0 binary itself measured
that much lower — so compare the ratios, not the 0.2 figures.)

### Added
- **Data-parallel replicas** (`--replicas N` for `train` and `bench`; 0, the default, is automatic:
  one per thread, never more than the windows of a batch, fewer if the extra gradient copies would
  not fit in memory; 1 is the ordinary step). The windows of a micro-batch are split among copies
  of the model, each with an autodiff graph of its own (`Module::fork`), run forward and backward on
  different threads; the gradients are added in a fixed pairwise order, and the initial weights are
  drawn in a fixed order, so a run is reproducible for a given number of threads and replicas. The
  attempt of 0.2, which shared one graph between threads, deadlocked and was rejected; a test now
  runs pools of one to five threads with two to eight replicas under a watchdog.
- **Checkpoints are written as a set**: every file is staged under a temporary name, flushed to
  disk, then renamed into place with `meta.json` last, so a save that fails while writing (disk
  full, I/O error, a kill) leaves the previous checkpoint exactly as it was. Non-finite weights are
  never written and never loaded. A `.germen.lock` file keeps a second run out of a directory that
  one is using.
- Available-memory probes for **Windows** (`GlobalMemoryStatusEx`) and **macOS** (`vm_stat`), so the
  "will it fit" check works there too (it only worked on Linux before).
- `docs/AUDIT.md`; `scripts/audit_repro.py` (black-box reproductions of the findings, runnable
  against any `germen` binary) and `examples/audit_lib.rs` (the library-level ones).
- Seven end-to-end tests of the real binary (`crates/germen-cli/tests/cli.rs`): 117 tests in all.
- CI: a job that builds with the minimum supported Rust (1.95), concurrency control, smoke runs
  with `--replicas 1` and `--replicas 3`, and a weekly RustSec advisory scan (`audit.yml`).
- API: `Batch::split`, `GptConfig::init_seeded`, `Gpt::init_parameters`, `checkpoint::save_all`,
  `checkpoint::ensure_finite`, `GptConfig::estimate_train_bytes_replicated`, `train::plan_replicas`,
  `sys::memory_limit_bytes`.

### Changed
- **The same `--seed` gives a different run than in 0.2**: the random generator is now seeded
  through SplitMix64 (finding A-14 in [docs/AUDIT.md](docs/AUDIT.md)) and the initial weights are
  drawn in a fixed order. Checkpoints written by 0.2 and 0.1 load and resume unchanged.
- `--resume` with a `--steps` smaller than the step already reached is an error that explains that
  `--steps` is the total (0.2.0 exited 0 and rewrote `meta.json`); exactly as many is a success that
  says there is nothing to do, and nothing is rewritten, so a "resume until done" loop ends.
- `--threads` above 1,024 is refused up front with a clear message (0.2.0 left it to the operating
  system, which refused too).
- Ctrl-C is honoured between micro-batches instead of between steps; the partial sum is dropped.
- The configuration of a model, the training options and the command-line numbers are validated
  with ceilings and finiteness checks (`GptConfig::validate`, `TrainOptions::validate`).
- Library API (the 0.x series makes no promise, but for the record): `TrainOptions`, `Event::Start`,
  `BenchParams` and `BenchResult` have a new `replicas` field and `bench_model` a new parameter.
- `#![deny(unsafe_code)]` in both crates; the single `unsafe` call (the Windows memory probe) is
  allowed in its own module.

### Fixed
Each item has an ID in [docs/AUDIT.md](docs/AUDIT.md), with the evidence and the test.
- A `tokenizer.json` of a few hundred bytes could make the loader allocate until the process was
  aborted (A-01); a `config.json` with absurd dimensions aborted on allocation (A-02); a
  `weights.bpk` with NaN weights was loaded without a word (A-03).
- A save that failed half-way left new weights next to an old optimizer state (A-04); a diverged
  run wrote NaN weights over the last good checkpoint (A-05); two `--resume` runs could share one
  output directory (A-06).
- `--grad-clip nan` silently trained without clipping (A-07); `--resume` with too few steps
  exited 0 and rewrote `meta.json` (A-08); `meta.json` stored the number of tokens, not bytes, as
  `corpus_bytes` with BPE (A-09).
- `--tokenizer bpe --val-fraction 2` panicked (A-10); `--val-fraction 0` reported "validation" loss
  measured on the training text without saying so (A-11); `--demo-mb 1e9` aborted (A-12);
  `generate --max-new 5000000000` aborted before the first token (A-13); the seeds 2k and 2k+1
  gave the same random stream (A-14); Ctrl-C with `--accum 60` took 29.8 s to act (A-16).
- On Windows and macOS the "will it fit in memory" check had no memory figure to compare with
  (A-17), and on Windows a save could fail when an antivirus scanner held the file just written
  (A-18). Both were found by reading the code; this machine cannot run them.
- Burn keeps one random generator for the whole process, and two threads that seeded and
  initialised a model at the same moment took shares of each other's stream, so the weights a seed
  promises depended on timing (A-24). Seeding and drawing are now one locked step
  (`GptConfig::init_seeded`). The 0.3 test suite itself was affected: a reproducibility test failed
  in 2 of 13 parallel runs, which a verification in a clean directory revealed (40 runs in a row
  pass now).

### Measured and rejected
Allocator tuning and `mimalloc` (slower), lowering the thresholds at which Flex splits element-wise
work across threads, a bigger batch, and a parallel or fused AdamW update (the update is about
1.8 % of a step) gave nothing; removing RMSNorm, rotary embeddings, the activation, the softmax or
the mask altogether saves less than 2–4 % of a step each. Details in [docs/AUDIT.md](docs/AUDIT.md).

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
