# GERMEN — plan and decisions

Status: version 0.4.0 (November 2026). The name is provisional; check it against trademark
databases (EUIPO / TMview) before any public launch.

## 1. Goal

Make "train a language model from scratch" something an ordinary person can do on an ordinary
computer: one binary, one text file, no Python, no GPU, no accounts, nothing leaving the machine.
The models are small on purpose. The point is to learn how training behaves, to experiment with
data, and to have a transparent, auditable base to build on.

## 2. The cost model — why sizes are what they are

* Training cost ≈ **6 × N × D** floating-point operations for N parameters and D training tokens
  (forward 2·N·D plus backward 4·N·D), plus a small attention term that grows with context length.
  `GptConfig::train_flops_per_token` in `config.rs` implements the exact estimate.
* With the byte vocabulary one byte = one token, so D equals the corpus size in bytes times the
  number of passes. With BPE a token covers several bytes (3–4 for ordinary prose), so a given
  context window and a given D cover that much more text; the output head grows by
  2·(V − 256)·width parameters.
* A model that has learned nothing has loss ln V nats per token (ln 256 = 5.545 nats = 8 bits per
  byte for bytes). Validation is printed in nats per token and in bits per byte, which stays
  comparable between byte and BPE runs.
* Rule of thumb for compute-optimal training: about 20 tokens per parameter. A 10.8 M-parameter
  model → about 216 M tokens. Small models are often trained on much more for better quality at
  a given size.
* `germen bench` measures real tokens per second; time = D / (tokens per second). Measure first,
  then pick a size. Reference numbers from a 2-core sandbox are in the README.

## 3. Decisions

| decision | choice | why |
|---|---|---|
| framework | Burn **0.22.1**, pinned (`=0.22.1`) | the new device-based API is stable within 0.22; a pin keeps builds reproducible |
| backend | `flex` | pure Rust, `gemm` matmul, run-time SIMD selection, rayon; builds on Windows, macOS, Linux, even WASM, with no native dependency |
| not used | `cpu` (CubeCL/LLVM) | needs an LLVM toolchain |
| not used | `ndarray` | legacy path in Burn 0.22 |
| not used | LibTorch | deprecated in 0.22, large native dependency |
| not used | Burn's `Learner` | a hand-written loop is ~100 lines, gives full control over events, checkpoints and Ctrl-C |
| tokens | bytes (default) or a trained byte-level BPE (`u16` ids, up to 65,536) | bytes need nothing to train or ship and round-trip any input; BPE gives 3–4× more text per window. The merges are learned from the training split only, so validation stays honest; merges never cross a pre-token, which keeps training and encoding cheap and parallel |
| feed-forward activation | squared ReLU, `relu(x)²` | on `flex` 0.22.1 `erf`/`tanh`/`exp` do not use more than one thread and cost several ns per element: one GELU call took ~13× longer than `relu(x)²` on the `mini` shapes (`profile_ops`), and that time does not shrink with more cores. In `profile_mlp` the squared-ReLU feed-forward block (forward + backward on the `mini` shapes) went from 381 to 220 ms on one → two threads (1.73×), a SwiGLU block with the same parameter count from 400–443 to 245–278 ms (≈ 1.6×). On the demo corpus both reached a similar validation loss after 300 steps (0.276 for squared ReLU, 0.285 for GELU; single runs) |
| positions | rotary (own implementation, rotate-half layout, constant tables) | no learned position table, works for any window up to the trained context |
| attention | training: block-wise causal (blocks of 2 × head dimension, scale folded into the queries, additive mask); evaluation: Burn's fused kernel; generation: key/value cache | under autodiff Burn's `attention` falls back to a long chain of ops that also computes the masked upper triangle; the block-wise version skips it (whole steps 8 % faster for `mini`, 28 % for `nano` on two cores; same values and gradients, tested). Of block sizes 32/64/128/256, twice the head dimension was best for head dimensions 32 and 64 |
| gradient clipping | global norm over all parameters, default 1 | the standard recipe; Burn's built-in clipping works per tensor |
| optimizer state | saved next to the weights (`optimizer.bpk`, keyed by parameter id) | resuming without it causes a loss bump; a test shows the step after a reload equals the uninterrupted one |
| threads | the whole command runs inside its own rayon pool (`sys::run_in_pool`) | +7 % (`mini`) to +25 % (`nano`) on two cores compared with driving the global pool from an outside thread, because every parallel loop otherwise needs a hand-off between threads |
| data parallelism | the windows of a micro-batch are split among *replicas* of the model, each with its own autodiff graph (`Module::fork`), run forward and backward on different threads; gradients are added in a fixed pairwise order (`train::micro_batch_gradients`) | + 23 % (`nano`) to + 44 % (`micro`) on two cores (`mini` + 34 %, `small` + 36 %; ratios of the medians of five interleaved runs, ± 10 %), with a peak memory 1–3 % lower: the small element-wise operations that do not scale inside one graph now run side by side. A fixed slot scheme, a fixed reduction tree and a fixed order of drawing the initial weights make a run reproducible for a given number of threads and replicas; a test compares the gradients with those of one big graph |
| checkpoint | burnpack + JSON; every file staged, fsynced and renamed together, `meta.json` last; a lock file per directory; non-finite weights are never written or loaded | a crash or a second run can no longer leave a mixed set of files, and a diverged model cannot replace a good checkpoint (see [AUDIT.md](AUDIT.md)) |
| UI language | Romanian messages, English code and docs, READMEs in eight languages | first user is Romanian; moving the messages to a catalogue is a starter issue |

## 4. Performance findings (2-core AVX-512 VM, `mini`, batch 16 × 256)

Where the time of one step goes at one thread: forward ≈ 33 %, backward ≈ 66 %, AdamW update
≈ 1.3 %. With two threads the whole step is about 1.5× faster (1.51× in `bench --scaling`), made up of:

| part | speed-up on 2 threads |
|---|---:|
| matrix products in the feed-forward block | 1.7–1.8× |
| attention (before the block-wise version) | 1.24× |
| RMSNorm, rotary embeddings, AdamW update | ≈ 1.0× |

By Amdahl's law the parts that do not scale decide what many cores can give, which is why
`germen bench --scaling` exists. What was done about it, and what was not:

* **Done — block-wise training attention:** whole step + 8 % (`mini`), + 22 % (`micro`), + 28 % (`nano`)
  on two threads (three interleaved runs each, against Burn's attention); + 5 % for `mini` at one
  thread. Blocks of twice the head dimension beat a fixed 128 by 3–6 % for `nano` and `micro`.
* **Done — run inside an own pool:** + 7 % (`mini`), + 17 % (`micro`), + 25 % (`nano`); `nano`
  went from about 5.5–6.1 k to 7.1–7.9 k tokens/s over three interleaved runs, which also shows
  how much of a small step was thread hand-offs.
* **Done in 0.3 — data parallelism, with a graph per replica.** The attempt of 0.2 (several
  threads each running a micro-batch forward and backward on the one model, gradients summed)
  deadlocked with four workers on a two-thread pool (every thread waiting in a futex, 0 % CPU) and
  gained 3.5 % with two: all threads shared a single autodiff graph and its lock. `Module::fork`
  gives every replica a graph (and lock) of its own, which removed both problems. Final reference,
  five interleaved rounds of the real `bench` against the ordinary step (`--replicas 1`), two
  threads, batch 16 × 256, ratios of the medians: `nano` + 23 %, `micro` + 44 %, `mini` + 34 %,
  `small` + 36 % (per-round ratios of `nano` ranged from + 2 % to + 40 %, so + 16 % by the median
  of the rounds). An earlier series of the same kind gave + 25 / + 45 / + 23 / + 35 %; that `mini`
  differs by eleven points between the two series is what ± 10 % looks like on this machine. Going
  from one to two threads, `mini` became 2.04× faster (733 → 1,494 tokens/s) instead of 1.41×.
  Peak memory with two replicas was 1–3 % lower than with the ordinary step in all four presets.
  Reproducibility needed three things, each found by a test: a fixed reduction order (rayon's
  `reduce` has no fixed tree), a fixed assignment of replicas to slots, and drawing the initial
  weights in module order before any thread uses them (Burn initialises parameters lazily, on
  first use, and every draw advances one device-wide generator). A test runs pools of one to five
  threads with two to eight replicas under a watchdog. The full table and the control runs are in
  [AUDIT.md](AUDIT.md).
* **Not a lever — bigger batches:** a larger `--batch` did not raise tokens per second.
* **Rejected by measurement in the 0.3 audit** (same virtual machine, interleaved runs, nothing
  kept that did not win): allocator tuning of the C library and `mimalloc` (the latter was slower);
  lowering Flex's thresholds for splitting element-wise work across threads; a bigger batch; a
  parallel or fused AdamW update (the update is about 1.8 % of a step, so that is the most it could
  give). Removing RMSNorm, rotary embeddings, the activation, the softmax or the mask altogether —
  an upper bound for anything that speeds them up — saved less than 2–4 % of a step each and no more
  than about 10 % together, so there is no cheap model-level win left in them. In Flex 0.22.1 the
  element-wise SIMD kernels only split work across threads for tensors of about 4 Mi elements or
  more (other operations use a 256 K-element threshold) and `erf`/`tanh`/`exp` are
  single-threaded; replicas hide this when the batch has at least as many windows as there are
  threads.
* **Open — how many replicas?** The default is one replica per thread (at most one per window).
  Forcing more replicas than threads, so that each replica is a single window, was measured at two
  threads against that default: `nano` + 40 %, `micro` + 6 %, `mini` + 8 %, but `small` − 8 % (batch
  16; four rounds, three for `small`), and for `nano` at batch 4 it was − 9 %; at one thread
  `nano` and `mini` gained + 9 %. Smaller graphs stay in the caches, while every replica adds one
  more gradient set to the sum, a cost that grows with the parameters; a parallel or in-place
  reduction might remove it and has not been tried. Until then the default stays conservative and
  `--replicas N` is the knob (table in [AUDIT.md](AUDIT.md)).
* **Open — a many-core measurement.** Everything above comes from two cores. How far replicas carry
  on 20 cores is unknown (`germen bench --scaling --json bench.json` will tell); at most `--batch`
  replicas exist, so use a batch at least as large as the number of threads.

`cargo run --release -p germen-core --example profile_ops -- N` times single operations with N
threads; `profile_step`, `profile_mlp` and `profile_attn` do the same for a whole step and for the
feed-forward and attention blocks.

## 5. Roadmap

* **0.2 (done)** — BPE tokenizer; AdamW state on `--resume`; global-norm clipping; key/value cache;
  gradient accumulation; `bench --scaling`; READMEs in eight languages.
* **0.3 (done)** — adversarial audit ([AUDIT.md](AUDIT.md)): hostile input files, atomic
  checkpoints, option validation, Windows/macOS memory probes; data-parallel replicas.
* **0.4 (done)** — second adversarial audit ([AUDIT.md](AUDIT.md), Part II): checkpoint manifest
  with SHA-256, `germen verify`, a journal for saves, strict `--resume`, no silent overwrite,
  bounded pre-token scan, hardware SHA-256.
* **0.5** — memory-mapped corpora larger than RAM; desktop GUI (egui) with live loss curve and
  samples; safetensors export/import; whatever the first many-core measurement calls for.
* **0.6** — opt-in GPU backend (`wgpu`) behind a Cargo feature; community benchmark table.
* **1.0** — the command-line messages in the project's usual languages; releases for Windows,
  Linux and macOS built by CI; a second adversarial audit (by someone other than the author of the
  code) before the announcement.

## 6. Open questions

1. **Name** — "GERMEN" is provisional; run a trademark search.
2. **Interoperability** — colibri's README lists thirteen engines (GLM, Kimi, DeepSeek, Qwen,
   OLMoE, …), loads language models from GGUF files as they are, and says checkpoints with the same
   architecture as a supported one run unchanged; for a new architecture it points to a new C
   engine file over shared headers, with no step-by-step guide. A GERMEN model (dense,
   squared-ReLU feed-forward, byte or BPE vocabulary) is not among the listed architectures, so it
   cannot be loaded there unchanged. Nothing has been tried; a GGUF or safetensors export would be
   the first step and is not planned before 0.5. Source: the README of
   <https://github.com/JustVugg/colibri>, read on 2026-11-17.
3. **Training data** — any pre-trained model that is shipped needs a corpus whose licence allows
   it (public-domain literature, openly licensed Wikipedia text, own writing). Not decided.
4. **Measured speed on the author's Windows PC (20 cores)** — not yet known.
   `germen bench --scaling --json bench.json` produces the curve to paste into the README and to
   decide what to parallelise next.

## 7. What has been verified so far

* 156 tests pass (release build, Linux): unit tests of the library and of the command line, and
  end-to-end runs of the real binary. The core suite ran 40 times in a row in parallel mode
  without a failure; before the fix of A-24 it failed in 2 of 13 runs.
* The adversarial audit of 0.2 ([AUDIT.md](AUDIT.md), Part I): 15 findings reproduced against a
  0.2.0 binary and gone in 0.3.0, two suspicions that did not reproduce, seven found by reading the
  code, and one (A-24) found by verifying 0.3.0 itself in a clean directory.
* The audit of 0.3.0 (Part II): nine findings reproduced against a 0.3.0 binary and gone in 0.4.0,
  one found by reading the code; a hostile-input harness that classifies every run (damaged
  checkpoints: 206 of 464 accepted by 0.3.0, 3 of 468 by 0.4.0, all three legitimate for
  `generate`); mutation proofs for every fix, which fail against the old defect.
* Short real runs on the synthetic corpus, byte and BPE vocabularies: the loss falls well below
  the untrained value in a few dozen steps and the generated text follows the grammar of the
  corpus; interrupting and resuming works, including from a 0.1 checkpoint.
* Cached generation produces the same greedy text as the full forward pass, and the same text as
  before the cache for the first window.
* Benchmarks on a 2-core AVX-512 virtual machine (table in the README).
* CI on the pushed commits `9195e4d` and `9fa10b2` (every job green): `rustfmt`, `clippy -D
  warnings`, the 117 tests and an end-to-end smoke run on Ubuntu 26.04, Windows and macOS, and a
  build with Rust 1.95. The weekly dependency audit is green since `9fa10b2` (red on the first
  commit; see A-20 in [AUDIT.md](AUDIT.md)).
* **Not verified:** Windows and macOS beyond the CI smoke run (no real-size training or speed
  figures there, and the rename retry has never met a real antivirus scanner); anything on more
  than two cores; anything beyond the 85 M `base` preset; any real-world corpus; loading a GERMEN
  model in another engine.
