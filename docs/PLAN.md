# GERMEN — plan and decisions

Status: version 0.2.0 (November 2026). The name is provisional; check it against trademark
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
| checkpoint | burnpack + JSON | native to Burn, atomic writes, no extra dependency |
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
* **Rejected — data parallelism with several threads that each drive Burn** (each thread runs a
  micro-batch forward and backward, gradients are summed): with four workers on a two-thread pool
  the process deadlocked (every thread waiting in a futex, 0 % CPU), with two it gained 3.5 %. The
  CPU backend is not meant to be driven by several threads at once while it also parallelises
  inside each operation. Everything now runs on one driver thread.
* **Not a lever — bigger batches:** a larger `--batch` did not raise tokens per second.
* **Open — make the serial parts parallel.** In Flex 0.22.1 the element-wise SIMD kernels only
  split work across threads for tensors of about 4 Mi elements or more (other operations use a
  256 K-element threshold), so RMSNorm and rotary embeddings on the `mini` shapes stay on one
  thread; `erf`/`tanh`/`exp` are single-threaded. Candidates: a parallel AdamW update over
  parameter tensors, fused RMSNorm and rotary kernels, fewer tensor copies.

`cargo run --release -p germen-core --example profile_ops -- N` times single operations with N
threads; `profile_step`, `profile_mlp` and `profile_attn` do the same for a whole step and for the
feed-forward and attention blocks.

## 5. Roadmap

* **0.2 (done)** — BPE tokenizer; AdamW state on `--resume`; global-norm clipping; key/value cache;
  gradient accumulation; `bench --scaling`; READMEs in eight languages.
* **0.3** — parallel non-matmul parts of the step (driven by the first many-core measurement);
  memory-mapped corpora larger than RAM.
* **0.4** — desktop GUI (egui) with live loss curve and samples; safetensors export/import.
* **0.5** — opt-in GPU backend (`wgpu`) behind a Cargo feature; community benchmark table.
* **1.0** — the command-line messages in the project's usual languages; releases for Windows,
  Linux and macOS built by CI; an extensive adversarial audit before the announcement.

## 6. Open questions

1. **Name** — "GERMEN" is provisional; run a trademark search.
2. **Interoperability** — colibri's README lists thirteen engines (GLM, Kimi, DeepSeek, Qwen,
   OLMoE, …), loads language models from GGUF files as they are, and says checkpoints with the same
   architecture as a supported one run unchanged; for a new architecture it points to a new C
   engine file over shared headers, with no step-by-step guide. A GERMEN model (dense,
   squared-ReLU feed-forward, byte or BPE vocabulary) is not among the listed architectures, so it
   cannot be loaded there unchanged. Nothing has been tried; a GGUF or safetensors export would be
   the first step and is not planned before 0.4. Source: the README of
   <https://github.com/JustVugg/colibri>, read on 2026-11-17.
3. **Training data** — any pre-trained model that is shipped needs a corpus whose licence allows
   it (public-domain literature, openly licensed Wikipedia text, own writing). Not decided.
4. **Measured speed on the author's Windows PC (20 cores)** — not yet known.
   `germen bench --scaling --json bench.json` produces the curve to paste into the README and to
   decide what to parallelise next.

## 7. What has been verified so far

* 70 unit tests pass (release build, Linux).
* Short real runs on the synthetic corpus, byte and BPE vocabularies: the loss falls well below
  the untrained value in a few dozen steps and the generated text follows the grammar of the
  corpus; interrupting and resuming works, including from a 0.1 checkpoint.
* Cached generation produces the same greedy text as the full forward pass, and the same text as
  before the cache for the first window.
* Benchmarks on a 2-core AVX-512 virtual machine (table in the README).
* **Not verified:** native Windows and macOS runs (the CI workflow builds and smoke-tests both but
  has not run on GitHub yet); anything on more than two cores; anything beyond the 85 M `base`
  preset; any real-world corpus; loading a GERMEN model in another engine.
