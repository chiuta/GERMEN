# GERMEN — plan and decisions

Status: version 0.1.0, a working prototype (November 2026). The name is provisional; check it
against trademark databases (EUIPO / TMview) before any public launch.

## 1. Goal

Make "train a language model from scratch" something an ordinary person can do on an ordinary
computer: one binary, one text file, no Python, no GPU, no accounts, nothing leaving the machine.
The models are small on purpose. The point is to learn how training behaves, to experiment with
data, and to have a transparent, auditable base to build on.

## 2. The cost model — why sizes are what they are

* Training cost ≈ **6 × N × D** floating-point operations for N parameters and D training tokens
  (forward 2·N·D plus backward 4·N·D), plus a small attention term that grows with context length.
  `GptConfig::train_flops_per_token` in `config.rs` implements the exact estimate.
* The byte-level vocabulary makes one byte = one token, so D equals the corpus size in bytes times
  the number of passes.
* A model that has learned nothing has loss ln 256 = 5.545 nats per byte (8 bits per byte). The
  validation loss is printed in nats and in bits per byte, so progress is easy to read.
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
| tokens | bytes (256) | nothing to train or ship; round-trips any UTF-8; BPE is the planned upgrade |
| feed-forward activation | squared ReLU, `relu(x)²` | on `flex` 0.22.1 `erf`/`tanh`/`exp` do not use more than one thread and cost several ns per element: one GELU call took ~13× longer than `relu(x)²` on the `mini` shapes (`profile_ops`), and that time does not shrink with more cores. End-to-end speed-ups on the noisy 2-core VM were within a few to ~15 %, so no precise figure is claimed. On the demo corpus both reached a similar validation loss after 300 steps (0.276 for squared ReLU, 0.285 for GELU; single runs) |
| positions | rotary (own implementation) | no learned position table, works for any window up to the trained context |
| attention | Burn's fused `attention`, causal | falls back to differentiable decomposed ops under autodiff |
| checkpoint | burnpack + JSON | native to Burn, atomic writes, no extra dependency |
| UI language | Romanian messages, English code and docs | first user is Romanian; i18n is a starter issue |

### Performance notes (measured on a 2-core AVX-512 VM, `mini`, batch 16 × 256)

* Going from 1 to 2 threads: 730 → 1,051 tokens/s (1.44×). Matrix products scale well
  (~1.8×); the remaining time is in element-wise and transcendental operations, attention's
  masked softmax, the optimizer and copies, which scale poorly or not at all.
* With many more cores the non-scaling part dominates (Amdahl's law), so the first task on the
  target machine is `profile_ops` with its thread count, then fixing the largest item (candidates:
  a fused attention kernel, a parallel element-wise path, fewer tensor copies in RoPE).

## 4. Roadmap

* **0.2** — trainable BPE tokenizer with byte fallback; AdamW state saved and restored on
  `--resume`; global-norm gradient clipping.
* **0.3** — key/value cache for generation; gradient accumulation; memory-mapped corpora larger
  than RAM.
* **0.4** — desktop GUI (egui) with live loss curve and samples; safetensors export/import.
* **0.5** — opt-in GPU backend (`wgpu`) behind a Cargo feature; community benchmark table.
* **1.0** — translations of the README and CLI into the project's usual languages; releases for
  Windows, Linux and macOS built by CI; an extensive adversarial audit before the announcement.

## 5. Open questions

1. **Name** — "GERMEN" is provisional; run a trademark search.
2. **Interoperability** — whether a model trained here can be loaded by local inference engines
   (for example colibri) depends on exporting a compatible weight layout *and* a tokenizer file.
   Not verified yet; the safetensors export (0.4) is the first step.
3. **Training data** — any pre-trained model that is shipped needs a corpus whose licence allows
   it (public-domain literature, openly licensed Wikipedia text, own writing). Not decided.
4. **Measured speed on the author's Windows PC** — not yet known. `germen bench --json` exists so
   that the first measurement can be pasted into the README and the benchmark table.

## 6. What has been verified so far

* 39 unit tests pass (release build, Linux).
* A short real run on the synthetic corpus: the loss falls from ≈ 5.5 to below 1 nat per byte in
  under 100 steps and the generated text follows the grammar of the corpus.
* A benchmark on a 2-core AVX-512 virtual machine (table in the README).
* **Not verified:** native Windows and macOS runs (the CI workflow builds and smoke-tests both);
  anything beyond the 85 M `base` preset; any real-world corpus.
