# GERMEN

**Train a small language model from scratch, in pure Rust, on an ordinary CPU.**

No Python, no CUDA, no C or C++ toolchain, no account, no network access, no telemetry.
One binary, one text file, one command. Built on [Burn](https://burn.dev) 0.22 with its pure-Rust
`flex` CPU backend.

> Other languages: [Română](README.ro.md) · more translations are planned.

## What it is — and what it is not

GERMEN trains a GPT-style network (RMSNorm, rotary positions, squared-ReLU feed-forward, no biases)
over **bytes**: the vocabulary is the 256 byte values, so any text — Romanian diacritics
included — works without a tokenizer to prepare.

It is a **workshop for learning and experimenting**, and a base for research: you can watch a
network go from random noise to spelling, punctuation and the style of your text within minutes.
It is **not** a chatbot factory. A model of 1–25 million parameters trained on a few hundred
megabytes learns the *form* of a text (words, grammar fragments, layout), not world knowledge or
conversation. Useful assistants need orders of magnitude more compute than a desktop CPU offers.

## Quick start

You need Rust 1.95 or newer ([rustup.rs](https://rustup.rs); on Windows it also asks for the
Visual Studio Build Tools). Then:

```bash
cargo build --release
./target/release/germen bench                      # how fast is THIS computer?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # text from the model you just trained
```

On Windows use `target\release\germen.exe`. Pre-built binaries are attached to the releases.

Train on your own text — any file, any language:

```bash
germen train --data my-corpus.txt --preset mini --steps 5000 --out my-model
germen generate --model my-model --prompt "Era odată " --max-new 400
```

Press **Ctrl-C** during training: it finishes the current step, saves, and exits. Continue later
with the same command plus `--resume`.

## Commands

| command | what it does |
|---|---|
| `germen info` | detected processors and SIMD level, model sizes and memory for the weights |
| `germen bench` | times real training steps for each model size and extrapolates the time for 100 M and 1 B tokens; `--json FILE` saves the numbers |
| `germen demo-corpus` | writes a synthetic Romanian corpus generated from a small grammar (`--size-mb`) |
| `germen train` | trains from scratch (`--data FILE` or `--demo`), or continues with `--resume` |
| `germen generate` | samples text from a trained model (`--temperature`, `--top-k`, `--prompt`) |

Add `--threads N` to any command to limit the number of worker threads. `germen train --help`
lists every training option (learning rate, warm-up, weight decay, batch, context, validation
and checkpoint intervals, …). The messages of the command line are currently in Romanian.

## Model sizes

| preset | parameters | layers × heads × width | weights (f32) |
|---|---:|---|---:|
| `nano` | 0.85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2.76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10.82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25.44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85.35 M | 12 × 12 × 768 | 326 MiB |

Training also keeps AdamW's two moment estimates (≈ 4× the weights) plus the activations, which
grow with `--batch` and `--seq`.

## How long does it take?

The cost of training is about **6 × parameters × tokens** floating-point operations (a byte is a
token, so 1 MB of text = 1 M tokens). Divide by what your CPU really delivers and you have the
time. `germen bench` measures exactly that on your machine.

Reference measurement — a small Linux virtual machine with **2 cores** (AVX-512), batch 16 × 256:

| preset | tokens/s | effective GFLOPS | 100 M tokens | 1 B tokens |
|---|---:|---:|---:|---:|
| `nano` | 6,040 | 39 | 4.6 h | 1.9 d |
| `micro` | 2,362 | 47 | 11.8 h | 4.9 d |
| `mini` | 1,051 | 75 | 1.0 d | 11 d |
| `small` | 522 | 86 | 2.1 d | 22 d |

Going from 1 to 2 threads made `mini` 1.44× faster here (730 → 1,051 tokens/s), so do not expect
20 cores to give 20×: some operations of the CPU backend use a single thread (the `exp`/`erf`
family, which is why the feed-forward block uses squared ReLU instead of GELU). A desktop with many
more cores should still be several times faster;
`cargo run --release -p germen-core --example profile_ops -- N` shows which operations scale.
**Please run `germen bench --json bench.json` on your machine** — it
is the only way to know. A common rule of thumb for compute-optimal training is about 20 tokens
per parameter; small models are often trained on much more.

## How it is built

```
crates/germen-core   the library: config, model, data, train, generate, checkpoint, bench
crates/germen-cli    the `germen` command line (clap)
```

* **Backend** — Burn's `flex`: pure Rust, matrix products through the `gemm` crate, SIMD kernels
  (AVX2/AVX-512/SSE, NEON) picked **at run time**, multi-threaded with rayon. A generic x86-64
  build therefore still uses AVX2 on a CPU that has it.
* **Model** — pre-norm blocks, fused QKV projection, causal attention, rotary embeddings,
  untied output head, GPT-2-style initialisation (output projections scaled by `1/√(2·layers)`).
* **Training** — AdamW (β₂ = 0.95), linear warm-up then cosine decay, validation on fixed windows
  (reported in nats and bits per byte), atomic checkpoints, progress through `Event` callbacks so
  a graphical front end can reuse the same loop.
* **Checkpoint** — a folder with `config.json`, `weights.bpk` (Burn's burnpack, f32) and
  `meta.json`.
* **Tests** — 39 unit tests, among them: attention is causal, the initial loss is ln 256, a
  checkpoint reloads to identical logits, training lowers the loss, `--resume` continues.

## Limitations (version 0.1)

* Byte-level only: sequences are 3–4× longer than with a subword vocabulary. A BPE tokenizer is
  the next step.
* `--resume` restores weights and step counter but not AdamW's moments (a small loss bump).
* No key/value cache: generation re-reads the context window for every new byte.
* No gradient accumulation, no mixed precision, no GPU backend; gradient clipping is per tensor.
* The `demo` corpus is synthetic and tiny: it proves the pipeline learns, nothing more.

See [CHANGELOG.md](CHANGELOG.md) and [docs/PLAN.md](docs/PLAN.md). Contributions are welcome —
the open issues are a good start ([CONTRIBUTING.md](CONTRIBUTING.md)).

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
