# GERMEN

**Train a small language model from scratch, in pure Rust, on an ordinary CPU.**

No Python, no CUDA, no C or C++ toolchain, no account, no network access, no telemetry.
One binary, one text file, one command. Built on [Burn](https://burn.dev) 0.22 with its pure-Rust
`flex` CPU backend.

> Languages: **English** · [Română](README.ro.md) · [Français](README.fr.md) · [Italiano](README.it.md) · [Español](README.es.md) · [Português](README.pt.md) · [Deutsch](README.de.md) · [Magyar](README.hu.md)

🌐 **Website:** <https://germen.chiuta.ro> · [live Studio demo](https://germen.chiuta.ro/demo.html)

## What it is — and what it is not

GERMEN trains a GPT-style network (RMSNorm, rotary positions, squared-ReLU feed-forward, no biases)
over **tokens**. By default a token is a **byte**: the vocabulary is the 256 byte values, so any
text — Romanian diacritics included — works with nothing to prepare. With `--tokenizer bpe` GERMEN
first learns a byte-level BPE vocabulary (up to 65,536 tokens) from your own text, so the same
context window holds several times more text (typically 3–4× for ordinary prose). The tokenizer is
stored with the model.

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
./target/release/germen bench --scaling            # ...and how does it grow with the number of threads?
./target/release/germen train --demo --preset micro --steps 300
./target/release/germen generate                   # text from the model you just trained
```

On Windows use `target\release\germen.exe`. Pre-built binaries are attached to the releases.

Train on your own text — any file, any language:

```bash
germen train --data my-corpus.txt --preset mini --tokenizer bpe --vocab 4096 --steps 5000 --out my-model
germen generate --model my-model --prompt "Era odată " --max-new 400
```

Press **Ctrl-C** during training: it finishes the current step, saves, and exits. Continue later
with the same command plus `--resume`: weights, step counter and AdamW state are restored, so
training goes on without the loss bump of a cold restart (the random text windows are drawn anew).

Running the same `train` command again *without* `--resume` is refused, so that it cannot replace a
trained model by mistake; add `--overwrite` to start over on purpose.

## The window: GERMEN Studio

Prefer a window to a command line?

```bash
germen ui
```

opens **GERMEN Studio**: choose a text, press **Start training** and watch a model learn — the loss
curve, the speed, and samples of what the network writes as it improves. The same window writes
text from your models, checks and repairs their files and measures the speed of your computer, with
the same code as the commands above. It comes in eight languages, light and dark, fits a
phone-sized window as well as a wide screen, and is built to be used from the keyboard and with a
screen reader.

It is one small server that listens on `127.0.0.1` only, and one page that it serves: nothing to
install, nothing downloaded, nothing sent anywhere. The address it prints holds a secret that is
good for this run only, so only the window opened with it (or you) can give it orders. Models and
texts are kept in `~/GERMEN` (`--workspace` changes that); `--no-open` only prints the address. How
it is secured, and the rest: [docs/STUDIO.md](docs/STUDIO.md).

## Commands

| command | what it does |
|---|---|
| `germen info` | detected processors and SIMD level, model sizes and memory for the weights |
| `germen bench` | times real training steps for each model size and extrapolates the time for 100 M and 1 B tokens; `--scaling` repeats the measurement with 1, 2, 4, … threads and prints the speed-up curve; `--json FILE` saves the numbers |
| `germen demo-corpus` | writes a synthetic Romanian corpus generated from a small grammar (`--size-mb`) |
| `germen train` | trains from scratch (`--data FILE` or `--demo`), or continues with `--resume` |
| `germen generate` | samples text from a trained model, with a key/value cache (`--temperature`, `--top-k`, `--prompt`) |
| `germen verify` | checks a checkpoint: the size and SHA-256 of every file against `meta.json` (what `sha256sum` prints); `--recover` finishes a save that was cut off, without needing any text |
| `germen ui` | opens GERMEN Studio, the graphical interface: a server on `127.0.0.1` only and a window of its own (`--workspace`, `--port`, `--no-open`); see [docs/STUDIO.md](docs/STUDIO.md) |

Training options worth knowing: `--tokenizer byte|bpe` and `--vocab N` (the BPE size); `--batch` and
`--accum` (gradient accumulation: the effective batch is their product, the memory is that of
`--batch` alone); `--replicas N` (copies of the model that share each micro-batch between threads; 0
= automatic, 1 = off; choose a `--batch` at least as large as the number of threads); `--grad-clip`
(global gradient norm, default 1); `--lr`, `--warmup`, `--weight-decay`. `germen train --help` lists
them all. Add `--threads N` to any command to limit the number of worker threads (the default is
every logical processor). The messages of the command line are currently in Romanian.

## Model sizes

| preset | parameters | layers × heads × width | weights (f32) |
|---|---:|---|---:|
| `nano` | 0.85 M | 4 × 4 × 128 | 3 MiB |
| `micro` | 2.76 M | 6 × 6 × 192 | 11 MiB |
| `mini` | 10.82 M | 6 × 6 × 384 | 41 MiB |
| `small` | 25.44 M | 8 × 8 × 512 | 97 MiB |
| `base` | 85.35 M | 12 × 12 × 768 | 326 MiB |

The sizes are for the 256-token byte vocabulary; a BPE vocabulary of V tokens adds
2·(V − 256)·width parameters (embedding and output head). Training also keeps AdamW's two moment
estimates (≈ 4× the weights) plus the activations, which grow with `--batch` and `--seq`.

## How long does it take?

The cost of training is about **6 × parameters × tokens** floating-point operations (with bytes,
1 MB of text = 1 M tokens). Divide by what your CPU really delivers and you have the time.
`germen bench` measures exactly that on your machine.

Reference measurement — a small Linux virtual machine with **2 cores** (AVX-512), batch 16 × 256,
byte vocabulary, default settings (median of five interleaved runs on a noisy VM, so ± 10 %):

| preset | tokens/s | effective GFLOPS | 100 M tokens | 1 B tokens |
|---|---:|---:|---:|---:|
| `nano` | 8,366 | 54 | 3.3 h | 1.4 d |
| `micro` | 3,981 | 79 | 7.0 h | 2.9 d |
| `mini` | 1,464 | 104 | 19.0 h | 7.9 d |
| `small` | 647 | 106 | 1.8 d | 18 d |

Going from 1 to 2 threads made `mini` 2.04× faster here (733 → 1,494 tokens/s; 1.41× with
`--replicas 1`), so do not expect 20 cores to give 20×. Inside one graph, matrix products scale well
(≈ 1.8× on two threads), attention less (≈ 1.2×), and RMSNorm, rotary embeddings and the AdamW
update hardly at all; the replicas of version 0.3 run these parts side by side, but they need at
least as many windows per batch as there are threads. How far that carries on many cores has not
been measured: that is what `germen bench --scaling` is for — it prints the whole curve and the
number of threads beyond which nothing is gained. **Please run `germen bench --scaling --json
bench.json` on your machine** — it is the only way to know. A common rule of thumb for
compute-optimal training is about 20 tokens per parameter; small models are often trained on much
more.

Generation uses a key/value cache: 400 tokens from `mini` take about 1 second instead of 15 on the
same two cores.

## How it is built

```
crates/germen-core   the library: config, model, tokenizer, bpe, data, train, gradients,
                     generate, checkpoint, bench
crates/germen-studio GERMEN Studio: the server and the page behind `germen ui` (standard library only)
crates/germen-cli    the `germen` command line (clap)
```

* **Backend** — Burn's `flex`: pure Rust, matrix products through the `gemm` crate, SIMD kernels
  (AVX2/AVX-512/SSE, NEON) picked **at run time**, multi-threaded with rayon. A generic x86-64
  build therefore still uses AVX2 on a CPU that has it.
* **Model** — pre-norm blocks, fused QKV projection, causal attention, rotary embeddings,
  untied output head, GPT-2-style initialisation (output projections scaled by `1/√(2·layers)`).
  Training uses a block-wise causal attention that never computes the part above the diagonal
  (whole steps 8–28 % faster on two cores, depending on the model size; same values and
  gradients); evaluation uses Burn's fused kernel, and generation a key/value cache.
* **Tokenizer** — bytes, or a byte-level BPE with a GPT-2-like pre-tokenizer (a space sticks to the
  word that follows, merges never cross a word). Training of the vocabulary is deterministic and
  uses the training part of the text only; encoding is parallel; any byte sequence round-trips.
* **Training** — AdamW (β₂ = 0.95), linear warm-up then cosine decay, global-norm gradient
  clipping, gradient accumulation, validation on fixed windows (reported in nats per token and
  bits per byte), atomic checkpoints, progress through `Event` callbacks so a graphical front end
  can reuse the same loop.
* **Threads** — the whole command runs as one worker of its own thread pool, which starts parallel
  loops without a hand-off between threads (+ 7 % for `mini` to + 25 % for `nano` on two cores). On
  top of that every micro-batch is split into **data-parallel replicas**: copies of the model, each
  with an autodiff graph of its own, run forward and backward on different threads and their
  gradients are added in a fixed order, so a run is reproducible for a given number of threads and
  replicas (+ 23 % to + 44 % on two cores, depending on the model, with a lower memory peak;
  `--replicas 1` gives the ordinary step). Spreading micro-batches over threads that all shared one
  graph was tried in 0.2 and rejected; see [docs/PLAN.md](docs/PLAN.md).
* **Checkpoint** — a folder with `config.json`, `weights.bpk` (Burn's burnpack, f32),
  `optimizer.bpk` (AdamW moments), `tokenizer.json` and `meta.json`. Checkpoints written by version
  0.1 still load. Since 0.4 `meta.json` also records the size and SHA-256 of the other files,
  checked on every load: a damaged, truncated or mixed-up file is named in the error; older
  checkpoints load unchecked.
* **Studio** — `germen ui` starts a small HTTP server written on the standard library alone, bound
  to `127.0.0.1`, and serves one self-contained page (style, script and eight languages inside; it
  loads nothing). The page starts the same training, generation and measuring code as the command
  line and follows it through events. Who may talk to the server: a secret in the address (after
  the `#`, which a browser never sends), the `Host` and `Origin` headers, and a
  Content-Security-Policy that lets exactly one script run; each of these gates is attacked by
  tests over a real socket and switched off in turn by 18 mutation proofs, which the tests must
  notice ([docs/STUDIO.md](docs/STUDIO.md)).
* **Tests** — 247 tests (unit tests of the library, of the command line and of the Studio, and
  end-to-end runs of the real binary and of the server over a socket), among them: attention is
  causal, block attention equals Burn's attention (values and gradients), cached generation equals
  the full forward pass, accumulated micro-batches equal one large batch, replicas give the
  gradients of one big graph and a run is reproducible bit for bit, an optimizer step after
  reloading the AdamW state equals the step of an uninterrupted run, a checkpoint is never left
  half-written and non-finite weights are never saved, hostile tokenizer and configuration files are
  refused, every byte sequence survives the BPE round trip. A flipped bit in a checkpoint file is
  caught and the file named, and a save cut off at any moment is completed or discarded. The
  adversarial audit is in [docs/AUDIT.md](docs/AUDIT.md).

## Limitations (version 0.5)

* The checksums guard against accidents (a flipped bit, a truncated file, files of two different
  saves), not against someone who can edit `meta.json` as well; checkpoints written before 0.4
  cannot be verified.
* The whole corpus is read into memory (as 16-bit token ids); memory-mapped corpora larger than
  RAM are planned.
* Inside one graph several operations of the step (RMSNorm, rotary embeddings, the AdamW update) do
  not use extra cores well; replicas hide this when `--batch` is at least the number of threads, but
  how far that carries on many cores has not been measured yet.
* No mixed precision and no GPU backend.
* The `demo` corpus is synthetic and tiny: it proves the pipeline learns, nothing more.
* Not yet measured on Windows, macOS or a machine with many cores — see above for how to help.
* GERMEN Studio has been run in a real browser on Linux only, and in Chromium only; on Windows and
  macOS its tests pass in CI, but nobody has used its window there yet, and nobody has used it with
  a screen reader.

See [CHANGELOG.md](CHANGELOG.md), [docs/PLAN.md](docs/PLAN.md), [docs/AUDIT.md](docs/AUDIT.md) and
[docs/STUDIO.md](docs/STUDIO.md). Contributions are welcome — the open issues are a good start
([CONTRIBUTING.md](CONTRIBUTING.md)).

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Alexandru-Ionuț Chiuță (Alexio).
