# Contributing to GERMEN

Thank you for looking! Small, focused contributions are the easiest to review.

## Build and test

```bash
cargo build --release
cargo test --release --workspace      # 39 tests, a few seconds in release mode
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

Rust 1.95 or newer is required (Burn 0.22). Please do not use a debug build to judge speed —
it is 20–100× slower.

## Ground rules

* **Clean room.** Write code from papers, public standards and your own understanding. Do not copy
  code from other projects unless its licence allows it and you say so in the pull request.
* **Pure Rust by default.** The default build must stay free of native dependencies, network access
  and telemetry. Optional features (for example a GPU backend) go behind a Cargo feature.
* **Tests.** New behaviour comes with a test; bug fixes with a test that failed before.
* **Honest numbers.** If a change claims to be faster, show `germen bench --json` before/after on the
  same machine.
* Code and documentation are in English; the command-line messages are Romanian for now, and
  moving them to a message catalogue is one of the starter issues.

## Good first contributions

Look at the open issues: BPE tokenizer, AdamW state on `--resume`, KV cache, gradient
accumulation, global-norm clipping, safetensors export, a GUI, translations, and community
benchmarks. A benchmark from your own computer (`germen bench --json my-cpu.json`) is a perfectly
good first pull request.

## License

By contributing you agree that your contribution is licensed under the MIT License of this project.
