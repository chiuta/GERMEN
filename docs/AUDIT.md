# Adversarial audit of GERMEN — findings, fixes and what remains

Four rounds, the first two on 17 November 2026 and the third and fourth on 18 November 2026, all done by the
people who wrote the code (the author and an AI assistant), so "adversarial" means *tried hard to
break it*, not an independent review: [PLAN.md](PLAN.md) schedules one by someone else before 1.0.

* **Part I — round 1** (sections 1–6): the 0.2.0 source, fixed in 0.3.0. Findings `A-01` … `A-24`.
* **Part II — round 2** (sections 7–12): the 0.3.0 source (commit `c7bbf61`), fixed in 0.4.0.
  Findings `B-01` … `B-10`. Where it changes something Part I said, §4 says so in place.
* **Part III — round 3** (sections 13–18): the Studio of 0.5.0 (`germen ui`), the only part of GERMEN
  that listens on a socket. Findings `C-01` … `C-12`, and 18 mutation proofs of its gates.
* **Part IV — round 4** (sections 19–25): the network, the optimizer and the tokenizer of 0.5.0 checked against
  **independent implementations**, and a comparison with nanoGPT. Findings `D-01` … `D-04`, fixed in 0.6.0, and
  6 more mutation proofs (24 in all).

# Part I — round 1: 0.2.0 → 0.3.0

## 1. Scope and method

**Audited:** the 0.2.0 source (commit `7859929`), its command line, its CI and the claims of its
documentation. GERMEN is a local tool with no network input, so the inputs an attacker, a typo or a
failing disk can control are:

* the files of a model directory: `config.json`, `tokenizer.json`, `weights.bpk`, `optimizer.bpk`,
  `meta.json`;
* the corpus file and every number on the command line;
* signals (Ctrl-C, kill), a full or failing disk, and a second run on the same directory.

**Procedure.** Read the code for places that trust one of these inputs, do arithmetic that can
overflow, or can stop half-way through a write. Turn every suspicion into a reproduction against a
**0.2.0 binary built from that commit**; a finding is *reproduced* only if that binary misbehaves.
Fix it, and keep a test that guards the fix. For speed, change nothing on a hunch: interleaved A/B
runs (A B B A …) of the real `bench` command on one machine, medians, and a change stays only if it
wins.

**Severity.** *High* — a bad file or a failure destroys data or takes the process down by
exhausting memory. *Medium* — a silent wrong result, or data left inconsistent after a plausible
failure. *Low* — a misleading success, a panic (exit code 101) where an error message belongs, a
test or process gap. *Info* — checked and found fine, or hardening.

**Evidence codes.** **R** — reproduced against the 0.2.0 binary by
[`scripts/audit_repro.py`](../scripts/audit_repro.py) (it fails on 0.2.0 and passes on 0.3.0).
**L** — reproduced against the 0.2.0 library by
[`examples/audit_lib.rs`](../crates/germen-core/examples/audit_lib.rs). **I** — found by reading the
code; not reproducible on the Linux machine used here. **N** — checked and found safe in 0.2.0.
**T** — found by the project's own tests or verification while 0.3.0 was being finished (the code
in question was already in 0.2.0, which was not run against it).

## 2. Findings

| ID | Severity | What happens in 0.2.0 | Evidence | In 0.3.0 |
|---|---|---|---|---|
| A-01 | High | a `tokenizer.json` of a few hundred bytes, each merge doubling the longest token, makes the loader ask for exabytes | R | refused with a message |
| A-02 | High | a `config.json` with `d_model = 2^40` is accepted by `validate` and aborts on allocation; `n_layer = 2^32` and a negative `rope_theta` are accepted too, and nothing limits `block_size` | R, L | refused with a message |
| A-03 | Medium | a `weights.bpk` with a NaN weight loads; `generate` exits 0 and prints garbage | R | refused with a message |
| A-04 | Medium | a save that fails half-way leaves new weights next to the old optimizer state | R | the previous checkpoint is untouched |
| A-04n | Info | SIGKILL ×25 during back-to-back saves | R (negative) | same |
| A-05 | High | a diverged run writes NaN weights over the last checkpoint | R | never written, never loaded |
| A-06 | Medium | two `--resume` runs in one directory train and save concurrently | R | the second is refused at once |
| A-07 | Medium | `--grad-clip nan` silently switches clipping off | R | refused |
| A-08 | Low | `--resume` with fewer `--steps` than done exits 0 and rewrites `meta.json` | R | an error; equal steps: nothing to do, nothing rewritten |
| A-09 | Low | with BPE, `meta.json` stores the number of *tokens* in `corpus_bytes` | R | bytes |
| A-10 | Low | `--tokenizer bpe --val-fraction 2` panics (exit 101); `split_point` returns offsets past the end | R, L | refused with a message |
| A-11 | Low | `--val-fraction 0` reports "validation" loss measured on the training text, without saying so | R | prints a warning |
| A-12 | Low | `--demo-mb 1e9` aborts in the allocator | R | refused with a message |
| A-13 | Medium | `generate --max-new 5000000000` aborts before the first token | R | starts generating |
| A-14 | Low | seeds `2k` and `2k+1` start the same random stream, so `--seed 2` and `--seed 3` train on identical data | L | every seed has its own stream |
| A-15 | Info | `--threads 100000000` | R (negative) | explicit ceiling of 1,024 |
| A-16 | Low | Ctrl-C with `--accum 60` takes 29.8 s to act | R | 0.1 s |
| A-17 | Medium | the "will it fit in memory" check answers only on Linux: on Windows and macOS (the platforms GERMEN is released for) it limits nothing but a fixed 256 GiB | I | Windows and macOS probes |
| A-18 | Low | on Windows a save fails when an antivirus scanner or the indexer still holds a file that was just written | I | the rename is retried for about 10 s on Windows |
| A-19 | Low | `rust-version = "1.95"` in `Cargo.toml` is checked by nothing | I | CI job `msrv` |
| A-20 | Low | no scan of the ≈ 780 locked crates for known vulnerabilities | I | weekly `audit.yml`, judging the crates that are compiled |
| A-21 | Info | `unsafe` was not forbidden | I | `#![deny(unsafe_code)]`; one allowed FFI call |
| A-22 | Low | no test ran the real binary: exit codes and messages were untested | I | 7 end-to-end tests |
| A-23 | Info | documentation drift: the module documentation of `generate.rs` still said there is no key/value cache (0.2 added one) | I | corrected |
| A-24 | Low | Burn has one random generator per process; two threads that seed it and initialise a model at the same moment take shares of each other's stream, so the weights a seed promises depend on timing | T | seeding and drawing are one locked step |

Reading the evidence column: 15 findings were reproduced against the 0.2.0 binary or library, two
suspicions did not reproduce (A-04n, A-15), seven (A-17 to A-23) could only be established by
reading the code, and one (A-24) came out of verifying 0.3.0 itself in a clean directory.

### Details

**A-01 — hostile tokenizer.** On 0.2.0 the process aborted in 1.4 s (exit −6) under a 3 GB
address-space limit; without the limit it would have allocated until the kernel killed it. Fix:
`Bpe::from_merges` checks the length of the token a merge would produce (maximum 64 bytes, which is
the longest token training can build) *before* copying it. Tests: `bpe::hostile_merge_lists_cannot_exhaust_memory`
(64-byte tokens still load, 128 do not), `tokenizer::corrupt_tokenizer_files_give_clear_errors`, and
the end-to-end `a_hostile_tokenizer_file_is_an_error_not_an_out_of_memory_kill`.

**A-02 — hostile configuration.** 0.2.0 aborted in 0.1 s (exit −6). `GptConfig::validate` now sets
ceilings (context 16,384; vocabulary 2^20; 1,024 layers; width 16,384; feed-forward 65,536; 2^32
parameters in all), demands finite positive `rope_theta` and `norm_eps`, and the parameter count is
computed in `u128` with saturating arithmetic so that no field can overflow it. Tests:
`config::hostile_or_corrupt_configs_are_rejected_before_anything_is_allocated` (14 cases),
`parameter_arithmetic_saturates_instead_of_overflowing`, `the_largest_accepted_context_is_not_a_trap`,
`checkpoint::a_hostile_config_is_refused_before_anything_is_built`. The ceilings are far beyond
anything a CPU can train; they are constants at the top of `config.rs`.

**A-03, A-05 — non-finite weights.** 0.2.0 loaded a file with one NaN weight and generated garbage
with exit 0 (A-03). With `--lr 1e39 --grad-clip 0 --save-every 1` it exited 1 but all 851,712
weight words on disk were NaN or infinite (A-05). `checkpoint::ensure_finite` checks every
parameter tensor (`sum(x · 0)` is exactly 0 only if every element is finite, so no copy of the data
is made) both before a save, which then writes nothing, and after a load. Tests:
`checkpoint::a_diverged_model_never_replaces_a_good_checkpoint` (the directory is byte-identical
afterwards, not even a `.tmp` is left), `train::a_non_finite_loss_is_reported_whatever_the_replica_count`,
and the end-to-end `a_model_with_nan_weights_is_refused`.

**A-04 — half-written checkpoint.** 0.2.0 saved weights, optimizer, tokenizer, configuration and
progress one after another, each renamed into place as soon as it was written. With the file size
limited so that the 3.4 MB weights fit and the 6.8 MB optimizer state does not, the save failed
but `weights.bpk` and `meta.json` had already been replaced. `fsutil::Staged` writes every file to
`<name>.tmp`, flushes it to the disk, and only then renames them all, `meta.json` last; a failure or
a kill while writing leaves the previous set exactly as it was ("files changed: none"). Tests:
`fsutil::staged_files_appear_only_at_commit_and_replace_the_old_ones`,
`an_abandoned_or_failed_set_leaves_the_previous_files_untouched`,
`files_are_published_in_order_so_a_failure_never_reaches_the_last_one`,
`checkpoint::save_all_writes_one_complete_set_and_leaves_nothing_else`.
**A-04n:** 25 runs killed with SIGKILL at random moments while saving at every step all left a
checkpoint that loads, on 0.2.0 as on 0.3.0, so the rename-based writing of 0.2.0 already survived
that; the check looks for a checkpoint that does not load, not for a loadable but mixed one.

**A-06 — two runs, one directory.** 0.2.0 already refused a *fresh* run into an existing
checkpoint, but a second `--resume` of a directory in use was accepted and kept running next to the
first (it had not stopped when it was killed after 8 s). A `.germen.lock` file, locked with the
operating system's file lock for as long as the run lives, now makes the second run fail at once
(exit 1, 0.0 s); a lock left behind by a crash is harmless because the lock belongs to the process,
not to the file. Tests: `fsutil::a_second_claim_on_a_directory_is_refused_until_the_first_ends`,
`train::a_directory_in_use_by_another_run_is_refused_and_released_afterwards`.

**A-07 — option validation.** A NaN compares false with `> 0`, so `--grad-clip nan` trained without
clipping and said nothing. `TrainOptions::validate` now also checks `beta2`, `weight_decay` and
`grad_clip` for finiteness and range, and `batch × seq × accum` against overflow. Test:
`train::options_validation_rejects_values_that_would_break_the_run`.

**A-08 — `--steps` is the total.** 0.2.0 answered "Gata: 0 pași" with exit 0 and rewrote
`meta.json` with the settings of the new run. Fewer steps than done is now an error that says what
`--steps` means; exactly as many is a success that says there is nothing to do, so a "resume until
finished" loop ends. Neither rewrites anything. Tests:
`train::resuming_past_the_requested_steps_is_an_error_and_at_them_a_quiet_no_op` and the
end-to-end `resuming_past_or_at_the_requested_steps_is_explained`.

**A-09 — `corpus_bytes`.** For the same 200 KB demo text 0.2.0 stored 73,799 (the number of BPE
tokens) and 0.3.0 stores 209,748. Test:
`train::metadata_counts_the_corpus_in_bytes_even_when_its_tokens_are_fewer`.

**A-10, A-11 — validation share.** With the byte tokenizer 0.2.0 already refused `--val-fraction 2`;
with `--tokenizer bpe` it panicked while slicing the text (exit 101), because the text is cut at
`split_point` before any check runs. `split_point` now clamps (and `Corpus::build`
and the command line reject) any NaN or out-of-range share; `--val-fraction 0` still works but
prints a warning that the "validation" loss is measured on the training text. Tests:
`data::split_point_survives_nonsense_fractions`, the end-to-end
`a_nonsense_validation_share_is_a_message_not_a_panic` and
`a_run_without_validation_text_says_so`, and `the_validation_share_is_checked` in the command line.

**A-12, A-13 — sizes typed by a person.** `--demo-mb 1e9` saturated to `usize::MAX` bytes and
aborted (exit −6); `--demo-mb 0` was already refused. `generate --max-new 5000000000` called
`Vec::with_capacity` with that number and aborted before the first token; at most 65,536 tokens are
now reserved up front. Tests: the end-to-end `nonsense_sizes_are_refused_not_attempted`,
`sizes_in_mib_are_checked`, and `generate::an_absurd_max_new_does_not_abort_before_the_first_token`
(`usize::MAX`, `usize::MAX / 8` and 2^40).

**A-14 — seeds.** `Rng::new` was `(seed ^ C) | 1`, which throws away the lowest bit of the seed.
The generator is now seeded through one round of SplitMix64. Tests:
`data::every_seed_gives_its_own_stream` (2,000 seeds, 2,000 different streams) and
`neighbouring_seeds_are_not_correlated` (about half of the bits differ between neighbours).
**Consequence:** the same `--seed` gives a different run than in 0.2 (checkpoints written by 0.2
and 0.1 still load and resume).

**Library-level evidence** (`examples/audit_lib.rs`, run against both libraries). 0.2.0 prints
"1000 of 1000 pairs of seeds (2k, 2k+1) give identical streams" (A-14);
`split_point(100, 2) = 18446744073709551516` and `split_point(100, inf) = 101`, both past the end
of the text (A-10); and `validate()` accepts `d_model = 2^40`, `n_layer = 2^32` and
`rope_theta = -1` with `norm_eps = 0` (A-02). 0.3.0 prints "0 of 1000", offsets inside the text
(0 and 0), and an error for each of the three configurations.

**A-15 — thread count.** 0.2.0 reported "nu pot porni 100000000 fire de lucru" (exit 1 in 0.4 s, under
a 3 GB limit): the operating system refused to create the threads. Not a defect; an explicit ceiling of 1,024
with a clear message is hardening. Tests: `sys::absurd_thread_counts_are_refused` and the
end-to-end `an_absurd_thread_count_is_refused`.

**A-16 — Ctrl-C.** The stop request was looked at once per optimizer step; with `--accum 60` a
step takes about 30 s here. It is now checked before every micro-batch and the partial sum is
dropped, so no half step is ever applied. 29.8 s became 0.1 s. Test:
`train::a_stop_request_is_honoured_between_micro_batches`.

**A-17 — memory check on Windows and macOS (by inspection).** `available_memory_bytes` had a body
for Linux only, so `check_memory_fits` — the check that refuses a run that cannot fit — fell back
to 256 GiB elsewhere. It now asks `GlobalMemoryStatusEx` on Windows (the only `unsafe` call of the
project, in its own module) and `/usr/bin/vm_stat` on macOS (free + inactive + speculative pages,
absolute path so that nothing on `PATH` is run). Tests:
`sys::available_memory_is_known_on_the_platforms_germen_is_built_for` (runs on all three CI
systems) and `sys::vm_stat_output_is_understood` (the parser, on any system).

**A-18 — Windows rename (by inspection).** `fs::rename` over a file that an antivirus scanner or
the search indexer has just opened fails with "access denied" or a sharing violation. On Windows
only, `fsutil::rename_over` retries for about ten seconds (ten attempts, delays doubling from
20 ms); elsewhere such an error is real and is never retried. Tests:
`fsutil::retry_waits_out_transient_errors_only` and `only_windows_retries_renames` check the
logic; the real antivirus case could not be tried here.

**A-19 to A-23 — process.** CI gained a job that builds the whole workspace with Rust 1.95
(`cargo check --all-targets --locked`), a weekly and on-change RustSec scan (`audit.yml`), and
concurrency control; both crates `deny(unsafe_code)`; `crates/germen-cli/tests/cli.rs` runs the real
binary; stale statements in the documentation were corrected.

**A-24 — one random generator for the whole process.** Burn keeps a single generator per device,
and every initial weight is a draw from it, taken the first time something reads the weight. 0.2.0
seeded the device and then built the model, so two runs that start at the same moment in one
process (two threads of a program that trains two models, or two tests) take shares of each other's
stream, and neither gets the weights its seed promises. Nothing in the command line can do this —
one run per process — which is why the audit of the binary missed it. It surfaced when the 0.3
suite was run in a clean copy of the repository: `replicated_training_is_reproducible_for_a_fixed_seed`
failed in 2 of 13 runs with the tests in parallel (never in 15 runs alone), with different losses
from the very first step; the tests that build models without seeding them were drawing from the
same stream too.
Fix: `GptConfig::init_seeded` seeds the device, builds the model and draws every weight under one
process-wide lock (`model::exclusive_generator`); `Gpt::init_parameters` takes the same lock;
`train` and `bench` use it and no longer seed the device by themselves; every model a test builds is
seeded, and the one test that draws from the generator directly goes through the lock. Tests:
`model::the_seed_alone_decides_the_weights_even_when_other_threads_draw` (eight threads, two seeds,
twelve rounds each: it failed in six of six runs with the lock switched off, in the first round, and
passes with it) and `checkpoint::loading_a_checkpoint_draws_nothing_from_the_random_generator`
(loading is outside the lock, which is right only while it draws nothing). After the fix the core
suite ran 40 times in a row in parallel mode without a failure. Limits: the lock covers GERMEN's own
draws; a program that seeds or draws from the same Burn device itself is not covered. 0.2.0 has the
same pattern; it was not run against this test.

**The first run of the dependency audit (A-20).** On the pushed 0.3.0 commit `cargo audit` (1,527
advisories, 786 crates in `Cargo.lock`) reported 7 vulnerabilities — six in `zip` 0.6.6 and one in
`zstd-safe` 5.0.2 — and 10 warnings: eight more `zstd-safe` advisories marked unsound, and `chrono`
and `paste` unmaintained. The job failed. None of the vulnerable crates is compiled: `zip` is used
only by `tch`, the LibTorch binding behind Burn's optional `burn-tch` backend, `zstd-safe` only
through `zip`, and `chrono` only through `turso`, an optional dependency of `cubecl-environment`.
`cargo tree --target all` reaches none of them, and a release build leaves no artifact of them in
`target/release/deps`; `Cargo.lock` lists them because it records the optional dependencies of
Burn's sub-crates, not only what is switched on. `paste` is compiled (a proc-macro used by
`burn-dispatch`, `gemm`, `pulp` and `macerator`); it is unmaintained and has no fixed version, which
is a notice, not a vulnerability.

Rather than switch the check off, or list ten advisory IDs to ignore, the workflow keeps the full
report for people to read and takes its verdict from `scripts/audit_graph.py`: a vulnerability or an
*unsound* warning about a compiled crate fails the job; other notices about compiled crates are
listed; advisories about crates that are only in the lock file are listed as such; and a report the
script cannot read ends with exit code 2, so that a broken check never passes as a clean one. The
script was tried with the real cargo-audit 0.23.0 against five hand-made advisory databases (only
never-compiled crates: pass; a compiled crate with an unmaintained notice: pass, notice printed; a
compiled crate with an unsound advisory, or with a vulnerability: fail; no advisory at all: pass)
and against an empty, a malformed and a missing report (exit 2). The real database cannot be reached
from the sandbox; the corrected workflow ran on GitHub (commit `9fa10b2`, green) and printed 17
advisories in `Cargo.lock`, 16 of them about three crates that are never compiled (`chrono`, `zip`,
`zstd-safe`), one notice about a compiled crate (`paste`, unmaintained) and none that applies.

**Added without a finding.** `damaged_weight_files_are_errors_not_crashes` (six kinds of damaged
`weights.bpk`) and `a_run_with_a_tiny_context_still_ends_with_its_sample` are regression tests that
revealed no defect in the current code; they were not run against 0.2.0.

## 3. Performance audit

All numbers: a 2-vCPU Linux virtual machine with AVX-512, release build, byte vocabulary, batch
16 × 256, `germen bench`. The machine is noisy (±10 %), so only interleaved comparisons count.

### 3.1 Kept: data-parallel replicas (P-01)

Every micro-batch is split among copies of the model (`--replicas`; 0 = one per thread, at most one
per window). Each copy
has an autodiff graph of its own (`Module::fork`), runs forward and backward on a different thread,
and the gradients are added in a fixed pairwise order. The attempt of 0.2 shared one graph between
threads; it deadlocked with four workers on two threads and gained 3.5 % with two. Five interleaved
rounds, `--replicas 1` (the ordinary step) against the default (two replicas on two threads):

| preset | `--replicas 1` tokens/s | default tokens/s | ratio of medians | median of per-round ratios | per-round range |
|---|---:|---:|---:|---:|---:|
| `nano` | 6,807 | 8,366 | 1.23 | 1.16 | 1.02–1.40 |
| `micro` | 2,769 | 3,981 | 1.44 | 1.35 | 1.28–1.52 |
| `mini` | 1,090 | 1,464 | 1.34 | 1.29 | 1.28–1.41 |
| `small` | 475 | 647 | 1.36 | 1.38 | 1.33–1.42 |

`nano` is the noisy one: in one round the gain was 2 %. The README quotes the ratio of medians; the
honest summary is "roughly 15–45 % faster on two cores, depending on the model and on how the noise
falls".

* **Scaling from one to two threads** for `mini` (three interleaved rounds): the ordinary step
  761 → 1,075 tokens/s (1.41×); with replicas 733 → 1,494 tokens/s (2.04×).
* **Memory:** peak resident memory with two replicas was 1–3 % *lower* than with the ordinary step
  in all four presets (for `mini`, 1,573 → 1,531 MiB; medians of five), so the extra
  `(2 · slots − 1) · parameters · 4` bytes that `estimate_train_bytes_replicated` adds are
  conservative.
* **Correctness:** `train::replicas_give_the_same_gradients_as_one_graph` (batch 7 split into 2, 3, 7
  and 50 parts, with accumulation) compares with the gradients of one big graph;
  `replicated_training_is_reproducible_for_a_fixed_seed` runs 1, 2 and 3 replicas twice and demands
  bit-identical losses and logits; `replicated_training_learns_like_the_ordinary_step` checks that the
  loss falls; `replicas_never_deadlock_whatever_the_pool_size` runs pools of one to five threads with
  two to eight replicas under a watchdog; `replica_planning_respects_requests_windows_and_memory`.
  Reproducibility needed three things, each found by a test: a fixed reduction tree (rayon's
  `reduce` has none), a fixed assignment of replicas to slots, and drawing the initial weights in
  module order before any thread touches the model (Burn initialises lazily, and every draw advances
  one device-wide generator): `Gpt::init_parameters`.
* **No regression.** The ordinary step of 0.3 (`--replicas 1`) against the 0.2.0 binary, two
  threads, four interleaved rounds, medians: `nano` 7,086 → 7,670 tokens/s (+8 %), `micro`
  2,894 → 2,965 (+2 %), `mini` 1,144 → 1,119 (−2 %); `small` was not part of this check.
  Differences of this size are noise: an A/A control (the same code path under the two spellings
  `--replicas 1` and `--replicas 0`, one thread) differed by +1.2 % for `nano` (six rounds) and
  −3.1 % for `mini` (three rounds). The 0.2.0 binary itself measured 9–14 % below the table published
  with 0.2 (7,086 against 8,261 for `nano`, 2,894 against 3,180, 1,144 against 1,260), so absolute
  figures in this document are not comparable with older ones; only the ratios are.
* **One thread.** The default gains nothing there: `--replicas 0` resolves to one replica, the
  ordinary step. Forcing replicas does help on one thread (next section).

#### How many replicas?

The default is one replica per thread (at most one per window). Forcing more replicas than threads,
down to one window per replica, was measured against that default, interleaved, medians of tokens
per second (ratio to the default in brackets). Two threads:

| preset | batch | rounds | default (2) | 4 | 8 | 16 or `--batch` windows |
|---|---:|---:|---:|---:|---:|---:|
| `nano` | 16 | 4 | 8,616 | 11,240 (1.30) | 11,254 (1.31) | 12,030 (1.40) |
| `micro` | 16 | 4 | 4,131 | 4,104 (0.99) | 4,198 (1.02) | 4,374 (1.06) |
| `mini` | 16 | 4 | 1,436 | 1,416 (0.99) | 1,458 (1.02) | 1,546 (1.08) |
| `small` | 16 | 3 | 662 | – | – | 607 (0.92) |
| `mini` | 32 | 3 | 1,460 | – | 1,479 (1.01) at 16 | 1,538 (1.05) at 32 |
| `nano` | 64 | 3 | 10,555 | – | 11,532 (1.09) at 16 | 12,684 (1.20) at 64 |
| `nano` | 4 | 3 | 10,743 | 9,746 (0.91) | – | – |

On one thread, three rounds: `nano` (batch 16) 5,640 with the ordinary step against 6,153 with 16
replicas (1.09); `mini` 743 against 809 (1.09), and in another series 745 with the ordinary step
against 787, 801 and 803 with 2, 4 and 8 replicas.

So more replicas than threads is neither a clear win nor a clear loss: up to + 40 % for the
smallest model, − 8 % for `small`, − 9 % for `nano` at batch 4. A smaller graph stays in the
processor caches, while every replica adds one more set of gradients to the sum, a cost that grows
with the parameters; that would explain the pattern but I did not measure the sum separately. The
default therefore stays at one replica per thread, which is already 23–44 % faster than the
ordinary step, and `--replicas N` is the knob (the help text says so). A parallel or in-place
reduction of the gradient sets might make the larger count a win for every model; it has not been
tried. On a machine with at least as many threads as windows, such as 20 threads with `--batch 16`,
the default already is one window per replica.

### 3.2 Tried and rejected

None of these beat the baseline in interleaved runs. The raw numbers were not kept, so read this as
a record of what was tried, not as data.

* allocator tuning of the C library, and `mimalloc` (slower);
* lowering the thresholds at which Flex splits element-wise work across threads;
* a larger `--batch` (tokens per second did not rise);
* a parallel or fused AdamW update — the update is about 1.8 % of a step, so that is the most it
  could give;
* removing RMSNorm, rotary embeddings, the activation, the softmax or the mask altogether, an upper
  bound for anything that makes them faster: less than 2–4 % of a step each and about 10 % together.

On this machine a `mini` step spends about three quarters of its time in matrix products; the small
element-wise operations around them run on one thread, which is what replicas hide.

### 3.3 Not measured

* **Many cores.** Everything above is two cores. How far replicas carry on 20 cores is unknown;
  `germen bench --scaling --json bench.json` will tell, and at most `--batch` replicas exist.
* Windows and macOS speed; BPE training and encoding speed; the `dist` profile (fat LTO) against
  `release`; a batch smaller than the number of threads (there are then fewer replicas than
  threads; whether the idle threads help inside the matrix products was not measured).
* Training quality with replicas beyond one small test; the gradients are equal to those of one big
  graph up to rounding, which is the reason to expect none.

## 4. Trade-offs and residual risks

* **Disk space.** Staged writing needs room for a second copy of the checkpoint while a save runs.
* **Not atomic across files.** The renames of one save are a handful of fast operations with
  `meta.json` last, but a crash between them can leave new weights next to an old `meta.json`:
  every file is complete and loads, yet the set is mixed, and no file records which save it belongs
  to. A generation number written into every file would close this; it is not done.
  *Closed in 0.4.0: the manifest of every file in `meta.json` and the journal of a save (B-01,
  B-04, §8).*
* **The lock is advisory.** A program that ignores it (an older GERMEN, for one) is not stopped, and
  where a file system cannot lock (some network shares) the claim is skipped rather than refused.
* **Limits refuse by size, not by proof.** There is no fuzzing; the hostile-input tests are
  hand-written, and a corrupt `weights.bpk` is only as safe as Burn's burnpack reader (six kinds of
  damage were tried).
  *0.4.0 adds a fuzz harness and refuses by checksum, §7; it is still a sample.*
* **The ceilings** of A-02 could in principle refuse a legitimate gigantic model (a context over
  16,384, a width over 16,384, more than 2^32 parameters); they are named constants in `config.rs`.
* **The corpus is read whole.** A text file larger than the available memory is not refused before
  it is read (planned: memory-mapped corpora, 0.5). *0.4.0 refuses a text whose tokens cannot fit
  before reading it (B-10); the corpus is still read whole.*
* **Reproducibility** holds for a given seed, thread count and replica count, not across machines
  with different core counts (the order of floating-point sums differs), and not between 0.2 and 0.3.
  It also assumes that nothing outside GERMEN draws from the same Burn device while a model is being
  initialised (A-24).
* **The dependency check judges the default build.** Someone who builds with another Burn backend
  (LibTorch, say) compiles crates that the check ignores (A-20).

## 5. What has not been verified

CI settled several things that the sandbox could not. On both pushed commits (`9195e4d`, `9fa10b2`)
every job of the CI workflow was green: `rustfmt` and `clippy -D warnings`; all 117 tests and an
end-to-end smoke run (bench, train, resume, BPE, generate, `--replicas 3` and `1`) on Ubuntu 26.04,
Windows and macOS, which includes the memory-probe test on each system; and a build of the whole
workspace with Rust 1.95 (the sandbox has 1.97). The dependency audit was red on the first commit
and is green on the second (A-20). What is left:

* Windows and macOS beyond that smoke run: no training of real size, no speed measurement, no
  comparison of the `vm_stat` and `GlobalMemoryStatusEx` figures with another source, and the
  rename retry has never met a real antivirus scanner.
* The third-party GitHub Actions are pinned by tag, not by commit hash.
* A bigger machine (§3.3), and a release build on any machine but this one; the release jobs of CI
  (draft pre-release, binaries) run only for tags and were skipped.

## 6. Reproducing this audit

```sh
cargo build --release
# every finding that has a command-line trigger, against the new build and an old one:
python3 scripts/audit_repro.py --new target/release/germen --old /path/to/germen-0.2.0
python3 scripts/audit_repro.py --new target/release/germen A-06 A-07     # only some
# the dependency audit as CI runs it (needs `cargo install cargo-audit`):
cargo audit --json > audit.json; python3 scripts/audit_graph.py audit.json
# library-level findings (copy the file into the examples folder of an old tree to compare):
cargo run --release -p germen-core --example audit_lib
# speed: interleaved A/B of the real command
target/release/germen bench --threads 2 --steps 8 --replicas 1 --json a.json
target/release/germen bench --threads 2 --steps 8 --json b.json
```

`audit_repro.py` needs a POSIX system (process groups, resource limits and signals); on Windows run
it in WSL. It exits 1 if the new binary fails a scenario that is not purely informational.

---

# Part II — round 2: 0.3.0 → 0.4.0

## 7. Scope and method

**Audited:** the 0.3.0 source (commit `c7bbf61`), the claims of its README and CHANGELOG, and every
place where §4 of round 1 admitted a residual risk. The inputs and the severity scale are those of
§1. Round 1 reproduced suspicions one at a time; round 2 added three instruments:

* **A hostile-input harness**, [`scripts/audit_fuzz.py`](../scripts/audit_fuzz.py). It throws
  thousands of odd inputs at the real binary, each in a scratch directory, under an address-space
  limit and a timeout, and classifies every run: *ok*, *error* (a refusal with a message: fine),
  *running* (still going at the time limit, for runs that are meant to be long), *PANIC*, *ABORT*,
  *SIGNAL*, *HANG*. Five suites: damaged and mixed checkpoints (`ckpt`: truncations, bit flips,
  length-field bombs, files of another run, each through `generate` and through `train --resume`),
  extreme values in `config.json`, `meta.json` and `tokenizer.json` (`json`), extreme values for
  every option (`cli`), odd training texts (`data`: empty, NUL bytes, invalid UTF-8, one huge line,
  a directory, a device) and saves that cannot finish (`disk`). For `ckpt` there is a fourth kind of
  failure that round 1 could not see, because an exit code of 0 looks like success: a damaged
  checkpoint that is *accepted*.
* **Differential tests** wherever code was rewritten: the bounded pre-token scan against the
  unbounded one it replaces, and each SHA-256 implementation against a textbook one, on random
  inputs.
* **Mutation proofs**, [`scripts/audit_mutations.py`](../scripts/audit_mutations.py). For each fix
  the old defect (or one broken line of the new code) is put back in the source, the tests that are
  meant to guard it are run and at least one must fail, then the file is restored. A test that
  still passes against the defect proves nothing. The script has 26 entries (20 for the fixes of B-01 to B-10, six for
  the fast SHA-256); every one is caught by at least one test.

**Evidence codes** are those of §1, plus **F** — found by the fuzz harness (and then reproduced by
hand against the 0.3.0 binary). The same machine as in round 1: Linux, 2 virtual CPUs (Intel Xeon
at 2.1 GHz, with the SHA extensions), 8 GB of memory.

## 8. Findings

| ID | Severity | What happens in 0.3.0 | Evidence | In 0.4.0 |
|---|---|---|---|---|
| B-01 | Medium | one flipped bit in `weights.bpk` (or any other damage that leaves the file readable) is accepted: `generate` exits 0 and prints text from the damaged weights | R, F | every file has its size and SHA-256 in `meta.json`; a mismatch is refused, naming the file |
| B-02 | Low | a tensor without a parameter id in `optimizer.bpk` makes `--resume` panic inside Burn (exit 101) | R, F | refused with a message |
| B-03 | Medium | `--resume` with the optimizer state of another run is accepted; Burn silently skips the states it cannot match, so training goes on with moments that belong to nothing | R, F | refused |
| B-04 | Medium | the weights of one run next to the `meta.json` of another load without a word (a save cut off between two renames would leave exactly that: §4) | R, F | refused, naming the file and the remedy; `--resume` and `verify --recover` finish an interrupted save |
| B-05 | Medium | `--resume` without `meta.json` starts again from step 0 and trains over the weights it just loaded | R, F | refused (`generate` does not need `meta.json`) |
| B-06 | Medium | running the same `train` command again replaces a trained model in `--out` at the first save | R | refused unless `--resume` or `--overwrite` |
| B-07 | Low | `--tokenizer bpe` on text without blanks scans quadratically: 5 MB of one letter had not finished after 90 s | R, F | linear: 0.2 s |
| B-08 | Low | `norm_eps = 1e-300` or `1e300` and `rope_theta = 0.5` or `1e30` pass `validate`, and the model computes with numbers that overflow or vanish | R | bounds: `rope_theta` in [1, 10¹²], `norm_eps` in [10⁻¹², 1] |
| B-09 | Low | `bench --steps 2147483648` aborts allocating 16 GiB for its timings, `--steps 2^63` panics, `--warmup 2^64-1` wraps and prints `NaN ms/step` | R, F | refused or run; nothing is reserved up front |
| B-10 | Low | a text whose tokens cannot fit in memory is read whole first, and then the allocator aborts the process (`memory allocation of N bytes failed`) | I | refused before it is read, with the size and the limit |

Reading the evidence column: nine findings are black-box scenarios of `audit_repro.py` (B-01 to
B-09) that fail against the 0.3.0 binary and pass against 0.4.0; the harness found seven of them on
its own (F). B-10 could only be established by reading the code: reproducing it means asking the
operating system for more memory than the machine has, which can take the machine down with it.

**What the harness saw**, per suite and binary (Linux, one run of each; "bad" is a panic, an abort,
a signal or a hang; for `ckpt`, damaged checkpoints that were accepted are counted on their own):

| suite | runs 0.3.0 / 0.4.0 | 0.3.0 | 0.4.0 |
|---|---|---|---|
| `ckpt` — damaged, truncated and mixed checkpoints, each through `generate` and `train --resume` | 464 / 468 | 4 panics (B-02); 206 damaged or mixed checkpoints **accepted** (B-01, B-03, B-04, B-05) | 0 bad; 3 accepted, all of them cases `generate` may accept (it needs neither `optimizer.bpk` nor `meta.json`, and says so when the manifest is missing) |
| `ckpt` on checkpoints *written by 0.3.0* (no manifest) | — / 468 | — | 0 bad; 177 accepted (nothing records what their bytes should be: B-01 can only be closed from the next save on) |
| `json` — extreme values in the three JSON files | 440 / 461 | 0 bad | 0 bad |
| `cli` — extreme values for every option | 1,448 / 1,448 | 6 bad: `bench --steps` of 2³¹ and more, 3 aborts and 3 panics (B-09) | 0 bad (42 runs still going at the time limit, as long runs should be) |
| `data` — odd training texts | 112 / 112 | 3 bad: BPE on 5 MB without a blank, still running after 60 s (B-07) | 0 bad |
| `disk` — saves that cannot finish | 12 / 12 | 0 bad | 0 bad |
| all | 2,476 / 2,969 | 13 bad, 206 accepted | 0 bad |

### Details

**B-01 — damage that still parses.** In 0.3.0 the loader checked that a file could be read and that
the weights were finite; nothing said whether they were the weights that had been saved. Of 464
mutated checkpoints, 206 ran to the end with exit 0. The mutations that matter: 45 bit flips in the
tensor data of `weights.bpk` that `generate` accepted without any change in its output (a flipped
low bit), 3 more that changed it, a `weights.bpk` of another run, an `optimizer.bpk` of another run
or of another size, a deleted `meta.json` or `tokenizer.json`, 4 KiB of garbage appended to a file.
Fix: `meta.json` now has a `files` map with the size and SHA-256 of every other file of the set,
written by the same save that wrote the files, and checked at every load for the files that the
command reads (`generate`: weights, configuration and tokenizer; `train --resume`: also the
optimizer). The error names the file and shows the first twelve digits of both sums. `germen
verify DIR` checks all of them and also loads the weights and the optimizer. The sums are plain
SHA-256 of the file as `sha256sum`, `shasum -a 256` or `Get-FileHash` print it, so a checkpoint that
is published or copied can be checked without GERMEN; the manifests written by the audit's own runs
were compared with `hashlib` and `sha256sum` (16 files). Checkpoints without a manifest (0.3.0 and
older) still load, with a one-line notice that nothing could be vouched for, and get one at the next
save.

**B-02 and B-03 — the optimizer file.** Burn's `load_record` panics on a tensor that has no
parameter id and silently skips a state it cannot match, so a file that was almost right, or right
for another model, was either a crash or an unannounced restart of the moments. Fix:
`check_optimizer_file` reads the burnpack header itself, before Burn sees the file: every id must
be one of the model's, the shapes must be equal, the tensor names must be `"{id}.…"`, and the
number of states Burn ends up with must equal the number stored (so that nothing was skipped); a
`catch_unwind` around Burn's loader is the last backstop. B-03 is also tested with the manifest
stripped, so that the structural check is what is exercised and not the checksum.

**B-04 — a set that does not belong together; a save cut short.** §4 of round 1 named this as the
residual risk of staged writing: every file is whole and loads, yet the set is mixed. The manifest
closes the case in which the files came from different saves, however they got there. For the case
that a run dies *during* the renames, a save is now a small journal: every temporary file is
complete and flushed, and `meta.json.tmp` with the manifest exists, before the first rename. What a
dead run leaves is therefore enough to finish the job, and `checkpoint::recover` does: it completes
a save that got as far as publishing its first file and discards one that did not, so the
checkpoint is always the old set or the new one. `train --resume` does it before it loads;
`germen verify --recover DIR` does it on request; a load that meets a torn set is refused with a
message that names this remedy. In the SIGKILL experiment of round 1 (A-04n: 25 kills during
back-to-back saves) all 25 left a checkpoint that loaded in 0.3.0; in 0.4.0 two of the 25 left a
set cut off between two renames, both were detected and refused as an interrupted save, and both
were completed by `--resume` (none of the 25 unrecoverable).

**B-05, B-06 — silent restarts.** Both are a misleading success rather than a crash, which is why
nothing flagged them. `--resume` now needs the progress record (`meta.json`): without it the step
counter, and everything that depends on it, would start again from zero over weights that are
already trained. And a fresh `train` into a
directory that already holds a checkpoint stops before it trains or writes anything, with a message that
offers `--resume` and `--overwrite`; the CI smoke test and the end-to-end tests check that nothing
on disk changed. This is a change of behaviour, and the reason for 0.4.0 rather than 0.3.1.

**B-07 — the cliff in the pre-tokenizer.** `pretoken_end` found where a run of same-class bytes
ends by walking all of it, then cut the result to the cap of 64 bytes: every position of a run of
*n* bytes scanned *n* bytes, so 5 MB of the letter `a` is about 10¹³ byte comparisons. The scan now stops at the cap
(`reach = n.min(i + MAX_PRETOKEN + 1)`). Boundaries must not move, or every vocabulary already
trained would change: the old function is kept in the tests as a reference and compared with the
new one on random text and on long runs of every class. 0.3.0: timed out after 90 s; 0.4.0: 0.2 s.

**B-08, B-09, B-10 — numbers.** B-08: two floating-point fields of `config.json` were only checked
to be finite and positive. B-09: `bench` summed `warmup + steps` unchecked (with one step, a
warm-up of 2⁶⁴−1 wraps to zero steps in all, and the mean of nothing is `NaN`) and reserved a vector
of `steps` timings up front (16 GiB for 2³¹, capacity overflow for 2⁶³). The sum is now checked and
the reservation is capped at 65,536 entries; the vector grows as steps are really taken. B-10: reading
a big text can succeed and the tokens made from it still not fit, and then the allocator aborts the
process after the slow part is done. The corpus pre-check (`sys::check_corpus_fits`) compares the
size of the text, times 3 for the byte tokenizer (the text plus a two-byte token for each of its
bytes) or times 2 when a BPE vocabulary is learned, with 90 % of the memory the operating system
reports as available, and refuses before `std::fs::read`.

## 9. Performance audit

The same rule as in §3: nothing stays on a hunch; interleaved A/B runs (A B B A …) of the real
binaries on one machine, medians.

**Q-01 — the pre-tokenizer cliff (B-07).** On ordinary text nothing changed (BPE vocabulary on the
2 MiB demo corpus plus one step: 0.196 s against 0.182 s, three interleaved pairs, inside the
noise). On text with long runs it is the difference between 0.2 s and "did not finish".

**Q-02 — SHA-256.** The manifest has a price: every load hashes the files it reads. The first
implementation, the textbook one in safe Rust, ran at about 205 MB/s here, which made `germen
verify` of a 1 GB checkpoint (the `base` preset: 341 MB of weights and 683 MB of optimizer state)
take 6.5 s (measured), and would make loading the model for `generate` about 1.5 s slower (computed
from the speed, not measured). Two steps were measured, each on 64 MiB, best of four, in turns, in
four runs of the final code:

| version | speed | against the textbook |
|---|---|---|
| textbook (64-word schedule, shuffled variables) | 200–212 MB/s | 1.0× |
| portable, unrolled (16-word window, rotated names) | 221–240 MB/s | 1.1–1.2× |
| the SHA instructions of the processor (x86-64, found at run time) | 1,277–1,282 MB/s | 6.0–6.4× |

The last is as fast as `sha256sum` on the same machine (1.02 GB in 0.87–0.97 s, reading included,
in two sessions). With it, on the `base` checkpoint: `germen verify` takes 2.3–2.8 s in those two
sessions (hashing is about 0.8 s of it; the rest is loading and checking the tensors; the machine is
shared, so the second session was slower throughout, `sha256sum` included); loading for `generate`
costs 0.26–0.27 s more than in 0.3.0 (0.34 s → 0.61 s and 0.38 s → 0.64 s, four interleaved pairs
each) with the same peak memory, because the file is hashed in pieces of 1 MiB; and a one-step
`base` training run including its final save of 977 MB took 12.7 s against 13.1 s for 0.3.0 (three
pairs, ranges overlapping: no measurable cost). `bench` on `nano` is unchanged (0.52 s against
0.51 s over four pairs).

**Tried and not done.**
* *The `sha2` crate* would have added seven crates to the build for something the in-house version
  now does as fast as `sha256sum`.
* *A `--no-verify` flag.* At 0.27 s per 341 MB it is not worth a way to load unchecked files.
* *Hashing the files of a set on several threads* is bounded by the largest file (two thirds of
  the total): at most 1.5× on top of the above; not done.
* *The SHA instructions of ARM processors* (Apple silicon, Windows on ARM, Graviton) are not used:
  there is no such machine here to test them on. They run the portable version, about 240 MB/s,
  so a 1 GB checkpoint takes about 4 s to hash there.

## 10. Trade-offs and residual risks

* **The sums protect against accidents, not against a person.** Anyone who can edit `meta.json`
  can also edit the sums. SHA-256 here is a checksum that anyone can recompute, not a signature.
* **Compatibility.** A 0.4.0 checkpoint can be read by 0.3.0 (the extra `files` field is ignored)
  but 0.3.0 does not verify it. A checkpoint from 0.3.0 or earlier loads in 0.4.0 with a notice and
  gets its manifest at the next save; once a manifest exists, an optimizer file it does not list is
  refused. Scripts that re-ran `train` into an existing directory to replace its model need
  `--overwrite`.
* **A torn save needs one action.** `generate` refuses it and names the remedy; it does not repair
  it silently. The journal makes the repair possible, not automatic.
* **The memory pre-check (B-10) is a guard against the obvious mistake.** It goes by the memory the
  operating system calls available, so a limit set with `ulimit -v` or by a container is invisible
  to it, and a text that fits in memory but not in the limit still fails at the allocation.
* **The harness is a sample.** "0 bad" in the table above means none of those runs; it is not a
  proof, and the mutations are the ones that someone thought of (plus the random ones: bit flips).
* **`unsafe`.** There are now two calls in the workspace, not one: the Windows memory probe of
  round 1 and the call into the SHA instructions, made only after the processor has said it has
  them. The second is checked three ways: a differential test of both implementations on 10,000
  random states and inputs against the textbook version, the NIST and `hashlib` vectors on both,
  and six mutation proofs (S-01 to S-06) that break one line of the fast path at a time.
* **The portable fallback is slow** (see Q-02) on processors without the instructions or outside
  x86-64.
* **Reproducibility** is as in round 1; nothing in this round changes what a seed produces.

## 11. What has not been verified

* **CI.** The first push of 0.4.0 (`25c666b`) passed every test job: Ubuntu 26.04, Windows and
  macOS (all 156 tests and the smoke run with `verify`, the refusal of a fresh run and
  `--overwrite`), the build with Rust 1.95, and the dependency audit. The lint job failed: CI
  installs the newest stable Rust (1.100 that day), whose clippy has a lint
  (`chunks_exact_to_as_chunks`) that the 1.97 used here does not, and `integrity.rs` had ten
  fixed-size `chunks_exact` loops. They use `as_chunks` now, and with that commit (`70d295c`)
  every job of the CI workflow is green. (The dependency audit ran on the first push, which changed
  `Cargo.lock`; the second left the lockfile alone.) Lint with the toolchain CI uses.
* **The SHA instructions on any processor but the one used here**, and the portable version on a
  real ARM machine (it is tested on this one by calling it directly).
* **Windows and macOS under a crash.** The torn-save tests construct the state of a cut-off save by
  hand and are platform independent, but the SIGKILL experiment ran only on Linux, and no one has
  watched NTFS or APFS meet a power cut half-way through the renames.
* **A hostile but self-consistent checkpoint.** Someone who recomputes `meta.json` gets past the
  manifest and meets Burn's reader, whose limits are those of §4.
* **A bigger machine and real training runs** (§3.3): nothing in this round was measured above two
  cores.

## 12. Reproducing round 2

```sh
cargo build --release
# the nine black-box scenarios, against the new build and the 0.3.0 one (B-01 to B-09):
python3 scripts/audit_repro.py --new target/release/germen --old /path/to/germen-0.3.0 B-
# the hostile-input harness: all suites (about ten minutes), and checkpoints written by 0.3.0
python3 scripts/audit_fuzz.py --bin target/release/germen --json fuzz.json
python3 scripts/audit_fuzz.py --bin target/release/germen --only ckpt --base-bin /path/to/germen-0.3.0
# mutation proofs (about ten minutes; edits the sources in place and restores them):
python3 scripts/audit_mutations.py
# the speed of SHA-256, every implementation in turn:
cargo test --release -p germen-core --lib sha256_throughput -- --ignored --nocapture
# a checkpoint can be checked without GERMEN; compare with the `files` map of meta.json:
sha256sum DIR/weights.bpk DIR/optimizer.bpk DIR/tokenizer.json DIR/config.json
```


# Part III — round 3: the Studio of 0.5.0

## 13. Scope and method

**Audited:** the Studio as it stands in the 0.5.0 source before its first commit — `crates/germen-studio`
(the server, the page with its CSS, JavaScript and eight languages), the command `germen ui`, and the
three additions to the library that it needed (`Bpe::train_cancellable`, `generate::generate_until`,
`data::sample_blocks`). Rounds 1 and 2 compared a new build with an old one; there is no earlier Studio,
so the findings below are defects of the *draft* that a test, a linter or a browser found while it was
being verified, each fixed before the release. The model of the attacker is in
[STUDIO.md](STUDIO.md) §4.1: a program, or a web page of another origin in the user's own browser
(including one that makes the browser resolve a hostile name to `127.0.0.1`: DNS rebinding), that can
send bytes to the port; and the bytes themselves, from a typo to a hostile head.

**Instruments.** Three, the first two in the repository:

* **Tests that attack a real server over a socket** (21 in `crates/germen-studio/tests/http.rs`), 60 unit
  tests of the crate and 2 tests of the command that start `germen ui` as a user would.
* **Mutation proofs**, the entries `T-01` … `T-18` of
  [`scripts/audit_mutations.py`](../scripts/audit_mutations.py): each gate of the server is switched off in
  turn and a test must fail (§15).
* **A real browser**: Chromium 153 driven by Playwright 1.63, axe-core 4.14.1 and ESLint 9 (their scripts and
  configuration are not in the repository; the sweeps are described in [STUDIO.md](STUDIO.md) §8). Every
  page and flow, in light and dark, in eight languages, at five widths; and the emulation of forced colours.

**Severity** is that of §1. **Evidence codes:** **R** — reproduced: a test of the command failed on the
draft. **T** — caught by a test of the page's own rules. **W** — found by driving the page in Chromium.
**F** — found in Chromium's emulation of forced colours (Windows High Contrast). The machine is that of
the other rounds: Linux, 2 virtual CPUs, 8 GB of memory.

## 14. Findings

| ID | Severity | What happens in the draft | Evidence | Now |
|---|---|---|---|---|
| C-01 | Low | `germen ui --max-upload-mb 1000001` starts the server although the message says the range is 1 … 1,000,000: only zero and an overflow of the product were refused | R | refused before anything starts, naming the option and the range; the test of the command tries 0, 1,000,001 and a number that overflows |
| C-02 | Low | the demonstration page keeps its settings in the browser's `localStorage`, which the page's rule test forbids for both pages | T | memory only: lost when the tab is closed |
| C-03 | Low | a copy button throws after its `await`: it reads `event.currentTarget` once the handler has returned | W | the element is taken before the `await` |
| C-04 | Low | on a narrow range the labels of the y axis are identical ("6E-4" four times, "10.9K" three times) | W | labels get as many digits as it takes to tell them apart (up to five more) |
| C-05 | Low | a logarithmic axis can have a single tick | W | at least three gridlines: powers of ten, then 2 and 5, then every digit, then round numbers |
| C-06 | Low | the tooltip of a curve that has no point under the cursor does not say at which step it was read | W | a series with no point at the cursor shows its nearest one and says where that is |
| C-07 | Low | long labels of buttons widen the whole page on a phone; the text of tabs is clipped in German and Portuguese | W | the sweep of every page × 8 languages × 5 widths (down to 320 px) finds no overflow and no clipped text |
| C-08 | Low | the side bar with icons only has no names: nothing says what an icon is | W | every item has an accessible name, and a label beside it on hover and on keyboard focus |
| C-09 | Low | the screen shown after Quit looks like a crash | W | it says that the Studio has closed, that the window can be closed, and how to come back |
| C-10 | Low | in forced colours the progress bars lose their track (a background is dropped there): a fill of 1 % floats in nothing | F | every bar has an edge in the system's text colour |
| C-11 | Low | in forced colours the sprout of the side bar, drawn in the page's pale ink, nearly vanishes on a white canvas | F | drawn in the system's colours |
| C-12 | Low | in forced colours the dot of a chart legend vanishes, for the same reason as the track | F | the marks of a legend keep the series' colours, as the charts do |

None is a flaw of the server's gates: those were attacked by the tests of §13 and by the proofs of
§15 and no gate was found missing. Everything above is the page, or the command's own checks.

**Also found, and fixed without a change of behaviour.** By clippy and ESLint: a function that nothing
called (`until`), a type alias kept alive with `#[allow(dead_code)]`, a timer variable of the demonstration
replay that was used before the line that assigns it, and two names that shadowed others.

**A false alarm that is not counted.** In the emulation of forced colours with a light palette axe-core's
contrast rule reports 9 elements of the side bar (1.12:1, "#eaf3fb on #ffffff"). axe reads the page's own
text colour; the browser's computed colour for the same elements is the system's (dark blue on white),
and a screenshot shows them legible. The report is a limit of the tool in that mode, and the three
defects C-10 … C-12 were found by looking at screenshots, not by it.

### Details

**C-01 — a range that was documented and not enforced.** `--max-upload-mb` is the size of the largest
text the page may upload; the help says "1 … 1,000,000" and the error for a bad value repeats it. The
command, though, refused only 0 and a number whose product with 1,048,576 overflows. The finding came
from the first test of the command, which gives the option 1,000,001 and waits for the command to end:
the server started instead, printed its address and went on serving, and the test hung for more than nine
minutes before the sandbox killed it. Fix: an explicit `(1..=MAX_UPLOAD_MB).contains(…)` in `cmd_ui`, with
the constant defined once above it. The test now tries `0`, `1000001` and `99999999999999999999999`: each
must fail without a panic; the refusal of `0` must also exit with code 1 and name `--max-upload-mb`; and a
refused start must not create the workspace.

**C-10 … C-12 — what "forced colours" removes.** In that mode the browser replaces the page's text,
border and background colours by the system's; a *background* that carries meaning disappears, and
(in Chromium) the colours that an SVG gets from `fill` and `stroke` are not replaced at all. The three defects are the two sides of that rule: a
track and a legend dot drawn *only* with a background, and a drawing filled with colours chosen for
a dark side bar that is white in this mode. Fixes, all in the `forced-colors` block of `style.css`:
`.bar` and `.bar-track` get a 1 px `CanvasText` edge; `.sprout` is coloured `CanvasText` and `Highlight`;
`.legend i` is exempted from the replacement, like `.chart .ser`. Checked again in both palettes
(light and dark) on the run, Write and Settings pages, and on the tooltip of the icons-only side bar.

## 15. Mutation proofs

`python3 scripts/audit_mutations.py T-` (about three minutes; it edits the sources in place and restores
them, then rebuilds). For each row the defect is put back, the named test runs, and it must fail. All 18 are
**caught**.

| ID | What is switched off | Test that must fail |
|---|---|---|
| T-01 | the `Host` header is no longer checked (DNS rebinding gets in) | `http::a_request_meant_for_another_host` |
| T-02 | the `Origin` header is no longer checked | `http::requests_that_another_page_made` |
| T-03 | `Sec-Fetch-Site` is no longer checked | `http::requests_that_another_page_made` |
| T-04 | the token is no longer required | `http::the_api_needs_the_token` |
| T-05 | any token is accepted | `http::the_api_needs_the_token` |
| T-06 | the comparison of tokens accepts a prefix of the real one | `auth::` (unit tests) |
| T-07 | the policy lets inline script run (its hash is replaced by `'unsafe-inline'`) | `the_page_is_served_with_a_policy` |
| T-08 | the policy no longer asks for Trusted Types | `page::` (unit tests) |
| T-09 | a model name may contain a slash (a way out of the workspace) | `names_from_the_network` |
| T-10 | an upload may be larger than the limit | `http::a_text_can_be_uploaded_within_the_limit` |
| T-11 | an upload may replace a file that is there | `http::a_text_can_be_uploaded_within_the_limit` |
| T-12 | a second `Host` header is merged instead of refused (request smuggling) | `malformed_requests…` |
| T-13 | the number of connections is not limited | `http::too_many_connections…` |
| T-14 | a JSON body may be of any size | `malformed_requests…` |
| T-15 | the server listens on every interface, not on `127.0.0.1` alone | `loopback_only` (unit test) |
| T-16 | a model that is being trained can be checked (read half-written) from outside | `http::a_model_in_training_is_not_touched` |
| T-17 | a fresh training over a trained model is accepted | `http::a_model_is_trained_from_scratch` |
| T-18 | the page's Close does not close the server | `http::the_page_can_close_the_server` |

**A proof that was bound to the wrong test.** At the first run T-14 was *not caught*. It was not a gap in
the tests: the entry named `bodies_are_checked_before_anything_runs`, a test of bodies that are the wrong
*kind* (not JSON, an array, deep nesting, a NUL byte, invalid UTF-8…), none of them about size, while the
case that T-14 removes — a declared `Content-Length` of 5,000,000 bytes, refused with 413 — lives in
`malformed_requests_are_refused_in_the_right_words_and_the_server_goes_on`. With the right filter it is caught. The lesson is the one of round 2: a mutation proof proves something only for the
test it names, so each row names its test.

## 16. Trade-offs and residual risks

What the Studio does not defend against is said in [STUDIO.md](STUDIO.md) §4.4 and is not repeated here:
a user who can run programs as you, a shoulder-surfer, a program that can sniff loopback with administrator
rights, a hostile but self-consistent checkpoint (it meets the reader of Burn as on the command line), and
the absence of a reviewer other than the authors. Beside those:

* **The server's HTTP is our own.** It handles one request per connection and refuses what it does not
  implement (no keep-alive, no chunked bodies, no upgrade), which keeps `http.rs` at about 570 lines and
  adds no dependency to the workspace. The price is that nobody else has read a parser that sits on a
  socket; the tests of §13, T-12 and T-14 and the limits of STUDIO.md §4.3 are what stands in for it.
* **Trusted Types are enforced by Chromium only.** In another browser the other directives of the policy and
  the absence of any way of writing markup from a string (a test of the page) still hold; the third line
  of defence is missing there.
* **The jobs live in memory.** Twelve jobs and 4,096 events each are kept; closing the Studio drops them. The
  models are on disk and are never lost that way; the history of a run and its curves are.
* **Loopback is not private on a shared computer** (STUDIO.md §4.4): the advice there is `--no-open` and
  pasting the address by hand.

## 17. What has not been verified

* **Windows and macOS beyond CI.** The tests of the command run there (every job of the CI workflow is green
  on `26d0581` and `303cfb3`, the clippy job with the newest stable Rust); nobody has used the page on them
  with a real browser (how the application window opens, the file picker's drives, how a notification is
  asked for).
* **Browsers other than Chromium.** Only Chromium was run.
* **Screen readers.** The structure passes axe and was read in the order a reader would; no person using
  NVDA, JAWS, VoiceOver or Orca has used it. **Forced colours** were looked at only in Chromium's emulation
  of its default light and dark palettes, not on a real Windows theme.
* **Many cores and long runs.** Trainings of a few minutes on two cores; not days, and not a page left open for
  days.
* **An independent review** of `http.rs` and `auth.rs`, small enough to be read in an afternoon.

## 18. Reproducing round 3

```sh
# the tests of the Studio and of the command (release build):
cargo test --release --locked -p germen-studio
cargo test --release --locked -p germen-cli --test cli
# the 18 mutation proofs (about three minutes; edits the sources in place and restores them):
python3 scripts/audit_mutations.py T-
# the demonstration page, written by the program itself (it has no program behind it):
germen ui --write-demo demo.html
# the page in a browser: start it, open the address it prints, and follow the sweeps of STUDIO.md §8
germen ui --workspace /tmp/gw
```


# Part IV — round 4: independent implementations, 0.5.0 → 0.6.0

## 19. Scope and method

Rounds 1 to 3 looked for ways to make GERMEN fail. Round 4 asks a different question: *is what it computes
the right thing?* A test written by the author of a function shares the author's idea of what the function
is for, so three parts were re-written by a second implementation that shares **no code and no automatic
differentiation** with the first, and compared number by number.

| Part | Reference | What is compared |
|---|---|---|
| the network | `scripts/reference_check.py`: the architecture of [PLAN.md](PLAN.md) in float64 NumPy (pre-norm blocks, RMSNorm with a gain, fused q/k/v, rotary embeddings in the rotate-half layout, causal attention scaled by 1/√head_dim, squared-ReLU feed-forward, untied head, mean cross-entropy) | the logits of the three paths of the model (training with blocked attention, evaluation with Burn's fused attention, cached generation with the key/value cache), the loss, and **the gradient of every parameter by central finite differences** (ε = 10⁻⁵) |
| the optimizer | the same script: AdamW written by hand (decoupled decay `p ← p·(1 − lr·wd)`, bias-corrected moments, ε outside the square root) | five steps at five different learning rates, every parameter |
| the tokenizer | `scripts/reference_bpe.py`: a naive byte-pair encoder that recounts every pair from scratch at each merge (no heap, no incremental counts, no cache, no threads) and follows only the rules written in `bpe.rs` | the list of merges, the tokens of the training text, the tokens of a different text, and decoding |

The Rust side is dumped by two examples that are not part of the program
(`crates/germen-core/examples/reference_dump.rs`, which gives the model deterministic random weights and
writes weights, batch, loss, logits, gradients and the AdamW steps as JSON, and `bpe_dump.rs`).

Also in this round: the fuzz harness of round 2 was run again on the 0.6.0 build (2,501 runs over checkpoints,
JSON, the command line, text and the disk: 0 bad by the harness's own criteria: no hang, no panic, no crash; the harness's own line is
in §20.2), and the Shakespeare runs of §21 were made on both versions. **The same limits apply as in the other parts**: the author and an AI
assistant, on one machine (Linux, 2 virtual CPUs, 8 GB), not an independent review.

**Severity** is that of §1; evidence codes: **I** — found by an independent implementation, **R** — reproduced
by a test of the command or the library, **M** — found by a mutation proof that the tests did not notice.

## 20. Findings

### 20.1 What agreed (not findings)

| Comparison | Largest difference | Allowed |
|---|---|---|
| logits, training path (blocked attention) | 9 × 10⁻⁶ (scale 6.7) | 2 × 10⁻⁴ of the scale |
| logits, evaluation path (fused attention) | 1.4 × 10⁻⁵ | the same |
| logits, cached path (three chunks: 20 + 17 + 1 × 33 tokens) | 7 × 10⁻⁶ | the same |
| loss | 2 × 10⁻⁷ | 2 × 10⁻⁴ |
| gradient of each of the 15 parameter tensors, by finite differences | 8 × 10⁻⁷ (scale 0.07 … 0.33) | 2 × 10⁻⁴ of the scale |
| AdamW, five steps, every parameter | 3 × 10⁻⁷ | 10⁻⁵ of the scale |
| the same on a model with `d_model` 128 and 300 tokens, so that attention is blocked in three | 5 × 10⁻⁴ on logits of scale 17 (3 × 10⁻⁵ relative); AdamW 3 × 10⁻⁷ | the same |
| causality: logits of earlier tokens when a later one is changed | 0 | 0 |
| BPE, 600 entries trained on 200 kB of text | 344 of 344 merges equal; the 92,812 tokens of the training text and the 30,849 of another text equal; decoding gives the text back | exact |

The weights are `f32` in Rust and `f64` in the reference, so differences of 10⁻⁵ to 10⁻⁷ are rounding; no
comparison was close to its limit, and none had to be loosened to pass. The finite-difference check covers
every element of every tensor, including the gains of the normalisations and the embedding. The larger
model's gradients were not checked this way (the reference is a Python loop: hours); its logits, loss and
optimizer were.

### 20.2 Findings

| ID | Severity | What 0.5.0 does | Evidence | Now (0.6.0) |
|---|---|---|---|---|
| D-01 | Medium | `--weight-decay` also pulls the RMSNorm gains towards zero (AdamW decays every parameter). The loss is trying to keep a gain near 1; the decay shrinks it by `lr·wd` of its value at every step: 0.7 % over 2,000 steps at the defaults, **10 %** with the nanoGPT recipe (`lr 1e-3`, `wd 0.1`), 6 % over 20,000 steps at the defaults. nanoGPT, llm.c and the GPT-3 recipe decay matrices only | I | `train::adamw_step`: the step is taken as usual and the decay that fell on the one-dimensional parameters is given back (`p_old·lr·wd` added), so a gain moves by its own update only. Verified against the hand-written AdamW (the matrices are still decayed exactly as before) and by two tests |
| D-02 | Medium | a run that is stopped and resumed is not the run that was not stopped: one generator drew the training windows and a resume restarted it, so the resumed half trained on other windows than the same steps of an uninterrupted run. Nothing is lost, but "the same seed gives the same model" was false across a pause, and the pause could not be told from a change of the data | R | every step draws from its own stream, `Rng::for_step(seed, step)`. A test trains 6 steps straight, and 3 + 3 with a stop between, and compares the weights as numbers (the files differ in their bytes: they carry the parameters' random ids): **equal**. *Behaviour change:* the same `--seed` draws other windows than in 0.5.0 |
| D-03 | Low | `--lr × --weight-decay` ≥ 1 is accepted. The factor `1 − lr·wd` is then zero or negative: the weights flip sign at every step and grow. `--weight-decay 1e9` trained a model of noise without a word | R | refused before anything starts, with a message that names both options and the product |
| D-04 | Test gap | the mutation "every step draws the same windows" (D-02b) was **not** caught: nothing tested that two steps, or two seeds, draw different data | M | a test that every (seed, step) pair has its own stream and the same one every time |

D-01 and D-02 are defects of what the program *does*, not of its arithmetic: both were invisible to every
earlier test, and both are found only by asking what a reference implementation of the *recipe* (nanoGPT's)
would do. Neither changes the format of a checkpoint: a model trained with 0.5.0 loads and generates as before.

**An error in this round's own draft, caught before release.** The comment that sizes D-01 first said "3 % in
all"; the computation (`1 − (1 − lr·wd)^steps`, 2,000 steps, `lr·wd = 10⁻⁴`) gives 18 % at the *peak* rate
and, with the cosine schedule that brings the rate down to a tenth, **10 %**. The comment, the changelog and
this page say 10 %.

**A line in the harness.** The fuzz harness tries `--out /` among its hostile values and expects a refusal.
Run as root (as in this sandbox) the request is legitimate and succeeded: it wrote a checkpoint into the
root of the machine, which then made a test of the Studio fail (`/` "holds a model"). It is not a defect of
GERMEN, which does what it is asked; the harness now skips that value when it runs as root.

## 21. Comparison with nanoGPT

nanoGPT is the reference that most readers know. Its README gives a recipe for a computer without a GPU:
character-level Shakespeare, 4 layers, 4 heads, 128 dimensions, a context of 64, batch 12, 2,000 iterations,
no dropout, validation loss about 1.88; on a GPU (A100, a model of 10.7 M parameters, about three minutes)
1.47. GERMEN was trained on the same text (`tinyshakespeare`, 1,115,394 bytes), with the same shape. The two
0.6.0 rows were run with exactly the command below; the two 0.5.0 rows were run earlier in the round with the
same preset, context, batch and steps, and options that were not written down, so they are shown as a spread
and not as a measurement:

```sh
germen train --data input.txt --out run --preset nano --seq 64 --batch 12 --steps 2000 \
             --lr 0.001 --eval-every 250 --eval-batches 20 --weight-decay W --seed 1
```

| Run | Validation loss (nats per byte) | Time (2 virtual CPUs) |
|---|---|---|
| GERMEN 0.5.0, seed 1 | 1.7166 | 125 s |
| GERMEN 0.5.0, seed 2 | 1.7250 | 133 s |
| GERMEN 0.6.0, seed 1, `--weight-decay 0.01` (the default) | 1.7262 | 211 s (other jobs were running) |
| GERMEN 0.6.0, seed 1, `--weight-decay 0.1` (nanoGPT's) | 1.7274 | 226 s (the same) |
| nanoGPT, CPU recipe (published) | ≈ 1.88 | not comparable (other machine) |

**Reading.** On this recipe GERMEN reaches a validation loss about 0.15 nat lower than the one nanoGPT
documents. The spread between seeds and versions (1.717 … 1.727) is about a hundredth, so the difference
is real but should not be over-read, for these reasons, all of which favour or disfavour GERMEN by unknown
amounts:

* The architectures differ (rotary embeddings, RMSNorm and a squared-ReLU feed-forward in GERMEN; learned
  positions, LayerNorm and GELU in nanoGPT), and so do the initialisation and the schedule.
* nanoGPT's tokens are the 65 characters of the file; GERMEN's are bytes (256, of which the file uses 65), so
  the two start from different untrained losses (ln 65 and ln 256) and the per-token figures are close but not
  strictly the same quantity.
* The validation text differs: the last 5 % of the file here, the last 10 % in nanoGPT.
* The 1.88 is the figure in nanoGPT's README, not a run on this machine; **nanoGPT was not run here**.

**What this does show:** that the network, once checked against a reference, learns at least as well as the
well-known recipe on a task small enough to run here; and that `--weight-decay 0.1` does not change the result
at this size (1.7274 against 1.7262), so the default `0.01` stays. A larger model, a longer run or another text
could rank them differently: that was not tried.

**Speed is not compared.** GERMEN's throughput is in the README (13,229 tokens/s for `nano` in the
recording of 0.6.0, on 2 virtual CPUs); no other framework was run on this machine.

## 22. Mutation proofs

Six more entries in [`scripts/audit_mutations.py`](../scripts/audit_mutations.py) (`D-xx`): the defect is put
back, the tests must fail, the source is restored.

| Entry | The mutation | Caught by |
|---|---|---|
| D-01a | the weight decay reaches the norm gains again (a plain AdamW step) | `weight_decay_reaches_the_matrices_but_not_the_norm_gains` |
| D-01b | the training loop takes the plain AdamW step | `the_training_loop_leaves_the_norm_gains_alone_whatever_the_weight_decay` |
| D-01c | the decay is given back to the matrices too | the same two tests |
| D-02a | a resume draws other windows | `a_run_that_is_stopped_and_resumed_equals_an_uninterrupted_one` |
| D-02b | every step draws the same windows | `every_step_of_every_run_has_its_own_stream_and_the_same_every_time` (new: D-04) |
| D-03 | `lr × weight-decay` of 1 or more is accepted | the table of bad options |

All six are CAUGHT. With the 18 of round 3 (§15) that makes 24 proofs for the last two rounds, besides those
of rounds 1 and 2, and every one is caught.

## 23. Trade-offs and residual risks

* **The same seed, other batches.** A model trained with 0.6.0 cannot be reproduced from a 0.5.0 command line.
  This is the price of D-02; the alternative (keeping the old stream) would have kept resume inexact.
* **Exact resume has conditions**: the same machine, thread count, replica count, options and seed. With a
  different number of threads or replicas the gradients are probably summed in another order, so the weights
  would differ by rounding but not by data; this was not tested.
* **Decaying only the matrices is a choice**, shared with the recipes it follows; it is not proven better for
  these sizes. At 2,000 steps it changed nothing measurable.
* **The gains' share is given back after the step**, not skipped, because Burn's AdamW has no per-parameter
  exemption: the result equals "no decay" up to one rounding of `f32`, not bit for bit.

## 24. What has not been verified

* **Speed against other frameworks** (candle, llm.c, PyTorch on the same CPU): not run.
* **Quality at 10 M parameters and above** against nanoGPT's GPU baseline (1.47): not run; the comparison of
  §21 is the small CPU recipe only.
* **The gradients of a large model** by finite differences; the logits, the loss and AdamW were checked there.
* **Windows and macOS numerics**: the tests pass there in CI, the reference comparison was run on Linux only.
* **The sampler's distribution** beyond the existing test (temperature, top-k, top-p against an independent
  sampler).
* **BPE against other tokenizers** (Hugging Face, tiktoken): their pre-tokenizers differ by design, so only the
  rules written in `bpe.rs` were checked, against a naive implementation of themselves.
* **An independent review**, still pending (see [PLAN.md](PLAN.md)).

## 25. Reproducing round 4

```sh
# the network and the optimizer against NumPy (about 35 s with the gradients; --skip-gradients: seconds):
cargo run --release -p germen-core --example reference_dump -- /tmp/ref.json
python3 scripts/reference_check.py /tmp/ref.json
# a model with attention in several blocks (d_model 128, 300 tokens); skip the Python gradient loop:
cargo run --release -p germen-core --example reference_dump -- /tmp/ref_big.json 128 2 256 2 300
python3 scripts/reference_check.py /tmp/ref_big.json --skip-gradients
# the tokenizer against the naive one:
cargo run --release -p germen-core --example bpe_dump -- train.txt 600 /tmp/bpe.json probe.txt
python3 scripts/reference_bpe.py train.txt 600 /tmp/bpe.json probe.txt
# the six mutation proofs of this round, and the hostile-input harness:
python3 scripts/audit_mutations.py D-
python3 scripts/audit_fuzz.py --bin target/release/germen
# the comparison of §21 (the text is Karpathy's tinyshakespeare, "input.txt"):
germen train --data input.txt --out run --preset nano --seq 64 --batch 12 --steps 2000 \
             --lr 0.001 --eval-every 250 --eval-batches 20 --seed 1
```

Sources of the published figures: [nanoGPT README](https://github.com/karpathy/nanoGPT) (the CPU recipe with
loss ≈ 1.88 and the GPU baseline of 1.47 for the 10.7 M-parameter model).
