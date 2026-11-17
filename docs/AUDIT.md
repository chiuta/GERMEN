# Adversarial audit of GERMEN 0.2 — findings, fixes and what remains

Written for version 0.3.0 (17 November 2026). The audit was done by the people who wrote the code
(the author and an AI assistant), so "adversarial" means *tried hard to break it*, not an
independent review: [PLAN.md](PLAN.md) schedules one by someone else before 1.0.

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
| A-20 | Low | no scan of the ≈ 780 locked crates for known vulnerabilities | I | weekly `audit.yml` |
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
* **The lock is advisory.** A program that ignores it (an older GERMEN, for one) is not stopped, and
  where a file system cannot lock (some network shares) the claim is skipped rather than refused.
* **Limits refuse by size, not by proof.** There is no fuzzing; the hostile-input tests are
  hand-written, and a corrupt `weights.bpk` is only as safe as Burn's burnpack reader (six kinds of
  damage were tried).
* **The ceilings** of A-02 could in principle refuse a legitimate gigantic model (a context over
  16,384, a width over 16,384, more than 2^32 parameters); they are named constants in `config.rs`.
* **The corpus is read whole.** A text file larger than the available memory is not refused before
  it is read (planned: memory-mapped corpora, 0.4).
* **Reproducibility** holds for a given seed, thread count and replica count, not across machines
  with different core counts (the order of floating-point sums differs), and not between 0.2 and 0.3.
  It also assumes that nothing outside GERMEN draws from the same Burn device while a model is being
  initialised (A-24).

## 5. What has not been verified

* Windows and macOS: the `GlobalMemoryStatusEx` call, the `vm_stat` parser on a real Mac, the file
  lock, and the rename retry under a real antivirus. CI compiles and tests on all three systems
  and will show failures; it cannot show what an antivirus does.
* The build with Rust 1.95: the sandbox has 1.97, so the new `msrv` job is the first test of the
  promise in `Cargo.toml`.
* `audit.yml` has not run yet; the third-party GitHub Actions are pinned by tag, not by commit hash.
* A bigger machine (§3.3), and a release build on any machine but this one.

## 6. Reproducing this audit

```sh
cargo build --release
# every finding that has a command-line trigger, against the new build and an old one:
python3 scripts/audit_repro.py --new target/release/germen --old /path/to/germen-0.2.0
python3 scripts/audit_repro.py --new target/release/germen A-06 A-07     # only some
# library-level findings (copy the file into the examples folder of an old tree to compare):
cargo run --release -p germen-core --example audit_lib
# speed: interleaved A/B of the real command
target/release/germen bench --threads 2 --steps 8 --replicas 1 --json a.json
target/release/germen bench --threads 2 --steps 8 --json b.json
```

`audit_repro.py` needs a POSIX system (process groups, resource limits and signals); on Windows run
it in WSL. It exits 1 if the new binary fails a scenario that is not purely informational.
