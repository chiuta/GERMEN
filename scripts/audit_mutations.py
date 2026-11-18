#!/usr/bin/env python3
"""Mutation proofs for docs/AUDIT.md (round 2, and the Studio of 0.5.0: entries T-xx).

For each entry one defect is put back (or one line of the fast SHA-256 is broken), the tests that
are meant to guard against it are run, and at least one of them must FAIL; the file is restored
afterwards and the build must work again.

    python3 scripts/audit_mutations.py            # all of them (about 10 minutes)
    python3 scripts/audit_mutations.py B-07 S-    # only the entries whose name starts so
    python3 scripts/audit_mutations.py T-         # only the Studio's (about 6 minutes)

The sources are edited in place and restored in a `finally`; do not run other builds meanwhile.
"""
import subprocess, shutil, sys, time, os, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
BACKUP = os.path.join(tempfile.mkdtemp(prefix="germen-mut-"), "backup")

def sh(cmd, timeout=900):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return "timeout", (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), time.time() - t0

M = []
def mut(name, path, old, new, cmd):
    M.append((name, path, old, new, cmd))

CORE = "cargo test --release --locked -p germen-core --lib "
CLI = "cargo test --release --locked -p germen-cli --test cli "

# --- BPE pre-token scan
mut("B-07a unbounded pre-token scan (the old code) -> linear-time tests must fail",
    "crates/germen-core/src/bpe.rs",
    "let reach = n.min(i + MAX_PRETOKEN + 1);", "let reach = n;",
    CORE + "a_long_run_without_blanks_is_cut_in_linear_time")
mut("B-07b unbounded pre-token scan -> end-to-end 2 MiB train+encode test must fail",
    "crates/germen-core/src/bpe.rs",
    "let reach = n.min(i + MAX_PRETOKEN + 1);", "let reach = n;",
    CORE + "text_without_blanks_trains_and_encodes_in_linear_time")
mut("B-07c off-by-one in the bound (reach = i + 64) -> the differential test must fail",
    "crates/germen-core/src/bpe.rs",
    "let reach = n.min(i + MAX_PRETOKEN + 1);", "let reach = n.min(i + MAX_PRETOKEN);",
    CORE + "bounding_the_scan_moves_no_boundary")
# --- config
mut("B-08 old norm_eps/rope_theta checks (finite and positive only)",
    "crates/germen-core/src/config.rs",
    "if !(1.0..=1e12).contains(&self.rope_theta) {", "if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {",
    CORE + "hostile_or_corrupt_configs_are_rejected")
mut("B-08b old norm_eps check",
    "crates/germen-core/src/config.rs",
    "if !(1e-12..=1.0).contains(&self.norm_eps) {", "if !(self.norm_eps.is_finite() && self.norm_eps > 0.0) {",
    CORE + "hostile_or_corrupt_configs_are_rejected")
# --- bench
mut("B-09a warmup + steps wraps (the old sum)",
    "crates/germen-core/src/bench.rs",
    "let total = warmup.checked_add(steps).ok_or_else(|| {\n        format!(\"warmup ({warmup}) + steps ({steps}) depășește ce se poate număra\")\n    })?;",
    "let total = warmup.wrapping_add(steps);",
    CORE + "counts_that_wrap_or_cannot_be_reserved_are_not_trusted")
mut("B-09b Vec::with_capacity(steps) (the old reservation)",
    "crates/germen-core/src/bench.rs",
    "Vec::with_capacity(steps.min(MAX_UPFRONT_STEPS))", "Vec::with_capacity(steps)",
    CORE + "counts_that_wrap_or_cannot_be_reserved_are_not_trusted")
# --- memory pre-check
mut("B-10a corpus size arithmetic wraps",
    "crates/germen-core/src/sys.rs",
    "text_bytes.saturating_mul(bytes_per_byte) as f64", "text_bytes.wrapping_mul(bytes_per_byte) as f64",
    CORE + "a_text_that_cannot_fit_is_refused_before_it_is_read")
mut("B-10b the CLI no longer checks the size of the text before reading it",
    "crates/germen-cli/src/main.rs",
    "sys::check_corpus_fits(info.len(), per_byte)?;", "let _ = per_byte;",
    CLI + "a_text_larger_than_memory_is_refused_before_it_is_read")
# --- checkpoint integrity
mut("B-01a load() no longer verifies weights.bpk against the manifest",
    "crates/germen-core/src/checkpoint.rs",
    "    verify(dir, &meta, &[WEIGHTS_FILE])?;", "    let _ = &meta;",
    CORE + "a_flipped_bit_in_a_file_is_reported_by_its_name")
mut("B-01b the tokenizer is no longer verified",
    "crates/germen-core/src/tokenizer.rs",
    "crate::checkpoint::verify_files(dir, &[TOKENIZER_FILE])?;", "let _ = dir;",
    CORE + "-- tokenizer:: checkpoint::")
mut("B-02 optimizer file with a nameless tensor is handed to Burn unchecked (panic)",
    "crates/germen-core/src/checkpoint.rs",
    "let stored = check_optimizer_file(&path, model)?;", "let stored = 0usize; let _ = model;",
    CORE + "an_optimizer_file_with_a_nameless_tensor_is_an_error_not_a_panic")
mut("B-03 foreign optimizer state is no longer refused",
    "crates/germen-core/src/checkpoint.rs",
    "let stored = check_optimizer_file(&path, model)?;", "let stored = 0usize; let _ = model;",
    CORE + "an_optimizer_that_does_not_belong_to_the_weights_is_refused")
mut("B-04 a save that stops half-way deletes its remaining files (no journal to complete)",
    "crates/germen-core/src/fsutil.rs",
    "                if i == 0 {\n                    for (rest, _) in &pending {",
    "                if true {\n                    for (rest, _) in &pending {",
    CORE + "-- fsutil:: checkpoint::")
mut("B-05 resume no longer needs meta.json",
    "crates/germen-core/src/checkpoint.rs",
    "pub fn load_for_resume(dir: &Path, device: &Device) -> Result<(Gpt, GptConfig, Meta), String> {",
    "pub fn load_for_resume(dir: &Path, device: &Device) -> Result<(Gpt, GptConfig, Meta), String> {\n    if !dir.join(META_FILE).exists() { let (m, c, _) = load(dir, device)?; return Ok((m, c, Meta::default())); }",
    CORE + "resume_needs_the_progress_record_but_generation_does_not")
mut("B-06 a fresh run replaces a trained checkpoint without asking",
    "crates/germen-core/src/train.rs",
    "if !resume && !opts.overwrite && checkpoint::exists(out_dir) {", "if false && !resume && !opts.overwrite && checkpoint::exists(out_dir) {",
    CORE + "a_fresh_run_does_not_silently_replace_a_trained_checkpoint")
mut("B-06b (CLI) the same, end to end",
    "crates/germen-core/src/train.rs",
    "if !resume && !opts.overwrite && checkpoint::exists(out_dir) {", "if false && !resume && !opts.overwrite && checkpoint::exists(out_dir) {",
    CLI + "a_fresh_run_does_not_silently_replace_a_checkpoint")


mut("B-04b a torn set no longer says how to put it right",
    "crates/germen-core/src/checkpoint.rs",
    ".map_err(|e| format!(\"{e}{}\", interrupted_save_hint(dir)))?;", ".map_err(|e| e)?;",
    CORE + "-- checkpoint::")
mut("B-04c recover_locked takes no claim on the directory",
    "crates/germen-core/src/checkpoint.rs",
    "    let _claim = DirLock::acquire(dir)?;\n    recover(dir)", "    recover(dir)",
    CORE + "-- checkpoint::")
mut("B-04d (CLI) verify --recover does nothing",
    "crates/germen-cli/src/main.rs",
    "println!(\"{}\", recovery_message(&checkpoint::recover_locked(&a.dir)?));", "let _ = &a.recover;",
    CLI + "a_save_cut_off_between_two_renames")

# --- SHA-256 on the processor's SHA extensions (x86-64 only: elsewhere these are not applicable)
INTEGRITY = "crates/germen-core/src/integrity.rs"
mut("S-01 SHA-NI: a round constant read from the wrong lane",
    INTEGRITY, "K[4 * $i + 3] as i32,", "K[4 * $i + 2] as i32,",
    CORE + "integrity::")
mut("S-02 SHA-NI: the message schedule is aligned by the wrong amount",
    INTEGRITY, "_mm_alignr_epi8::<4>($v3, $v2),", "_mm_alignr_epi8::<8>($v3, $v2),",
    CORE + "integrity::")
mut("S-03 SHA-NI: one word of the state is written back from the wrong lane",
    INTEGRITY, "state[5] = _mm_extract_epi32::<1>(hgef) as u32;", "state[5] = _mm_extract_epi32::<0>(hgef) as u32;",
    CORE + "integrity::")
mut("S-04 SHA-NI: the state is not added back after a block",
    INTEGRITY, "cdgh = _mm_add_epi32(cdgh, cdgh_before);", "let _ = cdgh_before;",
    CORE + "integrity::")
mut("S-05 SHA-NI: the bytes of a message word are not fully reversed",
    INTEGRITY, "0x0405_0607_0001_0203", "0x0405_0607_0001_0302",
    CORE + "integrity::")
mut("S-06 the fast backend is never chosen (the dispatch test must notice)",
    INTEGRITY, '&& std::arch::is_x86_feature_detected!("sse4.1")\n        {', '&& std::arch::is_x86_feature_detected!("sse4.1")\n            && false\n        {',
    CORE + "the_fast_backend_is_chosen")

# --- GERMEN Studio (0.5.0): who may talk to the server, and what it may make it do.
# Every gate is switched off in turn; the tests that attack that gate must notice.
STUDIO = "cargo test --release --locked -p germen-studio "
API = "crates/germen-studio/src/api.rs"
AUTH = "crates/germen-studio/src/auth.rs"
mut("T-01 the Host header is no longer checked (DNS rebinding gets in)",
    API, ".is_some_and(|host| auth::host_allowed(host, shared.port))", ".is_some_and(|host| !host.is_empty())",
    STUDIO + "--test http a_request_meant_for_another_host")
mut("T-02 the Origin header is no longer checked (another page's request gets in)",
    API, "&& !auth::origin_allowed(origin, shared.port)", "&& false",
    STUDIO + "--test http requests_that_another_page_made")
mut("T-03 Sec-Fetch-Site is no longer checked",
    API, "&& !auth::fetch_site_allowed(site)", "&& false",
    STUDIO + "--test http requests_that_another_page_made")
mut("T-04 the token is no longer required",
    API, "    if !allowed {", "    if false && !allowed {",
    STUDIO + "--test http the_api_needs_the_token")
mut("T-05 any token is accepted",
    API, ".is_some_and(|t| auth::constant_time_eq(t.as_bytes(), shared.token.as_bytes()));", ".is_some_and(|t| !t.is_empty());",
    STUDIO + "--test http the_api_needs_the_token")
mut("T-06 the comparison of tokens accepts a prefix of the real one",
    AUTH, "    let mut difference = a.len() ^ b.len();\n    for i in 0..a.len().max(b.len()) {", "    let mut difference = 0usize;\n    for i in 0..a.len().min(b.len()) {",
    STUDIO + "--lib auth::")
mut("T-07 the policy lets inline script run (its hash is replaced by 'unsafe-inline')",
    "crates/germen-studio/src/page.rs", "csp_hash(&script),", "\"'unsafe-inline'\",",
    STUDIO + "the_page_is_served_with_a_policy")
mut("T-08 the policy no longer asks for Trusted Types",
    "crates/germen-studio/src/page.rs", "object-src 'none'; require-trusted-types-for 'script'\"", "object-src 'none'\"",
    STUDIO + "page::")
mut("T-09 a model name may contain a slash (a way out of the workspace)",
    "crates/germen-studio/src/workspace.rs", "matches!(c, b'-' | b'_' | b'.'))", "matches!(c, b'-' | b'_' | b'.' | b'/'))",
    STUDIO + "names_from_the_network")
mut("T-10 an upload may be larger than the limit",
    API, "    if length > shared.max_upload {", "    if false && length > shared.max_upload {",
    STUDIO + "--test http a_text_can_be_uploaded_within_the_limit")
mut("T-11 an upload may replace a file that is there",
    "crates/germen-studio/src/workspace.rs", ".write(true).create_new(true).open(&path)", ".write(true).create(true).truncate(true).open(&path)",
    STUDIO + "--test http a_text_can_be_uploaded_within_the_limit")
mut("T-12 a second Host header is merged instead of refused (request smuggling)",
    "crates/germen-studio/src/http.rs", 'const SINGLE: [&str; 9] = [\n    "host",\n', 'const SINGLE: [&str; 9] = [\n    "x-host",\n',
    STUDIO + "malformed_requests")
mut("T-13 the number of connections is not limited",
    "crates/germen-studio/src/lib.rs", "const MAX_CONNECTIONS: usize = 64;", "const MAX_CONNECTIONS: usize = 100_000;",
    STUDIO + "--test http too_many_connections")
mut("T-14 a JSON body may be of any size",
    API, "pub(crate) const MAX_JSON_BODY: u64 = 1 << 20;", "pub(crate) const MAX_JSON_BODY: u64 = 1 << 40;",
    STUDIO + "--test http malformed_requests")
mut("T-15 the server listens on every interface, not on 127.0.0.1 alone",
    "crates/germen-studio/src/lib.rs", "TcpListener::bind((Ipv4Addr::LOCALHOST, p))", "TcpListener::bind((Ipv4Addr::UNSPECIFIED, p))",
    STUDIO + "--lib loopback_only")
mut("T-16 a model that is being trained can be checked (read half-written) from outside",
    API, "            if hub.dir_in_training(&dir) {\n                return Err(model_busy(&dir));\n            }\n            Ok(Response::ok(&work::verify_model(&dir, shared.threads)?))",
    "            Ok(Response::ok(&work::verify_model(&dir, shared.threads)?))",
    STUDIO + "--test http a_model_in_training_is_not_touched")
mut("T-17 a fresh training over a trained model is accepted",
    API, "Mode::New if checkpoint::exists(&dir) => {", "Mode::New if false && checkpoint::exists(&dir) => {",
    STUDIO + "--test http a_model_is_trained_from_scratch")
mut("T-18 the page's Close does not close the server",
    API, "            shared.request_shutdown();\n            Ok(Response::ok(&json!({ \"ok\": true })))", "            Ok(Response::ok(&json!({ \"ok\": true })))",
    STUDIO + "--test http the_page_can_close_the_server")

only = sys.argv[1:]
summary = []
for name, path, old, new, cmd in M:
    if only and not any(name.startswith(o) for o in only):
        continue
    src = open(path, encoding="utf-8").read()
    if src.count(old) != 1:
        print(f"!! {name}: the text to mutate occurs {src.count(old)} times in {path}", flush=True)
        summary.append((name, "NOT APPLICABLE"))
        continue
    shutil.copy(path, BACKUP)
    try:
        open(path, "w", encoding="utf-8").write(src.replace(old, new))
        rc, out, dt = sh(cmd, timeout=900)
        failed = [l for l in out.splitlines() if l.startswith("test ") and l.endswith("FAILED")]
        compiled = "error[" not in out and "could not compile" not in out
        verdict = "CAUGHT" if (rc != 0 and failed and compiled) else ("COMPILE ERROR" if not compiled else ("TIMEOUT" if rc == "timeout" else "NOT CAUGHT"))
        print(f"{verdict:14} {name}  ({dt:.0f}s)  {'; '.join(l[5:-7] for l in failed)[:200]}", flush=True)
        if not compiled:
            print(out[-1500:], flush=True)
        summary.append((name, verdict))
    finally:
        shutil.copy(BACKUP, path)
# restored sources must pass again
rc, out, dt = sh("cargo build --release --locked 2>&1 | tail -2", timeout=600)
print("restored build:", out.strip().splitlines()[-1] if out.strip() else rc, flush=True)
print("SUMMARY", summary, flush=True)
