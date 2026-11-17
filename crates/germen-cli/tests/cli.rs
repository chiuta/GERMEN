//! End-to-end checks of the command-line tool: it is run as a separate process, as a user would
//! run it, and what matters is the exit code (1 = a reported error; 101 would be a Rust panic)
//! and the message.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn germen(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_germen"))
        .args(args)
        .output()
        .expect("the germen binary must start")
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("germen-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Asserts a reported error (exit code 1, never a panic) whose message contains `needle`.
fn assert_refused(out: &Output, needle: &str) {
    let err = text(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "stderr: {err}");
    assert!(err.contains("eroare:"), "stderr: {err}");
    assert!(err.contains(needle), "expected {needle:?} in: {err}");
    assert!(!err.contains("panicked"), "stderr: {err}");
}

fn train_args<'a>(out: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec![
        "train",
        "--demo",
        "--demo-mb",
        "0.1",
        "--preset",
        "nano",
        "--seq",
        "32",
        "--batch",
        "2",
        "--eval-every",
        "0",
        "--log-every",
        "100",
        "--out",
        out,
    ];
    args.extend_from_slice(extra);
    args
}

#[test]
fn nonsense_sizes_are_refused_not_attempted() {
    let dir = scratch("sizes");
    let out = dir.to_str().unwrap();
    // 1e300 MiB used to saturate to usize::MAX bytes and abort inside the allocator.
    for bad in ["1e300", "-3", "0", "NaN"] {
        let flag = format!("--demo-mb={bad}"); // `=` so that "-3" is a value, not an option
        let result = germen(&["train", "--demo", &flag, "--out", out]);
        assert_refused(&result, "--demo-mb");
    }
    let result = germen(&["demo-corpus", "--size-mb", "1e300", "--out", out]);
    assert_refused(&result, "--size-mb");
    assert!(!Path::new(out).exists(), "nothing may be written");
}

#[test]
fn a_nonsense_validation_share_is_a_message_not_a_panic() {
    let dir = scratch("valfrac");
    let out = dir.to_str().unwrap();
    // With --tokenizer bpe this used to panic in the text slicing before any check ran.
    for fraction in ["7", "-0.2", "0.5", "NaN"] {
        let flag = format!("--val-fraction={fraction}");
        let args = train_args(out, &["--tokenizer", "bpe", &flag]);
        assert_refused(&germen(&args), "--val-fraction");
    }
}

#[test]
fn a_run_without_validation_text_says_so() {
    // With --val-fraction 0 there is no unseen text, so the "validation" loss is measured on the
    // training text; 0.2.0 reported it as validation without a word.
    let run = |name: &str, fraction: &str| {
        let dir = scratch(name);
        let args = [
            "train",
            "--demo",
            "--demo-mb",
            "0.1",
            "--preset",
            "nano",
            "--seq",
            "32",
            "--batch",
            "2",
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
            "--log-every",
            "100",
            "--save-every",
            "0",
            fraction,
            "--out",
            dir.to_str().unwrap(),
        ];
        let out = germen(&args);
        let _ = std::fs::remove_dir_all(&dir);
        out
    };
    let without = run("noval", "--val-fraction=0");
    assert!(without.status.success(), "{}", text(&without.stderr));
    let warning = text(&without.stderr);
    assert!(
        warning.contains("avertisment") && warning.contains("nu există text de validare"),
        "{warning}"
    );
    let with = run("withval", "--val-fraction=0.05");
    assert!(with.status.success(), "{}", text(&with.stderr));
    assert!(!text(&with.stderr).contains("nu există text de validare"));
}

#[test]
fn an_absurd_thread_count_is_refused() {
    assert_refused(&germen(&["--threads", "1000000", "info"]), "fire de lucru");
    // Zero is an argument error, reported with the usual "usage" exit code.
    let zero = germen(&["--threads", "0", "info"]);
    assert_eq!(zero.status.code(), Some(2));
    assert!(text(&zero.stderr).contains("--threads"));
}

#[test]
fn resuming_past_or_at_the_requested_steps_is_explained() {
    let dir = scratch("resume");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "4", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));

    // Fewer steps than were done: an error that says what --steps means.
    let fewer = germen(&train_args(out, &["--steps", "2", "--resume"]));
    assert_refused(&fewer, "--steps");
    assert!(text(&fewer.stderr).contains("pasul 4"));

    // Exactly as many: success, and it says there was nothing to do.
    let same = germen(&train_args(out, &["--steps", "4", "--resume"]));
    assert!(same.status.success(), "{}", text(&same.stderr));
    assert!(
        text(&same.stdout).contains("Nimic de făcut"),
        "{}",
        text(&same.stdout)
    );

    // More: it trains the difference.
    let more = germen(&train_args(out, &["--steps", "6", "--resume"]));
    assert!(more.status.success(), "{}", text(&more.stderr));
    assert!(
        text(&more.stdout).contains("Gata: 2 pași"),
        "{}",
        text(&more.stdout)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_hostile_tokenizer_file_is_an_error_not_an_out_of_memory_kill() {
    let dir = scratch("hostile-tokenizer");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));

    // Sixty merges that each double the longest token.
    let merges: Vec<String> = std::iter::once("[97,97]".to_string())
        .chain((0..60).map(|k| format!("[{},{}]", 256 + k, 256 + k)))
        .collect();
    let json = format!(
        r#"{{"kind":"bpe","vocab_size":317,"merges":[{}]}}"#,
        merges.join(",")
    );
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();

    assert_refused(
        &germen(&["generate", "--model", out, "--max-new", "3"]),
        "octeți",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_model_with_nan_weights_is_refused() {
    // The weights file is replaced by one whose bytes are all 0xFF (NaN as f32 data): whatever
    // the container format says about it, generating from it must stop with an error.
    let dir = scratch("nan-weights");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));
    let weights = dir.join("weights.bpk");
    let len = std::fs::metadata(&weights).unwrap().len() as usize;
    std::fs::write(&weights, vec![0xFFu8; len]).unwrap();
    let result = germen(&["generate", "--model", out, "--max-new", "3"]);
    assert_eq!(result.status.code(), Some(1), "{}", text(&result.stderr));
    assert!(
        !text(&result.stderr).contains("panicked"),
        "{}",
        text(&result.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
