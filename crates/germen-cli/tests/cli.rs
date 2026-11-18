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

/// Rewrites `meta.json` without its manifest, which is what a checkpoint of 0.3.0 or before
/// looks like.
fn strip_manifest(dir: &Path) {
    let path = dir.join("meta.json");
    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    meta.as_object_mut()
        .unwrap()
        .remove("files")
        .expect("a checkpoint of this version has a manifest");
    std::fs::write(&path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();
}

/// Copies the files of a checkpoint directory (not its lock file).
fn copy_checkpoint(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        if entry.file_name() != ".germen.lock" && entry.path().is_file() {
            std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
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

    // The manifest names the edited file before anything in it is read…
    let caught = germen(&["generate", "--model", out, "--max-new", "3"]);
    assert_refused(&caught, "tokenizer.json");
    assert!(
        text(&caught.stderr).contains("meta.json"),
        "{}",
        text(&caught.stderr)
    );

    // …and without one (a checkpoint of 0.3.0 or before) the tokenizer's own checks stand guard.
    strip_manifest(&dir);
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

#[test]
fn verify_vouches_for_a_checkpoint_and_names_the_file_that_is_wrong() {
    let dir = scratch("verify");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));

    let good = germen(&["verify", out]);
    assert!(good.status.success(), "{}", text(&good.stderr));
    let report = text(&good.stdout);
    for name in [
        "weights.bpk",
        "config.json",
        "tokenizer.json",
        "optimizator",
    ] {
        assert!(report.contains(name), "{name} is missing from: {report}");
    }
    assert!(report.contains("SHA-256 corect"), "{report}");

    // One flipped bit in the weights, and the report says which file, instead of a failure
    // somewhere deep in the loader (or, worse, a model that quietly generates something else).
    let weights = dir.join("weights.bpk");
    let mut bytes = std::fs::read(&weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&weights, bytes).unwrap();
    assert_refused(&germen(&["verify", out]), "weights.bpk");
    assert_refused(
        &germen(&["generate", "--model", out, "--max-new", "3"]),
        "weights.bpk",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_checkpoint_of_the_old_format_still_loads_and_says_it_cannot_be_vouched_for() {
    let dir = scratch("legacy");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));
    strip_manifest(&dir);

    let result = germen(&["generate", "--model", out, "--max-new", "3"]);
    assert!(result.status.success(), "{}", text(&result.stderr));
    assert!(
        text(&result.stderr).contains("nu poate fi verificat"),
        "{}",
        text(&result.stderr)
    );
    // `verify` has nothing to check it against, and says so with an error rather than a "fine".
    assert_refused(&germen(&["verify", out]), "meta.json");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fresh_run_does_not_silently_replace_a_checkpoint() {
    let dir = scratch("overwrite");
    let out = dir.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));
    let weights = dir.join("weights.bpk");
    let before = std::fs::read(&weights).unwrap();

    // The same command again used to replace the trained model at its first save, without a word.
    let again = germen(&train_args(
        out,
        &["--steps", "1", "--save-every", "0", "--seed", "99"],
    ));
    assert_refused(&again, "--overwrite");
    assert_eq!(
        std::fs::read(&weights).unwrap(),
        before,
        "not a byte changed"
    );

    // Asked for in so many words, it does.
    let forced = germen(&train_args(
        out,
        &[
            "--steps",
            "1",
            "--save-every",
            "0",
            "--seed",
            "99",
            "--overwrite",
        ],
    ));
    assert!(forced.status.success(), "{}", text(&forced.stderr));
    assert_ne!(std::fs::read(&weights).unwrap(), before);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_text_larger_than_memory_is_refused_before_it_is_read() {
    // A sparse file twice as large as the memory the system has free: nothing is written to disk
    // to make it, and reading it could only end in an out-of-memory error or a killed process.
    let Some(available) = germen_core::sys::available_memory_bytes() else {
        return;
    };
    let dir = scratch("huge-text");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("huge.txt");
    let created = std::fs::File::create(&file)
        .and_then(|f| f.set_len(available.saturating_mul(2).max(1 << 30)));
    if created.is_err() {
        // A file system that cannot hold a file that large (even sparse): nothing to test here.
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    let model = dir.join("model");
    let started = std::time::Instant::now();
    let out = germen(&[
        "train",
        "--data",
        file.to_str().unwrap(),
        "--steps",
        "1",
        "--out",
        model.to_str().unwrap(),
    ]);
    assert_refused(&out, "limita acceptată");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "the refusal must come before the file is read, not after"
    );
    assert!(!model.exists(), "a refused run leaves nothing behind");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_save_cut_off_between_two_renames_is_refused_with_a_remedy_and_completed_by_verify_recover() {
    // What a power cut leaves in the middle of publishing a save: the files of the new save are
    // either already in place (tokenizer, optimizer, weights) or waiting under their temporary
    // names (configuration, progress record), while the progress record that vouches for the
    // files is still the old one.
    let old = scratch("cut-old");
    let newer = scratch("cut-new");
    let torn = scratch("cut-torn");
    let out = old.to_str().unwrap();
    let first = germen(&train_args(out, &["--steps", "1", "--save-every", "0"]));
    assert!(first.status.success(), "{}", text(&first.stderr));
    copy_checkpoint(&old, &newer);
    let second = germen(&train_args(
        newer.to_str().unwrap(),
        &["--steps", "2", "--save-every", "0", "--resume"],
    ));
    assert!(second.status.success(), "{}", text(&second.stderr));

    copy_checkpoint(&old, &torn);
    for name in ["tokenizer.json", "optimizer.bpk", "weights.bpk"] {
        std::fs::copy(newer.join(name), torn.join(name)).unwrap();
    }
    for name in ["config.json", "meta.json"] {
        std::fs::copy(newer.join(name), torn.join(format!("{name}.tmp"))).unwrap();
    }
    let torn_dir = torn.to_str().unwrap();

    // Both commands refuse the mixture, and both say how to put it right.
    assert_refused(
        &germen(&["generate", "--model", torn_dir, "--max-new", "3"]),
        "verify --recover",
    );
    assert_refused(&germen(&["verify", torn_dir]), "verify --recover");

    // The remedy needs no text to train on, and leaves the set the new save would have made.
    let fixed = germen(&["verify", torn_dir, "--recover"]);
    assert!(fixed.status.success(), "{}", text(&fixed.stderr));
    assert!(
        text(&fixed.stdout).contains("am terminat-o"),
        "{}",
        text(&fixed.stdout)
    );
    for name in [
        "tokenizer.json",
        "optimizer.bpk",
        "weights.bpk",
        "config.json",
        "meta.json",
    ] {
        assert_eq!(
            std::fs::read(torn.join(name)).unwrap(),
            std::fs::read(newer.join(name)).unwrap(),
            "{name}"
        );
    }
    let generated = germen(&["generate", "--model", torn_dir, "--max-new", "3"]);
    assert!(generated.status.success(), "{}", text(&generated.stderr));
    let again = germen(&["verify", torn_dir, "--recover"]);
    assert!(again.status.success());
    assert!(
        text(&again.stdout).contains("Nicio salvare întreruptă"),
        "{}",
        text(&again.stdout)
    );
    for dir in [&old, &newer, &torn] {
        let _ = std::fs::remove_dir_all(dir);
    }
}

// ---- `germen ui`: the graphical interface, started and closed as a user would ----

/// One answer of the Studio, read to the end of the connection (it closes after every answer).
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// One HTTP/1.1 request over a plain socket: `host` is what the `Host` header says, `token` the
/// bearer token (none: no `Authorization` header).
fn ask(port: u16, method: &str, path: &str, host: &str, token: Option<&str>) -> Reply {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).expect("the Studio listens");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n");
    if let Some(token) = token {
        request.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if method == "POST" {
        request.push_str("Content-Length: 0\r\n");
    }
    request.push_str("\r\n");
    stream.write_all(request.as_bytes()).unwrap();
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .expect("the Studio answers and closes the connection");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a complete head");
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split(' ').nth(1))
        .and_then(|code| code.parse().ok())
        .expect("a status line");
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    Reply {
        status,
        headers,
        body: raw[split + 4..].to_vec(),
    }
}

/// Kills the child when the test ends any way at all, so that a failed assertion cannot leave a
/// server running.
struct Running(std::process::Child);

impl Drop for Running {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_ui_prints_its_secret_address_answers_only_to_those_who_have_it_and_closes_when_asked() {
    use std::io::{BufRead, BufReader, Read};
    use std::process::Stdio;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    let workspace = scratch("ui");
    let child = Command::new(env!("CARGO_BIN_EXE_germen"))
        .args(["ui", "--no-open", "--port", "0", "--workspace"])
        .arg(&workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the germen binary must start");
    let mut child = Running(child);

    // What it prints: the address with the port and the token after the `#`.
    let stdout = child.0.stdout.take().unwrap();
    let (lines_tx, lines) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines_tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut printed = Vec::new();
    let url = loop {
        let line = lines
            .recv_timeout(Duration::from_secs(60))
            .unwrap_or_else(|_| panic!("no address in time; printed so far: {printed:?}"));
        printed.push(line.clone());
        if let Some(at) = line.find("http://127.0.0.1:") {
            break line[at..].trim().to_string();
        }
    };
    let rest = url.strip_prefix("http://127.0.0.1:").unwrap();
    let (port, token) = rest.split_once("/#k=").expect("the token is after the #");
    let port: u16 = port.parse().expect("a port");
    assert_ne!(
        port, 0,
        "port 0 asks for any free port; the address names the real one"
    );
    assert_eq!(token.len(), 64, "{url}");
    assert!(token.bytes().all(|b| b.is_ascii_hexdigit()), "{url}");
    let host = format!("127.0.0.1:{port}");

    // The page: served to whoever asks, with a policy that names its one script and its one style.
    let page = ask(port, "GET", "/", &host, None);
    assert_eq!(page.status, 200);
    let policy = page.header("content-security-policy").expect("a policy");
    assert!(policy.contains("default-src 'none'"), "{policy}");
    assert!(policy.contains("'sha256-"), "{policy}");
    let html = text(&page.body);
    assert!(html.contains("<meta name=\"germen-mode\" content=\"app\">"));
    assert!(
        !html.contains("function installDemo"),
        "the demonstration is not in the program"
    );
    assert!(
        !html.contains(token),
        "the token is never written into the page"
    );

    // The interface: nothing without the token, nothing for another host, and the real answer.
    assert_eq!(ask(port, "GET", "/api/ping", &host, None).status, 401);
    let wrong = "0".repeat(64);
    assert_eq!(
        ask(port, "GET", "/api/ping", &host, Some(&wrong)).status,
        401
    );
    assert_eq!(
        ask(port, "GET", "/api/ping", "evil.example", Some(token)).status,
        421,
        "a name that someone made point to 127.0.0.1 is not this server"
    );
    let ping = ask(port, "GET", "/api/ping", &host, Some(token));
    assert_eq!(ping.status, 200);
    assert_eq!(text(&ping.body).replace(' ', ""), "{\"ok\":true}");
    let system = ask(port, "GET", "/api/system", &host, Some(token));
    assert_eq!(system.status, 200);
    let system: serde_json::Value = serde_json::from_slice(&system.body).unwrap();
    assert_eq!(system["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(system["presets"].as_array().map(Vec::len), Some(5));
    assert!(workspace.join("models").is_dir() && workspace.join("data").is_dir());

    // The window's "Close": the process ends by itself, with success.
    assert_eq!(
        ask(port, "POST", "/api/quit", &host, Some(token)).status,
        200
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the Studio did not close");
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    let _ = child.0.stderr.take().unwrap().read_to_string(&mut stderr);
    assert!(status.success(), "{status}; stderr: {stderr}");
    printed.extend(lines.iter());
    assert!(
        printed.iter().any(|line| line.contains("s-a închis")),
        "{printed:?}"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
fn the_ui_refuses_what_it_cannot_serve_before_it_starts_anything() {
    let workspace = scratch("ui-refused");
    let dir = workspace.to_string_lossy().into_owned();
    for bad in ["0", "1000001", "99999999999999999999999"] {
        let out = germen(&[
            "ui",
            "--no-open",
            "--workspace",
            &dir,
            "--max-upload-mb",
            bad,
        ]);
        assert!(!out.status.success(), "--max-upload-mb {bad}");
        assert!(
            !text(&out.stderr).contains("panicked"),
            "{}",
            text(&out.stderr)
        );
    }
    let out = germen(&[
        "ui",
        "--no-open",
        "--workspace",
        &dir,
        "--max-upload-mb",
        "0",
    ]);
    assert_refused(&out, "--max-upload-mb");
    assert!(!workspace.exists(), "a refused start creates nothing");

    // The one-file demonstration page: written, not served.
    let page = workspace.with_extension("html");
    let out = germen(&["ui", "--write-demo", page.to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let html = std::fs::read_to_string(&page).unwrap();
    assert!(html.contains("<meta name=\"germen-mode\" content=\"demo\">"));
    assert!(html.contains("function installDemo"));
    assert!(!workspace.exists());
    let _ = std::fs::remove_file(&page);
}
