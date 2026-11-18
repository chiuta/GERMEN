//! The Studio, from the outside: a real server on a real port, spoken to over raw TCP, the way a
//! browser, a script or an attacker would. Nothing here calls the crate's functions except to
//! start and stop it; every check goes through bytes on a socket.

use germen_studio::{Options, ShutdownHandle, Studio};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

static COUNTER: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------------------------
// A server to talk to
// ---------------------------------------------------------------------------------------------

struct Harness {
    port: u16,
    token: String,
    workspace: PathBuf,
    handle: ShutdownHandle,
    thread: Option<JoinHandle<()>>,
}

impl Harness {
    fn start() -> Harness {
        Harness::with(|_| {})
    }

    fn with(tweak: impl FnOnce(&mut Options)) -> Harness {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let workspace =
            std::env::temp_dir().join(format!("germen-studio-it-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&workspace);
        let mut options = Options {
            workspace: workspace.clone(),
            port: Some(0),
            threads: Some(2),
            max_upload: 1 << 20,
        };
        tweak(&mut options);
        let studio = Studio::start(options).expect("the Studio starts");
        let port = studio.port();
        let token = studio.token().to_string();
        let handle = studio.shutdown_handle();
        let thread = std::thread::spawn(move || studio.serve());
        Harness {
            port,
            token,
            workspace,
            handle,
            thread: Some(thread),
        }
    }

    /// Asks the server to end and waits (at most 30 s) for it to do so.
    fn stop(&mut self) {
        self.handle.request();
        if let Some(thread) = self.thread.take() {
            let started = Instant::now();
            while !thread.is_finished() {
                assert!(
                    started.elapsed() < Duration::from_secs(30),
                    "the server did not end"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            thread.join().expect("the server ends cleanly");
        }
    }

    fn host(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// Sends `bytes` as they are and reads the answer to the end.
    fn raw(&self, bytes: &[u8]) -> Reply {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port)).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .unwrap();
        stream.write_all(bytes).expect("send");
        read_reply(&mut stream)
    }

    /// A request with the right `Host`, the token (when `auth`) and the extra headers.
    fn request(
        &self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
        auth: bool,
    ) -> Reply {
        let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n", self.host());
        if auth {
            head.push_str(&format!("Authorization: Bearer {}\r\n", self.token));
        }
        for (name, value) in headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        if let Some(body) = body {
            head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        head.push_str("\r\n");
        let mut bytes = head.into_bytes();
        if let Some(body) = body {
            bytes.extend_from_slice(body);
        }
        self.raw(&bytes)
    }

    fn get(&self, path: &str) -> Reply {
        self.request("GET", path, &[], None, true)
    }

    fn send_json(&self, method: &str, path: &str, value: &Value) -> Reply {
        let body = serde_json::to_vec(value).unwrap();
        self.request(
            method,
            path,
            &[("Content-Type", "application/json")],
            Some(&body),
            true,
        )
    }

    fn post(&self, path: &str, value: &Value) -> Reply {
        self.send_json("POST", path, value)
    }

    /// Starts a job and returns its number.
    fn start_job(&self, path: &str, value: &Value) -> u64 {
        let reply = self.post(path, value);
        assert_eq!(reply.status, 202, "{path}: {}", reply.text());
        reply.json()["job"].as_u64().expect("a job number")
    }

    /// Follows a job through its events until it ends; returns every event and the final snapshot.
    fn follow(&self, job: u64, within: Duration) -> (Vec<Value>, Value) {
        let started = Instant::now();
        let mut events = Vec::new();
        let mut next = 0u64;
        loop {
            let reply = self.get(&format!("/api/jobs/{job}/events?after={next}&wait=400"));
            assert_eq!(reply.status, 200, "{}", reply.text());
            let body = reply.json();
            assert_eq!(body["gap"], false, "no events were lost");
            for event in body["events"].as_array().unwrap() {
                assert_eq!(event["seq"].as_u64().unwrap(), next, "events come in order");
                next += 1;
                events.push(event["e"].clone());
            }
            if matches!(
                body["state"].as_str().unwrap(),
                "done" | "stopped" | "failed"
            ) {
                break;
            }
            assert!(
                started.elapsed() < within,
                "job {job} did not end within {within:?}; last events: {:?}",
                &events[events.len().saturating_sub(3)..]
            );
        }
        let snapshot = self.get(&format!("/api/jobs/{job}")).json();
        (events, snapshot)
    }

    /// Waits until `condition` holds for the snapshot of the job.
    fn wait_for(
        &self,
        job: u64,
        within: Duration,
        what: &str,
        condition: impl Fn(&Value) -> bool,
    ) -> Value {
        let started = Instant::now();
        loop {
            let snapshot = self.get(&format!("/api/jobs/{job}")).json();
            if condition(&snapshot) {
                return snapshot;
            }
            assert!(
                !matches!(
                    snapshot["state"].as_str().unwrap(),
                    "done" | "stopped" | "failed"
                ),
                "the job ended before {what}: {snapshot}"
            );
            assert!(started.elapsed() < within, "no {what} within {within:?}");
            std::thread::sleep(Duration::from_millis(30));
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.handle.request();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = std::fs::remove_dir_all(&self.workspace);
    }
}

struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Reply {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body)
            .unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.text()))
    }

    fn error_code(&self) -> String {
        self.json()["error"]["code"]
            .as_str()
            .unwrap_or_else(|| panic!("no error code in {}", self.text()))
            .to_string()
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_reply(stream: &mut TcpStream) -> Reply {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => raw.extend_from_slice(&chunk[..n]),
            // A reset after the answer was written is how some refusals end.
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset && !raw.is_empty() => break,
            Err(e) => panic!("reading the answer: {e} (got {} bytes)", raw.len()),
        }
    }
    parse_reply(&raw)
}

fn parse_reply(raw: &[u8]) -> Reply {
    let mut rest = raw;
    loop {
        let end = find(rest, b"\r\n\r\n")
            .unwrap_or_else(|| panic!("no complete head in {:?}", String::from_utf8_lossy(rest)));
        let head = std::str::from_utf8(&rest[..end]).expect("an ASCII head");
        let mut lines = head.split("\r\n");
        let status_line = lines.next().unwrap();
        let status: u16 = status_line.split(' ').nth(1).unwrap().parse().unwrap();
        let body = &rest[end + 4..];
        if status == 100 {
            rest = body;
            continue;
        }
        let headers = lines
            .map(|l| {
                let (n, v) = l.split_once(':').expect("a header line");
                (n.to_string(), v.trim().to_string())
            })
            .collect();
        return Reply {
            status,
            headers,
            body: body.to_vec(),
        };
    }
}

/// A small, fast training run: nano model, 30 KB of demo text, a few steps.
fn tiny_train(name: &str, steps: usize) -> Value {
    json!({
        "name": name,
        "data": { "kind": "demo", "mb": 0.03, "seed": 3 },
        "preset": "nano",
        "seq": 32,
        "batch": 4,
        "steps": steps,
        "warmup": 2,
        "lr": 0.003,
        "eval_every": 10,
        "eval_batches": 2,
        "save_every": 10,
        "sample_every": 10,
        "log_every": 1,
        "seed": 5,
    })
}

const LONG: Duration = Duration::from_secs(240);

// ---------------------------------------------------------------------------------------------
// The page and the gates in front of the API
// ---------------------------------------------------------------------------------------------

#[test]
fn the_page_is_served_with_a_policy_that_names_its_own_script() {
    let h = Harness::start();
    let reply = h.request("GET", "/", &[], None, false);
    assert_eq!(
        reply.status, 200,
        "the page itself needs no token (it carries none)"
    );
    assert!(
        reply
            .header("content-type")
            .unwrap()
            .starts_with("text/html")
    );
    let csp = reply.header("content-security-policy").expect("a policy");
    for needle in [
        "default-src 'none'",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "require-trusted-types-for 'script'",
    ] {
        assert!(csp.contains(needle), "{needle} in {csp}");
    }
    assert!(!csp.contains("unsafe"), "{csp}");
    assert_eq!(reply.header("x-frame-options"), Some("DENY"));
    assert_eq!(reply.header("x-content-type-options"), Some("nosniff"));
    assert_eq!(reply.header("cache-control"), Some("no-store"));
    assert_eq!(reply.header("referrer-policy"), Some("no-referrer"));
    assert!(reply.header("cross-origin-resource-policy").is_some());
    let page = reply.text();
    assert!(page.contains("<script>") && page.contains("</script>"));
    assert!(
        !page.contains(&h.token),
        "the token is never in the page: it is in the address's #"
    );
    // The favicon is answered without a body so that the browser does not ask in vain.
    assert_eq!(
        h.request("GET", "/favicon.ico", &[], None, false).status,
        204
    );
    // Nothing else exists outside /api.
    for path in [
        "/index.html",
        "/static/app.js",
        "/..%2f..%2fetc%2fpasswd",
        "/.git/config",
        "//",
    ] {
        let r = h.request("GET", path, &[], None, false);
        assert_eq!(r.status, 404, "{path}");
    }
    assert_eq!(h.request("POST", "/", &[], Some(b"x"), false).status, 405);
}

#[test]
fn the_api_needs_the_token_and_nothing_less_will_do() {
    let h = Harness::start();
    let ok = h.get("/api/ping");
    assert_eq!(ok.status, 200);
    assert_eq!(ok.json(), json!({ "ok": true }));

    // A different last character, whatever the random token ends with (a fixed "0" would equal the
    // token one time in sixteen and make this test fail by chance).
    let last = if h.token.ends_with('0') { '1' } else { '0' };
    let wrong = format!("{}{last}", &h.token[..h.token.len() - 1]);
    let upper = h.token.to_uppercase();
    let attempts: Vec<(String, String)> = vec![
        ("no header at all".into(), String::new()),
        ("empty token".into(), "Bearer ".into()),
        ("not bearer".into(), format!("Basic {}", h.token)),
        ("token alone".into(), h.token.clone()),
        ("a wrong last character".into(), format!("Bearer {wrong}")),
        (
            "a prefix of the token".into(),
            format!("Bearer {}", &h.token[..32]),
        ),
        ("the token and more".into(), format!("Bearer {}00", h.token)),
        ("upper case".into(), format!("Bearer {upper}")),
        ("padded".into(), format!("Bearer  {}", h.token)),
    ];
    for (what, header) in attempts {
        let headers: Vec<(&str, &str)> = if header.is_empty() {
            vec![]
        } else {
            vec![("Authorization", header.as_str())]
        };
        let r = h.request("GET", "/api/ping", &headers, None, false);
        assert_eq!(r.status, 401, "{what}");
        assert_eq!(r.error_code(), "unauthorized", "{what}");
    }
    // The token in the query string or in a cookie is not a way in.
    let q = h.request("GET", &format!("/api/ping?k={}", h.token), &[], None, false);
    assert_eq!(q.status, 401);
    let c = h.request(
        "GET",
        "/api/ping",
        &[("Cookie", &format!("k={}", h.token))],
        None,
        false,
    );
    assert_eq!(c.status, 401);
    // Every route is behind the gate, not only the harmless ones.
    for (method, path) in [
        ("GET", "/api/system"),
        ("GET", "/api/settings"),
        ("PUT", "/api/settings"),
        ("GET", "/api/models"),
        ("GET", "/api/fs?path=/"),
        ("GET", "/api/jobs"),
        ("GET", "/api/jobs/1"),
        ("GET", "/api/jobs/1/events"),
        ("POST", "/api/jobs/1/stop"),
        ("POST", "/api/train"),
        ("POST", "/api/generate"),
        ("POST", "/api/bench"),
        ("POST", "/api/verify"),
        ("POST", "/api/recover"),
        ("POST", "/api/upload?name=x.txt"),
        ("POST", "/api/quit"),
        ("GET", "/api/estimate"),
        ("GET", "/api/nothing-here"),
    ] {
        let body: Option<&[u8]> = if method == "GET" { None } else { Some(b"{}") };
        let r = h.request(method, path, &[], body, false);
        assert_eq!(r.status, 401, "{method} {path} without a token");
    }
    // And the server is still there, not stopped by any of that.
    assert_eq!(h.get("/api/ping").status, 200);
}

#[test]
fn a_request_meant_for_another_host_is_refused_even_with_the_token() {
    let h = Harness::start();
    let port = h.port;
    let t = &h.token;
    let other_port = if port == 65000 { 65001 } else { port + 1 };
    for host in [
        "evil.example".to_string(),
        format!("evil.example:{port}"),
        format!("127.0.0.1:{other_port}"),
        "127.0.0.1".to_string(),
        format!("127.0.0.1.evil.example:{port}"),
        format!("[::1]:{port}"),
        format!("0.0.0.0:{port}"),
        format!("localhost.evil.example:{port}"),
        format!("127.0.0.1:{port}x"),
        String::new(),
    ] {
        for (path, auth) in [("/", false), ("/api/ping", true)] {
            let mut head = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n");
            if auth {
                head.push_str(&format!("Authorization: Bearer {t}\r\n"));
            }
            head.push_str("\r\n");
            let r = h.raw(head.as_bytes());
            assert!(
                r.status == 421 || r.status == 400,
                "Host {host:?} for {path}: {}",
                r.status
            );
            assert!(!r.text().contains("\"ok\":true"), "Host {host:?}");
        }
    }
    // The names of this very server are fine.
    for host in [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        format!("LOCALHOST:{port}"),
    ] {
        let r = h.raw(
            format!("GET /api/ping HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {t}\r\n\r\n")
                .as_bytes(),
        );
        assert_eq!(r.status, 200, "{host}");
    }
}

#[test]
fn requests_that_another_page_made_are_refused_even_with_the_token() {
    let h = Harness::start();
    let port = h.port;
    // What a page of another site sends: its own origin, and a browser's "cross-site" marker.
    for (name, value) in [
        ("Origin", "http://evil.example".to_string()),
        ("Origin", "https://127.0.0.1".to_string()),
        ("Origin", format!("http://127.0.0.1:{}", port + 1)),
        ("Origin", format!("https://127.0.0.1:{port}")),
        ("Origin", "null".to_string()),
        ("Origin", format!("http://127.0.0.1:{port}.evil.example")),
        ("Origin", String::new()),
        ("Sec-Fetch-Site", "cross-site".to_string()),
        ("Sec-Fetch-Site", "same-site".to_string()),
        ("Sec-Fetch-Site", "none".to_string()),
        ("Sec-Fetch-Site", String::new()),
    ] {
        let r = h.request("GET", "/api/ping", &[(name, value.as_str())], None, true);
        assert_eq!(r.status, 403, "{name}: {value:?}");
        assert_eq!(r.error_code(), "bad_origin", "{name}: {value:?}");
        let r = h.request(
            "POST",
            "/api/quit",
            &[(name, value.as_str())],
            Some(b"{}"),
            true,
        );
        assert_eq!(r.status, 403, "quit with {name}: {value:?}");
    }
    // The page itself, as a browser sends it.
    let origin = format!("http://127.0.0.1:{port}");
    let same = [
        ("Origin", origin.as_str()),
        ("Sec-Fetch-Site", "same-origin"),
    ];
    assert_eq!(h.request("GET", "/api/ping", &same, None, true).status, 200);
    let localhost = format!("http://localhost:{port}");
    assert_eq!(
        h.request(
            "GET",
            "/api/ping",
            &[("Origin", localhost.as_str())],
            None,
            true
        )
        .status,
        200
    );
    // A script of another page cannot even read the answer: no CORS header is ever sent.
    let r = h.request("GET", "/api/ping", &same, None, true);
    assert!(r.header("access-control-allow-origin").is_none());
    // A pre-flight does not get a "yes".
    let pre = h.request(
        "OPTIONS",
        "/api/train",
        &[
            ("Origin", "http://evil.example"),
            ("Access-Control-Request-Method", "POST"),
        ],
        None,
        false,
    );
    assert_eq!(pre.status, 405);
    assert!(pre.header("access-control-allow-origin").is_none());
    // A form post from another site (a "simple" request, no pre-flight) is stopped by the gates too.
    let form = h.request(
        "POST",
        "/api/quit",
        &[
            ("Origin", "http://evil.example"),
            ("Content-Type", "text/plain"),
        ],
        Some(b"{}"),
        false,
    );
    assert!(form.status == 401 || form.status == 403);
    assert_eq!(h.get("/api/ping").status, 200, "the server is still up");
}

// ---------------------------------------------------------------------------------------------
// Hostile and broken requests
// ---------------------------------------------------------------------------------------------

#[test]
fn malformed_requests_are_refused_in_the_right_words_and_the_server_goes_on() {
    let h = Harness::start();
    let host = h.host();
    let t = h.token.clone();
    let good_headers = format!("Host: {host}\r\nAuthorization: Bearer {t}\r\n");
    let cases: Vec<(&str, Vec<u8>, u16)> = vec![
        ("no Host", b"GET /api/ping HTTP/1.1\r\n\r\n".to_vec(), 400),
        (
            "HTTP/2",
            format!("GET /api/ping HTTP/2\r\n{good_headers}\r\n").into_bytes(),
            505,
        ),
        (
            "an unknown method",
            format!("BREW /api/ping HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            405,
        ),
        (
            "DELETE",
            format!("DELETE /api/jobs/1 HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            405,
        ),
        (
            "TRACE",
            format!("TRACE / HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            405,
        ),
        (
            "an absolute-form target",
            format!("GET http://evil.example/api/ping HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            400,
        ),
        (
            "an asterisk target",
            format!("GET * HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            400,
        ),
        (
            "a duplicate Host",
            format!("GET /api/ping HTTP/1.1\r\n{good_headers}Host: evil.example\r\n\r\n").into_bytes(),
            400,
        ),
        (
            "a duplicate Authorization",
            format!("GET /api/ping HTTP/1.1\r\n{good_headers}Authorization: Bearer x\r\n\r\n").into_bytes(),
            400,
        ),
        (
            "a duplicate Content-Length",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: 2\r\nContent-Length: 2\r\n\r\n{{}}").into_bytes(),
            400,
        ),
        (
            "chunked encoding",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Transfer-Encoding: chunked\r\n\r\n2\r\n{{}}\r\n0\r\n\r\n").into_bytes(),
            501,
        ),
        (
            "a body without a length",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}\r\n{{}}").into_bytes(),
            411,
        ),
        (
            "a negative length",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: -1\r\n\r\n").into_bytes(),
            400,
        ),
        (
            "a signed length",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: +2\r\n\r\n{{}}").into_bytes(),
            400,
        ),
        (
            "a length that is not a number",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: 0x2\r\n\r\n{{}}").into_bytes(),
            400,
        ),
        (
            "a length of a size no body has",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: 99999999999999999999\r\n\r\n").into_bytes(),
            400,
        ),
        (
            "a body far over the limit",
            format!("PUT /api/settings HTTP/1.1\r\n{good_headers}Content-Length: 5000000\r\n\r\n").into_bytes(),
            413,
        ),
        (
            "an obsolete line folding",
            format!("GET /api/ping HTTP/1.1\r\n{good_headers}X-A: b\r\n  folded\r\n\r\n").into_bytes(),
            400,
        ),
        (
            "binary rubbish",
            vec![0, 1, 2, 3, 255, 254, 253, b'\r', b'\n', b'\r', b'\n'],
            400,
        ),
        (
            "a bare line feed head",
            format!("GET /api/ping HTTP/1.1\nHost: {host}\n\n").into_bytes(),
            400,
        ),
        (
            "a NUL in the target",
            format!("GET /api/\0ping HTTP/1.1\r\n{good_headers}\r\n").into_bytes(),
            400,
        ),
        (
            "a target of 5000 characters",
            format!("GET /{} HTTP/1.1\r\n{good_headers}\r\n", "a".repeat(5000)).into_bytes(),
            414,
        ),
        (
            "a head of 20 KB",
            format!("GET /api/ping HTTP/1.1\r\n{good_headers}X-Pad: {}\r\n\r\n", "a".repeat(20_000)).into_bytes(),
            431,
        ),
        (
            "a hundred headers",
            {
                let mut s = format!("GET /api/ping HTTP/1.1\r\n{good_headers}");
                for i in 0..100 {
                    s.push_str(&format!("X-{i}: v\r\n"));
                }
                s.push_str("\r\n");
                s.into_bytes()
            },
            431,
        ),
    ];
    for (what, bytes, expected) in cases {
        let r = h.raw(&bytes);
        let alternatives: &[u16] = match expected {
            // Some refusals are named differently by different, equally right readings.
            414 => &[414, 400, 431],
            431 => &[431, 400],
            _ => &[],
        };
        assert!(
            r.status == expected || alternatives.contains(&r.status),
            "{what}: expected {expected}, got {} ({})",
            r.status,
            r.text()
        );
        assert!(r.status >= 400, "{what}");
        if expected == 405 {
            assert!(
                r.header("allow").is_some(),
                "{what}: a 405 says what is allowed"
            );
        }
        // Every refusal is JSON the page can read, and says nothing about the system.
        if !r.body.is_empty() {
            let text = r.text();
            assert!(
                !text.contains("panicked") && !text.contains("src/"),
                "{what}: {text}"
            );
        }
        // And the next request is served as if nothing had happened.
        assert_eq!(h.get("/api/ping").status, 200, "after {what}");
    }
}

#[test]
fn a_client_that_leaves_halfway_leaves_nothing_behind() {
    let h = Harness::start();
    let host = h.host();
    // Gone in the middle of the head.
    {
        let mut s = TcpStream::connect(("127.0.0.1", h.port)).unwrap();
        s.write_all(format!("GET /api/ping HTTP/1.1\r\nHost: {host}\r\nAuth").as_bytes())
            .unwrap();
    }
    // Gone before sending anything.
    drop(TcpStream::connect(("127.0.0.1", h.port)).unwrap());
    // Gone in the middle of an upload: no half file may stay in data/.
    {
        let mut s = TcpStream::connect(("127.0.0.1", h.port)).unwrap();
        s.write_all(
            format!(
                "POST /api/upload?name=half.txt HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {}\r\nContent-Length: 1000\r\n\r\nonly ten b",
                h.token
            )
            .as_bytes(),
        )
        .unwrap();
        s.shutdown(Shutdown::Write).unwrap();
        let mut sink = Vec::new();
        let _ = s.read_to_end(&mut sink);
    }
    let started = Instant::now();
    loop {
        let listed = h
            .get(&format!(
                "/api/fs?path={}",
                h.workspace.join("data").display()
            ))
            .json();
        let names: Vec<&str> = listed["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        if names.is_empty() {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a partial upload stayed: {names:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(h.get("/api/ping").status, 200);
}

#[test]
fn names_from_the_network_cannot_reach_outside_the_workspace() {
    let h = Harness::start();
    // A model's name becomes a folder under models/; anything that is not a plain name is refused.
    for bad in [
        "../evil",
        "..",
        ".",
        "a/b",
        "a\\b",
        "",
        "CON",
        "nul.txt",
        "aux",
        "com1",
        "-x",
        ".hidden",
        "a b",
        "é",
        "x\u{0}y",
        "trailing.",
        &"n".repeat(65),
    ] {
        let body = tiny_train(bad, 2);
        let r = h.post("/api/train", &body);
        assert_eq!(r.status, 400, "name {bad:?}: {}", r.text());
        assert_eq!(r.error_code(), "bad_name", "name {bad:?}");
    }
    assert!(!h.workspace.parent().unwrap().join("evil").exists());
    // A model to read must be a name in the workspace or an absolute folder that holds a model.
    for bad in [
        "../../etc/passwd",
        "..",
        "relative/dir",
        "/",
        "/etc",
        "/proc/self",
        "C:\\Windows",
    ] {
        for path in ["/api/generate", "/api/verify", "/api/recover"] {
            let r = h.post(path, &json!({ "model": bad }));
            assert!(
                (400..500).contains(&r.status),
                "{path} with {bad:?} gave {}: {}",
                r.status,
                r.text()
            );
        }
    }
    // Segments of a path are decoded one by one, so an encoded slash cannot make a new level.
    for path in [
        "/api/jobs/..%2F..%2Fetc",
        "/api/jobs/%2e%2e",
        "/api/jobs/1%2F..%2F..",
        "/api/jobs/1/events/../../..",
        "/api/jobs//",
        "/api/%2e%2e/%2e%2e/etc/passwd",
    ] {
        let r = h.get(path);
        assert!(r.status == 404 || r.status == 400, "{path}: {}", r.status);
    }
    // A job number is a number, however it is written.
    for path in [
        "/api/jobs/-1",
        "/api/jobs/1e3",
        "/api/jobs/0x1",
        "/api/jobs/99999999999999999999",
        "/api/jobs/%201",
    ] {
        assert_eq!(h.get(path).status, 404, "{path}");
    }
    // The picker only takes absolute paths.
    for path in ["relative", "..", "~", "", "data/../.."] {
        let r = h.get(&format!("/api/fs?path={path}"));
        assert!(
            r.status == 400 || (r.status == 200 && path.is_empty()),
            "fs path {path:?}: {}",
            r.status
        );
    }
    // Training data is a file somewhere, never a URL or a relative path.
    for path in [
        "relative.txt",
        "http://evil.example/x",
        "file:///etc/passwd",
        "../x",
    ] {
        let mut body = tiny_train("ok", 2);
        body["data"] = json!({ "kind": "file", "path": path });
        let r = h.post("/api/train", &body);
        assert_eq!(r.status, 400, "data path {path:?}: {}", r.text());
    }
    let mut missing = tiny_train("ok", 2);
    missing["data"] =
        json!({ "kind": "file", "path": h.workspace.join("data/nope.txt").display().to_string() });
    assert_eq!(h.post("/api/train", &missing).status, 400);
    assert!(
        h.get("/api/jobs").json()["jobs"]
            .as_array()
            .unwrap()
            .is_empty(),
        "no job was started by any of this"
    );
}

#[test]
fn bodies_are_checked_before_anything_runs() {
    let h = Harness::start();
    let wrong: Vec<(&str, Vec<u8>)> = vec![
        ("not JSON", b"hello".to_vec()),
        ("empty", Vec::new()),
        ("an array", b"[1,2]".to_vec()),
        ("a number", b"7".to_vec()),
        ("null", b"null".to_vec()),
        ("truncated", b"{\"name\":\"a\"".to_vec()),
        (
            "deeply nested",
            format!("{}1{}", "[".repeat(5000), "]".repeat(5000)).into_bytes(),
        ),
        ("a NUL", b"{\"name\":\"a\0\"}".to_vec()),
        (
            "invalid UTF-8",
            vec![b'{', b'"', 0xff, 0xfe, b'"', b':', b'1', b'}'],
        ),
    ];
    for (what, body) in wrong {
        for path in ["/api/train", "/api/generate", "/api/bench", "/api/verify"] {
            let r = h.request("POST", path, &[], Some(&body), true);
            assert!(
                (400..500).contains(&r.status),
                "{what} to {path}: {} {}",
                r.status,
                r.text()
            );
        }
    }
    // An unknown field is a mistake (a typo in a script), not something to ignore.
    let mut body = tiny_train("t", 2);
    body["steeps"] = json!(5);
    let r = h.post("/api/train", &body);
    assert_eq!(r.status, 400);
    // A value out of range names the field.
    for (field, value) in [
        ("steps", json!(0)),
        ("batch", json!(100_000)),
        ("lr", json!(-1)),
        ("lr", json!(1e9)),
        ("preset", json!("gigantic")),
        ("tokenizer", json!("word")),
        ("val_fraction", json!(0.9)),
    ] {
        let mut body = tiny_train("t", 2);
        body[field] = value.clone();
        let r = h.post("/api/train", &body);
        assert_eq!(r.status, 400, "{field} = {value}");
        assert_eq!(r.error_code(), "invalid_field", "{field} = {value}");
        assert_eq!(r.json()["error"]["params"]["field"], field);
    }
    // Numbers that are not finite cannot be written in JSON at all, and are refused as text.
    let r = h.request(
        "POST",
        "/api/train",
        &[],
        Some(b"{\"name\":\"t\",\"data\":{\"kind\":\"demo\"},\"lr\":NaN}"),
        true,
    );
    assert_eq!(r.status, 400);
    assert!(
        h.get("/api/jobs").json()["jobs"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

// ---------------------------------------------------------------------------------------------
// Settings, files
// ---------------------------------------------------------------------------------------------

#[test]
fn the_system_describes_itself() {
    let h = Harness::start();
    let s = h.get("/api/system").json();
    assert!(s["version"].as_str().unwrap().starts_with("0."));
    assert_eq!(s["presets"].as_array().unwrap().len(), 5);
    assert_eq!(s["languages"].as_array().unwrap().len(), 8);
    assert!(s["logical_cpus"].as_u64().unwrap() >= 1);
    assert_eq!(s["workers"], 2, "the thread count that was asked for");
    assert_eq!(s["workspace"], h.workspace.display().to_string());
    assert!(s["limits"]["upload_bytes"].as_u64().unwrap() > 0);
    let mut sizes: Vec<u64> = s["presets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["params"].as_u64().unwrap())
        .collect();
    let sorted = {
        let mut x = sizes.clone();
        x.sort_unstable();
        x
    };
    sizes.dedup();
    assert_eq!(sizes, sorted, "the presets grow");
    // The size of a run is told before it is started.
    let e = h
        .get("/api/estimate?preset=nano&batch=4&seq=32&vocab=256")
        .json();
    assert_eq!(e["fits"], true);
    assert!(e["train_bytes"].as_u64().unwrap() > e["weights_bytes"].as_u64().unwrap());
    let huge = h
        .get("/api/estimate?preset=base&batch=4096&seq=16384&vocab=65536")
        .json();
    assert_eq!(huge["fits"], false);
    assert!(huge["problem"].is_string());
    assert_eq!(h.get("/api/estimate?preset=nope").status, 400);
    assert_eq!(h.get("/api/estimate?batch=abc").status, 400);
}

#[test]
fn settings_are_kept_checked_and_survive_a_restart() {
    let mut h = Harness::start();
    let defaults = h.get("/api/settings").json();
    assert_eq!(defaults["lang"], "auto");
    assert_eq!(defaults["theme"], "auto");
    let saved = h.send_json(
        "PUT",
        "/api/settings",
        &json!({ "lang": "ro", "theme": "dark", "text_scale": 1.25 }),
    );
    assert_eq!(saved.status, 200, "{}", saved.text());
    assert_eq!(saved.json()["lang"], "ro");
    assert_eq!(h.get("/api/settings").json()["theme"], "dark");
    // Bad values are refused whole: nothing of the patch is applied.
    for patch in [
        json!({ "lang": "klingon" }),
        json!({ "lang": "fr", "theme": "neon" }),
        json!({ "text_scale": 9 }),
        json!({ "text_scale": "big" }),
        json!({ "notify": "yes" }),
        json!({ "unknown": 1 }),
        json!({ "form": [1, 2] }),
        json!([1]),
    ] {
        let r = h.send_json("PUT", "/api/settings", &patch);
        assert_eq!(r.status, 400, "{patch}: {}", r.text());
    }
    let now = h.get("/api/settings").json();
    assert_eq!(now["lang"], "ro", "a refused patch changed nothing");
    assert_eq!(now["theme"], "dark");
    // A file edited by hand into nonsense falls back to the defaults, key by key.
    std::fs::write(
        h.workspace.join("settings.json"),
        r#"{"lang":"ro","theme":"mauve","text_scale":50}"#,
    )
    .unwrap();
    let read = h.get("/api/settings").json();
    assert_eq!(read["lang"], "ro");
    assert_eq!(read["theme"], "auto");
    assert_eq!(read["text_scale"], 1.0);
    // The Expect header: a client that waits for 100-continue before sending its body.
    let host = h.host();
    let body = br#"{"density":"compact"}"#;
    let mut s = TcpStream::connect(("127.0.0.1", h.port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(
        format!(
            "PUT /api/settings HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {}\r\nExpect: 100-continue\r\nContent-Length: {}\r\n\r\n",
            h.token,
            body.len()
        )
        .as_bytes(),
    )
    .unwrap();
    let mut interim = [0u8; 25];
    s.read_exact(&mut interim).unwrap();
    assert!(String::from_utf8_lossy(&interim).starts_with("HTTP/1.1 100 Continue"));
    s.write_all(body).unwrap();
    assert_eq!(read_reply(&mut s).status, 200);
    assert_eq!(h.get("/api/settings").json()["density"], "compact");
    h.stop();
}

#[test]
fn a_text_can_be_uploaded_within_the_limit_and_never_over_a_file() {
    let h = Harness::start();
    let data = h.workspace.join("data");
    let text: Vec<u8> = "Ana are mere. ".repeat(100).into_bytes();
    let up = |name: &str, body: &[u8]| {
        h.request(
            "POST",
            &format!("/api/upload?name={name}"),
            &[],
            Some(body),
            true,
        )
    };
    let first = up("carte.txt", &text);
    assert_eq!(first.status, 200, "{}", first.text());
    let first = first.json();
    assert_eq!(first["bytes"], text.len());
    let path = PathBuf::from(first["path"].as_str().unwrap());
    assert!(path.starts_with(&data), "{}", path.display());
    assert_eq!(std::fs::read(&path).unwrap(), text);
    // The same name again is a new file, not the old one overwritten.
    let second = up("carte.txt", b"alt text").json();
    assert_ne!(second["path"], first["path"]);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        text,
        "the first file is untouched"
    );
    // Names that try to leave the folder, or to mean something else, are made plain.
    for name in [
        "../../escape.txt",
        "..%2F..%2Fescape.txt",
        "%2e%2e%2fescape",
        "CON.txt",
        "....",
        "%5Cwindows%5Csystem32%5Cx",
        ".bashrc",
        "",
    ] {
        let r = up(name, b"x");
        assert_eq!(r.status, 200, "{name}: {}", r.text());
        let p = PathBuf::from(r.json()["path"].as_str().unwrap());
        assert_eq!(p.parent().unwrap(), data, "{name} -> {}", p.display());
        let file = p.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            !file.starts_with('.') && !file.contains(['/', '\\', '\0']),
            "{file}"
        );
    }
    // A NUL byte in the name is refused outright (in the query, before any file is made).
    assert_eq!(up("a%00b.txt", b"x").status, 400);
    assert_eq!(
        up("%ff%fe.txt", b"x").status,
        400,
        "and so is a name that is not UTF-8"
    );
    assert!(
        !h.workspace.join("escape.txt").exists()
            && !h.workspace.parent().unwrap().join("escape.txt").exists()
    );
    // Over the limit: refused from the head alone, with nothing written.
    let before = std::fs::read_dir(&data).unwrap().count();
    let host = h.host();
    let r = h.raw(
        format!(
            "POST /api/upload?name=big.txt HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {}\r\nContent-Length: 2000000\r\n\r\nxxxx",
            h.token
        )
        .as_bytes(),
    );
    assert_eq!(r.status, 413);
    assert_eq!(r.error_code(), "too_large");
    assert_eq!(std::fs::read_dir(&data).unwrap().count(), before);
    // An empty text is not a text.
    let r = h.request("POST", "/api/upload?name=empty.txt", &[], Some(b""), true);
    assert!(r.status == 400 || r.status == 411, "{}", r.status);
    assert_eq!(std::fs::read_dir(&data).unwrap().count(), before);
}

#[test]
fn the_file_picker_lists_what_is_there() {
    let h = Harness::start();
    let data = h.workspace.join("data");
    std::fs::write(data.join("b.txt"), "x").unwrap();
    std::fs::write(data.join("a.txt"), "yy").unwrap();
    std::fs::write(data.join(".secret"), "z").unwrap();
    std::fs::create_dir(data.join("sub")).unwrap();
    let listing = h.get(&format!("/api/fs?path={}", data.display())).json();
    let names: Vec<&str> = listing["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["sub", "a.txt", "b.txt"],
        "folders first, hidden files hidden"
    );
    let all = h
        .get(&format!("/api/fs?path={}&hidden=1", data.display()))
        .json();
    assert_eq!(all["entries"].as_array().unwrap().len(), 4);
    let a = listing["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == "a.txt")
        .unwrap();
    assert_eq!(a["size"], 2);
    assert_eq!(a["dir"], false);
    assert!(
        !listing["roots"].as_array().unwrap().is_empty(),
        "places to start from"
    );
    assert_eq!(
        h.get(&format!("/api/fs?path={}", data.join("nope").display()))
            .status,
        404
    );
    assert_eq!(
        h.get(&format!("/api/fs?path={}", data.join("a.txt").display()))
            .status,
        400,
        "a file is not a folder"
    );
    // An enormous folder is cut, and says so.
    let many = h.workspace.join("many");
    std::fs::create_dir(&many).unwrap();
    for i in 0..3100 {
        std::fs::write(many.join(format!("f{i:05}.txt")), "").unwrap();
    }
    let big = h.get(&format!("/api/fs?path={}", many.display())).json();
    assert!(big["entries"].as_array().unwrap().len() <= 3000);
    assert_eq!(big["truncated"], true);
}

// ---------------------------------------------------------------------------------------------
// The work itself
// ---------------------------------------------------------------------------------------------

#[test]
fn a_model_is_trained_from_scratch_checked_used_and_trained_further_all_over_http() {
    let h = Harness::start();
    assert!(
        h.get("/api/models").json()["models"]
            .as_array()
            .unwrap()
            .is_empty()
    );

    // Train.
    let job = h.start_job("/api/train", &tiny_train("t1", 30));
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    for needed in ["stage", "start", "step", "eval", "sample", "saved", "end"] {
        assert!(types.contains(&needed), "no {needed} event in {types:?}");
    }
    assert_eq!(types.iter().filter(|t| **t == "step").count(), 30);
    assert_eq!(types.last(), Some(&"end"));
    let start = events.iter().find(|e| e["type"] == "start").unwrap();
    assert_eq!(start["resume"], false);
    let nano_layers = h.get("/api/system").json()["presets"][0]["layers"].clone();
    assert_eq!(start["layers"], nano_layers);
    assert_eq!(start["tokenizer"], "byte");
    assert_eq!(start["total_steps"], 30);
    assert!(start["params"].as_u64().unwrap() > 1000);
    let evals: Vec<&Value> = events.iter().filter(|e| e["type"] == "eval").collect();
    assert!(evals.len() >= 3, "evaluated at steps 10, 20, 30");
    assert!(evals[0]["bpb"].as_f64().unwrap() > 0.0);
    // The snapshot is the sum of the events: the same curve, in columns.
    let steps = &snap["train"]["steps"];
    assert_eq!(steps["step"].as_array().unwrap().len(), 30);
    let loss: Vec<f64> = steps["loss"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert!(loss.iter().all(|l| l.is_finite() && *l > 0.0));
    assert!(
        loss.last().unwrap() < &(loss[0] - 0.3),
        "it learned something: {loss:?}"
    );
    assert_eq!(snap["result"]["final_step"], 30);
    assert_eq!(snap["result"]["interrupted"], false);
    assert_eq!(snap["train"]["stage"]["name"], "training");
    assert!(!snap["train"]["samples"].as_array().unwrap().is_empty());
    assert_eq!(
        snap["seq"].as_u64().unwrap() as usize,
        events.len(),
        "the snapshot names the next event"
    );
    // The same run, asked for again, from the middle: a page that arrives late sees all of it.
    let (late, _) = h.follow(job, LONG);
    assert_eq!(late.len(), events.len());

    // The model is listed.
    let listed = h.get("/api/models").json();
    let models = listed["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "t1");
    assert_eq!(models[0]["step"], 30);
    assert_eq!(models[0]["tokenizer"], "byte");
    assert_eq!(models[0]["has_manifest"], true);

    // A new model may not take the place of this one.
    let again = h.post("/api/train", &tiny_train("t1", 5));
    assert_eq!(again.status, 409);
    assert_eq!(again.error_code(), "model_exists");
    // Nothing to continue where there is no model.
    let mut nothing = tiny_train("nope", 5);
    nothing["mode"] = json!("resume");
    assert_eq!(h.post("/api/train", &nothing).status, 404);

    // Verify, from top to bottom.
    let verified = h.post("/api/verify", &json!({ "model": "t1" }));
    assert_eq!(verified.status, 200, "{}", verified.text());
    assert_eq!(verified.json()["step"], 30);
    assert!(!verified.json()["files"].as_array().unwrap().is_empty());
    // A damaged file is found.
    let model_dir = h.workspace.join("models/t1");
    let weights = std::fs::read_dir(&model_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "bpk"))
        .expect("a weights file");
    let original = std::fs::read(&weights).unwrap();
    let mut damaged = original.clone();
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0x55;
    std::fs::write(&weights, &damaged).unwrap();
    let bad = h.post("/api/verify", &json!({ "model": "t1" }));
    assert_eq!(bad.status, 422, "a flipped bit is found: {}", bad.text());
    assert_eq!(bad.error_code(), "engine");
    std::fs::write(&weights, &original).unwrap();
    assert_eq!(h.post("/api/verify", &json!({ "model": "t1" })).status, 200);

    // Generate, twice with the same seed: the same text.
    let ask = json!({ "model": "t1", "prompt": "Ana ", "max_new": 40, "temperature": 0.8, "top_k": 20, "window": 32, "seed": 7 });
    let first = h.start_job("/api/generate", &ask);
    let (gen_events, first_snap) = h.follow(first, LONG);
    assert_eq!(first_snap["state"], "done", "{}", first_snap["error"]);
    assert_eq!(gen_events[0]["type"], "loaded");
    assert_eq!(first_snap["generate"]["loaded"]["step"], 30);
    let text = first_snap["generate"]["text"].as_str().unwrap().to_string();
    assert!(!text.is_empty());
    assert_eq!(first_snap["result"]["tokens"], 40);
    assert!(first_snap["result"]["tokens_per_sec"].as_f64().unwrap() > 0.0);
    let joined: String = gen_events
        .iter()
        .filter(|e| e["type"] == "text")
        .map(|e| e["text"].as_str().unwrap())
        .collect();
    assert_eq!(joined, text, "the pieces add up to the text");
    let second = h.start_job("/api/generate", &ask);
    let (_, second_snap) = h.follow(second, LONG);
    assert_eq!(
        second_snap["generate"]["text"], text,
        "the same seed gives the same text"
    );
    // The prompt is limited, and a model that does not exist is not found.
    let long_prompt = json!({ "model": "t1", "prompt": "x".repeat(70_000) });
    assert_eq!(h.post("/api/generate", &long_prompt).status, 400);
    assert_eq!(
        h.post("/api/generate", &json!({ "model": "nope" })).status,
        404
    );

    // Go on training: `steps` is the total, so 36 means six more.
    let mut more = tiny_train("t1", 36);
    more["mode"] = json!("resume");
    let job = h.start_job("/api/train", &more);
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let start = events.iter().find(|e| e["type"] == "start").unwrap();
    assert_eq!(start["resume"], true);
    assert_eq!(start["from_step"], 30);
    assert_eq!(snap["result"]["final_step"], 36);
    assert_eq!(snap["result"]["steps_run"], 6);
    assert_eq!(h.get("/api/models").json()["models"][0]["step"], 36);

    // Start again in the same folder, on purpose: the old model is replaced.
    let mut replace = tiny_train("t1", 4);
    replace["mode"] = json!("overwrite");
    replace["save_every"] = json!(2);
    let job = h.start_job("/api/train", &replace);
    let (_, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    assert_eq!(h.get("/api/models").json()["models"][0]["step"], 4);
    assert_eq!(h.post("/api/verify", &json!({ "model": "t1" })).status, 200);

    // The list of jobs remembers them, newest first.
    let jobs = h.get("/api/jobs").json();
    let kinds: Vec<&str> = jobs["jobs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["train", "train", "generate", "generate", "train"]);
}

#[test]
fn a_model_with_a_learned_vocabulary_trains_and_speaks_utf8() {
    let h = Harness::start();
    let mut body = tiny_train("bpe1", 12);
    body["tokenizer"] = json!("bpe");
    body["vocab"] = json!(300);
    body["data"] = json!({ "kind": "demo", "mb": 0.06, "seed": 2 });
    let job = h.start_job("/api/train", &body);
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let stages: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "stage")
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(
        stages.contains(&"bpe") && stages.contains(&"corpus"),
        "{stages:?}"
    );
    let start = events.iter().find(|e| e["type"] == "start").unwrap();
    assert_eq!(start["tokenizer"], "bpe");
    assert_eq!(start["vocab_size"], 300);
    assert!(
        start["bytes_per_token"].as_f64().unwrap() > 1.0,
        "merged tokens hold more than a byte"
    );
    let m = &h.get("/api/models").json()["models"][0];
    assert_eq!(m["tokenizer"], "bpe");
    let job = h.start_job(
        "/api/generate",
        &json!({ "model": "bpe1", "prompt": "Într-o zi ", "max_new": 60, "seed": 1, "window": 32 }),
    );
    let (_, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let text = snap["generate"]["text"].as_str().unwrap();
    assert!(!text.is_empty());
    assert!(
        !text.contains('\u{fffd}') || text.chars().filter(|c| *c == '\u{fffd}').count() < 3,
        "no stray replacement marks"
    );
}

#[test]
fn a_running_training_stops_on_request_and_leaves_a_model_that_checks_out() {
    let h = Harness::start();
    let mut body = tiny_train("long", 2_000_000);
    body["save_every"] = json!(0);
    body["eval_every"] = json!(0);
    body["sample_every"] = json!(0);
    let job = h.start_job("/api/train", &body);
    let snap = h.wait_for(job, LONG, "three steps", |s| {
        s["train"]["last"]["step"].as_u64().unwrap_or(0) >= 3
    });
    assert_eq!(snap["state"], "running");
    // A second heavy job is turned away, and told which one is in the way.
    let busy = h.post(
        "/api/bench",
        &json!({ "preset": "nano", "steps": 1, "warmup": 0, "batch": 2, "seq": 32 }),
    );
    assert_eq!(busy.status, 409);
    assert_eq!(busy.error_code(), "busy");
    assert_eq!(busy.json()["error"]["params"]["job"], job);
    assert_eq!(busy.json()["error"]["params"]["kind"], "train");
    let busy2 = h.post("/api/train", &tiny_train("other", 5));
    assert_eq!(busy2.status, 409);
    // The folder is not for anyone else while it is being written.
    // Stop.
    let stopped = h.post(&format!("/api/jobs/{job}/stop"), &json!({}));
    assert_eq!(stopped.status, 200);
    let stopped_twice = h.post(&format!("/api/jobs/{job}/stop"), &json!({}));
    assert_eq!(stopped_twice.status, 200);
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "stopped", "{}", snap["error"]);
    assert_eq!(snap["result"]["interrupted"], true);
    let stopping = events.iter().filter(|e| e["type"] == "stopping").count();
    assert_eq!(stopping, 1, "however many times it was asked");
    let reached = snap["result"]["final_step"].as_u64().unwrap();
    assert!(reached >= 3);
    // What it saved when it stopped is a whole model.
    let verified = h.post("/api/verify", &json!({ "model": "long" }));
    assert_eq!(verified.status, 200, "{}", verified.text());
    assert_eq!(verified.json()["step"], reached);
    // And the slot is free again.
    let next = h.start_job("/api/train", &tiny_train("after", 3));
    let (_, snap) = h.follow(next, LONG);
    assert_eq!(snap["state"], "done");
}

#[test]
fn a_model_in_training_is_not_touched_by_anything_else_but_may_be_used() {
    let h = Harness::start();
    // First a model that exists and is stable.
    let first = h.start_job("/api/train", &tiny_train("steady", 10));
    assert_eq!(h.follow(first, LONG).1["state"], "done");
    // Train it further, without saving until the end.
    let mut more = tiny_train("steady", 5_000_000);
    more["mode"] = json!("resume");
    more["save_every"] = json!(0);
    more["eval_every"] = json!(0);
    more["sample_every"] = json!(0);
    let job = h.start_job("/api/train", &more);
    h.wait_for(job, LONG, "a step", |s| {
        s["train"]["last"]["step"].as_u64().unwrap_or(0) >= 11
    });
    // Checking or repairing a folder that is being written would race with the writer.
    for path in ["/api/verify", "/api/recover"] {
        let r = h.post(path, &json!({ "model": "steady" }));
        assert_eq!(r.status, 409, "{path}: {}", r.text());
        assert_eq!(r.error_code(), "model_busy");
    }
    // Using it is fine: generating loads the last whole save.
    let speak = h.start_job(
        "/api/generate",
        &json!({ "model": "steady", "prompt": "Ana ", "max_new": 20, "seed": 3, "window": 32 }),
    );
    let (_, snap) = h.follow(speak, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    assert_eq!(
        snap["generate"]["loaded"]["step"], 10,
        "the save of the first run, not a half-written one"
    );
    assert_eq!(snap["result"]["tokens"], 20);
    // A generation can be stopped too, and keeps what it had.
    let long_gen = h.start_job("/api/generate", &json!({ "model": "steady", "prompt": "Ana ", "max_new": 20_000, "seed": 3, "window": 32, "temperature": 1.0 }));
    h.wait_for(long_gen, LONG, "some text", |s| {
        s["generate"]["text"].as_str().is_some_and(|t| t.len() > 5)
    });
    assert_eq!(
        h.post(&format!("/api/jobs/{long_gen}/stop"), &json!({}))
            .status,
        200
    );
    let (_, snap) = h.follow(long_gen, LONG);
    assert_eq!(snap["state"], "stopped");
    assert!(snap["result"]["tokens"].as_u64().unwrap() < 20_000);
    assert!(snap["generate"]["text"].as_str().unwrap().len() > 5);
    h.post(&format!("/api/jobs/{job}/stop"), &json!({}));
    assert_eq!(h.follow(job, LONG).1["state"], "stopped");
    assert_eq!(
        h.post("/api/verify", &json!({ "model": "steady" })).status,
        200,
        "free again"
    );
}

#[test]
fn speed_can_be_measured_and_the_threads_swept() {
    let h = Harness::start();
    let job = h.start_job(
        "/api/bench",
        &json!({ "preset": "nano", "batch": 2, "seq": 32, "vocab": 256, "steps": 1, "warmup": 0 }),
    );
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types[0], "bench_plan");
    assert!(
        types.contains(&"bench_running") && types.contains(&"bench_result"),
        "{types:?}"
    );
    assert_eq!(snap["bench"]["results"].as_array().unwrap().len(), 1);
    assert!(snap["result"]["text"].as_str().unwrap().contains("nano"));
    assert!(snap["result"]["report"].is_object());
    // A sweep over thread counts reports each point as it is measured.
    let job = h.start_job("/api/bench", &json!({ "preset": "nano", "batch": 2, "seq": 32, "steps": 1, "warmup": 0, "scaling": [1, 2] }));
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "done", "{}", snap["error"]);
    let points = events
        .iter()
        .filter(|e| e["type"] == "scaling_point")
        .count();
    assert_eq!(points, 2);
    assert_eq!(snap["bench"]["points"].as_array().unwrap().len(), 2);
}

#[test]
fn a_stuck_job_that_ends_in_error_says_why_and_frees_the_slot() {
    let h = Harness::start();
    // A text file that is too short for the context.
    let file = h.workspace.join("data/short.txt");
    std::fs::write(&file, "abc").unwrap();
    let mut body = tiny_train("short", 3);
    body["data"] = json!({ "kind": "file", "path": file.display().to_string() });
    let job = h.start_job("/api/train", &body);
    let (events, snap) = h.follow(job, LONG);
    assert_eq!(snap["state"], "failed");
    assert_eq!(snap["error"]["code"], "engine");
    assert!(snap["error"]["message"].as_str().unwrap().len() > 10);
    assert_eq!(events.last().unwrap()["type"], "end");
    assert_eq!(events.last().unwrap()["state"], "failed");
    assert!(
        !h.workspace.join("models/short/meta.json").exists(),
        "nothing half-made is left as a model"
    );
    // The slot is free for the next one.
    let next = h.start_job("/api/train", &tiny_train("fine", 3));
    assert_eq!(h.follow(next, LONG).1["state"], "done");
}

// ---------------------------------------------------------------------------------------------
// Load, and the end
// ---------------------------------------------------------------------------------------------

#[test]
fn too_many_connections_are_turned_away_and_the_server_recovers() {
    let h = Harness::start();
    // Eighty clients that connect and say nothing.
    let idle: Vec<TcpStream> = (0..80)
        .map(|_| TcpStream::connect(("127.0.0.1", h.port)).unwrap())
        .collect();
    std::thread::sleep(Duration::from_millis(300));
    let mut turned_away = 0;
    for mut s in idle.iter().map(|s| s.try_clone().unwrap()) {
        s.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
        let mut buf = [0u8; 64];
        if let Ok(n) = s.read(&mut buf)
            && n > 0
            && buf.starts_with(b"HTTP/1.1 503")
        {
            turned_away += 1;
        }
    }
    assert!(turned_away >= 10, "only {turned_away} were turned away");
    drop(idle);
    // The seats come back as the idle connections close.
    let started = Instant::now();
    loop {
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            h.get("/api/ping").status == 200
        }))
        .unwrap_or(false);
        if ok {
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the server did not recover"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn the_page_can_close_the_server_and_a_running_training_is_saved_first() {
    let mut h = Harness::start();
    // Without the token, no.
    assert_eq!(
        h.request("POST", "/api/quit", &[], Some(b"{}"), false)
            .status,
        401
    );
    assert_eq!(h.get("/api/ping").status, 200);
    let mut body = tiny_train("closing", 3_000_000);
    body["save_every"] = json!(0);
    let job = h.start_job("/api/train", &body);
    h.wait_for(job, LONG, "two steps", |s| {
        s["train"]["last"]["step"].as_u64().unwrap_or(0) >= 2
    });
    let r = h.post("/api/quit", &json!({}));
    assert_eq!(r.status, 200);
    // The server ends by itself, after the training has saved.
    let thread = h.thread.take().unwrap();
    let started = Instant::now();
    while !thread.is_finished() {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the server did not end"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    thread.join().unwrap();
    assert!(
        TcpStream::connect(("127.0.0.1", h.port)).is_err(),
        "the port is closed"
    );
    let meta = std::fs::read_to_string(h.workspace.join("models/closing/meta.json"))
        .expect("the model was saved on the way out");
    let meta: Value = serde_json::from_str(&meta).unwrap();
    assert!(meta["step"].as_u64().unwrap() >= 2);
}

#[test]
fn two_studios_do_not_share_a_token_or_a_job_list() {
    let a = Harness::start();
    let b = Harness::start();
    assert_ne!(a.token, b.token);
    assert_ne!(a.port, b.port);
    // A's token does not open B.
    let r = b.raw(
        format!(
            "GET /api/ping HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {}\r\n\r\n",
            b.host(),
            a.token
        )
        .as_bytes(),
    );
    assert_eq!(r.status, 401);
    // And A's host name is not B's.
    let r = b.raw(
        format!(
            "GET /api/ping HTTP/1.1\r\nHost: {}\r\nAuthorization: Bearer {}\r\n\r\n",
            a.host(),
            b.token
        )
        .as_bytes(),
    );
    assert_eq!(r.status, 421);
}
