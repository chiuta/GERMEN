//! The HTTP/1.1 subset the Studio speaks.
//!
//! One request per connection (`Connection: close`): no keep-alive, no pipelining, no chunked
//! bodies, no `Upgrade`. The browser and the page are the only clients, and everything that is not
//! needed is one more way in for a malformed request. Every limit is checked before the data it
//! guards against is read, and every header that decides something (`Host`, `Content-Length`,
//! `Authorization`, `Origin`) may appear only once, so that two parties along the way cannot
//! disagree about which of two copies counts.

use serde_json::{Value, json};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::Duration;

/// Longest request head (request line and headers together) that is read.
pub(crate) const MAX_HEAD_BYTES: usize = 16 * 1024;
/// Most header lines accepted.
const MAX_HEADERS: usize = 64;
/// Longest request target (path and query).
const MAX_TARGET_BYTES: usize = 4096;
/// Most query parameters accepted.
const MAX_QUERY_PARAMS: usize = 32;
/// How long a client may take to send the head of a request.
pub(crate) const HEAD_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a client may stay silent in the middle of a body.
pub(crate) const BODY_TIMEOUT: Duration = Duration::from_secs(30);
/// How much unread input is thrown away after an answer, so that the close is not a reset.
const DRAIN_BYTES: usize = 256 * 1024;

/// A failure that becomes an answer: status, a code the page translates, and a readable message.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
    pub params: Value,
}

impl ApiError {
    pub(crate) fn new(status: u16, code: &'static str, message: impl Into<String>) -> ApiError {
        ApiError {
            status,
            code,
            message: message.into(),
            params: Value::Null,
        }
    }

    /// Attaches values the page puts into its translated text.
    pub(crate) fn with(mut self, params: Value) -> ApiError {
        self.params = params;
        self
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> ApiError {
        ApiError::new(400, "bad_request", message)
    }

    /// A field of the request body that is not acceptable.
    pub(crate) fn field(field: &str, message: impl Into<String>) -> ApiError {
        ApiError::new(400, "invalid_field", message).with(json!({ "field": field }))
    }

    /// An error message of the engine itself (checkpoints, memory, corpus): shown as it is.
    pub(crate) fn engine(message: impl Into<String>) -> ApiError {
        ApiError::new(422, "engine", message)
    }

    pub(crate) fn not_found() -> ApiError {
        ApiError::new(404, "not_found", "nu există")
    }

    pub(crate) fn to_response(&self) -> Response {
        let mut error = json!({ "code": self.code, "message": self.message });
        if !self.params.is_null() {
            error["params"] = self.params.clone();
        }
        Response::json(self.status, &json!({ "error": error }))
    }
}

/// Request methods the Studio knows. Anything else is answered with 405.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Get,
    Post,
    Put,
}

/// The head of a request, parsed and checked.
#[derive(Debug)]
pub(crate) struct Head {
    pub method: Method,
    /// Path segments, each percent-decoded on its own (so `%2F` stays inside a segment).
    pub segments: Vec<String>,
    pub query: Vec<(String, String)>,
    /// Header names in lowercase.
    pub headers: Vec<(String, String)>,
    pub content_length: Option<u64>,
}

impl Head {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    pub(crate) fn param(&self, key: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The path segments as `&str`, for matching.
    pub(crate) fn path(&self) -> Vec<&str> {
        self.segments.iter().map(String::as_str).collect()
    }
}

/// A connection: the socket, and what was read past the end of the head.
pub(crate) struct Conn {
    pub stream: TcpStream,
    pub rest: Vec<u8>,
}

impl Conn {
    pub(crate) fn new(stream: TcpStream) -> Conn {
        Conn {
            stream,
            rest: Vec::new(),
        }
    }
}

/// The body of a request, as far as the head announced it.
pub(crate) struct Body<'a> {
    conn: &'a mut Conn,
    remaining: u64,
}

impl<'a> Body<'a> {
    pub(crate) fn new(conn: &'a mut Conn, length: u64) -> Body<'a> {
        Body {
            conn,
            remaining: length,
        }
    }
}

impl Read for Body<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || out.is_empty() {
            return Ok(0);
        }
        let want = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(out.len());
        let got = if self.conn.rest.is_empty() {
            self.conn.stream.read(&mut out[..want])?
        } else {
            let k = want.min(self.conn.rest.len());
            out[..k].copy_from_slice(&self.conn.rest[..k]);
            self.conn.rest.drain(..k);
            k
        };
        if got == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        self.remaining -= got as u64;
        Ok(got)
    }
}

/// Reads the whole body into memory, refusing one longer than `limit` before reading any of it.
pub(crate) fn read_all(body: &mut Body<'_>, limit: u64) -> Result<Vec<u8>, ApiError> {
    if body.remaining > limit {
        return Err(ApiError::new(
            413,
            "too_large",
            format!("cererea e prea mare (cel mult {limit} octeți)"),
        )
        .with(json!({ "limit": limit })));
    }
    let mut out = Vec::with_capacity(body.remaining as usize);
    body.read_to_end(&mut out)
        .map_err(|e| ApiError::bad_request(format!("cererea s-a întrerupt: {e}")))?;
    Ok(out)
}

fn bad(status: u16, code: &'static str, message: &str) -> ApiError {
    ApiError::new(status, code, message)
}

/// Reads and checks the head of a request. `Ok(None)`: the client left without saying anything.
pub(crate) fn read_head(conn: &mut Conn) -> Result<Option<Head>, ApiError> {
    let _ = conn.stream.set_read_timeout(Some(HEAD_TIMEOUT));
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    let end = loop {
        if let Some(i) = find_head_end(&buf) {
            break i;
        }
        // A head that ends its lines with a bare line feed never contains the blank line that
        // ends a head here; say so at once instead of waiting for the timeout.
        if buf.windows(2).any(|w| w == b"\n\n") {
            return Err(bad(
                400,
                "bad_request",
                "liniile antetului se termină cu CRLF",
            ));
        }
        if buf.len() > MAX_HEAD_BYTES {
            return Err(bad(431, "head_too_large", "antetul cererii e prea mare"));
        }
        match conn.stream.read(&mut chunk) {
            Ok(0) if buf.is_empty() => return Ok(None),
            Ok(0) => return Err(bad(400, "bad_request", "cerere întreruptă")),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                return if buf.is_empty() {
                    Ok(None)
                } else {
                    Err(bad(408, "timeout", "cererea a venit prea încet"))
                };
            }
            Err(_) => return Ok(None),
        }
    };
    if end > MAX_HEAD_BYTES {
        return Err(bad(431, "head_too_large", "antetul cererii e prea mare"));
    }
    conn.rest = buf.split_off(end + 4);
    buf.truncate(end);
    let _ = conn.stream.set_read_timeout(Some(BODY_TIMEOUT));
    parse_head(&buf).map(Some)
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

/// Headers that decide something: a second copy is refused, not merged or ignored.
const SINGLE: [&str; 9] = [
    "host",
    "content-length",
    "authorization",
    "origin",
    "content-type",
    "transfer-encoding",
    "expect",
    "sec-fetch-site",
    "sec-fetch-mode",
];

fn parse_head(raw: &[u8]) -> Result<Head, ApiError> {
    if !raw.is_ascii() || raw.contains(&0) {
        return Err(bad(400, "bad_request", "antet cu caractere nepermise"));
    }
    let text = std::str::from_utf8(raw).map_err(|_| bad(400, "bad_request", "antet invalid"))?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad(400, "bad_request", "linie de cerere invalidă"));
    };
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(bad(505, "bad_version", "versiune HTTP neacceptată"));
    }
    let method = match method {
        "GET" => Method::Get,
        "POST" => Method::Post,
        "PUT" => Method::Put,
        _ => return Err(bad(405, "method_not_allowed", "metodă neacceptată")),
    };
    if target.len() > MAX_TARGET_BYTES {
        return Err(bad(414, "uri_too_long", "adresă prea lungă"));
    }
    if !target.starts_with('/') || target.bytes().any(|b| b <= b' ' || b == 0x7f || b == b'#') {
        return Err(bad(400, "bad_request", "adresă invalidă"));
    }
    let (raw_path, raw_query) = match target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (target, ""),
    };
    let segments = split_path(raw_path)?;
    let query = parse_query(raw_query)?;

    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        if line.is_empty() {
            return Err(bad(400, "bad_request", "rând gol în antet"));
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            return Err(bad(
                400,
                "bad_request",
                "antete continuate pe rândul următor",
            ));
        }
        if headers.len() >= MAX_HEADERS {
            return Err(bad(431, "head_too_large", "prea multe antete"));
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(bad(400, "bad_request", "antet fără „:”"));
        };
        if name.is_empty() || !name.bytes().all(is_token_byte) {
            return Err(bad(400, "bad_request", "nume de antet invalid"));
        }
        let value = value.trim_matches(|c| c == ' ' || c == '\t');
        if value.bytes().any(|b| (b < b' ' && b != b'\t') || b == 0x7f) {
            return Err(bad(400, "bad_request", "valoare de antet invalidă"));
        }
        headers.push((name.to_ascii_lowercase(), value.to_string()));
    }
    for single in SINGLE {
        if headers.iter().filter(|(n, _)| n == single).count() > 1 {
            return Err(bad(400, "bad_request", "antet repetat"));
        }
    }
    let head = Head {
        method,
        segments,
        query,
        headers,
        content_length: None,
    };
    if head.header("host").is_none() {
        return Err(bad(400, "bad_request", "lipsește antetul Host"));
    }
    if head.header("transfer-encoding").is_some() {
        return Err(bad(
            501,
            "not_implemented",
            "corpul în bucăți nu e acceptat",
        ));
    }
    let content_length = match head.header("content-length") {
        None => None,
        Some(v) if !v.is_empty() && v.len() <= 19 && v.bytes().all(|b| b.is_ascii_digit()) => {
            Some(v.parse::<u64>().expect("up to 19 digits fit in a u64"))
        }
        Some(_) => return Err(bad(400, "bad_request", "Content-Length invalid")),
    };
    if method != Method::Get && content_length.is_none() {
        return Err(bad(411, "length_required", "lipsește Content-Length"));
    }
    Ok(Head {
        content_length,
        ..head
    })
}

fn split_path(raw: &str) -> Result<Vec<String>, ApiError> {
    if raw == "/" {
        return Ok(Vec::new());
    }
    raw[1..]
        .split('/')
        .map(|segment| {
            let decoded = percent_decode(segment, false)
                .ok_or_else(|| bad(400, "bad_request", "adresă codificată greșit"))?;
            if decoded.is_empty() || decoded.contains(['/', '\\']) {
                return Err(bad(404, "not_found", "nu există"));
            }
            Ok(decoded)
        })
        .collect()
}

fn parse_query(raw: &str) -> Result<Vec<(String, String)>, ApiError> {
    let mut out = Vec::new();
    for pair in raw.split('&').filter(|p| !p.is_empty()) {
        if out.len() >= MAX_QUERY_PARAMS {
            return Err(bad(400, "bad_request", "prea mulți parametri"));
        }
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let k = percent_decode(k, true);
        let v = percent_decode(v, true);
        let (Some(k), Some(v)) = (k, v) else {
            return Err(bad(400, "bad_request", "parametri codificați greșit"));
        };
        out.push((k, v));
    }
    Ok(out)
}

/// Decodes `%XX`; `plus` also turns `+` into a space. `None` for a bad escape, for bytes that are
/// not UTF-8 once decoded, and for a NUL.
pub(crate) fn percent_decode(text: &str, plus: bool) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                let hi = hex_value(*bytes.get(i + 1)?)?;
                let lo = hex_value(*bytes.get(i + 2)?)?;
                out.push(hi * 16 + lo);
                i += 3;
            }
            b'+' if plus => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    if out.contains(&0) {
        return None;
    }
    String::from_utf8(out).ok()
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// An answer, ready to be written.
pub(crate) struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub headers: Vec<(&'static str, String)>,
    pub body: Vec<u8>,
}

impl Response {
    pub(crate) fn json(status: u16, value: &Value) -> Response {
        Response {
            status,
            content_type: "application/json; charset=utf-8",
            headers: Vec::new(),
            body: serde_json::to_vec(value).unwrap_or_else(|_| b"null".to_vec()),
        }
    }

    pub(crate) fn ok(value: &Value) -> Response {
        Response::json(200, value)
    }

    pub(crate) fn html(body: String, csp: &str) -> Response {
        Response {
            status: 200,
            content_type: "text/html; charset=utf-8",
            headers: vec![("Content-Security-Policy", csp.to_string())],
            body: body.into_bytes(),
        }
    }

    pub(crate) fn empty(status: u16) -> Response {
        Response {
            status,
            content_type: "text/plain; charset=utf-8",
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        411 => "Length Required",
        413 => "Content Too Large",
        414 => "URI Too Long",
        421 => "Misdirected Request",
        422 => "Unprocessable Content",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        507 => "Insufficient Storage",
        _ => "Status",
    }
}

/// Writes the answer and closes the connection politely: what the client has not yet sent is read
/// and thrown away for a moment, because closing a socket with unread input resets it, and a reset
/// can destroy an answer (a 413, say) that the client has not read yet.
pub(crate) fn finish(conn: &mut Conn, response: &Response) {
    let mut head = String::with_capacity(512);
    head.push_str(&format!(
        "HTTP/1.1 {} {}\r\n",
        response.status,
        reason(response.status)
    ));
    head.push_str(&format!("Content-Type: {}\r\n", response.content_type));
    head.push_str(&format!("Content-Length: {}\r\n", response.body.len()));
    head.push_str("Connection: close\r\n");
    head.push_str("Cache-Control: no-store\r\n");
    head.push_str("X-Content-Type-Options: nosniff\r\n");
    head.push_str("Referrer-Policy: no-referrer\r\n");
    head.push_str("Cross-Origin-Resource-Policy: same-origin\r\n");
    head.push_str("Cross-Origin-Opener-Policy: same-origin\r\n");
    head.push_str("X-Frame-Options: DENY\r\n");
    head.push_str("Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()\r\n");
    if response.status == 405 {
        head.push_str("Allow: GET, POST, PUT\r\n");
    }
    for (name, value) in &response.headers {
        // Values come from this program, never from a request; refuse a line break all the same.
        if !value.contains(['\r', '\n']) {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
    }
    head.push_str("\r\n");
    let _ = conn.stream.set_write_timeout(Some(BODY_TIMEOUT));
    let sent = conn
        .stream
        .write_all(head.as_bytes())
        .and_then(|()| conn.stream.write_all(&response.body))
        .and_then(|()| conn.stream.flush());
    if sent.is_err() {
        return;
    }
    let _ = conn.stream.shutdown(Shutdown::Write);
    let _ = conn
        .stream
        .set_read_timeout(Some(Duration::from_millis(300)));
    let mut sink = [0u8; 8192];
    let mut drained = 0usize;
    while drained < DRAIN_BYTES {
        match conn.stream.read(&mut sink) {
            Ok(0) | Err(_) => break,
            Ok(n) => drained += n,
        }
    }
}

/// Answers `Expect: 100-continue` so that a client waiting for it sends its body at once.
pub(crate) fn send_continue(conn: &mut Conn) {
    let _ = conn.stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Head, ApiError> {
        parse_head(text.as_bytes())
    }

    fn status(text: &str) -> u16 {
        parse(text).map(|_| 200).unwrap_or_else(|e| e.status)
    }

    #[test]
    fn a_plain_request_is_parsed() {
        let h = parse("GET /api/jobs/3/events?after=7&wait=1500 HTTP/1.1\r\nHost: 127.0.0.1:7870\r\nAuthorization: Bearer abc").unwrap();
        assert_eq!(h.method, Method::Get);
        assert_eq!(h.path(), ["api", "jobs", "3", "events"]);
        assert_eq!(h.param("after"), Some("7"));
        assert_eq!(h.param("wait"), Some("1500"));
        assert_eq!(h.header("authorization"), Some("Bearer abc"));
        assert_eq!(h.header("host"), Some("127.0.0.1:7870"));
        assert_eq!(h.content_length, None);
    }

    #[test]
    fn the_root_has_no_segments() {
        let h = parse("GET / HTTP/1.1\r\nHost: x").unwrap();
        assert!(h.segments.is_empty());
    }

    #[test]
    fn malformed_request_lines_are_refused() {
        for line in [
            "",
            "GET",
            "GET /",
            "GET / HTTP/1.1 extra",
            "GET  / HTTP/1.1",
            "get / HTTP/1.1",
            "GET http://evil/ HTTP/1.1",
            "GET * HTTP/1.1",
            "GET /a b HTTP/1.1",
            "GET /a#frag HTTP/1.1",
            "GET /\u{e9} HTTP/1.1",
        ] {
            let text = format!("{line}\r\nHost: x");
            assert!(
                matches!(status(&text), 400 | 405),
                "accepted {line:?}: {}",
                status(&text)
            );
        }
        assert_eq!(status("GET / HTTP/2\r\nHost: x"), 505);
        assert_eq!(status("DELETE /api/jobs/1 HTTP/1.1\r\nHost: x"), 405);
        assert_eq!(status("OPTIONS / HTTP/1.1\r\nHost: x"), 405);
        assert_eq!(status("HEAD / HTTP/1.1\r\nHost: x"), 405);
    }

    #[test]
    fn a_missing_host_is_refused_and_so_is_a_repeated_one() {
        assert_eq!(status("GET / HTTP/1.1\r\nAccept: */*"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nHost: b"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nhost: b"), 400);
    }

    #[test]
    fn repeated_security_headers_are_refused() {
        for name in [
            "Authorization: Bearer a",
            "Origin: http://a",
            "Content-Type: text/plain",
        ] {
            let text =
                format!("POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\n{name}\r\n{name}");
            assert_eq!(status(&text), 400, "{name}");
        }
        assert_eq!(
            status("POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\nContent-Length: 1"),
            400
        );
    }

    #[test]
    fn content_length_must_be_plain_digits() {
        for bad in [
            "-1",
            "+1",
            "1 2",
            "0x10",
            "1e3",
            "",
            "99999999999999999999",
            "١٢",
        ] {
            let text = format!("POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: {bad}");
            assert_eq!(status(&text), 400, "{bad:?}");
        }
        let h = parse("POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 0042").unwrap();
        assert_eq!(h.content_length, Some(42));
    }

    #[test]
    fn a_body_without_a_length_or_with_chunks_is_refused() {
        assert_eq!(status("POST /x HTTP/1.1\r\nHost: a"), 411);
        assert_eq!(status("PUT /x HTTP/1.1\r\nHost: a"), 411);
        assert_eq!(
            status("POST /x HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: chunked"),
            501
        );
        assert_eq!(
            status(
                "POST /x HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\nTransfer-Encoding: chunked"
            ),
            501
        );
    }

    #[test]
    fn header_syntax_is_strict() {
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\n folded: x"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nX:\ty\r\n\tmore"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nno colon here"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nBad Name: x"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nName : x"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nX: a\u{1}b"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nX: a\rb"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nX: a\nb"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: \u{e9}"), 400);
        assert_eq!(status("GET / HTTP/1.1\r\nHost: a\r\nX: ok\twith tab"), 200);
    }

    #[test]
    fn too_many_headers_are_refused() {
        let mut text = String::from("GET / HTTP/1.1\r\nHost: a");
        for i in 0..MAX_HEADERS {
            text.push_str(&format!("\r\nX-{i}: v"));
        }
        assert_eq!(status(&text), 431);
    }

    #[test]
    fn path_segments_are_decoded_one_by_one() {
        let h = parse("GET /api/models/caf%C3%A9 HTTP/1.1\r\nHost: a").unwrap();
        assert_eq!(h.segments, ["api", "models", "café"]);
        // An encoded slash or backslash never becomes a separator; it is refused.
        assert_eq!(status("GET /api/models/a%2Fb HTTP/1.1\r\nHost: a"), 404);
        assert_eq!(status("GET /api/models/a%5Cb HTTP/1.1\r\nHost: a"), 404);
        assert_eq!(
            status("GET /api/models/%2e%2e%2fx HTTP/1.1\r\nHost: a"),
            404
        );
        // Empty segments, bad escapes, NUL and invalid UTF-8 are refused.
        assert_eq!(status("GET /api//jobs HTTP/1.1\r\nHost: a"), 404);
        assert_eq!(status("GET /api/jobs/ HTTP/1.1\r\nHost: a"), 404);
        assert_eq!(status("GET /api/%zz HTTP/1.1\r\nHost: a"), 400);
        assert_eq!(status("GET /api/% HTTP/1.1\r\nHost: a"), 400);
        assert_eq!(status("GET /api/a%00b HTTP/1.1\r\nHost: a"), 400);
        assert_eq!(status("GET /api/%ff HTTP/1.1\r\nHost: a"), 400);
    }

    #[test]
    fn query_strings_are_decoded_and_bounded() {
        let h = parse("GET /x?name=a%20b+c&flag&k=%C8%99 HTTP/1.1\r\nHost: a").unwrap();
        assert_eq!(h.param("name"), Some("a b c"));
        assert_eq!(h.param("flag"), Some(""));
        assert_eq!(h.param("k"), Some("ș"));
        assert_eq!(status("GET /x?a=%zz HTTP/1.1\r\nHost: a"), 400);
        let many = (0..=MAX_QUERY_PARAMS)
            .map(|i| format!("k{i}=v"))
            .collect::<Vec<_>>()
            .join("&");
        assert_eq!(status(&format!("GET /x?{many} HTTP/1.1\r\nHost: a")), 400);
        let long = "a".repeat(MAX_TARGET_BYTES);
        assert_eq!(status(&format!("GET /{long} HTTP/1.1\r\nHost: a")), 414);
    }

    #[test]
    fn percent_decoding_is_exact() {
        assert_eq!(percent_decode("a%41%61", false).as_deref(), Some("aAa"));
        assert_eq!(percent_decode("a+b", false).as_deref(), Some("a+b"));
        assert_eq!(percent_decode("a+b", true).as_deref(), Some("a b"));
        assert_eq!(percent_decode("%2b", true).as_deref(), Some("+"));
        assert!(percent_decode("%4", false).is_none());
        assert!(percent_decode("%g0", false).is_none());
        assert!(percent_decode("%00", false).is_none());
        assert!(
            percent_decode("%c3", false).is_none(),
            "half of a UTF-8 character"
        );
    }
}
