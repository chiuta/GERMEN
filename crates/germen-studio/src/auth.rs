//! Who may talk to the Studio.
//!
//! The server listens on the loopback interface only, but that alone does not keep out (a) other
//! programs and other users of the same computer, nor (b) web pages that the browser opens, which
//! can send requests to `127.0.0.1` too, directly or through a name that they make point there
//! (DNS rebinding). Three independent checks answer that:
//!
//! * a **token**, random for every start, that only the page the user opened knows (it travels in
//!   the part of the address after `#`, which a browser never sends to any server);
//! * the **`Host`** header must name this very server, which a rebound name does not;
//! * the **`Origin`** and **`Sec-Fetch-Site`** headers, when the browser sends them, must say that
//!   the request comes from the Studio's own page.

use germen_core::integrity::sha256_hex;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

/// Length of a token in characters (256 bits as hexadecimal).
pub(crate) const TOKEN_LEN: usize = 64;

/// A fresh random token.
///
/// The randomness comes from the operating system twice over: the keys of the standard library's
/// hash maps are drawn from it (so a hash of a counter under a fresh `RandomState` cannot be
/// predicted without them), and `/dev/urandom` is read where it exists. Both, the clock and the
/// process are mixed through SHA-256, so a weakness in one of the sources does not decide.
pub(crate) fn new_token() -> String {
    let mut seed: Vec<u8> = Vec::with_capacity(128);
    for i in 0u64..8 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(i);
        seed.extend_from_slice(&hasher.finish().to_le_bytes());
    }
    if let Ok(mut file) = std::fs::File::open("/dev/urandom") {
        let mut bytes = [0u8; 32];
        if file.read_exact(&mut bytes).is_ok() {
            seed.extend_from_slice(&bytes);
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    seed.extend_from_slice(&now.to_le_bytes());
    seed.extend_from_slice(&u64::from(std::process::id()).to_le_bytes());
    let on_the_stack = 0u8;
    seed.extend_from_slice(&(std::ptr::addr_of!(on_the_stack) as usize).to_le_bytes());
    sha256_hex(&seed)
}

/// Whether `a` and `b` are equal, in a time that does not depend on where they first differ.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut difference = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        difference |= usize::from(x ^ y);
    }
    std::hint::black_box(difference) == 0
}

/// The token in an `Authorization: Bearer <token>` header, if that is what it holds.
pub(crate) fn bearer(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then_some(token)
}

/// Whether a request carrying this `Host` header was meant for this server.
pub(crate) fn host_allowed(host: &str, port: u16) -> bool {
    let host = host.to_ascii_lowercase();
    host == format!("127.0.0.1:{port}")
        || host == format!("localhost:{port}")
        || (port == 80 && (host == "127.0.0.1" || host == "localhost"))
}

/// Whether a request carrying this `Origin` header was made by the Studio's own page.
pub(crate) fn origin_allowed(origin: &str, port: u16) -> bool {
    let origin = origin.to_ascii_lowercase();
    origin == format!("http://127.0.0.1:{port}")
        || origin == format!("http://localhost:{port}")
        || (port == 80 && (origin == "http://127.0.0.1" || origin == "http://localhost"))
}

/// Whether a `Sec-Fetch-Site` header (sent by browsers, never by the page itself) says that the
/// request did not come from another site.
pub(crate) fn fetch_site_allowed(value: &str) -> bool {
    value.eq_ignore_ascii_case("same-origin")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_long_hexadecimal_and_never_repeat() {
        let tokens: Vec<String> = (0..50).map(|_| new_token()).collect();
        for t in &tokens {
            assert_eq!(t.len(), TOKEN_LEN);
            assert!(t.bytes().all(|b| b.is_ascii_hexdigit()));
        }
        let mut sorted = tokens.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), tokens.len());
        // Not obviously structured: every hexadecimal digit shows up in 50 × 64 characters.
        let all: String = tokens.concat();
        for digit in "0123456789abcdef".chars() {
            assert!(all.contains(digit), "no {digit} in 3200 random digits");
        }
    }

    #[test]
    fn comparison_is_exact_whatever_the_lengths() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(!constant_time_eq(b"ab", b"abc"));
        assert!(!constant_time_eq(b"abc", b""));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(
            !constant_time_eq(b"a\0", b"a"),
            "a trailing NUL is a difference"
        );
    }

    #[test]
    fn the_bearer_scheme_is_parsed() {
        assert_eq!(bearer("Bearer abc"), Some("abc"));
        assert_eq!(bearer("bearer abc"), Some("abc"));
        assert_eq!(bearer("BEARER abc"), Some("abc"));
        assert_eq!(bearer("Basic abc"), None);
        assert_eq!(bearer("Bearerabc"), None);
        assert_eq!(bearer(""), None);
        assert_eq!(bearer("Bearer"), None);
        // Extra spaces become part of the token, which then does not match.
        assert_eq!(bearer("Bearer  abc"), Some(" abc"));
    }

    #[test]
    fn only_this_server_is_a_valid_host() {
        assert!(host_allowed("127.0.0.1:7870", 7870));
        assert!(host_allowed("localhost:7870", 7870));
        assert!(host_allowed("LOCALHOST:7870", 7870));
        for wrong in [
            "127.0.0.1:7871",
            "127.0.0.1",
            "localhost",
            "evil.example:7870",
            "127.0.0.1.evil.example:7870",
            "evil.example",
            "[::1]:7870",
            "0.0.0.0:7870",
            "127.0.0.1:7870.evil.example",
            "127.0.0.1:7870 ",
            "",
        ] {
            assert!(!host_allowed(wrong, 7870), "accepted {wrong:?}");
        }
        assert!(
            host_allowed("127.0.0.1", 80),
            "port 80 is the default and may be left out"
        );
        assert!(!host_allowed("127.0.0.1", 8080));
    }

    #[test]
    fn only_the_studio_page_is_a_valid_origin() {
        assert!(origin_allowed("http://127.0.0.1:7870", 7870));
        assert!(origin_allowed("http://localhost:7870", 7870));
        for wrong in [
            "null",
            "http://127.0.0.1:7871",
            "https://127.0.0.1:7870",
            "http://evil.example",
            "http://evil.example:7870",
            "http://127.0.0.1:7870.evil.example",
            "http://127.0.0.1:7870/",
            "",
        ] {
            assert!(!origin_allowed(wrong, 7870), "accepted {wrong:?}");
        }
    }

    #[test]
    fn only_same_origin_fetches_pass() {
        assert!(fetch_site_allowed("same-origin"));
        assert!(fetch_site_allowed("Same-Origin"));
        for wrong in ["cross-site", "same-site", "none", ""] {
            assert!(!fetch_site_allowed(wrong), "accepted {wrong:?}");
        }
    }
}
