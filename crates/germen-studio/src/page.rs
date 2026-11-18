//! The page: one HTML document with its style and script inside, assembled when the Studio starts.
//!
//! Nothing is fetched from anywhere when the page loads, and the server tells the browser so: the
//! Content-Security-Policy allows exactly one inline script and one inline style, named by their
//! SHA-256, and nothing else to run or load except calls back to this server. A script injected
//! into the page by any means (a model's text, a file name) would not match the hash and would not
//! run; and the page does not build its markup from text either, see `assets/js/`.

use germen_core::integrity::Sha256;
use serde_json::{Map, Value};

const INDEX: &str = include_str!("../assets/index.html");
const STYLE: &str = include_str!("../assets/style.css");

/// The script, in the order its parts are joined. They share one scope. The first parts are the
/// interface itself.
const SCRIPT_HEAD: [&str; 7] = [
    include_str!("../assets/js/00-core.js"),
    include_str!("../assets/js/10-i18n.js"),
    include_str!("../assets/js/20-api.js"),
    include_str!("../assets/js/30-ui.js"),
    include_str!("../assets/js/40-charts.js"),
    include_str!("../assets/js/50-pages.js"),
    include_str!("../assets/js/60-train.js"),
];

/// A stand-in for the program that replays a recorded training. Only the demonstration page has it:
/// the page of the program itself never contains this code.
const SCRIPT_DEMO: &str = include_str!("../assets/js/70-demo.js");

/// The last part: starts everything.
const SCRIPT_BOOT: &str = include_str!("../assets/js/99-boot.js");

/// Which page is built.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    /// The page the program serves, which talks to the program.
    App,
    /// The same interface with a recorded training in place of the program, for looking at it
    /// without installing anything.
    Demo,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::App => "app",
            Mode::Demo => "demo",
        }
    }
}

/// The texts of the interface, one file per language.
pub(crate) const LANGUAGE_FILES: [(&str, &str); 8] = [
    ("en", include_str!("../assets/i18n/en.json")),
    ("ro", include_str!("../assets/i18n/ro.json")),
    ("fr", include_str!("../assets/i18n/fr.json")),
    ("it", include_str!("../assets/i18n/it.json")),
    ("es", include_str!("../assets/i18n/es.json")),
    ("pt", include_str!("../assets/i18n/pt.json")),
    ("de", include_str!("../assets/i18n/de.json")),
    ("hu", include_str!("../assets/i18n/hu.json")),
];

/// The finished page and the policy that goes with it.
pub(crate) struct Page {
    pub html: String,
    pub csp: String,
}

/// Standard base64, with padding.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let n = (u32::from(group[0]) << 16)
            | (u32::from(*group.get(1).unwrap_or(&0)) << 8)
            | u32::from(*group.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= group.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The CSP source expression for an inline block with exactly this text.
pub(crate) fn csp_hash(text: &str) -> String {
    let mut sha = Sha256::new();
    sha.update(text.as_bytes());
    format!("'sha256-{}'", base64(&sha.finish()))
}

/// All the languages as one JSON object, safe to place inside a `<script>` element.
fn languages_json() -> String {
    let mut all = Map::new();
    for (code, text) in LANGUAGE_FILES {
        let parsed: Value = serde_json::from_str(text)
            .unwrap_or_else(|e| panic!("assets/i18n/{code}.json is not valid JSON: {e}"));
        all.insert(code.to_string(), parsed);
    }
    let text = serde_json::to_string(&Value::Object(all)).expect("a JSON value serialises");
    // `</script` would end the element, `<!--` would open a comment inside it.
    text.replace("</", "<\\/").replace("<!--", "<\\!--")
}

/// The text with `\n` line ends. A browser reads every line end of a page as `\n`, so a hash of
/// the text with `\r\n` (the files of a Windows checkout) would not be the hash the browser takes.
fn lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

/// The whole script of the page: the texts of the languages, then the parts.
fn script_text(mode: Mode) -> String {
    let mut parts: Vec<&str> = SCRIPT_HEAD.to_vec();
    if mode == Mode::Demo {
        parts.push(SCRIPT_DEMO);
    }
    parts.push(SCRIPT_BOOT);
    lf(&format!(
        "(()=>{{'use strict';\nconst I18N={};\n{}\n}})();",
        languages_json(),
        parts.join("\n")
    ))
}

/// Assembles the page.
pub(crate) fn build() -> Page {
    assemble(Mode::App)
}

/// The demonstration page: one file that replays a recorded training, with no program behind it.
pub(crate) fn demo_html() -> String {
    assemble(Mode::Demo).html
}

fn assemble(mode: Mode) -> Page {
    let script = script_text(mode);
    let style = lf(STYLE);
    let html = lf(INDEX)
        .replace("__MODE__", mode.name())
        .replace("/*__STYLE__*/", &style)
        .replace("/*__SCRIPT__*/", &script);
    let csp = format!(
        "default-src 'none'; script-src {}; style-src {}; img-src 'self' data:; font-src 'self' data:; \
         connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; \
         object-src 'none'; require-trusted-types-for 'script'",
        csp_hash(&script),
        csp_hash(&style)
    );
    Page { html, csp }
}

#[cfg(test)]
mod tests {
    use super::*;
    use germen_core::integrity::sha256_hex;

    #[test]
    fn base64_matches_the_standard_vectors() {
        // RFC 4648 §10.
        for (plain, coded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(plain.as_bytes()), coded);
        }
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
        assert_eq!(base64(&[0xfb, 0xef, 0xbe]), "++++");
    }

    #[test]
    fn a_hash_source_is_the_base64_of_the_digest() {
        // sha256("abc") = ba7816bf 8f01cfea 414140de 5dae2223 b00361a3 96177a9c b410ff61 f20015ad
        assert_eq!(
            csp_hash("abc"),
            "'sha256-ungWv48Bz+pBQUDeXa4iI7ADYaOWF3qctBD/YfIAFa0='"
        );
        assert_eq!(sha256_hex(b"abc").len(), 64);
    }

    /// The text between `<tag ...>` and `</tag>`, for every such element.
    fn blocks<'a>(html: &'a str, tag: &str) -> Vec<&'a str> {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut out = Vec::new();
        let mut rest = html;
        while let Some(i) = rest.find(&open) {
            let after = &rest[i..];
            let start = after.find('>').expect("a tag ends") + 1;
            let end = after.find(&close).expect("an element is closed");
            out.push(&after[start..end]);
            rest = &after[end + close.len()..];
        }
        out
    }

    #[test]
    fn the_policy_names_exactly_the_inline_script_and_style_of_the_page() {
        let page = build();
        let scripts = blocks(&page.html, "script");
        let styles = blocks(&page.html, "style");
        assert_eq!(scripts.len(), 1, "one inline script, no other");
        assert_eq!(styles.len(), 1, "one inline style, no other");
        assert!(
            page.csp
                .contains(&format!("script-src {};", csp_hash(scripts[0])))
        );
        assert!(
            page.csp
                .contains(&format!("style-src {};", csp_hash(styles[0])))
        );
        assert!(
            !page.csp.contains("unsafe"),
            "no 'unsafe-inline', no 'unsafe-eval': {}",
            page.csp
        );
        for directive in [
            "default-src 'none'",
            "connect-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
            "object-src 'none'",
            "require-trusted-types-for 'script'",
        ] {
            assert!(page.csp.contains(directive), "missing {directive}");
        }
    }

    #[test]
    fn the_page_loads_nothing_from_anywhere() {
        let page = build();
        let html = &page.html;
        // No element fetches a resource: no src, no stylesheet link, no import, no font URL.
        assert!(!html.contains(" src="), "no src attributes");
        assert!(!html.contains("<link rel=\"stylesheet"));
        assert!(!html.contains("@import"));
        assert!(!html.contains("url(http"));
        for needle in ["<iframe", "<object", "<embed", "<img ", "<form", "<base"] {
            assert!(!html.contains(needle), "{needle} has no place in the page");
        }
        // No inline event handler (the policy would refuse it anyway) and no inline style attribute.
        for attribute in [
            " onclick=",
            " onload=",
            " onerror=",
            " oninput=",
            " onchange=",
            " style=",
        ] {
            assert!(!html.contains(attribute), "{attribute}");
        }
        // The two blocks cannot be closed early by their own content.
        for (what, text) in [
            ("script", blocks(html, "script")[0]),
            ("style", blocks(html, "style")[0]),
        ] {
            assert!(
                !text.to_ascii_lowercase().contains(&format!("</{what}")),
                "{what}"
            );
            assert!(!text.contains("<!--"), "{what}");
        }
    }

    /// Whether `script` calls `name(` as a function (not as the end of a longer name).
    fn calls(script: &str, name: &str) -> bool {
        let call = format!("{name}(");
        script.match_indices(&call).any(|(i, _)| {
            !script[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$')
        })
    }

    #[test]
    fn the_script_never_builds_markup_from_text() {
        // The page writes every dynamic text with `textContent`; these are the ways of writing
        // markup from a string, none of which may appear (Trusted Types would refuse them too).
        for mode in [Mode::App, Mode::Demo] {
            let script = script_text(mode);
            for sink in [
                "innerHTML",
                "outerHTML",
                "insertAdjacentHTML",
                "document.write",
                "createContextualFragment",
                "srcdoc",
                "new Function",
                "setTimeout(\"",
                "setInterval(\"",
                "importScripts",
                "localStorage",
            ] {
                assert!(!script.contains(sink), "the {mode:?} script uses {sink}");
            }
            for call in ["eval", "Function", "setTimeout('", "setInterval('"] {
                assert!(!calls(&script, call), "the {mode:?} script calls {call}(");
            }
        }
    }

    #[test]
    fn a_page_has_only_unix_line_ends_whatever_the_files_have() {
        assert_eq!(lf("a\r\nb\rc\nd"), "a\nb\nc\nd");
        assert!(!build().html.contains('\r'));
        assert!(!demo_html().contains('\r'));
    }

    #[test]
    fn the_marks_of_the_template_are_all_filled_in() {
        for (what, text) in [("style", STYLE), ("script", &script_text(Mode::Demo))] {
            for mark in ["/*__STYLE__*/", "/*__SCRIPT__*/", "__MODE__"] {
                assert!(!text.contains(mark), "the {what} contains {mark}");
            }
        }
        for html in [build().html, demo_html()] {
            assert!(!html.contains("/*__") && !html.contains("__MODE__"));
        }
    }

    #[test]
    fn only_the_demonstration_page_carries_the_demonstration() {
        let app = build().html;
        let demo = demo_html();
        assert!(app.contains("<meta name=\"germen-mode\" content=\"app\">"));
        assert!(demo.contains("<meta name=\"germen-mode\" content=\"demo\">"));
        assert!(demo.contains("function installDemo"));
        assert!(
            !app.contains("function installDemo"),
            "the program's own page has no demonstration in it"
        );
    }

    // ------------------------------------------------------------------ the texts

    fn texts(code: &str) -> Map<String, Value> {
        let (_, text) = LANGUAGE_FILES
            .iter()
            .find(|(c, _)| *c == code)
            .unwrap_or_else(|| panic!("no language {code}"));
        match serde_json::from_str::<Value>(text).expect("valid JSON") {
            Value::Object(map) => map,
            other => panic!("{code}.json holds {other}, not an object"),
        }
    }

    /// The names between braces in a text: `{name}`.
    fn placeholders(text: &str) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        let mut rest = text;
        while let Some(open) = rest.find('{') {
            let after = &rest[open + 1..];
            match after.find('}') {
                Some(close) => {
                    let name = &after[..close];
                    assert!(
                        !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'),
                        "a stray brace in {text:?}"
                    );
                    names.insert(name.to_string());
                    rest = &after[close + 1..];
                }
                None => panic!("an open brace in {text:?}"),
            }
        }
        assert!(!rest.contains('}'), "a closing brace in {text:?}");
        names
    }

    #[test]
    fn every_language_has_the_texts_of_english_and_only_those() {
        let english = texts("en");
        assert!(english.len() > 400, "English has {} texts", english.len());
        for (code, _) in LANGUAGE_FILES {
            let other = texts(code);
            let missing: Vec<_> = english.keys().filter(|k| !other.contains_key(*k)).collect();
            let extra: Vec<_> = other.keys().filter(|k| !english.contains_key(*k)).collect();
            assert!(missing.is_empty(), "{code}.json lacks {missing:?}");
            assert!(
                extra.is_empty(),
                "{code}.json has texts English has not: {extra:?}"
            );
            for (key, value) in &other {
                let text = value
                    .as_str()
                    .unwrap_or_else(|| panic!("{code}.json: {key} is not a text"));
                assert!(!text.trim().is_empty(), "{code}.json: {key} is empty");
                assert_eq!(text, text.trim(), "{code}.json: {key} has spaces around it");
                assert_eq!(
                    placeholders(text),
                    placeholders(english[key].as_str().expect("a text")),
                    "{code}.json: {key} has other {{names}} than English"
                );
            }
        }
    }

    /// Texts whose key is built at run time: the beginning, and what can follow it.
    const BUILT_KEYS: [(&str, &[&str]); 11] = [
        ("err.", &["unknown", "network", "lost"]),
        ("kind.", &["train", "bench", "generate"]),
        ("models.repair.", &["nothing", "completed", "discarded"]),
        (
            "nav.",
            &["train", "models", "generate", "speed", "settings", "about"],
        ),
        ("pick.place.", &["data", "models", "home"]),
        ("run.log.end.", &["done", "stopped", "failed"]),
        ("run.log.recovered.", &["completed", "discarded"]),
        (
            "run.stage.",
            &["queued", "reading", "bpe", "corpus", "training", "progress"],
        ),
        (
            "run.state.",
            &[
                "idle", "starting", "running", "stopping", "done", "stopped", "failed",
            ],
        ),
        ("train.size.", &["nano", "micro", "mini", "small", "base"]),
        ("train.tok.", &["byteHint", "bpeHint"]),
    ];

    /// Quoted strings of the script that look like the key of a text.
    fn key_literals(script: &str, namespaces: &[&str]) -> std::collections::BTreeSet<String> {
        let bytes = script.as_bytes();
        let mut found = std::collections::BTreeSet::new();
        let mut i = 0;
        while i < bytes.len() {
            if matches!(bytes[i], b'\'' | b'"' | b'`') {
                let quote = bytes[i];
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() && bytes[j] != quote && bytes[j] != b'\n' && bytes[j] != b'\\'
                {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == quote {
                    let literal = &script[start..j];
                    let shaped = literal.split('.').enumerate().all(|(n, part)| {
                        part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                            && (!part.is_empty() || n > 0)
                    }) && literal.contains('.');
                    let namespace = literal.split('.').next().unwrap_or("");
                    if shaped && namespaces.contains(&namespace) {
                        found.insert(literal.to_string());
                    }
                    i = j + 1;
                    continue;
                }
                i = j.max(i) + 1;
                continue;
            }
            i += 1;
        }
        found
    }

    #[test]
    fn every_text_the_script_asks_for_exists() {
        let english = texts("en");
        let namespaces: Vec<&str> = english
            .keys()
            .map(|k| k.split('.').next().expect("a key"))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        // `nav.crumbs` and `nav.rail` are element names with a class (`h('nav.rail')`), not texts.
        let not_texts = ["nav.crumbs", "nav.rail"];
        let script = script_text(Mode::Demo);
        let literals = key_literals(&script, &namespaces);
        assert!(
            literals.len() > 300,
            "found only {} keys in the script",
            literals.len()
        );
        for literal in &literals {
            if not_texts.contains(&literal.as_str()) {
                continue;
            }
            if literal.ends_with('.') {
                let (_, ends) = BUILT_KEYS
                    .iter()
                    .find(|(start, _)| start == literal)
                    .unwrap_or_else(|| {
                        panic!(
                            "the script builds keys that begin {literal:?}; add them to BUILT_KEYS"
                        )
                    });
                for end in *ends {
                    let key = format!("{literal}{end}");
                    assert!(english.contains_key(&key), "no English text for {key}");
                }
            } else {
                assert!(
                    english.contains_key(literal),
                    "no English text for {literal}"
                );
            }
        }
    }

    /// The codes after a 4xx or 5xx status in the program (`(400, "not_a_folder")`), tests left out.
    fn error_codes(source: &str) -> std::collections::BTreeSet<String> {
        let source = source.split("#[cfg(test)]").next().unwrap_or(source);
        let bytes = source.as_bytes();
        let mut codes = std::collections::BTreeSet::new();
        for i in 0..bytes.len().saturating_sub(6) {
            let status = matches!(bytes[i], b'4' | b'5')
                && bytes[i + 1].is_ascii_digit()
                && bytes[i + 2].is_ascii_digit()
                && bytes[i + 3] == b',';
            let alone = i == 0
                || !(bytes[i - 1].is_ascii_alphanumeric() || matches!(bytes[i - 1], b'_' | b'.'));
            if !(status && alone) {
                continue;
            }
            let mut j = i + 4;
            while bytes.get(j).is_some_and(u8::is_ascii_whitespace) {
                j += 1;
            }
            if bytes.get(j) != Some(&b'"') {
                continue;
            }
            let start = j + 1;
            if let Some(len) = source[start..].find('"') {
                let code = &source[start..start + len];
                if !code.is_empty() && code.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') {
                    codes.insert(code.to_string());
                }
            }
        }
        codes
    }

    #[test]
    fn every_error_the_server_can_send_has_a_text() {
        let english = texts("en");
        let mut codes = std::collections::BTreeSet::new();
        for source in [
            include_str!("api.rs"),
            include_str!("auth.rs"),
            include_str!("http.rs"),
            include_str!("jobs.rs"),
            include_str!("lib.rs"),
            include_str!("work.rs"),
            include_str!("workspace.rs"),
        ] {
            codes.extend(error_codes(source));
        }
        assert!(codes.len() >= 20, "found only {codes:?}");
        for code in &codes {
            assert!(
                english.contains_key(&format!("err.{code}")),
                "no text for the error {code}"
            );
        }
    }
}
