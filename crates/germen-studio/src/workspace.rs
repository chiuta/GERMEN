//! The folder the Studio keeps its things in: models, texts, settings.
//!
//! ```text
//! <workspace>/models/<name>/   one checkpoint per folder (what `germen train --out` writes)
//! <workspace>/data/            texts that were uploaded or pasted
//! <workspace>/settings.json    the choices made in the interface
//! ```
//!
//! Everything the page sends that names something on disk goes through this module, which is
//! where a name stops being text from the network and becomes a path.

use crate::http::ApiError;
use germen_core::checkpoint;
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Most entries a folder listing returns.
const MAX_LISTED: usize = 3000;
/// Largest `settings.json` that is written.
const MAX_SETTINGS_BYTES: usize = 8 * 1024;
/// Interface languages.
pub(crate) const LANGUAGES: [&str; 8] = ["en", "ro", "fr", "it", "es", "pt", "de", "hu"];

/// The workspace folder.
#[derive(Debug, Clone)]
pub(crate) struct Workspace {
    root: PathBuf,
}

/// Whether `name` is acceptable as the name of a model folder: 1–64 letters, digits, `-`, `_`
/// and `.`, starting with a letter or digit, not ending in a dot, and not a name that Windows
/// reserves for devices. Nothing in it can mean "another folder" on any system.
pub(crate) fn valid_model_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    (1..=64).contains(&bytes.len())
        && bytes[0].is_ascii_alphanumeric()
        && bytes
            .iter()
            .all(|&c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        && !name.ends_with('.')
        && !is_reserved_device_name(name)
}

/// Names that Windows treats as devices, with or without an extension (`CON`, `nul.txt`, `COM3`).
fn is_reserved_device_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    match stem.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" => true,
        s if s.len() == 4 && (s.starts_with("COM") || s.starts_with("LPT")) => {
            matches!(s.as_bytes()[3], b'1'..=b'9')
        }
        _ => false,
    }
}

/// A name for an uploaded text: the last component of what the browser said, with everything but
/// letters, digits and a few marks replaced, never empty, never hidden, never a device name.
pub(crate) fn sanitize_file_name(name: &str) -> String {
    let last = name.rsplit(['/', '\\']).next().unwrap_or("");
    let cleaned: String = last
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '.' | '-' | '_' | '(' | ')') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(|c| c == '.' || c == ' ');
    let mut out: String = if trimmed.is_empty() {
        "text.txt".to_string()
    } else {
        trimmed.chars().take(100).collect()
    };
    let out_trimmed = out.trim_end_matches(['.', ' ']).to_string();
    out = if out_trimmed.is_empty() {
        "text.txt".to_string()
    } else {
        out_trimmed
    };
    if is_reserved_device_name(&out) {
        out.insert(0, '_');
    }
    out
}

impl Workspace {
    /// Opens (and creates, if need be) the workspace at `root`.
    pub(crate) fn open(root: PathBuf) -> Result<Workspace, String> {
        let root = std::path::absolute(&root)
            .map_err(|e| format!("nu pot stabili calea {}: {e}", root.display()))?;
        for dir in [root.clone(), root.join("models"), root.join("data")] {
            fs::create_dir_all(&dir)
                .map_err(|e| format!("nu pot crea folderul {}: {e}", dir.display()))?;
        }
        Ok(Workspace { root })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    pub(crate) fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    fn settings_path(&self) -> PathBuf {
        self.root.join("settings.json")
    }

    /// The folder of the model called `name` (which need not exist yet).
    pub(crate) fn model_dir(&self, name: &str) -> Result<PathBuf, ApiError> {
        if !valid_model_name(name) {
            return Err(ApiError::new(
                400,
                "bad_name",
                "numele modelului: 1–64 de litere ASCII, cifre, „-”, „_” sau „.”",
            ));
        }
        Ok(self.models_dir().join(name))
    }

    /// A model that must already exist: a name in the workspace, or a folder (absolute path) that
    /// the user picked elsewhere.
    pub(crate) fn existing_model(&self, reference: &str) -> Result<PathBuf, ApiError> {
        let dir = if valid_model_name(reference) {
            self.models_dir().join(reference)
        } else if Path::new(reference).is_absolute() {
            PathBuf::from(reference)
        } else {
            return Err(ApiError::new(
                400,
                "bad_name",
                "modelul se alege din listă sau cu un folder (cale absolută)",
            ));
        };
        if !dir.is_dir() || !checkpoint::exists(&dir) {
            return Err(ApiError::new(
                404,
                "model_missing",
                format!("în {} nu e niciun model", dir.display()),
            )
            .with(json!({ "path": dir.display().to_string() })));
        }
        Ok(dir)
    }

    /// Summaries of the models in `models/`, newest first.
    pub(crate) fn list_models(&self) -> Vec<Value> {
        let Ok(entries) = fs::read_dir(self.models_dir()) else {
            return Vec::new();
        };
        let mut found: Vec<(u64, Value)> = Vec::new();
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !valid_model_name(&name) || !entry.path().is_dir() {
                continue;
            }
            let dir = entry.path();
            if !checkpoint::exists(&dir) {
                continue;
            }
            let summary = summarize(&name, &dir);
            let modified = summary["modified"].as_u64().unwrap_or(0);
            found.push((modified, summary));
        }
        found.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        found.into_iter().map(|(_, v)| v).collect()
    }

    /// The saved settings, with a default for everything not saved.
    pub(crate) fn settings(&self) -> Value {
        let mut out = default_settings();
        if let Ok(text) = fs::read_to_string(self.settings_path())
            && let Ok(Value::Object(saved)) = serde_json::from_str::<Value>(&text)
        {
            for (key, value) in saved {
                if validate_setting(&key, &value).is_ok() {
                    out[&key] = value;
                }
            }
        }
        out
    }

    /// Applies `patch` (an object with some of the settings) and saves the result.
    pub(crate) fn update_settings(&self, patch: &Value) -> Result<Value, ApiError> {
        let Some(patch) = patch.as_object() else {
            return Err(ApiError::bad_request("setările sunt un obiect JSON"));
        };
        for (key, value) in patch {
            validate_setting(key, value)
                .map_err(|reason| ApiError::field(key, format!("setarea „{key}”: {reason}")))?;
        }
        let mut current = self.settings();
        for (key, value) in patch {
            current[key] = value.clone();
        }
        let text = serde_json::to_string_pretty(&current)
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        if text.len() > MAX_SETTINGS_BYTES {
            return Err(ApiError::new(
                413,
                "too_large",
                "setările ocupă prea mult loc",
            ));
        }
        let tmp = self.root.join("settings.json.tmp");
        fs::write(&tmp, &text)
            .and_then(|()| fs::rename(&tmp, self.settings_path()))
            .map_err(|e| ApiError::new(500, "io", format!("nu pot salva setările: {e}")))?;
        Ok(current)
    }

    /// Creates a new, empty file in `data/` for an upload, named after `wanted` (made safe, and
    /// numbered when the name is taken). Returns the open file and its path.
    pub(crate) fn new_data_file(&self, wanted: &str) -> Result<(File, PathBuf), ApiError> {
        let name = sanitize_file_name(wanted);
        let (stem, ext) = match name.rfind('.') {
            Some(i) if i > 0 => (&name[..i], &name[i..]),
            _ => (name.as_str(), ""),
        };
        for n in 1..=999u32 {
            let candidate = if n == 1 {
                name.clone()
            } else {
                format!("{stem}-{n}{ext}")
            };
            let path = self.data_dir().join(&candidate);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((file, path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => {
                    return Err(ApiError::new(
                        500,
                        "io",
                        format!("nu pot crea {}: {e}", path.display()),
                    ));
                }
            }
        }
        Err(ApiError::new(
            409,
            "too_many_files",
            "prea multe fișiere cu același nume în data/",
        ))
    }
}

/// What the list of models shows about one checkpoint folder.
fn summarize(name: &str, dir: &Path) -> Value {
    let modified = fs::metadata(dir.join(checkpoint::META_FILE))
        .or_else(|_| fs::metadata(dir))
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let mut out = json!({
        "name": name,
        "path": dir.display().to_string(),
        "modified": modified,
    });
    match (checkpoint::load_meta(dir), checkpoint::load_config(dir)) {
        (Ok(meta), Ok(cfg)) => {
            let bytes: u64 = meta.files.values().map(|f| f.bytes).sum();
            out["step"] = json!(meta.step);
            out["tokens_seen"] = json!(meta.tokens_seen);
            out["params"] = json!(cfg.num_params());
            out["layers"] = json!(cfg.n_layer);
            out["heads"] = json!(cfg.n_head);
            out["d_model"] = json!(cfg.d_model);
            out["context"] = json!(cfg.block_size);
            out["vocab_size"] = json!(cfg.vocab_size);
            out["tokenizer"] = json!(if cfg.vocab_size == 256 { "byte" } else { "bpe" });
            out["last_train_loss"] = json!(meta.last_train_loss);
            out["last_val_loss"] = json!(meta.last_val_loss);
            out["best_val_loss"] = json!(meta.best_val_loss);
            out["corpus_bytes"] = json!(meta.corpus_bytes);
            out["elapsed_secs"] = json!(meta.elapsed_secs);
            out["bytes"] = json!(bytes);
            out["has_manifest"] = json!(!meta.files.is_empty());
        }
        (meta, cfg) => {
            let reason = meta.err().or(cfg.err()).unwrap_or_default();
            out["error"] = json!(reason);
        }
    }
    out
}

fn default_settings() -> Value {
    json!({
        "lang": "auto",
        "theme": "auto",
        "contrast": "auto",
        "motion": "auto",
        "density": "comfortable",
        "text_scale": 1.0,
        "rail": "full",
        "notify": false,
        "sound": false,
        "form": {},
        "bench": {},
    })
}

fn one_of(value: &Value, allowed: &[&str]) -> Result<(), String> {
    match value.as_str() {
        Some(s) if allowed.contains(&s) => Ok(()),
        _ => Err(format!("una dintre: {}", allowed.join(", "))),
    }
}

/// Checks one setting; the error says what is expected.
fn validate_setting(key: &str, value: &Value) -> Result<(), String> {
    match key {
        "lang" => {
            let mut allowed = vec!["auto"];
            allowed.extend(LANGUAGES);
            one_of(value, &allowed)
        }
        "theme" => one_of(value, &["auto", "dark", "light"]),
        "contrast" => one_of(value, &["auto", "more"]),
        "motion" => one_of(value, &["auto", "reduce"]),
        "density" => one_of(value, &["comfortable", "compact"]),
        "rail" => one_of(value, &["full", "icons"]),
        "text_scale" => match value.as_f64() {
            Some(x) if (0.85..=1.6).contains(&x) => Ok(()),
            _ => Err("un număr între 0.85 și 1.6".into()),
        },
        "notify" | "sound" => value
            .is_boolean()
            .then_some(())
            .ok_or_else(|| "true sau false".into()),
        "form" => validate_flat_object(value, 48),
        "bench" => validate_bench(value),
        _ => Err("setare necunoscută".into()),
    }
}

/// An object of at most `max_keys` short keys whose values are numbers, booleans or short texts.
fn validate_flat_object(value: &Value, max_keys: usize) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Err("un obiect".into());
    };
    if object.len() > max_keys {
        return Err(format!("cel mult {max_keys} de chei"));
    }
    for (k, v) in object {
        let key_ok = !k.is_empty()
            && k.len() <= 40
            && k.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        let value_ok = match v {
            Value::Bool(_) => true,
            Value::Number(n) => n.as_f64().is_some_and(f64::is_finite),
            Value::String(s) => s.chars().count() <= 512,
            _ => false,
        };
        if !key_ok || !value_ok {
            return Err(format!("cheia „{k}” nu e acceptată"));
        }
    }
    Ok(())
}

/// The last benchmark per model size: `{ "nano": { "tps": 1234.5, "ms": 80.0 }, … }`.
fn validate_bench(value: &Value) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Err("un obiect".into());
    };
    if object.len() > 8 {
        return Err("cel mult 8 de dimensiuni".into());
    }
    for (preset, entry) in object {
        if germen_core::config::Preset::parse(preset).is_none() {
            return Err(format!("dimensiune necunoscută „{preset}”"));
        }
        let Some(entry) = entry.as_object() else {
            return Err("fiecare dimensiune are un obiect".into());
        };
        for (k, v) in entry {
            let ok = matches!(
                k.as_str(),
                "tps" | "ms" | "threads" | "batch" | "seq" | "at"
            ) && v.as_f64().is_some_and(|x| x.is_finite() && x >= 0.0);
            if !ok {
                return Err(format!("câmpul „{k}” nu e acceptat"));
            }
        }
    }
    Ok(())
}

/// A listing of one folder for the file picker.
pub(crate) fn list_dir(
    workspace: &Workspace,
    path: &str,
    show_hidden: bool,
) -> Result<Value, ApiError> {
    let dir = if path.is_empty() {
        workspace.data_dir()
    } else {
        PathBuf::from(path)
    };
    if !dir.is_absolute() {
        return Err(ApiError::new(
            400,
            "bad_path",
            "calea trebuie să fie absolută",
        ));
    }
    let entries = fs::read_dir(&dir).map_err(|e| {
        let (status, code) = if dir.is_file() {
            (400, "not_a_folder")
        } else {
            match e.kind() {
                std::io::ErrorKind::NotFound => (404, "bad_path"),
                std::io::ErrorKind::PermissionDenied => (403, "denied"),
                _ => (400, "bad_path"),
            }
        };
        ApiError::new(status, code, format!("nu pot citi {}: {e}", dir.display()))
            .with(json!({ "path": dir.display().to_string() }))
    })?;
    let mut items: Vec<(bool, String, Value)> = Vec::new();
    let mut truncated = false;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !show_hidden && is_hidden(&name, &entry) {
            continue;
        }
        if items.len() >= MAX_LISTED {
            truncated = true;
            break;
        }
        // `metadata` follows links, so a link to a folder is a folder; a broken link is skipped.
        let Ok(meta) = fs::metadata(entry.path()) else {
            continue;
        };
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let is_dir = meta.is_dir();
        let is_model = is_dir && checkpoint::exists(&entry.path());
        items.push((
            is_dir,
            name.to_lowercase(),
            json!({
                "name": name,
                "dir": is_dir,
                "size": if is_dir { 0 } else { meta.len() },
                "modified": modified,
                "model": is_model,
            }),
        ));
    }
    items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    Ok(json!({
        "path": dir.display().to_string(),
        "parent": dir.parent().map(|p| p.display().to_string()),
        "entries": items.into_iter().map(|(_, _, v)| v).collect::<Vec<_>>(),
        "truncated": truncated,
        "roots": roots(workspace),
    }))
}

#[cfg(windows)]
fn is_hidden(name: &str, entry: &fs::DirEntry) -> bool {
    use std::os::windows::fs::MetadataExt;
    const HIDDEN: u32 = 0x2;
    name.starts_with('.')
        || entry
            .metadata()
            .is_ok_and(|m| m.file_attributes() & HIDDEN != 0)
}

#[cfg(not(windows))]
fn is_hidden(name: &str, _entry: &fs::DirEntry) -> bool {
    name.starts_with('.')
}

/// Places to start from: the workspace folders, the home folder, and the drives or the root.
fn roots(workspace: &Workspace) -> Vec<Value> {
    let mut out = vec![
        json!({ "kind": "data", "path": workspace.data_dir().display().to_string() }),
        json!({ "kind": "models", "path": workspace.models_dir().display().to_string() }),
    ];
    if let Some(home) = std::env::home_dir() {
        out.push(json!({ "kind": "home", "path": home.display().to_string() }));
    }
    for drive in drives() {
        out.push(json!({ "kind": "drive", "path": drive }));
    }
    out
}

#[cfg(windows)]
fn drives() -> Vec<String> {
    use std::sync::mpsc;
    use std::time::Duration;
    // A disconnected network drive can make `exists` wait for many seconds, so each letter is
    // asked on its own thread and what has not answered in half a second is left out.
    let (tx, rx) = mpsc::channel();
    for letter in 'A'..='Z' {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let root = format!("{letter}:\\");
            if Path::new(&root).exists() {
                let _ = tx.send(root);
            }
        });
    }
    drop(tx);
    let deadline = std::time::Instant::now() + Duration::from_millis(500);
    let mut found = Vec::new();
    while let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) {
        match rx.recv_timeout(left) {
            Ok(root) => found.push(root),
            Err(_) => break,
        }
    }
    found.sort();
    found
}

#[cfg(not(windows))]
fn drives() -> Vec<String> {
    vec!["/".to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-ws-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn model_names_cannot_mean_another_place() {
        for ok in [
            "a",
            "model-1",
            "carte_2",
            "v1.5",
            "A9",
            "x".repeat(64).as_str(),
        ] {
            assert!(valid_model_name(ok), "{ok}");
        }
        for bad in [
            "",
            ".",
            "..",
            ".hidden",
            "-x",
            "_x",
            "a/b",
            "a\\b",
            "a b",
            "a:b",
            "ș",
            "a\0b",
            "end.",
            "con",
            "NUL",
            "nul.txt",
            "COM1",
            "lpt9",
            "x".repeat(65).as_str(),
            "../x",
            "C:",
            "a*b",
            "a?b",
        ] {
            assert!(!valid_model_name(bad), "{bad:?} must be refused");
        }
        // COM0 and LPT0 are not reserved; COM10 is not a device name either.
        assert!(valid_model_name("com0"));
        assert!(valid_model_name("com10"));
    }

    #[test]
    fn uploaded_names_are_made_safe() {
        assert_eq!(sanitize_file_name("carte.txt"), "carte.txt");
        assert_eq!(
            sanitize_file_name("Amintiri din copilărie.txt"),
            "Amintiri din copilărie.txt"
        );
        assert_eq!(sanitize_file_name("../../etc/passwd"), "passwd");
        assert_eq!(sanitize_file_name("C:\\Users\\x\\a.txt"), "a.txt");
        assert_eq!(sanitize_file_name(".bashrc"), "bashrc");
        assert_eq!(sanitize_file_name("..."), "text.txt");
        assert_eq!(sanitize_file_name(""), "text.txt");
        assert_eq!(
            sanitize_file_name("a<b>:c|d?e*f\"g.txt"),
            "a_b__c_d_e_f_g.txt"
        );
        assert_eq!(sanitize_file_name("nul.txt"), "_nul.txt");
        assert_eq!(sanitize_file_name("CON"), "_CON");
        assert_eq!(sanitize_file_name("trailing. . "), "trailing");
        assert_eq!(sanitize_file_name("a\nb\r\0c.txt"), "a_b__c.txt");
        let long = "x".repeat(300) + ".txt";
        assert_eq!(sanitize_file_name(&long).chars().count(), 100);
        for name in ["a/b", "..", "a\\..\\b", "%2e%2e"] {
            let safe = sanitize_file_name(name);
            assert!(
                !safe.contains(['/', '\\']) && safe != ".." && safe != ".",
                "{name} → {safe}"
            );
        }
    }

    #[test]
    fn uploads_never_overwrite_and_stay_inside_data() {
        let root = scratch("upload");
        let ws = Workspace::open(root.clone()).unwrap();
        let (mut f1, p1) = ws.new_data_file("carte.txt").unwrap();
        let (_f2, p2) = ws.new_data_file("carte.txt").unwrap();
        let (_f3, p3) = ws.new_data_file("../../carte.txt").unwrap();
        use std::io::Write;
        writeln!(f1, "x").unwrap();
        assert_eq!(p1.file_name().unwrap(), "carte.txt");
        assert_eq!(p2.file_name().unwrap(), "carte-2.txt");
        assert_eq!(p3.file_name().unwrap(), "carte-3.txt");
        for p in [&p1, &p2, &p3] {
            assert_eq!(p.parent().unwrap(), ws.data_dir());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn settings_are_validated_merged_and_survive_a_restart() {
        let root = scratch("settings");
        let ws = Workspace::open(root.clone()).unwrap();
        assert_eq!(ws.settings()["lang"], "auto");
        let saved = ws
            .update_settings(&json!({ "lang": "ro", "theme": "dark", "text_scale": 1.2 }))
            .unwrap();
        assert_eq!(saved["lang"], "ro");
        assert_eq!(
            saved["density"], "comfortable",
            "untouched settings keep their value"
        );
        let again = Workspace::open(root.clone()).unwrap();
        assert_eq!(again.settings()["theme"], "dark");
        assert_eq!(again.settings()["text_scale"], 1.2);

        for (key, value) in [
            ("lang", json!("xx")),
            ("lang", json!(3)),
            ("theme", json!("blue")),
            ("text_scale", json!(5.0)),
            ("text_scale", json!("1")),
            ("notify", json!("yes")),
            ("form", json!([1])),
            ("form", json!({ "Bad-Key": 1 })),
            ("form", json!({ "nested": { "a": 1 } })),
            ("form", json!({ "a": null })),
            ("bench", json!({ "huge": { "tps": 1 } })),
            ("bench", json!({ "nano": { "evil": 1 } })),
            ("bench", json!({ "nano": { "tps": -1 } })),
            ("unknown", json!(1)),
        ] {
            assert!(
                ws.update_settings(&json!({ key: value.clone() })).is_err(),
                "{key} = {value}"
            );
        }
        assert!(ws.update_settings(&json!([1, 2])).is_err());
        assert_eq!(
            ws.settings()["lang"],
            "ro",
            "a refused update changes nothing"
        );
        // A hand-edited file with a bad value falls back to the default for that key only.
        fs::write(
            root.join("settings.json"),
            r#"{"lang":"zz","theme":"light"}"#,
        )
        .unwrap();
        let s = ws.settings();
        assert_eq!(s["lang"], "auto");
        assert_eq!(s["theme"], "light");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn models_are_found_by_name_or_by_absolute_path_only() {
        let root = scratch("models");
        let ws = Workspace::open(root.clone()).unwrap();
        assert!(ws.existing_model("nothing").is_err());
        assert!(ws.existing_model("../models").is_err());
        assert!(ws.existing_model("relative/path").is_err());
        let dir = ws.model_dir("m1").unwrap();
        fs::create_dir_all(&dir).unwrap();
        assert_eq!(ws.existing_model("m1").unwrap_err().code, "model_missing");
        fs::write(dir.join("meta.json"), "{}").unwrap();
        assert_eq!(ws.existing_model("m1").unwrap(), dir);
        assert_eq!(ws.existing_model(&dir.display().to_string()).unwrap(), dir);
        assert!(ws.model_dir("../x").is_err());
        let listed = ws.list_models();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["name"], "m1");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn folder_listings_put_folders_first_and_hide_dotfiles() {
        let root = scratch("list");
        let ws = Workspace::open(root.clone()).unwrap();
        let d = ws.data_dir();
        fs::create_dir_all(d.join("Zeta")).unwrap();
        fs::create_dir_all(d.join("alfa")).unwrap();
        fs::write(d.join("b.txt"), "hello").unwrap();
        fs::write(d.join("A.txt"), "x").unwrap();
        fs::write(d.join(".secret"), "x").unwrap();
        let listing = list_dir(&ws, &d.display().to_string(), false).unwrap();
        let names: Vec<&str> = listing["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["alfa", "Zeta", "A.txt", "b.txt"]);
        let all = list_dir(&ws, &d.display().to_string(), true).unwrap();
        assert_eq!(all["entries"].as_array().unwrap().len(), 5);
        assert_eq!(
            listing["entries"][3]["size"], 5,
            "files carry their size, in bytes"
        );
        assert!(list_dir(&ws, "relative", false).is_err());
        assert!(list_dir(&ws, &d.join("nope").display().to_string(), false).is_err());
        assert!(list_dir(&ws, &d.join("b.txt").display().to_string(), false).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
