//! The routes: the page, and the JSON interface it talks to.
//!
//! Every request passes the same gates in the same order before any route sees it: the `Host`
//! header must name this server; on `/api`, the `Origin` and `Sec-Fetch-Site` headers, when sent,
//! must say the request comes from the Studio's page, and the bearer token must be right.

use crate::Shared;
use crate::auth;
use crate::http::{ApiError, Body, Conn, Head, Method, Response, read_all, send_continue};
use crate::jobs::Kind;
use crate::work::{self, BenchReq, DataReq, GenReq, MAX_PROMPT_BYTES, Mode, TrainReq};
use crate::workspace::{self, LANGUAGES};
use germen_core::checkpoint;
use germen_core::config::Preset;
use germen_core::sys;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::Duration;

/// Largest JSON body accepted.
pub(crate) const MAX_JSON_BODY: u64 = 1 << 20;

pub(crate) fn handle(shared: &Arc<Shared>, head: &Head, conn: &mut Conn) -> Response {
    match route(shared, head, conn) {
        Ok(response) => response,
        Err(error) => error.to_response(),
    }
}

fn route(shared: &Arc<Shared>, head: &Head, conn: &mut Conn) -> Result<Response, ApiError> {
    // Gate 1: the request must have been meant for this server (not for a name that someone
    // pointed at 127.0.0.1).
    if !head
        .header("host")
        .is_some_and(|host| auth::host_allowed(host, shared.port))
    {
        return Err(ApiError::new(
            421,
            "bad_host",
            "cererea nu e adresată acestui server",
        ));
    }
    let path = head.path();
    match path.as_slice() {
        [] => {
            return if head.method == Method::Get {
                Ok(Response::html(shared.page.html.clone(), &shared.page.csp))
            } else {
                Err(ApiError::new(
                    405,
                    "method_not_allowed",
                    "metodă neacceptată",
                ))
            };
        }
        ["favicon.ico"] => return Ok(Response::empty(204)),
        ["api", ..] => {}
        _ => return Err(ApiError::not_found()),
    }
    // Gate 2: where the request comes from.
    if let Some(origin) = head.header("origin")
        && !auth::origin_allowed(origin, shared.port)
    {
        return Err(ApiError::new(403, "bad_origin", "cerere din altă pagină"));
    }
    if let Some(site) = head.header("sec-fetch-site")
        && !auth::fetch_site_allowed(site)
    {
        return Err(ApiError::new(403, "bad_origin", "cerere din altă pagină"));
    }
    // Gate 3: the token.
    let allowed = head
        .header("authorization")
        .and_then(auth::bearer)
        .is_some_and(|t| auth::constant_time_eq(t.as_bytes(), shared.token.as_bytes()));
    if !allowed {
        return Err(ApiError::new(
            401,
            "unauthorized",
            "lipsește sau e greșit tokenul; deschide adresa afișată de `germen ui`",
        ));
    }

    let ws = &shared.workspace;
    let hub = &shared.hub;
    match (head.method, &path[1..]) {
        (Method::Get, ["ping"]) => Ok(Response::ok(&json!({ "ok": true }))),
        (Method::Get, ["system"]) => Ok(Response::ok(&system(shared))),
        (Method::Get, ["settings"]) => Ok(Response::ok(&ws.settings())),
        (Method::Put, ["settings"]) => {
            let patch = json_body(head, conn)?;
            Ok(Response::ok(&ws.update_settings(&patch)?))
        }
        (Method::Get, ["models"]) => Ok(Response::ok(&json!({ "models": ws.list_models() }))),
        (Method::Post, ["verify"]) => {
            let dir = model_from_body(shared, head, conn)?;
            if hub.dir_in_training(&dir) {
                return Err(model_busy(&dir));
            }
            Ok(Response::ok(&work::verify_model(&dir, shared.threads)?))
        }
        (Method::Post, ["recover"]) => {
            let dir = model_from_body(shared, head, conn)?;
            if hub.dir_in_training(&dir) {
                return Err(model_busy(&dir));
            }
            Ok(Response::ok(&work::recover_model(&dir)?))
        }
        (Method::Get, ["estimate"]) => estimate(head),
        (Method::Get, ["fs"]) => {
            let hidden = head.param("hidden") == Some("1");
            let listing = workspace::list_dir(ws, head.param("path").unwrap_or(""), hidden)?;
            Ok(Response::ok(&listing))
        }
        (Method::Post, ["upload"]) => upload(shared, head, conn),
        (Method::Get, ["jobs"]) => Ok(Response::ok(&json!({ "jobs": hub.list() }))),
        (Method::Post, ["train"]) => {
            let raw = json_body(head, conn)?;
            let req: TrainReq = serde_json::from_value(raw.clone())
                .map_err(|e| ApiError::bad_request(format!("cerere invalidă: {e}")))?;
            req.validate()?;
            let dir = ws.model_dir(&req.name)?;
            match req.mode {
                Mode::New if checkpoint::exists(&dir) => {
                    let step = checkpoint::load_meta(&dir).map_or(0, |m| m.step);
                    return Err(ApiError::new(
                        409,
                        "model_exists",
                        format!("{} conține deja un model (pasul {step})", dir.display()),
                    )
                    .with(json!({ "name": req.name, "step": step })));
                }
                Mode::Resume if !checkpoint::exists(&dir) => {
                    return Err(ApiError::new(
                        404,
                        "model_missing",
                        format!("în {} nu e niciun model de continuat", dir.display()),
                    )
                    .with(json!({ "name": req.name })));
                }
                _ => {}
            }
            if let DataReq::File { path } = &req.data
                && !std::path::Path::new(path).is_file()
            {
                return Err(ApiError::field(
                    "data.path",
                    format!("{path} nu e un fișier"),
                ));
            }
            let threads = shared.threads;
            let label = req.name.clone();
            let work_dir = dir.clone();
            let job = hub.start(Kind::Train, label, Some(dir), raw, move |job| {
                work::run_train(job, &req, &work_dir, threads)
            })?;
            Ok(Response::json(202, &json!({ "job": job.id })))
        }
        (Method::Post, ["generate"]) => {
            let raw = json_body(head, conn)?;
            let req: GenReq = serde_json::from_value(raw.clone())
                .map_err(|e| ApiError::bad_request(format!("cerere invalidă: {e}")))?;
            req.validate()?;
            let dir = ws.existing_model(&req.model)?;
            let being_trained = hub.dir_in_training(&dir);
            let label = dir
                .file_name()
                .map_or_else(|| req.model.clone(), |n| n.to_string_lossy().into_owned());
            let threads = shared.threads;
            let job = hub.start(Kind::Generate, label, None, raw, move |job| {
                work::run_generate(job, &req, &dir, being_trained, threads)
            })?;
            Ok(Response::json(202, &json!({ "job": job.id })))
        }
        (Method::Post, ["bench"]) => {
            let raw = json_body(head, conn)?;
            let req: BenchReq = serde_json::from_value(raw.clone())
                .map_err(|e| ApiError::bad_request(format!("cerere invalidă: {e}")))?;
            req.validate()?;
            let job = hub.start(Kind::Bench, req.preset.clone(), None, raw, move |job| {
                work::run_bench(job, &req)
            })?;
            Ok(Response::json(202, &json!({ "job": job.id })))
        }
        (Method::Get, ["jobs", id]) => {
            let job = find_job(shared, id)?;
            Ok(Response::ok(&job.snapshot()))
        }
        (Method::Get, ["jobs", id, "events"]) => {
            let job = find_job(shared, id)?;
            let after = number_param(head, "after", 0)?;
            let wait = number_param(head, "wait", 0)?;
            let poll = job.poll(after, Duration::from_millis(wait));
            Ok(Response::ok(&json!({
                "events": poll.events,
                "next": poll.next,
                "state": poll.state.as_str(),
                "gap": poll.gap,
            })))
        }
        (Method::Post, ["jobs", id, "stop"]) => {
            let job = find_job(shared, id)?;
            // The body of a stop is empty; read it so that the connection ends cleanly.
            let _ = read_all(&mut Body::new(conn, head.content_length.unwrap_or(0)), 1024)?;
            job.request_stop();
            Ok(Response::ok(&json!({ "state": job.state().as_str() })))
        }
        (Method::Post, ["quit"]) => {
            let _ = read_all(&mut Body::new(conn, head.content_length.unwrap_or(0)), 1024)?;
            shared.request_shutdown();
            Ok(Response::ok(&json!({ "ok": true })))
        }
        _ => Err(ApiError::not_found()),
    }
}

fn model_busy(dir: &std::path::Path) -> ApiError {
    ApiError::new(
        409,
        "model_busy",
        format!("{} e folosit de o antrenare în curs", dir.display()),
    )
}

fn find_job(shared: &Shared, id: &str) -> Result<Arc<crate::jobs::Job>, ApiError> {
    id.parse::<u64>()
        .ok()
        .and_then(|id| shared.hub.get(id))
        .ok_or_else(ApiError::not_found)
}

fn number_param(head: &Head, key: &str, default: u64) -> Result<u64, ApiError> {
    match head.param(key) {
        None => Ok(default),
        Some(text) => text
            .parse::<u64>()
            .map_err(|_| ApiError::field(key, format!("„{key}” trebuie să fie un număr întreg"))),
    }
}

/// The JSON body of the request.
fn json_body(head: &Head, conn: &mut Conn) -> Result<Value, ApiError> {
    let length = head.content_length.unwrap_or(0);
    if head
        .header("expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
        && length > 0
        && length <= MAX_JSON_BODY
    {
        send_continue(conn);
    }
    let bytes = read_all(&mut Body::new(conn, length), MAX_JSON_BODY)?;
    serde_json::from_slice(&bytes).map_err(|e| ApiError::bad_request(format!("JSON invalid: {e}")))
}

/// `{ "model": "<name or absolute path>" }` → the folder of an existing model.
fn model_from_body(
    shared: &Shared,
    head: &Head,
    conn: &mut Conn,
) -> Result<std::path::PathBuf, ApiError> {
    let body = json_body(head, conn)?;
    let Some(reference) = body["model"].as_str() else {
        return Err(ApiError::field("model", "lipsește „model”"));
    };
    shared.workspace.existing_model(reference)
}

fn system(shared: &Shared) -> Value {
    let workers = shared.threads.unwrap_or_else(sys::logical_cpus);
    let presets: Vec<Value> = [
        Preset::Nano,
        Preset::Micro,
        Preset::Mini,
        Preset::Small,
        Preset::Base,
    ]
    .iter()
    .map(|p| {
        let cfg = p.config(256);
        json!({
            "name": p.name(),
            "params": cfg.num_params(),
            "layers": cfg.n_layer,
            "heads": cfg.n_head,
            "d_model": cfg.d_model,
            "weights_bytes": cfg.num_params() * 4,
            "train_bytes": cfg.estimate_train_bytes(16, 256),
        })
    })
    .collect();
    json!({
        "version": germen_core::VERSION,
        "platform": sys::platform(),
        "logical_cpus": sys::logical_cpus(),
        "workers": workers,
        "simd": sys::simd_summary(),
        "optimised": sys::is_optimised_build(),
        "memory": {
            "available": sys::available_memory_bytes(),
            "limit": sys::memory_limit_bytes(),
        },
        "workspace": shared.workspace.root().display().to_string(),
        "models_dir": shared.workspace.models_dir().display().to_string(),
        "data_dir": shared.workspace.data_dir().display().to_string(),
        "presets": presets,
        "languages": LANGUAGES,
        "limits": {
            "upload_bytes": shared.max_upload,
            "prompt_bytes": MAX_PROMPT_BYTES,
            "max_threads": sys::MAX_THREADS,
        },
    })
}

/// `GET /api/estimate?preset=&batch=&seq=&vocab=&replicas=[&data_bytes=&tokenizer=]`
fn estimate(head: &Head) -> Result<Response, ApiError> {
    let num = |key: &str, default: usize| -> Result<usize, ApiError> {
        match head.param(key) {
            None => Ok(default),
            Some(text) => text.parse::<usize>().map_err(|_| {
                ApiError::field(key, format!("„{key}” trebuie să fie un număr întreg"))
            }),
        }
    };
    let mut out = work::estimate(
        head.param("preset").unwrap_or("mini"),
        num("batch", 16)?,
        num("seq", 256)?,
        num("vocab", 256)?,
        num("replicas", 0)?,
    )?;
    if let Some(bytes) = head.param("data_bytes") {
        let bytes: u64 = bytes
            .parse()
            .map_err(|_| ApiError::field("data_bytes", "un număr întreg"))?;
        let per_byte = if head.param("tokenizer") == Some("bpe") {
            2
        } else {
            3
        };
        let verdict = sys::check_corpus_fits(bytes, per_byte);
        out["corpus_fits"] = json!(verdict.is_ok());
        out["corpus_problem"] = json!(verdict.err());
    }
    Ok(Response::ok(&out))
}

/// `POST /api/upload?name=…` with the text itself as the body.
fn upload(shared: &Shared, head: &Head, conn: &mut Conn) -> Result<Response, ApiError> {
    let length = head.content_length.unwrap_or(0);
    if length == 0 {
        return Err(ApiError::new(400, "empty", "fișierul e gol"));
    }
    if length > shared.max_upload {
        return Err(ApiError::new(
            413,
            "too_large",
            format!(
                "fișierul are {length} de octeți, iar limita e {}",
                shared.max_upload
            ),
        )
        .with(json!({ "limit": shared.max_upload })));
    }
    let (mut file, path) = shared
        .workspace
        .new_data_file(head.param("name").unwrap_or("text.txt"))?;
    if head
        .header("expect")
        .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        send_continue(conn);
    }
    let mut body = Body::new(conn, length);
    let mut buffer = vec![0u8; 1 << 20];
    let mut written = 0u64;
    let copied = loop {
        match body.read(&mut buffer) {
            Ok(0) => break Ok(()),
            Ok(n) => {
                if let Err(e) = file.write_all(&buffer[..n]) {
                    break Err(e);
                }
                written += n as u64;
            }
            Err(e) => break Err(e),
        }
    };
    let finished = copied.and_then(|()| file.flush());
    if let Err(e) = finished {
        drop(file);
        let _ = std::fs::remove_file(&path);
        return Err(if e.kind() == std::io::ErrorKind::StorageFull {
            ApiError::new(507, "disk_full", "nu mai e loc pe disc")
        } else {
            ApiError::new(
                500,
                "upload_failed",
                format!("încărcarea s-a întrerupt: {e}"),
            )
        });
    }
    Ok(Response::ok(&json!({
        "path": path.display().to_string(),
        "name": path.file_name().map(|n| n.to_string_lossy().into_owned()),
        "bytes": written,
    })))
}
