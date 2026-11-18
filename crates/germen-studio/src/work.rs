//! What the jobs do: the same steps as `germen train`, `generate`, `bench` and `verify`, but
//! reporting through events instead of printing, and listening for a stop instead of Ctrl-C.
//!
//! Requests arrive as JSON from the page; every field is checked here, with the same limits as
//! the command line and a few more where a person at a keyboard would have noticed a typo and a
//! script would not. The engine checks them once more.

use crate::http::ApiError;
use crate::jobs::{Job, Outcome, number};
use germen_core::bench::{
    BenchParams, BenchReport, bench_model, bench_scaling, default_thread_counts,
};
use germen_core::bpe::Bpe;
use germen_core::checkpoint;
use germen_core::config::{self, GptConfig, Preset};
use germen_core::data::{Corpus, Rng, demo_corpus, sample_blocks, split_point};
use germen_core::generate::{SampleOptions, generate_until};
use germen_core::sys;
use germen_core::tokenizer::{Tokenizer, Utf8Stream};
use germen_core::train::{Event, TrainOptions, train};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: f64 = 1024.0 * 1024.0;

// ---------------------------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------------------------

/// Where the text comes from.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum DataReq {
    /// The built-in synthetic text.
    Demo {
        #[serde(default = "d_demo_mb")]
        mb: f64,
        #[serde(default = "d_one64")]
        seed: u64,
    },
    /// A file on this computer.
    File { path: String },
}

/// What to do about a model that is already in the folder.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Mode {
    /// A new model; refused if the folder holds one.
    #[default]
    New,
    /// Go on training the model in the folder.
    Resume,
    /// A new model that replaces the one in the folder at the first save.
    Overwrite,
}

fn d_demo_mb() -> f64 {
    2.0
}
fn d_one64() -> u64 {
    1
}
fn d_one() -> usize {
    1
}
fn d_preset() -> String {
    "mini".into()
}
fn d_tokenizer() -> String {
    "byte".into()
}
fn d_vocab() -> usize {
    2048
}
fn d_bpe_mb() -> f64 {
    64.0
}
fn d_batch() -> usize {
    16
}
fn d_steps() -> usize {
    2000
}
fn d_lr() -> f64 {
    6e-4
}
fn d_min_lr() -> f64 {
    0.1
}
fn d_warmup() -> usize {
    100
}
fn d_wd() -> f32 {
    0.01
}
fn d_clip() -> f32 {
    1.0
}
fn d_eval_every() -> usize {
    100
}
fn d_eval_batches() -> usize {
    8
}
fn d_save_every() -> usize {
    500
}
fn d_log_every() -> usize {
    10
}
fn d_val() -> f64 {
    0.05
}

/// A training request, as the page sends it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TrainReq {
    pub name: String,
    pub data: DataReq,
    #[serde(default = "d_preset")]
    pub preset: String,
    #[serde(default = "d_tokenizer")]
    pub tokenizer: String,
    #[serde(default = "d_vocab")]
    pub vocab: usize,
    #[serde(default = "d_bpe_mb")]
    pub bpe_sample_mb: f64,
    #[serde(default)]
    pub seq: Option<usize>,
    #[serde(default = "d_batch")]
    pub batch: usize,
    #[serde(default = "d_one")]
    pub accum: usize,
    #[serde(default = "d_steps")]
    pub steps: usize,
    #[serde(default)]
    pub replicas: usize,
    #[serde(default = "d_lr")]
    pub lr: f64,
    #[serde(default = "d_min_lr")]
    pub min_lr_ratio: f64,
    #[serde(default = "d_warmup")]
    pub warmup: usize,
    #[serde(default = "d_wd")]
    pub weight_decay: f32,
    #[serde(default = "d_clip")]
    pub grad_clip: f32,
    #[serde(default = "d_eval_every")]
    pub eval_every: usize,
    #[serde(default = "d_eval_batches")]
    pub eval_batches: usize,
    #[serde(default = "d_save_every")]
    pub save_every: usize,
    #[serde(default = "d_log_every")]
    pub log_every: usize,
    #[serde(default)]
    pub sample_every: usize,
    #[serde(default = "d_one64")]
    pub seed: u64,
    #[serde(default = "d_val")]
    pub val_fraction: f64,
    #[serde(default)]
    pub mode: Mode,
}

fn check_range(field: &str, ok: bool, expected: &str) -> Result<(), ApiError> {
    if ok {
        Ok(())
    } else {
        Err(ApiError::field(field, format!("„{field}”: {expected}")))
    }
}

impl TrainReq {
    /// Checks every number. The engine checks them again against the model.
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        check_range(
            "steps",
            (1..=10_000_000).contains(&self.steps),
            "între 1 și 10000000",
        )?;
        check_range("batch", (1..=4096).contains(&self.batch), "între 1 și 4096")?;
        check_range("accum", (1..=4096).contains(&self.accum), "între 1 și 4096")?;
        if let Some(seq) = self.seq {
            check_range(
                "seq",
                (8..=config::MAX_BLOCK_SIZE).contains(&seq),
                "între 8 și lungimea maximă a contextului",
            )?;
        }
        check_range(
            "replicas",
            self.replicas <= sys::MAX_THREADS,
            "cel mult numărul maxim de fire",
        )?;
        check_range(
            "lr",
            self.lr.is_finite() && self.lr > 0.0 && self.lr <= 1.0,
            "între 0 (exclus) și 1",
        )?;
        check_range(
            "min_lr_ratio",
            self.min_lr_ratio.is_finite() && (0.0..=1.0).contains(&self.min_lr_ratio),
            "între 0 și 1",
        )?;
        check_range(
            "warmup",
            self.warmup <= self.steps,
            "cel mult numărul de pași",
        )?;
        check_range(
            "weight_decay",
            self.weight_decay.is_finite() && (0.0..=1.0).contains(&self.weight_decay),
            "între 0 și 1",
        )?;
        check_range(
            "grad_clip",
            self.grad_clip.is_finite() && (0.0..=1000.0).contains(&self.grad_clip),
            "între 0 și 1000",
        )?;
        check_range(
            "eval_every",
            self.eval_every <= 10_000_000,
            "cel mult 10000000",
        )?;
        check_range(
            "save_every",
            self.save_every <= 10_000_000,
            "cel mult 10000000",
        )?;
        check_range(
            "sample_every",
            self.sample_every <= 10_000_000,
            "cel mult 10000000",
        )?;
        check_range(
            "log_every",
            (1..=10_000_000).contains(&self.log_every),
            "cel puțin 1",
        )?;
        check_range(
            "eval_batches",
            (1..=1000).contains(&self.eval_batches),
            "între 1 și 1000",
        )?;
        check_range(
            "val_fraction",
            self.val_fraction.is_finite() && (0.0..0.5).contains(&self.val_fraction),
            "între 0 și 0,5 (exclus)",
        )?;
        check_range(
            "preset",
            Preset::parse(&self.preset).is_some(),
            "nano, micro, mini, small sau base",
        )?;
        check_range(
            "tokenizer",
            self.tokenizer == "byte" || self.tokenizer == "bpe",
            "byte sau bpe",
        )?;
        check_range(
            "vocab",
            (257..=65536).contains(&self.vocab),
            "între 257 și 65536",
        )?;
        check_range(
            "bpe_sample_mb",
            self.bpe_sample_mb.is_finite()
                && self.bpe_sample_mb > 0.0
                && self.bpe_sample_mb <= 4096.0,
            "între 0 (exclus) și 4096",
        )?;
        match &self.data {
            DataReq::Demo { mb, .. } => check_range(
                "data.mb",
                mb.is_finite() && *mb > 0.0 && *mb <= 1024.0,
                "între 0 (exclus) și 1024",
            )?,
            DataReq::File { path } => check_range(
                "data.path",
                Path::new(path).is_absolute(),
                "o cale absolută",
            )?,
        }
        Ok(())
    }

    fn resume(&self) -> bool {
        self.mode == Mode::Resume
    }
}

/// Throttles progress events: at most one per interval.
struct Pace {
    every: Duration,
    last: Option<Instant>,
}

impl Pace {
    fn new(every: Duration) -> Pace {
        Pace { every, last: None }
    }

    fn due(&mut self) -> bool {
        let now = Instant::now();
        match self.last {
            Some(t) if now.duration_since(t) < self.every => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }
}

fn stage(job: &Job, name: &str, done: Option<usize>, total: Option<usize>) {
    job.emit(json!({ "type": "stage", "name": name, "done": done, "total": total }));
}

/// The end of a run that failed with `message`: a failure, unless the person had asked to stop
/// (the engine gives up on the work it was doing, which is not an error).
fn failed(job: &Job, message: String) -> Outcome {
    if job.stop_requested() {
        Outcome::Stopped(json!({ "interrupted": true, "before_training": true }))
    } else {
        Outcome::Failed(ApiError::engine(message))
    }
}

/// Trains. Runs on the job's own thread, inside a pool of `threads` workers.
pub(crate) fn run_train(
    job: &Arc<Job>,
    req: &TrainReq,
    dir: &Path,
    threads: Option<usize>,
) -> Outcome {
    match sys::run_in_pool(threads, || train_inner(job, req, dir)) {
        Ok(outcome) => outcome,
        Err(e) => Outcome::Failed(ApiError::engine(e)),
    }
}

fn train_inner(job: &Arc<Job>, req: &TrainReq, dir: &Path) -> Outcome {
    let resume = req.resume();
    // Text.
    stage(job, "reading", None, None);
    let text: Vec<u8> = match &req.data {
        DataReq::File { path } => {
            let per_byte = if req.tokenizer == "bpe" && !resume {
                2
            } else {
                3
            };
            match std::fs::metadata(path) {
                Ok(info) if info.is_file() => {
                    if let Err(e) = sys::check_corpus_fits(info.len(), per_byte) {
                        return failed(job, e);
                    }
                }
                Ok(_) => return failed(job, format!("{path} nu e un fișier")),
                Err(e) => return failed(job, format!("nu pot citi {path}: {e}")),
            }
            match std::fs::read(path) {
                Ok(bytes) => bytes,
                Err(e) => return failed(job, format!("nu pot citi {path}: {e}")),
            }
        }
        DataReq::Demo { mb, seed } => demo_corpus((mb * MIB) as usize, *seed).into_bytes(),
    };
    if job.stop_requested() {
        return Outcome::Stopped(json!({ "interrupted": true, "before_training": true }));
    }

    // Tokenizer: the saved one when resuming, otherwise as requested.
    let tokenizer = if resume {
        match Tokenizer::load(dir) {
            Ok(t) => t,
            Err(e) => return failed(job, e),
        }
    } else if req.tokenizer == "bpe" {
        let budget = (req.bpe_sample_mb * MIB) as usize;
        let train_part = &text[..split_point(text.len(), req.val_fraction)];
        let sample = sample_blocks(train_part, budget);
        stage(job, "bpe", Some(0), Some(req.vocab.saturating_sub(256)));
        let mut pace = Pace::new(Duration::from_millis(120));
        let learned = Bpe::train_cancellable(
            &sample,
            req.vocab,
            &mut |done, wanted| {
                if pace.due() || done == wanted {
                    stage(job, "bpe", Some(done), Some(wanted));
                }
            },
            &job.stop,
        );
        match learned {
            Ok(bpe) => Tokenizer::Bpe(bpe),
            Err(e) => return failed(job, e),
        }
    } else {
        Tokenizer::Byte
    };
    if job.stop_requested() {
        return Outcome::Stopped(json!({ "interrupted": true, "before_training": true }));
    }

    stage(job, "corpus", None, None);
    let corpus = match Corpus::build(&text, req.val_fraction, tokenizer) {
        Ok(c) => c,
        Err(e) => return failed(job, e),
    };
    drop(text);

    // Architecture: from the checkpoint when resuming, otherwise from the preset.
    let cfg: GptConfig = if resume {
        match checkpoint::load_config(dir) {
            Ok(c) => c,
            Err(e) => return failed(job, e),
        }
    } else {
        let preset = Preset::parse(&req.preset).expect("validated");
        preset
            .config(req.seq.unwrap_or(256))
            .with_vocab(corpus.tokenizer.vocab_size())
    };
    let opts = TrainOptions {
        steps: req.steps,
        batch: req.batch,
        accum: req.accum,
        seq: req.seq.unwrap_or(cfg.block_size),
        lr: req.lr,
        min_lr_ratio: req.min_lr_ratio,
        warmup: req.warmup,
        weight_decay: req.weight_decay,
        grad_clip: req.grad_clip,
        beta2: 0.95,
        eval_every: req.eval_every,
        eval_batches: req.eval_batches,
        save_every: req.save_every,
        log_every: req.log_every,
        sample_every: req.sample_every,
        seed: req.seed,
        replicas: req.replicas,
        overwrite: req.mode == Mode::Overwrite,
    };

    let ln2 = std::f32::consts::LN_2;
    let mut bytes_per_token = 1.0f64;
    let tokenizer_name = corpus.tokenizer.name();
    let corpus_bytes = corpus.bytes;
    let corpus_tokens = corpus.len();
    let report = train(
        &corpus,
        &cfg,
        &opts,
        dir,
        resume,
        &job.stop,
        &mut |event| match event {
            Event::Start {
                params,
                tokens_per_step,
                est_memory_bytes,
                from_step,
                total_steps,
                train_tokens,
                bytes_per_token: bpt,
                vocab_size,
                optimizer_restored,
                replicas,
            } => {
                bytes_per_token = bpt;
                job.emit(json!({
                    "type": "start",
                    "params": params,
                    "tokens_per_step": tokens_per_step,
                    "est_memory_bytes": est_memory_bytes,
                    "from_step": from_step,
                    "total_steps": total_steps,
                    "train_tokens": train_tokens,
                    "bytes_per_token": bpt,
                    "vocab_size": vocab_size,
                    "optimizer_restored": optimizer_restored,
                    "replicas": replicas,
                    "workers": sys::worker_threads(),
                    "resume": resume,
                    "tokenizer": tokenizer_name,
                    "corpus_bytes": corpus_bytes,
                    "corpus_tokens": corpus_tokens,
                    "layers": cfg.n_layer,
                    "heads": cfg.n_head,
                    "d_model": cfg.d_model,
                    "context": cfg.block_size,
                }));
            }
            Event::Step {
                step,
                total,
                loss,
                lr,
                grad_norm,
                tokens_per_sec,
                elapsed,
                eta,
            } => job.emit(json!({
                "type": "step",
                "step": step,
                "total": total,
                "loss": number(f64::from(loss)),
                "lr": number(lr),
                "grad_norm": number(f64::from(grad_norm)),
                "tps": number(tokens_per_sec),
                "elapsed": number(elapsed),
                "eta": number(eta),
            })),
            Event::Eval {
                step,
                val_loss,
                best,
            } => job.emit(json!({
                "type": "eval",
                "step": step,
                "val_loss": number(f64::from(val_loss)),
                "bpb": number(f64::from(val_loss / ln2) / bytes_per_token),
                "best": best,
            })),
            Event::Sample { step, text } => job.emit(json!({
                "type": "sample",
                "step": step,
                "text": text.chars().take(6000).collect::<String>(),
            })),
            Event::Saved { step, .. } => job.emit(json!({ "type": "saved", "step": step })),
            Event::Recovered(found) => {
                let (kind, files) = match found {
                    checkpoint::Recovery::Nothing => return,
                    checkpoint::Recovery::Completed(n) => ("completed", n),
                    checkpoint::Recovery::Discarded(n) => ("discarded", n),
                };
                job.emit(json!({ "type": "recovered", "kind": kind, "files": files }));
            }
        },
    );
    match report {
        Err(e) => Outcome::Failed(ApiError::engine(e)),
        Ok(r) => {
            let result = json!({
                "final_step": r.final_step,
                "steps_run": r.steps_run,
                "tokens_seen": r.tokens_seen,
                "last_train_loss": number(f64::from(r.last_train_loss)),
                "last_val_loss": r.last_val_loss.map(|v| number(f64::from(v))),
                "best_val_loss": r.best_val_loss.map(|v| number(f64::from(v))),
                "elapsed_secs": number(r.elapsed_secs),
                "tokens_per_sec": number(r.tokens_per_sec),
                "interrupted": r.interrupted,
                "bytes_per_token": bytes_per_token,
                "name": req.name,
            });
            if r.interrupted {
                Outcome::Stopped(result)
            } else {
                Outcome::Done(result)
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Generating
// ---------------------------------------------------------------------------------------------

fn d_max_new() -> usize {
    300
}
fn d_temperature() -> f32 {
    0.8
}
fn d_top_k() -> usize {
    40
}
fn d_window() -> usize {
    128
}

/// A generation request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenReq {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "d_max_new")]
    pub max_new: usize,
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    #[serde(default = "d_top_k")]
    pub top_k: usize,
    #[serde(default = "d_window")]
    pub window: usize,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// Longest prompt accepted, in bytes.
pub(crate) const MAX_PROMPT_BYTES: usize = 64 * 1024;

impl GenReq {
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        check_range(
            "prompt",
            self.prompt.len() <= MAX_PROMPT_BYTES,
            "cel mult 65536 de octeți",
        )?;
        check_range(
            "max_new",
            (1..=20_000).contains(&self.max_new),
            "între 1 și 20000",
        )?;
        check_range(
            "temperature",
            self.temperature.is_finite() && (0.0..=5.0).contains(&self.temperature),
            "între 0 și 5",
        )?;
        check_range("top_k", self.top_k <= 65_536, "cel mult 65536")?;
        check_range(
            "window",
            (2..=config::MAX_BLOCK_SIZE).contains(&self.window),
            "între 2 și lungimea maximă a contextului",
        )?;
        Ok(())
    }
}

/// Loads a model for generating. A model that is being trained is written from time to time; for
/// the moment of the save the files do not match each other, so a failed load is tried again.
fn load_model(
    dir: &Path,
    being_trained: bool,
) -> Result<(germen_core::model::Gpt, GptConfig, checkpoint::Meta), String> {
    let device = germen_core::cpu_device();
    let mut attempt = 0;
    loop {
        match checkpoint::load(dir, &device) {
            Ok(loaded) => return Ok(loaded),
            Err(_) if being_trained && attempt < 4 => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(400));
            }
            Err(e) if being_trained => {
                return Err(format!(
                    "modelul se salvează chiar acum, încearcă din nou în câteva secunde ({e})"
                ));
            }
            Err(e) => return Err(e),
        }
    }
}

/// A seed for a text the person did not ask to repeat. It stays below 2^53, so that the page, whose
/// numbers are doubles, shows it exactly and the same seed given back makes the same text.
fn fresh_seed() -> u64 {
    const PAGE_SAFE: u64 = (1 << 53) - 1;
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
        & PAGE_SAFE
}

/// Generates text, streaming it as events.
pub(crate) fn run_generate(
    job: &Arc<Job>,
    req: &GenReq,
    dir: &Path,
    being_trained: bool,
    threads: Option<usize>,
) -> Outcome {
    match sys::run_in_pool(threads, || generate_inner(job, req, dir, being_trained)) {
        Ok(outcome) => outcome,
        Err(e) => Outcome::Failed(ApiError::engine(e)),
    }
}

fn generate_inner(job: &Arc<Job>, req: &GenReq, dir: &Path, being_trained: bool) -> Outcome {
    let started = Instant::now();
    let (model, cfg, meta) = match load_model(dir, being_trained) {
        Ok(loaded) => loaded,
        Err(e) => return Outcome::Failed(ApiError::engine(e)),
    };
    let tokenizer = match Tokenizer::load(dir) {
        Ok(t) => t,
        Err(e) => return Outcome::Failed(ApiError::engine(e)),
    };
    if tokenizer.vocab_size() != cfg.vocab_size {
        return Outcome::Failed(ApiError::engine(format!(
            "tokenizatorul ({} tokeni) nu se potrivește cu modelul (vocabular {})",
            tokenizer.vocab_size(),
            cfg.vocab_size
        )));
    }
    job.emit(json!({
        "type": "loaded",
        "params": cfg.num_params(),
        "step": meta.step,
        "context": cfg.block_size,
        "vocab_size": cfg.vocab_size,
        "tokenizer": tokenizer.name(),
        "load_secs": started.elapsed().as_secs_f64(),
    }));

    let seed = req.seed.unwrap_or_else(fresh_seed);
    let opts = SampleOptions {
        max_new: req.max_new,
        temperature: req.temperature,
        top_k: req.top_k,
        window: req.window,
    };
    let device = germen_core::cpu_device();
    let mut stream = Utf8Stream::new();
    let mut pending = String::new();
    let mut last_flush = Instant::now();
    let flush = |pending: &mut String| {
        if !pending.is_empty() {
            job.emit(json!({ "type": "text", "text": std::mem::take(pending) }));
        }
    };
    let generating = Instant::now();
    let tokens = generate_until(
        &model,
        &tokenizer,
        req.prompt.as_bytes(),
        &opts,
        &mut Rng::new(seed),
        &device,
        || job.stop_requested(),
        |token| {
            pending.push_str(&stream.push_token(&tokenizer, token));
            if pending.len() >= 256 || last_flush.elapsed() >= Duration::from_millis(40) {
                flush(&mut pending);
                last_flush = Instant::now();
            }
        },
    );
    // A character cut off by the end is dropped, not shown as a replacement mark.
    flush(&mut pending);
    let secs = generating.elapsed().as_secs_f64();
    let result = json!({
        "tokens": tokens.len(),
        "secs": secs,
        "tokens_per_sec": if secs > 0.0 { tokens.len() as f64 / secs } else { 0.0 },
        "seed": seed,
        "interrupted": job.stop_requested() && tokens.len() < req.max_new,
    });
    if job.stop_requested() && tokens.len() < req.max_new {
        Outcome::Stopped(result)
    } else {
        Outcome::Done(result)
    }
}

// ---------------------------------------------------------------------------------------------
// Measuring speed
// ---------------------------------------------------------------------------------------------

fn d_bench_preset() -> String {
    "all".into()
}
fn d_bench_steps() -> usize {
    4
}
fn d_bench_warmup() -> usize {
    2
}
fn d_bench_seq() -> usize {
    256
}
fn d_bench_vocab() -> usize {
    256
}

/// A speed measurement request.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BenchReq {
    #[serde(default = "d_bench_preset")]
    pub preset: String,
    #[serde(default = "d_batch")]
    pub batch: usize,
    #[serde(default = "d_bench_seq")]
    pub seq: usize,
    #[serde(default = "d_bench_vocab")]
    pub vocab: usize,
    #[serde(default = "d_bench_steps")]
    pub steps: usize,
    #[serde(default = "d_bench_warmup")]
    pub warmup: usize,
    #[serde(default)]
    pub replicas: usize,
    /// `None`: one measurement per model. `Some([])`: a sweep over 1, 2, 4, 8… threads. `Some(list)`:
    /// a sweep over those thread counts.
    #[serde(default)]
    pub scaling: Option<Vec<usize>>,
}

impl BenchReq {
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        check_range(
            "preset",
            self.preset == "all" || Preset::parse(&self.preset).is_some(),
            "all, nano, micro, mini, small sau base",
        )?;
        check_range("batch", (1..=4096).contains(&self.batch), "între 1 și 4096")?;
        check_range(
            "seq",
            (8..=config::MAX_BLOCK_SIZE).contains(&self.seq),
            "între 8 și lungimea maximă a contextului",
        )?;
        check_range(
            "vocab",
            (256..=65536).contains(&self.vocab),
            "între 256 și 65536",
        )?;
        check_range("steps", (1..=1000).contains(&self.steps), "între 1 și 1000")?;
        check_range("warmup", self.warmup <= 1000, "cel mult 1000")?;
        check_range(
            "replicas",
            self.replicas <= sys::MAX_THREADS,
            "cel mult numărul maxim de fire",
        )?;
        if let Some(list) = &self.scaling {
            check_range(
                "scaling",
                list.len() <= 32 && list.iter().all(|n| (1..=sys::MAX_THREADS).contains(n)),
                "cel mult 32 de numere de fire, fiecare cel puțin 1",
            )?;
        }
        Ok(())
    }
}

/// Measures speed, reporting each result as it comes.
pub(crate) fn run_bench(job: &Arc<Job>, req: &BenchReq) -> Outcome {
    let scaling = req.scaling.is_some();
    let presets: Vec<Preset> = if req.preset == "all" && !scaling {
        Preset::DEFAULT_BENCH.to_vec()
    } else if req.preset == "all" {
        vec![Preset::Mini]
    } else {
        vec![Preset::parse(&req.preset).expect("validated")]
    };
    job.emit(json!({
        "type": "bench_plan",
        "presets": presets.iter().map(|p| p.name()).collect::<Vec<_>>(),
        "scaling": scaling,
        "batch": req.batch,
        "seq": req.seq,
        "steps": req.steps,
        "warmup": req.warmup,
        "optimised_build": sys::is_optimised_build(),
    }));
    let mut report = BenchReport::new();
    if let Some(list) = &req.scaling {
        let counts = if list.is_empty() {
            default_thread_counts(sys::logical_cpus())
        } else {
            list.clone()
        };
        let params = BenchParams {
            batch: req.batch,
            seq: req.seq,
            warmup: req.warmup,
            steps: req.steps,
            replicas: req.replicas,
        };
        for p in presets {
            if job.stop_requested() {
                return Outcome::Stopped(json!({ "interrupted": true }));
            }
            job.emit(json!({ "type": "bench_running", "preset": p.name() }));
            let series = bench_scaling(
                p.name(),
                &p.config(req.seq).with_vocab(req.vocab),
                &params,
                &counts,
                |point| {
                    job.emit(json!({
                        "type": "scaling_point",
                        "preset": p.name(),
                        "point": serde_json::to_value(point).unwrap_or(Value::Null),
                    }));
                },
            );
            match series {
                Ok(s) => report.scaling.push(s),
                Err(e) => return Outcome::Failed(ApiError::engine(e)),
            }
        }
    } else {
        for p in presets {
            if job.stop_requested() {
                return Outcome::Stopped(json!({ "interrupted": true }));
            }
            job.emit(json!({ "type": "bench_running", "preset": p.name() }));
            let measured = sys::run_in_pool(None, || {
                bench_model(
                    p.name(),
                    &p.config(req.seq).with_vocab(req.vocab),
                    req.batch,
                    req.seq,
                    req.warmup,
                    req.steps,
                    req.replicas,
                )
            });
            match measured {
                Ok(Ok(result)) => {
                    job.emit(json!({
                        "type": "bench_result",
                        "result": serde_json::to_value(&result).unwrap_or(Value::Null),
                    }));
                    report.results.push(result);
                }
                Ok(Err(e)) | Err(e) => return Outcome::Failed(ApiError::engine(e)),
            }
        }
    }
    let as_json = serde_json::from_str::<Value>(&report.to_json()).unwrap_or(Value::Null);
    Outcome::Done(json!({ "report": as_json, "text": report.render() }))
}

// ---------------------------------------------------------------------------------------------
// Checking a model
// ---------------------------------------------------------------------------------------------

/// Checks a checkpoint from top to bottom, as `germen verify` does.
pub(crate) fn verify_model(dir: &Path, threads: Option<usize>) -> Result<Value, ApiError> {
    let found = sys::run_in_pool(threads, || checkpoint::inspect(dir))
        .map_err(ApiError::engine)?
        .map_err(ApiError::engine)?;
    Ok(json!({
        "step": found.meta.step,
        "params": found.config.num_params(),
        "files": found.files.iter().map(|(name, bytes)| json!({ "name": name, "bytes": bytes })).collect::<Vec<_>>(),
        "optimizer": found.optimizer,
        "leftovers": found.leftovers,
    }))
}

/// Finishes or discards a save that was interrupted, as `germen verify --recover` does.
pub(crate) fn recover_model(dir: &Path) -> Result<Value, ApiError> {
    let found = checkpoint::recover_locked(dir).map_err(ApiError::engine)?;
    let (kind, files) = match found {
        checkpoint::Recovery::Nothing => ("nothing", 0),
        checkpoint::Recovery::Completed(n) => ("completed", n),
        checkpoint::Recovery::Discarded(n) => ("discarded", n),
    };
    Ok(json!({ "kind": kind, "files": files }))
}

/// What a run of this size needs, for the form that asks for it.
pub(crate) fn estimate(
    preset: &str,
    batch: usize,
    seq: usize,
    vocab: usize,
    replicas: usize,
) -> Result<Value, ApiError> {
    let preset =
        Preset::parse(preset).ok_or_else(|| ApiError::field("preset", "dimensiune necunoscută"))?;
    check_range("batch", (1..=4096).contains(&batch), "între 1 și 4096")?;
    check_range(
        "seq",
        (8..=config::MAX_BLOCK_SIZE).contains(&seq),
        "între 8 și lungimea maximă a contextului",
    )?;
    check_range(
        "vocab",
        (256..=65536).contains(&vocab),
        "între 256 și 65536",
    )?;
    let cfg = preset.config(seq).with_vocab(vocab);
    cfg.validate().map_err(ApiError::engine)?;
    let used = germen_core::train::plan_replicas(&cfg, batch, seq, replicas);
    let workers = sys::worker_threads();
    let bytes = cfg.estimate_train_bytes_replicated(batch, seq, used.min(workers));
    let fits = sys::check_memory_fits(bytes, batch, seq);
    Ok(json!({
        "params": cfg.num_params(),
        "layers": cfg.n_layer,
        "heads": cfg.n_head,
        "d_model": cfg.d_model,
        "weights_bytes": cfg.num_params() * 4,
        "train_bytes": bytes,
        "replicas": used,
        "fits": fits.is_ok(),
        "problem": fits.err(),
        "flops_per_token": cfg.train_flops_per_token(seq),
        "memory_limit_bytes": sys::memory_limit_bytes(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: Value) -> Result<TrainReq, serde_json::Error> {
        serde_json::from_value(v)
    }

    fn base() -> Value {
        json!({ "name": "m", "data": { "kind": "demo" } })
    }

    #[test]
    fn a_fresh_seed_is_one_the_page_can_hold_exactly() {
        for _ in 0..100 {
            let seed = fresh_seed();
            assert!(seed < 1 << 53, "{seed}");
            assert_eq!(seed as f64 as u64, seed);
        }
    }

    #[test]
    fn a_minimal_request_gets_the_defaults_of_the_command_line() {
        let r = parse(base()).unwrap();
        r.validate().unwrap();
        assert_eq!(r.steps, 2000);
        assert_eq!(r.batch, 16);
        assert_eq!(r.preset, "mini");
        assert_eq!(r.tokenizer, "byte");
        assert_eq!(r.warmup, 100);
        assert_eq!(r.mode, Mode::New);
        assert!(matches!(r.data, DataReq::Demo { mb, seed } if mb == 2.0 && seed == 1));
    }

    #[test]
    fn unknown_fields_and_wrong_types_are_refused_before_anything_runs() {
        for bad in [
            json!({ "name": "m", "data": { "kind": "demo" }, "evil": 1 }),
            json!({ "name": "m", "data": { "kind": "demo", "path": "/x" } }),
            json!({ "name": "m", "data": { "kind": "url", "href": "http://x" } }),
            json!({ "name": "m", "data": { "kind": "file" } }),
            json!({ "name": "m" }),
            json!({ "data": { "kind": "demo" } }),
            json!({ "name": "m", "data": { "kind": "demo" }, "steps": "10" }),
            json!({ "name": "m", "data": { "kind": "demo" }, "steps": -1 }),
            json!({ "name": "m", "data": { "kind": "demo" }, "steps": 1.5 }),
            json!({ "name": "m", "data": { "kind": "demo" }, "mode": "destroy" }),
            json!({ "name": 3, "data": { "kind": "demo" } }),
        ] {
            assert!(parse(bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn every_number_has_its_limits() {
        let ok = |patch: Value| {
            let mut v = base();
            for (k, x) in patch.as_object().unwrap() {
                v[k] = x.clone();
            }
            parse(v).unwrap().validate()
        };
        assert!(ok(json!({})).is_ok());
        for (field, value) in [
            ("steps", json!(0)),
            ("steps", json!(10_000_001)),
            ("batch", json!(0)),
            ("batch", json!(4097)),
            ("accum", json!(0)),
            ("seq", json!(7)),
            ("seq", json!(1_000_000)),
            ("replicas", json!(100_000)),
            ("lr", json!(0.0)),
            ("lr", json!(-1.0)),
            ("lr", json!(2.0)),
            ("min_lr_ratio", json!(1.5)),
            ("warmup", json!(5000)),
            ("weight_decay", json!(-0.1)),
            ("grad_clip", json!(1e9)),
            ("log_every", json!(0)),
            ("eval_batches", json!(0)),
            ("val_fraction", json!(0.5)),
            ("val_fraction", json!(-0.1)),
            ("preset", json!("huge")),
            ("tokenizer", json!("word")),
            ("vocab", json!(256)),
            ("vocab", json!(70_000)),
            ("bpe_sample_mb", json!(0.0)),
            ("bpe_sample_mb", json!(5000.0)),
            ("data", json!({ "kind": "demo", "mb": 0.0 })),
            ("data", json!({ "kind": "demo", "mb": 5000.0 })),
            ("data", json!({ "kind": "file", "path": "relative.txt" })),
        ] {
            let err = ok(json!({ field: value.clone() })).expect_err(&format!("{field} = {value}"));
            assert_eq!(err.code, "invalid_field");
            assert!(err.params["field"].as_str().unwrap().starts_with(field));
        }
        assert!(ok(json!({ "steps": 1, "warmup": 1 })).is_ok());
        assert!(ok(json!({ "val_fraction": 0.0, "eval_every": 0, "save_every": 0 })).is_ok());
    }

    #[test]
    fn generation_requests_are_bounded_too() {
        let g = |v: Value| -> Result<(), ApiError> {
            serde_json::from_value::<GenReq>(v).unwrap().validate()
        };
        assert!(g(json!({ "model": "m" })).is_ok());
        assert!(g(json!({ "model": "m", "prompt": "x".repeat(MAX_PROMPT_BYTES) })).is_ok());
        assert!(g(json!({ "model": "m", "prompt": "x".repeat(MAX_PROMPT_BYTES + 1) })).is_err());
        assert!(g(json!({ "model": "m", "max_new": 0 })).is_err());
        assert!(g(json!({ "model": "m", "max_new": 20_001 })).is_err());
        assert!(g(json!({ "model": "m", "temperature": -0.1 })).is_err());
        assert!(g(json!({ "model": "m", "temperature": 9.0 })).is_err());
        assert!(g(json!({ "model": "m", "window": 1 })).is_err());
        assert!(g(json!({ "model": "m", "top_k": 70_000 })).is_err());
        assert!(serde_json::from_value::<GenReq>(json!({ "model": "m", "x": 1 })).is_err());
    }

    #[test]
    fn benchmark_requests_are_bounded_too() {
        let b = |v: Value| -> Result<(), ApiError> {
            serde_json::from_value::<BenchReq>(v).unwrap().validate()
        };
        assert!(b(json!({})).is_ok());
        assert!(b(json!({ "preset": "nano", "scaling": [] })).is_ok());
        assert!(b(json!({ "scaling": [1, 2, 4] })).is_ok());
        assert!(b(json!({ "preset": "x" })).is_err());
        assert!(b(json!({ "scaling": [0] })).is_err());
        assert!(b(json!({ "scaling": [100_000] })).is_err());
        assert!(b(json!({ "scaling": vec![1; 33] })).is_err());
        assert!(b(json!({ "steps": 0 })).is_err());
        assert!(b(json!({ "vocab": 100 })).is_err());
    }

    #[test]
    fn the_estimate_matches_the_engine_and_refuses_what_cannot_fit() {
        let e = estimate("nano", 16, 128, 256, 1).unwrap();
        let cfg = Preset::Nano.config(128);
        assert_eq!(e["params"], cfg.num_params());
        assert_eq!(e["train_bytes"], cfg.estimate_train_bytes(16, 128));
        assert_eq!(e["weights_bytes"], cfg.num_params() * 4);
        assert_eq!(e["fits"], true);
        let huge = estimate("base", 4096, 16384, 65536, 1).unwrap();
        assert_eq!(huge["fits"], false);
        assert!(huge["problem"].is_string());
        assert!(estimate("nope", 16, 128, 256, 1).is_err());
        assert!(estimate("nano", 0, 128, 256, 1).is_err());
    }

    #[test]
    fn the_pace_lets_one_through_per_interval() {
        let mut p = Pace::new(Duration::from_millis(50));
        assert!(p.due());
        assert!(!p.due());
        std::thread::sleep(Duration::from_millis(60));
        assert!(p.due());
    }
}
