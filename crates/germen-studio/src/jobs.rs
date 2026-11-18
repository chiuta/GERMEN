//! Jobs: the long pieces of work (training, generating, measuring speed), the events they report,
//! and what a page that arrives late — or is reloaded — needs in order to catch up.
//!
//! A job writes numbered **events**. A page asks for the events after the last number it has
//! seen and waits (a long poll) until there is one. Because a page can also be opened in the middle
//! of a run, every job keeps a **snapshot** that is the sum of its events (the curve so far, the
//! samples so far, the last state) together with the number of the next event, taken under one
//! lock: the page draws the snapshot and then asks for the events from that number on, and no
//! event is lost or counted twice. The long curves are thinned out as they grow, so a run of a
//! million steps costs no more memory than one of a thousand.

use crate::http::ApiError;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Events kept for pages that are a little behind; a page further behind reloads the snapshot.
const EVENT_RING: usize = 4096;
/// Finished jobs kept in memory.
const KEPT_JOBS: usize = 12;
/// Points of a curve kept before the curve is thinned to half.
const CURVE_CAP: usize = 2400;
/// Samples of generated text kept in a snapshot.
const KEPT_SAMPLES: usize = 16;
/// Longest sample kept, in characters.
const SAMPLE_CHARS: usize = 6000;
/// Entries of the log kept in a snapshot.
const KEPT_LOG: usize = 300;
/// Longest text a generation job keeps in its snapshot, in bytes.
const MAX_TEXT_BYTES: usize = 512 * 1024;
/// Longest a poll waits for the next event.
pub(crate) const MAX_POLL_WAIT: Duration = Duration::from_secs(5);

/// Takes a lock even if a thread panicked while holding it: the data is only ever progress
/// numbers and text, and one failed job must not take the whole interface down with it.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a job does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Train,
    Generate,
    Bench,
}

impl Kind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Kind::Train => "train",
            Kind::Generate => "generate",
            Kind::Bench => "bench",
        }
    }

    /// Training and measuring use every core and run one at a time; generating is light enough to
    /// run beside them.
    fn heavy(self) -> bool {
        !matches!(self, Kind::Generate)
    }
}

/// Where a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Starting,
    Running,
    Stopping,
    Done,
    Stopped,
    Failed,
}

impl State {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            State::Starting => "starting",
            State::Running => "running",
            State::Stopping => "stopping",
            State::Done => "done",
            State::Stopped => "stopped",
            State::Failed => "failed",
        }
    }

    pub(crate) fn is_final(self) -> bool {
        matches!(self, State::Done | State::Stopped | State::Failed)
    }
}

/// How a job ended.
pub(crate) enum Outcome {
    /// Ran to the end; the value is its result.
    Done(Value),
    /// Was stopped on request; the value says how far it got.
    Stopped(Value),
    /// Could not do what it was asked.
    Failed(ApiError),
}

/// A curve kept at a bounded size: when it has more than `cap` points, every other point is dropped
/// and from then on only every second new point is kept (and so on, doubling).
struct Curve {
    columns: Vec<Vec<f64>>,
    stride: usize,
    seen: usize,
    cap: usize,
}

impl Curve {
    fn new(columns: usize, cap: usize) -> Curve {
        Curve {
            columns: vec![Vec::new(); columns],
            stride: 1,
            seen: 0,
            cap,
        }
    }

    fn push(&mut self, row: &[f64]) {
        let keep = self.seen.is_multiple_of(self.stride);
        self.seen += 1;
        if !keep {
            return;
        }
        for (column, value) in self.columns.iter_mut().zip(row) {
            column.push(*value);
        }
        if self.columns[0].len() > self.cap {
            for column in &mut self.columns {
                *column = column.iter().copied().step_by(2).collect();
            }
            self.stride *= 2;
        }
    }

    fn json(&self, names: &[&str]) -> Value {
        let mut out = serde_json::Map::new();
        for (name, column) in names.iter().zip(&self.columns) {
            out.insert(
                (*name).to_string(),
                Value::Array(column.iter().map(|&x| number(x)).collect()),
            );
        }
        Value::Object(out)
    }
}

/// A float as JSON, `null` for NaN and infinity (which JSON cannot hold).
pub(crate) fn number(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn f(event: &Value, key: &str) -> f64 {
    event[key].as_f64().unwrap_or(f64::NAN)
}

/// The sum of a training job's events.
struct TrainSum {
    stage: Value,
    start: Value,
    last: Value,
    steps: Curve,
    evals: Curve,
    samples: VecDeque<Value>,
    log: VecDeque<Value>,
}

impl TrainSum {
    fn new() -> TrainSum {
        TrainSum {
            stage: json!({ "name": "queued" }),
            start: Value::Null,
            last: Value::Null,
            steps: Curve::new(5, CURVE_CAP),
            evals: Curve::new(4, CURVE_CAP),
            samples: VecDeque::new(),
            log: VecDeque::new(),
        }
    }

    fn apply(&mut self, event: &Value, at: f64) {
        match event["type"].as_str().unwrap_or("") {
            "stage" => {
                self.stage = json!({
                    "name": event["name"],
                    "done": event["done"],
                    "total": event["total"],
                });
                if event["total"].is_null() {
                    self.log_entry(json!({ "t": at, "type": "stage", "name": event["name"] }));
                }
            }
            "start" => {
                self.start = event.clone();
                self.stage = json!({ "name": "training" });
                self.log_entry(json!({ "t": at, "type": "start" }));
            }
            "step" => {
                self.last = event.clone();
                self.steps.push(&[
                    f(event, "step"),
                    f(event, "loss"),
                    f(event, "lr"),
                    f(event, "tps"),
                    f(event, "grad_norm"),
                ]);
            }
            "eval" => {
                self.evals.push(&[
                    f(event, "step"),
                    f(event, "val_loss"),
                    f(event, "bpb"),
                    if event["best"].as_bool().unwrap_or(false) {
                        1.0
                    } else {
                        0.0
                    },
                ]);
            }
            "sample" => {
                if self.samples.len() == KEPT_SAMPLES {
                    self.samples.pop_front();
                }
                let text: String = event["text"]
                    .as_str()
                    .unwrap_or("")
                    .chars()
                    .take(SAMPLE_CHARS)
                    .collect();
                self.samples.push_back(json!({
                    "step": event["step"],
                    "text": text,
                }));
            }
            "saved" => {
                self.log_entry(json!({ "t": at, "type": "saved", "step": event["step"] }));
            }
            "recovered" => {
                self.log_entry(json!({
                    "t": at,
                    "type": "recovered",
                    "kind": event["kind"],
                    "files": event["files"],
                }));
            }
            "stopping" => self.log_entry(json!({ "t": at, "type": "stopping" })),
            _ => {}
        }
    }

    fn log_entry(&mut self, entry: Value) {
        if self.log.len() == KEPT_LOG {
            self.log.pop_front();
        }
        self.log.push_back(entry);
    }

    fn json(&self) -> Value {
        json!({
            "stage": self.stage,
            "start": self.start,
            "last": self.last,
            "steps": self.steps.json(&["step", "loss", "lr", "tps", "grad_norm"]),
            "evals": self.evals.json(&["step", "val_loss", "bpb", "best"]),
            "samples": self.samples,
            "log": self.log,
        })
    }
}

/// The sum of a generation job's events.
struct GenSum {
    loaded: Value,
    text: String,
    truncated: bool,
}

impl GenSum {
    fn apply(&mut self, event: &Value) {
        match event["type"].as_str().unwrap_or("") {
            "loaded" => self.loaded = event.clone(),
            "text" => {
                let piece = event["text"].as_str().unwrap_or("");
                if self.text.len() + piece.len() <= MAX_TEXT_BYTES {
                    self.text.push_str(piece);
                } else {
                    self.truncated = true;
                }
            }
            _ => {}
        }
    }
}

/// The sum of a benchmark job's events.
struct BenchSum {
    plan: Value,
    results: Vec<Value>,
    points: Vec<Value>,
    current: Value,
}

impl BenchSum {
    fn apply(&mut self, event: &Value) {
        match event["type"].as_str().unwrap_or("") {
            "bench_plan" => self.plan = event.clone(),
            "bench_running" => self.current = event.clone(),
            "bench_result" => {
                self.results.push(event["result"].clone());
                self.current = Value::Null;
            }
            "scaling_point" => {
                self.points.push(event.clone());
                self.current = Value::Null;
            }
            _ => {}
        }
    }
}

enum Sum {
    Train(Box<TrainSum>),
    Generate(GenSum),
    Bench(BenchSum),
}

struct Inner {
    state: State,
    next_seq: u64,
    base_seq: u64,
    events: VecDeque<Value>,
    sum: Sum,
    result: Value,
    error: Value,
    ended: Option<Instant>,
}

/// One piece of work, and everything a page may ask about it.
pub(crate) struct Job {
    pub(crate) id: u64,
    pub(crate) kind: Kind,
    label: String,
    request: Value,
    /// The checkpoint folder it writes (training only).
    dir: Option<PathBuf>,
    started: Instant,
    started_unix: u64,
    /// Set to ask the job to end soon and, if it trains, to save first.
    pub(crate) stop: AtomicBool,
    inner: Mutex<Inner>,
    changed: Condvar,
}

/// What a poll returns.
pub(crate) struct Poll {
    /// `{ "seq": n, "e": event }` in order.
    pub(crate) events: Vec<Value>,
    /// The number to ask for next.
    pub(crate) next: u64,
    pub(crate) state: State,
    /// Events the page asked for have been dropped from the ring; it must reload the snapshot.
    pub(crate) gap: bool,
}

impl Job {
    fn new(id: u64, kind: Kind, label: String, dir: Option<PathBuf>, request: Value) -> Job {
        let sum = match kind {
            Kind::Train => Sum::Train(Box::new(TrainSum::new())),
            Kind::Generate => Sum::Generate(GenSum {
                loaded: Value::Null,
                text: String::new(),
                truncated: false,
            }),
            Kind::Bench => Sum::Bench(BenchSum {
                plan: Value::Null,
                results: Vec::new(),
                points: Vec::new(),
                current: Value::Null,
            }),
        };
        Job {
            id,
            kind,
            label,
            request,
            dir,
            started: Instant::now(),
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            stop: AtomicBool::new(false),
            inner: Mutex::new(Inner {
                state: State::Starting,
                next_seq: 0,
                base_seq: 0,
                events: VecDeque::new(),
                sum,
                result: Value::Null,
                error: Value::Null,
                ended: None,
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn state(&self) -> State {
        lock(&self.inner).state
    }

    pub(crate) fn stop_requested(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Records an event: numbers it, adds it to the sum, wakes the pages that are waiting.
    pub(crate) fn emit(&self, event: Value) {
        let mut g = lock(&self.inner);
        let at = self.started.elapsed().as_secs_f64();
        match &mut g.sum {
            Sum::Train(t) => t.apply(&event, at),
            Sum::Generate(s) => s.apply(&event),
            Sum::Bench(b) => b.apply(&event),
        }
        if g.state == State::Starting && event["type"] != "stage" {
            g.state = State::Running;
        }
        let seq = g.next_seq;
        g.next_seq += 1;
        if g.events.len() == EVENT_RING {
            g.events.pop_front();
            g.base_seq += 1;
        }
        g.events.push_back(json!({ "seq": seq, "e": event }));
        drop(g);
        self.changed.notify_all();
    }

    /// Asks the job to end: it saves what it can and stops.
    pub(crate) fn request_stop(&self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            return;
        }
        let running = {
            let mut g = lock(&self.inner);
            if g.state.is_final() {
                false
            } else {
                g.state = State::Stopping;
                true
            }
        };
        if running {
            self.emit(json!({ "type": "stopping" }));
        }
    }

    fn finish(&self, outcome: Outcome) {
        let (state, result, error) = match outcome {
            Outcome::Done(v) => (State::Done, v, Value::Null),
            Outcome::Stopped(v) => (State::Stopped, v, Value::Null),
            Outcome::Failed(e) => {
                let mut error = json!({ "code": e.code, "message": e.message });
                if !e.params.is_null() {
                    error["params"] = e.params;
                }
                (State::Failed, Value::Null, error)
            }
        };
        {
            let mut g = lock(&self.inner);
            g.state = state;
            g.result = result.clone();
            g.error = error.clone();
            g.ended = Some(Instant::now());
        }
        self.emit(json!({
            "type": "end",
            "state": state.as_str(),
            "result": result,
            "error": error,
        }));
        // `emit` leaves the state alone once it is final; wake everyone who waits for the end.
        self.changed.notify_all();
    }

    /// The whole state of the job, with the number of the next event.
    pub(crate) fn snapshot(&self) -> Value {
        let g = lock(&self.inner);
        let elapsed = g
            .ended
            .unwrap_or_else(Instant::now)
            .duration_since(self.started)
            .as_secs_f64();
        let mut out = json!({
            "id": self.id,
            "kind": self.kind.as_str(),
            "state": g.state.as_str(),
            "label": self.label,
            "started": self.started_unix,
            "elapsed": elapsed,
            "seq": g.next_seq,
            "request": self.request,
            "result": g.result,
            "error": g.error,
        });
        match &g.sum {
            Sum::Train(t) => out["train"] = t.json(),
            Sum::Generate(s) => {
                out["generate"] = json!({
                    "loaded": s.loaded,
                    "text": s.text,
                    "truncated": s.truncated,
                });
            }
            Sum::Bench(b) => {
                out["bench"] = json!({
                    "plan": b.plan,
                    "results": b.results,
                    "points": b.points,
                    "current": b.current,
                });
            }
        }
        out
    }

    /// A line for the list of jobs.
    fn summary(&self) -> Value {
        let g = lock(&self.inner);
        let mut out = json!({
            "id": self.id,
            "kind": self.kind.as_str(),
            "state": g.state.as_str(),
            "label": self.label,
            "started": self.started_unix,
        });
        if let Sum::Train(t) = &g.sum {
            out["step"] = t.last["step"].clone();
            out["total"] = t.last["total"].clone();
        }
        out
    }

    /// The events from number `after` on, waiting up to `wait` for the first one.
    pub(crate) fn poll(&self, after: u64, wait: Duration) -> Poll {
        let wait = wait.min(MAX_POLL_WAIT);
        let deadline = Instant::now() + wait;
        let mut g = lock(&self.inner);
        while g.next_seq <= after && !g.state.is_final() {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            if left.is_zero() {
                break;
            }
            g = match self.changed.wait_timeout(g, left) {
                Ok((guard, _)) => guard,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        let gap = after < g.base_seq;
        let from = after.max(g.base_seq);
        let skip = usize::try_from(from - g.base_seq).unwrap_or(usize::MAX);
        let events: Vec<Value> = g.events.iter().skip(skip).cloned().collect();
        Poll {
            events,
            next: g.next_seq,
            state: g.state,
            gap,
        }
    }
}

struct HubInner {
    next_id: u64,
    jobs: VecDeque<Arc<Job>>,
    threads: Vec<JoinHandle<()>>,
    closing: bool,
}

/// All the jobs of one Studio.
pub(crate) struct Hub {
    inner: Mutex<HubInner>,
}

impl Hub {
    pub(crate) fn new() -> Hub {
        Hub {
            inner: Mutex::new(HubInner {
                next_id: 1,
                jobs: VecDeque::new(),
                threads: Vec::new(),
                closing: false,
            }),
        }
    }

    /// Starts `work` on a thread of its own, unless a job of the same weight is already running.
    ///
    /// `dir` is the checkpoint folder a training job writes; nothing else may read or repair that
    /// folder while the job lives (see [`Hub::dir_in_training`]).
    pub(crate) fn start<F>(
        &self,
        kind: Kind,
        label: String,
        dir: Option<PathBuf>,
        request: Value,
        work: F,
    ) -> Result<Arc<Job>, ApiError>
    where
        F: FnOnce(&Arc<Job>) -> Outcome + Send + 'static,
    {
        let mut hub = lock(&self.inner);
        if hub.closing {
            return Err(ApiError::new(
                503,
                "shutting_down",
                "GERMEN Studio se închide",
            ));
        }
        if let Some(running) = hub
            .jobs
            .iter()
            .find(|j| j.kind.heavy() == kind.heavy() && !j.state().is_final())
        {
            return Err(ApiError::new(
                409,
                "busy",
                format!("o altă lucrare ({}) e în curs", running.kind.as_str()),
            )
            .with(json!({ "job": running.id, "kind": running.kind.as_str() })));
        }
        let id = hub.next_id;
        hub.next_id += 1;
        let job = Arc::new(Job::new(id, kind, label, dir, request));
        hub.jobs.push_front(Arc::clone(&job));
        while hub.jobs.len() > KEPT_JOBS {
            match hub.jobs.iter().rposition(|j| j.state().is_final()) {
                Some(i) => {
                    hub.jobs.remove(i);
                }
                None => break,
            }
        }
        let runner = Arc::clone(&job);
        let spawned = std::thread::Builder::new()
            .name(format!("germen-job-{id}"))
            .spawn(move || {
                let outcome =
                    catch_unwind(AssertUnwindSafe(|| work(&runner))).unwrap_or_else(|panic| {
                        let what = panic
                            .downcast_ref::<&str>()
                            .map(|s| (*s).to_string())
                            .or_else(|| panic.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "eroare necunoscută".to_string());
                        Outcome::Failed(ApiError::new(
                            500,
                            "internal",
                            format!("lucrarea s-a oprit pe neașteptate: {what}"),
                        ))
                    });
                runner.finish(outcome);
            });
        match spawned {
            Ok(handle) => {
                hub.threads.retain(|h| !h.is_finished());
                hub.threads.push(handle);
                Ok(job)
            }
            Err(e) => {
                hub.jobs.pop_front();
                Err(ApiError::new(
                    503,
                    "internal",
                    format!("nu pot porni lucrarea: {e}"),
                ))
            }
        }
    }

    pub(crate) fn get(&self, id: u64) -> Option<Arc<Job>> {
        lock(&self.inner).jobs.iter().find(|j| j.id == id).cloned()
    }

    /// Summaries of the kept jobs, newest first.
    pub(crate) fn list(&self) -> Vec<Value> {
        let jobs: Vec<Arc<Job>> = lock(&self.inner).jobs.iter().cloned().collect();
        jobs.iter().map(|j| j.summary()).collect()
    }

    /// Whether a training job that has not ended is writing to `dir`.
    pub(crate) fn dir_in_training(&self, dir: &std::path::Path) -> bool {
        lock(&self.inner).jobs.iter().any(|j| {
            j.kind == Kind::Train && !j.state().is_final() && j.dir.as_deref() == Some(dir)
        })
    }

    /// Asks every job to end, and refuses new ones.
    pub(crate) fn close(&self) {
        let jobs: Vec<Arc<Job>> = {
            let mut hub = lock(&self.inner);
            hub.closing = true;
            hub.jobs.iter().cloned().collect()
        };
        for job in jobs {
            job.request_stop();
        }
    }

    /// Waits for the threads of all jobs (they save before they end).
    pub(crate) fn join(&self) {
        let threads = std::mem::take(&mut lock(&self.inner).threads);
        for handle in threads {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn wait_final(job: &Job) {
        let mut next = 0;
        while !job.state().is_final() {
            next = job.poll(next, Duration::from_millis(50)).next;
        }
    }

    #[test]
    fn a_curve_stays_small_and_keeps_its_shape() {
        let mut c = Curve::new(2, 100);
        for i in 0..100_000 {
            c.push(&[i as f64, (i * 2) as f64]);
            assert!(
                c.columns[0].len() <= 101,
                "len {} at {i}",
                c.columns[0].len()
            );
        }
        let x = &c.columns[0];
        let y = &c.columns[1];
        assert!(x.len() > 50, "thinned, not emptied: {}", x.len());
        assert_eq!(x[0], 0.0);
        assert!(x.windows(2).all(|w| w[1] > w[0]), "steps keep their order");
        assert!(
            x.iter().zip(y).all(|(a, b)| *b == 2.0 * *a),
            "columns stay aligned"
        );
        let gaps: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            gaps.iter().all(|g| *g == gaps[0]),
            "the points stay evenly spaced"
        );
        assert!(
            *x.last().unwrap() > 90_000.0,
            "the curve reaches almost to the end"
        );
    }

    #[test]
    fn a_curve_that_is_still_small_is_kept_whole() {
        let mut c = Curve::new(1, 100);
        for i in 0..100 {
            c.push(&[i as f64]);
        }
        assert_eq!(c.columns[0].len(), 100);
    }

    #[test]
    fn events_are_numbered_and_a_snapshot_names_the_next_one() {
        let job = Job::new(1, Kind::Train, "m".into(), None, json!({}));
        job.emit(json!({ "type": "stage", "name": "reading" }));
        job.emit(
            json!({ "type": "step", "step": 10, "total": 100, "loss": 3.5, "lr": 0.001,
                         "tps": 5000.0, "grad_norm": null }),
        );
        job.emit(json!({ "type": "eval", "step": 10, "val_loss": 3.6, "bpb": 5.2, "best": true }));
        let snap = job.snapshot();
        assert_eq!(snap["seq"], 3);
        assert_eq!(
            snap["state"], "running",
            "a stage event alone does not make a run 'running'"
        );
        assert_eq!(snap["train"]["steps"]["loss"], json!([3.5]));
        assert_eq!(snap["train"]["steps"]["grad_norm"], json!([null]));
        assert_eq!(snap["train"]["evals"]["best"], json!([1.0]));
        let p = job.poll(1, Duration::ZERO);
        assert_eq!(p.events.len(), 2);
        assert_eq!(p.events[0]["seq"], 1);
        assert_eq!(p.next, 3);
        assert!(!p.gap);
        let p = job.poll(3, Duration::ZERO);
        assert!(p.events.is_empty());
    }

    #[test]
    fn a_page_that_fell_far_behind_is_told_to_reload() {
        let job = Job::new(1, Kind::Train, "m".into(), None, json!({}));
        for i in 0..(EVENT_RING + 10) {
            job.emit(json!({ "type": "sample", "step": i, "text": "x" }));
        }
        let p = job.poll(0, Duration::ZERO);
        assert!(p.gap);
        assert_eq!(p.events.len(), EVENT_RING);
        assert_eq!(p.events[0]["seq"], 10);
        let snap = job.snapshot();
        assert_eq!(
            snap["train"]["samples"].as_array().unwrap().len(),
            KEPT_SAMPLES
        );
        assert_eq!(snap["seq"], EVENT_RING + 10);
        let p = job.poll(EVENT_RING as u64 + 5, Duration::ZERO);
        assert!(!p.gap);
        assert_eq!(p.events.len(), 5);
    }

    #[test]
    fn a_poll_waits_for_the_next_event_and_wakes_at_once() {
        let job = Arc::new(Job::new(1, Kind::Generate, "m".into(), None, json!({})));
        let writer = Arc::clone(&job);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            writer.emit(json!({ "type": "text", "text": "ab" }));
        });
        let started = Instant::now();
        let p = job.poll(0, Duration::from_secs(5));
        let waited = started.elapsed();
        assert_eq!(p.events.len(), 1);
        assert!(waited >= Duration::from_millis(100), "{waited:?}");
        assert!(
            waited < Duration::from_secs(3),
            "woken by the event, not by the timeout: {waited:?}"
        );
        handle.join().unwrap();
        // Nothing new: the poll returns empty after its wait.
        let started = Instant::now();
        let p = job.poll(1, Duration::from_millis(120));
        assert!(p.events.is_empty());
        assert!(started.elapsed() >= Duration::from_millis(100));
    }

    #[test]
    fn generated_text_is_summed_and_capped() {
        let job = Job::new(1, Kind::Generate, "m".into(), None, json!({}));
        job.emit(json!({ "type": "text", "text": "Ana " }));
        job.emit(json!({ "type": "text", "text": "are mere" }));
        assert_eq!(job.snapshot()["generate"]["text"], "Ana are mere");
        let big = "x".repeat(MAX_TEXT_BYTES);
        job.emit(json!({ "type": "text", "text": big }));
        let snap = job.snapshot();
        assert_eq!(snap["generate"]["truncated"], true);
        assert!(snap["generate"]["text"].as_str().unwrap().len() <= MAX_TEXT_BYTES);
    }

    #[test]
    fn only_one_heavy_job_runs_at_a_time_but_a_light_one_may_join_it() {
        let hub = Hub::new();
        let (release, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        let first = hub
            .start(Kind::Train, "a".into(), None, json!({}), move |_| {
                let _ = lock(&gate).recv();
                Outcome::Done(json!({}))
            })
            .unwrap();
        let busy = hub
            .start(Kind::Bench, "b".into(), None, json!({}), |_| {
                Outcome::Done(json!({}))
            })
            .err()
            .expect("a second heavy job is refused");
        assert_eq!((busy.status, busy.code), (409, "busy"));
        assert_eq!(busy.params["job"], first.id);
        let light = hub
            .start(Kind::Generate, "c".into(), None, json!({}), |_| {
                Outcome::Done(json!({}))
            })
            .expect("generation may run beside training");
        wait_final(&light);
        release.send(()).unwrap();
        wait_final(&first);
        let second = hub
            .start(Kind::Bench, "d".into(), None, json!({}), |_| {
                Outcome::Done(json!({}))
            })
            .expect("the slot is free again");
        wait_final(&second);
        hub.join();
    }

    #[test]
    fn a_job_that_panics_fails_cleanly_and_the_hub_goes_on() {
        let hub = Hub::new();
        let job = hub
            .start(Kind::Train, "p".into(), None, json!({}), |_| panic!("boom"))
            .unwrap();
        wait_final(&job);
        let snap = job.snapshot();
        assert_eq!(snap["state"], "failed");
        assert_eq!(snap["error"]["code"], "internal");
        assert!(snap["error"]["message"].as_str().unwrap().contains("boom"));
        let next = hub
            .start(Kind::Train, "q".into(), None, json!({}), |_| {
                Outcome::Done(json!(1))
            })
            .expect("the slot is free after a panic");
        wait_final(&next);
        assert_eq!(next.snapshot()["result"], 1);
        hub.join();
    }

    #[test]
    fn stopping_marks_the_job_and_ending_records_the_outcome() {
        let hub = Hub::new();
        let job = hub
            .start(Kind::Train, "s".into(), None, json!({}), |job| {
                while !job.stop_requested() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Outcome::Stopped(json!({ "step": 7 }))
            })
            .unwrap();
        job.request_stop();
        job.request_stop();
        wait_final(&job);
        let snap = job.snapshot();
        assert_eq!(snap["state"], "stopped");
        assert_eq!(snap["result"]["step"], 7);
        let all = job.poll(0, Duration::ZERO);
        let kinds: Vec<&str> = all
            .events
            .iter()
            .map(|e| e["e"]["type"].as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            ["stopping", "end"],
            "one stopping event however often it was asked"
        );
        hub.join();
    }

    #[test]
    fn a_folder_being_trained_is_reported_until_the_job_ends() {
        let hub = Hub::new();
        let dir = PathBuf::from("/some/model");
        let (release, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        let job = hub
            .start(
                Kind::Train,
                "t".into(),
                Some(dir.clone()),
                json!({}),
                move |_| {
                    let _ = lock(&gate).recv();
                    Outcome::Done(json!({}))
                },
            )
            .unwrap();
        assert!(hub.dir_in_training(&dir));
        assert!(!hub.dir_in_training(std::path::Path::new("/other")));
        release.send(()).unwrap();
        wait_final(&job);
        assert!(!hub.dir_in_training(&dir));
        hub.join();
    }

    #[test]
    fn closing_stops_the_jobs_and_refuses_new_ones() {
        let hub = Hub::new();
        let job = hub
            .start(Kind::Train, "c".into(), None, json!({}), |job| {
                while !job.stop_requested() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Outcome::Stopped(json!({}))
            })
            .unwrap();
        hub.close();
        hub.join();
        assert_eq!(job.state(), State::Stopped);
        let refused = hub
            .start(Kind::Bench, "x".into(), None, json!({}), |_| {
                Outcome::Done(json!({}))
            })
            .err()
            .unwrap();
        assert_eq!(refused.code, "shutting_down");
    }

    #[test]
    fn old_finished_jobs_are_forgotten_but_running_ones_never() {
        let hub = Hub::new();
        let (release, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        let running = hub
            .start(Kind::Train, "keep".into(), None, json!({}), move |_| {
                let _ = lock(&gate).recv();
                Outcome::Done(json!({}))
            })
            .unwrap();
        for i in 0..(KEPT_JOBS + 5) {
            let j = hub
                .start(Kind::Generate, format!("g{i}"), None, json!({}), |_| {
                    Outcome::Done(json!({}))
                })
                .unwrap();
            wait_final(&j);
        }
        let listed = hub.list();
        assert!(listed.len() <= KEPT_JOBS + 1, "{}", listed.len());
        assert!(
            hub.get(running.id).is_some(),
            "a job that is still running is kept"
        );
        release.send(()).unwrap();
        hub.join();
    }
}
