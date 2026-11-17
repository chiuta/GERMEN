//! Benchmark: how fast does training run on THIS machine?
//!
//! Every measured step is a complete training step (forward, backward, AdamW update) on random
//! tokens — the content of the data does not change the speed, so no corpus is needed. The
//! result is what decides which model size is realistic on a given computer.

use crate::config::GptConfig;
use crate::data::{Batch, Rng};
use crate::sys;
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams};
use serde::Serialize;
use std::fmt::Write as _;
use std::time::Instant;

/// Measurements for one model size.
#[derive(Clone, Debug, Serialize)]
pub struct BenchResult {
    /// Preset name (or `"custom"`).
    pub preset: String,
    /// Trainable parameters.
    pub params: usize,
    /// Windows per step.
    pub batch: usize,
    /// Tokens per window.
    pub seq: usize,
    /// Number of timed steps.
    pub steps: usize,
    /// Mean milliseconds per training step.
    pub ms_per_step: f64,
    /// Fastest step in milliseconds.
    pub ms_per_step_best: f64,
    /// Training throughput (tokens per second).
    pub tokens_per_sec: f64,
    /// Effective speed in GFLOPS, counting only the model's own matrix multiplications
    /// (`6 · params + attention` per token); a lower bound of what the CPU really executes.
    pub gflops: f64,
    /// Peak resident memory of the process so far in MiB (Linux only).
    pub peak_rss_mib: Option<f64>,
}

impl BenchResult {
    /// Estimated wall-clock seconds to train on `tokens` tokens at the measured speed.
    pub fn eta_secs(&self, tokens: f64) -> f64 {
        tokens / self.tokens_per_sec
    }
}

/// A full benchmark run: machine description plus one result per model size.
#[derive(Clone, Debug, Serialize)]
pub struct BenchReport {
    /// GERMEN version.
    pub germen_version: String,
    /// `"<os>/<arch>"`.
    pub platform: String,
    /// Logical processors reported by the OS.
    pub logical_cpus: usize,
    /// Worker threads used by the tensor backend.
    pub worker_threads: usize,
    /// Detected vector instruction sets.
    pub simd: String,
    /// Whether this is an optimised (release) build.
    pub optimised_build: bool,
    /// One entry per measured model.
    pub results: Vec<BenchResult>,
    /// Thread-count sweeps (`germen bench --scaling`), one series per model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scaling: Vec<ScalingSeries>,
}

/// What every measurement of a benchmark run has in common.
#[derive(Clone, Copy, Debug)]
pub struct BenchParams {
    /// Windows per step.
    pub batch: usize,
    /// Tokens per window.
    pub seq: usize,
    /// Untimed steps before measuring.
    pub warmup: usize,
    /// Timed steps.
    pub steps: usize,
}

/// One measurement of a thread-count sweep.
#[derive(Clone, Debug, Serialize)]
pub struct ScalingPoint {
    /// Worker threads of the pool this measurement ran on.
    pub threads: usize,
    /// The measurement itself.
    pub result: BenchResult,
    /// Throughput relative to the first point of the series.
    pub speedup: f64,
    /// `speedup` divided by the ratio of thread counts to the first point
    /// (1.0 = perfect scaling).
    pub efficiency: f64,
}

/// Training throughput of one model size at several thread counts.
#[derive(Clone, Debug, Serialize)]
pub struct ScalingSeries {
    /// Preset name (or `"custom"`).
    pub preset: String,
    /// Measurements, fewest threads first.
    pub points: Vec<ScalingPoint>,
}

impl ScalingSeries {
    /// The fastest measurement.
    pub fn best(&self) -> Option<&ScalingPoint> {
        self.points
            .iter()
            .max_by(|a, b| a.result.tokens_per_sec.total_cmp(&b.result.tokens_per_sec))
    }

    /// The point with the fewest threads that is still within 3 % of the fastest: beyond it,
    /// extra threads buy almost nothing.
    pub fn sensible(&self) -> Option<&ScalingPoint> {
        let best = self.best()?.result.tokens_per_sec;
        self.points
            .iter()
            .filter(|p| p.result.tokens_per_sec >= 0.97 * best)
            .min_by_key(|p| p.threads)
    }

    /// Human-readable table.
    pub fn render(&self) -> String {
        let mut s = String::new();
        if let Some(first) = self.points.first() {
            let r = &first.result;
            let _ = writeln!(
                s,
                "Scalare pe fire — model {} ({:.2}M parametri), lot {}×{}",
                self.preset,
                r.params as f64 / 1e6,
                r.batch,
                r.seq
            );
        }
        let _ = writeln!(
            s,
            "{:>6} {:>9} {:>10} {:>11} {:>10}",
            "fire", "ms/pas", "tokeni/s", "accelerare", "eficiență"
        );
        for p in &self.points {
            let _ = writeln!(
                s,
                "{:>6} {:>9.0} {:>10.0} {:>11} {:>10}",
                p.threads,
                p.result.ms_per_step,
                p.result.tokens_per_sec,
                format!("{:.2}×", p.speedup),
                format!("{:.0}%", p.efficiency * 100.0),
            );
        }
        if let (Some(best), Some(sensible)) = (self.best(), self.sensible()) {
            let _ = writeln!(
                s,
                "Cel mai rapid: {} fire ({:.0} tokeni/s).",
                best.threads, best.result.tokens_per_sec
            );
            if sensible.threads < best.threads {
                let _ = writeln!(
                    s,
                    "Cu {} fire se ajunge la {:.0}% din acea viteză; peste atât, firele în plus aduc puțin.",
                    sensible.threads,
                    100.0 * sensible.result.tokens_per_sec / best.result.tokens_per_sec
                );
            }
        }
        s
    }
}

/// Thread counts for a sweep up to `max`: 1, 2, 4, 8, … and `max` itself.
pub fn default_thread_counts(max: usize) -> Vec<usize> {
    let max = max.max(1);
    let mut counts: Vec<usize> = std::iter::successors(Some(1usize), |&n| n.checked_mul(2))
        .take_while(|&n| n < max)
        .collect();
    counts.push(max);
    counts
}

impl BenchReport {
    /// Creates an empty report describing the current machine.
    pub fn new() -> Self {
        BenchReport {
            germen_version: crate::VERSION.to_string(),
            platform: sys::platform(),
            logical_cpus: sys::logical_cpus(),
            worker_threads: sys::worker_threads(),
            simd: sys::simd_summary(),
            optimised_build: sys::is_optimised_build(),
            results: Vec::new(),
            scaling: Vec::new(),
        }
    }

    /// Pretty JSON, handy for sharing measurements.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Human-readable table with time estimates for training on 100 M and 1 B tokens.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "GERMEN {} — benchmark de antrenare", self.germen_version);
        let _ = writeln!(
            s,
            "Mașina: {}, {} procesoare logice, {} fire de lucru, {}",
            self.platform, self.logical_cpus, self.worker_threads, self.simd
        );
        if !self.optimised_build {
            let _ = writeln!(
                s,
                "ATENȚIE: acesta e un build de depanare (debug) — de zeci de ori mai lent. \
                 Folosiți `cargo build --release`."
            );
        }
        if !self.results.is_empty() {
            let _ = writeln!(s);
            let _ = writeln!(
                s,
                "{:<7} {:>9} {:>11} {:>9} {:>10} {:>8} {:>10} {:>13} {:>13}",
                "model",
                "parametri",
                "lot×context",
                "ms/pas",
                "tokeni/s",
                "GFLOPS",
                "RAM vârf",
                "100M tokeni",
                "1G tokeni"
            );
            for r in &self.results {
                let ram = r
                    .peak_rss_mib
                    .map_or("—".to_string(), |m| format!("{:.0} MiB", m));
                let _ = writeln!(
                    s,
                    "{:<7} {:>8.2}M {:>11} {:>9.0} {:>10.0} {:>8.1} {:>10} {:>13} {:>13}",
                    r.preset,
                    r.params as f64 / 1e6,
                    format!("{}×{}", r.batch, r.seq),
                    r.ms_per_step,
                    r.tokens_per_sec,
                    r.gflops,
                    ram,
                    sys::human_duration(r.eta_secs(1e8)),
                    sys::human_duration(r.eta_secs(1e9)),
                );
            }
            let _ = writeln!(s);
            let _ = writeln!(
                s,
                "Ultimele două coloane sunt estimări liniare din viteza măsurată (fără validare/salvări)."
            );
        }
        for series in &self.scaling {
            let _ = writeln!(s);
            s.push_str(&series.render());
        }
        s
    }
}

impl Default for BenchReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Times `steps` full training steps of `cfg` (after `warmup` untimed ones).
///
/// `name` only labels the result. Returns an error if the batch does not fit the context or the
/// loss turns non-finite.
pub fn bench_model(
    name: &str,
    cfg: &GptConfig,
    batch: usize,
    seq: usize,
    warmup: usize,
    steps: usize,
) -> Result<BenchResult, String> {
    cfg.validate()?;
    if batch == 0 || steps == 0 {
        return Err("batch și steps trebuie să fie cel puțin 1".into());
    }
    if seq < 2 || seq > cfg.block_size {
        return Err(format!(
            "seq ({seq}) trebuie să fie între 2 și {}",
            cfg.block_size
        ));
    }

    sys::check_memory_fits(cfg.estimate_train_bytes(batch, seq), batch, seq)?;

    let device = crate::train_device();
    device.seed(1);
    let mut model = cfg.init(&device);
    let mut optim = AdamWConfig::new()
        .with_beta_2(0.95)
        .with_epsilon(1e-8)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut rng = Rng::new(7);

    let mut times = Vec::with_capacity(steps);
    for i in 0..warmup + steps {
        let (x, y) = Batch::random(batch, seq, cfg.vocab_size, &mut rng).to_tensors(&device);
        let started = Instant::now();
        let loss = model.loss(x, y);
        let value = loss.clone().into_scalar::<f32>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optim.step(1e-4, model, grads);
        let dt = started.elapsed().as_secs_f64();
        if !value.is_finite() {
            return Err(format!("pierderea a devenit {value} în benchmark"));
        }
        if i >= warmup {
            times.push(dt);
        }
    }

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let best = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let tokens = (batch * seq) as f64;
    Ok(BenchResult {
        preset: name.to_string(),
        params: cfg.num_params(),
        batch,
        seq,
        steps,
        ms_per_step: mean * 1e3,
        ms_per_step_best: best * 1e3,
        tokens_per_sec: tokens / mean,
        gflops: cfg.train_flops_per_token(seq) * tokens / mean / 1e9,
        peak_rss_mib: sys::peak_rss_mib(),
    })
}

/// Runs [`bench_model`] once per entry of `threads`, each time on a fresh pool of exactly that
/// many worker threads, and relates every result to the one with the fewest threads.
///
/// `on_point` is called after each measurement (for progress output).
pub fn bench_scaling(
    name: &str,
    cfg: &GptConfig,
    params: &BenchParams,
    threads: &[usize],
    mut on_point: impl FnMut(&ScalingPoint),
) -> Result<ScalingSeries, String> {
    let mut counts = threads.to_vec();
    counts.sort_unstable();
    counts.dedup();
    if counts.first().is_none_or(|&n| n == 0) {
        return Err("numărul de fire trebuie să fie cel puțin 1".into());
    }

    let mut points: Vec<ScalingPoint> = Vec::with_capacity(counts.len());
    for &n in &counts {
        let pool = sys::build_pool(n)?;
        // `install` makes the pool the one every parallel loop inside the closure runs on.
        let result = pool.install(|| {
            bench_model(
                name,
                cfg,
                params.batch,
                params.seq,
                params.warmup,
                params.steps,
            )
        })?;
        let (base_speed, base_threads) = points.first().map_or((result.tokens_per_sec, n), |p| {
            (p.result.tokens_per_sec, p.threads)
        });
        let speedup = result.tokens_per_sec / base_speed;
        let point = ScalingPoint {
            threads: n,
            result,
            speedup,
            efficiency: speedup / (n as f64 / base_threads as f64),
        };
        on_point(&point);
        points.push(point);
    }
    Ok(ScalingSeries {
        preset: name.to_string(),
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    #[test]
    fn default_thread_counts_double_up_to_the_maximum() {
        assert_eq!(default_thread_counts(0), vec![1]);
        assert_eq!(default_thread_counts(1), vec![1]);
        assert_eq!(default_thread_counts(2), vec![1, 2]);
        assert_eq!(default_thread_counts(8), vec![1, 2, 4, 8]);
        assert_eq!(default_thread_counts(20), vec![1, 2, 4, 8, 16, 20]);
        assert_eq!(default_thread_counts(12), vec![1, 2, 4, 8, 12]);
    }

    #[test]
    fn scaling_sweeps_measure_each_thread_count_on_its_own_pool() {
        let cfg = Preset::Nano.config(32);
        let params = BenchParams {
            batch: 2,
            seq: 16,
            warmup: 0,
            steps: 1,
        };
        let mut seen = Vec::new();
        let series =
            bench_scaling("nano", &cfg, &params, &[2, 1, 2], |p| seen.push(p.threads)).unwrap();
        assert_eq!(seen, vec![1, 2], "sorted, without duplicates");
        assert_eq!(series.points.len(), 2);
        assert!((series.points[0].speedup - 1.0).abs() < 1e-12);
        assert!((series.points[0].efficiency - 1.0).abs() < 1e-12);
        assert!(series.points[1].speedup > 0.0);
        assert!(series.best().is_some() && series.sensible().is_some());
        assert!(series.sensible().unwrap().threads <= series.best().unwrap().threads);
        let text = series.render();
        assert!(text.contains("Scalare pe fire") && text.contains("accelerare"));

        let mut report = BenchReport::new();
        report.scaling.push(series);
        assert!(report.to_json().contains("\"efficiency\""));
        assert!(report.render().contains("Scalare pe fire"));

        assert!(bench_scaling("nano", &cfg, &params, &[], |_| {}).is_err());
        assert!(bench_scaling("nano", &cfg, &params, &[0, 1], |_| {}).is_err());
    }

    #[test]
    fn a_pool_install_changes_the_thread_count_the_backend_sees() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        assert_eq!(pool.install(sys::worker_threads), 3);
    }

    #[test]
    fn bench_runs_and_reports_sane_numbers() {
        let cfg = Preset::Nano.config(32);
        let r = bench_model("nano", &cfg, 2, 16, 1, 2).unwrap();
        assert_eq!(r.params, cfg.num_params());
        assert!(r.ms_per_step > 0.0 && r.tokens_per_sec > 0.0 && r.gflops > 0.0);
        assert!(r.ms_per_step_best <= r.ms_per_step + 1e-9);
        assert!((r.eta_secs(r.tokens_per_sec) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bench_rejects_bad_arguments() {
        let cfg = Preset::Nano.config(32);
        assert!(bench_model("x", &cfg, 2, 64, 0, 1).is_err());
        assert!(bench_model("x", &cfg, 0, 16, 0, 1).is_err());
    }

    #[test]
    fn report_renders_a_table_and_json() {
        let cfg = Preset::Nano.config(32);
        let mut rep = BenchReport::new();
        rep.results
            .push(bench_model("nano", &cfg, 2, 16, 0, 1).unwrap());
        let text = rep.render();
        assert!(text.contains("nano") && text.contains("tokeni/s"));
        let json = rep.to_json();
        assert!(json.contains("\"tokens_per_sec\""));
    }
}
