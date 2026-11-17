//! The training loop: AdamW with linear warm-up and cosine decay, periodic validation,
//! checkpointing and a clean stop on request.
//!
//! The loop reports progress through [`Event`] callbacks instead of printing, so the same code
//! can drive the command line today and a graphical interface later.

use crate::checkpoint::{self, Meta};
use crate::config::GptConfig;
use crate::data::{Corpus, Rng, Split};
use crate::generate::{SampleOptions, generate};
use crate::gradients;
use crate::model::Gpt;
use crate::tokenizer::Tokenizer;
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Hyper-parameters of a training run.
#[derive(Clone, Debug)]
pub struct TrainOptions {
    /// Total number of optimizer steps (the run ends when this step number is reached).
    pub steps: usize,
    /// Windows per micro-batch (what is held in memory at once).
    pub batch: usize,
    /// Micro-batches whose gradients are summed before each optimizer step. The effective batch
    /// is `batch * accum` windows with the memory of `batch`.
    pub accum: usize,
    /// Tokens per window (context length used for training; at most the model's `block_size`).
    pub seq: usize,
    /// Peak learning rate (reached at the end of the warm-up).
    pub lr: f64,
    /// The learning rate decays to `lr * min_lr_ratio` at the final step.
    pub min_lr_ratio: f64,
    /// Number of warm-up steps with a linearly rising learning rate (capped at a quarter of
    /// `steps`, see [`TrainOptions::effective_warmup`]).
    pub warmup: usize,
    /// Decoupled weight decay of AdamW.
    pub weight_decay: f32,
    /// Clip of the global gradient norm: when the L2 norm of all gradients together exceeds this,
    /// they are scaled down to it (`0` disables clipping).
    pub grad_clip: f32,
    /// Second-moment decay of AdamW. 0.95 suits short runs better than the 0.999 default.
    pub beta2: f32,
    /// Validate every this many steps (`0` disables validation).
    pub eval_every: usize,
    /// Number of validation batches averaged per evaluation.
    pub eval_batches: usize,
    /// Write a checkpoint every this many steps (`0` = only at the end).
    pub save_every: usize,
    /// Report training loss every this many steps.
    pub log_every: usize,
    /// Generate a text sample every this many steps (`0` = only at the end).
    pub sample_every: usize,
    /// Seed for initialisation and batch sampling.
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            steps: 1000,
            batch: 16,
            accum: 1,
            seq: 256,
            lr: 6e-4,
            min_lr_ratio: 0.1,
            warmup: 50,
            weight_decay: 0.01,
            grad_clip: 1.0,
            beta2: 0.95,
            eval_every: 100,
            eval_batches: 8,
            save_every: 500,
            log_every: 10,
            sample_every: 0,
            seed: 1,
        }
    }
}

impl TrainOptions {
    /// Checks the options against the model; returns a readable error otherwise.
    pub fn validate(&self, cfg: &GptConfig) -> Result<(), String> {
        if self.steps == 0 || self.batch == 0 || self.accum == 0 {
            return Err("--steps, --batch și --accum trebuie să fie cel puțin 1".into());
        }
        if self.seq < 2 || self.seq > cfg.block_size {
            return Err(format!(
                "--seq ({}) trebuie să fie între 2 și contextul modelului ({})",
                self.seq, cfg.block_size
            ));
        }
        if !(self.lr.is_finite() && self.lr > 0.0) {
            return Err("--lr trebuie să fie un număr pozitiv".into());
        }
        if !(0.0..=1.0).contains(&self.min_lr_ratio) {
            return Err("min_lr_ratio trebuie să fie în [0, 1]".into());
        }
        if self.eval_every > 0 && self.eval_batches == 0 {
            return Err("--eval-batches trebuie să fie cel puțin 1 când validarea e activă".into());
        }
        Ok(())
    }

    /// Warm-up length actually used: `warmup`, but never more than a quarter of the run, so that
    /// a short run still reaches its peak learning rate and decays afterwards.
    pub fn effective_warmup(&self) -> usize {
        self.warmup.min(self.steps / 4)
    }

    /// Learning rate for the step with 0-based index `step`: linear warm-up, then cosine decay
    /// from `lr` down to `lr * min_lr_ratio` at the last step.
    pub fn lr_at(&self, step: usize) -> f64 {
        let min_lr = self.lr * self.min_lr_ratio;
        let warmup = self.effective_warmup();
        if warmup > 0 && step < warmup {
            return self.lr * (step + 1) as f64 / warmup as f64;
        }
        if step + 1 >= self.steps {
            return min_lr;
        }
        let span = (self.steps - warmup).max(1) as f64;
        let progress = ((step - warmup.min(step)) as f64 / span).clamp(0.0, 1.0);
        min_lr + 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()) * (self.lr - min_lr)
    }
}

/// Progress notifications emitted by [`train`].
#[derive(Debug)]
pub enum Event<'a> {
    /// The run is about to start.
    Start {
        /// Trainable parameters.
        params: usize,
        /// Tokens processed per optimizer step.
        tokens_per_step: usize,
        /// Estimated peak memory of the run in bytes (a conservative guide, see
        /// [`GptConfig::estimate_train_bytes`]).
        est_memory_bytes: u64,
        /// First step of this run (non-zero when resuming).
        from_step: usize,
        /// Step number at which the run ends.
        total_steps: usize,
        /// Tokens in the training split.
        train_tokens: usize,
        /// Average bytes of text per token (1 for the byte tokenizer).
        bytes_per_token: f64,
        /// Vocabulary size of the tokenizer.
        vocab_size: usize,
        /// When resuming: whether the AdamW moment estimates were restored (checkpoints from
        /// before GERMEN 0.2 have none and start them from zero).
        optimizer_restored: bool,
    },
    /// Periodic training progress.
    Step {
        /// Steps completed so far.
        step: usize,
        /// Step number at which the run ends.
        total: usize,
        /// Mean training loss over the last reporting interval (nats per token).
        loss: f32,
        /// Learning rate of the latest step.
        lr: f64,
        /// Mean global gradient norm over the interval, before clipping (`NaN` when clipping
        /// is off, because the norm is then not computed).
        grad_norm: f32,
        /// Training throughput over the interval.
        tokens_per_sec: f64,
        /// Seconds spent training in this run.
        elapsed: f64,
        /// Estimated seconds until the end of the run.
        eta: f64,
    },
    /// A validation pass finished.
    Eval {
        /// Steps completed so far.
        step: usize,
        /// Mean validation loss (nats per token).
        val_loss: f32,
        /// Whether this is the best validation loss so far.
        best: bool,
    },
    /// A text sample was generated.
    Sample {
        /// Steps completed so far.
        step: usize,
        /// The generated text.
        text: &'a str,
    },
    /// A checkpoint was written.
    Saved {
        /// Steps completed so far.
        step: usize,
        /// Checkpoint directory.
        dir: &'a Path,
    },
}

/// Outcome of a training run.
#[derive(Clone, Debug)]
pub struct TrainReport {
    /// Step number reached (counting steps from earlier runs when resuming).
    pub final_step: usize,
    /// Steps executed by this call.
    pub steps_run: usize,
    /// Training tokens processed overall (including earlier runs).
    pub tokens_seen: u64,
    /// Training loss of the final step.
    pub last_train_loss: f32,
    /// Last validation loss, if validation ran.
    pub last_val_loss: Option<f32>,
    /// Best validation loss, if validation ran.
    pub best_val_loss: Option<f32>,
    /// Seconds spent training in this call.
    pub elapsed_secs: f64,
    /// Average training throughput of this call.
    pub tokens_per_sec: f64,
    /// Whether the run was stopped early on request.
    pub interrupted: bool,
}

/// Mean validation loss (nats per token) over `batches` fixed random windows.
///
/// The windows are the same on every call, so successive evaluations are comparable.
pub fn eval_loss(
    model: &Gpt,
    corpus: &Corpus,
    batches: usize,
    batch: usize,
    seq: usize,
    device: &Device,
) -> f32 {
    let mut rng = Rng::new(0xE7A1_0001);
    let mut total = 0.0f32;
    for _ in 0..batches {
        let b = corpus.sample_batch(Split::Val, batch, seq, &mut rng);
        let (x, y) = b.to_tensors(device);
        total += model.loss(x, y).into_scalar::<f32>();
    }
    total / batches as f32
}

fn make_sample(model: &Gpt, tokenizer: &Tokenizer, device: &Device, seed: u64) -> String {
    let opts = SampleOptions {
        max_new: 160,
        temperature: 0.8,
        top_k: 40,
        window: 128,
    };
    let mut rng = Rng::new(seed);
    let tokens = generate(model, tokenizer, b"", &opts, &mut rng, device, |_| {});
    tokenizer.decode_complete(&tokens)
}

fn save_checkpoint(
    out_dir: &Path,
    cfg: &GptConfig,
    model: &Gpt,
    optim: &ModuleOptimizer,
    tokenizer: &Tokenizer,
    meta: &Meta,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<(), String> {
    checkpoint::save(out_dir, model, cfg, meta)?;
    checkpoint::save_optimizer(out_dir, optim)?;
    tokenizer.save(out_dir)?;
    on_event(Event::Saved {
        step: meta.step,
        dir: out_dir,
    });
    Ok(())
}

/// Trains `cfg` on `corpus` and writes checkpoints into `out_dir`.
///
/// With `resume`, weights, step counter, statistics and the optimizer's moment estimates are
/// restored from `out_dir`. Setting `stop` to `true` from another
/// thread (e.g. a Ctrl-C handler) ends the run after the current step with a final checkpoint.
pub fn train(
    corpus: &Corpus,
    cfg: &GptConfig,
    opts: &TrainOptions,
    out_dir: &Path,
    resume: bool,
    stop: &AtomicBool,
    on_event: &mut dyn FnMut(Event<'_>),
) -> Result<TrainReport, String> {
    cfg.validate()?;
    opts.validate(cfg)?;
    if corpus.tokenizer.vocab_size() != cfg.vocab_size {
        return Err(format!(
            "tokenizatorul ({} tokeni) nu se potrivește cu modelul (vocabular {})",
            corpus.tokenizer.vocab_size(),
            cfg.vocab_size
        ));
    }
    corpus.check_seq(opts.seq)?;
    let est_memory_bytes = cfg.estimate_train_bytes(opts.batch, opts.seq);
    crate::sys::check_memory_fits(est_memory_bytes, opts.batch, opts.seq)?;

    let device = crate::train_device();
    device.seed(opts.seed);
    let eval_device = device.clone().without_autodiff();

    let (mut model, mut meta) = if resume {
        let (m, saved_cfg, meta) = checkpoint::load(out_dir, &device)?;
        let same = serde_json::to_string(&saved_cfg).ok() == serde_json::to_string(cfg).ok();
        if !same {
            return Err("arhitectura din punctul de control diferă de cea cerută; \
                        la --resume folosiți aceeași dimensiune de model"
                .into());
        }
        if Tokenizer::load(out_dir)? != corpus.tokenizer {
            return Err(
                "tokenizatorul din punctul de control diferă de cel al corpusului; \
                        la --resume trebuie folosit același tokenizator"
                    .into(),
            );
        }
        (m, meta)
    } else {
        (cfg.init(&device), Meta::default())
    };
    meta.germen_version = crate::VERSION.to_string();
    meta.seed = opts.seed;
    meta.corpus_bytes = corpus.len();
    meta.seq = opts.seq;
    meta.batch = opts.batch;
    meta.accum = opts.accum;

    let start_step = meta.step;
    let tokens_per_step = opts.batch * opts.seq * opts.accum;

    // Clipping is done on the global norm by `gradients::clip_global_norm`, not by Burn.
    let mut optim = AdamWConfig::new()
        .with_beta_2(opts.beta2)
        .with_epsilon(1e-8)
        .with_weight_decay(opts.weight_decay)
        .init();
    let mut optimizer_restored = false;
    if resume {
        (optim, optimizer_restored) = checkpoint::load_optimizer(out_dir, optim)?;
    }

    on_event(Event::Start {
        params: cfg.num_params(),
        tokens_per_step,
        est_memory_bytes,
        from_step: start_step,
        total_steps: opts.steps,
        train_tokens: corpus.train.len(),
        bytes_per_token: corpus.bytes_per_token(),
        vocab_size: cfg.vocab_size,
        optimizer_restored,
    });

    let mut rng = Rng::new(opts.seed ^ (start_step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let log_every = opts.log_every.max(1);
    let elapsed_before = meta.elapsed_secs;

    let mut train_time = 0.0f64; // seconds spent inside training steps only
    let mut interval_time = 0.0f64;
    let mut interval_loss = 0.0f64;
    let mut interval_grad_norm = 0.0f64;
    let mut interval_steps = 0usize;
    let mut last_loss = f32::NAN;
    let mut steps_run = 0usize;
    let mut interrupted = false;

    for step in start_step..opts.steps {
        if stop.load(Ordering::Relaxed) {
            interrupted = true;
            break;
        }

        let started = Instant::now();
        let lr = opts.lr_at(step);

        // Forward and backward over `accum` micro-batches; each loss is divided by `accum`, so
        // the summed gradients equal those of one batch `accum` times larger.
        let mut accumulator = GradientsAccumulator::new();
        let mut step_loss = 0.0f32;
        for _ in 0..opts.accum {
            let batch = corpus.sample_batch(Split::Train, opts.batch, opts.seq, &mut rng);
            let (x, y) = batch.to_tensors(&device);
            let loss = model.loss(x, y);
            let value = loss.clone().into_scalar::<f32>();
            if !value.is_finite() {
                return Err(format!(
                    "pierderea a devenit {value} la pasul {step}; reduceți --lr (acum {lr:.2e}) \
                     sau activați --grad-clip"
                ));
            }
            step_loss += value / opts.accum as f32;
            let scaled = if opts.accum > 1 {
                loss.div_scalar(opts.accum as f32)
            } else {
                loss
            };
            let grads = GradientsParams::from_grads(scaled.backward(), &model);
            accumulator.accumulate(&model, grads);
        }
        last_loss = step_loss;
        let mut grads = accumulator.grads();
        let grad_norm = if opts.grad_clip > 0.0 {
            gradients::clip_global_norm(&mut grads, &model, opts.grad_clip)
        } else {
            f32::NAN
        };
        if opts.grad_clip > 0.0 && !grad_norm.is_finite() {
            return Err(format!(
                "gradientul a devenit non-finit ({grad_norm}) la pasul {step}; reduceți --lr \
                 (acum {lr:.2e})"
            ));
        }
        model = optim.step(lr, model, grads);

        let dt = started.elapsed().as_secs_f64();
        train_time += dt;
        interval_time += dt;
        interval_loss += last_loss as f64;
        interval_grad_norm += grad_norm as f64;
        interval_steps += 1;
        steps_run += 1;
        let done = step + 1;
        meta.step = done;
        meta.tokens_seen += tokens_per_step as u64;
        meta.last_train_loss = Some(last_loss);
        meta.elapsed_secs = elapsed_before + train_time;

        let is_last = done == opts.steps;

        if done % log_every == 0 || is_last {
            let per_step = train_time / steps_run as f64;
            on_event(Event::Step {
                step: done,
                total: opts.steps,
                loss: (interval_loss / interval_steps as f64) as f32,
                lr,
                grad_norm: (interval_grad_norm / interval_steps as f64) as f32,
                tokens_per_sec: (interval_steps * tokens_per_step) as f64 / interval_time.max(1e-9),
                elapsed: train_time,
                eta: per_step * (opts.steps - done) as f64,
            });
            interval_time = 0.0;
            interval_loss = 0.0;
            interval_grad_norm = 0.0;
            interval_steps = 0;
        }

        if opts.eval_every > 0 && (done % opts.eval_every == 0 || is_last) {
            let val = eval_loss(
                &model.valid(),
                corpus,
                opts.eval_batches,
                opts.batch,
                opts.seq,
                &eval_device,
            );
            let best = meta.best_val_loss.is_none_or(|b| val < b);
            if best {
                meta.best_val_loss = Some(val);
            }
            meta.last_val_loss = Some(val);
            on_event(Event::Eval {
                step: done,
                val_loss: val,
                best,
            });
        }

        if opts.sample_every > 0 && done % opts.sample_every == 0 && !is_last {
            let text = make_sample(
                &model.valid(),
                &corpus.tokenizer,
                &eval_device,
                opts.seed ^ done as u64,
            );
            on_event(Event::Sample {
                step: done,
                text: &text,
            });
        }

        if opts.save_every > 0 && done % opts.save_every == 0 && !is_last {
            save_checkpoint(
                out_dir,
                cfg,
                &model,
                &optim,
                &corpus.tokenizer,
                &meta,
                on_event,
            )?;
        }
    }

    // Final checkpoint and sample (also after an interruption).
    meta.elapsed_secs = elapsed_before + train_time;
    save_checkpoint(
        out_dir,
        cfg,
        &model,
        &optim,
        &corpus.tokenizer,
        &meta,
        on_event,
    )?;
    if steps_run > 0 {
        let text = make_sample(
            &model.valid(),
            &corpus.tokenizer,
            &eval_device,
            opts.seed ^ 0xF1A1,
        );
        on_event(Event::Sample {
            step: meta.step,
            text: &text,
        });
    }

    Ok(TrainReport {
        final_step: meta.step,
        steps_run,
        tokens_seen: meta.tokens_seen,
        last_train_loss: last_loss,
        last_val_loss: meta.last_val_loss,
        best_val_loss: meta.best_val_loss,
        elapsed_secs: train_time,
        tokens_per_sec: if train_time > 0.0 {
            (steps_run * tokens_per_step) as f64 / train_time
        } else {
            0.0
        },
        interrupted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;
    use crate::data::demo_corpus;

    #[test]
    fn learning_rate_schedule_warms_up_and_decays() {
        let o = TrainOptions {
            steps: 100,
            warmup: 10,
            lr: 1e-3,
            min_lr_ratio: 0.1,
            ..TrainOptions::default()
        };
        assert!(o.lr_at(0) < o.lr_at(5) && o.lr_at(5) < o.lr_at(9));
        assert!(
            (o.lr_at(9) - 1e-3).abs() < 1e-9,
            "warm-up must end at the peak"
        );
        let mut prev = f64::MAX;
        for s in 10..100 {
            let l = o.lr_at(s);
            assert!(
                l <= prev + 1e-12,
                "lr must not increase during decay (step {s})"
            );
            assert!((1e-4 - 1e-12..=1e-3 + 1e-12).contains(&l));
            prev = l;
        }
        assert!(
            (o.lr_at(99) - 1e-4).abs() < 1e-9,
            "last step uses the minimum lr"
        );
    }

    #[test]
    fn warmup_is_capped_so_short_runs_still_reach_the_peak() {
        let o = TrainOptions {
            steps: 40,
            warmup: 100,
            lr: 1e-3,
            ..TrainOptions::default()
        };
        assert_eq!(o.effective_warmup(), 10);
        assert!((o.lr_at(9) - 1e-3).abs() < 1e-9);
        assert!(o.lr_at(39) < 2e-4, "and it decays by the end");
    }

    #[test]
    fn schedule_without_warmup_starts_at_peak() {
        let o = TrainOptions {
            steps: 50,
            warmup: 0,
            lr: 2e-3,
            ..TrainOptions::default()
        };
        assert!((o.lr_at(0) - 2e-3).abs() < 1e-6);
    }

    #[test]
    fn options_validation_catches_mistakes() {
        let cfg = Preset::Nano.config(64);
        assert!(
            TrainOptions {
                seq: 128,
                ..TrainOptions::default()
            }
            .validate(&cfg)
            .is_err()
        );
        assert!(
            TrainOptions {
                batch: 0,
                ..TrainOptions::default()
            }
            .validate(&cfg)
            .is_err()
        );
        assert!(
            TrainOptions {
                lr: -1.0,
                ..TrainOptions::default()
            }
            .validate(&cfg)
            .is_err()
        );
        assert!(
            TrainOptions {
                seq: 64,
                ..TrainOptions::default()
            }
            .validate(&cfg)
            .is_ok()
        );
    }

    #[test]
    fn training_lowers_the_loss_and_resume_continues() {
        let dir = std::env::temp_dir().join(format!("germen-train-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let corpus = Corpus::from_bytes(demo_corpus(60_000, 5).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(48);
        let opts = TrainOptions {
            steps: 30,
            batch: 8,
            seq: 48,
            lr: 3e-3,
            warmup: 3,
            eval_every: 15,
            eval_batches: 2,
            save_every: 0,
            log_every: 5,
            ..TrainOptions::default()
        };

        let mut losses = Vec::new();
        let mut saved = 0;
        let mut grad_norms = Vec::new();
        let stop = AtomicBool::new(false);
        let report = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |e| match e {
            Event::Step {
                loss, grad_norm, ..
            } => {
                losses.push(loss);
                grad_norms.push(grad_norm);
            }
            Event::Saved { .. } => saved += 1,
            _ => {}
        })
        .unwrap();
        assert!(
            grad_norms.iter().all(|g| g.is_finite() && *g > 0.0),
            "gradient norms must be reported: {grad_norms:?}"
        );
        assert!(dir.join(checkpoint::OPTIMIZER_FILE).exists());

        assert_eq!(report.final_step, 30);
        assert_eq!(report.steps_run, 30);
        assert!(!report.interrupted);
        assert!(saved >= 1);
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(first > 4.5, "should start near ln(256) = 5.55, got {first}");
        assert!(last < first - 0.5, "loss should fall: {first} -> {last}");
        assert!(report.best_val_loss.is_some());

        // Resume for 10 more steps: the step counter continues from the checkpoint.
        let more = TrainOptions {
            steps: 40,
            ..opts.clone()
        };
        let mut restored = None;
        let report2 = train(&corpus, &cfg, &more, &dir, true, &stop, &mut |e| {
            if let Event::Start {
                optimizer_restored, ..
            } = e
            {
                restored = Some(optimizer_restored);
            }
        })
        .unwrap();
        assert_eq!(restored, Some(true), "AdamW state must be restored");
        assert_eq!(report2.final_step, 40);
        assert_eq!(report2.steps_run, 10);
        assert!(report2.tokens_seen > report.tokens_seen);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restored_optimizer_state_continues_exactly_like_uninterrupted_training() {
        let dir = std::env::temp_dir().join(format!("germen-optim-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let device = crate::train_device();
        let cfg = Preset::Nano.config(16);
        let adam = || AdamWConfig::new().with_beta_2(0.95).init();
        let window = |i: usize| {
            let make = |shift: usize| {
                let data: Vec<i32> = (0..3 * 16)
                    .map(|k| ((i * 53 + k * 7 + shift) % 256) as i32)
                    .collect();
                Tensor::<2, Int>::from_data(TensorData::new(data, [3, 16]), &device)
            };
            (make(0), make(1))
        };
        let step = |model: Gpt, optim: &mut ModuleOptimizer, i: usize| {
            let (x, y) = window(i);
            let grads = GradientsParams::from_grads(model.loss(x, y).backward(), &model);
            optim.step(1e-2, model, grads)
        };

        // Three steps, then a checkpoint.
        let mut model = cfg.init(&device);
        let mut optim = adam();
        for i in 0..3 {
            model = step(model, &mut optim, i);
        }
        checkpoint::save(&dir, &model, &cfg, &Meta::default()).unwrap();
        checkpoint::save_optimizer(&dir, &optim).unwrap();

        // A: carry on in the same process. B: reload weights and optimizer from disk.
        // C: reload the weights only (what GERMEN 0.1 did).
        let a = step(model, &mut optim, 3);
        let (mb, _, _) = checkpoint::load(&dir, &device).unwrap();
        let (mut ob, found) = checkpoint::load_optimizer(&dir, adam()).unwrap();
        assert!(found);
        let b = step(mb, &mut ob, 3);
        let (mc, _, _) = checkpoint::load(&dir, &device).unwrap();
        let c = step(mc, &mut adam(), 3);

        let probe = {
            let data: Vec<i32> = (0..32).map(|k| k * 13 % 256).collect();
            Tensor::<2, Int>::from_data(TensorData::new(data, [2, 16]), &eval_dev())
        };
        fn eval_dev() -> Device {
            crate::cpu_device()
        }
        let logits = |m: Gpt| m.valid().forward(probe.clone());
        let max_diff = |p: Tensor<3>, q: Tensor<3>| (p - q).abs().max().into_scalar::<f32>();
        let (la, lb, lc) = (logits(a), logits(b), logits(c));
        let restored = max_diff(la.clone(), lb);
        let fresh = max_diff(la, lc);
        assert!(restored < 1e-4, "restored run drifted by {restored}");
        assert!(
            fresh > 10.0 * restored.max(1e-6),
            "a fresh optimizer should visibly differ (restored {restored}, fresh {fresh})"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bpe_run_stores_its_tokenizer_and_resume_insists_on_the_same_one() {
        use crate::bpe::Bpe;
        let dir = std::env::temp_dir().join(format!("germen-bpe-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = demo_corpus(150_000, 21).into_bytes();
        let bpe = Bpe::train(&text, 400, &mut |_, _| {}).unwrap();
        let corpus = Corpus::build(&text, 0.1, Tokenizer::Bpe(bpe)).unwrap();
        assert!(
            corpus.bytes_per_token() > 2.5,
            "{}",
            corpus.bytes_per_token()
        );
        let vocab = corpus.tokenizer.vocab_size();
        let cfg = Preset::Nano.config(32).with_vocab(vocab);
        let opts = TrainOptions {
            steps: 30,
            batch: 8,
            seq: 32,
            lr: 3e-3,
            warmup: 3,
            eval_every: 0,
            save_every: 0,
            log_every: 5,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        let mut losses = Vec::new();
        let mut seen_vocab = 0;
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |e| match e {
            Event::Step { loss, .. } => losses.push(loss),
            Event::Start { vocab_size, .. } => seen_vocab = vocab_size,
            _ => {}
        })
        .unwrap();
        assert_eq!(seen_vocab, vocab);
        let start = (vocab as f32).ln();
        assert!(losses[0] > start - 0.6, "{} vs ln(V) = {start}", losses[0]);
        assert!(losses.last().unwrap() < &(losses[0] - 0.4), "{losses:?}");
        assert_eq!(Tokenizer::load(&dir).unwrap(), corpus.tokenizer);

        // Resuming with the same tokenizer works...
        let more = TrainOptions {
            steps: 36,
            ..opts.clone()
        };
        train(&corpus, &cfg, &more, &dir, true, &stop, &mut |_| {}).unwrap();
        // ...with a different one (here: bytes, whose vocabulary does not fit the model) it is
        // refused instead of silently producing garbage.
        let bytes = Corpus::from_bytes(text.clone(), 0.1).unwrap();
        let err = train(&bytes, &cfg, &more, &dir, true, &stop, &mut |_| {}).unwrap_err();
        assert!(err.contains("tokenizator"), "{err}");
        // Same vocabulary size but a different merge list is caught as well.
        let other = Bpe::train(&text[..60_000], 400, &mut |_, _| {}).unwrap();
        if other.vocab_size() == vocab {
            let c2 = Corpus::build(&text, 0.1, Tokenizer::Bpe(other)).unwrap();
            let err = train(&c2, &cfg, &more, &dir, true, &stop, &mut |_| {}).unwrap_err();
            assert!(err.contains("tokenizator"), "{err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gradient_accumulation_counts_tokens_and_still_learns() {
        let dir = std::env::temp_dir().join(format!("germen-accum-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(60_000, 8).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let opts = TrainOptions {
            steps: 24,
            batch: 4,
            accum: 3,
            seq: 32,
            lr: 3e-3,
            warmup: 3,
            eval_every: 0,
            save_every: 0,
            log_every: 4,
            ..TrainOptions::default()
        };
        let mut tokens_per_step = 0;
        let mut losses = Vec::new();
        let stop = AtomicBool::new(false);
        let report = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |e| match e {
            Event::Start {
                tokens_per_step: t, ..
            } => tokens_per_step = t,
            Event::Step { loss, .. } => losses.push(loss),
            _ => {}
        })
        .unwrap();
        assert_eq!(tokens_per_step, 4 * 32 * 3);
        assert_eq!(report.tokens_seen, 24 * 4 * 32 * 3);
        assert!(losses.last().unwrap() < &(losses[0] - 0.3), "{losses:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn absurd_memory_requests_are_refused_before_allocating() {
        let dir = std::env::temp_dir().join(format!("germen-mem-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(200_000, 4).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(10_000);
        let opts = TrainOptions {
            seq: 10_000,
            batch: 64,
            steps: 1,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        let err = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap_err();
        assert!(err.contains("--batch") && err.contains("GiB"), "{err}");
        assert!(
            !dir.exists(),
            "nothing may be written when the request is refused"
        );
    }

    #[test]
    fn stop_flag_ends_the_run_with_a_checkpoint() {
        let dir = std::env::temp_dir().join(format!("germen-stop-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(20_000, 6).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let opts = TrainOptions {
            steps: 1000,
            batch: 4,
            seq: 32,
            eval_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(true); // stop requested before the first step
        let report = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        assert!(report.interrupted);
        assert_eq!(report.steps_run, 0);
        assert!(dir.join(checkpoint::WEIGHTS_FILE).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
