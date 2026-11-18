//! The training loop: AdamW with linear warm-up and cosine decay, periodic validation,
//! checkpointing and a clean stop on request.
//!
//! The loop reports progress through [`Event`] callbacks instead of printing, so the same code
//! can drive the command line today and a graphical interface later.

use crate::checkpoint::{self, Meta};
use crate::config::GptConfig;
use crate::data::{Batch, Corpus, Rng, Split};
use crate::fsutil::DirLock;
use crate::generate::{SampleOptions, generate};
use crate::gradients;
use crate::model::Gpt;
use crate::tokenizer::Tokenizer;
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use rayon::prelude::*;
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
    /// Data-parallel replicas per micro-batch: the windows of a micro-batch are split among this
    /// many copies of the model, each running forward and backward on its own autodiff graph on a
    /// different worker thread, and their gradients are summed. The result equals one big graph
    /// (up to the rounding of the sums), but the serial parts of a step (the many small
    /// element-wise operations, which do not scale with threads) now run side by side.
    /// `0` chooses automatically, see [`plan_replicas`]; `1` is the ordinary single-graph step.
    pub replicas: usize,
    /// Allow a run that starts from scratch (no `resume`) to replace a checkpoint that is already
    /// in the output directory. Without it such a run is refused: its first save would otherwise
    /// destroy a trained model because `--resume` was forgotten.
    pub overwrite: bool,
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
            replicas: 0,
            overwrite: false,
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
        if !(self.beta2.is_finite() && (0.0..1.0).contains(&self.beta2)) {
            return Err("beta2 trebuie să fie în [0, 1)".into());
        }
        if !(self.weight_decay.is_finite() && self.weight_decay >= 0.0) {
            return Err("--weight-decay trebuie să fie un număr finit, cel puțin 0".into());
        }
        // AdamW multiplies every weight by `1 - lr * weight_decay` at each step: from 1 on that no
        // longer shrinks the weights, it flips their sign and makes them grow (`--weight-decay 1e9`
        // used to be accepted and trained a model of noise).
        let decay = self.lr * self.weight_decay as f64;
        if decay >= 1.0 {
            return Err(format!(
                "--lr × --weight-decay ({decay}) trebuie să fie sub 1: de la 1 în sus descompunerea                  ponderilor le răstoarnă semnul în loc să le micșoreze"
            ));
        }
        // NaN used to compare false with `> 0` and switch the clipping off without a word.
        if !(self.grad_clip.is_finite() && self.grad_clip >= 0.0) {
            return Err(
                "--grad-clip trebuie să fie un număr finit, cel puțin 0 (0 = oprit)".into(),
            );
        }
        // Far beyond any real run, but small enough that the products below cannot overflow.
        const MAX_TOKENS_PER_STEP: usize = 1 << 40;
        let tokens_per_step = self
            .batch
            .checked_mul(self.seq)
            .and_then(|n| n.checked_mul(self.accum))
            .filter(|&n| n <= MAX_TOKENS_PER_STEP);
        if tokens_per_step.is_none() {
            return Err(format!(
                "--batch × --seq × --accum ({} × {} × {}) depășește {MAX_TOKENS_PER_STEP} de tokeni pe pas",
                self.batch, self.seq, self.accum
            ));
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
        /// Data-parallel replicas per micro-batch actually used (1 = the ordinary step).
        replicas: usize,
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
    /// The output directory held the leftovers of a save that was interrupted, which have been
    /// completed or discarded (see [`checkpoint::recover`]); sent before [`Event::Start`].
    Recovered(checkpoint::Recovery),
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
    // One set: weights, optimizer, tokenizer, architecture and (last) the progress record.
    checkpoint::save_all(out_dir, model, cfg, meta, Some(optim), Some(tokenizer))?;
    on_event(Event::Saved {
        step: meta.step,
        dir: out_dir,
    });
    Ok(())
}

/// Values of the one-dimensional parameters of a model, in visiting order. In a [`Gpt`] those are
/// exactly the gains of the RMSNorm layers; every other parameter is a matrix.
struct Gains(Vec<Tensor<1>>);

impl ModuleVisitor for Gains {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if D == 1 {
            let t = param.val().detach();
            let n = t.dims().iter().product::<usize>();
            self.0.push(t.reshape([n]));
        }
    }
}

/// Puts back, on every one-dimensional parameter, the share of its own old value that AdamW's
/// decoupled decay took away: `p * (1 - lr * wd) - lr * u` becomes `p - lr * u`.
struct GiveBack {
    old: std::vec::IntoIter<Tensor<1>>,
    by: f32,
}

impl ModuleMapper for GiveBack {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        if D != 1 {
            return param;
        }
        let old = self.old.next().expect("one old gain for every gain");
        let by = self.by;
        param.map(|t| {
            let (dims, device, tracked) = (t.dims(), t.device(), t.is_require_grad());
            let n = dims.iter().product::<usize>();
            let back = (t.detach().reshape([n]) + old.mul_scalar(by)).reshape(dims);
            // A computed tensor is not a leaf of the autodiff graph, and a parameter must be one:
            // go through the values (a handful of numbers per layer).
            let leaf = Tensor::<D>::from_data(back.into_data(), &device);
            if tracked { leaf.require_grad() } else { leaf }
        })
    }
}

/// One AdamW step in which the weight decay reaches the matrices (and the embedding) but not the
/// RMSNorm gains, as in nanoGPT, llm.c and the GPT-3 recipe: a decay pulls a weight towards zero,
/// which regularises a matrix and only shrinks a gain that the loss is trying to hold at about one.
/// Up to 0.5.0 the gains were decayed too, by `lr * weight_decay` of their value at every step:
/// 10 % in all over 2 000 steps at `lr = 1e-3`, `weight_decay = 0.1` (the nanoGPT recipe), 0.7 %
/// at the defaults for 2 000 steps and 6 % for 20 000.
///
/// Burn's AdamW has no per-parameter exemption, so the step is taken as usual and the decay that
/// was applied to the gains is given back: the decay is `p ← p · (1 − lr · wd)` before the update
/// (verified against a hand-written AdamW in `scripts/reference_check.py`), so adding `p_old · lr ·
/// wd` afterwards leaves exactly `p_old − lr · update`, up to one rounding.
pub fn adamw_step(
    optim: &mut ModuleOptimizer,
    lr: f64,
    weight_decay: f32,
    model: Gpt,
    grads: GradientsParams,
) -> Gpt {
    if weight_decay <= 0.0 {
        return optim.step(lr, model, grads);
    }
    let mut gains = Gains(Vec::new());
    model.visit(&mut gains);
    let stepped = optim.step(lr, model, grads);
    stepped.map(&mut GiveBack {
        old: gains.0.into_iter(),
        by: (lr * weight_decay as f64) as f32,
    })
}

/// How many replicas to run per micro-batch of `batch` windows of `seq` tokens.
///
/// A `requested` number (non-zero) is honoured, only capped at the number of windows. `0` means
/// automatic: one replica per worker thread, but never more than there are windows, and fewer
/// if the extra sets of gradients would not fit in memory.
pub fn plan_replicas(cfg: &GptConfig, batch: usize, seq: usize, requested: usize) -> usize {
    let batch = batch.max(1);
    if requested > 0 {
        return requested.min(batch);
    }
    let workers = crate::sys::worker_threads();
    let limit = crate::sys::memory_limit_bytes();
    let mut replicas = workers.min(batch);
    while replicas > 1
        && cfg.estimate_train_bytes_replicated(batch, seq, replicas.min(workers)) as f64 > limit
    {
        replicas -= 1;
    }
    replicas
}

/// `a + b`, gradient by gradient.
fn add_gradients(model: &Gpt, a: GradientsParams, b: GradientsParams) -> GradientsParams {
    let mut sum = GradientsAccumulator::new();
    sum.accumulate(model, a);
    sum.accumulate(model, b);
    sum.grads()
}

/// Forward and backward of one replica: a copy of the model with an autodiff graph of its own
/// (`fork`), so that replicas on different threads never share a graph or its lock.
fn replica_gradients(
    model: &Gpt,
    part: &Batch,
    weight: f32,
    accum: usize,
    device: &Device,
) -> Result<(GradientsParams, f32), f32> {
    let replica = model.clone().fork(device);
    let (x, y) = part.to_tensors(device);
    let loss = replica.loss(x, y);
    let value = loss.clone().into_scalar::<f32>();
    if !value.is_finite() {
        return Err(value);
    }
    let weighted = loss.mul_scalar(weight);
    let scaled = if accum > 1 {
        weighted.div_scalar(accum as f32)
    } else {
        weighted
    };
    Ok((
        GradientsParams::from_grads(scaled.backward(), &replica),
        value,
    ))
}

/// Gradients of `loss / accum` for one micro-batch, and the micro-batch's mean loss (not scaled).
///
/// With `replicas > 1` the windows are split among that many replicas, each weighted by its share
/// of the windows, so the sum equals the gradient of the mean over the whole micro-batch (up to
/// rounding). The replicas are dealt to as many slots as there are worker threads; a slot runs its
/// replicas in turn, adding their gradients as it goes, and the slots' sums are then added in a
/// fixed pairwise order: the result does not depend on which thread ran what, only on how many
/// threads there are. `Err(loss)` is a loss that was NaN or infinite.
pub(crate) fn micro_batch_gradients(
    model: &Gpt,
    batch: &Batch,
    replicas: usize,
    accum: usize,
    device: &Device,
) -> Result<(GradientsParams, f32), f32> {
    let replicas = replicas.clamp(1, batch.batch.max(1));
    if replicas == 1 {
        let (x, y) = batch.to_tensors(device);
        let loss = model.loss(x, y);
        let value = loss.clone().into_scalar::<f32>();
        if !value.is_finite() {
            return Err(value);
        }
        let scaled = if accum > 1 {
            loss.div_scalar(accum as f32)
        } else {
            loss
        };
        return Ok((GradientsParams::from_grads(scaled.backward(), model), value));
    }

    let parts = batch.split(replicas);
    let slots = replicas.min(crate::sys::worker_threads()).max(1);
    let total = batch.batch as f32;
    let sums: Vec<Result<Option<(GradientsParams, f32)>, f32>> = (0..slots)
        .into_par_iter()
        .map(|slot| {
            let mut sum: Option<(GradientsParams, f32)> = None;
            for part in parts.iter().skip(slot).step_by(slots) {
                let weight = part.batch as f32 / total;
                let (grads, value) = replica_gradients(model, part, weight, accum, device)?;
                sum = Some(match sum {
                    None => (grads, value * weight),
                    Some((acc, loss)) => (add_gradients(model, acc, grads), loss + value * weight),
                });
            }
            Ok(sum)
        })
        .collect();
    let mut level: Vec<(GradientsParams, f32)> = Vec::with_capacity(slots);
    for sum in sums {
        level.extend(sum?);
    }
    while level.len() > 1 {
        level = level
            .into_par_iter()
            .chunks(2)
            .map(|mut pair| {
                let last = pair.pop().expect("chunks are not empty");
                match pair.pop() {
                    Some((acc, loss)) => (add_gradients(model, acc, last.0), loss + last.1),
                    None => last,
                }
            })
            .collect();
    }
    Ok(level.pop().expect("a micro-batch has at least one replica"))
}

/// Forward and backward over `opts.accum` micro-batches; returns the summed gradients (each
/// micro-batch's loss is divided by `accum`, so they equal those of one batch `accum` times
/// larger) and the mean loss.
///
/// `should_stop` is asked before every micro-batch. When it says yes, the partial sum is dropped
/// and `None` is returned: applying it would be a step on a smaller batch than the schedule
/// assumes. This is what makes Ctrl-C answer within one micro-batch rather than one whole step
/// (which with `--accum 64` on a big model can be minutes).
#[allow(clippy::too_many_arguments)]
fn accumulate_gradients(
    model: &Gpt,
    corpus: &Corpus,
    opts: &TrainOptions,
    replicas: usize,
    rng: &mut Rng,
    device: &Device,
    step: usize,
    lr: f64,
    should_stop: &mut dyn FnMut() -> bool,
) -> Result<Option<(GradientsParams, f32)>, String> {
    let mut accumulator = GradientsAccumulator::new();
    let mut step_loss = 0.0f32;
    for _ in 0..opts.accum {
        if should_stop() {
            return Ok(None);
        }
        let batch = corpus.sample_batch(Split::Train, opts.batch, opts.seq, rng);
        let (grads, value) = micro_batch_gradients(model, &batch, replicas, opts.accum, device)
            .map_err(|bad| {
                format!(
                    "pierderea a devenit {bad} la pasul {step}; reduceți --lr (acum {lr:.2e}) \
                     sau activați --grad-clip"
                )
            })?;
        step_loss += value / opts.accum as f32;
        accumulator.accumulate(model, grads);
    }
    Ok(Some((accumulator.grads(), step_loss)))
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
    let replicas = plan_replicas(cfg, opts.batch, opts.seq, opts.replicas);
    let est_memory_bytes = cfg.estimate_train_bytes_replicated(
        opts.batch,
        opts.seq,
        replicas.min(crate::sys::worker_threads()),
    );
    crate::sys::check_memory_fits(est_memory_bytes, opts.batch, opts.seq)?;
    // Held until this function returns: a second run in the same directory would interleave its
    // files with ours into a model that never existed.
    let _claim = DirLock::acquire(out_dir)?;
    // A save that died half-way is finished (or thrown away) before anything is read from, or
    // refused because of, what the directory holds.
    match checkpoint::recover(out_dir)? {
        checkpoint::Recovery::Nothing => {}
        recovery => on_event(Event::Recovered(recovery)),
    }
    if !resume && !opts.overwrite && checkpoint::exists(out_dir) {
        let step = checkpoint::load_meta(out_dir).map(|m| m.step).unwrap_or(0);
        return Err(format!(
            "{} conține deja un punct de control (pasul {step}); o antrenare nouă l-ar înlocui la \
             prima salvare. Continuați-l cu --resume, alegeți alt --out sau adăugați --overwrite \
             dacă vreți să-l înlocuiți",
            out_dir.display()
        ));
    }

    let device = crate::train_device();
    let eval_device = device.clone().without_autodiff();

    let (mut model, mut meta) = if resume {
        let (m, saved_cfg, meta) = checkpoint::load_for_resume(out_dir, &device)?;
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
        // Seeded and drawn in one step, not lazily and not racing: with replicas, several threads
        // would otherwise be the first to touch the parameters, and another run in this process
        // could take its share of the device's one random generator in between.
        (cfg.init_seeded(&device, opts.seed), Meta::default())
    };
    meta.germen_version = crate::VERSION.to_string();
    meta.seed = opts.seed;
    meta.corpus_bytes = corpus.bytes;
    meta.seq = opts.seq;
    meta.batch = opts.batch;
    meta.accum = opts.accum;

    let start_step = meta.step;
    if start_step > opts.steps {
        return Err(format!(
            "punctul de control e deja la pasul {start_step}, iar --steps ({}) e numărul TOTAL \
             de pași, nu cel de pași în plus; pentru a continua dați --steps mai mare decât \
             {start_step}",
            opts.steps
        ));
    }
    let tokens_per_step = opts.batch * opts.seq * opts.accum;

    // Clipping is done on the global norm by `gradients::clip_global_norm`, not by Burn.
    let mut optim = AdamWConfig::new()
        .with_beta_2(opts.beta2)
        .with_epsilon(1e-8)
        .with_weight_decay(opts.weight_decay)
        .init();
    let mut optimizer_restored = false;
    if resume {
        (optim, optimizer_restored) = checkpoint::load_optimizer(out_dir, optim, &model)?;
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
        replicas,
    });

    if start_step == opts.steps {
        // Already done (typically a `--resume` that is repeated until it has nothing left to do):
        // succeed without touching the checkpoint.
        return Ok(TrainReport {
            final_step: start_step,
            steps_run: 0,
            tokens_seen: meta.tokens_seen,
            last_train_loss: meta.last_train_loss.unwrap_or(f32::NAN),
            last_val_loss: meta.last_val_loss,
            best_val_loss: meta.best_val_loss,
            elapsed_secs: 0.0,
            tokens_per_sec: 0.0,
            interrupted: false,
        });
    }

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
        let mut rng = Rng::for_step(opts.seed, step);

        let Some((mut grads, step_loss)) = accumulate_gradients(
            &model,
            corpus,
            opts,
            replicas,
            &mut rng,
            &device,
            step,
            lr,
            &mut || stop.load(Ordering::Relaxed),
        )?
        else {
            interrupted = true;
            break;
        };
        last_loss = step_loss;
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
        model = adamw_step(&mut optim, lr, opts.weight_decay, model, grads);

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

    /// A scratch directory name that no other test (or run of the suite) uses.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-train-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Tiny corpus, model and options for the tests of how a run treats its directory.
    fn small_run(steps: usize) -> (Corpus, GptConfig, TrainOptions) {
        let corpus = Corpus::from_bytes(demo_corpus(20_000, 5).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let opts = TrainOptions {
            steps,
            batch: 2,
            seq: 32,
            warmup: 1,
            eval_every: 0,
            save_every: 0,
            log_every: 1,
            ..TrainOptions::default()
        };
        (corpus, cfg, opts)
    }

    /// Gains (one-dimensional parameters) and everything else, flattened, of a model.
    fn gains_and_matrices(model: &Gpt) -> (Vec<f32>, Vec<f32>) {
        struct Split(Vec<f32>, Vec<f32>);
        impl ModuleVisitor for Split {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                let v = p.val().into_data().try_to_vec::<f32>().unwrap();
                if D == 1 {
                    self.0.extend(v);
                } else {
                    self.1.extend(v);
                }
            }
        }
        let mut s = Split(vec![], vec![]);
        model.visit(&mut s);
        (s.0, s.1)
    }

    #[test]
    fn weight_decay_reaches_the_matrices_but_not_the_norm_gains() {
        let device = crate::train_device();
        let cfg = Preset::Nano.config(16);
        let model = cfg.init_seeded(&device, 1);
        let data = |shift: usize| {
            let v: Vec<i32> = (0..2 * 16)
                .map(|k| ((k * 7 + shift) % 256) as i32)
                .collect();
            Tensor::<2, Int>::from_data(TensorData::new(v, [2, 16]), &device)
        };
        let grads = || GradientsParams::from_grads(model.loss(data(0), data(1)).backward(), &model);
        // A decay big enough to see: 5 % of every weight per step.
        let (lr, wd) = (0.1f64, 0.5f32);
        let adam = |wd: f32| {
            AdamWConfig::new()
                .with_beta_2(0.95)
                .with_weight_decay(wd)
                .init()
        };

        let without = adam(0.0).step(lr, model.clone(), grads());
        let with = adam(wd).step(lr, model.clone(), grads());
        let spared = adamw_step(&mut adam(wd), lr, wd, model.clone(), grads());
        let (g_without, _) = gains_and_matrices(&without);
        let (g_with, m_with) = gains_and_matrices(&with);
        let (g_spared, m_spared) = gains_and_matrices(&spared);
        assert!(!g_without.is_empty());

        // The plain step does decay the gains (this is what the test would catch) ...
        let moved = g_with
            .iter()
            .zip(&g_without)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            moved > 0.04,
            "the decay should be visible on the gains: {moved}"
        );
        // ... the spared step leaves them as if there were no decay ...
        for (a, b) in g_spared.iter().zip(&g_without) {
            assert!((a - b).abs() < 1e-6, "gain {a} vs {b}");
        }
        // ... and every matrix is decayed exactly as before.
        assert_eq!(m_spared, m_with);
    }

    #[test]
    fn the_training_loop_leaves_the_norm_gains_alone_whatever_the_weight_decay() {
        // A decay of 100 * lr = 10 % per step would take the gains from 1 to about 0.6 in five
        // steps; their own updates (lr = 1e-3) move them by a few thousandths.
        let dir = scratch("gains");
        let (corpus, cfg, mut opts) = small_run(5);
        opts.weight_decay = 100.0;
        opts.lr = 1e-3;
        opts.warmup = 0;
        let never = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &never, &mut |_| {}).unwrap();
        let (model, _, _) = checkpoint::load(&dir, &crate::train_device()).unwrap();
        let (gains, matrices) = gains_and_matrices(&model);
        assert!(!gains.is_empty());
        for g in &gains {
            assert!((g - 1.0).abs() < 0.05, "a gain moved to {g}");
        }
        // ... while the matrices were decayed hard (their initial scale is about 0.02).
        let rms = (matrices.iter().map(|v| v * v).sum::<f32>() / matrices.len() as f32).sqrt();
        assert!(rms < 0.02 * 0.7, "the matrices were not decayed: rms {rms}");
    }

    #[test]
    fn a_run_that_is_stopped_and_resumed_equals_an_uninterrupted_one() {
        // Until 0.6.0 the windows of a run came from one generator that a resume restarted, so a
        // resumed run trained on other data than the same run without the pause.
        for replicas in [1usize, 2] {
            let (corpus, cfg, mut opts) = small_run(10);
            opts.replicas = replicas;
            let straight = scratch(&format!("straight-{replicas}"));
            let paused = scratch(&format!("paused-{replicas}"));
            let never = AtomicBool::new(false);
            train(&corpus, &cfg, &opts, &straight, false, &never, &mut |_| {}).unwrap();

            let stop = AtomicBool::new(false);
            let first = train(&corpus, &cfg, &opts, &paused, false, &stop, &mut |e| {
                if let Event::Step { step: 4, .. } = e {
                    stop.store(true, Ordering::Relaxed);
                }
            })
            .unwrap();
            assert!(first.interrupted && first.final_step == 4, "{first:?}");
            let rest = train(&corpus, &cfg, &opts, &paused, true, &never, &mut |_| {}).unwrap();
            assert_eq!(rest.final_step, 10);

            // The files themselves differ in their bytes (they carry the parameters' random ids),
            // so the weights are compared as numbers: not "about equal" but equal.
            let device = crate::train_device();
            let (a, _, _) = checkpoint::load(&straight, &device).unwrap();
            let (b, _, _) = checkpoint::load(&paused, &device).unwrap();
            assert_eq!(
                gains_and_matrices(&a),
                gains_and_matrices(&b),
                "weights differ after a pause ({replicas} replicas)"
            );
        }
    }

    #[test]
    fn a_fresh_run_does_not_silently_replace_a_trained_checkpoint() {
        // The commonest slip: running the same command again without --resume. Until 0.4.0 the
        // first save of the second run replaced the trained model without a word.
        let dir = scratch("no-clobber");
        let (corpus, cfg, opts) = small_run(3);
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        let weights = dir.join(checkpoint::WEIGHTS_FILE);
        let before = std::fs::read(&weights).unwrap();

        let err = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap_err();
        assert!(
            err.contains("--resume") && err.contains("--overwrite"),
            "{err}"
        );
        assert!(err.contains("pasul 3"), "{err}");
        assert_eq!(
            std::fs::read(&weights).unwrap(),
            before,
            "not a byte changed"
        );

        // Asking for it is allowed.
        let again = TrainOptions {
            overwrite: true,
            seed: 2,
            ..opts.clone()
        };
        train(&corpus, &cfg, &again, &dir, false, &stop, &mut |_| {}).unwrap();
        assert_ne!(std::fs::read(&weights).unwrap(), before);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_without_the_progress_record_is_refused_not_restarted_from_zero() {
        // With meta.json gone, `--resume` used to carry on from step 0 on the trained weights:
        // warm-up again, statistics wiped, "untrained model" announced, and the next save
        // recorded the mixture as the checkpoint.
        let dir = scratch("no-meta");
        let (corpus, cfg, opts) = small_run(3);
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        std::fs::remove_file(dir.join(checkpoint::META_FILE)).unwrap();
        let weights = std::fs::read(dir.join(checkpoint::WEIGHTS_FILE)).unwrap();

        let more = TrainOptions { steps: 5, ..opts };
        let err = train(&corpus, &cfg, &more, &dir, true, &stop, &mut |_| {}).unwrap_err();
        assert!(err.contains("meta.json"), "{err}");
        assert_eq!(
            std::fs::read(dir.join(checkpoint::WEIGHTS_FILE)).unwrap(),
            weights
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_refuses_files_of_two_different_saves() {
        // The optimizer of step 3 next to the weights of step 5: both intact, neither complaining.
        let (a, b) = (scratch("mix-a"), scratch("mix-b"));
        let (corpus, cfg, opts) = small_run(3);
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &a, false, &stop, &mut |_| {}).unwrap();
        for entry in std::fs::read_dir(&a).unwrap() {
            let entry = entry.unwrap();
            std::fs::create_dir_all(&b).unwrap();
            std::fs::copy(entry.path(), b.join(entry.file_name())).unwrap();
        }
        let more = TrainOptions {
            steps: 5,
            ..opts.clone()
        };
        train(&corpus, &cfg, &more, &b, true, &stop, &mut |_| {}).unwrap();

        std::fs::copy(
            b.join(checkpoint::OPTIMIZER_FILE),
            a.join(checkpoint::OPTIMIZER_FILE),
        )
        .unwrap();
        let err = train(&corpus, &cfg, &more, &a, true, &stop, &mut |_| {}).unwrap_err();
        assert!(err.contains(checkpoint::OPTIMIZER_FILE), "{err}");
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
    }

    #[test]
    fn resume_completes_a_save_that_was_cut_short() {
        // A run that died after writing every file of step 5 but before publishing any of them:
        // the next `--resume` must finish that save, not silently go back to step 3.
        let (a, b) = (scratch("cut-a"), scratch("cut-b"));
        let (corpus, cfg, opts) = small_run(3);
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &a, false, &stop, &mut |_| {}).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        for entry in std::fs::read_dir(&a).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), b.join(entry.file_name())).unwrap();
        }
        let more = TrainOptions {
            steps: 5,
            ..opts.clone()
        };
        train(&corpus, &cfg, &more, &b, true, &stop, &mut |_| {}).unwrap();
        for name in [
            "tokenizer.json",
            "optimizer.bpk",
            "weights.bpk",
            "config.json",
            "meta.json",
        ] {
            let tmp = crate::fsutil::tmp_path(&a.join(name));
            std::fs::copy(b.join(name), tmp).unwrap();
        }

        let mut recovered = None;
        let report = train(&corpus, &cfg, &more, &a, true, &stop, &mut |e| {
            if let Event::Recovered(r) = e {
                recovered = Some(r);
            }
        })
        .unwrap();
        assert_eq!(recovered, Some(checkpoint::Recovery::Completed(5)));
        assert_eq!(report.final_step, 5);
        assert_eq!(report.steps_run, 0, "step 5 was already on disk");
        assert_eq!(
            std::fs::read(a.join(checkpoint::WEIGHTS_FILE)).unwrap(),
            std::fs::read(b.join(checkpoint::WEIGHTS_FILE)).unwrap()
        );
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
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
    fn options_validation_rejects_values_that_would_break_the_run() {
        let cfg = Preset::Nano.config(64);
        let base = TrainOptions {
            seq: 64,
            ..TrainOptions::default()
        };
        assert!(base.validate(&cfg).is_ok());
        let bad: Vec<(&str, TrainOptions)> = vec![
            (
                "beta2 = 1",
                TrainOptions {
                    beta2: 1.0,
                    ..base.clone()
                },
            ),
            (
                "beta2 < 0",
                TrainOptions {
                    beta2: -0.1,
                    ..base.clone()
                },
            ),
            (
                "beta2 NaN",
                TrainOptions {
                    beta2: f32::NAN,
                    ..base.clone()
                },
            ),
            (
                "lr x weight decay of 1 or more",
                TrainOptions {
                    lr: 0.5,
                    weight_decay: 2.0,
                    ..base.clone()
                },
            ),
            (
                "negative weight decay",
                TrainOptions {
                    weight_decay: -0.5,
                    ..base.clone()
                },
            ),
            (
                "NaN weight decay",
                TrainOptions {
                    weight_decay: f32::NAN,
                    ..base.clone()
                },
            ),
            (
                "negative clip",
                TrainOptions {
                    grad_clip: -1.0,
                    ..base.clone()
                },
            ),
            (
                "NaN clip (used to switch clipping off silently)",
                TrainOptions {
                    grad_clip: f32::NAN,
                    ..base.clone()
                },
            ),
            (
                "infinite clip",
                TrainOptions {
                    grad_clip: f32::INFINITY,
                    ..base.clone()
                },
            ),
            (
                "accum = usize::MAX",
                TrainOptions {
                    accum: usize::MAX,
                    ..base.clone()
                },
            ),
            (
                "tokens per step overflow",
                TrainOptions {
                    batch: usize::MAX / 2,
                    ..base.clone()
                },
            ),
        ];
        for (what, opts) in bad {
            assert!(opts.validate(&cfg).is_err(), "{what} must be rejected");
        }
        // Zero clipping is the documented way to switch it off.
        assert!(
            TrainOptions {
                grad_clip: 0.0,
                ..base.clone()
            }
            .validate(&cfg)
            .is_ok()
        );
    }

    #[test]
    fn resuming_past_the_requested_steps_is_an_error_and_at_them_a_quiet_no_op() {
        let dir = std::env::temp_dir().join(format!("germen-resume-end-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(30_000, 2).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let opts = TrainOptions {
            steps: 6,
            batch: 4,
            seq: 32,
            eval_every: 0,
            save_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        let before = checkpoint::load_meta(&dir).unwrap();
        assert_eq!(before.step, 6);

        // `--steps` is the TOTAL: asking for fewer than were done is a mistake worth a message
        // (it used to rewrite the checkpoint with the new run's settings and say nothing).
        let fewer = TrainOptions {
            steps: 3,
            batch: 8,
            ..opts.clone()
        };
        let err = train(&corpus, &cfg, &fewer, &dir, true, &stop, &mut |_| {}).unwrap_err();
        assert!(err.contains("pasul 6") && err.contains("--steps"), "{err}");
        assert_eq!(
            checkpoint::load_meta(&dir).unwrap(),
            before,
            "nothing may be rewritten"
        );

        // Asking for exactly what is done is how a "keep resuming until finished" script ends:
        // success, no work, no rewrite, no sample.
        let (mut saved, mut sampled) = (0, 0);
        let same = TrainOptions {
            batch: 8,
            ..opts.clone()
        };
        let report = train(&corpus, &cfg, &same, &dir, true, &stop, &mut |e| match e {
            Event::Saved { .. } => saved += 1,
            Event::Sample { .. } => sampled += 1,
            _ => {}
        })
        .unwrap();
        assert_eq!((report.final_step, report.steps_run), (6, 0));
        assert!(!report.interrupted);
        assert_eq!(
            (saved, sampled),
            (0, 0),
            "no checkpoint and no sample for a no-op"
        );
        assert_eq!(
            checkpoint::load_meta(&dir).unwrap(),
            before,
            "nothing may be rewritten"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn metadata_counts_the_corpus_in_bytes_even_when_its_tokens_are_fewer() {
        let dir = std::env::temp_dir().join(format!("germen-meta-bytes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let text = demo_corpus(40_000, 3).into_bytes();
        let bpe = crate::bpe::Bpe::train(&text, 400, &mut |_, _| {}).unwrap();
        let corpus = Corpus::build(&text, 0.1, Tokenizer::Bpe(bpe)).unwrap();
        assert!(
            corpus.len() * 2 < corpus.bytes,
            "BPE must compress for this test to mean anything"
        );
        let cfg = Preset::Nano
            .config(32)
            .with_vocab(corpus.tokenizer.vocab_size());
        let opts = TrainOptions {
            steps: 2,
            batch: 4,
            seq: 32,
            eval_every: 0,
            save_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        // The field is called `corpus_bytes`; up to 0.2.0 it held the number of TOKENS.
        assert_eq!(
            checkpoint::load_meta(&dir).unwrap().corpus_bytes,
            corpus.bytes
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_run_with_a_tiny_context_still_ends_with_its_sample() {
        // The end-of-run sample asks for a 128-token window; the model only has 4 positions.
        let dir = std::env::temp_dir().join(format!("germen-tiny-ctx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(20_000, 4).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(4);
        let opts = TrainOptions {
            steps: 2,
            batch: 2,
            seq: 4,
            eval_every: 0,
            save_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        let mut sampled = false;
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |e| {
            sampled |= matches!(e, Event::Sample { .. });
        })
        .unwrap();
        assert!(sampled);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every gradient of `model`, flattened in parameter order.
    fn flat_gradients(grads: &GradientsParams, model: &Gpt) -> Vec<f32> {
        struct Collect<'a>(&'a GradientsParams, Vec<f32>);
        impl burn::module::ModuleVisitor for Collect<'_> {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
                if let Some(g) = self.0.get::<D>(param.id) {
                    self.1.extend(g.into_data().try_to_vec::<f32>().unwrap());
                }
            }
        }
        let mut all = Collect(grads, Vec::new());
        burn::module::Module::visit(model, &mut all);
        all.1
    }

    #[test]
    fn replicas_give_the_same_gradients_as_one_graph() {
        let device = crate::train_device();
        let cfg = Preset::Nano.config(16);
        let model = cfg.init_seeded(&device, 3);
        let mut rng = Rng::new(11);
        // Seven windows: split in 2, 3 and 7 the parts are unequal, which the weights must handle.
        let batch = Batch::random(7, 16, cfg.vocab_size, &mut rng);

        let (reference, reference_loss) =
            micro_batch_gradients(&model, &batch, 1, 1, &device).unwrap();
        let want = flat_gradients(&reference, &model);
        let scale = want.iter().fold(0f32, |m, g| m.max(g.abs()));
        assert!(scale > 0.0 && want.iter().all(|g| g.is_finite()));

        for replicas in [2, 3, 7, 50] {
            let (grads, loss) =
                micro_batch_gradients(&model, &batch, replicas, 1, &device).unwrap();
            let got = flat_gradients(&grads, &model);
            assert_eq!(got.len(), want.len());
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 2e-4 * scale,
                "{replicas} replicas: worst difference {worst} (scale {scale})"
            );
            assert!(
                (loss - reference_loss).abs() < 1e-4,
                "{replicas}: loss {loss} vs {reference_loss}"
            );
        }

        // With accumulation the gradient is that of loss / accum, whatever the replica count.
        let (quarter, _) = micro_batch_gradients(&model, &batch, 3, 4, &device).unwrap();
        let got = flat_gradients(&quarter, &model);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b / 4.0).abs())
            .fold(0f32, f32::max);
        assert!(
            worst <= 2e-4 * scale / 4.0,
            "accum 4: worst difference {worst}"
        );
    }

    /// Trains a nano model for `steps` steps; returns the loss of every step and the logits of
    /// the saved model on a fixed input.
    fn nano_run(name: &str, opts: &TrainOptions) -> (Vec<f32>, Vec<f32>) {
        let dir = std::env::temp_dir().join(format!("germen-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(30_000, 4).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(opts.seq);
        let mut losses = Vec::new();
        let stop = AtomicBool::new(false);
        train(&corpus, &cfg, opts, &dir, false, &stop, &mut |e| {
            if let Event::Step { loss, .. } = e {
                losses.push(loss);
            }
        })
        .unwrap();
        let (model, _, _) = checkpoint::load(&dir, &Device::flex()).unwrap();
        let ids: Vec<i32> = (0..opts.seq as i32).map(|i| (i * 7 + 3) % 256).collect();
        let x = Tensor::<2, Int>::from_data(TensorData::new(ids, [1, opts.seq]), &Device::flex());
        let logits = model.forward(x).into_data().try_to_vec::<f32>().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        (losses, logits)
    }

    #[test]
    fn replicated_training_is_reproducible_for_a_fixed_seed() {
        // The initial weights are drawn before the replicas race to use them, and the gradient
        // sums have a fixed order: two runs with the same settings must agree to the last bit.
        for replicas in [1, 2, 3] {
            let opts = TrainOptions {
                steps: 6,
                batch: 6,
                seq: 16,
                lr: 3e-3,
                warmup: 2,
                eval_every: 0,
                save_every: 0,
                log_every: 1,
                replicas,
                ..TrainOptions::default()
            };
            let a = nano_run(&format!("repro-a-{replicas}"), &opts);
            let b = nano_run(&format!("repro-b-{replicas}"), &opts);
            assert_eq!(a.0.len(), 6);
            assert_eq!(
                a.0, b.0,
                "{replicas} replicas: losses differ between identical runs"
            );
            assert_eq!(
                a.1, b.1,
                "{replicas} replicas: weights differ between identical runs"
            );
        }
    }

    #[test]
    fn replicated_training_learns_like_the_ordinary_step() {
        let mk = |replicas| TrainOptions {
            steps: 24,
            batch: 6,
            accum: 2,
            seq: 32,
            lr: 3e-3,
            warmup: 3,
            eval_every: 0,
            save_every: 0,
            log_every: 4,
            replicas,
            ..TrainOptions::default()
        };
        let (single, _) = nano_run("learn-1", &mk(1));
        let (many, _) = nano_run("learn-3", &mk(3));
        for losses in [&single, &many] {
            assert!(losses.last().unwrap() < &(losses[0] - 0.3), "{losses:?}");
        }
        // Same data, same maths, different rounding: the curves stay close.
        for (a, b) in single.iter().zip(&many) {
            assert!((a - b).abs() < 0.15, "{single:?} vs {many:?}");
        }
    }

    /// Runs `work` on a thread of its own and fails the test if it has not finished in `secs`
    /// seconds: a deadlock must fail one test, not hang the whole run.
    fn within<T: Send + 'static>(secs: u64, work: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
        rx.recv_timeout(std::time::Duration::from_secs(secs))
            .expect("deadlock: the work did not finish in time")
    }

    #[test]
    fn replicas_never_deadlock_whatever_the_pool_size() {
        // The attempt at data parallelism that 0.2 rejected let several threads share one autodiff
        // graph, and hung (every thread waiting, 0 % CPU) with more workers than pool threads.
        // Replicas have graphs of their own: pools of one to five threads, with fewer, as many and
        // more replicas than threads, must all get through.
        for threads in [1usize, 2, 3, 5] {
            for replicas in [2usize, 3, 4, 8] {
                within(180, move || {
                    let cfg = Preset::Nano.config(16);
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    pool.install(|| {
                        let device = crate::train_device();
                        let model = cfg.init_seeded(&device, 5);
                        let mut rng = Rng::new(2);
                        let batch = Batch::random(8, 16, cfg.vocab_size, &mut rng);
                        for _ in 0..2 {
                            let (grads, loss) =
                                micro_batch_gradients(&model, &batch, replicas, 1, &device)
                                    .unwrap();
                            assert!(loss.is_finite(), "{threads} threads, {replicas} replicas");
                            drop(grads);
                        }
                    });
                });
            }
        }
    }

    /// Makes the first float parameter NaN and leaves the others alone.
    struct PoisonFirst(bool);

    impl burn::module::ModuleMapper for PoisonFirst {
        fn map_float<const D: usize>(
            &mut self,
            param: burn::module::Param<Tensor<D>>,
        ) -> burn::module::Param<Tensor<D>> {
            if std::mem::replace(&mut self.0, true) {
                param
            } else {
                param.map(|t| t.mul_scalar(f32::NAN))
            }
        }
    }

    #[test]
    fn a_non_finite_loss_is_reported_whatever_the_replica_count() {
        let device = crate::train_device();
        let cfg = Preset::Nano.config(16);
        let model = cfg.init_seeded(&device, 9);
        let mut rng = Rng::new(4);
        let batch = Batch::random(6, 16, cfg.vocab_size, &mut rng);
        let poisoned = burn::module::Module::map(model, &mut PoisonFirst(false));
        for replicas in [1, 3, 6] {
            match micro_batch_gradients(&poisoned, &batch, replicas, 1, &device) {
                Ok(_) => panic!("{replicas} replicas: NaN weights must give an error"),
                Err(bad) => assert!(!bad.is_finite(), "{replicas} replicas: {bad}"),
            }
        }
    }

    #[test]
    fn replica_planning_respects_requests_windows_and_memory() {
        let cfg = Preset::Nano.config(64);
        assert_eq!(plan_replicas(&cfg, 8, 64, 1), 1);
        assert_eq!(plan_replicas(&cfg, 8, 64, 5), 5);
        assert_eq!(
            plan_replicas(&cfg, 3, 64, 100),
            3,
            "never more than the windows"
        );
        assert_eq!(plan_replicas(&cfg, 1, 64, 0), 1);
        let auto = plan_replicas(&cfg, 8, 64, 0);
        assert!(
            (1..=8.min(crate::sys::worker_threads())).contains(&auto),
            "{auto}"
        );

        // A model whose gradient copies cannot fit is not replicated by itself.
        let huge = GptConfig::new(256, 64, 1024, 16, 16_384, 65_536);
        assert_eq!(plan_replicas(&huge, 8, 64, 0), 1);
        assert!(
            cfg.estimate_train_bytes_replicated(8, 64, 4)
                > cfg.estimate_train_bytes_replicated(8, 64, 2)
        );
        assert_eq!(
            cfg.estimate_train_bytes_replicated(8, 64, 1),
            cfg.estimate_train_bytes(8, 64)
        );
    }

    #[test]
    fn the_start_event_reports_the_replicas_in_use() {
        let dir = std::env::temp_dir().join(format!("germen-start-rep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(20_000, 4).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(16);
        let opts = TrainOptions {
            steps: 1,
            batch: 4,
            seq: 16,
            replicas: 3,
            eval_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        let mut seen = 0;
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |e| {
            if let Event::Start { replicas, .. } = e {
                seen = replicas;
            }
        })
        .unwrap();
        assert_eq!(seen, 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stop_request_is_honoured_between_micro_batches() {
        let device = crate::train_device();
        let corpus = Corpus::from_bytes(demo_corpus(30_000, 9).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init_seeded(&device, 1);
        let opts = TrainOptions {
            batch: 2,
            seq: 32,
            accum: 8,
            ..TrainOptions::default()
        };

        // Stop requested while the third micro-batch is about to start: no half step comes back
        // and exactly two windows were drawn, not eight.
        let mut rng = Rng::new(5);
        let mut asked = 0;
        let out = accumulate_gradients(
            &model,
            &corpus,
            &opts,
            1,
            &mut rng,
            &device,
            0,
            1e-3,
            &mut || {
                asked += 1;
                asked > 2
            },
        )
        .unwrap();
        assert!(out.is_none());
        assert_eq!(asked, 3);
        let mut reference = Rng::new(5);
        for _ in 0..2 {
            corpus.sample_batch(Split::Train, 2, 32, &mut reference);
        }
        assert_eq!(rng.next_u64(), reference.next_u64());

        // Never asked to stop: all eight micro-batches, a sensible loss and real gradients.
        let mut rng = Rng::new(5);
        let (grads, loss) = accumulate_gradients(
            &model,
            &corpus,
            &opts,
            1,
            &mut rng,
            &device,
            0,
            1e-3,
            &mut || false,
        )
        .unwrap()
        .unwrap();
        assert!((loss - 256f32.ln()).abs() < 0.5, "loss {loss}");
        assert!(gradients::global_norm(&grads, &model) > 0.0);
    }

    #[test]
    fn a_directory_in_use_by_another_run_is_refused_and_released_afterwards() {
        let dir = std::env::temp_dir().join(format!("germen-claim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let corpus = Corpus::from_bytes(demo_corpus(20_000, 6).into_bytes(), 0.1).unwrap();
        let cfg = Preset::Nano.config(32);
        let opts = TrainOptions {
            steps: 2,
            batch: 2,
            seq: 32,
            eval_every: 0,
            ..TrainOptions::default()
        };
        let stop = AtomicBool::new(false);
        {
            let _other_run = DirLock::acquire(&dir).unwrap();
            let err = train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap_err();
            assert!(err.contains("altă rulare"), "{err}");
            assert!(!dir.join(checkpoint::WEIGHTS_FILE).exists());
        }
        // The other run has ended: the directory is ours again.
        train(&corpus, &cfg, &opts, &dir, false, &stop, &mut |_| {}).unwrap();
        assert!(dir.join(checkpoint::WEIGHTS_FILE).exists());
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
        let mut model = cfg.init_seeded(&device, 1);
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
        let (mut ob, found) = checkpoint::load_optimizer(&dir, adam(), &mb).unwrap();
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
