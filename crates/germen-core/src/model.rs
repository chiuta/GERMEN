//! The GPT-style network, written directly on Burn tensors.

use crate::config::GptConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::{relu, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use std::sync::{Mutex, PoisonError};

/// Standard deviation of the weight initialisation (GPT-2 convention).
const INIT_STD: f64 = 0.02;

fn linear(d_in: usize, d_out: usize, std: f64, device: &Device) -> Linear {
    LinearConfig::new(d_in, d_out)
        .with_bias(false)
        .with_initializer(Initializer::Normal { mean: 0.0, std })
        .init(device)
}

/// Rotates pairs of channels by a position-dependent angle (rotary embeddings, "rotate-half"
/// layout). `x` is `[batch, heads, seq, head_dim]`; `cos`/`sin` are `[1, 1, seq, head_dim / 2]`.
fn apply_rope(x: Tensor<4>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<4> {
    let [b, h, t, hd] = x.dims();
    let half = hd / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..t, 0..half]);
    let x2 = x.slice([0..b, 0..h, 0..t, half..hd]);
    let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let o2 = x1 * sin.clone() + x2 * cos.clone();
    Tensor::cat(vec![o1, o2], 3)
}

/// Value added to the scores of positions a query must not see. Finite on purpose: `-inf` would
/// make `-inf - max` and `0 * -inf` appear in the backward pass.
const MASKED: f32 = -1.0e30;

/// Row-major `[n, n]` additive causal mask: 0 on and below the diagonal, [`MASKED`] above it.
fn causal_mask_values(n: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; n * n];
    for row in 0..n {
        mask[row * n + row + 1..(row + 1) * n].fill(MASKED);
    }
    mask
}

/// Queries are processed in blocks of this many positions during training (see
/// [`causal_attention_blocks`]): twice the head dimension. More blocks skip more of the masked
/// upper triangle but cost more small operations; on two threads at 256 tokens this was the best
/// of 32, 64, 128 and 256 for head dimensions 32 (`nano`, `micro`) and 64 (`mini`).
fn attention_block(head_dim: usize) -> usize {
    (2 * head_dim).clamp(32, 256)
}

/// Causal attention for training, `softmax(q kᵀ / √d + mask) v`, written with ordinary
/// differentiable tensor operations.
///
/// Compared with Burn's `attention`, which under autodiff expands into a long chain of
/// operations over the full `[seq, seq]` score matrix, this version
/// - never computes the masked upper-right blocks: queries are processed in blocks and a block
///   only looks at the keys up to its own end (about 25–30 % faster on `mini`, see
///   `examples/profile_attn.rs`);
/// - folds the `1/√d` scale into the queries instead of scaling every score.
///
/// The result equals Burn's attention up to floating-point rounding (tested).
/// `mask` is `[1, 1, seq, seq]` with `0` where attention is allowed and [`MASKED`] elsewhere.
fn causal_attention_blocks(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    mask: &Tensor<4>,
    block: usize,
) -> Tensor<4> {
    let [b, h, t, hd] = q.dims();
    let q = q.mul_scalar(1.0 / (hd as f32).sqrt());
    if t <= block {
        let scores = q.matmul(k.swap_dims(2, 3)) + mask.clone();
        return softmax(scores, 3).matmul(v);
    }
    let mut outs = Vec::with_capacity(t.div_ceil(block));
    let mut lo = 0;
    while lo < t {
        let hi = (lo + block).min(t);
        let qi = q.clone().slice([0..b, 0..h, lo..hi, 0..hd]);
        let ki = k.clone().slice([0..b, 0..h, 0..hi, 0..hd]);
        let vi = v.clone().slice([0..b, 0..h, 0..hi, 0..hd]);
        let mi = mask.clone().slice([0..1, 0..1, lo..hi, 0..hi]);
        let scores = qi.matmul(ki.swap_dims(2, 3)) + mi;
        outs.push(softmax(scores, 3).matmul(vi));
        lo = hi;
    }
    Tensor::cat(outs, 2)
}

/// Part `which` (0 = queries, 1 = keys, 2 = values) of a fused `[b, t, 3d]` projection, as
/// `[b, heads, t, head_dim]`.
fn split_heads(qkv: &Tensor<3>, which: usize, heads: usize) -> Tensor<4> {
    let [b, t, d3] = qkv.dims();
    let d = d3 / 3;
    qkv.clone()
        .slice([0..b, 0..t, which * d..(which + 1) * d])
        .reshape([b, t, heads, d / heads])
        .swap_dims(1, 2)
}

/// Keys and values of the tokens the model has already read — one `[batch, heads, len, head_dim]`
/// pair per layer — so that generating the next token costs one token's worth of computation
/// instead of re-reading the whole context. Create one with [`Gpt::new_cache`].
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<Option<(Tensor<4>, Tensor<4>)>>,
    len: usize,
}

impl KvCache {
    /// Number of tokens read so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing has been read yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Forgets everything.
    pub fn clear(&mut self) {
        self.len = 0;
        self.layers.iter_mut().for_each(|l| *l = None);
    }
}

/// Causal multi-head self-attention.
#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    qkv: Linear,
    proj: Linear,
    n_head: usize,
}

impl CausalSelfAttention {
    fn new(cfg: &GptConfig, device: &Device) -> Self {
        let d = cfg.d_model;
        // The output projections of residual branches are scaled down (GPT-2) so the
        // variance of the residual stream does not grow with depth.
        let proj_std = INIT_STD / ((2 * cfg.n_layer) as f64).sqrt();
        CausalSelfAttention {
            qkv: linear(d, 3 * d, INIT_STD, device),
            proj: linear(d, d, proj_std, device),
            n_head: cfg.n_head,
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.n_head;

        let qkv = self.qkv.forward(x); // [b, t, 3d]
        let q = apply_rope(split_heads(&qkv, 0, h), cos, sin);
        let k = apply_rope(split_heads(&qkv, 1, h), cos, sin);
        let v = split_heads(&qkv, 2, h);

        let y = if q.is_autodiff() {
            causal_attention_blocks(q, k, v, mask, attention_block(d / h)) // training
        } else {
            // Evaluation and generation: Burn's fused, multi-threaded kernel.
            let options = AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            };
            attention(q, k, v, None, None, options)
        }; // [b, h, t, hd]
        let y = y.swap_dims(1, 2).reshape([b, t, d]);
        self.proj.forward(y)
    }

    /// Attention for `t` new tokens that come after what `cache` already holds. `cos`/`sin` are
    /// the rotary tables for the new positions; `mask` is `[1, 1, t, total]`.
    fn forward_cached(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
        cache: &mut Option<(Tensor<4>, Tensor<4>)>,
    ) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.n_head;
        let hd = d / h;

        let qkv = self.qkv.forward(x);
        let q = apply_rope(split_heads(&qkv, 0, h), cos, sin);
        let k = apply_rope(split_heads(&qkv, 1, h), cos, sin);
        let v = split_heads(&qkv, 2, h);
        let (k, v) = match cache.take() {
            Some((pk, pv)) => (Tensor::cat(vec![pk, k], 2), Tensor::cat(vec![pv, v], 2)),
            None => (k, v),
        };
        *cache = Some((k.clone(), v.clone()));

        let scores = q
            .mul_scalar(1.0 / (hd as f32).sqrt())
            .matmul(k.swap_dims(2, 3))
            + mask.clone();
        let y = softmax(scores, 3).matmul(v); // [b, h, t, hd]
        self.proj.forward(y.swap_dims(1, 2).reshape([b, t, d]))
    }
}

/// Position-wise feed-forward block with a squared-ReLU activation (`relu(x)²`).
///
/// Chosen over GELU for speed on CPUs. Measured with `examples/profile_ops.rs` on Burn 0.22.1's
/// `flex` backend: `erf`, `tanh` and `exp` cost several nanoseconds per element and do not use
/// more than one thread, so GELU took ~13× longer than `relu(x)²` on the `mini` shapes, while
/// `relu` and a multiplication are cheap. The Primer paper (So et al., 2021) also reported better
/// training efficiency for squared ReLU than for GELU.
#[derive(Module, Debug)]
pub struct Mlp {
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &GptConfig, device: &Device) -> Self {
        let proj_std = INIT_STD / ((2 * cfg.n_layer) as f64).sqrt();
        Mlp {
            up: linear(cfg.d_model, cfg.d_ff, INIT_STD, device),
            down: linear(cfg.d_ff, cfg.d_model, proj_std, device),
        }
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = relu(self.up.forward(x));
        self.down.forward(hidden.clone() * hidden)
    }
}

/// One pre-norm transformer block.
#[derive(Module, Debug)]
pub struct Block {
    norm1: RmsNorm,
    attn: CausalSelfAttention,
    norm2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &GptConfig, device: &Device) -> Self {
        Block {
            norm1: RmsNormConfig::new(cfg.d_model)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            attn: CausalSelfAttention::new(cfg, device),
            norm2: RmsNormConfig::new(cfg.d_model)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            mlp: Mlp::new(cfg, device),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
    ) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), cos, sin, mask);
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }

    fn forward_cached(
        &self,
        x: Tensor<3>,
        cos: &Tensor<4>,
        sin: &Tensor<4>,
        mask: &Tensor<4>,
        cache: &mut Option<(Tensor<4>, Tensor<4>)>,
    ) -> Tensor<3> {
        let a = self
            .attn
            .forward_cached(self.norm1.forward(x.clone()), cos, sin, mask, cache);
        let x = x + a;
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }
}

/// The complete language model: byte embeddings → N blocks → RMSNorm → vocabulary logits.
#[derive(Module, Debug)]
pub struct Gpt {
    wte: Embedding,
    blocks: Vec<Block>,
    norm_f: RmsNorm,
    head: Linear,
    /// Rotary tables, `[block_size, head_dim / 2]`. Constants, not trained.
    rope_cos: Tensor<2>,
    rope_sin: Tensor<2>,
    /// Additive causal mask, `[block_size, block_size]`: 0 on and below the diagonal, [`MASKED`]
    /// above it. Constant, not trained.
    causal_mask: Tensor<2>,
    block_size: usize,
}

/// Burn keeps ONE random generator per device, shared by every thread of the process, and each
/// initial weight is a draw from it. Seeding it and drawing the weights therefore has to be one
/// uninterrupted step: two threads doing it at once (two training runs in one process, or two
/// tests) would each take a share of the other's stream, and neither would get the weights its seed
/// promises. This lock makes the step uninterrupted; everything in GERMEN that draws goes through
/// [`exclusive_generator`].
static GENERATOR: Mutex<()> = Mutex::new(());

/// Runs `f` while no other thread of this crate seeds or draws from the device generator.
pub(crate) fn exclusive_generator<R>(f: impl FnOnce() -> R) -> R {
    // The guard protects no data, so a panic in another holder leaves nothing to distrust.
    let _turn = GENERATOR.lock().unwrap_or_else(PoisonError::into_inner);
    f()
}

impl GptConfig {
    /// Builds a freshly initialised model on `device`, whose weights are drawn from the device's
    /// random generator the first time something reads them.
    ///
    /// That is the right constructor for a model that is about to be overwritten from a
    /// checkpoint (nothing is drawn for it), and the wrong one for a model that is going to be
    /// trained: its weights would depend on when, and on which thread, they are first touched. Use
    /// [`GptConfig::init_seeded`] for that.
    ///
    /// Panics if the configuration is invalid; call [`GptConfig::validate`] first for a
    /// recoverable error.
    pub fn init(&self, device: &Device) -> Gpt {
        self.validate().expect("invalid GptConfig");
        let half = self.head_dim() / 2;
        // Inverse frequencies theta^(-2i/d), i in 0..half, then angle = position * freq.
        let mut cos = Vec::with_capacity(self.block_size * half);
        let mut sin = Vec::with_capacity(self.block_size * half);
        for pos in 0..self.block_size {
            for i in 0..half {
                let freq =
                    (self.rope_theta as f64).powf(-(2.0 * i as f64) / self.head_dim() as f64);
                let angle = pos as f64 * freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        let table = |v: Vec<f32>| {
            Tensor::<2>::from_data(TensorData::new(v, [self.block_size, half]), device)
        };
        let n = self.block_size;
        let causal_mask =
            Tensor::<2>::from_data(TensorData::new(causal_mask_values(n), [n, n]), device);

        Gpt {
            wte: EmbeddingConfig::new(self.vocab_size, self.d_model)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: INIT_STD,
                })
                .init(device),
            blocks: (0..self.n_layer)
                .map(|_| Block::new(self, device))
                .collect(),
            norm_f: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.norm_eps)
                .init(device),
            head: linear(self.d_model, self.vocab_size, INIT_STD, device),
            rope_cos: table(cos),
            rope_sin: table(sin),
            causal_mask,
            block_size: self.block_size,
        }
    }

    /// Builds a model on `device` whose initial weights depend on `seed` alone.
    ///
    /// Seeds the device's generator and draws every weight, in a fixed order, as one step that no
    /// other thread can interleave with (see [`GptConfig::init`] for why that matters), so two
    /// runs started at the same moment from different threads still get the weights of their own
    /// seeds, and the same seed always gives the same model.
    ///
    /// Panics if the configuration is invalid.
    pub fn init_seeded(&self, device: &Device, seed: u64) -> Gpt {
        exclusive_generator(|| {
            device.seed(seed);
            let model = self.init(device);
            model.draw_all_parameters();
            model
        })
    }
}

/// Reads every parameter once, which is what draws its initial value.
struct InitAll;

impl burn::module::ModuleVisitor for InitAll {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
        let _ = param.val();
    }
}

impl Gpt {
    /// Draws the initial value of every parameter now, in a fixed order.
    ///
    /// Burn initialises a parameter the first time something reads it, and each draw advances
    /// the device's random generator. When several threads touch a fresh model at once (the
    /// data-parallel training step does), the order of the draws, and with it every initial
    /// weight, would depend on thread timing. Calling this first fixes the order; seeding the
    /// device first, under the same lock, is what [`GptConfig::init_seeded`] does, and is usually
    /// what you want. A model that is about to be loaded from a file must not call it: that would
    /// waste the draws.
    pub fn init_parameters(&self) {
        exclusive_generator(|| self.draw_all_parameters());
    }

    /// [`Gpt::init_parameters`] for a caller that already holds the generator.
    fn draw_all_parameters(&self) {
        use burn::module::Module;
        self.visit(&mut InitAll);
    }

    /// Maximum context length this model was built for.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Logits for every position: `[batch, seq] -> [batch, seq, vocab]`.
    ///
    /// Panics if `seq` exceeds [`Gpt::block_size`].
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [_, t] = tokens.dims();
        assert!(
            t <= self.block_size,
            "sequence of {t} tokens exceeds block_size {}",
            self.block_size
        );
        let half = self.rope_cos.dims()[1];
        let table = |tab: &Tensor<2>| tab.clone().slice([0..t, 0..half]).reshape([1, 1, t, half]);
        let (cos, sin) = (table(&self.rope_cos), table(&self.rope_sin));
        let mask = self
            .causal_mask
            .clone()
            .slice([0..t, 0..t])
            .reshape([1, 1, t, t]);

        let mut x = self.wte.forward(tokens);
        for block in &self.blocks {
            x = block.forward(x, &cos, &sin, &mask);
        }
        self.head.forward(self.norm_f.forward(x))
    }

    /// An empty cache for [`Gpt::forward_cached`].
    pub fn new_cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.blocks.len()).map(|_| None).collect(),
            len: 0,
        }
    }

    /// Reads `tokens` (`[batch, t]`) as the continuation of whatever `cache` already holds and
    /// returns their logits (`[batch, t, vocab]`), identical to the corresponding rows of
    /// [`Gpt::forward`] over the whole sequence. Use it for generation: first the prompt, then
    /// one token at a time.
    ///
    /// Panics if the cache would grow beyond [`Gpt::block_size`]. For inference only (it records
    /// no gradients).
    pub fn forward_cached(&self, tokens: Tensor<2, Int>, cache: &mut KvCache) -> Tensor<3> {
        let [_, t] = tokens.dims();
        let start = cache.len;
        let total = start + t;
        assert!(
            total <= self.block_size,
            "context of {total} tokens exceeds block_size {}",
            self.block_size
        );
        let half = self.rope_cos.dims()[1];
        let table = |tab: &Tensor<2>| {
            tab.clone()
                .slice([start..total, 0..half])
                .reshape([1, 1, t, half])
        };
        let (cos, sin) = (table(&self.rope_cos), table(&self.rope_sin));
        let mask = self
            .causal_mask
            .clone()
            .slice([start..total, 0..total])
            .reshape([1, 1, t, total]);

        let mut x = self.wte.forward(tokens);
        for (block, layer) in self.blocks.iter().zip(cache.layers.iter_mut()) {
            x = block.forward_cached(x, &cos, &sin, &mask, layer);
        }
        cache.len = total;
        self.head.forward(self.norm_f.forward(x))
    }

    /// Mean cross-entropy (nats per token) of predicting `targets` from `tokens`.
    pub fn loss(&self, tokens: Tensor<2, Int>, targets: Tensor<2, Int>) -> Tensor<1> {
        let logits = self.forward(tokens);
        let [b, t, v] = logits.dims();
        let logits = logits.reshape([b * t, v]);
        let targets = targets.reshape([b * t]);
        CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    fn tokens(
        b: usize,
        t: usize,
        device: &Device,
        f: impl Fn(usize, usize) -> i32,
    ) -> Tensor<2, Int> {
        let data: Vec<i32> = (0..b * t).map(|i| f(i / t, i % t)).collect();
        Tensor::<2, Int>::from_data(TensorData::new(data, [b, t]), device)
    }

    #[test]
    fn parameter_count_matches_formula() {
        let device = Device::flex();
        for p in [Preset::Nano, Preset::Micro] {
            let cfg = p.config(32);
            let model = cfg.init_seeded(&device, 1);
            assert_eq!(model.num_params(), cfg.num_params(), "{}", p.name());
        }
    }

    #[test]
    fn cached_forward_reproduces_the_full_forward_pass() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(40);
        let model = cfg.init_seeded(&device, 1);
        let t = 30;
        let all = tokens(1, t, &device, |_, c| ((c * 37 + 11) % 256) as i32);
        let full: Vec<f32> = model.forward(all.clone()).into_data().try_to_vec().unwrap();

        // A prompt of 12 tokens, then 5 at once (a non-empty cache with several new tokens),
        // then one by one.
        let mut cache = model.new_cache();
        assert!(cache.is_empty());
        let mut got: Vec<f32> = Vec::new();
        let mut at = 0;
        for chunk in [12usize, 5]
            .into_iter()
            .chain(std::iter::repeat_n(1, t - 17))
        {
            let ids = all.clone().slice([0..1, at..at + chunk]);
            let logits = model.forward_cached(ids, &mut cache);
            got.extend(logits.into_data().try_to_vec::<f32>().unwrap());
            at += chunk;
            assert_eq!(cache.len(), at);
        }
        assert_eq!(at, t);
        assert_eq!(got.len(), full.len());
        let worst = got
            .iter()
            .zip(&full)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst < 1e-4,
            "cached logits differ from the full pass by {worst}"
        );

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    #[should_panic(expected = "exceeds block_size")]
    fn cached_forward_refuses_to_outgrow_the_context() {
        let device = Device::flex();
        let model = Preset::Nano.config(8).init_seeded(&device, 1);
        let mut cache = model.new_cache();
        let x = tokens(1, 6, &device, |_, c| c as i32);
        model.forward_cached(x.clone(), &mut cache);
        model.forward_cached(x, &mut cache);
    }

    #[test]
    fn attention_blocks_are_twice_the_head_dimension_within_limits() {
        assert_eq!(attention_block(8), 32);
        assert_eq!(attention_block(32), 64);
        assert_eq!(attention_block(64), 128);
        assert_eq!(attention_block(512), 256);
    }

    #[test]
    fn parameter_count_follows_a_larger_vocabulary() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(32).with_vocab(1000);
        let model = cfg.init_seeded(&device, 1);
        assert_eq!(model.num_params(), cfg.num_params());
        assert_eq!(
            cfg.num_params() - Preset::Nano.config(32).num_params(),
            2 * (1000 - 256) * cfg.d_model,
            "embedding and head both grow with the vocabulary"
        );
        let x = tokens(2, 9, &device, |r, c| ((r * 301 + c * 97) % 1000) as i32);
        assert_eq!(model.forward(x).dims(), [2, 9, 1000]);
    }

    #[test]
    fn forward_has_expected_shape_and_is_finite() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init_seeded(&device, 1);
        let x = tokens(3, 20, &device, |r, c| ((r * 31 + c * 7) % 256) as i32);
        let logits = model.forward(x);
        assert_eq!(logits.dims(), [3, 20, 256]);
        let v: Vec<f32> = logits.into_data().try_to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn attention_is_causal() {
        // Changing a token must not affect the logits of EARLIER positions.
        let device = Device::flex();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init_seeded(&device, 1);
        let a = tokens(1, 16, &device, |_, c| (c * 13 % 256) as i32);
        let b = tokens(1, 16, &device, |_, c| {
            if c >= 10 { 200 } else { (c * 13 % 256) as i32 }
        });
        let la: Vec<f32> = model.forward(a).into_data().try_to_vec::<f32>().unwrap();
        let lb: Vec<f32> = model.forward(b).into_data().try_to_vec::<f32>().unwrap();
        let v = 256;
        for pos in 0..10 {
            for k in 0..v {
                let (x, y) = (la[pos * v + k], lb[pos * v + k]);
                assert!((x - y).abs() < 1e-4, "position {pos} changed: {x} vs {y}");
            }
        }
        // ...while later positions do change.
        let diff: f32 = (10 * v..16 * v).map(|i| (la[i] - lb[i]).abs()).sum();
        assert!(diff > 1e-3);
    }

    fn max_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
        (a - b).abs().max().into_scalar::<f32>()
    }

    fn reference_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
        attention(
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            },
        )
    }

    #[test]
    fn block_attention_matches_burn_attention_values_and_gradients() {
        use burn::tensor::Distribution;
        let device = Device::flex().autodiff();
        let (b, h, t, hd) = (2usize, 3usize, 20usize, 8usize);
        let dist = Distribution::Normal(0.0, 1.0);
        // Draws from the generator that every thread of the process shares: one at a time.
        let (q, k, v) = exclusive_generator(|| {
            let draw = || Tensor::<4>::random([b, h, t, hd], dist, &device).require_grad();
            (draw(), draw(), draw())
        });
        let mask = Tensor::<2>::from_data(TensorData::new(causal_mask_values(t), [t, t]), &device)
            .reshape([1, 1, t, t]);

        let run = |f: &dyn Fn(Tensor<4>, Tensor<4>, Tensor<4>) -> Tensor<4>| {
            let out = f(q.clone(), k.clone(), v.clone());
            let grads = out.clone().powf_scalar(2.0).sum().backward();
            (
                out.inner(),
                q.grad(&grads).unwrap(),
                k.grad(&grads).unwrap(),
                v.grad(&grads).unwrap(),
            )
        };
        let reference = run(&reference_attention);
        // 7 gives a ragged last block (20 = 7 + 7 + 6); 64 exercises the single-block path.
        for block in [7usize, 5, 64] {
            let got = run(&|q, k, v| causal_attention_blocks(q, k, v, &mask, block));
            assert!(
                max_diff(got.0, reference.0.clone()) < 1e-4,
                "output, block {block}"
            );
            assert!(
                max_diff(got.1, reference.1.clone()) < 1e-3,
                "dq, block {block}"
            );
            assert!(
                max_diff(got.2, reference.2.clone()) < 1e-3,
                "dk, block {block}"
            );
            assert!(
                max_diff(got.3, reference.3.clone()) < 1e-3,
                "dv, block {block}"
            );
        }
    }

    #[test]
    fn training_and_evaluation_paths_give_the_same_logits() {
        // The autodiff device uses the block attention, the plain device Burn's fused kernel.
        let train_dev = Device::flex().autodiff();
        let cfg = Preset::Nano.config(300); // longer than the attention block (64): several blocks
        let model = cfg.init_seeded(&train_dev, 1);
        let x = tokens(2, 300, &train_dev, |r, c| ((r * 29 + c * 3) % 256) as i32);
        let a = model.forward(x.clone());

        let eval_model = model.valid();
        let x_eval = x.inner();
        let b = eval_model.forward(x_eval);
        assert!(max_diff(a.inner(), b) < 1e-3);
    }

    /// Every parameter of a model, in the module's own order.
    fn weights(model: &Gpt) -> Vec<f32> {
        struct Collect(Vec<f32>);
        impl burn::module::ModuleVisitor for Collect {
            fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<D>>) {
                self.0
                    .extend(param.val().into_data().try_to_vec::<f32>().unwrap());
            }
        }
        let mut all = Collect(Vec::new());
        burn::module::Module::visit(model, &mut all);
        all.0
    }

    #[test]
    fn the_seed_alone_decides_the_weights_even_when_other_threads_draw() {
        // Burn has one random generator for the whole process. Eight threads build models at once,
        // four with one seed and four with another, round after round; whatever the interleaving,
        // every model must be the one its seed promises. Without the lock around "seed, then
        // draw" the threads take shares of each other's streams.
        let (cfg, device) = (Preset::Nano.config(16), Device::flex());
        let alone = |seed| weights(&cfg.init_seeded(&device, seed));
        let (want_a, want_b) = (alone(11), alone(22));
        assert_ne!(
            want_a, want_b,
            "different seeds must give different weights"
        );
        assert_eq!(
            want_a,
            alone(11),
            "the same seed must give the same weights"
        );

        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8usize)
                .map(|i| {
                    let (cfg, device) = (&cfg, &device);
                    scope.spawn(move || {
                        let seed = if i % 2 == 0 { 11 } else { 22 };
                        (0..12)
                            .map(|_| weights(&cfg.init_seeded(device, seed)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for (i, thread) in threads.into_iter().enumerate() {
                let want = if i % 2 == 0 { &want_a } else { &want_b };
                for (round, got) in thread.join().unwrap().iter().enumerate() {
                    assert!(
                        got == want,
                        "thread {i}, round {round}: not the seed's weights"
                    );
                }
            }
        });
    }

    #[test]
    fn initial_loss_is_close_to_uniform_entropy() {
        let device = Device::flex();
        let model = Preset::Nano.config(32).init_seeded(&device, 1);
        let x = tokens(4, 24, &device, |r, c| ((r * 17 + c * 5) % 256) as i32);
        let y = tokens(4, 24, &device, |r, c| ((r * 3 + c * 11 + 1) % 256) as i32);
        let loss = model.loss(x, y).into_scalar::<f32>();
        let uniform = (256f32).ln(); // 5.545
        assert!((loss - uniform).abs() < 0.3, "loss {loss} vs {uniform}");
    }
}
