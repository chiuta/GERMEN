//! The GPT-style network, written directly on Burn tensors.

use crate::config::GptConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{
    Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;

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

    fn forward(&self, x: Tensor<3>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let h = self.n_head;
        let hd = d / h;

        let qkv = self.qkv.forward(x); // [b, t, 3d]
        let split = |lo: usize| {
            qkv.clone()
                .slice([0..b, 0..t, lo..lo + d])
                .reshape([b, t, h, hd])
                .swap_dims(1, 2) // [b, h, t, hd]
        };
        let q = apply_rope(split(0), cos, sin);
        let k = apply_rope(split(d), cos, sin);
        let v = split(2 * d);

        let options = AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        };
        let y = attention(q, k, v, None, None, options); // [b, h, t, hd]
        let y = y.swap_dims(1, 2).reshape([b, t, d]);
        self.proj.forward(y)
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

    fn forward(&self, x: Tensor<3>, cos: &Tensor<4>, sin: &Tensor<4>) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), cos, sin);
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
    block_size: usize,
}

impl GptConfig {
    /// Builds a freshly initialised model on `device`.
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
            block_size: self.block_size,
        }
    }
}

impl Gpt {
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

        let mut x = self.wte.forward(tokens);
        for block in &self.blocks {
            x = block.forward(x, &cos, &sin);
        }
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
            let model = cfg.init(&device);
            assert_eq!(model.num_params(), cfg.num_params(), "{}", p.name());
        }
    }

    #[test]
    fn forward_has_expected_shape_and_is_finite() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init(&device);
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
        let model = cfg.init(&device);
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

    #[test]
    fn initial_loss_is_close_to_uniform_entropy() {
        let device = Device::flex();
        let model = Preset::Nano.config(32).init(&device);
        let x = tokens(4, 24, &device, |r, c| ((r * 17 + c * 5) % 256) as i32);
        let y = tokens(4, 24, &device, |r, c| ((r * 3 + c * 11 + 1) % 256) as i32);
        let loss = model.loss(x, y).into_scalar::<f32>();
        let uniform = (256f32).ln(); // 5.545
        assert!((loss - uniform).abs() < 0.3, "loss {loss} vs {uniform}");
    }
}
