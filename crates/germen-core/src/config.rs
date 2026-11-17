//! Model architecture configuration and size presets.
//!
//! GERMEN trains a decoder-only transformer ("GPT-style") with pre-norm RMSNorm,
//! rotary position embeddings, a squared-ReLU feed-forward block and no bias terms.
//! The presets assume the byte-level vocabulary (256 symbols); with a BPE tokenizer use
//! [`GptConfig::with_vocab`] to match the tokenizer's size.

use burn::prelude::*;

/// Longest context a configuration may ask for. The causal mask is `block_size²` floats, so this
/// limit (1 GiB of mask) keeps a mistyped or hostile `config.json` from reserving terabytes.
pub const MAX_BLOCK_SIZE: usize = 16_384;
/// Largest vocabulary a configuration may declare (the tokenizers stop at 65 536).
pub const MAX_VOCAB_SIZE: usize = 1 << 20;
/// Largest number of transformer blocks.
pub const MAX_LAYERS: usize = 1_024;
/// Largest residual-stream width.
pub const MAX_D_MODEL: usize = 16_384;
/// Largest feed-forward width.
pub const MAX_D_FF: usize = 65_536;
/// Largest parameter count (4 Gi parameters = 16 GiB of weights, far beyond what a CPU run can
/// train; the limit exists to refuse nonsense, not to ration real models).
pub const MAX_PARAMS: usize = 1 << 32;

/// Architecture of the model. Serialised as JSON next to every checkpoint.
#[derive(Config, Debug)]
pub struct GptConfig {
    /// Number of distinct tokens (256 for the byte-level tokenizer).
    pub vocab_size: usize,
    /// Maximum context length (positions the model can attend to).
    pub block_size: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads. `d_model` must be divisible by it.
    pub n_head: usize,
    /// Width of the residual stream.
    pub d_model: usize,
    /// Hidden width of the feed-forward block (usually `4 * d_model`).
    pub d_ff: usize,
    /// Base of the rotary-embedding frequencies.
    #[config(default = 10000.0)]
    pub rope_theta: f32,
    /// Epsilon of the RMS normalisation.
    #[config(default = 1e-5)]
    pub norm_eps: f64,
}

impl GptConfig {
    /// The same architecture with a different vocabulary size (the presets assume the 256
    /// bytes; a BPE tokenizer brings its own size, which changes the embedding and the output
    /// head and so the parameter count).
    pub fn with_vocab(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// Checks internal consistency and sanity; returns a human-readable error otherwise.
    ///
    /// A `config.json` is input like any other file: besides self-consistency this enforces
    /// upper limits, so that a corrupt or hostile one is refused here instead of making
    /// [`GptConfig::init`] allocate an absurd amount of memory (or divide by zero).
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 || self.block_size < 2 || self.n_layer == 0 {
            return Err("vocab_size, block_size și n_layer trebuie să fie pozitive".into());
        }
        if self.d_model == 0 || self.d_ff == 0 {
            return Err("d_model și d_ff trebuie să fie pozitive".into());
        }
        if self.n_head == 0 || !self.d_model.is_multiple_of(self.n_head) {
            return Err(format!(
                "d_model ({}) trebuie să fie multiplu de n_head ({})",
                self.d_model, self.n_head
            ));
        }
        if !self.head_dim().is_multiple_of(2) {
            return Err(
                "dimensiunea unui cap (d_model / n_head) trebuie să fie pară (RoPE)".into(),
            );
        }
        for (name, value, max) in [
            ("vocab_size", self.vocab_size, MAX_VOCAB_SIZE),
            ("block_size", self.block_size, MAX_BLOCK_SIZE),
            ("n_layer", self.n_layer, MAX_LAYERS),
            ("d_model", self.d_model, MAX_D_MODEL),
            ("d_ff", self.d_ff, MAX_D_FF),
        ] {
            if value > max {
                return Err(format!(
                    "{name} ({value}) depășește maximul acceptat ({max})"
                ));
            }
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err("rope_theta trebuie să fie un număr finit pozitiv".into());
        }
        if !(self.norm_eps.is_finite() && self.norm_eps > 0.0) {
            return Err("norm_eps trebuie să fie un număr finit pozitiv".into());
        }
        let params = self.param_count();
        if params > MAX_PARAMS as u128 {
            return Err(format!(
                "modelul ar avea {params} de parametri; maximul acceptat este {MAX_PARAMS}"
            ));
        }
        Ok(())
    }

    /// Parameter count in `u128` with saturating arithmetic, so that it cannot overflow however
    /// absurd the fields are (only [`GptConfig::validate`] decides whether they are acceptable).
    fn param_count(&self) -> u128 {
        let (d, ff, v, l) = (
            self.d_model as u128,
            self.d_ff as u128,
            self.vocab_size as u128,
            self.n_layer as u128,
        );
        let mul = |a: u128, b: u128| a.saturating_mul(b);
        let per_layer = mul(mul(d, 3), d) // fused q,k,v projection
            .saturating_add(mul(d, d)) // attention output projection
            .saturating_add(mul(mul(2, d), ff)) // feed-forward up + down
            .saturating_add(mul(2, d)); // two RMSNorm gains
        mul(v, d) // token embedding
            .saturating_add(mul(l, per_layer))
            .saturating_add(d) // final RMSNorm gain
            .saturating_add(mul(d, v)) // output head (untied)
    }

    /// Number of trainable parameters (all weights; there are no biases). Saturates at
    /// `usize::MAX` for a configuration that [`GptConfig::validate`] would refuse anyway.
    pub fn num_params(&self) -> usize {
        usize::try_from(self.param_count()).unwrap_or(usize::MAX)
    }

    /// Parameters that take part in matrix multiplications (everything except the
    /// embedding lookup table and the norm gains), as a float so it cannot overflow.
    fn matmul_params(&self) -> f64 {
        let d = self.d_model as f64;
        self.n_layer as f64 * (4.0 * d * d + 2.0 * d * self.d_ff as f64)
            + d * self.vocab_size as f64
    }

    /// Rough memory (bytes) needed to train with `batch` windows of `seq` tokens: weights,
    /// gradients and AdamW's two moment buffers, plus the activations kept for the backward pass.
    ///
    /// Calibrated on measurements of the presets (it comes out about 25 % above the real peak),
    /// so it is a conservative order-of-magnitude guide, not an exact figure.
    pub fn estimate_train_bytes(&self, batch: usize, seq: usize) -> u64 {
        let (d, h, t, b) = (
            self.d_model as f64,
            self.n_head as f64,
            seq as f64,
            batch as f64,
        );
        let per_layer_token = 17.0 * d + 3.0 * h * t + 3.0 * self.d_ff as f64;
        let activations =
            b * t * (self.n_layer as f64 * per_layer_token + 3.0 * self.vocab_size as f64) * 4.0;
        let weights_and_optimizer = 4.0 * self.num_params() as f64 * 4.0;
        (activations + weights_and_optimizer + 64.0 * 1024.0 * 1024.0) as u64
    }

    /// [`GptConfig::estimate_train_bytes`] for a step run as `slots` data-parallel replicas at a
    /// time (see `TrainOptions::replicas`): every slot holds a set of gradients of its own, and
    /// the one it is adding them to, on top of the single set the estimate already counts. The
    /// activations do not grow: the same windows are merely split among the replicas.
    pub fn estimate_train_bytes_replicated(&self, batch: usize, seq: usize, slots: usize) -> u64 {
        let single = self.estimate_train_bytes(batch, seq);
        if slots <= 1 {
            return single;
        }
        let extra = (2 * slots - 1) as f64 * self.num_params() as f64 * 4.0;
        single.saturating_add(extra as u64)
    }

    /// Estimated floating-point operations to train on ONE token at context length
    /// `seq_len` (forward + backward): `6 * N_matmul + 12 * L * d * T`.
    ///
    /// The second term counts the two attention matmuls (QKᵀ and AV) of a dense,
    /// non-causal-skipping implementation. This is the usual "model FLOPs" estimate;
    /// real hardware executes extra element-wise work that is not counted.
    pub fn train_flops_per_token(&self, seq_len: usize) -> f64 {
        6.0 * self.matmul_params()
            + 12.0 * self.n_layer as f64 * self.d_model as f64 * seq_len as f64
    }
}

/// Named model sizes. Parameter counts assume the byte-level vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// ≈ 0.85 M parameters — smoke tests and the fastest benchmark.
    Nano,
    /// ≈ 2.8 M parameters.
    Micro,
    /// ≈ 10.8 M parameters — the size GERMEN was designed around.
    Mini,
    /// ≈ 25 M parameters.
    Small,
    /// ≈ 85 M parameters (GPT-2 "small" shape) — realistically needs a GPU.
    Base,
}

impl Preset {
    /// Presets measured by `germen bench --preset all` (Base is opt-in: it is slow on CPUs).
    pub const DEFAULT_BENCH: [Preset; 4] =
        [Preset::Nano, Preset::Micro, Preset::Mini, Preset::Small];

    /// Short lowercase name used on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Preset::Nano => "nano",
            Preset::Micro => "micro",
            Preset::Mini => "mini",
            Preset::Small => "small",
            Preset::Base => "base",
        }
    }

    /// Parses a preset name (case-insensitive).
    pub fn parse(s: &str) -> Option<Preset> {
        match s.trim().to_ascii_lowercase().as_str() {
            "nano" => Some(Preset::Nano),
            "micro" => Some(Preset::Micro),
            "mini" => Some(Preset::Mini),
            "small" => Some(Preset::Small),
            "base" => Some(Preset::Base),
            _ => None,
        }
    }

    /// The architecture for this preset with the given context length.
    pub fn config(self, block_size: usize) -> GptConfig {
        let (n_layer, n_head, d_model) = match self {
            Preset::Nano => (4, 4, 128),
            Preset::Micro => (6, 6, 192),
            Preset::Mini => (6, 6, 384),
            Preset::Small => (8, 8, 512),
            Preset::Base => (12, 12, 768),
        };
        GptConfig::new(256, block_size, n_layer, n_head, d_model, 4 * d_model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_valid_and_have_expected_sizes() {
        for (p, lo, hi) in [
            (Preset::Nano, 0.7e6, 1.0e6),
            (Preset::Micro, 2.5e6, 3.1e6),
            (Preset::Mini, 10.0e6, 11.5e6),
            (Preset::Small, 24.0e6, 27.0e6),
            (Preset::Base, 80.0e6, 90.0e6),
        ] {
            let c = p.config(256);
            c.validate().unwrap();
            let n = c.num_params() as f64;
            assert!(n > lo && n < hi, "{}: {n}", p.name());
        }
    }

    #[test]
    fn hostile_or_corrupt_configs_are_rejected_before_anything_is_allocated() {
        let base = || Preset::Nano.config(64);
        assert!(base().validate().is_ok());
        type Tweak = fn(&mut GptConfig);
        let broken: [(&str, Tweak); 14] = [
            ("zero width", |c| c.d_model = 0),
            ("zero feed-forward", |c| c.d_ff = 0),
            ("NaN rope_theta", |c| c.rope_theta = f32::NAN),
            ("zero rope_theta", |c| c.rope_theta = 0.0),
            ("infinite rope_theta", |c| c.rope_theta = f32::INFINITY),
            ("negative rope_theta", |c| c.rope_theta = -1.0),
            ("NaN norm_eps", |c| c.norm_eps = f64::NAN),
            ("zero norm_eps", |c| c.norm_eps = 0.0),
            ("negative norm_eps", |c| c.norm_eps = -1e-5),
            // The causal mask is block_size² floats: this would be 4 exabytes.
            ("huge block_size", |c| c.block_size = 1 << 31),
            ("huge vocabulary", |c| c.vocab_size = usize::MAX),
            ("huge width", |c| {
                c.d_model = 1 << 40;
                c.n_head = 1;
            }),
            ("huge depth", |c| c.n_layer = usize::MAX),
            // Every field is within its own limit, yet the weights alone would take ~13 TB.
            ("huge total", |c| {
                c.d_model = 16_384;
                c.n_head = 16;
                c.d_ff = 65_536;
                c.n_layer = 1_024;
            }),
        ];
        for (what, tweak) in broken {
            let mut c = base();
            tweak(&mut c);
            assert!(c.validate().is_err(), "{what} must be rejected");
        }
    }

    #[test]
    fn parameter_arithmetic_saturates_instead_of_overflowing() {
        let mut c = Preset::Nano.config(64);
        c.d_model = usize::MAX - 1;
        c.n_head = 2;
        c.d_ff = usize::MAX;
        c.n_layer = usize::MAX;
        c.vocab_size = usize::MAX;
        assert_eq!(c.num_params(), usize::MAX);
        assert!(c.estimate_train_bytes(usize::MAX, usize::MAX) > 0);
    }

    #[test]
    fn the_largest_accepted_context_is_not_a_trap() {
        // A 16384-token context is the limit; its mask (1 GiB) must not be reachable by a typo
        // of one more digit.
        let mut c = Preset::Nano.config(MAX_BLOCK_SIZE);
        assert!(c.validate().is_ok());
        c.block_size = MAX_BLOCK_SIZE + 1;
        assert!(c.validate().is_err());
    }

    #[test]
    fn memory_estimate_is_plausible_and_monotonic() {
        const MIB: u64 = 1024 * 1024;
        // Measured peaks at batch 16 x 256: mini 1432 MiB, small 2524 MiB (the estimate is a bit high).
        let mini = Preset::Mini.config(256).estimate_train_bytes(16, 256) / MIB;
        assert!((1200..2300).contains(&mini), "mini: {mini} MiB");
        let small = Preset::Small.config(256).estimate_train_bytes(16, 256) / MIB;
        assert!((2200..3800).contains(&small), "small: {small} MiB");
        let c = Preset::Nano.config(1024);
        assert!(c.estimate_train_bytes(32, 256) > c.estimate_train_bytes(16, 256));
        assert!(c.estimate_train_bytes(16, 512) > c.estimate_train_bytes(16, 256));
        assert!(
            c.estimate_train_bytes(16, 9999) > 100 * 1024 * MIB,
            "absurd settings look absurd"
        );
    }

    #[test]
    fn validate_rejects_bad_heads() {
        let mut c = Preset::Nano.config(64);
        c.n_head = 5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn parse_roundtrip() {
        for p in [
            Preset::Nano,
            Preset::Micro,
            Preset::Mini,
            Preset::Small,
            Preset::Base,
        ] {
            assert_eq!(Preset::parse(p.name()), Some(p));
        }
        assert_eq!(Preset::parse("MINI"), Some(Preset::Mini));
        assert_eq!(Preset::parse("huge"), None);
    }

    #[test]
    fn flops_estimate_is_dominated_by_weights_for_short_context() {
        let c = Preset::Mini.config(256);
        let f = c.train_flops_per_token(256);
        let six_n = 6.0 * c.num_params() as f64;
        assert!(f > 0.9 * six_n && f < 1.3 * six_n);
    }
}
