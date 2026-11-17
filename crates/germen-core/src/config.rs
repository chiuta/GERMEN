//! Model architecture configuration and size presets.
//!
//! GERMEN trains a decoder-only transformer ("GPT-style") with pre-norm RMSNorm,
//! rotary position embeddings, a squared-ReLU feed-forward block and no bias terms.
//! The vocabulary is byte-level (256 symbols), so any UTF-8 text — Romanian
//! diacritics included — is representable without an out-of-vocabulary token.

use burn::prelude::*;

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
    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// Checks internal consistency; returns a human-readable error otherwise.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 || self.block_size < 2 || self.n_layer == 0 {
            return Err("vocab_size, block_size și n_layer trebuie să fie pozitive".into());
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
        Ok(())
    }

    /// Number of trainable parameters (all weights; there are no biases).
    pub fn num_params(&self) -> usize {
        let d = self.d_model;
        let per_layer = d * 3 * d // fused q,k,v projection
            + d * d // attention output projection
            + 2 * d * self.d_ff // feed-forward up + down
            + 2 * d; // two RMSNorm gains
        self.vocab_size * d // token embedding
            + self.n_layer * per_layer
            + d // final RMSNorm gain
            + d * self.vocab_size // output head (untied)
    }

    /// Parameters that take part in matrix multiplications (everything except the
    /// embedding lookup table and the norm gains).
    fn matmul_params(&self) -> usize {
        let d = self.d_model;
        self.n_layer * (d * 3 * d + d * d + 2 * d * self.d_ff) + d * self.vocab_size
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

    /// Estimated floating-point operations to train on ONE token at context length
    /// `seq_len` (forward + backward): `6 * N_matmul + 12 * L * d * T`.
    ///
    /// The second term counts the two attention matmuls (QKᵀ and AV) of a dense,
    /// non-causal-skipping implementation. This is the usual "model FLOPs" estimate;
    /// real hardware executes extra element-wise work that is not counted.
    pub fn train_flops_per_token(&self, seq_len: usize) -> f64 {
        6.0 * self.matmul_params() as f64
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
