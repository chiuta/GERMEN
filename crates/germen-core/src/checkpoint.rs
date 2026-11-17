//! Saving and loading a trained model.
//!
//! A checkpoint is a directory with three files:
//!
//! | file          | content                                                              |
//! |---------------|----------------------------------------------------------------------|
//! | `config.json` | architecture ([`GptConfig`]) — enough to rebuild the empty network   |
//! | `weights.bpk` | all parameters in Burn's "burnpack" format (f32)                     |
//! | `meta.json`   | training progress ([`Meta`]): step, tokens seen, last losses, …      |
//!
//! Files are written to a temporary name and renamed, so an interrupted save (Ctrl-C, power
//! loss) never leaves a half-written checkpoint behind.

use crate::config::GptConfig;
use crate::model::Gpt;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// File name of the weights inside a checkpoint directory.
pub const WEIGHTS_FILE: &str = "weights.bpk";
/// File name of the architecture description.
pub const CONFIG_FILE: &str = "config.json";
/// File name of the training metadata.
pub const META_FILE: &str = "meta.json";

/// Training progress stored next to the weights.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    /// Version of GERMEN that wrote the checkpoint.
    pub germen_version: String,
    /// Optimizer steps completed.
    pub step: usize,
    /// Training tokens processed so far.
    pub tokens_seen: u64,
    /// Training loss (nats/token) of the most recent step.
    pub last_train_loss: Option<f32>,
    /// Validation loss (nats/token) at the most recent evaluation.
    pub last_val_loss: Option<f32>,
    /// Lowest validation loss seen so far.
    pub best_val_loss: Option<f32>,
    /// Seed of the run.
    pub seed: u64,
    /// Size of the corpus in bytes.
    pub corpus_bytes: usize,
    /// Context length used for training.
    pub seq: usize,
    /// Windows per step used for training.
    pub batch: usize,
    /// Wall-clock seconds spent training (sums over resumed runs).
    pub elapsed_secs: f64,
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = tmp_path(path);
    fs::write(&tmp, bytes).map_err(|e| format!("nu pot scrie {}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("nu pot redenumi în {}: {e}", path.display()))
}

/// Writes `config.json`, `weights.bpk` and `meta.json` into `dir` (created if missing).
///
/// `model` may carry autodiff tracking; only the plain values are stored.
pub fn save(dir: &Path, model: &Gpt, cfg: &GptConfig, meta: &Meta) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("nu pot crea {}: {e}", dir.display()))?;

    let weights = dir.join(WEIGHTS_FILE);
    let weights_tmp = tmp_path(&weights);
    model
        .clone()
        .valid()
        .save_file(&weights_tmp)
        .map_err(|e| format!("nu pot salva ponderile: {e}"))?;
    fs::rename(&weights_tmp, &weights)
        .map_err(|e| format!("nu pot redenumi în {}: {e}", weights.display()))?;

    let cfg_json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    write_atomic(&dir.join(CONFIG_FILE), cfg_json.as_bytes())?;
    let meta_json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    write_atomic(&dir.join(META_FILE), meta_json.as_bytes())
}

/// Reads only the architecture from a checkpoint directory.
pub fn load_config(dir: &Path) -> Result<GptConfig, String> {
    let path = dir.join(CONFIG_FILE);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    let cfg: GptConfig =
        serde_json::from_str(&text).map_err(|e| format!("{} e invalid: {e}", path.display()))?;
    cfg.validate()?;
    Ok(cfg)
}

/// Reads only the training metadata (missing file → defaults).
pub fn load_meta(dir: &Path) -> Result<Meta, String> {
    let path = dir.join(META_FILE);
    if !path.exists() {
        return Ok(Meta::default());
    }
    let text =
        fs::read_to_string(&path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{} e invalid: {e}", path.display()))
}

/// Rebuilds the network from a checkpoint directory and loads its weights.
pub fn load(dir: &Path, device: &Device) -> Result<(Gpt, GptConfig, Meta), String> {
    let cfg = load_config(dir)?;
    let weights = dir.join(WEIGHTS_FILE);
    if !weights.exists() {
        return Err(format!("lipsește {}", weights.display()));
    }
    let model = cfg
        .init(device)
        .try_load_file(&weights)
        .map_err(|e| format!("nu pot încărca {}: {e}", weights.display()))?;
    Ok((model, cfg, load_meta(dir)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn save_and_load_roundtrip_reproduces_the_same_logits() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init(&device);
        let dir = scratch_dir("roundtrip");
        let meta = Meta {
            germen_version: crate::VERSION.into(),
            step: 7,
            tokens_seen: 1234,
            last_val_loss: Some(1.5),
            ..Meta::default()
        };
        save(&dir, &model, &cfg, &meta).unwrap();
        assert!(
            !dir.join("weights.bpk.tmp").exists(),
            "temporary file must be gone"
        );

        let (loaded, cfg2, meta2) = load(&dir, &device).unwrap();
        assert_eq!(cfg2.num_params(), cfg.num_params());
        assert_eq!(meta2, meta);

        let data: Vec<i32> = (0..24).map(|i| i * 11 % 256).collect();
        let x = Tensor::<2, Int>::from_data(TensorData::new(data, [2, 12]), &device);
        let a: Vec<f32> = model
            .forward(x.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let b: Vec<f32> = loaded.forward(x).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f32::max);
        assert!(max_diff < 1e-6, "logits differ after reload: {max_diff}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_reports_missing_files() {
        let dir = scratch_dir("missing");
        assert!(load(&dir, &Device::flex()).is_err());
    }
}
