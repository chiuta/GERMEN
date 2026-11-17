//! Saving and loading a trained model.
//!
//! A checkpoint is a directory with these files:
//!
//! | file             | content                                                              |
//! |------------------|----------------------------------------------------------------------|
//! | `config.json`    | architecture ([`GptConfig`]) — enough to rebuild the empty network   |
//! | `weights.bpk`    | all parameters in Burn's "burnpack" format (f32)                     |
//! | `meta.json`      | training progress ([`Meta`]): step, tokens seen, last losses, …      |
//! | `optimizer.bpk`  | AdamW moment estimates, needed to resume training exactly            |
//! | `tokenizer.json` | the vocabulary the model was trained with                            |
//!
//! [`save_all`] writes every file under a temporary name, flushes them to the disk and only then
//! renames them into place, `meta.json` last. An interrupted save (Ctrl-C, a crash, a power cut)
//! leaves the previous checkpoint complete; and weights that contain NaN or infinity are refused
//! instead of replacing a good checkpoint, both when saving and when loading.

use crate::config::GptConfig;
use crate::fsutil::Staged;
use crate::model::Gpt;
use crate::tokenizer::Tokenizer;
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::ModuleOptimizer;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// File name of the weights inside a checkpoint directory.
pub const WEIGHTS_FILE: &str = "weights.bpk";
/// File name of the architecture description.
pub const CONFIG_FILE: &str = "config.json";
/// File name of the training metadata.
pub const META_FILE: &str = "meta.json";
/// File name of the AdamW moment estimates (needed to resume training exactly).
pub const OPTIMIZER_FILE: &str = "optimizer.bpk";

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
    /// Windows per micro-batch used for training.
    pub batch: usize,
    /// Micro-batches accumulated per optimizer step (`0` in checkpoints written before 0.2,
    /// which means 1).
    #[serde(default)]
    pub accum: usize,
    /// Wall-clock seconds spent training (sums over resumed runs).
    pub elapsed_secs: f64,
}

struct FiniteCheck {
    tensors: usize,
    bad: usize,
}

impl ModuleVisitor for FiniteCheck {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.tensors += 1;
        // `x * 0` is 0 for every finite `x` and NaN for NaN and for ±infinity, so the sum is
        // exactly 0 when (and only when) every element is finite. No copy of the data is made.
        if param.val().mul_scalar(0.0).sum().into_scalar::<f32>() != 0.0 {
            self.bad += 1;
        }
    }
}

/// Fails if any weight of `model` is NaN or infinite. `model` should be a plain (not
/// autodiff-tracked) one, e.g. the result of `valid()`.
pub fn ensure_finite(model: &Gpt) -> Result<(), String> {
    let mut check = FiniteCheck { tensors: 0, bad: 0 };
    model.visit(&mut check);
    if check.bad > 0 {
        return Err(format!(
            "{} din {} tensori de ponderi conțin NaN sau infinit; antrenarea a divergent \
             (reduceți --lr)",
            check.bad, check.tensors
        ));
    }
    Ok(())
}

/// Writes `config.json`, `weights.bpk` and `meta.json` into `dir` (created if missing), as one
/// set that is published together (see the module documentation).
///
/// `model` may carry autodiff tracking; only the plain values are stored. To save the optimizer
/// state and the tokenizer in the same set, use [`save_all`].
pub fn save(dir: &Path, model: &Gpt, cfg: &GptConfig, meta: &Meta) -> Result<(), String> {
    save_all(dir, model, cfg, meta, None, None)
}

/// Writes a whole checkpoint — weights, architecture, progress and, when given, the optimizer
/// state and the tokenizer — as one set: every file is complete and on disk before any of them
/// replaces its predecessor, and `meta.json`, which says how far training got, is replaced last.
///
/// Refuses (writing nothing) if the weights contain NaN or infinity.
pub fn save_all(
    dir: &Path,
    model: &Gpt,
    cfg: &GptConfig,
    meta: &Meta,
    optimizer: Option<&ModuleOptimizer>,
    tokenizer: Option<&Tokenizer>,
) -> Result<(), String> {
    let plain = model.clone().valid();
    ensure_finite(&plain)?;
    let cfg_json = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
    let meta_json = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;

    let mut staged = Staged::new(dir)?;
    if let Some(tokenizer) = tokenizer {
        staged.bytes(
            crate::tokenizer::TOKENIZER_FILE,
            tokenizer.to_json()?.as_bytes(),
        )?;
    }
    if let Some(optimizer) = optimizer {
        staged.with(OPTIMIZER_FILE, |tmp| {
            optimizer
                .save(tmp)
                .map_err(|e| format!("nu pot salva starea optimizatorului: {e}"))
        })?;
    }
    staged.with(WEIGHTS_FILE, |tmp| {
        plain
            .save_file(tmp)
            .map_err(|e| format!("nu pot salva ponderile: {e}"))
    })?;
    staged.bytes(CONFIG_FILE, cfg_json.as_bytes())?;
    staged.bytes(META_FILE, meta_json.as_bytes())?;
    staged.commit()
}

/// Writes the optimizer state (AdamW moments) to `optimizer.bpk` in `dir`, atomically.
pub fn save_optimizer(dir: &Path, optim: &ModuleOptimizer) -> Result<(), String> {
    let mut staged = Staged::new(dir)?;
    staged.with(OPTIMIZER_FILE, |tmp| {
        optim
            .save(tmp)
            .map_err(|e| format!("nu pot salva starea optimizatorului: {e}"))
    })?;
    staged.commit()
}

/// Loads the optimizer state saved next to a checkpoint into `optim`.
///
/// Returns the optimizer and whether a state was found; checkpoints written before GERMEN 0.2
/// have none, and then `optim` is returned unchanged.
pub fn load_optimizer(
    dir: &Path,
    optim: ModuleOptimizer,
) -> Result<(ModuleOptimizer, bool), String> {
    let path = dir.join(OPTIMIZER_FILE);
    if !path.exists() {
        return Ok((optim, false));
    }
    let optim = optim
        .load(&path)
        .map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    Ok((optim, true))
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
    ensure_finite(&model.clone().valid())
        .map_err(|e| format!("{} este corupt: {e}", weights.display()))?;
    Ok((model, cfg, load_meta(dir)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;
    use std::path::PathBuf;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("germen-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn loading_a_checkpoint_draws_nothing_from_the_random_generator() {
        // The device's generator is shared by every thread, and `exclusive_generator` keeps the
        // draws of one initialisation apart from another's. Loading takes no part in that lock, which
        // is only right if loading never draws: the next number after a load must be the one
        // the generator would have given without it.
        use burn::tensor::Distribution;
        let device = Device::flex();
        let cfg = Preset::Nano.config(16);
        let dir = scratch_dir("load-draws-nothing");
        save(&dir, &cfg.init_seeded(&device, 1), &cfg, &Meta::default()).unwrap();
        let next = || {
            Tensor::<1>::random([8], Distribution::Default, &device)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
        };
        crate::model::exclusive_generator(|| {
            device.seed(5);
            let untouched = next();
            device.seed(5);
            load(&dir, &device).unwrap();
            assert_eq!(next(), untouched, "loading used up draws of the generator");
        });
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_and_load_roundtrip_reproduces_the_same_logits() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(32);
        let model = cfg.init_seeded(&device, 1);
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

    /// Makes the first float parameter NaN and leaves the others alone.
    struct PoisonFirst(bool);

    impl burn::module::ModuleMapper for PoisonFirst {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            if std::mem::replace(&mut self.0, true) {
                param
            } else {
                param.map(|t| t.mul_scalar(f32::NAN))
            }
        }
    }

    /// Every file of a directory with its content, sorted by name.
    fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<(String, Vec<u8>)> = fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        files
    }

    #[test]
    fn a_diverged_model_never_replaces_a_good_checkpoint() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(16);
        let dir = scratch_dir("poison");
        let meta = Meta {
            step: 5,
            ..Meta::default()
        };
        save(&dir, &cfg.init_seeded(&device, 1), &cfg, &meta).unwrap();
        let before = snapshot(&dir);

        let poisoned = cfg.init_seeded(&device, 1).map(&mut PoisonFirst(false));
        let err = save_all(
            &dir,
            &poisoned,
            &cfg,
            &Meta {
                step: 9,
                ..Meta::default()
            },
            None,
            Some(&Tokenizer::Byte),
        )
        .unwrap_err();
        assert!(err.contains("NaN"), "{err}");
        assert_eq!(
            snapshot(&dir),
            before,
            "not a byte may change, not even a .tmp"
        );
        assert_eq!(load_meta(&dir).unwrap().step, 5);

        // Weights that reached the disk some other way are refused when loading.
        poisoned.save_file(dir.join(WEIGHTS_FILE)).unwrap();
        let err = load(&dir, &device).unwrap_err();
        assert!(err.contains("corupt"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_all_writes_one_complete_set_and_leaves_nothing_else() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(16);
        let dir = scratch_dir("set");
        let optim = burn::optim::AdamWConfig::new().init();
        let meta = Meta {
            step: 3,
            ..Meta::default()
        };
        save_all(
            &dir,
            &cfg.init_seeded(&device, 1),
            &cfg,
            &meta,
            Some(&optim),
            Some(&Tokenizer::Byte),
        )
        .unwrap();
        let names: Vec<String> = snapshot(&dir).into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            [
                CONFIG_FILE,
                META_FILE,
                OPTIMIZER_FILE,
                crate::tokenizer::TOKENIZER_FILE,
                WEIGHTS_FILE
            ]
        );
        assert_eq!(load_meta(&dir).unwrap(), meta);
        assert_eq!(Tokenizer::load(&dir).unwrap(), Tokenizer::Byte);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_weight_files_are_errors_not_crashes() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(16);
        let dir = scratch_dir("damaged");
        save(&dir, &cfg.init_seeded(&device, 1), &cfg, &Meta::default()).unwrap();
        let good = fs::read(dir.join(WEIGHTS_FILE)).unwrap();
        let mut flipped = good.clone();
        flipped.iter_mut().take(64).for_each(|b| *b ^= 0xFF);
        let mut hole = good.clone();
        hole[good.len() / 3..good.len() / 3 + 4096].fill(0xFF);
        for (what, bytes) in [
            ("empty", Vec::new()),
            ("one byte", vec![0x42]),
            ("truncated", good[..good.len() / 2].to_vec()),
            ("garbage", vec![0xAB; 4096]),
            ("damaged header", flipped),
            ("0xFF in the middle", hole),
        ] {
            fs::write(dir.join(WEIGHTS_FILE), &bytes).unwrap();
            assert!(load(&dir, &device).is_err(), "{what} must be an error");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_hostile_config_is_refused_before_anything_is_built() {
        let device = Device::flex();
        let cfg = Preset::Nano.config(16);
        let dir = scratch_dir("hostile-config");
        save(&dir, &cfg.init_seeded(&device, 1), &cfg, &Meta::default()).unwrap();
        let text = fs::read_to_string(dir.join(CONFIG_FILE)).unwrap();
        for (field, value) in [
            ("block_size", serde_json::json!(4_000_000_000u64)),
            ("d_model", serde_json::json!(0)),
            ("n_layer", serde_json::json!(u64::MAX)),
            ("norm_eps", serde_json::json!(-1.0)),
        ] {
            let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
            json[field] = value;
            fs::write(dir.join(CONFIG_FILE), json.to_string()).unwrap();
            assert!(load_config(&dir).is_err(), "{field}");
            assert!(load(&dir, &device).is_err(), "{field}");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_reports_missing_files() {
        let dir = scratch_dir("missing");
        assert!(load(&dir, &Device::flex()).is_err());
    }
}
