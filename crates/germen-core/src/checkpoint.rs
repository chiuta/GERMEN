//! Saving and loading a trained model.
//!
//! A checkpoint is a directory with these files:
//!
//! | file             | content                                                              |
//! |------------------|----------------------------------------------------------------------|
//! | `config.json`    | architecture ([`GptConfig`]) — enough to rebuild the empty network   |
//! | `weights.bpk`    | all parameters in Burn's "burnpack" format (f32)                     |
//! | `meta.json`      | training progress ([`Meta`]) and the size and SHA-256 of the others  |
//! | `optimizer.bpk`  | AdamW moment estimates, needed to resume training exactly            |
//! | `tokenizer.json` | the vocabulary the model was trained with                            |
//!
//! [`save_all`] writes every file under a temporary name, flushes them to the disk and only then
//! renames them into place, `meta.json` last. Weights that contain NaN or infinity are refused
//! instead of replacing a good checkpoint, both when saving and when loading.
//!
//! Two things keep the files from drifting apart:
//!
//! * **The manifest.** `meta.json` records the size and SHA-256 of every other file of the set
//!   ([`Meta::files`]). Loading checks them, so a bit that rotted on the disk, a half-finished
//!   copy, or the weights of one save next to the optimizer of another is an error with a name
//!   on it, not a model that quietly behaves a little worse. Checkpoints from before 0.4.0 have no
//!   manifest and load as they always did ([`Verification::NoManifest`]).
//! * **The journal.** The temporary files of a save are all complete, and `meta.json.tmp`
//!   (with the manifest) exists, before the first one is renamed. A run that dies in the middle of
//!   the renames therefore leaves everything needed to finish the job, and [`recover`] does: it
//!   completes such a save, or discards one that never got that far, so that the checkpoint is
//!   always either the old set or the new one.

use crate::config::GptConfig;
use crate::fsutil::{DirLock, Staged, rename_over, sync_dir, tmp_path};
use crate::integrity;
use crate::model::Gpt;
use crate::tokenizer::{TOKENIZER_FILE, Tokenizer};
use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::ModuleOptimizer;
use burn::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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

/// Every file a checkpoint can consist of, in the order a save publishes them.
const SET_FILES: [&str; 5] = [
    TOKENIZER_FILE,
    OPTIMIZER_FILE,
    WEIGHTS_FILE,
    CONFIG_FILE,
    META_FILE,
];

/// Size and SHA-256 of one file of a checkpoint, as recorded in `meta.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSum {
    /// Length of the file in bytes.
    pub bytes: u64,
    /// SHA-256 of its content, lowercase hexadecimal (what `sha256sum` prints).
    pub sha256: String,
}

impl FileSum {
    fn of_file(path: &Path) -> Result<FileSum, String> {
        let (bytes, sha256) = integrity::file_sum(path)
            .map_err(|e| format!("nu pot citi {} pentru suma de control: {e}", path.display()))?;
        Ok(FileSum { bytes, sha256 })
    }
}

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
    /// Size and SHA-256 of the other files of the set, by file name. Written by every save since
    /// GERMEN 0.4.0; empty in older checkpoints, which cannot be verified. A file that is not
    /// listed here does not belong to the checkpoint.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, FileSum>,
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
/// replaces its predecessor, and `meta.json`, which says how far training got and carries the
/// size and SHA-256 of all the others ([`Meta::files`], replaced here by the fresh ones), is
/// replaced last.
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

    let mut staged = Staged::new(dir)?;
    let mut names: Vec<&str> = Vec::new();
    if let Some(tokenizer) = tokenizer {
        staged.bytes(TOKENIZER_FILE, tokenizer.to_json()?.as_bytes())?;
        names.push(TOKENIZER_FILE);
    }
    if let Some(optimizer) = optimizer {
        staged.with(OPTIMIZER_FILE, |tmp| {
            optimizer
                .save(tmp)
                .map_err(|e| format!("nu pot salva starea optimizatorului: {e}"))
        })?;
        names.push(OPTIMIZER_FILE);
    }
    staged.with(WEIGHTS_FILE, |tmp| {
        plain
            .save_file(tmp)
            .map_err(|e| format!("nu pot salva ponderile: {e}"))
    })?;
    names.push(WEIGHTS_FILE);
    staged.bytes(CONFIG_FILE, cfg_json.as_bytes())?;
    names.push(CONFIG_FILE);

    // The sums are taken from the files as they now lie on the disk (the big ones in parallel),
    // so that they describe exactly what will be published.
    let sums: Vec<Result<(String, FileSum), String>> = names
        .par_iter()
        .map(|name| Ok((name.to_string(), FileSum::of_file(&staged.tmp_of(name))?)))
        .collect();
    let mut meta = meta.clone();
    meta.files = sums
        .into_iter()
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    staged.bytes(META_FILE, meta_json.as_bytes())?;
    staged.commit()
}

/// Writes the optimizer state (AdamW moments) to `optimizer.bpk` in `dir`, atomically. If `dir`
/// holds a checkpoint with a manifest, its `meta.json` is replaced in the same set, so that the
/// new file is the one the manifest vouches for.
pub fn save_optimizer(dir: &Path, optim: &ModuleOptimizer) -> Result<(), String> {
    let mut staged = Staged::new(dir)?;
    staged.with(OPTIMIZER_FILE, |tmp| {
        optim
            .save(tmp)
            .map_err(|e| format!("nu pot salva starea optimizatorului: {e}"))
    })?;
    if let Some(mut meta) = read_meta(dir)?.filter(|m| !m.files.is_empty()) {
        meta.files.insert(
            OPTIMIZER_FILE.to_string(),
            FileSum::of_file(&staged.tmp_of(OPTIMIZER_FILE))?,
        );
        let meta_json = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
        staged.bytes(META_FILE, meta_json.as_bytes())?;
    }
    staged.commit()
}

/// What [`verify`] could say about a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verification {
    /// Every file that was asked for matches the size and SHA-256 recorded in `meta.json`.
    Verified,
    /// `meta.json` has no manifest (the checkpoint was written before GERMEN 0.4.0, or has no
    /// `meta.json` at all), so nothing could be checked. Not an error; the next save adds one.
    NoManifest,
}

/// Checks the files `names` of the checkpoint in `dir` against the manifest in `meta`.
///
/// Files that the manifest does not list are skipped here (see [`Meta::files`] for what that
/// means); a listed file that is missing, has another length, or has another SHA-256 is an error
/// that names the file.
pub fn verify(dir: &Path, meta: &Meta, names: &[&str]) -> Result<Verification, String> {
    if meta.files.is_empty() {
        return Ok(Verification::NoManifest);
    }
    let checks: Vec<Result<(), String>> = names
        .par_iter()
        .map(|name| {
            let Some(want) = meta.files.get(*name) else {
                return Ok(());
            };
            let path = dir.join(name);
            let length = match fs::metadata(&path) {
                Ok(m) => m.len(),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(format!(
                        "lipsește {}, deși meta.json îl listează: punctul de control e \
                         incomplet (copiat pe jumătate sau cu o salvare întreruptă)",
                        path.display()
                    ));
                }
                Err(e) => return Err(format!("nu pot citi {}: {e}", path.display())),
            };
            if length != want.bytes {
                return Err(format!(
                    "{} are {length} de octeți, dar meta.json înregistrează {}: fișier trunchiat, \
                     prelungit sau dintr-o altă salvare",
                    path.display(),
                    want.bytes
                ));
            }
            let have = FileSum::of_file(&path)?;
            if have.sha256 != want.sha256 {
                // `chars`, not a byte slice: the recorded text comes from a file anyone can edit.
                let short = |sum: &str| sum.chars().take(12).collect::<String>();
                return Err(format!(
                    "{}: suma SHA-256 ({}…) nu corespunde celei din meta.json ({}…): fișier \
                     deteriorat sau dintr-o altă salvare",
                    path.display(),
                    short(&have.sha256),
                    short(&want.sha256)
                ));
            }
            Ok(())
        })
        .collect();
    checks
        .into_iter()
        .collect::<Result<Vec<()>, String>>()
        .map_err(|e| format!("{e}{}", interrupted_save_hint(dir)))?;
    Ok(Verification::Verified)
}

/// What to add to a verification error when `dir` holds the temporary files of an interrupted
/// save: the files that do not match are then very likely the two halves of that save, and there
/// is a command that finishes it. (`meta.json.tmp` is always among them: it is renamed last.)
fn interrupted_save_hint(dir: &Path) -> &'static str {
    if tmp_path(&dir.join(META_FILE)).exists() {
        "; directorul conține o salvare întreruptă (meta.json.tmp): \
         `germen verify --recover <director>` o termină"
    } else {
        ""
    }
}

/// Reads `meta.json` of `dir` and checks the files `names` against it (see [`verify`]). A
/// directory without `meta.json` has no manifest.
pub(crate) fn verify_files(dir: &Path, names: &[&str]) -> Result<Verification, String> {
    match read_meta(dir)? {
        Some(meta) => verify(dir, &meta, names),
        None => Ok(Verification::NoManifest),
    }
}

/// What [`recover`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recovery {
    /// Nothing was left over from an earlier save.
    Nothing,
    /// An earlier save had written all its files but died before publishing them all; the
    /// remaining `usize` files have now been published, so the checkpoint is that save's.
    Completed(usize),
    /// An earlier save died while still writing; its `usize` unfinished files were removed and
    /// the checkpoint is the previous one, untouched.
    Discarded(usize),
}

/// Finishes or abandons a save that was interrupted, so that `dir` holds one whole checkpoint.
///
/// A save writes all its files as `<name>.tmp`, `meta.json.tmp` last, and only then renames them
/// into place. If `meta.json.tmp` is not there, the writing never finished and nothing was
/// published: the leftovers are deleted. If it is there, every other temporary file that remains
/// is checked against its manifest and renamed into place, and `meta.json` goes last. Nothing
/// here can make a checkpoint worse: when the leftovers do not add up, they are deleted and the
/// verification at load time reports what is wrong with what is left.
///
/// Call this only while no other process writes to `dir`, i.e. under the directory claim.
pub fn recover(dir: &Path) -> Result<Recovery, String> {
    let left: Vec<&str> = SET_FILES
        .iter()
        .copied()
        .filter(|name| tmp_path(&dir.join(name)).exists())
        .collect();
    if left.is_empty() {
        return Ok(Recovery::Nothing);
    }
    let discard = || {
        for name in &left {
            let _ = fs::remove_file(tmp_path(&dir.join(name)));
        }
        Ok(Recovery::Discarded(left.len()))
    };
    let meta_tmp = tmp_path(&dir.join(META_FILE));
    let Some(meta) = fs::read_to_string(&meta_tmp)
        .ok()
        .and_then(|text| serde_json::from_str::<Meta>(&text).ok())
    else {
        return discard();
    };
    let others: Vec<&str> = left.iter().copied().filter(|n| *n != META_FILE).collect();
    let all_match = others.iter().all(|name| {
        meta.files.get(*name).is_some_and(|want| {
            FileSum::of_file(&tmp_path(&dir.join(name))).is_ok_and(|have| have == *want)
        })
    });
    if !all_match {
        return discard();
    }
    for name in &others {
        rename_over(&tmp_path(&dir.join(name)), &dir.join(name))?;
    }
    rename_over(&meta_tmp, &dir.join(META_FILE))?;
    sync_dir(dir);
    Ok(Recovery::Completed(left.len()))
}

/// [`recover`] for a caller that is not training (`germen verify --recover`).
///
/// It takes the directory claim first, so that it cannot rename files under a run that is writing
/// to `dir`; while another run holds the directory it fails without touching anything, and it
/// does not create `dir` when there is none.
pub fn recover_locked(dir: &Path) -> Result<Recovery, String> {
    if !dir.is_dir() {
        return Err(format!("{} nu există sau nu e un director", dir.display()));
    }
    let _claim = DirLock::acquire(dir)?;
    recover(dir)
}

/// Whether `dir` holds a checkpoint (weights or progress record).
pub fn exists(dir: &Path) -> bool {
    dir.join(WEIGHTS_FILE).exists() || dir.join(META_FILE).exists()
}

/// Loads the optimizer state saved next to a checkpoint into `optim`, after checking that it
/// belongs to `model`.
///
/// Returns the optimizer and whether a state was found; checkpoints written before GERMEN 0.2
/// have none, and then `optim` is returned unchanged. The file must hold state for parameters of
/// `model` only, with the shapes `model` has: the state of another model, or of the same one with
/// other parameter identities (a `weights.bpk` and an `optimizer.bpk` from two different runs),
/// would otherwise be accepted and silently never used, and a file whose entries are damaged
/// would stop the process inside the library. Those are errors here.
pub fn load_optimizer(
    dir: &Path,
    optim: ModuleOptimizer,
    model: &Gpt,
) -> Result<(ModuleOptimizer, bool), String> {
    let path = dir.join(OPTIMIZER_FILE);
    if !path.exists() {
        return Ok((optim, false));
    }
    let stored = check_optimizer_file(&path, model)?;
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| optim.load(&path)))
        .map_err(|_| {
            format!(
                "{} e corupt: starea optimizatorului nu poate fi reconstruită",
                path.display()
            )
        })?;
    let optim = loaded.map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    // Burn drops, without a word, the state of any parameter whose entries it cannot put back
    // together (a damaged name, a missing piece). Nothing may go missing: what was in the file
    // is what the optimizer now holds.
    let kept = optim.to_record().len();
    if kept != stored {
        return Err(format!(
            "{} e corupt: {} din {stored} tensori de stare nu au putut fi reconstruiți",
            path.display(),
            stored - kept.min(stored)
        ));
    }
    Ok((optim, true))
}

/// The identity and shape of every parameter of a model.
struct ParamShapes(BTreeMap<u64, Vec<usize>>);

impl ModuleVisitor for ParamShapes {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.0.insert(param.id.val(), param.lazy_shape().to_vec());
    }
}

/// Reads the header of an optimizer state file and checks that every tensor in it is the state of
/// a parameter of `model`, with the shape of that parameter. Returns the number of tensors.
fn check_optimizer_file(path: &Path, model: &Gpt) -> Result<usize, String> {
    let shown = path.display();
    let reader = burn_pack::Reader::from_file_exact(path).map_err(|e| format!("{shown}: {e}"))?;
    let tensors = reader.into_tensors().map_err(|e| format!("{shown}: {e}"))?;
    let mut params = ParamShapes(BTreeMap::new());
    model.visit(&mut params);
    let mut foreign = 0usize;
    for tensor in &tensors {
        let Some(id) = tensor.param_id else {
            return Err(format!(
                "{shown} e corupt: tensorul „{}” nu spune cărui parametru îi aparține",
                tensor.name
            ));
        };
        // Every state tensor is named after its parameter: "<id>.momentum.moment_1".
        if !tensor.name.starts_with(&format!("{id}.")) {
            return Err(format!(
                "{shown} e corupt: tensorul „{}” poartă alt identificator decât {id}",
                tensor.name
            ));
        }
        match params.0.get(&id) {
            None => foreign += 1,
            Some(shape) if *shape != tensor.shape.to_vec() => {
                return Err(format!(
                    "{shown}: starea „{}” are forma {:?}, dar parametrul modelului are {:?}: \
                     optimizatorul e al altui model",
                    tensor.name,
                    tensor.shape.to_vec(),
                    shape
                ));
            }
            Some(_) => {}
        }
    }
    if foreign > 0 {
        return Err(format!(
            "{shown}: {foreign} din {} tensori de stare nu aparțin niciunui parametru al \
             acestor ponderi (optimizatorul și ponderile vin din rulări sau din salvări \
             diferite); ștergeți {} ca să reluați cu un optimizator proaspăt",
            tensors.len(),
            OPTIMIZER_FILE
        ));
    }
    Ok(tensors.len())
}

/// Reads only the architecture from a checkpoint directory.
pub fn load_config(dir: &Path) -> Result<GptConfig, String> {
    verify_files(dir, &[CONFIG_FILE])?;
    let path = dir.join(CONFIG_FILE);
    let text =
        fs::read_to_string(&path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    let cfg: GptConfig =
        serde_json::from_str(&text).map_err(|e| format!("{} e invalid: {e}", path.display()))?;
    cfg.validate()?;
    Ok(cfg)
}

/// `meta.json` of `dir`, or `None` if there is none.
fn read_meta(dir: &Path) -> Result<Option<Meta>, String> {
    let path = dir.join(META_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let text =
        fs::read_to_string(&path).map_err(|e| format!("nu pot citi {}: {e}", path.display()))?;
    let meta =
        serde_json::from_str(&text).map_err(|e| format!("{} e invalid: {e}", path.display()))?;
    Ok(Some(meta))
}

/// Reads only the training metadata (missing file → defaults).
pub fn load_meta(dir: &Path) -> Result<Meta, String> {
    Ok(read_meta(dir)?.unwrap_or_default())
}

/// Rebuilds the network from a checkpoint directory and loads its weights, after checking them
/// against the manifest in `meta.json` when there is one.
pub fn load(dir: &Path, device: &Device) -> Result<(Gpt, GptConfig, Meta), String> {
    let cfg = load_config(dir)?;
    let weights = dir.join(WEIGHTS_FILE);
    if !weights.exists() {
        return Err(format!("lipsește {}", weights.display()));
    }
    let meta = load_meta(dir)?;
    verify(dir, &meta, &[WEIGHTS_FILE])?;
    let model = cfg
        .init(device)
        .try_load_file(&weights)
        .map_err(|e| format!("nu pot încărca {}: {e}", weights.display()))?;
    ensure_finite(&model.clone().valid())
        .map_err(|e| format!("{} este corupt: {e}", weights.display()))?;
    Ok((model, cfg, meta))
}

/// What a checkpoint directory holds, after every check that can be made without training.
#[derive(Clone, Debug)]
pub struct Inspection {
    /// The progress record (with the manifest).
    pub meta: Meta,
    /// The architecture.
    pub config: GptConfig,
    /// Each file of the checkpoint with its size, all of them checked against the manifest.
    pub files: Vec<(String, u64)>,
    /// Whether an optimizer state is present (and belongs to these weights).
    pub optimizer: bool,
    /// Temporary files of a save that was interrupted; `train --resume` finishes or discards them.
    pub leftovers: Vec<String>,
}

/// Checks a checkpoint from top to bottom without changing anything: the manifest against every
/// file, the weights for NaN and infinity, the tokenizer against the model, the optimizer state
/// against the weights. A checkpoint without a manifest (written before 0.4.0) is an error here,
/// because there is nothing to check it against.
pub fn inspect(dir: &Path) -> Result<Inspection, String> {
    let leftovers: Vec<String> = SET_FILES
        .iter()
        .filter(|name| tmp_path(&dir.join(name)).exists())
        .map(|name| format!("{name}.tmp"))
        .collect();
    let checked = || -> Result<Inspection, String> {
        let device = crate::cpu_device();
        let (model, config, meta) = load_for_resume(dir, &device)?;
        if meta.files.is_empty() {
            return Err(format!(
                "{} nu are sume de control (punctul de control e scris de GERMEN înainte de \
                 0.4.0), deci nu se poate verifica; `germen train --resume` îi adaugă un manifest \
                 la următoarea salvare",
                dir.join(META_FILE).display()
            ));
        }
        let tokenizer = Tokenizer::load(dir)?;
        if tokenizer.vocab_size() != config.vocab_size {
            return Err(format!(
                "tokenizatorul ({} tokeni) nu se potrivește cu modelul (vocabular {})",
                tokenizer.vocab_size(),
                config.vocab_size
            ));
        }
        let (_, optimizer) = load_optimizer(dir, burn::optim::AdamWConfig::new().init(), &model)?;
        let files = meta
            .files
            .iter()
            .map(|(name, sum)| (name.clone(), sum.bytes))
            .collect();
        Ok(Inspection {
            meta,
            config,
            files,
            optimizer,
            leftovers: Vec::new(),
        })
    };
    match checked() {
        Ok(found) => Ok(Inspection { leftovers, ..found }),
        Err(e) if leftovers.is_empty() => Err(e),
        Err(e) => Err(format!(
            "{e}; în director sunt fișiere {} ale unei salvări întrerupte: `germen train \
             --resume` o termină sau o șterge",
            leftovers.join(", ")
        )),
    }
}

/// Like [`load`], for continuing training: the checkpoint must be whole.
///
/// `meta.json` is required (without it the step counter, the schedule and the statistics would
/// silently start again from zero on weights that are already trained), and so is every file
/// the manifest lists. If there is a manifest, an `optimizer.bpk` it does not list is refused as
/// not belonging to this checkpoint.
pub fn load_for_resume(dir: &Path, device: &Device) -> Result<(Gpt, GptConfig, Meta), String> {
    let Some(meta) = read_meta(dir)? else {
        return Err(format!(
            "lipsește {}: punctul de control e incomplet (copiat pe jumătate sau cu o salvare \
             întreruptă), iar fără el pasul, programul de învățare și statisticile ar reporni \
             de la zero pe ponderi deja antrenate",
            dir.join(META_FILE).display()
        ));
    };
    if !meta.files.is_empty()
        && !meta.files.contains_key(OPTIMIZER_FILE)
        && dir.join(OPTIMIZER_FILE).exists()
    {
        return Err(format!(
            "{} nu e în lista din meta.json: aparține altei salvări; mutați-l sau ștergeți-l \
             ca să reluați fără starea optimizatorului",
            dir.join(OPTIMIZER_FILE).display()
        ));
    }
    verify(dir, &meta, &[OPTIMIZER_FILE])?;
    load(dir, device)
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

        let (loaded, cfg2, mut meta2) = load(&dir, &device).unwrap();
        assert_eq!(cfg2.num_params(), cfg.num_params());
        assert_eq!(
            meta2.files.len(),
            2,
            "weights and configuration are in the manifest"
        );
        meta2.files.clear();
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

        // Weights that reached the disk some other way are refused when loading: by the manifest
        // first ...
        poisoned.save_file(dir.join(WEIGHTS_FILE)).unwrap();
        let err = load(&dir, &device).unwrap_err();
        assert!(err.contains("SHA-256"), "{err}");
        // ... and, in a checkpoint that has no manifest (written before 0.4.0), by the check for
        // NaN and infinity itself.
        let mut json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(META_FILE)).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("files").unwrap();
        fs::write(dir.join(META_FILE), json.to_string()).unwrap();
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
        let mut written = load_meta(&dir).unwrap();
        assert_eq!(
            written.files.len(),
            4,
            "the manifest covers the other four files"
        );
        written.files.clear();
        assert_eq!(written, meta);
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

    // ---- integrity: the manifest, the journal, the optimizer's identity -------------------------

    use burn::optim::{AdamWConfig, GradientsParams};

    /// A tiny model after one AdamW step, so that there is optimizer state to save.
    fn after_one_step(seed: u64) -> (Gpt, ModuleOptimizer, GptConfig) {
        let device = crate::train_device();
        let cfg = Preset::Nano.config(16);
        let model = cfg.init_seeded(&device, seed);
        let mut optim: ModuleOptimizer = AdamWConfig::new().init();
        let data: Vec<i32> = (0..32).map(|i| (i * 7 + seed as i32) % 256).collect();
        let x = Tensor::<2, Int>::from_data(TensorData::new(data, [2, 16]), &device);
        let grads = GradientsParams::from_grads(model.loss(x.clone(), x).backward(), &model);
        let model = optim.step(1e-2, model, grads);
        (model, optim, cfg)
    }

    /// Writes a whole checkpoint (all five files) for the model of `seed` at `step`.
    fn write_set(dir: &Path, seed: u64, step: usize) {
        let (model, optim, cfg) = after_one_step(seed);
        let meta = Meta {
            step,
            seed,
            ..Meta::default()
        };
        save_all(
            dir,
            &model,
            &cfg,
            &meta,
            Some(&optim),
            Some(&Tokenizer::Byte),
        )
        .unwrap();
    }

    /// Flips one bit of the file, in the middle of its tensor data.
    fn flip_a_bit(path: &Path) {
        let mut bytes = fs::read(path).unwrap();
        let at = bytes.len() * 3 / 4;
        bytes[at] ^= 0x10;
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn a_manifest_lists_every_file_of_the_set_and_matches_it() {
        let dir = scratch_dir("manifest");
        write_set(&dir, 1, 4);
        let meta = load_meta(&dir).unwrap();
        let listed: Vec<&str> = meta.files.keys().map(String::as_str).collect();
        assert_eq!(
            listed,
            [
                CONFIG_FILE,
                OPTIMIZER_FILE,
                crate::tokenizer::TOKENIZER_FILE,
                WEIGHTS_FILE
            ],
            "meta.json vouches for the other four files, and not for itself"
        );
        for (name, sum) in &meta.files {
            let (bytes, sha) = integrity::file_sum(&dir.join(name)).unwrap();
            assert_eq!((sum.bytes, &sum.sha256), (bytes, &sha), "{name}");
        }
        assert_eq!(
            verify(&dir, &meta, &SET_FILES).unwrap(),
            Verification::Verified
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flipped_bit_in_a_file_is_reported_by_its_name() {
        // Before the manifest every one of these was accepted without a word: a bit that rotted
        // on the disk changed the model, or the optimizer, and nobody was told.
        for victim in [WEIGHTS_FILE, OPTIMIZER_FILE] {
            let dir = scratch_dir(&format!("bitflip-{victim}"));
            write_set(&dir, 1, 4);
            flip_a_bit(&dir.join(victim));
            let err = match victim {
                WEIGHTS_FILE => load(&dir, &Device::flex()).unwrap_err(),
                _ => load_for_resume(&dir, &Device::flex()).unwrap_err(),
            };
            assert!(err.contains(victim) && err.contains("SHA-256"), "{err}");
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn a_file_of_another_save_is_never_mixed_in() {
        // The weights of step 9 beside the progress record (and optimizer) of step 4: the shape
        // of a torn checkpoint, or of a careless copy. Both files are intact and plausible.
        let (a, b) = (scratch_dir("mix-a"), scratch_dir("mix-b"));
        write_set(&a, 1, 4);
        write_set(&b, 2, 9);
        for file in [WEIGHTS_FILE, OPTIMIZER_FILE, CONFIG_FILE] {
            let mixed = scratch_dir(&format!("mix-{file}"));
            copy_dir(&a, &mixed);
            // Both saves share one configuration, so the "other" one gets a different, valid one.
            let other: Vec<u8> = if file == CONFIG_FILE {
                let mut cfg: serde_json::Value =
                    serde_json::from_str(&fs::read_to_string(a.join(file)).unwrap()).unwrap();
                cfg["rope_theta"] = serde_json::json!(5000.0);
                cfg.to_string().into_bytes()
            } else {
                fs::read(b.join(file)).unwrap()
            };
            fs::write(mixed.join(file), other).unwrap();
            let err = load_for_resume(&mixed, &Device::flex()).unwrap_err();
            assert!(err.contains(file), "{file}: {err}");
            let _ = fs::remove_dir_all(&mixed);
        }
        let _ = fs::remove_dir_all(&a);
        let _ = fs::remove_dir_all(&b);
    }

    fn copy_dir(from: &Path, to: &Path) {
        fs::create_dir_all(to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }

    #[test]
    fn a_missing_or_resized_file_the_manifest_lists_is_an_error() {
        let dir = scratch_dir("listed-missing");
        write_set(&dir, 1, 4);
        let weights = fs::read(dir.join(WEIGHTS_FILE)).unwrap();

        // Appended junk used to be accepted by the burnpack reader.
        let mut longer = weights.clone();
        longer.extend_from_slice(&[0u8; 4096]);
        fs::write(dir.join(WEIGHTS_FILE), longer).unwrap();
        assert!(load(&dir, &Device::flex()).unwrap_err().contains("octeți"));

        fs::remove_file(dir.join(WEIGHTS_FILE)).unwrap();
        assert!(load(&dir, &Device::flex()).is_err());
        fs::write(dir.join(WEIGHTS_FILE), &weights).unwrap();
        assert!(load(&dir, &Device::flex()).is_ok());

        // A listed tokenizer that is gone must not turn into "the byte tokenizer".
        fs::remove_file(dir.join(crate::tokenizer::TOKENIZER_FILE)).unwrap();
        let err = Tokenizer::load(&dir).unwrap_err();
        assert!(err.contains("lipsește"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_checkpoint_without_a_manifest_loads_as_it_always_did() {
        // Everything written before 0.4.0: no `files` in meta.json. Still loadable, but nothing
        // can be vouched for, and the caller is told so.
        let dir = scratch_dir("legacy");
        write_set(&dir, 1, 4);
        let mut json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(META_FILE)).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("files").unwrap();
        fs::write(dir.join(META_FILE), json.to_string()).unwrap();
        flip_a_bit(&dir.join(WEIGHTS_FILE)); // cannot be noticed without a manifest, and is not

        let (_, _, meta) = load(&dir, &Device::flex()).unwrap();
        assert!(meta.files.is_empty());
        assert_eq!(meta.step, 4);
        assert_eq!(
            verify(&dir, &meta, &SET_FILES).unwrap(),
            Verification::NoManifest
        );
        assert!(load_for_resume(&dir, &Device::flex()).is_ok());
        // The next save puts a manifest in.
        let (model, optim, cfg) = after_one_step(1);
        save_all(
            &dir,
            &model,
            &cfg,
            &meta,
            Some(&optim),
            Some(&Tokenizer::Byte),
        )
        .unwrap();
        assert!(!load_meta(&dir).unwrap().files.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_manifest_written_by_hand_cannot_make_the_loader_panic() {
        let dir = scratch_dir("hostile-manifest");
        write_set(&dir, 1, 4);
        let text = fs::read_to_string(dir.join(META_FILE)).unwrap();
        let weights_sum = load_meta(&dir).unwrap().files[WEIGHTS_FILE].clone();
        for (what, files) in [
            (
                "wrong types",
                serde_json::json!({ "weights.bpk": { "bytes": -1, "sha256": 5 } }),
            ),
            ("not an object", serde_json::json!([1, 2, 3])),
            (
                "non-ASCII sum of the right length",
                serde_json::json!({ "weights.bpk": { "bytes": weights_sum.bytes, "sha256": "ăîșțâăîșțâăîșțâ" } }),
            ),
            (
                "empty sum",
                serde_json::json!({ "weights.bpk": { "bytes": weights_sum.bytes, "sha256": "" } }),
            ),
            (
                "a name that points out of the directory",
                serde_json::json!({ "../x": { "bytes": 1, "sha256": "00" } }),
            ),
        ] {
            let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
            json["files"] = files;
            fs::write(dir.join(META_FILE), json.to_string()).unwrap();
            let result = load(&dir, &Device::flex());
            match what {
                // Listing a name that is not one of ours is harmless: it is never opened.
                "a name that points out of the directory" => {
                    assert!(result.is_ok(), "{what}: {result:?}");
                }
                _ => assert!(result.is_err(), "{what} must be refused"),
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_needs_the_progress_record_but_generation_does_not() {
        let dir = scratch_dir("no-meta");
        write_set(&dir, 1, 4);
        fs::remove_file(dir.join(META_FILE)).unwrap();
        // Sampling from the weights is fine without it ...
        assert!(load(&dir, &Device::flex()).is_ok());
        // ... continuing training is not: step 0 on trained weights would silently restart the
        // learning-rate schedule and wipe the statistics.
        let err = load_for_resume(&dir, &Device::flex()).unwrap_err();
        assert!(err.contains("meta.json"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_optimizer_that_does_not_belong_to_the_weights_is_refused() {
        // Both files are healthy; they just come from two runs, so their parameter identities
        // differ. This used to be accepted, announced as "restored", and never used.
        let (model_a, optim_a, cfg) = after_one_step(1);
        let (_, optim_b, _) = after_one_step(2);
        let dir = scratch_dir("foreign-optimizer");
        save_all(&dir, &model_a, &cfg, &Meta::default(), Some(&optim_a), None).unwrap();
        let (_, found) = load_optimizer(&dir, AdamWConfig::new().init(), &model_a).unwrap();
        assert!(found, "the optimizer of the same run is accepted");

        optim_b.save(dir.join(OPTIMIZER_FILE)).unwrap();
        let err = load_optimizer(&dir, AdamWConfig::new().init(), &model_a)
            .err()
            .expect("another run's optimizer must be refused");
        assert!(err.contains("nu aparțin"), "{err}");

        // State for the right parameter but of the wrong shape (an optimizer of another size).
        let ids = {
            let mut p = ParamShapes(BTreeMap::new());
            model_a.visit(&mut p);
            p.0
        };
        let (id, shape) = ids.iter().next().unwrap();
        let mut odd = shape.clone();
        odd.push(1);
        write_pack(
            &dir.join(OPTIMIZER_FILE),
            &[(format!("{id}.momentum.moment_1"), Some(*id), odd)],
        );
        let err = load_optimizer(&dir, AdamWConfig::new().init(), &model_a)
            .err()
            .expect("a state of the wrong shape must be refused");
        assert!(err.contains("forma"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Writes a burnpack with the given `(name, parameter id, shape)` f32 tensors, all zeros.
    fn write_pack(path: &Path, tensors: &[(String, Option<u64>, Vec<usize>)]) {
        let tensors = tensors
            .iter()
            .map(|(name, id, shape)| {
                let bytes = vec![0u8; shape.iter().product::<usize>() * 4];
                burn_pack::Tensor::new(
                    name.clone(),
                    burn_pack::DType::F32,
                    shape.clone(),
                    *id,
                    burn_pack::Bytes::from_bytes_vec(bytes),
                )
            })
            .collect();
        burn_pack::Writer::new(tensors).write_to_file(path).unwrap();
    }

    #[test]
    fn an_optimizer_file_with_a_nameless_tensor_is_an_error_not_a_panic() {
        // Found by flipping bits in optimizer.bpk: Burn's loader `expect`s every tensor to carry
        // a parameter id and stops the whole process when one does not (4 of 464 mutations).
        let (model, _, _) = after_one_step(1);
        let dir = scratch_dir("nameless");
        fs::create_dir_all(&dir).unwrap();
        write_pack(
            &dir.join(OPTIMIZER_FILE),
            &[("7.momentum.moment_1".to_string(), None, vec![2])],
        );
        let err = load_optimizer(&dir, AdamWConfig::new().init(), &model)
            .err()
            .expect("must be an error");
        assert!(err.contains("nu spune cărui parametru"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_optimizer_keeps_the_manifest_true() {
        let (model, optim, cfg) = after_one_step(1);
        let dir = scratch_dir("save-optimizer");
        save(&dir, &model, &cfg, &Meta::default()).unwrap();
        save_optimizer(&dir, &optim).unwrap();
        let meta = load_meta(&dir).unwrap();
        assert!(meta.files.contains_key(OPTIMIZER_FILE));
        assert_eq!(
            verify(&dir, &meta, &[OPTIMIZER_FILE, WEIGHTS_FILE]).unwrap(),
            Verification::Verified
        );
        assert!(load_for_resume(&dir, &Device::flex()).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_optimizer_the_manifest_does_not_list_is_refused_on_resume() {
        let (model, optim, cfg) = after_one_step(1);
        let dir = scratch_dir("unlisted-optimizer");
        save(&dir, &model, &cfg, &Meta::default()).unwrap();
        optim.save(dir.join(OPTIMIZER_FILE)).unwrap(); // written behind the manifest's back
        let err = load_for_resume(&dir, &Device::flex()).unwrap_err();
        assert!(err.contains("nu e în lista"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    // ---- the journal ------------------------------------------------------------------------

    /// Turns `dir`, which holds checkpoint A, into what a save of checkpoint B (in `b`) looks
    /// like when the process dies after publishing the first `renamed` of its files: those are
    /// in place, the rest sit under their temporary names, and `meta.json` is still A's.
    fn die_after(dir: &Path, b: &Path, renamed: usize) {
        let order = [
            crate::tokenizer::TOKENIZER_FILE,
            OPTIMIZER_FILE,
            WEIGHTS_FILE,
            CONFIG_FILE,
            META_FILE,
        ];
        for (i, name) in order.iter().enumerate() {
            let bytes = fs::read(b.join(name)).unwrap();
            if i < renamed {
                fs::write(dir.join(name), bytes).unwrap();
            } else {
                fs::write(tmp_path(&dir.join(name)), bytes).unwrap();
            }
        }
    }

    #[test]
    fn a_torn_set_names_the_remedy_and_recover_locked_applies_it() {
        let b = scratch_dir("locked-b");
        write_set(&b, 2, 9);
        let dir = scratch_dir("locked");
        write_set(&dir, 1, 4);
        // Tokenizer, optimizer and weights of the new save are published; the configuration and
        // the progress record that vouches for them still wait under their temporary names.
        die_after(&dir, &b, 3);

        // Generation refuses the mixture, and says what to do about it.
        let err = load(&dir, &Device::flex()).unwrap_err();
        assert!(
            err.contains("weights.bpk") && err.contains("verify --recover"),
            "{err}"
        );

        // While another run holds the directory nothing is touched.
        let claim = DirLock::acquire(&dir).unwrap();
        let busy = recover_locked(&dir).unwrap_err();
        assert!(busy.contains("folosit"), "{busy}");
        assert!(tmp_path(&dir.join(META_FILE)).exists());
        drop(claim);

        // A path that is not a directory is an error, and nothing is created there.
        let nowhere = dir.join("nowhere");
        assert!(recover_locked(&nowhere).is_err());
        assert!(!nowhere.exists());

        assert_eq!(recover_locked(&dir).unwrap(), Recovery::Completed(2));
        let (_, _, meta) = load(&dir, &Device::flex()).unwrap();
        assert_eq!(meta.step, 9);
        assert_eq!(recover_locked(&dir).unwrap(), Recovery::Nothing);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&b);
    }

    #[test]
    fn a_save_that_dies_at_any_point_is_completed_and_the_result_verifies() {
        // Before the journal, every one of these left new weights next to an old progress record
        // (or the reverse) with nothing to say so; now recovery rolls the save forward.
        let b = scratch_dir("journal-b");
        write_set(&b, 2, 9);
        for renamed in 0..=4 {
            let dir = scratch_dir(&format!("journal-{renamed}"));
            write_set(&dir, 1, 4);
            die_after(&dir, &b, renamed);
            if renamed >= 2 {
                // The torn state is detected, not trusted. (With one file published it is the
                // tokenizer, identical in both saves, so that set is whole and loads; the
                // configuration, published fourth, is identical too.)
                assert!(
                    load_for_resume(&dir, &Device::flex()).is_err(),
                    "{renamed} files published: a mixed set must not load"
                );
            }
            let found = recover(&dir).unwrap();
            assert_eq!(
                found,
                Recovery::Completed(5 - renamed),
                "{renamed} published"
            );
            let (_, _, meta) = load_for_resume(&dir, &Device::flex()).unwrap();
            assert_eq!(meta.step, 9, "the new save won ({renamed} published)");
            assert_eq!(meta.seed, 2);
            for name in SET_FILES {
                assert_eq!(
                    fs::read(dir.join(name)).unwrap(),
                    fs::read(b.join(name)).unwrap(),
                    "{name} is B's ({renamed} published)"
                );
                assert!(!tmp_path(&dir.join(name)).exists(), "no .tmp left");
            }
            assert_eq!(recover(&dir).unwrap(), Recovery::Nothing, "idempotent");
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&b);
    }

    #[test]
    fn a_save_that_died_while_writing_is_discarded_and_the_old_set_stays() {
        let dir = scratch_dir("journal-discard");
        write_set(&dir, 1, 4);
        let before = snapshot(&dir);
        // Died while writing: some staged files, but no meta.json.tmp (it is written last).
        fs::write(tmp_path(&dir.join(WEIGHTS_FILE)), b"half of the weights").unwrap();
        fs::write(tmp_path(&dir.join(OPTIMIZER_FILE)), b"all of the optimizer").unwrap();
        assert_eq!(recover(&dir).unwrap(), Recovery::Discarded(2));
        assert_eq!(snapshot(&dir), before, "nothing but the leftovers is gone");
        assert!(load_for_resume(&dir, &Device::flex()).is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn leftovers_that_do_not_add_up_are_discarded_never_published() {
        let b = scratch_dir("journal-bad-b");
        write_set(&b, 2, 9);
        for what in [
            "flipped bit in a staged file",
            "garbage meta.json.tmp",
            "file not in manifest",
        ] {
            let dir = scratch_dir("journal-bad");
            write_set(&dir, 1, 4);
            let before = snapshot(&dir);
            die_after(&dir, &b, 0);
            match what {
                "flipped bit in a staged file" => flip_a_bit(&tmp_path(&dir.join(WEIGHTS_FILE))),
                "garbage meta.json.tmp" => {
                    fs::write(tmp_path(&dir.join(META_FILE)), b"{ not json").unwrap();
                }
                _ => {
                    // A staged file the staged progress record knows nothing about.
                    let mut json: serde_json::Value = serde_json::from_str(
                        &fs::read_to_string(tmp_path(&dir.join(META_FILE))).unwrap(),
                    )
                    .unwrap();
                    json["files"].as_object_mut().unwrap().remove(WEIGHTS_FILE);
                    fs::write(tmp_path(&dir.join(META_FILE)), json.to_string()).unwrap();
                }
            }
            assert_eq!(recover(&dir).unwrap(), Recovery::Discarded(5), "{what}");
            assert_eq!(snapshot(&dir), before, "{what}: the old set is untouched");
            let _ = fs::remove_dir_all(&dir);
        }
        let _ = fs::remove_dir_all(&b);
    }

    #[test]
    fn inspect_vouches_for_a_whole_checkpoint_and_names_what_is_wrong_with_a_bad_one() {
        let dir = scratch_dir("inspect");
        write_set(&dir, 1, 4);
        let found = inspect(&dir).unwrap();
        assert_eq!(found.meta.step, 4);
        assert!(found.optimizer);
        assert!(found.leftovers.is_empty());
        assert_eq!(found.files.len(), 4);

        // Bit rot in the optimizer: the weights are fine, and the message says which file is not.
        let good = fs::read(dir.join(OPTIMIZER_FILE)).unwrap();
        flip_a_bit(&dir.join(OPTIMIZER_FILE));
        let err = inspect(&dir).unwrap_err();
        assert!(err.contains(OPTIMIZER_FILE), "{err}");
        fs::write(dir.join(OPTIMIZER_FILE), good).unwrap();

        // A cut-short save is reported together with the way out.
        fs::write(tmp_path(&dir.join(WEIGHTS_FILE)), b"x").unwrap();
        assert_eq!(inspect(&dir).unwrap().leftovers, ["weights.bpk.tmp"]);
        flip_a_bit(&dir.join(WEIGHTS_FILE));
        let err = inspect(&dir).unwrap_err();
        assert!(
            err.contains("weights.bpk.tmp") && err.contains("--resume"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn inspect_cannot_vouch_for_a_checkpoint_without_a_manifest() {
        let dir = scratch_dir("inspect-legacy");
        write_set(&dir, 1, 4);
        let mut json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(META_FILE)).unwrap()).unwrap();
        json.as_object_mut().unwrap().remove("files").unwrap();
        fs::write(dir.join(META_FILE), json.to_string()).unwrap();
        let err = inspect(&dir).unwrap_err();
        assert!(err.contains("nu are sume de control"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn exists_means_weights_or_progress_record() {
        let dir = scratch_dir("exists");
        assert!(!exists(&dir));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".germen.lock"), b"").unwrap();
        assert!(!exists(&dir), "the lock file alone is not a checkpoint");
        fs::write(dir.join(META_FILE), b"{}").unwrap();
        assert!(exists(&dir));
        let _ = fs::remove_dir_all(&dir);
    }
}
