//! Writes everything an independent implementation needs to check GERMEN's network: the weights
//! (replaced by deterministic, deliberately large values so that no layer is close to a no-op), a
//! batch, and what GERMEN computes from them: the logits of the training path (queries in blocks,
//! under autodiff), the logits of the evaluation path (Burn's fused attention), the logits of the
//! cached path (the prompt in pieces), the loss and the gradient of the loss for every parameter.
//!
//! `cargo run --release -p germen-core --example reference_dump -- OUT.json`, then
//! `python3 scripts/reference_check.py OUT.json` (see docs/AUDIT.md, round 4).
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::AdamWConfig;
use burn::optim::GradientsParams;
use burn::prelude::*;
use germen_core::config::GptConfig;
use serde_json::json;

/// splitmix64, the same generator everywhere in this harness (the Python side does not need it:
/// the values are written out).
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f32 {
    (next(state) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

struct Fill {
    path: Vec<String>,
    seed: u64,
    device: Device,
}

impl ModuleMapper for Fill {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let device = self.device.clone();
        let name = self.path.join(".");
        let mut state = self.seed
            ^ name
                .bytes()
                .fold(7u64, |h, b| h.wrapping_mul(131) ^ b as u64);
        let gain = name.ends_with("gamma");
        param.map(|t| {
            let dims = t.dims();
            let n: usize = dims.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    let u = uniform(&mut state);
                    if gain { 1.0 + 0.5 * u } else { 0.8 * u }
                })
                .collect();
            Tensor::<D>::from_data(TensorData::new(v, dims), &device).require_grad()
        })
    }
}

/// A parameter: its name, shape, values and (when tracked) gradient.
type Entry = (String, Vec<usize>, Vec<f32>, Option<Vec<f32>>);

struct Names {
    path: Vec<String>,
    out: Vec<Entry>,
    grads: GradientsParams,
}

impl ModuleVisitor for Names {
    fn enter_module(&mut self, name: &str, _: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let t = param.val();
        let dims = t.dims().to_vec();
        let data = t.into_data().try_to_vec::<f32>().unwrap();
        let g = self
            .grads
            .get::<D>(param.id)
            .map(|g| g.into_data().try_to_vec::<f32>().unwrap());
        self.out.push((self.path.join("."), dims, data, g));
    }
}

struct Flat(Vec<Vec<f32>>);

impl ModuleVisitor for Flat {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.0
            .push(param.val().into_data().try_to_vec::<f32>().unwrap());
    }
}

fn weights_of(model: &germen_core::model::Gpt) -> Vec<Vec<f32>> {
    let mut f = Flat(vec![]);
    model.visit(&mut f);
    f.0
}

fn ints(device: &Device, data: Vec<i32>, b: usize, t: usize) -> Tensor<2, Int> {
    Tensor::<2, Int>::from_data(TensorData::new(data, [b, t]), device)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args
        .get(1)
        .expect("usage: reference_dump OUT.json [D_MODEL N_HEAD D_FF N_LAYER SEQ]");
    // The default geometry is small enough for finite differences over every parameter; a larger
    // one (for example `64 2 160 3 150`: head dimension 32, so blocks of 64 queries) is checked on
    // the logits and AdamW alone (`--skip-gradients`). SEQ must be at least 37.
    let g = |i: usize, default: usize| args.get(i).map_or(default, |v| v.parse().unwrap());
    let (d_model, n_head, d_ff, n_layer, seq) = (g(2, 16), g(3, 2), g(4, 40), g(5, 2), g(6, 70));
    let cfg = GptConfig {
        vocab_size: 24,
        block_size: seq + 10,
        n_layer,
        n_head,
        d_model,
        d_ff,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
    };
    cfg.validate().unwrap();
    // 70 tokens is more than two attention blocks of 32 (head dimension 8), so the blocked
    // training path really has several blocks to get wrong.
    let (b, t) = (2usize, seq);
    let mut s = 12345u64;
    let toks: Vec<i32> = (0..b * t).map(|_| (next(&mut s) % 24) as i32).collect();
    let tgts: Vec<i32> = (0..b * t).map(|_| (next(&mut s) % 24) as i32).collect();

    let device = Device::flex().autodiff();
    let model = cfg.init_seeded(&device, 1);
    let model = model.map(&mut Fill {
        path: vec![],
        seed: 99,
        device: device.clone(),
    });
    let logits_train: Vec<f32> = model
        .forward(ints(&device, toks.clone(), b, t))
        .into_data()
        .try_to_vec()
        .unwrap();
    let loss = model.loss(
        ints(&device, toks.clone(), b, t),
        ints(&device, tgts.clone(), b, t),
    );
    let loss_value = loss.clone().into_scalar::<f32>();
    let grads = GradientsParams::from_grads(loss.backward(), &model);

    let plain = model.clone().valid();
    let pdev = plain.devices()[0].clone();
    let logits_eval: Vec<f32> = plain
        .forward(ints(&pdev, toks.clone(), b, t))
        .into_data()
        .try_to_vec()
        .unwrap();
    // The cached path: 20 tokens, then 17, then one at a time.
    let mut cache = plain.new_cache();
    let mut cached: Vec<f32> = Vec::new();
    let row0: Vec<i32> = toks[..t].to_vec();
    let mut at = 0;
    for chunk in [20usize, 17]
        .into_iter()
        .chain(std::iter::repeat_n(1, t - 37))
    {
        let ids = ints(&pdev, row0[at..at + chunk].to_vec(), 1, chunk);
        cached.extend(
            plain
                .forward_cached(ids, &mut cache)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
        );
        at += chunk;
    }

    let mut names = Names {
        path: vec![],
        out: vec![],
        grads,
    };
    model.visit(&mut names);
    // AdamW as GERMEN configures it (train.rs), five steps with the same gradient and a changing
    // learning rate, so that bias correction, the decoupled decay and epsilon all matter.
    let (wd, beta2) = (0.1f32, 0.95f32);
    let lrs = [3e-4f64, 6e-4, 9e-4, 3e-3, 1e-3];
    let mut optim = AdamWConfig::new()
        .with_beta_2(beta2)
        .with_epsilon(1e-8)
        .with_weight_decay(wd)
        .init();
    let mut stepped = model.clone();
    let mut after: Vec<Vec<Vec<f32>>> = vec![];
    for lr in lrs {
        let g = GradientsParams::from_grads(
            model
                .loss(
                    ints(&device, toks.clone(), b, t),
                    ints(&device, tgts.clone(), b, t),
                )
                .backward(),
            &model,
        );
        // The gradient belongs to `model`'s parameter ids, which `stepped` shares.
        stepped = optim.step(lr, stepped, g);
        after.push(weights_of(&stepped));
    }
    let tensors: Vec<_> = names
        .out
        .iter()
        .map(|(n, d, v, g)| json!({"name": n, "shape": d, "data": v, "grad": g}))
        .collect();
    let doc = json!({
        "config": cfg,
        "batch": b, "seq": t,
        "tokens": toks, "targets": tgts,
        "tensors": tensors,
        "loss": loss_value,
        "logits_train": logits_train,
        "logits_eval": logits_eval,
        "logits_cached_row0": cached,
        "adamw": {"lrs": lrs, "weight_decay": wd, "beta1": 0.9, "beta2": beta2, "epsilon": 1e-8, "after": after},
    });
    std::fs::write(out, serde_json::to_string(&doc).unwrap()).unwrap();
    println!("{} tensors, loss {loss_value}", names.out.len());
}
