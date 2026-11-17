//! Splits one real training step (`mini` preset, batch 16 × context 256) into phases and times
//! the building blocks (forward + backward) separately, to find what does not scale with threads.
//!
//! Run with:  cargo run --release -p germen-core --example profile_step -- [threads]

use burn::grad_clipping::GradientClippingConfig;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{RmsNorm, RmsNormConfig};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use germen_core::config::Preset;
use germen_core::data::{Batch, Rng};
use std::time::Instant;

fn time(label: &str, reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("  {label:<46} {ms:>9.1} ms");
    ms
}

fn main() {
    if let Some(n) = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        germen_core::sys::set_worker_threads(n);
    }
    println!("worker threads: {}", germen_core::sys::worker_threads());

    let (b, t) = (16usize, 256usize);
    let device = Device::flex().autodiff();
    let cfg = Preset::Mini.config(t);
    let mut model = cfg.init(&device);
    let mut optim = AdamWConfig::new()
        .with_beta_2(0.95)
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let mut rng = Rng::new(7);

    // ---- phases of a real training step ------------------------------------------------------
    println!("phases of one real step (mini, 16 x 256), mean of 4 steps:");
    let (mut fwd, mut bwd, mut grads_t, mut opt) = (0.0, 0.0, 0.0, 0.0);
    let steps = 4;
    for i in 0..steps + 1 {
        let batch = Batch::random(b, t, 256, &mut rng);
        let (x, y) = batch.to_tensors(&device);
        let t0 = Instant::now();
        let loss = model.loss(x, y);
        let _ = loss.clone().into_scalar::<f32>();
        let t1 = Instant::now();
        let g = loss.backward();
        let t2 = Instant::now();
        let gp = GradientsParams::from_grads(g, &model);
        let t3 = Instant::now();
        model = optim.step(1e-4, model, gp);
        let t4 = Instant::now();
        if i > 0 {
            fwd += (t1 - t0).as_secs_f64();
            bwd += (t2 - t1).as_secs_f64();
            grads_t += (t3 - t2).as_secs_f64();
            opt += (t4 - t3).as_secs_f64();
        }
    }
    let n = steps as f64;
    let total = (fwd + bwd + grads_t + opt) / n * 1e3;
    for (name, v) in [
        ("forward", fwd),
        ("backward", bwd),
        ("collect gradients", grads_t),
        ("optimizer step (AdamW + clip)", opt),
    ] {
        println!(
            "  {name:<46} {:>9.1} ms  ({:>4.1} %)",
            v / n * 1e3,
            v / n * 1e3 / total * 100.0
        );
    }
    println!(
        "  {:<46} {total:>9.1} ms  = {:.0} tok/s",
        "total",
        (b * t) as f64 / (total / 1e3)
    );

    // ---- building blocks, forward + backward -------------------------------------------------
    println!("building blocks, forward + backward (one layer unless noted):");
    let (d, h) = (cfg.d_model, cfg.n_head);
    let hd = d / h;
    let nrm = Distribution::Normal(0.0, 0.5);
    let x3 = Tensor::<3>::random([b, t, d], nrm, &device).require_grad();

    let norm: RmsNorm = RmsNormConfig::new(d).init(&device);
    time("RMSNorm [16,256,384]", 5, || {
        let _ = norm.forward(x3.clone()).sum().backward();
    });

    let q = Tensor::<4>::random([b, h, t, hd], nrm, &device).require_grad();
    let k = Tensor::<4>::random([b, h, t, hd], nrm, &device).require_grad();
    let v = Tensor::<4>::random([b, h, t, hd], nrm, &device).require_grad();
    time("causal attention [16,6,256,64]", 3, || {
        let o = attention(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            },
        );
        let _ = o.sum().backward();
    });

    let cos = Tensor::<4>::random([1, 1, t, hd / 2], nrm, &device);
    let sin = Tensor::<4>::random([1, 1, t, hd / 2], nrm, &device);
    time("RoPE on q [16,6,256,64]", 5, || {
        let half = hd / 2;
        let x1 = q.clone().slice([0..b, 0..h, 0..t, 0..half]);
        let x2 = q.clone().slice([0..b, 0..h, 0..t, half..hd]);
        let o1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
        let o2 = x1 * sin.clone() + x2 * cos.clone();
        let _ = Tensor::cat(vec![o1, o2], 3).sum().backward();
    });

    time("head + cross-entropy [4096,384]x[384,256]", 5, || {
        let w = Tensor::<2>::random([d, 256], nrm, &device).require_grad();
        let tokens = Tensor::<2, Int>::zeros([b * t, 1], &device).reshape([b * t]);
        let logits = x3.clone().reshape([b * t, d]).matmul(w);
        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, tokens);
        let _ = loss.backward();
    });

    let split_src = Tensor::<3>::random([b, t, 3 * d], nrm, &device).require_grad();
    time("split qkv + reshape + swap_dims (x3)", 5, || {
        let mut acc = None;
        for lo in [0, d, 2 * d] {
            let s = split_src
                .clone()
                .slice([0..b, 0..t, lo..lo + d])
                .reshape([b, t, h, hd])
                .swap_dims(1, 2);
            acc = Some(match acc {
                None => s.sum(),
                Some(a) => a + s.sum(),
            });
        }
        let _ = acc.unwrap().backward();
    });

    let big = Tensor::<2>::random([b * t, cfg.d_ff], nrm, &device).require_grad();
    time("elementwise relu(x)*relu(x) [4096,1536]", 5, || {
        let r = burn::tensor::activation::relu(big.clone());
        let _ = (r.clone() * r).sum().backward();
    });
}
