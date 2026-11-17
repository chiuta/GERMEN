//! Times the individual tensor operations of a training step, to see which ones use all cores.
//!
//! Run with:  cargo run --release -p germen-core --example profile_ops -- [threads]
//! Shapes are those of the `mini` preset at batch 16 × context 256.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::{gelu, relu, sigmoid, silu, softmax};
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use std::time::Instant;

fn time<T>(label: &str, reps: usize, mut f: impl FnMut() -> T) {
    let _ = f(); // warm-up
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("{label:<44} {ms:>9.2} ms");
}

fn main() {
    if let Some(n) = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        germen_core::sys::set_worker_threads(n);
    }
    println!("worker threads: {}", germen_core::sys::worker_threads());

    let device = Device::flex();
    let (b, t, d, h, dff) = (16usize, 256usize, 384usize, 6usize, 1536usize);
    let hd = d / h;
    let n = Distribution::Normal(0.0, 1.0);

    let x = Tensor::<2>::random([b * t, d], n, &device);
    let w_up = Tensor::<2>::random([d, dff], n, &device);
    let up = Tensor::<2>::random([b * t, dff], n, &device);
    let x3 = Tensor::<3>::random([b, t, d], n, &device);
    let scores = Tensor::<4>::random([b, h, t, t], n, &device);
    let q = Tensor::<4>::random([b, h, t, hd], n, &device);
    let k = Tensor::<4>::random([b, h, t, hd], n, &device);
    let v = Tensor::<4>::random([b, h, t, hd], n, &device);
    let norm: RmsNorm = RmsNormConfig::new(d).init(&device);

    time("matmul [4096x384] @ [384x1536]", 5, || {
        x.clone().matmul(w_up.clone())
    });
    time("gelu [4096x1536]", 5, || gelu(up.clone()));
    time("add (residual) [4096x384]", 5, || x.clone() + x.clone());
    time("mul_scalar [4096x1536]", 5, || up.clone().mul_scalar(0.5));
    time("relu [4096x1536]", 5, || relu(up.clone()));
    time("relu squared [4096x1536]", 5, || {
        let r = relu(up.clone());
        r.clone() * r
    });
    time("silu [4096x1536]", 5, || silu(up.clone()));
    time("sigmoid [4096x1536]", 5, || sigmoid(up.clone()));
    time("tanh [4096x1536]", 5, || up.clone().tanh());
    time("exp [4096x1536]", 5, || up.clone().exp());
    time("erf [4096x1536]", 5, || up.clone().erf());
    time("quick-gelu x*sigmoid(1.702x)", 5, || {
        up.clone() * sigmoid(up.clone().mul_scalar(1.702))
    });
    time("tanh-gelu composed from basic ops", 5, || {
        let x = up.clone();
        let inner =
            (x.clone() + x.clone().powi_scalar(3).mul_scalar(0.044715)).mul_scalar(0.797_884_6);
        x.mul_scalar(0.5) * (inner.tanh().add_scalar(1.0))
    });
    time("rmsnorm [16x256x384]", 5, || norm.forward(x3.clone()));
    time("softmax [16x6x256x256], dim 3", 5, || {
        softmax(scores.clone(), 3)
    });
    time("transpose copy [16x256x6x64] -> [16x6x256x64]", 5, || {
        x3.clone()
            .reshape([b, t, h, hd])
            .swap_dims(1, 2)
            .reshape([b, h, t, hd])
    });
    time("attention fallback (causal) forward", 3, || {
        attention(
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
        )
    });
    time("sum [4096x1536]", 5, || up.clone().sum());
}
