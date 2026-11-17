//! Compares the feed-forward variants (forward + backward, with autodiff) at `mini` shapes.
//!
//! Run with:  cargo run --release -p germen-core --example profile_mlp -- [threads]

use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::{relu, silu};
use std::time::Instant;

fn time(label: &str, reps: usize, mut f: impl FnMut()) {
    f(); // warm-up
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("{label:<52} {ms:>9.1} ms");
}

fn main() {
    if let Some(n) = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        germen_core::sys::set_worker_threads(n);
    }
    println!("worker threads: {}", germen_core::sys::worker_threads());

    let device = Device::flex().autodiff();
    let (tokens, d, dff, hid) = (4096usize, 384usize, 1536usize, 1024usize);
    let n = Distribution::Normal(0.0, 0.5);
    let x = Tensor::<2>::random([tokens, d], n, &device).require_grad();
    let w_up = Tensor::<2>::random([d, dff], n, &device).require_grad();
    let w_down = Tensor::<2>::random([dff, d], n, &device).require_grad();
    let g_gate = Tensor::<2>::random([d, hid], n, &device).require_grad();
    let g_up = Tensor::<2>::random([d, hid], n, &device).require_grad();
    let g_down = Tensor::<2>::random([hid, d], n, &device).require_grad();

    time("relu^2 MLP (d_ff 1536): forward + backward", 3, || {
        let h = relu(x.clone().matmul(w_up.clone()));
        let y = (h.clone() * h).matmul(w_down.clone());
        let _ = y.sum().backward();
    });
    time("relu^2 MLP: forward only", 3, || {
        let h = relu(x.clone().matmul(w_up.clone()));
        let _ = (h.clone() * h).matmul(w_down.clone());
    });
    time("SwiGLU, burn silu (hidden 1024): fwd + bwd", 3, || {
        let gate = x.clone().matmul(g_gate.clone());
        let up = x.clone().matmul(g_up.clone());
        let y = (silu(gate) * up).matmul(g_down.clone());
        let _ = y.sum().backward();
    });
    time("SwiGLU, silu = x / (1 + exp(-x)): fwd + bwd", 3, || {
        let gate = x.clone().matmul(g_gate.clone());
        let up = x.clone().matmul(g_up.clone());
        let s = gate.clone() / (gate.neg().exp().add_scalar(1.0));
        let y = (s * up).matmul(g_down.clone());
        let _ = y.sum().backward();
    });
    time("SwiGLU, silu forward only", 3, || {
        let gate = x.clone().matmul(g_gate.clone());
        let up = x.clone().matmul(g_up.clone());
        let _ = (silu(gate) * up).matmul(g_down.clone());
    });
}
