//! Compares ways of writing causal attention under autodiff (forward + backward) on the `mini`
//! shapes, and checks that they agree numerically.
//!
//! Run with:  cargo run --release -p germen-core --example profile_attn -- [threads]

use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::softmax;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use std::time::Instant;

const NEG: f32 = -1.0e30;

/// A: Burn's `attention`, which under autodiff is a long chain of differentiable ops.
fn attn_burn(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    attention(
        q,
        k,
        v,
        None,
        None,
        AttentionModuleOptions {
            scale: None,
            softcap: None,
            is_causal: true,
        },
    )
}

/// `bias`: `[1, 1, t, t]`, 0 where attention is allowed and a huge negative number elsewhere.
fn attn_manual(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, bias: &Tensor<4>) -> Tensor<4> {
    let hd = q.dims()[3];
    let q = q.mul_scalar(1.0 / (hd as f32).sqrt());
    let scores = q.matmul(k.swap_dims(2, 3)) + bias.clone();
    softmax(scores, 3).matmul(v)
}

/// Same, but queries are processed in blocks and each block only looks at the keys up to its own
/// end, so the (masked-out) upper-right part of the score matrix is never computed.
fn attn_chunked(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    bias: &Tensor<4>,
    chunk: usize,
) -> Tensor<4> {
    let [b, h, t, hd] = q.dims();
    let q = q.mul_scalar(1.0 / (hd as f32).sqrt());
    let mut outs = Vec::new();
    let mut lo = 0;
    while lo < t {
        let hi = (lo + chunk).min(t);
        let qi = q.clone().slice([0..b, 0..h, lo..hi, 0..hd]);
        let ki = k.clone().slice([0..b, 0..h, 0..hi, 0..hd]);
        let vi = v.clone().slice([0..b, 0..h, 0..hi, 0..hd]);
        let bi = bias.clone().slice([0..1, 0..1, lo..hi, 0..hi]);
        let s = qi.matmul(ki.swap_dims(2, 3)) + bi;
        outs.push(softmax(s, 3).matmul(vi));
        lo = hi;
    }
    Tensor::cat(outs, 2)
}

fn time(label: &str, reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("  {label:<40} {ms:>8.1} ms");
    ms
}

fn max_abs_diff(a: Tensor<4>, b: Tensor<4>) -> f32 {
    (a - b).abs().max().into_scalar::<f32>()
}

/// One way of computing attention, to be timed.
type AttentionFn = Box<dyn Fn(Tensor<4>, Tensor<4>, Tensor<4>) -> Tensor<4>>;

fn main() {
    if let Some(n) = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
    {
        germen_core::sys::set_worker_threads(n);
    }
    println!("worker threads: {}", germen_core::sys::worker_threads());

    let device = Device::flex().autodiff();
    let (b, h, t, hd) = (16usize, 6usize, 256usize, 64usize);
    let dist = Distribution::Normal(0.0, 1.0);
    let q = Tensor::<4>::random([b, h, t, hd], dist, &device).require_grad();
    let k = Tensor::<4>::random([b, h, t, hd], dist, &device).require_grad();
    let v = Tensor::<4>::random([b, h, t, hd], dist, &device).require_grad();
    let mut mask = vec![0.0f32; t * t];
    for r in 0..t {
        for c in r + 1..t {
            mask[r * t + c] = NEG;
        }
    }
    let bias = Tensor::<4>::from_data(TensorData::new(mask, [1, 1, t, t]), &device);

    // Numerical agreement (outputs and input gradients) with Burn's own attention.
    let grads_of = |f: &dyn Fn(Tensor<4>, Tensor<4>, Tensor<4>) -> Tensor<4>| {
        let out = f(q.clone(), k.clone(), v.clone());
        let g = out.clone().powf_scalar(2.0).sum().backward();
        (
            out.inner(),
            q.grad(&g).unwrap(),
            k.grad(&g).unwrap(),
            v.grad(&g).unwrap(),
        )
    };
    let (o_ref, gq_ref, gk_ref, gv_ref) = grads_of(&|q, k, v| attn_burn(q, k, v));
    let candidates: Vec<(String, AttentionFn)> = vec![
        ("manual".to_string(), {
            let bias = bias.clone();
            Box::new(move |q, k, v| attn_manual(q, k, v, &bias))
        }),
        ("chunked 128".to_string(), {
            let bias = bias.clone();
            Box::new(move |q, k, v| attn_chunked(q, k, v, &bias, 128))
        }),
        ("chunked 64".to_string(), {
            let bias = bias.clone();
            Box::new(move |q, k, v| attn_chunked(q, k, v, &bias, 64))
        }),
        ("chunked 32".to_string(), {
            let bias = bias.clone();
            Box::new(move |q, k, v| attn_chunked(q, k, v, &bias, 32))
        }),
    ];
    println!("agreement with Burn's attention (max |difference|):");
    for (name, f) in &candidates {
        let (o, gq, gk, gv) = grads_of(f.as_ref());
        println!(
            "  {name:<14} out {:.2e}  dq {:.2e}  dk {:.2e}  dv {:.2e}",
            max_abs_diff(o, o_ref.clone()),
            max_abs_diff(gq, gq_ref.clone()),
            max_abs_diff(gk, gk_ref.clone()),
            max_abs_diff(gv, gv_ref.clone()),
        );
    }

    println!("forward + backward, [16, 6, 256, 64]:");
    let base = time("A  burn attention (fallback)", 3, || {
        let _ = attn_burn(q.clone(), k.clone(), v.clone()).sum().backward();
    });
    for (name, f) in &candidates {
        let ms = time(name, 3, || {
            let _ = f(q.clone(), k.clone(), v.clone()).sum().backward();
        });
        println!("      -> {:.2}x vs A", base / ms);
    }
}
