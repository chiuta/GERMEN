#!/usr/bin/env python3
"""An independent implementation of GERMEN's network, used to check the Rust one (docs/AUDIT.md, round 4).

It reads the file written by `cargo run --release -p germen-core --example reference_dump -- OUT.json`
(weights, a batch, and what GERMEN computed from them) and recomputes everything in float64 with NumPy
only: the logits, the loss and, by central finite differences (no automatic differentiation, so nothing
shared with the Rust side), the gradient of the loss for every parameter.

    python3 scripts/reference_check.py OUT.json [--tolerance 2e-4] [--skip-gradients]

Exit status 0 when every comparison is inside the tolerance. The architecture written here is the one
documented in docs/PLAN.md and nothing else: pre-norm blocks, RMSNorm with a gain, fused q/k/v
projection, rotary embeddings in the "rotate-half" layout, causal attention scaled by 1/sqrt(head_dim),
squared-ReLU feed-forward, untied output projection, mean cross-entropy.
"""
import json, sys, time
import numpy as np


def rmsnorm(x, gamma, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * gamma


def forward(W, cfg, tokens):
    b, t = tokens.shape
    d, h = cfg["d_model"], cfg["n_head"]
    hd = d // h
    half = hd // 2
    freq = cfg["rope_theta"] ** (-(2.0 * np.arange(half)) / hd)
    ang = np.arange(t)[:, None] * freq[None, :]
    cos, sin = np.cos(ang)[None, :, None, :], np.sin(ang)[None, :, None, :]  # [1,t,1,half]

    def rope(z):  # z: [b,t,h,hd]
        z1, z2 = z[..., :half], z[..., half:]
        return np.concatenate([z1 * cos - z2 * sin, z1 * sin + z2 * cos], -1)

    mask = np.triu(np.full((t, t), -np.inf), 1)
    x = W["wte.weight"][tokens]
    for i in range(cfg["n_layer"]):
        p = f"blocks.{i}."
        n = rmsnorm(x, W[p + "norm1.gamma"], cfg["norm_eps"])
        qkv = n @ W[p + "attn.qkv.weight"]
        q, k, v = (qkv[..., j * d:(j + 1) * d].reshape(b, t, h, hd) for j in range(3))
        q, k = rope(q), rope(k)
        s = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(hd) + mask
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        y = np.einsum("bhqk,bkhd->bqhd", a, v).reshape(b, t, d)
        x = x + y @ W[p + "attn.proj.weight"]
        n = rmsnorm(x, W[p + "norm2.gamma"], cfg["norm_eps"])
        u = np.maximum(n @ W[p + "mlp.up.weight"], 0.0)
        x = x + (u * u) @ W[p + "mlp.down.weight"]
    return rmsnorm(x, W["norm_f.gamma"], cfg["norm_eps"]) @ W["head.weight"]


def loss_of(W, cfg, tokens, targets):
    lg = forward(W, cfg, tokens)
    lg = lg - lg.max(-1, keepdims=True)
    lse = np.log(np.exp(lg).sum(-1))
    pick = np.take_along_axis(lg, targets[..., None], -1)[..., 0]
    return float((lse - pick).mean())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tol = 2e-4
    if "--tolerance" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tolerance") + 1])
        args = [a for a in args if a != str(tol) and a != sys.argv[sys.argv.index("--tolerance") + 1]]
    doc = json.load(open(args[0]))
    cfg, b, t = doc["config"], doc["batch"], doc["seq"]
    tokens = np.array(doc["tokens"]).reshape(b, t)
    targets = np.array(doc["targets"]).reshape(b, t)
    W = {x["name"]: np.array(x["data"], dtype=np.float64).reshape(x["shape"]) for x in doc["tensors"]}
    bad = 0

    def report(name, got, want, scale=None):
        nonlocal bad
        got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
        scale = scale if scale is not None else max(np.abs(want).max(), 1e-12)
        err = np.abs(got - want).max()
        ok = err <= tol * scale
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name:<44} max error {err:.3e}  (scale {scale:.3e}, limit {tol * scale:.3e})")

    ref = forward(W, cfg, tokens)
    report("logits, training path (blocked attention)", np.array(doc["logits_train"]).reshape(ref.shape), ref)
    report("logits, evaluation path (fused attention)", np.array(doc["logits_eval"]).reshape(ref.shape), ref)
    cached = np.array(doc["logits_cached_row0"]).reshape(t, -1)
    report("logits, cached path (20 + 17 + 1 x 33 tokens)", cached, ref[0], scale=np.abs(ref).max())
    report("loss", doc["loss"], loss_of(W, cfg, tokens, targets), scale=1.0)

    # Positions after a change: the first rows must not move when only later tokens change.
    t2 = tokens.copy()
    t2[:, 40:] = (t2[:, 40:] + 5) % cfg["vocab_size"]
    same = np.abs(forward(W, cfg, t2)[:, :40] - ref[:, :40]).max()
    print(f"{'ok  ' if same == 0 else 'FAIL'} reference is causal (earlier logits unchanged: {same:.1e})")
    bad += same != 0

    # AdamW (decoupled weight decay, bias-corrected moments, epsilon outside the square root), five
    # steps with the same gradient and a changing learning rate, written out from the paper and
    # from PyTorch's documented update, not from Burn.
    ad = doc["adamw"]
    m = {x["name"]: np.zeros(x["shape"]) for x in doc["tensors"]}
    v = {x["name"]: np.zeros(x["shape"]) for x in doc["tensors"]}
    p = {x["name"]: np.array(x["data"], dtype=np.float64).reshape(x["shape"]) for x in doc["tensors"]}
    for step, lr in enumerate(ad["lrs"], start=1):
        worst = 0.0
        for k, x in enumerate(doc["tensors"]):
            n = x["name"]
            g = np.array(x["grad"], dtype=np.float64).reshape(x["shape"])
            p[n] *= 1.0 - lr * ad["weight_decay"]
            m[n] = ad["beta1"] * m[n] + (1 - ad["beta1"]) * g
            v[n] = ad["beta2"] * v[n] + (1 - ad["beta2"]) * g * g
            mh, vh = m[n] / (1 - ad["beta1"] ** step), v[n] / (1 - ad["beta2"] ** step)
            p[n] -= lr * mh / (np.sqrt(vh) + ad["epsilon"])
            got = np.array(ad["after"][step - 1][k], dtype=np.float64).reshape(x["shape"])
            worst = max(worst, float(np.abs(got - p[n]).max()))
        scale = max(float(np.abs(p[n]).max()) for n in p)
        ok = worst <= 1e-5 * scale + 1e-9
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} AdamW step {step} (lr {lr:g}), every parameter       max error {worst:.3e}")

    if "--skip-gradients" not in sys.argv:
        t0 = time.time()
        eps = 1e-5
        for x in doc["tensors"]:
            name = x["name"]
            flat = W[name].reshape(-1)
            fd = np.empty_like(flat)
            for i in range(flat.size):
                old = flat[i]
                flat[i] = old + eps
                up = loss_of(W, cfg, tokens, targets)
                flat[i] = old - eps
                down = loss_of(W, cfg, tokens, targets)
                flat[i] = old
                fd[i] = (up - down) / (2 * eps)
            report(f"gradient of {name}", np.array(x["grad"]), fd.reshape(-1))
        print(f"(finite differences took {time.time() - t0:.0f} s)")
    print("ALL AGREE" if not bad else f"{int(bad)} DISAGREEMENT(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
