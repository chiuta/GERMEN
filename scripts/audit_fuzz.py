#!/usr/bin/env python3
"""Hostile-input harness for GERMEN (audit, round 2).

Round 1 (`audit_repro.py`) reproduced known suspicions one by one. This script goes the other way: it
throws many odd inputs at the real binary and records HOW it fails, not whether a particular thing
fails. Every run happens in a scratch directory, under an address-space limit (so an allocation bomb
becomes an abort instead of a swap storm) and a timeout.

Classes of outcome, from fine to bad:
  ok       exit 0
  error    exit 1 or 2 with a message (a refusal; fine)
  running  killed by the harness after the time limit, for runs that legitimately go on for long
  PANIC    exit 101 or "panicked at": a bug, whatever the input was
  ABORT    SIGABRT/allocation failure: a bug
  SIGNAL   any other fatal signal: a bug
  HANG     still running after the time limit although it should have finished: a bug

Suites (`--only`):
  ckpt     mutated checkpoints: truncations, bit flips, length-field bombs, wrong tokenizer, files of
           another generation; run through `generate` and through `train --resume`
  json     extreme values in every field of config.json, meta.json and tokenizer.json
  cli      extreme values for the options of generate, train, bench and demo-corpus
  data     odd training texts (empty, tiny, NUL, invalid UTF-8, one huge line, directories, devices)
  disk     a save that cannot be completed (file-size limit) must leave the old checkpoint intact

    python3 scripts/audit_fuzz.py --bin target/release/germen [--only ckpt,json] [--json out.json]
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

AS_LIMIT = 8 << 30
TIMEOUT = 90
DEMO = ["--demo", "--demo-mb", "0.2", "--preset", "nano", "--seq", "64", "--batch", "4",
        "--eval-batches", "1", "--threads", "1"]


def run(binary, args, timeout=TIMEOUT, fsize=None, stdin=None, cwd=None):
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
        if fsize is not None:
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        os.setsid()

    t0 = time.time()
    try:
        # Never the caller's directory: some cases pass `--out .` on purpose.
        p = subprocess.run([str(binary)] + [str(a) for a in args], capture_output=True,
                           timeout=timeout, preexec_fn=limits, stdin=stdin,
                           cwd=str(cwd) if cwd else tempfile.gettempdir())
    except subprocess.TimeoutExpired as e:
        return dict(rc=None, secs=time.time() - t0, out=(e.stdout or b"")[:200].decode("utf-8", "replace"),
                    err=(e.stderr or b"")[:300].decode("utf-8", "replace"))
    return dict(rc=p.returncode, secs=time.time() - t0, out=p.stdout.decode("utf-8", "replace"),
                err=p.stderr.decode("utf-8", "replace"))


def classify(r, may_run_long=False):
    if r["rc"] is None:
        return "running" if may_run_long else "HANG"
    if r["rc"] == 0:
        return "ok"
    if r["rc"] < 0:
        return "ABORT" if -r["rc"] == signal.SIGABRT else "SIGNAL"
    if "panicked at" in r["err"] or r["rc"] == 101:
        return "PANIC"
    if "memory allocation of" in r["err"]:
        return "ABORT"
    return "error"


class Case:
    def __init__(self, suite, name, cls, r, extra=None):
        self.suite, self.name, self.cls = suite, name, cls
        self.rc, self.secs = r["rc"], round(r["secs"], 2)
        self.err = " ".join(r["err"].split())[:240]
        self.extra = extra or {}

    def dict(self):
        return dict(suite=self.suite, name=self.name, cls=self.cls, rc=self.rc, secs=self.secs,
                    err=self.err, **self.extra)


# ------------------------------------------------------------------------------------ checkpoints

BASE_BIN = None  # a binary that builds the base checkpoints, when it is not the one under test


def make_base(binary, root, name, extra):
    out = root / name
    r = run(BASE_BIN or binary, ["train", *DEMO, "--steps", "6", "--save-every", "3", "--eval-every", "3",
                                 "--log-every", "3", "--out", out, *extra])
    if r["rc"] != 0:
        sys.exit(f"cannot build the base checkpoint {name}: {r['err'][:300]}")
    return out


def gen_cmd(d):
    return ["generate", "--model", d, "--max-new", "24", "--temperature", "0", "--seed", "1",
            "--threads", "1"]


BPE_ARGS = ["--tokenizer", "bpe", "--vocab", "400"]


def resume_cmd(d, bpe=False, steps="8"):
    return ["train", *DEMO, "--steps", steps, "--eval-every", "4", "--log-every", "4", "--out", d,
            "--resume", *(BPE_ARGS if bpe else [])]


def burnpack_layout(path):
    b = path.read_bytes()
    meta_len = struct.unpack_from("<I", b, 6)[0] if b[:4] == b"NRUB" else 0
    return len(b), 10, 10 + meta_len


def mutations(path, rng):
    """(name, function(bytes) -> bytes or None for 'delete') for a binary file."""
    size, head_end, data_start = burnpack_layout(path)
    out = [("delete", lambda b: None), ("empty", lambda b: b""),
           ("trunc_half", lambda b: b[: len(b) // 2]), ("trunc_last_byte", lambda b: b[:-1]),
           ("trunc_8", lambda b: b[:8]), ("trunc_to_header", lambda b: b[:data_start]),
           ("trunc_in_header", lambda b: b[: data_start - 7]),
           ("append_4k", lambda b: b + bytes(rng.randrange(256) for _ in range(4096))),
           ("zero_first_32", lambda b: bytes(32) + b[32:])]
    for off in (0, 4, 6, 7, 8, 9):
        out.append((f"ff_at_{off}", lambda b, o=off: b[:o] + b"\xff" * min(4, len(b) - o) + b[o + 4:]))
    out.append(("metalen_huge", lambda b: b[:6] + b"\xff\xff\xff\xff" + b[10:]))
    out.append(("metalen_zero", lambda b: b[:6] + bytes(4) + b[10:]))
    out.append(("metalen_plus1", lambda b: b[:6] + struct.pack("<I", data_start - 10 + 1) + b[10:]))
    out.append(("metalen_minus1", lambda b: b[:6] + struct.pack("<I", data_start - 10 - 1) + b[10:]))
    for i in range(24):
        pos = rng.randrange(head_end, data_start)
        bit = 1 << rng.randrange(8)
        out.append((f"flip_meta_{i}", lambda b, p=pos, m=bit: b[:p] + bytes([b[p] ^ m]) + b[p + 1:]))
    for i in range(24):
        pos = rng.randrange(data_start, size)
        bit = 1 << rng.randrange(8)
        out.append((f"flip_data_{i}", lambda b, p=pos, m=bit: b[:p] + bytes([b[p] ^ m]) + b[p + 1:]))
    return out


def ckpt_suite(binary, work, rng, workers):
    cases = []
    byte = make_base(binary, work, "byte", [])
    byte2 = make_base(binary, work, "byte_other", ["--seed", "7"])
    bpe = make_base(binary, work, "bpe", BPE_ARGS)
    micro = work / "micro"
    r = run(BASE_BIN or binary, ["train", "--demo", "--demo-mb", "0.2", "--preset", "micro", "--seq", "64",
                                 "--batch", "4", "--steps", "2", "--eval-every", "0", "--out", micro,
                                 "--threads", "1"])
    assert r["rc"] == 0, r["err"][:200]
    base_out = {d.name: run(binary, gen_cmd(d))["out"] for d in (byte, bpe)}


    jobs = []  # (label, dir to copy, mutate function(dir), command, base name for output comparison)

    def add(label, src, mutate, cmd_kind):
        jobs.append((label, src, mutate, cmd_kind))

    for src in (byte, bpe):  # the controls: an untouched checkpoint must work
        for k in ("generate", "resume"):
            add(f"control/{src.name}/{k}", src, lambda d: None, k)

    def file_mut(fname, fn):
        def apply(d):
            p = d / fname
            data = p.read_bytes()
            new = fn(data)
            if new is None:
                p.unlink()
            else:
                p.write_bytes(new)
        return apply

    for src in (byte, bpe):
        for fname in ("weights.bpk", "optimizer.bpk"):
            for mname, fn in mutations(src / fname, rng):
                kinds = ("generate", "resume") if fname == "weights.bpk" else ("resume",)
                for k in kinds:
                    add(f"{src.name}/{fname}/{mname}/{k}", src, file_mut(fname, fn), k)
    for fname in ("config.json", "meta.json", "tokenizer.json"):
        for mname, fn in [("delete", lambda b: None), ("empty", lambda b: b""),
                          ("trunc_half", lambda b: b[: len(b) // 2]), ("garbage", lambda b: b"\x00\xff{[" * 9),
                          ("append_junk", lambda b: b + b"\nxx"), ("utf8_bom", lambda b: b"\xef\xbb\xbf" + b),
                          ("not_utf8", lambda b: b[:3] + b"\xff\xfe" + b[3:])]:
            for k in ("generate", "resume"):
                add(f"byte/{fname}/{mname}/{k}", byte, file_mut(fname, fn), k)

    def swap(label, src, fname, donor, cmd_kind):
        def apply(d):
            shutil.copy(donor / fname, d / fname)
        add(label, src, apply, cmd_kind)

    # files of another generation or of another model
    for fname in ("weights.bpk", "optimizer.bpk", "meta.json"):
        for k in ("generate", "resume"):
            swap(f"mix/byte+{fname}-of-another-run/{k}", byte, fname, byte2, k)
    for k in ("generate", "resume"):
        swap(f"mix/byte+weights-of-micro/{k}", byte, "weights.bpk", micro, k)
        swap(f"mix/byte+optimizer-of-micro/{k}", byte, "optimizer.bpk", micro, k)
        swap(f"mix/byte+tokenizer-of-bpe/{k}", byte, "tokenizer.json", bpe, k)
        swap(f"mix/bpe+tokenizer-of-byte/{k}", bpe, "tokenizer.json", byte, k)
        swap(f"mix/bpe+weights-of-byte/{k}", bpe, "weights.bpk", byte, k)
        swap(f"mix/byte+config-of-micro/{k}", byte, "config.json", micro, k)
        add(f"mix/bpe-without-tokenizer/{k}", bpe, file_mut("tokenizer.json", lambda b: None), k)

    def one(job):
        label, src, mutate, kind = job
        d = Path(tempfile.mkdtemp(dir=work, prefix="m-")) / "m"
        shutil.copytree(src, d)
        try:
            mutate(d)
            r = run(binary, gen_cmd(d) if kind == "generate" else resume_cmd(d, src.name == "bpe"))
            cls = classify(r)
            if label.startswith("control/") and cls != "ok":
                cls = "BAD"  # the untouched checkpoint was refused: that is a bug of its own
            extra = {}
            if cls == "ok" and kind == "generate":
                extra["output"] = "same" if r["out"] == base_out[src.name] else "DIFFERENT"
            return Case("ckpt", label, cls, r, extra)
        finally:
            shutil.rmtree(d.parent, ignore_errors=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        cases = list(ex.map(one, jobs))
    return cases


# --------------------------------------------------------------------------------------- json fields

EXTREMES = [0, -1, 1, 2, 3, 2**31, 2**32, 2**53, 2**63, 2**64, 10**30, 1e308, -1e308, 1e-300,
            "x", "", None, [], {}, True]


def json_suite(binary, work, rng, workers):
    byte = make_base(binary, work, "jbyte", [])
    bpe = make_base(binary, work, "jbpe", BPE_ARGS)
    jobs = []
    for src in (byte,):
        for fname in ("config.json", "meta.json"):
            doc = json.loads((src / fname).read_text())
            for key in doc:
                for v in EXTREMES:
                    jobs.append((f"{src.name}/{fname}:{key}={json.dumps(v)}", src, fname, {**doc, key: v}))
                missing = {k: x for k, x in doc.items() if k != key}
                jobs.append((f"{src.name}/{fname}:{key} missing", src, fname, missing))
            jobs.append((f"{src.name}/{fname}: unknown field", src, fname, {**doc, "zzz": 1}))
    tok = json.loads((bpe / "tokenizer.json").read_text())
    merges = tok["merges"]
    tv = {"merges missing": {k: v for k, v in tok.items() if k != "merges"},
          "kind=foo": {**tok, "kind": "foo"}, "kind=byte with merges": {**tok, "kind": "byte"},
          "vocab_size=0": {**tok, "vocab_size": 0}, "vocab_size=2^40": {**tok, "vocab_size": 2**40},
          "vocab_size=257": {**tok, "vocab_size": 257}, "vocab_size=65537": {**tok, "vocab_size": 65537},
          "vocab_size=-1": {**tok, "vocab_size": -1},
          "merges=[]": {**tok, "merges": []}, "merges=[[0,0]]": {**tok, "merges": [[0, 0]]},
          "merge out of range": {**tok, "merges": [[0, 10**9]] + merges},
          "merge negative": {**tok, "merges": [[-1, 0]] + merges}, "merge pair of 3": {**tok, "merges": [[1, 2, 3]] + merges},
          "merge strings": {**tok, "merges": [["a", "b"]] + merges},
          "merges 10x duplicate": {**tok, "merges": merges + merges[:10] * 10},
          "merge refers forward": {**tok, "merges": [[256, 257]] + merges},
          "merges null": {**tok, "merges": None}, "merges huge count": {**tok, "merges": [[97, 98]] * 100000}}
    for name, doc in tv.items():
        jobs.append((f"bpe/tokenizer.json:{name}", bpe, "tokenizer.json", doc))

    def one(job):
        label, src, fname, doc = job
        d = Path(tempfile.mkdtemp(dir=work, prefix="j-")) / "m"
        shutil.copytree(src, d)
        try:
            (d / fname).write_text(json.dumps(doc))
            kind = "resume" if fname == "meta.json" else "generate"
            if src.name == "jbpe":
                r = run(binary, gen_cmd(d))
            else:
                r = run(binary, gen_cmd(d) if kind == "generate" else resume_cmd(d))
            return Case("json", label, classify(r), r)
        finally:
            shutil.rmtree(d.parent, ignore_errors=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


# ------------------------------------------------------------------------------------------- the CLI

NUMS = ["0", "1", "2", "3", "-1", "255", "256", "257", "65535", "65536", "65537", str(2**31),
        str(2**32), str(2**53), str(2**63 - 1), str(2**63), str(2**64 - 1), str(2**64), "1e9", "1e30",
        "1e308", "1e400", "-1e30", "1e-45", "1e-300", "nan", "inf", "-inf", "0.5", "1.5", "abc", "",
        "0x10", "1_000", " 7", "٣"]


def cli_suite(binary, work, rng, workers):
    byte = make_base(binary, work, "cbyte", [])
    base_train = ["train", "--demo", "--demo-mb", "0.05", "--preset", "nano", "--seq", "32", "--batch",
                  "2", "--steps", "2", "--eval-every", "0", "--save-every", "0", "--log-every", "1",
                  "--threads", "1"]
    long_ok = {"--steps", "--eval-every", "--save-every", "--log-every", "--sample-every", "--warmup",
               "--eval-batches", "--batch", "--seq", "--accum", "--demo-mb", "--bpe-sample-mb", "--vocab"}

    def train_with(opt, val):
        args = list(base_train)
        if opt in args:
            args[args.index(opt) + 1] = val
        else:
            args += [opt, val]
        return args

    train_opts = ["--seq", "--batch", "--accum", "--steps", "--replicas", "--lr", "--min-lr-ratio",
                  "--warmup", "--weight-decay", "--grad-clip", "--eval-every", "--eval-batches",
                  "--save-every", "--log-every", "--sample-every", "--seed", "--val-fraction",
                  "--demo-mb", "--threads", "--vocab", "--bpe-sample-mb"]
    gen_opts = ["--max-new", "--temperature", "--top-k", "--window", "--seed", "--threads"]
    bench_base = ["bench", "--preset", "nano", "--batch", "2", "--seq", "32", "--steps", "1",
                  "--warmup", "0", "--threads", "1"]
    bench_opts = ["--batch", "--seq", "--steps", "--warmup", "--vocab", "--replicas", "--threads"]

    jobs = []
    for opt in train_opts:
        for v in NUMS:
            args = train_with(opt, v)
            if opt in ("--vocab", "--bpe-sample-mb"):
                args = args + ["--tokenizer", "bpe"]
            jobs.append((f"train {opt} {v!r}", args, opt in long_ok or opt == "--steps"))
    for opt, vals in (("--preset", ["", "xxl", "NANO", "all", "nano,micro"]),
                      ("--tokenizer", ["", "foo", "BPE", "byte,bpe"]),
                      ("--out", ["", "/", "/dev/null", "/proc/x", "/nonexistent/x/y", "."]),
                      ("--data", ["", "/nonexistent", "/", "/dev/null", "/dev/zero", "/etc/shadow"])):
        for v in vals:
            args = [a for a in base_train if a != "--demo"]
            i = args.index("--demo-mb")
            del args[i:i + 2]
            if opt == "--data":
                jobs.append((f"train --data {v!r}", args + ["--data", v], False))
            else:
                a2 = list(base_train)
                a2 += [opt, v] if opt not in a2 else []
                jobs.append((f"train {opt} {v!r}", a2, False))
    for opt in gen_opts:
        for v in NUMS:
            args = ["generate", "--model", str(byte), "--max-new", "8", "--seed", "1", "--threads", "1"]
            if opt in args:
                args[args.index(opt) + 1] = v
            else:
                args += [opt, v]
            jobs.append((f"generate {opt} {v!r}", args, opt == "--max-new"))
    for prompt in ["", "a", "é", "😀", "‮", "x" * 100_000, "é" * 50_000, "\n" * 10_000, "\t", " ",
                   "﻿", "\U0010ffff", "퟿"]:
        jobs.append((f"generate --prompt len={len(prompt)} {prompt[:6]!r}",
                     ["generate", "--model", str(byte), "--prompt", prompt, "--max-new", "8", "--seed", "1",
                      "--threads", "1"], False))
    for opt in bench_opts:
        for v in NUMS:
            args = list(bench_base)
            if opt in args:
                args[args.index(opt) + 1] = v
            else:
                args += [opt, v]
            # A huge count of steps, warm-up steps or windows is a long job, not a hang.
            jobs.append((f"bench {opt} {v!r}", args, opt != "--threads"))
    for v in ["", "0", "1", "1,1", "2,1", "999999", "a,b", "1,", ",1", "-1", "1 2", "1;2", "1,1,1,1,1,1"]:
        jobs.append((f"bench --scaling {v!r}", bench_base + ["--scaling", v], False))
    for target in ["/nonexistent/dir/x.json", "/dev/full", "/", "/proc/version", ""]:
        jobs.append((f"bench --json {target!r}", bench_base + ["--json", target], False))
    for v in NUMS:
        jobs.append((f"demo-corpus --size-mb {v!r}", ["demo-corpus", "--size-mb", v, "--out", "{tmp}/c.txt"], False))
    for target in ["/dev/full", "/nonexistent/x/c.txt", "/", "", "/proc/version"]:
        jobs.append((f"demo-corpus --out {target!r}", ["demo-corpus", "--size-mb", "0.1", "--out", target], False))
    for v in ["", "0", "1", "1024", "1025", "99999", "-1", "x"]:
        jobs.append((f"info --threads {v!r}", ["--threads", v, "info"], False))
    jobs.append(("info extra arg", ["info", "extra"], False))
    jobs.append(("no command", [], False))
    jobs.append(("--version", ["--version"], False))

    # random pairs of extremes on train, to catch interactions the single factors miss
    for i in range(120):
        opts = rng.sample(train_opts, 2)
        args = list(base_train)
        for o in opts:
            v = rng.choice(NUMS)
            args = train_with_args(args, o, v)
        jobs.append((f"train pair #{i}: {' '.join(args[len(base_train):]) or '(base)'}", args, True))

    def one(job):
        label, args, long_ok_ = job
        d = Path(tempfile.mkdtemp(dir=work, prefix="c-"))
        try:
            a = [str(x).replace("{tmp}", str(d)) for x in args]
            if a and a[0] == "train" and "--out" not in a:
                a += ["--out", str(d / "out")]
            r = run(binary, a, timeout=15 if long_ok_ else TIMEOUT, cwd=d)
            return Case("cli", label, classify(r, may_run_long=long_ok_), r)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


def train_with_args(args, opt, val):
    args = list(args)
    if opt in args:
        args[args.index(opt) + 1] = val
    else:
        args += [opt, val]
    return args


# ------------------------------------------------------------------------------------------- texts

def data_suite(binary, work, rng, workers):
    big = 5 << 20
    files = {
        "empty": b"", "1 byte": b"a", "2 bytes": b"ab", "seq-1 bytes (31)": b"x" * 31,
        "seq bytes (32)": b"x" * 32, "seq+1 bytes (33)": b"x" * 33, "100 bytes": bytes(range(100)),
        "1 MB of NUL": bytes(1 << 20), "1 MB of 0xFF": b"\xff" * (1 << 20),
        "1 MB random bytes": bytes(rng.randrange(256) for _ in range(1 << 20)),
        "BOM + text": b"\xef\xbb\xbf" + b"Salut lume. " * 5000, "CRLF text": b"Salut lume.\r\n" * 5000,
        "overlong C0 80": b"\xc0\x80" * 40000, "lone surrogate ED A0 80": b"\xed\xa0\x80" * 30000,
        "1e6 newlines": b"\n" * 1_000_000, "5 MB one line of 'a'": b"a" * big,
        "5 MB one line, random letters": bytes(rng.choice(b"abcdefghij") for _ in range(big)),
        "5 MB of 'ab' repeated": b"ab" * (big // 2),
        "5 MB of the 256 byte values in order": bytes(range(256)) * (big // 256),
        "200000 distinct words": " ".join(f"w{i}" for i in range(200000)).encode(),
    }
    jobs = []
    for name, content in files.items():
        for tk in ("byte", "bpe"):
            for vf in ("0.05", "0", "0.5"):
                if vf != "0.05" and len(content) > (1 << 20) and tk == "bpe":
                    continue
                jobs.append((name, content, tk, vf))
    for special in ("/dev/zero", "dir"):
        for tk in ("byte",):
            jobs.append((special, None, tk, "0.05"))

    def one(job):
        name, content, tk, vf = job
        d = Path(tempfile.mkdtemp(dir=work, prefix="d-"))
        try:
            if content is None:
                path = {"/dev/zero": "/dev/zero", "dir": str(d)}[name]
            else:
                path = d / "t.txt"
                path.write_bytes(content)
            args = ["train", "--data", path, "--preset", "nano", "--seq", "32", "--batch", "2", "--steps", "2",
                    "--eval-every", "0", "--save-every", "0", "--log-every", "1", "--threads", "1",
                    "--out", d / "out", "--tokenizer", tk, "--val-fraction", vf]
            if tk == "bpe":
                args += ["--vocab", "300"]
            stdin = subprocess.DEVNULL
            r = run(binary, args, timeout=60, stdin=stdin)
            c = classify(r, may_run_long=name == "/dev/zero")
            return Case("data", f"{name} [{tk}, val {vf}]", c, r)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, jobs))


# --------------------------------------------------------------------------------------------- disk

def disk_suite(binary, work, rng, workers):
    cases = []
    for tk, extra in (("byte", []), ("bpe", BPE_ARGS)):
        base = make_base(binary, work, f"disk-{tk}", extra)
        want = json.loads((base / "meta.json").read_text())["step"]
        before = run(binary, gen_cmd(base))["out"]
        for limit in (1 << 10, 1 << 16, 1 << 20, 2 << 20, 3 << 20, 4 << 20):
            d = Path(tempfile.mkdtemp(dir=work, prefix="s-")) / "m"
            shutil.copytree(base, d)
            r = run(binary, ["train", *DEMO, "--steps", "12", "--save-every", "4", "--eval-every", "0",
                             "--log-every", "4", "--out", d, "--resume", *extra], fsize=limit)
            cls = classify(r)
            after = json.loads((d / "meta.json").read_text())["step"] if (d / "meta.json").exists() else None
            gen = run(binary, gen_cmd(d))
            leftovers = sorted(p.name for p in d.iterdir() if p.name.endswith(".tmp"))
            torn = after == want and gen["rc"] == 0 and gen["out"] != before
            extra_info = {"limit": limit, "step_before": want, "step_after": after,
                          "generate_ok": gen["rc"] == 0, "tmp_left": leftovers, "torn": torn}
            bad = cls in ("PANIC", "ABORT", "SIGNAL", "HANG") or gen["rc"] != 0 or leftovers or torn
            cases.append(Case("disk", f"{tk} file-size limit {limit}", "BAD" if bad else cls, r, extra_info))
            shutil.rmtree(d.parent, ignore_errors=True)
    return cases


# --------------------------------------------------------------------------------------------- main

SUITES = {"ckpt": ckpt_suite, "json": json_suite, "cli": cli_suite, "data": data_suite, "disk": disk_suite}
BAD = {"PANIC", "ABORT", "SIGNAL", "HANG", "BAD"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--only", default=",".join(SUITES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--base-bin", help="build the base checkpoints with this binary (e.g. an older "
                                       "version), and run the binary under test on them")
    ap.add_argument("--json", help="write every case to this file")
    ns = ap.parse_args()
    global BASE_BIN
    binary = Path(ns.bin).resolve()
    if not binary.exists():
        sys.exit(f"{binary} not found")
    if ns.base_bin:
        BASE_BIN = Path(ns.base_bin).resolve()
    work = Path(tempfile.mkdtemp(prefix="germen-fuzz-"))
    rng = random.Random(ns.seed)
    allcases = []
    try:
        for name in ns.only.split(","):
            t0 = time.time()
            cases = SUITES[name](binary, work, rng, ns.workers)
            allcases += cases
            c = Counter(x.cls for x in cases)
            print(f"\n== {name}: {len(cases)} runs in {time.time() - t0:.0f} s: "
                  + ", ".join(f"{k} {v}" for k, v in sorted(c.items())), flush=True)
            bad = [x for x in cases if x.cls in BAD]
            for x in bad[:60]:
                print(f"  {x.cls:6} {x.name}  [rc={x.rc}, {x.secs}s] {x.err[:200]}")
            if len(bad) > 60:
                print(f"  ... and {len(bad) - 60} more")
            if name == "ckpt":
                print_acceptance(cases)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if ns.json:
        Path(ns.json).write_text(json.dumps([c.dict() for c in allcases], indent=1))
    nbad = sum(1 for c in allcases if c.cls in BAD)
    print(f"\n{len(allcases)} runs, {nbad} bad")
    return 1 if nbad else 0


def print_acceptance(cases):
    """Which damaged checkpoints were accepted without a word?"""
    groups = defaultdict(Counter)
    for c in cases:
        if c.cls != "ok":
            continue
        parts = c.name.split("/")
        if parts[0] == "control":
            continue
        kind = parts[-1]
        what = parts[0] if parts[0] == "mix" else parts[1]
        mut = parts[2] if parts[0] != "mix" else parts[1]
        what2 = "flip_data" if mut.startswith("flip_data") else "flip_meta" if mut.startswith("flip_meta") else mut
        key = (what, what2, kind)
        groups[key]["accepted" + ("-DIFFERENT" if c.extra.get("output") == "DIFFERENT" else "")] += 1
    print("\n  damaged or mixed checkpoints that were ACCEPTED (exit 0):")
    for (what, mut, kind), c in sorted(groups.items()):
        print(f"    {what:24} {mut:34} {kind:9} {dict(c)}")


if __name__ == "__main__":
    sys.exit(main())
