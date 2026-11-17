#!/usr/bin/env python3
"""Black-box reproductions of the findings in docs/AUDIT.md.

Every scenario feeds one hostile input or failure to a `germen` binary and states the SAFE
behaviour; a binary passes when it shows it. Run it against the current build, and against an
older one to see which of the findings that build still has:

    python3 scripts/audit_repro.py --new target/release/germen
    python3 scripts/audit_repro.py --new target/release/germen --old /path/to/germen-0.2.0
    python3 scripts/audit_repro.py --new target/release/germen A-06 A-07     # only some scenarios

POSIX only (it uses process groups, resource limits and signals); on Windows, run it in WSL.
The exit status is 1 if the --new binary fails a scenario that is not purely informational.
"""
import argparse, json, os, random, resource, shutil, signal, struct, subprocess, sys, tempfile, time

TINY = ["--demo", "--demo-mb", "0.2", "--preset", "nano", "--seq", "64", "--batch", "8",
        "--steps", "30", "--eval-every", "15", "--eval-batches", "2", "--log-every", "10"]


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, help="the germen binary to check")
    p.add_argument("--old", help="an older germen binary to compare with (optional)")
    p.add_argument("--work", help="scratch directory (default: a temporary one)")
    p.add_argument("--json", help="write the results to this file")
    p.add_argument("only", nargs="*", help="run only the scenarios whose names start with these (e.g. A-06)")
    return p.parse_args()


ARGS = parse_arguments()
ROOT = ARGS.work or tempfile.mkdtemp(prefix="germen-audit-")


def limits(mem_mb):
    def f():
        os.setsid()
        if mem_mb:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
    return f


def run(binary, args, timeout=300, mem_mb=None):
    t0 = time.time()
    p = subprocess.Popen([binary] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         preexec_fn=limits(mem_mb))
    try:
        out, _ = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        rc = "timeout"
    return rc, out.decode("utf-8", "replace"), time.time() - t0


def clean_refusal(rc, out):
    """Exit status 1 or 2 (an error the program reported itself), not a panic, abort or kill."""
    return rc in (1, 2) and "panicked" not in out and "memory allocation" not in out


def scratch(name, tag):
    d = os.path.join(ROOT, tag, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.dirname(d), exist_ok=True)
    return d


def train_tiny(binary, d, extra=()):
    rc, out, _ = run(binary, ["train", *TINY, "--out", d, *extra])
    assert rc == 0, out[-500:]


def nonfinite_words(path, skip=8192):
    """Non-finite 32-bit words in the tensor data (the first 8 KiB are the burnpack header)."""
    data = open(path, "rb").read()[skip:]
    n = len(data) // 4
    words = struct.unpack("<%dI" % n, data[: n * 4])
    return sum(1 for w in words if (w & 0x7F800000) == 0x7F800000), n


results = []


def scenario(name, severity):
    def deco(fn):
        if ARGS.only and not any(name.startswith(p) for p in ARGS.only):
            return fn
        row = {"name": name, "severity": severity}
        for tag, binary in (("old", ARGS.old), ("new", ARGS.new)):
            if not binary:
                continue
            try:
                ok, detail = fn(binary, tag)
            except Exception as e:  # a harness error is reported, never hidden
                ok, detail = None, "harness error: %r" % (e,)
            row[tag] = (ok, detail)
        results.append(row)
        print("%-52s %s" % (name, "  ".join("%s: %-5s" % (t, row[t][0]) for t in ("old", "new") if t in row)), flush=True)
        for tag in ("old", "new"):
            if tag in row:
                print("      %s: %s" % (tag, row[tag][1].replace("\n", " ")[:230]), flush=True)
        return fn
    return deco


@scenario("A-01 hostile tokenizer.json (doubling merges)", "High")
def _(binary, tag):
    d = scratch("tok", tag)
    train_tiny(binary, d, ["--tokenizer", "bpe", "--vocab", "400"])
    merges = [[97, 97]] + [[256 + k, 256 + k] for k in range(143)]
    json.dump({"kind": "bpe", "vocab_size": 400, "merges": merges}, open(d + "/tokenizer.json", "w"))
    rc, out, dt = run(binary, ["generate", "--model", d, "--max-new", "5", "--seed", "1"], timeout=60, mem_mb=3000)
    return clean_refusal(rc, out), "exit %s after %.1fs: %s" % (rc, dt, out.strip().splitlines()[-1][:120] if out.strip() else "")


@scenario("A-02 hostile config.json (absurd dimensions)", "High")
def _(binary, tag):
    d = scratch("cfg", tag)
    train_tiny(binary, d)
    cfg = json.load(open(d + "/config.json"))
    for k in ("d_model", "d_ff"):
        if k in cfg:
            cfg[k] = 1 << 40
    json.dump(cfg, open(d + "/config.json", "w"))
    rc, out, dt = run(binary, ["generate", "--model", d, "--max-new", "5", "--seed", "1"], timeout=60, mem_mb=3000)
    return clean_refusal(rc, out), "exit %s after %.1fs: %s" % (rc, dt, out.strip().splitlines()[-1][:120] if out.strip() else "")


@scenario("A-03 NaN weights in a checkpoint are accepted", "Medium")
def _(binary, tag):
    d = scratch("poison", tag)
    train_tiny(binary, d)
    data = bytearray(open(d + "/weights.bpk", "rb").read())
    data[20000:20004] = struct.pack("<I", 0x7FC00000)
    open(d + "/weights.bpk", "wb").write(bytes(data))
    rc, out, _ = run(binary, ["generate", "--model", d, "--max-new", "20", "--seed", "1"], timeout=60)
    return clean_refusal(rc, out), "exit %s: %s" % (rc, out.strip().splitlines()[-1][:110] if out.strip() else "")


@scenario("A-04 a failed save leaves a mixed checkpoint", "Medium")
def _(binary, tag):
    import hashlib
    d = scratch("mixed", tag)
    rc, out, _ = run(binary, ["train", *TINY[:TINY.index("--steps")], "--steps", "10", *TINY[TINY.index("--steps") + 2:],
                              "--save-every", "10", "--out", d])
    assert rc == 0, out[-300:]
    files = ["weights.bpk", "optimizer.bpk", "meta.json", "config.json", "tokenizer.json"]
    h = lambda: {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest() for f in files}
    before = h()
    def lim():
        os.setsid()
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (5 << 20, 5 << 20))   # weights (3.4 MB) fit, optimizer (6.8 MB) does not
    p = subprocess.Popen([binary, "train", *TINY[:TINY.index("--steps")], "--steps", "20", *TINY[TINY.index("--steps") + 2:],
                          "--save-every", "20", "--out", d, "--resume"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=lim)
    out, _ = p.communicate(timeout=120)
    after = h()
    changed = [f for f in files if before[f] != after[f]]
    consistent = len(changed) in (0, len(files))
    return consistent, "save failed (exit %s); files changed: %s" % (p.returncode, ", ".join(changed) or "none")


@scenario("A-04n SIGKILL x25 during back-to-back saves", "Info")
def _(binary, tag):
    bad, trials = 0, 25
    rnd = random.Random(12345)
    for i in range(trials):
        d = scratch("kill%d" % i, tag)
        p = subprocess.Popen([binary, "train", "--demo", "--demo-mb", "0.1", "--preset", "nano", "--seq", "16",
                              "--batch", "1", "--steps", "100000", "--save-every", "1", "--eval-every", "100000",
                              "--eval-batches", "1", "--log-every", "100000", "--out", d],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=limits(None))
        t0 = time.time()
        while not os.path.exists(d + "/meta.json") and time.time() - t0 < 30:
            time.sleep(0.01)
        time.sleep(rnd.uniform(0.0, 0.4))
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        rc, out, _ = run(binary, ["generate", "--model", d, "--max-new", "3", "--seed", "1"], timeout=60)
        if rc != 0:
            bad += 1
        shutil.rmtree(d, ignore_errors=True)
    return bad == 0, "%d of %d killed runs left a checkpoint that does not load (negative result for both)" % (bad, trials)


@scenario("A-05 a diverged model is written over the checkpoint", "High")
def _(binary, tag):
    d = scratch("nan", tag)
    rc, out, _ = run(binary, ["train", "--demo", "--demo-mb", "0.1", "--preset", "nano", "--seq", "32", "--batch", "4",
                              "--steps", "5", "--lr", "1e39", "--grad-clip", "0", "--warmup", "0", "--save-every", "1",
                              "--eval-every", "100000", "--eval-batches", "1", "--log-every", "1", "--out", d], timeout=120)
    w = d + "/weights.bpk"
    if os.path.exists(w):
        bad, total = nonfinite_words(w)
        return bad == 0, "train exit %s; %d of %d weight words on disk are NaN/inf" % (rc, bad, total)
    return True, "train exit %s; no weights written: %s" % (rc, out.strip().splitlines()[-1][:110])


@scenario("A-06 two runs in one output directory", "Medium")
def _(binary, tag):
    d = scratch("two", tag)
    args = ["train", "--demo", "--demo-mb", "0.1", "--preset", "nano", "--seq", "32", "--batch", "4", "--steps", "100000",
            "--save-every", "1", "--eval-every", "100000", "--eval-batches", "1", "--log-every", "100000", "--out", d]
    first = subprocess.Popen([binary, *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=limits(None))
    try:
        t0 = time.time()
        while not os.path.exists(d + "/meta.json") and time.time() - t0 < 30:
            time.sleep(0.05)
        rc, out, dt = run(binary, args + ["--resume"], timeout=8)     # a second `--resume` of the same directory
    finally:
        os.killpg(first.pid, signal.SIGKILL)
        first.wait()
    return rc in (1, 2), "second run: exit %s after %.1fs" % (rc, dt)


@scenario("A-07 --grad-clip nan", "Medium")
def _(binary, tag):
    d = scratch("clipnan", tag)
    rc, out, _ = run(binary, ["train", *TINY, "--grad-clip", "nan", "--out", d], timeout=120)
    return clean_refusal(rc, out), "exit %s: %s" % (rc, out.strip().splitlines()[-1][:120] if out.strip() else "")


@scenario("A-08 --resume with fewer steps than done", "Low")
def _(binary, tag):
    d = scratch("fewer", tag)
    train_tiny(binary, d)
    before = {f: open(os.path.join(d, f), "rb").read() for f in ("meta.json", "weights.bpk")}
    rc, out, _ = run(binary, ["train", *TINY[:TINY.index("--steps")], "--steps", "10", *TINY[TINY.index("--steps") + 2:],
                              "--out", d, "--resume"])
    after = {f: open(os.path.join(d, f), "rb").read() for f in ("meta.json", "weights.bpk")}
    same = before == after
    return (rc in (1, 2)) and same, "exit %s; checkpoint files %s" % (rc, "untouched" if same else "REWRITTEN (%s)" % ", ".join(f for f in before if before[f] != after[f]))


@scenario("A-09 meta.corpus_bytes counts bytes (BPE)", "Low")
def _(binary, tag):
    d = scratch("meta", tag)
    train_tiny(binary, d, ["--tokenizer", "bpe", "--vocab", "400"])
    n = json.load(open(d + "/meta.json")).get("corpus_bytes")
    return n is not None and n >= 150_000, "corpus_bytes = %s (the demo text is about 200,000 bytes)" % n


@scenario("A-10 --val-fraction out of range (byte and BPE)", "Low")
def _(binary, tag):
    notes, ok = [], True
    for tok in (["--tokenizer", "byte"], ["--tokenizer", "bpe", "--vocab", "300"]):
        for v in ("2", "nan"):
            d = scratch("val%s%s" % (tok[1], v), tag)
            rc, out, _ = run(binary, ["train", *TINY, *tok, "--val-fraction=" + v, "--out", d], timeout=120)
            good = clean_refusal(rc, out)
            ok &= good
            notes.append("%s/%s→exit %s%s" % (tok[1], v, rc, "" if good else " (panic/abort)"))
    return ok, "; ".join(notes)


@scenario("A-11 --val-fraction 0 says it validates on training text", "Low")
def _(binary, tag):
    d = scratch("val0", tag)
    rc, out, _ = run(binary, ["train", *TINY, "--val-fraction=0", "--out", d], timeout=120)
    warned = "avertisment" in out and "validare" in out
    return rc == 0 and warned, "exit %s; warning %s" % (rc, "shown" if warned else "absent: the 'validation' numbers are silently measured on the training text")


@scenario("A-12 --demo-mb 0 and 1e9", "Low")
def _(binary, tag):
    notes, ok = [], True
    for v in ("0", "1e9"):
        d = scratch("demo" + v, tag)
        rc, out, dt = run(binary, ["train", "--demo", "--demo-mb=" + v, "--preset", "nano", "--steps", "2", "--out", d],
                          timeout=60, mem_mb=3000)
        good = clean_refusal(rc, out)
        ok &= good
        notes.append("%s→exit %s in %.1fs%s" % (v, rc, dt, "" if good else " (panic/abort/hang)"))
    return ok, "; ".join(notes)


@scenario("A-13 generate --max-new 5000000000", "Medium")
def _(binary, tag):
    d = scratch("maxnew", tag)
    train_tiny(binary, d)
    p = subprocess.Popen([binary, "generate", "--model", d, "--max-new", "5000000000", "--seed", "1"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=limits(3000))
    try:
        out, _ = p.communicate(timeout=6)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        rc = "still generating"
    text = out.decode("utf-8", "replace")
    return rc == "still generating" and len(text) > 20, "%s; %d bytes of output; %s" % (rc, len(text), text.strip().splitlines()[-1][:100] if text.strip() else "")


@scenario("A-15 --threads 100000000 (capped to 3 GB of memory)", "Info")
def _(binary, tag):
    rc, out, dt = run(binary, ["info", "--threads", "100000000"], timeout=30, mem_mb=3000)
    return clean_refusal(rc, out), "exit %s in %.1fs: %s" % (rc, dt, out.strip().splitlines()[-1][:100] if out.strip() else "")


@scenario("A-16 Ctrl-C with --accum 60", "Low")
def _(binary, tag):
    d = scratch("sigint", tag)
    p = subprocess.Popen([binary, "train", "--demo", "--demo-mb", "0.5", "--preset", "nano", "--seq", "256", "--batch", "16",
                          "--accum", "60", "--steps", "5", "--save-every", "100000", "--eval-every", "100000",
                          "--eval-batches", "1", "--log-every", "100000", "--out", d],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=limits(None))
    time.sleep(5)
    t0 = time.time()
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(timeout=90)
        dt = time.time() - t0
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        dt = 999
    return dt < 5, "exited %.1fs after Ctrl-C" % dt


def summary():
    print("\n%-52s %-8s %s" % ("scenario", "severity", "  ".join(t for t in ("old", "new") if getattr(ARGS, t))))
    failed = False
    for row in results:
        cells = []
        for tag in ("old", "new"):
            if tag in row:
                cells.append({True: "safe", False: "UNSAFE", None: "error"}[row[tag][0]])
        print("%-52s %-8s %s" % (row["name"], row["severity"], "  ".join("%-6s" % c for c in cells)))
        if "new" in row and row["new"][0] is not True and row["severity"] != "Info":
            failed = True
    if ARGS.json:
        json.dump(results, open(ARGS.json, "w"), indent=1, default=str)
    if not ARGS.work:
        shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failed else 0


sys.exit(summary())
