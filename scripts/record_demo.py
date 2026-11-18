#!/usr/bin/env python3
"""Records the demonstration page of GERMEN Studio from a real run.

The demonstration page (`germen ui --write-demo`) has no program behind it. What answers its requests is
`crates/germen-studio/assets/js/70-demo.js`, which replays a recording made here: a real training of the
smallest model, a text written by the model that came out of it, and two speed measurements.

    germen ui --no-open --workspace /tmp/germen-rec --port 7872      # prints an address ending in #k=<key>
    python3 scripts/record_demo.py 'http://127.0.0.1:7872/#k=<key>'

The workspace must be a new, empty one: the recorded model is called `demo-nano`. Standard library only.
With --from DIR the raw answers saved by an earlier run are used instead of a program; --save DIR keeps them.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "crates/germen-studio/assets/js/70-demo.js"
BEGIN, END = "/* RECORDING:BEGIN", "/* RECORDING:END */"

TRAIN = {
    "accum": 1, "batch": 8, "bpe_sample_mb": 64, "data": {"kind": "demo", "mb": 0.5, "seed": 1}, "eval_batches": 8,
    "eval_every": 40, "grad_clip": 1, "log_every": 5, "lr": 0.0006, "min_lr_ratio": 0.1, "mode": "new", "name": "demo-nano",
    "preset": "nano", "replicas": 0, "sample_every": 60, "save_every": 500, "seed": 1, "steps": 240, "tokenizer": "bpe",
    "val_fraction": 0.05, "vocab": 2048, "warmup": 12, "weight_decay": 0.01,
}
GENERATE = {"model": "demo-nano", "prompt": "Primăvara", "max_new": 700, "temperature": 0.8, "top_k": 40, "window": 256, "seed": 7}
BENCH_ALL = {"preset": "all", "batch": 8, "seq": 128, "steps": 2, "warmup": 1}
BENCH_SCALING = {"preset": "nano", "batch": 8, "seq": 128, "steps": 2, "warmup": 1, "scaling": [1, 2]}


class Studio:
    def __init__(self, address):
        found = re.search(r"#k=([0-9a-f]{64})", address)
        if not found:
            sys.exit("the address must end in #k=<64 hex digits>, as printed by `germen ui`")
        self.base = address.split("#")[0].rstrip("/")
        self.key = found.group(1)

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path, method=method, data=data,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read() or "null")
        except urllib.error.HTTPError as error:
            sys.exit(f"{method} {path}: {error.code} {error.read().decode()[:300]}")

    def record(self, job, limit=900):
        """Every event of a job, each with the second (since the start of the recording) it arrived at."""
        started, after, events = time.time(), 0, []
        while time.time() - started < limit:
            reply = self.call("GET", f"/api/jobs/{job}/events?after={after}&wait=3000")
            now = time.time() - started
            events += [[round(now, 3), item["e"]] for item in reply["events"]]
            after = reply["next"]
            if reply["state"] in ("done", "stopped", "failed"):
                if reply["state"] != "done":
                    sys.exit(f"job {job} ended as {reply['state']}")
                return events
        sys.exit(f"job {job} took too long")

    def run(self, path, body):
        events = self.record(self.call("POST", path, body)["job"])
        return {"request": body, "events": events}


def rounded(value, digits=6):
    """Floats with six significant digits: the recording is read by people and shipped in a web page."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, list):
        return [rounded(x, digits) for x in value]
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    return value


def build(raw):
    system = dict(raw["system"])
    system.update(
        data_dir="/home/demo/GERMEN/data", models_dir="/home/demo/GERMEN/models", workspace="/home/demo/GERMEN",
        memory={"available": 17179869184, "limit": 15461882265},  # a demonstration computer that can hold every size
    )
    model = next(m for m in raw["models"]["models"] if m["name"] == "demo-nano")
    model = {k: v for k, v in model.items() if k != "modified"}
    model["path"] = "/home/demo/GERMEN/models/demo-nano"
    text = "".join(e["text"] for _, e in raw["generate"]["events"] if e["type"] == "text")
    loaded = next(e for _, e in raw["generate"]["events"] if e["type"] == "loaded")
    results = [{"t": t, **e["result"]} for t, e in raw["bench"]["events"] if e["type"] == "bench_result"]
    start = 0.0
    for r in results:  # how long each size took: the replay waits as long
        r["secs"], start = round(r["t"] - start, 3), r["t"]
        del r["t"]
    points = [e["point"] for _, e in raw["scaling"]["events"] if e["type"] == "scaling_point"]
    return rounded({
        "system": system,
        "model": model,
        "train": {"request": raw["train"]["request"], "events": raw["train"]["events"]},
        "generate": {"loaded": loaded, "text": text},
        "bench": {"results": results, "scaling": points},
    })


def write(rec):
    lines = [f"{BEGIN} (written by scripts/record_demo.py from a real run; do not edit by hand) */", "const REC = {"]
    for i, (key, value) in enumerate(rec.items()):
        comma = "," if i < len(rec) - 1 else ""
        lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}{comma}")
    lines += ["};", END]
    source = TARGET.read_text(encoding="utf-8")
    start, stop = source.index(BEGIN), source.index(END) + len(END)
    TARGET.write_text(source[:start] + "\n".join(lines) + source[stop:], encoding="utf-8", newline="\n")
    print(f"wrote {TARGET} ({TARGET.stat().st_size // 1024} KiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address", nargs="?", help="the address printed by `germen ui`, with its #k=… key")
    parser.add_argument("--from", dest="source", help="use the answers saved in this folder instead of a program")
    parser.add_argument("--save", help="keep the raw answers in this folder")
    args = parser.parse_args()
    if args.source:
        raw = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in Path(args.source).glob("*.json")}
    elif args.address:
        studio = Studio(args.address)
        raw = {"system": studio.call("GET", "/api/system")}
        raw["train"] = studio.run("/api/train", TRAIN)
        raw["generate"] = studio.run("/api/generate", GENERATE)
        raw["bench"] = studio.run("/api/bench", BENCH_ALL)
        raw["scaling"] = studio.run("/api/bench", BENCH_SCALING)
        raw["models"] = studio.call("GET", "/api/models")
    else:
        parser.error("give the address of a running program, or --from")
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        for name, value in raw.items():
            Path(args.save, name + ".json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    write(build(raw))


if __name__ == "__main__":
    main()
