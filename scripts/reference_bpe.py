#!/usr/bin/env python3
"""A deliberately naive byte-pair-encoding trainer and encoder, to check GERMEN's fast one.

    cargo run --release -p germen-core --example bpe_dump -- TRAIN.txt VOCAB OUT.json [PROBE.txt]
    python3 scripts/reference_bpe.py TRAIN.txt VOCAB OUT.json [PROBE.txt]

The reference recounts every pair from scratch at every merge (no heap, no incremental counts, no
word cache, no threads) and follows only what src/bpe.rs documents: the pre-token rules, "most
frequent pair, ties to the smaller pair", at least two occurrences, and merging left to right.
It compares the list of merges, the tokens of the training file and the tokens of the probe file.
"""
import json, sys
from collections import Counter

MAXP = 64


def ws(b):
    return b in b" \n\r\t\x0b\x0c"


def cls(b):
    if 65 <= b <= 90 or 97 <= b <= 122 or b >= 0x80:
        return 0
    return 1 if 48 <= b <= 57 else 2


def pretokens(d):
    n, i, out = len(d), 0, []
    while i < n:
        j = i
        if d[j] == 32 and j + 1 < n and not ws(d[j + 1]):  # one space glues to the word after it
            j += 1
        if ws(d[j]):
            while j < n and ws(d[j]) and j - i < MAXP + 1:
                j += 1
            if j < n and j - i > 1 and d[j - 1] == 32:  # "  word": the last space goes with the word
                j -= 1
        else:
            c = cls(d[j])
            while j < n and not ws(d[j]) and cls(d[j]) == c and j - i < MAXP + 1:
                j += 1
        j = max(min(j, i + MAXP), i + 1)
        out.append(d[i:j])
        i = j
    return out


def merge(syms, pair, new):
    out, k = [], 0
    while k < len(syms):
        if k + 1 < len(syms) and syms[k] == pair[0] and syms[k + 1] == pair[1]:
            out.append(new)
            k += 2
        else:
            out.append(syms[k])
            k += 1
    return out


def train(data, vocab):
    words = Counter(w for w in pretokens(data) if len(w) > 1)
    words = {tuple(w): c for w, c in words.items()}
    merges = []
    while len(merges) < vocab - 256:
        pc = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pc[p] += c
        if not pc:
            break
        best = max(pc.values())
        if best < 2:
            break
        pair = min(p for p, c in pc.items() if c == best)
        new = 256 + len(merges)
        merges.append(list(pair))
        nw = {}
        for w, c in words.items():
            nw[tuple(merge(list(w), pair, new))] = nw.get(tuple(merge(list(w), pair, new)), 0) + c
        words = nw
    return merges


def encode(data, merges):
    rank = {tuple(m): r for r, m in enumerate(merges)}
    out = []
    for w in pretokens(data):
        s = list(w)
        if len(s) > 1:
            while True:
                cand = [rank[p] for p in zip(s, s[1:]) if p in rank]
                if not cand:
                    break
                r = min(cand)
                s = merge(s, tuple(merges[r]), 256 + r)
        out.extend(s)
    return out


def main():
    train_path, vocab, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    data = open(train_path, "rb").read()
    got = json.load(open(out))
    merges = train(data, vocab)
    bad = 0

    def check(name, ok, detail=""):
        nonlocal bad
        bad += not ok
        print(("ok   " if ok else "FAIL ") + name + (" " + detail if detail else ""))

    check(f"merges ({len(merges)} of {len(got['merges'])})", merges == got["merges"],
          "" if merges == got["merges"] else f"first difference at {next((i for i,(a,b) in enumerate(zip(merges,got['merges'])) if a!=b), min(len(merges),len(got['merges'])))}")
    ids = encode(data, merges)
    check(f"tokens of the training file ({len(ids)})", ids == got["train_tokens"])
    check("decoding the tokens gives the file back", got["decoded_ok"])
    if len(sys.argv) > 4:
        probe = open(sys.argv[4], "rb").read()
        pids = encode(probe, merges)
        check(f"tokens of the probe file ({len(pids)})", pids == got["probe_tokens"])
    print("ALL AGREE" if not bad else f"{bad} DISAGREEMENT(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
