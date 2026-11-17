#!/usr/bin/env python3
"""Fails only for RustSec advisories about crates that GERMEN really compiles.

`cargo audit` checks Cargo.lock, and Cargo.lock also lists crates that are never built: Burn's
sub-crates have optional backends (LibTorch pulls in `zip` and `zstd`, a database engine pulls in
`chrono`) that no feature of GERMEN switches on. An advisory about such a crate cannot reach the
binary, and a weekly check that is red for that reason teaches everybody to ignore it.

    cargo audit --json > audit.json
    python3 scripts/audit_graph.py audit.json

The crates that are compiled are those `cargo tree` reaches for any target (normal and build
dependencies, default features, which is what a release build uses).

Verdict:
  * a vulnerability, or an `unsound` warning, about a compiled crate: exit 1;
  * `unmaintained`, `yanked` and other notices about a compiled crate: listed, exit 0;
  * anything about a crate that is only in Cargo.lock: listed as such, exit 0;
  * a report that is not what this script expects (cargo-audit failed, the format changed): exit 2,
    so that a broken check can never pass as a clean one.
"""

import json
import re
import subprocess
import sys

FAILING_WARNINGS = {"unsound"}


def compiled_crates():
    """The set of (name, version) that a build of the workspace compiles."""
    out = subprocess.run(
        ["cargo", "tree", "--locked", "--target", "all", "-e", "normal,build",
         "--prefix", "none", "--format", "{p}"],
        check=True, capture_output=True, text=True,
    ).stdout
    crates = set()
    for line in out.splitlines():
        m = re.match(r"(\S+) v(\S+)", line.strip())
        if m:
            crates.add((m.group(1), m.group(2)))
    if len(crates) < 50:  # Burn alone brings several hundred
        raise SystemExit(f"error: `cargo tree` listed only {len(crates)} crates; refusing to judge")
    return crates


def entries(report):
    """(kind, entry) for every vulnerability and warning of a `cargo audit --json` report."""
    vulns = report["vulnerabilities"]
    listed = vulns["list"]
    if vulns.get("count", len(listed)) != len(listed):
        raise SystemExit("error: the report's vulnerability count does not match its list")
    for v in listed:
        yield "vulnerability", v
    for kind, items in report["warnings"].items():
        for w in items:
            yield kind, w


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        with open(argv[1], encoding="utf-8") as f:
            report = json.load(f)
        found = list(entries(report))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"error: {argv[1]} is not a cargo-audit report ({e!r})", file=sys.stderr)
        return 2

    compiled = compiled_crates()
    applies, lock_only = [], []
    for kind, e in found:
        pkg = e["package"]
        advisory = e.get("advisory") or {}
        row = (kind, pkg["name"], pkg["version"], advisory.get("id", "-"), advisory.get("title", ""))
        (applies if (pkg["name"], pkg["version"]) in compiled else lock_only).append(row)

    total = (report.get("lockfile") or {}).get("dependency-count", "?")
    print(f"{len(found)} advisories in Cargo.lock ({total} crates); "
          f"{len(compiled)} crates are compiled.")
    if lock_only:
        names = sorted({f"{n} {v}" for _, n, v, _, _ in lock_only})
        print(f"\nIn Cargo.lock only, never compiled — {len(lock_only)} advisories about "
              f"{len(names)} crates, ignored: {', '.join(names)}")
    failing = [r for r in applies if r[0] == "vulnerability" or r[0] in FAILING_WARNINGS]
    notices = [r for r in applies if r not in failing]
    if notices:
        print("\nAbout compiled crates, for information:")
        for kind, name, version, adv, title in notices:
            print(f"  {kind}: {name} {version}  {adv}  {title}")
    if failing:
        print("\nABOUT COMPILED CRATES — must be fixed:")
        for kind, name, version, adv, title in failing:
            print(f"  {kind}: {name} {version}  {adv}  {title}")
        return 1
    print("\nNo advisory concerns a compiled crate.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
