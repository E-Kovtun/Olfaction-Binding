"""What a sweep root actually contains: which dial, which alphas, seeds and folds.

The tables assume a sweep was produced with particular flags -- `--dial nodes`,
`--seed-graph`, alpha 1.0, five model seeds -- and two roots compared cell by cell
must agree on all of them. Shell history is the wrong place to check that; the CSVs
record it, and this prints it.

    python scripts/analysis/sweep_provenance.py --root results/graph/v13_esm3
    python scripts/analysis/sweep_provenance.py --root results/graph/v9_seeded \
                                                --root results/graph/v13_esm3

With two roots it also names every field they disagree on, which is the question
worth asking before putting their numbers in the same sentence.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

_repo = pathlib.Path(__file__).resolve()
while not (_repo / "pyproject.toml").exists():
    _repo = _repo.parent
sys.path.insert(0, str(_repo))

#: Columns worth reporting, in the order a reader cares about them. Missing ones are
#: skipped rather than faked: an older series genuinely has no `seeded_graph`, and
#: printing "False" for it would assert something the file does not say.
FIELDS = ["alpha", "seed", "fold", "seeded_graph", "dial", "mol_source", "variant"]


def summarise(root: pathlib.Path) -> dict:
    files = sorted(root.glob("metrics_*.csv"))
    if not files:
        return {"cells": 0}
    out, frames = {"cells": len(files)}, []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return out
    df = pd.concat(frames, ignore_index=True)
    out["rows"] = len(df)
    for col in FIELDS:
        if col in df.columns:
            vals = sorted(pd.unique(df[col].dropna()).tolist())
            out[col] = vals if len(vals) <= 8 else [vals[0], "...", vals[-1]]
    out["files"] = [f.name for f in files]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True,
                    help="a sweep root, repeatable; two roots are also diffed")
    ap.add_argument("--files", action="store_true", help="also list the metrics files")
    args = ap.parse_args(argv)

    summaries = {}
    for r in args.root:
        root = pathlib.Path(r)
        if not root.is_absolute():
            root = _repo / root
        s = summarise(root)
        summaries[r] = s
        print(f"\n=== {r}")
        if not s["cells"]:
            print("  no metrics_*.csv here")
            continue
        print(f"  {s['cells']} cell file(s), {s.get('rows', 0)} rows")
        for col in FIELDS:
            if col in s:
                print(f"  {col:<14} {s[col]}")
        if args.files:
            for name in s["files"]:
                print(f"    {name}")

    if len(summaries) == 2:
        (na, a), (nb, b) = summaries.items()
        diffs = [c for c in FIELDS if a.get(c) != b.get(c)]
        print(f"\n=== {na} vs {nb}")
        if not diffs:
            print("  same on every recorded field -- comparable")
        for c in diffs:
            print(f"  {c:<14} {a.get(c, '<absent>')}  !=  {b.get(c, '<absent>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
