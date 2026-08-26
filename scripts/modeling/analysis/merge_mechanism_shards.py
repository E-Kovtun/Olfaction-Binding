#!/usr/bin/env python
"""Merge per-class shards of mechanism_holdout.py back into one artifact directory.

`mechanism_holdout.py` is a single process, and on M2OR the dominant cost is
(classes x models x seeds) graph trainings that share nothing -- no checkpoints, no
state. So the cheap way onto N GPUs is to give each GPU its own class:

    CUDA_VISIBLE_DEVICES=0 python .../mechanism_holdout.py --dataset m2or \
        --classes carboxylic_acid --out results/mh_shards/carboxylic_acid &
    ... one per class ...

Each shard writes a COMPLETE artifact directory for its own class. This script
concatenates them into the layout the notebook expects, so
`notebooks/graph/mechanism_holdout/` cannot tell a sharded run from a serial one:

    python .../merge_mechanism_shards.py --dataset m2or results/mh_shards/*

Sharding is safe because the only cross-class object is the panel figure, which is
built from ONE class anyway; the merge keeps the canonical one and drops the rest.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from scripts.modeling.analysis.mechanism_holdout import DATASETS  # noqa: E402

TABLES = ["readouts.csv", "nulls.csv", "ood.csv"]


def merge(ds, shards, out, panel_class=None):
    order = DATASETS[ds]["classes"]
    metas = {}
    for s in shards:
        mp = s / "meta.json"
        if not mp.exists():
            raise SystemExit(f"{s} has no meta.json -- is it a finished shard?")
        m = json.loads(mp.read_text(encoding="utf-8"))
        if m["dataset"] != ds:
            raise SystemExit(f"{s} is dataset {m['dataset']!r}, not {ds!r}")
        metas[s] = m

    seen = [c for s in shards for c in metas[s]["classes"]]
    dup = {c for c in seen if seen.count(c) > 1}
    if dup:
        raise SystemExit(f"class(es) {sorted(dup)} appear in more than one shard -- "
                         f"merging would double-count rows")
    classes = [c for c in order if c in seen] + [c for c in seen if c not in order]
    missing = [c for c in order if c not in seen]

    out.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        parts = [pd.read_csv(s / t) for s in shards if (s / t).exists()]
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df = df.assign(_o=df["cls"].map({c: i for i, c in enumerate(classes)}))
        df = df.sort_values(["_o"] + [c for c in ("model", "feat", "seed") if c in df],
                            kind="stable").drop(columns="_o")
        df.to_csv(out / t, index=False)
        print(f"  {t:14} {len(df):5d} rows from {len(parts)} shard(s)")

    # molecules.csv: same (inchikey, smiles) rows everywhere, one boolean column per class
    mols = [pd.read_csv(s / "molecules.csv") for s in shards if (s / "molecules.csv").exists()]
    if mols:
        base = mols[0][["inchikey", "smiles"]]
        for m in mols:
            for c in [c for c in m.columns if c not in ("inchikey", "smiles")]:
                base = base.merge(m[["inchikey", c]], on="inchikey", how="left")
        base = base.assign(**{c: base[c].fillna(False).astype(bool)
                              for c in classes if c in base})
        base[["inchikey", "smiles"] + [c for c in classes if c in base]].to_csv(
            out / "molecules.csv", index=False)
        print(f"  molecules.csv  {len(base):5d} rows, {len(classes)} class column(s)")

    emb = {}
    for s in shards:
        f = s / "embeddings.npz"
        if f.exists():
            with np.load(f) as z:
                emb.update({k: z[k] for k in z.files})
    if emb:
        np.savez_compressed(out / "embeddings.npz", **emb)
        print(f"  embeddings.npz {len(emb):5d} arrays")

    # panels/overview are single-class by construction -- keep the canonical one
    want = panel_class or next((c for c in classes), None)
    for f in ("panels.npz", "overview.npz"):
        src = None
        for s in shards:
            if (s / f).exists():
                with np.load(s / f, allow_pickle=True) as z:
                    if str(z["cls"]) == want:
                        src = s / f
                        break
        if src is None:
            src = next((s / f for s in shards if (s / f).exists()), None)
        if src is not None:
            shutil.copy(src, out / f)
            print(f"  {f:14} from {src.parent.name}")

    m0 = dict(metas[shards[0]])
    m0.update(classes=classes, panel_class=want,
              seconds=round(max(m["seconds"] for m in metas.values()), 1),
              seconds_serial=round(sum(m["seconds"] for m in metas.values()), 1),
              sharded=[s.name for s in shards])
    (out / "meta.json").write_text(json.dumps(m0, indent=2), encoding="utf-8")
    print(f"=== {ds.upper()} merged into {out}: {len(classes)} class(es) "
          f"{classes}, wall {m0['seconds']:.0f}s vs {m0['seconds_serial']:.0f}s serial")
    if missing:
        print(f"!!! no shard covered {missing} -- the merged run is INCOMPLETE")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="per-class artifact directories to merge")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--out", default=None,
                    help="destination (default results/mechanism_holdout/<dataset>)")
    ap.add_argument("--panel-class", default=None,
                    help="which class keeps its panels (default: first in canonical order)")
    args = ap.parse_args()
    shards = list(dict.fromkeys(pathlib.Path(s).resolve() for s in args.shards))
    bad = [s for s in shards if not s.is_dir()]
    if bad:
        ap.error(f"not a directory: {bad}")
    out = pathlib.Path(args.out) if args.out else _root / "results" / "mechanism_holdout" / args.dataset
    merge(args.dataset, shards, out, args.panel_class)


if __name__ == "__main__":
    main()
