#!/usr/bin/env python
"""Fit the boosting head over [ONE-HOT receptor || molecule] -- the row no sweep writes.

The sweep's `boost_full` is the head over [ESM || molecule]. This is the same head with
the protein block replaced by a one-hot receptor identity, which is the honest floor for
our graph at alpha=0: identity with no refinement at all. A graph at alpha=0 that does
not beat this has learned nothing from the response profile.

Everything except that block is the sweep's own code -- the same fold indices, the same
coverage mask, the same `fit_boost` hyperparameters, the same metric battery
(`_score_split`) -- so the row lands beside `boost_full` as a like-for-like control and
not as a second implementation of boosting.

This is the SLOW half, and it is separated for that reason: `05_alpha0_vs_boost.py`
only reads what this writes. On M2OR the one-hot block is 1237 columns wide over ~46k
rows, so budget minutes per fold; the insect panels are seconds.

    .venv/bin/python scripts/article_tables/05a_onehot_boost.py \\
        --dataset m2or cc hc \\
        --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz'

One CSV per (dataset, regime) under results/article_tables/onehot_boost/, one row per
fold. An existing file is skipped unless --force.

**Pass the same protein npz the sweep used.** The one-hot block replaces ESM in the
features, but the file still decides the COVERAGE MASK, and a different mask is a
different set of rows in every fold -- which would make this row incomparable with the
sweep's own, quietly.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

sys.path.insert(0, str(tk.ROOT))


def _module(rel, name):
    spec = importlib.util.spec_from_file_location(name, tk.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def fold_row(ds, regime, fold, sw, ns, data, seed, task):
    """One fold: fit on train, score test, return the sweep's own metric columns."""
    from orbind.baselines import fit_boost, predict_scores
    P = sw._fold_prep(ds, regime, fold, ns, data)
    order = {r: i for i, r in enumerate(P["order"])}
    eye = np.eye(len(order), dtype=np.float32)

    def block(key):
        idx = [order[r] for r in P[f"rec_{key}"]]
        return np.concatenate([eye[idx], P[f"Xm_{key}"]], axis=1)

    Xtr = block("tr")
    t0 = time.time()
    est = fit_boost(Xtr, P["y_tr"], seed=seed, task=task)
    dt = time.time() - t0
    scores = sw._score_split(P, predict_scores(est, block("te"), task), task, "test")
    print(f"  {ds}/{regime} fold {fold}: {len(order)} receptors, "
          f"{Xtr.shape[1]} features, {len(Xtr)} train rows, {dt:.1f}s")
    return dict(fold=int(fold), seed=int(seed), n_receptors=len(order),
                n_features=int(Xtr.shape[1]), t_head=dt, **scores)


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["m2or", "cc", "hc"])
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"])
    ap.add_argument("--mol-source", default="chemberta", choices=tk.MOL_SOURCES)
    ap.add_argument("--seed", type=int, default=42,
                    help="42 is what every reported boosting number uses; "
                         "fit_boost draws subsample/colsample from it")
    ap.add_argument("--prot-embeddings", default=None,
                    help="THE SAME npz the sweep was run with -- it decides the "
                         "coverage mask. ESM3: 'data/embeddings/proteins/esm3_{ds}.npz'")
    ap.add_argument("--out", default="results/article_tables/onehot_boost")
    ap.add_argument("--force", action="store_true", help="recompute existing CSVs")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    sw = _module("scripts/modeling/train/run_alpha_gate_sweep.py", "_sweep_for_onehot")
    ns = argparse.Namespace(mol_source=a.mol_source, mol_embeddings=None,
                            prot_embeddings=a.prot_embeddings, pool_fold=1)
    out = tk.out_dir(a.out)
    for ds in a.dataset:
        todo = [r for r in a.regime
                if r in sw.REPEATS.get(ds, {})
                and (a.force or not (out / f"{ds}_{r}.csv").exists())]
        if not todo:
            print(f"{ds}: nothing to do (--force to redo)")
            continue
        data = sw._prepare(ds, ns)
        print(f"{ds}: {len(pd.unique(data[0]['receptor']))} receptors in the pool")
        for regime in todo:
            rows = [fold_row(ds, regime, f, sw, ns, data, a.seed, tk.TASK[ds])
                    for f in sw.REPEATS[ds][regime]]
            path = out / f"{ds}_{regime}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
