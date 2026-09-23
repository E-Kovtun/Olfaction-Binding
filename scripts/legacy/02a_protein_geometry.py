#!/usr/bin/env python
"""Geometry of FROZEN protein embeddings against the functional response profile -- the
rows of the geometry table that no sweep computes.

The sweep writes RSA / CCA / Procrustes of OUR receptor cloud against the TRAIN response
profile in every cell. This computes the same three numbers, with the same functions,
on the same receptor order and the same profile matrix -- it calls the sweep's own
`_prepare` / `_fold_prep` -- for the receptor features nothing trains: ESM-1b, ProtT5,
ESM-2 (each only where its npz exists and covers every receptor), the six classical
descriptors of tab:t4, and one-hot identity. No model, no GPU.

    .venv/bin/python scripts/legacy/02a_protein_geometry.py --dataset cc hc
    .venv/bin/python scripts/legacy/02a_protein_geometry.py --dataset m2or --regime transductive

One CSV per (dataset, regime): results/article_tables/protein_geometry/<ds>_<regime>.csv,
one row per (fold, embedding), `{rsa,cca,procrustes}_fun` and their `_z` against a
row-permutation null. An existing file is skipped unless --force.

M2OR CAUTION: the profile is sparse there and the imputed matrix partly encodes WHICH
pairs were assayed; see the `tested mask` control in mechanism_holdout. Read M2OR rows of
this table as contaminated by assay design.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

GEOMS = ["rsa", "cca", "procrustes"]


def _module(rel, name):
    spec = importlib.util.spec_from_file_location(name, tk.ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def embeddings(ds, recs, pf, load_npz_dict):
    """{name: {receptor: vector}} in tab:t4's order: pLMs, classical descriptors, one-hot.
    Receptor ids in these pools are the sequences, which is what the descriptors read."""
    out = {}
    for name, path in pf.DATASETS[ds]["plms"].items():
        p = tk.ROOT / path
        if not p.exists():
            print(f"  [skip {name}] {path} not found")
            continue
        emb = load_npz_dict(str(p))
        miss = sum(r not in emb for r in recs)
        if miss:
            print(f"  [skip {name}] covers {len(recs) - miss}/{len(recs)} receptors")
            continue
        out[name] = {r: np.asarray(emb[r], np.float64) for r in recs}
    for name, fn in pf.DESC.items():
        out[name] = {r: np.asarray(fn(r), np.float64) for r in recs}
    eye = np.eye(len(recs))
    out["onehot"] = {r: eye[i] for i, r in enumerate(recs)}
    return out


def fold_rows(ds, regime, fold, P, vecs, n_perm, GEOMETRY, geometry_nulls):
    M, order = P["ref"]["fun"], P["order"]
    rows = []
    for name, vec in vecs.items():
        X = np.stack([vec[r] for r in order])
        rec = dict(dataset=ds, regime=regime, fold=int(fold), embedding=name,
                   dim=int(X.shape[1]), n_receptors=len(order), n_train_mols=int(M.shape[1]))
        nulls = geometry_nulls(X, M, n_perm) if n_perm else {}
        for g in GEOMS:
            v = float(GEOMETRY[g](X, M))
            rec[f"{g}_fun"] = v
            if g in nulls:
                mu, sd = nulls[g]
                rec[f"{g}_fun_z"] = float((v - mu) / sd) if sd and sd > 1e-12 else np.nan
        rows.append(rec)
    return rows


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["cc", "hc"], choices=tk.DATASETS)
    ap.add_argument("--regime", nargs="+", default=tk.REGIMES, choices=tk.REGIMES)
    ap.add_argument("--mol-source", default="chemberta", choices=tk.MOL_SOURCES,
                    help="only decides the coverage mask, exactly as in the sweep")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=f"{tk.OUT_ROOT}/protein_geometry")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    from orbind.dataset import load_npz_dict
    from scripts.modeling.analysis.mechanism_holdout import GEOMETRY, geometry_nulls
    sw = _module("scripts/modeling/train/run_alpha_gate_sweep.py", "_sweep_for_geometry")
    pf = _module("scripts/modeling/analysis/prot_floor_sweep.py", "_prot_floor_for_geometry")
    ns = argparse.Namespace(mol_source=a.mol_source, mol_embeddings=None,
                            prot_embeddings=None, pool_fold=1)
    out = tk.out_dir(a.out)
    for ds in a.dataset:
        todo = [r for r in a.regime if a.force or not (out / f"{ds}_{r}.csv").exists()]
        if not todo:
            print(f"{ds}: every regime already written (--force to redo)")
            continue
        data = sw._prepare(ds, ns)
        recs = list(pd.unique(data[0]["receptor"]))
        print(f"{ds}: {len(recs)} receptors")
        vecs = embeddings(ds, recs, pf, load_npz_dict)
        for regime in todo:
            rows = []
            for fold in sw.REPEATS[ds].get(regime, [1, 2, 3, 4, 5]):
                P = sw._fold_prep(ds, regime, fold, ns, data)
                rows += fold_rows(ds, regime, fold, P, vecs, a.n_perm, GEOMETRY,
                                  geometry_nulls)
                print(f"  {ds}/{regime} fold {fold}: {len(vecs)} embeddings")
            path = out / f"{ds}_{regime}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  -> {path}")


if __name__ == "__main__":
    main()
