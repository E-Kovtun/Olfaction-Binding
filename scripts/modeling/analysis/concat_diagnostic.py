#!/usr/bin/env python
"""Why does GNN + PCA(ESM) score below the GNN alone? Three candidate causes, separated.

Reads an existing run's `embeddings.npz` -- no training, seconds to run:

    .venv/bin/python scripts/modeling/analysis/concat_diagnostic.py --dataset hc

The three explanations make different predictions, and each column below is the one that
tells them apart:

  DILUTION      the concat is an equal-weight blend by construction, so if ESM sits at the
                null the blend is dragged halfway to it. Predicts: the alpha sweep falls
                MONOTONICALLY from pure GNN to pure ESM, with no dip and no bump.
  DIMENSION     more columns hurt per se. Predicts: the score falls as the ESM block gets
                wider, and the RANDOM control (same width, same norm, no structure) hurts
                just as much as ESM does.
  WRONG SHAPE   ESM's similarity structure is strong and coherent but unrelated to the
                class -- receptor family, not ligand chemistry. Predicts: ESM hurts MORE
                than random noise of identical width and norm, because noise averages out
                across dimensions while a coherent wrong structure does not.

The last block reports how much of the top-k PCA -- the only part CCA and Procrustes ever
see -- each side actually owns. If ESM owns the leading subspace, those two measures are
scoring ESM no matter what the other 250 columns hold.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from scripts.modeling.analysis.mechanism_holdout import (  # noqa: E402
    DATASETS, GEOMETRY, _pca, block_concat, geometry_k, geometry_nulls, pca_esm)


def unit(B):
    B = np.asarray(B, np.float64)
    B = B - B.mean(0)
    return B / (np.sqrt((B ** 2).sum(1).mean()) + 1e-12)


def mix(G, E, alpha):
    """The same construction as block_concat, with the blocks weighted alpha : 1 - alpha."""
    if alpha >= 1.0:
        return unit(G)
    if alpha <= 0.0:
        return unit(E)
    return np.hstack([alpha * unit(G), (1.0 - alpha) * unit(E)])


def block_share(G, E, k):
    """Fraction of the top-k principal subspace's energy contributed by each block."""
    X = block_concat(G, E)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    V = Vt[:k].T                                   # (dim, k) leading directions
    w = (S[:k] ** 2) / max((S[:k] ** 2).sum(), 1e-12)
    g = (V[:G.shape[1]] ** 2).sum(0)               # energy of each direction in the G block
    e = (V[G.shape[1]:] ** 2).sum(0)
    return float((w * g).sum()), float((w * e).sum())


def run(ds, args):
    out = pathlib.Path(args.out) / ds
    f = out / "embeddings.npz"
    if not f.exists():
        raise SystemExit(f"no {f} -- run mechanism_holdout.py first (without --no-embeddings)")
    with np.load(f, allow_pickle=True) as z:
        emb = {k: z[k] for k in z.files}
    classes = [k[len("target__"):] for k in emb if k.startswith("target__")]
    seed = str(args.seed)
    rng = np.random.default_rng(0)

    rows, shares = [], []
    for c in classes:
        M = np.asarray(emb[f"target__{c}"], np.float64)
        gk = f"emb__{c}__esm__{seed}"
        if gk not in emb:
            print(f"  {c}: no graph embedding for seed {seed} -- skipped")
            continue
        G = np.asarray(emb[gk], np.float64)
        Xe = np.asarray(emb[f"emb__{c}__esm__0"], np.float64)
        E, edim = pca_esm(Xe)
        null = {g: geometry_nulls(mix(G, E, 0.5), M, args.n_perm)[g][0] for g in GEOMETRY}

        for label, X in [(f"alpha {a:.2f}", mix(G, E, a)) for a in args.alphas] + \
                        [(f"ESM PCA dim {d}", block_concat(G, _pca(Xe, min(d, edim))))
                         for d in args.dims if d <= edim] + \
                        [("random, ESM width", block_concat(
                            G, rng.normal(size=(len(G), E.shape[1]))))]:
            rows.append(dict(cls=c, variant=label, dim=int(X.shape[1]),
                             **{g: fn(X, M) for g, fn in GEOMETRY.items()},
                             **{f"{g}_null": null[g] for g in GEOMETRY}))
        k = geometry_k(block_concat(G, E), M)
        sg, se = block_share(G, E, k)
        shares.append(dict(cls=c, k=k, gnn_share=sg, esm_share=se,
                           gnn_dim=G.shape[1], esm_dim=E.shape[1]))

    if not rows:
        return
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for g in GEOMETRY:
        t = res.pivot_table(index="variant", columns="cls", values=g, sort=False)
        t["MEAN"] = t.mean(1)
        t["null"] = res.groupby("variant", sort=False)[f"{g}_null"].mean()
        print(f"\n=== {ds.upper()} / {g} " + "-" * 40)
        print(t.round(3).to_string())
    print(f"\n=== {ds.upper()} / who owns the top-k subspace CCA and Procrustes see "
          + "-" * 10)
    print(pd.DataFrame(shares).round(3).to_string(index=False))
    if args.csv:
        res.to_csv(out / "concat_diagnostic.csv", index=False)
        print(f"\nwritten to {out/'concat_diagnostic.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["all"])
    ap.add_argument("--seed", type=int, default=42, help="which graph seed to diagnose")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--csv", action="store_true", help="also write concat_diagnostic.csv")
    ap.add_argument("--out", default="results/mechanism_holdout")
    args = ap.parse_args()
    todo = list(DATASETS) if "all" in args.dataset else args.dataset
    for ds in todo:
        run(ds, args)


if __name__ == "__main__":
    main()
