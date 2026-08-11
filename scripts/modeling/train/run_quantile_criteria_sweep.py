"""Quantile x criterion sweep of the APPROVED pipeline GNN.

Runs the exact ensemble-pipeline signed GNN (`orbind.gnn_extractor.GnnSignedExtractor`
-> `_SignedSage`, emit=prot, 5-model bag) across every molecule-ranking criterion
and a set of coverage quantiles, and writes the compact per-(criterion, quantile,
seed) CSV that `notebooks/graph/alternatives/protein_based_graph.ipynb` DISPLAYS.
The notebook no longer trains anything -- this script is the single producer.

Why this exists: the notebook used to reimplement the GNN inline (hetero.HeteroLink,
its own split/ChemBERTa file), which drifted from the pipeline. Here the feature is
built exactly as the pipeline's `cls+mol` combo: `[ bagged graph-refined protein ||
raw molecule ]` (column order matches `--combos "12"`, source 1 = gnn cls, source 2
= mol), fitted with the fixed head `train_boost` at `seed=repeat` -- the same seed
`orbind.ensemble.run_ensemble` uses for a combo (baselines.fit_boost(..., seed=seed)).

Configurable: regime (inductive/transductive), molecule-embedding source (also the
GNN's MP node features, matching the pipeline), quantiles, criteria, seeds.
Hardwired: the 7 molecule-ranking criteria (`orbind.mol_selection`) and the
established GNN architecture + hyperparameters (`GnnSignedExtractor` defaults:
signed 2-layer SAGE, hidden 256, dropout 0.3, lr 3e-3, grad-clip 1.0, 900 epochs,
5-model bag, seed_offset 5000).

Resumable: rows are keyed by (criterion, quantile, seed) and written incrementally
(atomic tmp-replace); re-running skips finished cells.

    python scripts/modeling/train/run_quantile_criteria_sweep.py \
        --regime inductive \
        --mol-embeddings data/embeddings/molecules/chemberta_77m_m2or.npz \
        --quantiles 50 80 85 90 95 99 --boost-full
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

from orbind.gnn_extractor import GnnSignedExtractor          # noqa: E402
from orbind.mol_selection import CRITERIA, quality_K         # noqa: E402
from orbind.baselines import train_boost                     # noqa: E402
from orbind.dataset import metrics, load_npz_dict            # noqa: E402
from orbind.regimes import full_full_pairs, load_split       # noqa: E402

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
# CLI regime -> the pipeline split key persisted in full_full_split_indices.npz
REGIME_KEY = {"inductive": "inductive_molecule_v5", "transductive": "transductive"}
DEFAULT_REPEATS = {"inductive": [42, 43, 44, 45, 46], "transductive": [1, 2, 3, 4, 5]}


def _pair_matrix(emb: dict, keys) -> np.ndarray:
    return np.stack([emb[k] for k in keys]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", choices=["inductive", "transductive"], default="inductive")
    ap.add_argument("--mol-embeddings", default="data/embeddings/molecules/chemberta_77m_m2or.npz",
                    help="molecule embedding npz (inchikey-keyed): both the GNN's MP node "
                         "features AND the raw-molecule half of the boost feature")
    ap.add_argument("--prot-embeddings", default="data/embeddings/proteins/esm1b_650m_mean.npz")
    ap.add_argument("--quantiles", type=float, nargs="+", default=[50, 80, 85, 90, 95, 99],
                    help="coverage quantiles as PERCENTS (matches the notebook x-axis)")
    ap.add_argument("--criteria", nargs="+", default=list(CRITERIA),
                    help=f"subset of {list(CRITERIA)}")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override; default = inductive 42-46 / transductive folds 1-5")
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--pool-fold", type=int, default=1)
    ap.add_argument("--boost-full", action="store_true",
                    help="also write the no-graph [protein||molecule] baseline "
                         "(quantile-independent, replicated at every quantile for the flat line)")
    ap.add_argument("--out", default=None,
                    help="CSV path; default results/graph/full_full/v6/protein_based_graph/"
                         "metrics_{regime_key}.csv (what the notebook reads)")
    args = ap.parse_args()

    unknown = [c for c in args.criteria if c not in CRITERIA]
    if unknown:
        ap.error(f"unknown criteria {unknown}; choose from {list(CRITERIA)}")

    regime_key = REGIME_KEY[args.regime]
    seeds = args.seeds or DEFAULT_REPEATS[args.regime]
    # default CSV name carries the molecule-embedding stem so the ChemBERTa-node
    # and GIN-node bagged sweeps land in separate files (the notebook reads both).
    mol_stem = pathlib.Path(args.mol_embeddings).stem
    out = pathlib.Path(args.out) if args.out else (
        _root / "results/graph/full_full/v6/protein_based_graph"
        / f"metrics_{regime_key}__{mol_stem}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"regime={args.regime} ({regime_key})  seeds={seeds}  "
          f"quantiles={args.quantiles}  criteria={args.criteria}", flush=True)
    print(f"mol={args.mol_embeddings}\nprot={args.prot_embeddings}\nout={out}", flush=True)

    pairs = full_full_pairs(pool_fold=args.pool_fold)
    mol_emb = load_npz_dict(str(_root / args.mol_embeddings))
    prot_emb = load_npz_dict(str(_root / args.prot_embeddings))

    ik = pairs["inchikey"].to_numpy()
    rc = pairs["receptor"].to_numpy()
    lab = pairs["label"].to_numpy().astype(np.float32)
    cov = pairs["inchikey"].isin(mol_emb).to_numpy() & pairs["receptor"].isin(prot_emb).to_numpy()

    def covered(idx):
        idx = np.asarray(idx)
        return idx[cov[idx]]

    # ---- resumable state -------------------------------------------------
    rows = []
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["criterion"], float(r["quantile"]), int(r["seed"])) for _, r in prev.iterrows()}
        print(f"resuming: {len(done)} cells already in {out.name}", flush=True)

    def save():
        tmp = out.with_suffix(".tmp.csv")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(out)

    def record(crit, q, K, seed, m):
        rows.append({"criterion": crit, "quantile": q, "K": K, "seed": seed,
                     **{k: float(m[k]) for k in METRICS}})
        done.add((crit, float(q), int(seed)))
        save()

    # ---- sweep -----------------------------------------------------------
    for seed in seeds:
        tr, va, te = (covered(a) for a in load_split(regime_key, seed))
        y_tr, y_te = lab[tr], lab[te]
        Xm_tr, Xm_te = _pair_matrix(mol_emb, ik[tr]), _pair_matrix(mol_emb, ik[te])
        Xp_tr_raw, Xp_te_raw = _pair_matrix(prot_emb, rc[tr]), _pair_matrix(prot_emb, rc[te])

        # K per quantile (kept-molecule count) for the display column: coverage
        # of train molecules; cold (zero-coverage) molecules never count either way.
        uniq = pd.unique(ik[tr])
        loc = {k: i for i, k in enumerate(uniq)}
        cov_counts = np.bincount([loc[k] for k in ik[tr]], minlength=len(uniq))
        Kq = {q: quality_K(cov_counts, q / 100.0) for q in args.quantiles}
        n_train_mols = len(uniq)

        if args.boost_full and any(("boost_full", float(q), int(seed)) not in done for q in args.quantiles):
            # no-graph reference: [raw protein || raw molecule], column order as pipeline
            sc = train_boost(np.concatenate([Xp_tr_raw, Xm_tr], 1), y_tr,
                             np.concatenate([Xp_te_raw, Xm_te], 1), seed=seed)
            mb = metrics(y_te, sc)
            for q in args.quantiles:
                if ("boost_full", float(q), int(seed)) not in done:
                    record("boost_full", q, n_train_mols, seed, mb)
            print(f"  seed {seed} boost_full: " + " ".join(f"{k}={mb[k]:.3f}" for k in METRICS), flush=True)

        for q in args.quantiles:
            for crit in args.criteria:
                if (crit, float(q), int(seed)) in done:
                    continue
                ext = GnnSignedExtractor(
                    name="gnn", protein_path=args.prot_embeddings,
                    molecule_path=args.mol_embeddings, q=q / 100.0, criterion=crit,
                    n_models=args.n_models, epochs=args.epochs, emit="prot")
                Zp_tr, _Zp_va, Zp_te = ext.fit_transform(pairs, tr, va, te, seed)
                # pipeline `cls+mol` combo "12": source1=gnn(prot) then source2=mol
                feat_tr = np.concatenate([Zp_tr, Xm_tr], 1)
                feat_te = np.concatenate([Zp_te, Xm_te], 1)
                sc = train_boost(feat_tr, y_tr, feat_te, seed=seed)
                m = metrics(y_te, sc)
                record(crit, q, int(Kq[q]), seed, m)
                print(f"  seed {seed} q{int(q)} {crit:18} "
                      + " ".join(f"{k}={m[k]:.3f}" for k in METRICS), flush=True)

    print(f"\ndone -> {out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
