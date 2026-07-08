"""Unified global 5-fold CV + per-receptor GIN-only XGBoost per fold.

Single global StratifiedKFold over ALL curated pairs (same style as the
'stratified' baseline split) -> for each fold, for each receptor: train a
GIN-only XGBoost on that receptor's OWN train-fold pairs, score its own
test-fold pairs. No protein embedding, no cross-receptor sharing at all —
this isolates exactly what per-receptor molecular chemistry buys you, under
the same global split protocol as the standard ESM+GIN boost baseline.

Fallback (score=0, i.e. predict negative) when a receptor's train slice this
fold is unusable:
  - zero train pairs for this receptor this fold (all its pairs landed in test), or
  - train slice is single-class (can't fit a binary classifier).

    uv run python scripts/modeling/per_receptor_unified_kfold.py --folds 1
    uv run python scripts/modeling/per_receptor_unified_kfold.py --folds 5
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import dataset as D
from orbind.baselines import train_boost

OUT = ROOT / "notebooks" / "cache" / "per_receptor_unified_kfold.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def run_fold(pairs, GIN, fold_id, tr_idx, te_idx, seed=42):
    tr = pairs.iloc[tr_idx]
    te = pairs.iloc[te_idx]
    scores = np.zeros(len(te), dtype=np.float32)
    n_model = n_fallback = 0

    for rec, te_sub in te.groupby("receptor"):
        tr_sub = tr[tr["receptor"] == rec]
        y_tr = tr_sub["label"].to_numpy(np.float32)
        te_pos = te_sub.index

        if len(tr_sub) == 0 or len(np.unique(y_tr)) < 2:
            # fallback: unusable train slice -> predict negative
            scores[te.index.get_indexer(te_pos)] = 0.0
            n_fallback += len(te_sub)
            continue

        X_tr = np.stack([GIN[i] for i in tr_sub["inchikey"]]).astype(np.float32)
        X_te = np.stack([GIN[i] for i in te_sub["inchikey"]]).astype(np.float32)
        pred = train_boost(X_tr, y_tr, X_te, seed=seed)
        scores[te.index.get_indexer(te_pos)] = pred
        n_model += len(te_sub)

    y_te = te["label"].to_numpy(np.float32)
    m = D.metrics(y_te, scores)
    return m, n_model, n_fallback, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=1, help="how many of the 5 folds to run now")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    GIN = D.load_npz_dict(ROOT / "data" / "embeddings" / "molecules" / "gin_supervised_contextpred.npz")
    PAIRS = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    PAIRS = PAIRS[PAIRS["inchikey"].isin(GIN)].reset_index(drop=True)
    y_all = PAIRS["label"].to_numpy()
    print(f"pairs={len(PAIRS)}  pos_rate={y_all.mean():.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(skf.split(PAIRS, y_all))

    rows = []
    for fold_id in range(args.folds):
        tr_idx, te_idx = splits[fold_id]
        print(f"\n=== FOLD {fold_id}  (train={len(tr_idx)} test={len(te_idx)}) ===")
        m, n_model, n_fb, scores = run_fold(PAIRS, GIN, fold_id, tr_idx, te_idx, seed=args.seed)
        print(f"  model-covered test pairs: {n_model}   fallback(negative): {n_fb}")
        print(f"  AUROC={m['AUROC']:.4f}  AUPRC={m['AUPRC']:.4f}  MCC={m['MCC']:.4f}  F1={m['F1']:.4f}")
        rows.append({"fold": fold_id, "n_model": n_model, "n_fallback": n_fb,
                     **{k: round(float(v), 4) for k, v in m.items()}})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    if len(df) > 1:
        print("\n=== summary across folds (mean ± std) ===")
        for k in ["AUROC", "AUPRC", "MCC", "F1"]:
            print(f"  {k}: {df[k].mean():.4f} ± {df[k].std():.4f}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
