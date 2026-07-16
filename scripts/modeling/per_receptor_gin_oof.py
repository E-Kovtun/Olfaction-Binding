"""Per-receptor GIN XGBoost + honest fallback, then score the WHOLE curated
dataset in the 'crooked transductive' regime and read the pooled AUROC.

Per receptor:
  - if both classes have >= 3 examples: GIN XGBoost, stratified CV (5 folds,
    or 3 if min-class in [3,5)), out-of-fold predictions  [MODEL]
  - else (too few of one class): leave-one-out receptor base rate
    (predict this receptor's label frequency computed WITHOUT the point itself)
    -- the "most frequent label from train", made honest by LOO.            [FALLBACK]

Every receptor is SEEN (its own model or its own base rate), molecules held out
within-receptor. Then pool all pair-level scores and compute a single AUROC.

Outputs: notebooks/cache/per_receptor_gin_oof.csv  (idx, inchikey, label, score, source)

    uv run python scripts/modeling/per_receptor_gin_oof.py
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import dataset as D
from orbind.baselines import train_boost

OOF = ROOT / "notebooks" / "cache" / "per_receptor_gin_oof.csv"
OOF.parent.mkdir(parents=True, exist_ok=True)


def loo_base_rate(y: np.ndarray) -> np.ndarray:
    """Leave-one-out label frequency: score_i = (sum(y) - y_i) / (n - 1)."""
    n = len(y)
    if n == 1:
        return np.full(1, float(y.mean()))
    return (y.sum() - y) / (n - 1)


def main():
    GIN = D.load_npz_dict(ROOT / "data" / "embeddings" / "molecules" / "gin_supervised_contextpred_all_m2or.npz")
    PAIRS = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    PAIRS = PAIRS[PAIRS["inchikey"].isin(GIN)]
    n_pairs_total = len(PAIRS)

    g = PAIRS.groupby("receptor")["label"]
    RSTAT = pd.DataFrame({"n_pos": g.apply(lambda s: int((s == 1).sum())),
                          "n_neg": g.apply(lambda s: int((s == 0).sum()))})
    RSTAT = RSTAT.sort_values(["n_pos", "n_neg"], ascending=False, kind="stable").reset_index()

    rows, per_rec_model = [], []
    n_model = n_fallback = 0
    for idx, row in RSTAT.iterrows():
        seq = row["receptor"]
        sub = PAIRS[PAIRS["receptor"] == seq].reset_index(drop=True)
        y = sub["label"].to_numpy(np.float32)
        minc = min(int(y.sum()), int((y == 0).sum()))
        folds = 5 if minc >= 5 else (3 if minc >= 3 else 0)

        if folds:
            X = np.stack([GIN[i] for i in sub["inchikey"]]).astype(np.float32)
            score = np.zeros(len(y))
            for tr, te in StratifiedKFold(folds, shuffle=True, random_state=42).split(X, y):
                score[te] = train_boost(X[tr], y[tr], X[te], seed=42)
            src = "model"; n_model += 1
            per_rec_model.append(D.metrics(y, score)["AUROC"])
        else:
            score = loo_base_rate(y)                 # honest-ish majority/base-rate
            src = "fallback"; n_fallback += 1

        for ik, lab, sc in zip(sub["inchikey"], y, score):
            rows.append((int(idx), ik, int(lab), float(sc), src))
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(RSTAT)}", flush=True)

    df = pd.DataFrame(rows, columns=["idx", "inchikey", "label", "score", "source"])
    df.to_csv(OOF, index=False)

    def au(d):
        return D.metrics(d["label"].to_numpy(), d["score"].to_numpy())

    allm = au(df)
    modm = au(df[df.source == "model"])
    fbm  = au(df[df.source == "fallback"])
    print("\n=============== per-receptor GIN + LOO-base-rate fallback ===============")
    print(f"receptors: model={n_model}  fallback={n_fallback}  |  "
          f"pairs: model={int((df.source=='model').sum())}  fallback={int((df.source=='fallback').sum())}  "
          f"total={len(df)}/{n_pairs_total}")
    print(f"pooled positive rate = {df['label'].mean():.3f}\n")
    print(f"POOLED AUROC  — ALL pairs           : {allm['AUROC']:.3f}   AUPRC {allm['AUPRC']:.3f}")
    print(f"POOLED AUROC  — model pairs only     : {modm['AUROC']:.3f}   AUPRC {modm['AUPRC']:.3f}")
    print(f"POOLED AUROC  — fallback pairs only  : {fbm['AUROC']:.3f}   AUPRC {fbm['AUPRC']:.3f}")
    print(f"MACRO mean per-receptor AUROC (model): {np.mean(per_rec_model):.3f}  "
          f"(median {np.median(per_rec_model):.3f})")
    print(f"-> {OOF}")


if __name__ == "__main__":
    main()
