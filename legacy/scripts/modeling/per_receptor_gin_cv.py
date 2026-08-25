"""Per-receptor GIN-only XGBoost: how predictable is bind/no-bind from molecule
chemistry alone, for EACH receptor. Collects metrics + class counts so we can
map the receptor population by (data quantity) x (predictability).

Stratified CV, adaptive folds: 5 if min-class >= 5, else 3 if >= 3, else skip
(too few of the minority class to cross-validate). Out-of-fold predictions.

Incremental + resumable -> notebooks/cache/per_receptor_gin_cv.csv

    uv run python legacy/scripts/modeling/per_receptor_gin_cv.py
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

OUT = ROOT / "notebooks" / "cache" / "per_receptor_gin_cv.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    GIN = D.load_npz_dict(ROOT / "data" / "embeddings" / "molecules" / "gin_supervised_contextpred_all_m2or.npz")
    PAIRS = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    PAIRS = PAIRS[PAIRS["inchikey"].isin(GIN)]

    g = PAIRS.groupby("receptor")["label"]
    RSTAT = pd.DataFrame({"n_pos": g.apply(lambda s: int((s == 1).sum())),
                          "n_neg": g.apply(lambda s: int((s == 0).sum()))})
    RSTAT = RSTAT.sort_values(["n_pos", "n_neg"], ascending=False, kind="stable").reset_index()

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT)["idx"].tolist())

    for idx, row in RSTAT.iterrows():
        if idx in done:
            continue
        seq = row["receptor"]
        sub = PAIRS[PAIRS["receptor"] == seq]
        y = sub["label"].to_numpy(np.float32)
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        minc = min(n_pos, n_neg)
        folds = 5 if minc >= 5 else (3 if minc >= 3 else 0)

        rec = {"idx": int(idx), "n": len(y), "n_pos": n_pos, "n_neg": n_neg,
               "pos_rate": round(float(y.mean()), 4), "folds": folds,
               "AUROC": np.nan, "AUPRC": np.nan, "MCC": np.nan, "F1": np.nan}
        if folds:
            X = np.stack([GIN[i] for i in sub["inchikey"]]).astype(np.float32)
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(folds, shuffle=True, random_state=42).split(X, y):
                oof[te] = train_boost(X[tr], y[tr], X[te], seed=42)
            m = D.metrics(y, oof)
            rec.update({k: round(float(m[k]), 4) for k in ["AUROC", "AUPRC", "MCC", "F1"]})

        pd.DataFrame([rec]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        done.add(idx)
        if (idx + 1) % 25 == 0 or idx == len(RSTAT) - 1:
            print(f"  {idx + 1}/{len(RSTAT)} done", flush=True)

    df = pd.read_csv(OUT)
    ev = df[df["folds"] > 0]
    print(f"\nreceptors: {len(df)}  evaluable(CV): {len(ev)}  too-few-data: {int((df.folds==0).sum())}")
    print(f"AUROC median={ev.AUROC.median():.3f}  frac>0.7={ (ev.AUROC>0.7).mean():.2f}  "
          f"frac>0.8={(ev.AUROC>0.8).mean():.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
