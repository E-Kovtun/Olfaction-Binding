"""No-graph XGBoost baseline on full_full, for BOTH regimes (fold 1, seed 42).

This is the grey "no graph" bar in graph_evaluation_full_full.ipynb. Features are
the raw node embeddings with NO message passing: [ChemBERTa mol || ESM-mean prot].
Splits come from orbind.lorax.build, so they are byte-identical to the splits the
graph models train on — an apples-to-apples "does the graph add anything?" bar.

  * transductive       — reproduces (≈) the LORAX boost@0.5 number, fold 1.
  * inductive_molecule — cold-molecule baseline (no LORAX equivalent exists).

  uv run python scripts/modeling/eval/eval_full_full_baseline.py

Writes results/full_full/tables/baselines.csv.
"""
import argparse, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import lorax as L
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def _xy(split, Xm, Xp):
    idx = np.concatenate([split["pos"], split["neg"]], axis=0)
    y = np.concatenate([np.ones(len(split["pos"])), np.zeros(len(split["neg"]))])
    X = np.concatenate([Xm[idx[:, 0]], Xp[idx[:, 1]]], axis=1)
    return X.astype(np.float32), y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    esm, chem = L.load_embeddings()
    rows = []
    for regime in ("transductive", "inductive_molecule"):
        print(f"\n[{regime}]")
        Xm, Xp, splits = L.build(regime, args.fold, esm, chem, seed=args.seed)
        Xm, Xp = Xm.numpy(), Xp.numpy()
        Xtr, ytr = _xy(splits["train"], Xm, Xp)
        Xte, yte = _xy(splits["test"], Xm, Xp)
        scores = train_boost(Xtr, ytr, Xte, seed=args.seed)
        m = metrics(yte, scores)
        print("  no-graph boost EC50-TEST " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
        rows.append({"regime": regime, **{k: round(m[k], 4) for k in METRICS}})

    out = pd.DataFrame(rows)
    od = _root / "results" / "full_full" / "tables"; od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "baselines.csv", index=False)
    print("\nsaved -> results/full_full/tables/baselines.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
