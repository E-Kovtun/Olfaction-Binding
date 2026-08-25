"""Run our plain XGBoost baseline on the LORAX/Hladis M2OR splits.

Reproduces the LORAX Table 3 evaluation protocol (McConachie et al., ICLR 2026)
with our own boosting head so the numbers are directly comparable:

  * Their data, their 5 random folds (data/splits_indexes/lorax_m2or/rand_split_*).
  * Train on the FULL noisy mix (primary + secondary + ec50);
    TEST only on the held-out EC50 pairs (~22% positive) — exactly their split.
  * Features: putative ESM-1b 650M mean-pooled protein  ||  ChemBERTa-77M molecule
    (their provided embeddings; the paper shows the molecule encoder is
    interchangeable).
  * Head: our `train_boost` — a single XGBoost, NO Hladis quality/class/pair
    weighting, NO hyperopt ensemble.

Models are trained on `train` only; `val` is used purely to pick decision
thresholds (so threshold selection sees no test data). Variants reported:
  * boost@0.5        — standard head (scale_pos_weight = neg/pos), threshold 0.5.
  * boost@bestF1     — same model, threshold maximizing F1 on val.
  * boost@bestMCC    — same model, threshold maximizing MCC on val.
  * boost_noweight@0.5 — strictly unweighted (scale_pos_weight = 1), threshold 0.5.

AUROC and AUPRC are threshold-free (identical across the boost@* rows).

  uv run python legacy/scripts/modeling/eval/eval_on_lorax_splits.py

Writes results/lorax/lorax_compare.csv.
"""
import pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             matthews_corrcoef, f1_score, precision_score, recall_score)

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind.legacy import lorax as L

DATA = L.LORAX
METRIC_KEYS = ["AUROC", "AUPRC", "precision", "recall", "F1", "MCC"]


def _train_boost(Xtr, ytr, Xte, seed=42, weight=True):
    import xgboost as xgb
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if weight else 1.0
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="aucpr", tree_method="hist", n_jobs=-1,
                            random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _best_threshold(y, scores, objective="f1"):
    """Threshold in (0,1) maximizing F1 or MCC on (y, scores)."""
    fn = f1_score if objective == "f1" else matthews_corrcoef
    grid = np.linspace(0.01, 0.99, 99)
    vals = [fn(y, (scores >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(vals))])


def _metrics_at(y, scores, thr):
    pred = (scores >= thr).astype(int)
    return {"AUROC": roc_auc_score(y, scores),
            "AUPRC": average_precision_score(y, scores),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred)}


def _load_embeddings():
    return L.load_embeddings()

def _featurize(df, esm, chem):
    """Build X=[chemberta || esm], y from a split dataframe; drop rows lacking embeddings."""
    keep = df["Protein sequence"].isin(esm) & df["SMILES"].isin(chem)
    d = df[keep]
    X = np.stack([np.concatenate([chem[s], esm[p]])
                  for s, p in zip(d["SMILES"], d["Protein sequence"])]).astype(np.float32)
    y = d["output"].to_numpy(dtype=np.float32)
    return X, y, int((~keep).sum())


def main():
    esm, chem = _load_embeddings()
    print(f"ESM proteins={len(esm)} | ChemBERTa mols={len(chem)}")

    # variant -> list of per-fold metric dicts
    variants = ["boost@0.5", "boost@bestF1", "boost@bestMCC", "boost_noweight@0.5"]
    acc = {v: [] for v in variants}

    for s in range(1, 6):
        base = DATA / f"rand_split_{s}"
        tr = pd.read_csv(base / "train_df.csv")          # train ONLY
        va = pd.read_csv(base / "val_df.csv")            # for threshold pick
        te = pd.read_csv(base / "test_df.csv")           # EC50 test
        Xtr, ytr, _ = _featurize(tr, esm, chem)
        Xva, yva, _ = _featurize(va, esm, chem)
        Xte, yte, drop_te = _featurize(te, esm, chem)

        # weighted boost: one model, three thresholds (0.5 / bestF1 / bestMCC on val)
        sc_va = _train_boost(Xtr, ytr, Xva, seed=42, weight=True)
        sc_te = _train_boost(Xtr, ytr, Xte, seed=42, weight=True)
        thr_f1  = _best_threshold(yva, sc_va, "f1")
        thr_mcc = _best_threshold(yva, sc_va, "mcc")
        acc["boost@0.5"].append(_metrics_at(yte, sc_te, 0.5))
        acc["boost@bestF1"].append(_metrics_at(yte, sc_te, thr_f1))
        acc["boost@bestMCC"].append(_metrics_at(yte, sc_te, thr_mcc))

        # unweighted boost @0.5
        sc_te_nw = _train_boost(Xtr, ytr, Xte, seed=42, weight=False)
        acc["boost_noweight@0.5"].append(_metrics_at(yte, sc_te_nw, 0.5))

        m = acc["boost@0.5"][-1]
        print(f"  fold{s}: train={len(ytr)} | test={len(yte)} "
              f"(pos {yte.mean():.3f}, drop {drop_te}) | thr_F1={thr_f1:.2f} thr_MCC={thr_mcc:.2f} | "
              f"AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} F1={m['F1']:.3f} MCC={m['MCC']:.3f}")

    rows = []
    for v in variants:
        per_fold = acc[v]
        mean = {k: float(np.mean([f[k] for f in per_fold])) for k in METRIC_KEYS}
        std  = {k: float(np.std([f[k] for f in per_fold]))  for k in METRIC_KEYS}
        rows.append({"method": v,
                     **{k: round(mean[k], 4) for k in METRIC_KEYS},
                     **{f"{k}_std": round(std[k], 4) for k in METRIC_KEYS}})

    out = pd.DataFrame(rows)
    res_dir = _root / "results" / "full_full" / "article_results"; res_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(res_dir / "lorax_compare.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== our boost on LORAX splits (mean over 5 folds, EC50 test) ===")
    print(out[["method", *METRIC_KEYS]].to_string(index=False))
    print(f"\nsaved -> results/full_full/article_results/lorax_compare.csv")


if __name__ == "__main__":
    main()
