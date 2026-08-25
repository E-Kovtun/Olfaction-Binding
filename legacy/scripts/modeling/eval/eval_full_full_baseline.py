"""No-graph XGBoost baseline on full_full with repeat-level results and 95% CI.

Features are the original full-size node embeddings, with no message passing:
[ChemBERTa-77M 384-d molecule || putative ESM-1b 1280-d protein].

Resampling differs by regime for a deliberate reason:
  * transductive: the five genuine LORAX random folds;
  * inductive_molecule: five cold-molecule split seeds. Merely changing the
    nominal LORAX fold does not create an independent cold split because the
    builder reconstructs the same deduplicated full pair pool.

Writes:
  results/full_full/tables/baseline_runs.csv
  results/full_full/tables/baselines_ci95.csv
  results/full_full/tables/baselines.csv  (backward-compatible mean table)
"""
import argparse, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import t as student_t

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind.legacy import lorax as L
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]


def _xy(split, Xm, Xp):
    idx = np.concatenate([split["pos"], split["neg"]], axis=0)
    y = np.concatenate([np.ones(len(split["pos"])), np.zeros(len(split["neg"]))])
    X = np.concatenate([Xm[idx[:, 0]], Xp[idx[:, 1]]], axis=1)
    return X.astype(np.float32), y.astype(np.float32)


def _ci95(values):
    x = np.asarray(values, dtype=float); n = len(x); mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    half = float(student_t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return mean, sd, mean-half, mean+half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--cold-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--boost-seed", type=int, default=42)
    args = ap.parse_args()

    esm, chem = L.load_embeddings()
    print(f"full baseline embeddings: protein={len(esm)} x {next(iter(esm.values())).shape[0]} | "
          f"molecule={len(chem)} x {next(iter(chem.values())).shape[0]}")
    rows = []

    for fold in args.folds:
        Xm, Xp, splits = L.build("transductive", fold, esm, chem, seed=args.boost_seed)
        Xtr, ytr = _xy(splits["train"], Xm.numpy(), Xp.numpy())
        Xte, yte = _xy(splits["test"], Xm.numpy(), Xp.numpy())
        score = train_boost(Xtr, ytr, Xte, seed=args.boost_seed)
        m = metrics(yte, score)
        rows.append({"regime":"transductive", "repeat":fold, "resampling":"lorax_fold",
                     "split_seed":np.nan, **{k:float(m[k]) for k in METRICS}})
        print(f"transductive fold {fold}: " + " ".join(f"{k}={m[k]:.3f}" for k in METRICS))

    # fold=1 is only a source container here; independence comes from cold split seed.
    for seed in args.cold_seeds:
        Xm, Xp, splits = L.build("inductive_molecule", 1, esm, chem, seed=seed)
        Xtr, ytr = _xy(splits["train"], Xm.numpy(), Xp.numpy())
        Xte, yte = _xy(splits["test"], Xm.numpy(), Xp.numpy())
        score = train_boost(Xtr, ytr, Xte, seed=args.boost_seed)
        m = metrics(yte, score)
        rows.append({"regime":"inductive_molecule", "repeat":seed, "resampling":"cold_split_seed",
                     "split_seed":seed, **{k:float(m[k]) for k in METRICS}})
        print(f"inductive cold seed {seed}: " + " ".join(f"{k}={m[k]:.3f}" for k in METRICS))

    runs = pd.DataFrame(rows)
    ci_rows = []
    for regime, group in runs.groupby("regime", sort=False):
        for metric in METRICS:
            mean, sd, low, high = _ci95(group[metric])
            ci_rows.append({"regime":regime, "metric":metric, "mean":mean, "sd":sd,
                            "ci_low":low, "ci_high":high, "n":len(group),
                            "resampling":group.resampling.iloc[0],
                            "features":"full ChemBERTa384 + putative ESM1b1280"})
    ci = pd.DataFrame(ci_rows)
    legacy = runs.groupby("regime", sort=False)[METRICS].mean().reset_index()

    out = _root / "results/full_full/tables"; out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "baseline_runs.csv", index=False)
    ci.to_csv(out / "baselines_ci95.csv", index=False)
    legacy.to_csv(out / "baselines.csv", index=False)
    print("\nsaved baseline_runs.csv, baselines_ci95.csv, baselines.csv")
    print(ci.to_string(index=False))


if __name__ == "__main__":
    main()
