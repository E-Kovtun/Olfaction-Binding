"""Compute ONLY the concat22 variant (3 splits) and append to the results CSV,
reusing build_variants() from eval_protein_variants. Avoids recomputing the
other 7 variants already in results/protein_variants_boost.csv.

concat22 is 28160-d -> the default XGBoost (max_bin=256) OOMs (~6 GB histogram).
We free the other variants + per-residue npz first and drop max_bin to 64.

    uv run python legacy/scripts/modeling/eval/_append_concat22.py
"""
from __future__ import annotations
import pathlib, sys, gc
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from orbind import dataset as D
from eval_protein_variants import build_variants, DATA, SPLITS, SEED

MAX_BIN = 64   # default is 256; lowered to fit the 28160-d matrix in RAM


def boost_lowmem(Xtr, ytr, Xte, seed=42, max_bin=MAX_BIN):
    import xgboost as xgb
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="aucpr", tree_method="hist", max_bin=max_bin,
                            n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
gin   = D.load_npz_dict(DATA / "embeddings" / "molecules" / "gin_supervised_contextpred_all_m2or.npz")

V = build_variants()
prot = V["concat22"]
print(f"concat22 dim = {next(iter(prot.values())).shape[0]}  (max_bin={MAX_BIN})")
# free everything else (per-residue npz handle + other variant dicts) before fit
for kk in list(V.keys()):
    if kk != "concat22":
        del V[kk]
del V; gc.collect()

# assemble [molecule || protein] once
mask = pairs["receptor"].isin(prot) & pairs["inchikey"].isin(gin)
p    = pairs[mask].reset_index(drop=True)
Xm = np.stack([gin[i] for i in p["inchikey"]]).astype(np.float32)
Xp = np.stack([prot[r] for r in p["receptor"]]).astype(np.float32)
X  = np.concatenate([Xm, Xp], axis=1)
y  = p["label"].to_numpy().astype(np.float32)
del Xm, Xp, prot; gc.collect()
print(f"X={X.shape}  positives={int(y.sum())}/{len(y)}")

rows = []
for split in SPLITS:
    tr, te = D.split(p, y, kind=split, test_size=0.2, seed=SEED)
    pred = boost_lowmem(X[tr], y[tr], X[te], seed=SEED)
    m = D.metrics(y[te], pred)
    info = {"train": int(tr.sum()), "test": int(te.sum()), "test_pos": int(y[te].sum())}
    rows.append({"variant": "concat22", "split": split,
                 **{k: round(float(v), 4) for k, v in m.items()}, **info})
    print(f"  [concat22 | {split:<15}] AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} MCC={m['MCC']:.3f}")

new = pd.DataFrame(rows)
out = ROOT / "results" / "analysis" / "protein_variants_boost.csv"
old = pd.read_csv(out)
old = old[old["variant"] != "concat22"]            # replace if rerun
pd.concat([old, new], ignore_index=True).to_csv(out, index=False)
print(f"\nappended {len(new)} rows -> {out}")
