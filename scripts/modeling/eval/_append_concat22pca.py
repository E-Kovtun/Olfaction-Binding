"""concat22_pca: PCA-reduce the per-position 1280-d pocket space, then concat.

  1. Pool all 22*409 pocket-position embeddings -> cloud [8998, 1280].
  2. PCA on that cloud; keep first N components that explain >= 95% variance.
  3. Each receptor = concatenation of its 22 reduced segments -> dim 22*N.

Small enough to boost (unlike the 28160-d raw concat22). Appends to
results/protein_variants_boost.csv as variant 'concat22_pca'.

    uv run python scripts/modeling/eval/_append_concat22pca.py
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
from orbind.baselines import run_one
from eval_protein_variants import build_variants, DATA, SPLITS, SEED

VAR_KEEP = 0.95

pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
gin   = D.load_npz_dict(DATA / "embeddings" / "molecules" / "gin_supervised_contextpred.npz")

V = build_variants()
keys = list(V["mean"].keys())
concat = V["concat22"]                       # seq -> [28160]
for kk in list(V.keys()):
    if kk != "concat22":
        del V[kk]
del V; gc.collect()

# pooled pocket cloud [409*22, 1280]
P_all = np.stack([concat[k].reshape(22, 1280) for k in keys]).astype(np.float64)  # [409,22,1280]
flat  = P_all.reshape(-1, 1280)                                                   # [8998,1280]
mu    = flat.mean(0, keepdims=True)
Xc    = flat - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var = S ** 2
cum = np.cumsum(var) / var.sum()
N = int(np.searchsorted(cum, VAR_KEEP) + 1)
print(f"pooled pocket cloud: {flat.shape[0]} векторов x 1280")
print(f"N компонент для {VAR_KEEP:.0%} дисперсии: {N}  ->  итоговый dim = 22*{N} = {22 * N}")

comps = Vt[:N]                                                       # [N, 1280]
red = ((P_all.reshape(-1, 1280) - mu) @ comps.T).reshape(len(keys), 22 * N).astype(np.float32)
prot = dict(zip(keys, red))
del P_all, flat, Xc, U, S, Vt, red; gc.collect()

rows = []
for split in SPLITS:
    m, info = run_one(pairs, prot, gin, "boost", split, seed=SEED)
    rows.append({"variant": "concat22_pca", "split": split,
                 **{k: round(float(v), 4) for k, v in m.items()}, **info})
    print(f"  [concat22_pca | {split:<15}] AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} MCC={m['MCC']:.3f}")

new = pd.DataFrame(rows)
out = ROOT / "results" / "analysis" / "protein_variants_boost.csv"
old = pd.read_csv(out)
old = old[old["variant"] != "concat22_pca"]
pd.concat([old, new], ignore_index=True).to_csv(out, index=False)
print(f"\nN={N}  dim={22 * N}  appended {len(new)} rows -> {out}")
