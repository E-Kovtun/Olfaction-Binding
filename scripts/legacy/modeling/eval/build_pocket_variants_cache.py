"""Derive the pocket/mlp22/background receptor-block variants from the heavy
per-residue ESM npz ONCE and cache them in a tiny (~5 MB) npz.

Rationale: pocket22/mlp22/background are all small per-receptor vectors, but
deriving them needs the 1.86 GB `esm2_650m_per_residue_full_full.npz`, which is
not a stable dependency (it gets deleted to save space). After this cache exists,
the receptor-representation probe notebook reads it and no longer needs the heavy
file at all.

    uv run python scripts/modeling/eval/build_pocket_variants_cache.py
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from orbind import dataset as D

DATA    = ROOT / "data"
PER_RES = DATA / "embeddings" / "proteins" / "esm2_650m_per_residue_full_full.npz"
CACHE   = DATA / "embeddings" / "proteins" / "receptor_pocket_variants_cache.npz"


def main():
    if not PER_RES.exists():
        raise FileNotFoundError(f"{PER_RES} absent — nothing to derive the cache from")

    pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
    mean  = {k: v for k, v in D.load_npz_dict(DATA / "embeddings" / "proteins" / "esm2_650m_mean.npz").items()
             if k in set(pairs["receptor"])}
    keys  = list(mean.keys())

    pr = np.load(str(PER_RES), allow_pickle=False)
    pos_df = pd.read_csv(DATA / "processed" / "bw_pocket_positions_curated.csv", index_col="receptor")
    bw_cols = pos_df.columns.tolist()

    rng = np.random.RandomState(42)
    W1 = (rng.randn(1280, 256) * np.sqrt(2 / 1280)).astype(np.float32); b1 = np.zeros(256, np.float32)
    W2 = (rng.randn(256, 64)   * np.sqrt(2 / 256 )).astype(np.float32); b2 = np.zeros(64,  np.float32)
    mlp = lambda x: np.maximum(0, np.maximum(0, x @ W1 + b1) @ W2 + b2)

    bw = pd.read_csv(DATA / "processed" / "bw_numbering_curated.csv").dropna(subset=["seq_pos"])
    bw["helix"] = bw["bw_number"].str.split(".").str[0]; bw["seq_pos"] = bw["seq_pos"].astype(int)
    tm4e = bw[bw["helix"] == "4"].groupby("receptor")["seq_pos"].max()
    tm5s = bw[bw["helix"] == "5"].groupby("receptor")["seq_pos"].min()

    pocket, mlp22, background, n_bg = [], [], [], []
    for k in keys:
        if k not in pr.files:
            fb = mean[k]
            pocket.append(fb); mlp22.append(mlp(np.tile(fb, (len(bw_cols), 1))).ravel()); background.append(fb)
            continue
        mat = pr[k].astype(np.float32); L = len(mat)
        valid = [int(v) for v in pos_df.loc[k].dropna().astype(int).values if 0 <= v < L] if k in pos_df.index else []
        fb = mat[valid].mean(0) if valid else mat.mean(0)
        pocket.append(fb)
        if k in pos_df.index:
            row = pos_df.loc[k]
            embs = []
            for col in bw_cols:
                raw = row[col]; idx = int(raw) if pd.notna(raw) and 0 <= int(raw) < L else -1
                embs.append(mat[idx] if idx >= 0 else fb)
        else:
            embs = [fb] * len(bw_cols)
        mlp22.append(mlp(np.stack(embs)).ravel())
        excl = set(valid)
        if k in tm4e.index and k in tm5s.index:
            s0, e0 = int(tm4e[k]) + 1, int(tm5s[k]) - 1
            if e0 >= s0: excl.update(range(s0, e0 + 1))
        incl = [i for i in range(L) if i not in excl]
        background.append(mat[incl].mean(0) if incl else mat.mean(0)); n_bg.append(len(incl))

    np.savez_compressed(CACHE, ids=np.array(keys),
                        pocket22=np.stack(pocket).astype(np.float32),
                        mlp22=np.stack(mlp22).astype(np.float32),
                        background=np.stack(background).astype(np.float32))
    print(f"cached {len(keys)} receptors -> {CACHE.name}  ({CACHE.stat().st_size/1e6:.1f} MB)")
    if n_bg:
        print(f"  background residues kept: min={min(n_bg)} mean={sum(n_bg)/len(n_bg):.0f} max={max(n_bg)}")


if __name__ == "__main__":
    main()
