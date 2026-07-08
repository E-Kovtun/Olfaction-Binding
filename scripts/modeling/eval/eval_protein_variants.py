"""Test-run XGBoost baseline across all protein-embedding variants.

Variants compared (molecule GIN  ||  protein vector):
  mean        — ESM-2 mean-pool over ALL residues            (1280-d)
  pocket22    — ESM-2 mean-pool over the 22 BW pocket pos.   (1280-d)
  mlp22       — random MLP 1280->64 per BW pos, concatenated (1408-d)
  concat22    — raw concatenation of the 22 BW-pos embeddings (22*1280=28160-d)
  ecl2_hydro  — ESM-2 mean-pool over ECL2 (hydrophobicity)   (1280-d)
  ecl2_bw     — ESM-2 mean-pool over ECL2 (BW TM4/TM5)       (1280-d)
  background  — ESM-2 mean-pool over residues NOT in BW-22 ∪ ECL2 (1280-d)
  random      — per-row Gaussian noise floor (control)       (1280-d)

NOTE: the ecl2_hydro/ecl2_bw source npz files (esm2_650m_ecl2_*_curated.npz)
were removed from data/embeddings/proteins/ — the ECL2 investigation showed no
edge over background/pocket (see protein-embedding-variants-boost.md), so we
stopped maintaining that derived data. The variants stay documented here for
context; build_variants() skips them gracefully (with a printed note) if the
files are absent instead of failing.

Runs head=boost across 3 splits (stratified / group_molecule / group_receptor).

    uv run python scripts/modeling/eval/eval_protein_variants.py
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
from orbind.baselines import run_one

DATA = ROOT / "data"
SPLITS = ["stratified", "group_molecule", "group_receptor"]
SEED = 42

POCKET_PFXS = [
    "2.53", "2.57", "3.28", "3.29", "3.32", "3.33", "3.36", "3.37",
    "4.57", "4.61", "5.38", "5.39", "5.42", "5.43", "5.46",
    "6.44", "6.48", "6.51", "6.52", "6.55", "7.35", "7.39",
]


def build_variants() -> dict[str, dict]:
    """Return {variant_name: {receptor_seq: vector}}."""
    prot = DATA / "embeddings" / "proteins"
    V: dict[str, dict] = {}

    # full (780 receptors) is the single source of truth; restrict to the
    # curated 409 here (bw_pocket_positions_curated.csv is curated-only anyway)
    # so we don't waste work deriving pocket/mlp22/concat22/background for
    # receptors this script's PAIRS (pairs_curated.csv) never uses.
    mean_full = D.load_npz_dict(prot / "esm2_650m_mean_full.npz")
    curated_recs = set(pd.read_csv(DATA / "processed" / "pairs_curated.csv")["receptor"])
    V["mean"] = {k: v for k, v in mean_full.items() if k in curated_recs}

    # ecl2_hydro/ecl2_bw source npz removed (direction closed, see module
    # docstring) — skip gracefully rather than error if they're absent.
    for name, fname in [("ecl2_hydro", "esm2_650m_ecl2_mean_curated.npz"),
                        ("ecl2_bw", "esm2_650m_ecl2_bw_mean_curated.npz")]:
        fp = prot / fname
        if fp.exists():
            V[name] = D.load_npz_dict(fp)
        else:
            print(f"  {name}: {fname} not found (removed, direction closed) — skipping")
    keys = list(V["mean"].keys())

    # --- per-residue derived variants -------------------------------------
    pr_npz = np.load(str(prot / "esm2_650m_per_residue_full.npz"), allow_pickle=False)
    pos_df = pd.read_csv(DATA / "processed" / "bw_pocket_positions_curated.csv",
                         index_col="receptor")

    # random MLP 1280 -> 256 -> 64, He-init, seed 42
    MLP_OUT = 64
    rng = np.random.RandomState(42)
    W1 = (rng.randn(1280, 256) * np.sqrt(2 / 1280)).astype(np.float32); b1 = np.zeros(256, np.float32)
    W2 = (rng.randn(256, MLP_OUT) * np.sqrt(2 / 256)).astype(np.float32); b2 = np.zeros(MLP_OUT, np.float32)
    def mlp(x):  # [N,1280] -> [N,64]
        return np.maximum(0, np.maximum(0, x @ W1 + b1) @ W2 + b2)
    bw_cols = pos_df.columns.tolist()

    # ECL2 boundaries from BW numbering
    bw = pd.read_csv(DATA / "processed" / "bw_numbering_curated.csv").dropna(subset=["seq_pos"])
    bw["helix"] = bw["bw_number"].str.split(".").str[0]
    bw["seq_pos"] = bw["seq_pos"].astype(int)
    tm4e = bw[bw["helix"] == "4"].groupby("receptor")["seq_pos"].max()
    tm5s = bw[bw["helix"] == "5"].groupby("receptor")["seq_pos"].min()

    pocket, mlp22, concat22, background = {}, {}, {}, {}
    n_bg = []
    for k in keys:
        if k not in pr_npz.files:
            pocket[k] = V["mean"][k]
            mlp22[k] = mlp(np.tile(V["mean"][k], (len(bw_cols), 1))).ravel()
            concat22[k] = np.tile(V["mean"][k], (len(bw_cols), 1)).ravel()
            background[k] = V["mean"][k]; continue
        mat = pr_npz[k].astype(np.float32); L = len(mat)

        # pocket positions
        valid = []
        if k in pos_df.index:
            valid = [int(v) for v in pos_df.loc[k].dropna().astype(int).values if 0 <= v < L]
        fallback = mat[valid].mean(0) if valid else mat.mean(0)
        pocket[k] = fallback

        # per-position embeddings (or fallback) over 22 BW positions
        embs = []
        if k in pos_df.index:
            row = pos_df.loc[k]
            for col in bw_cols:
                raw = row[col]
                idx = int(raw) if pd.notna(raw) and 0 <= int(raw) < L else -1
                embs.append(mat[idx] if idx >= 0 else fallback)
        else:
            embs = [fallback] * len(bw_cols)
        pocket_mat = np.stack(embs)            # [22, 1280]
        mlp22[k]    = mlp(pocket_mat).ravel()  # [22*64   = 1408]
        concat22[k] = pocket_mat.ravel()       # [22*1280 = 28160]

        # background = all residues except BW-22 ∪ ECL2
        excl = set(valid)
        if k in tm4e.index and k in tm5s.index:
            s0, e0 = int(tm4e[k]) + 1, int(tm5s[k]) - 1
            if e0 >= s0:
                excl.update(range(s0, e0 + 1))
        incl = [i for i in range(L) if i not in excl]
        background[k] = mat[incl].mean(0) if incl else mat.mean(0)
        n_bg.append(len(incl))

    V["pocket22"]   = pocket
    V["mlp22"]      = mlp22
    V["concat22"]   = concat22
    V["background"] = background
    print(f"background residues kept: min={min(n_bg)} mean={sum(n_bg)/len(n_bg):.0f} max={max(n_bg)}")
    for name, d in V.items():
        print(f"  {name:<12} {len(d)} receptors  dim={next(iter(d.values())).shape[0]}")
    return V


def main():
    pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
    gin   = D.load_npz_dict(DATA / "embeddings" / "molecules" / "gin_supervised_contextpred.npz")
    print(f"pairs={len(pairs)}  molecules={len(gin)}")

    V = build_variants()

    order = [n for n in ["random", "mean", "pocket22", "mlp22", "concat22",
                        "ecl2_hydro", "ecl2_bw", "background"] if n == "random" or n in V]
    rows = []
    for name in order:
        is_random = name == "random"
        prot = V["mean"] if is_random else V[name]
        for split in SPLITS:
            m, info = run_one(pairs, prot, gin, "boost", split,
                              random_prot=is_random, seed=SEED)
            rows.append({"variant": name, "split": split,
                         **{k: round(float(v), 4) for k, v in m.items()}, **info})
            print(f"  [{name:<11} | {split:<15}] "
                  f"AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} MCC={m['MCC']:.3f}")

    res = pd.DataFrame(rows)
    out = ROOT / "results" / "analysis" / "protein_variants_boost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + "=" * 70)
    for metric in ["AUROC", "AUPRC", "MCC", "F1"]:
        piv = res.pivot_table(index="variant", columns="split", values=metric)
        piv = piv.reindex(order)[SPLITS]
        print(f"\n### {metric}")
        print(piv.round(3).to_string())
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
