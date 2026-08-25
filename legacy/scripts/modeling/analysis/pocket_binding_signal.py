"""Does pocket / ECL2 divergence predict binding-profile divergence, AFTER
controlling for overall sequence similarity (phylogeny)?

For every receptor pair (within the same OR subfamily, so global similarity is
~constant by construction) we compute:
  d_bind   = 1 - Jaccard of their bound-odorant sets over commonly-tested molecules
  d_pocket = cosine distance of the BW-22 pocket ESM embedding
  d_pocketH= Hamming distance of the 22 BW pocket residues (identity-based)
  d_ecl2   = cosine distance of the ECL2 (BW) ESM embedding
  d_bg     = cosine distance of the background embedding (residues outside BW-22 ∪ ECL2)
  d_global = cosine distance of the mean (whole-sequence) ESM embedding  [the control]

Then partial Spearman correlation with d_bind, controlling for d_global.

Falsifiable prediction (structure is necessary):
    rho(d_pocket, d_bind | d_global) > 0   and   > rho(d_bg, d_bind | d_global) ~ 0

    uv run python legacy/scripts/modeling/analysis/pocket_binding_signal.py
"""
from __future__ import annotations
import pathlib, re, sys, gc, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from orbind import dataset as D
from eval_protein_variants import build_variants, DATA

K_SHARED = 5          # min commonly-tested molecules per receptor pair
WITHIN_FAMILY = True  # restrict to same-subfamily pairs


def _family(slug: str) -> str:
    slug = slug.replace("_human", "")
    m = re.match(r"^(or?\d+)", slug)
    return m.group(1).upper() if m else slug.upper()


def partial_spearman(x, y, z):
    """rho(x, y | z) via Spearman correlations."""
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / denom if denom > 0 else np.nan, rxy


def cos_dist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return 1.0 - float(u @ v) / (nu * nv)


def main():
    pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
    print(f"pairs={len(pairs)}  receptors={pairs['receptor'].nunique()}")

    # binding profile per receptor: {inchikey: label}
    prof = {r: dict(zip(g["inchikey"], g["label"]))
            for r, g in pairs.groupby("receptor")}
    posset = {r: {m for m, l in d.items() if l == 1} for r, d in prof.items()}

    # embeddings
    V = build_variants()
    mean, pocket, bg = V["mean"], V["pocket22"], V["background"]
    ecl2 = V["ecl2_bw"]
    recs = list(mean.keys())
    del V; gc.collect()

    # BW-22 residue matrix for Hamming pocket distance
    bwm = pd.read_csv(DATA / "processed" / "bw_pocket_matrix_curated.csv", index_col="receptor")

    # family
    ref = pd.read_csv(DATA / "processed" / "bw_ref_used_curated.csv")
    fam = {r: _family(s) for r, s in zip(ref["receptor"], ref["ref_entry"])}

    # candidate pairs
    if WITHIN_FAMILY:
        by_fam: dict[str, list] = {}
        for r in recs:
            by_fam.setdefault(fam.get(r, "?"), []).append(r)
        cand = []
        for f, members in by_fam.items():
            cand.extend(itertools.combinations(members, 2))
        print(f"within-family pairs (candidate): {len(cand)}  "
              f"families={sum(1 for m in by_fam.values() if len(m) >= 2)}")
    else:
        cand = list(itertools.combinations(recs, 2))

    rows = []
    for a, b in cand:
        da, db = prof[a], prof[b]
        shared = da.keys() & db.keys()
        if len(shared) < K_SHARED:
            continue
        pa = posset[a] & shared
        pb = posset[b] & shared
        union = pa | pb
        if not union:                          # no binding info in shared set
            continue
        d_bind = 1.0 - len(pa & pb) / len(union)

        # Hamming over BW-22 residues (ignore NaN positions)
        if a in bwm.index and b in bwm.index:
            ra, rb = bwm.loc[a].values, bwm.loc[b].values
            ok = pd.notna(ra) & pd.notna(rb)
            d_pH = float((ra[ok] != rb[ok]).mean()) if ok.sum() else np.nan
        else:
            d_pH = np.nan

        rows.append((
            d_bind,
            cos_dist(pocket[a], pocket[b]),
            d_pH,
            cos_dist(ecl2[a], ecl2[b]),
            cos_dist(bg[a], bg[b]),
            cos_dist(mean[a], mean[b]),
            len(shared), len(union),
        ))

    cols = ["d_bind", "d_pocket", "d_pocketH", "d_ecl2", "d_bg", "d_global", "n_shared", "n_union"]
    df = pd.DataFrame(rows, columns=cols).dropna(
        subset=["d_bind", "d_pocket", "d_ecl2", "d_bg", "d_global"])
    print(f"usable pairs (>= {K_SHARED} shared, >=1 positive): {len(df)}")
    print(f"  median shared molecules: {df['n_shared'].median():.0f}")
    print(f"  d_bind: mean={df['d_bind'].mean():.3f}  (0=identical odorant set, 1=disjoint)")

    print("\n=== Spearman with d_bind (raw  /  partial controlling d_global) ===")
    print(f"{'predictor':<12} {'raw rho':>9} {'partial rho':>13} {'n':>7}")
    for col in ["d_pocket", "d_pocketH", "d_ecl2", "d_bg", "d_global"]:
        sub = df.dropna(subset=[col])
        if col == "d_global":
            raw = spearmanr(sub[col], sub["d_bind"]).correlation
            print(f"{col:<12} {raw:>9.3f} {'(control)':>13} {len(sub):>7}")
            continue
        prho, raw = partial_spearman(sub[col], sub["d_bind"], sub["d_global"])
        print(f"{col:<12} {raw:>9.3f} {prho:>13.3f} {len(sub):>7}")

    out = ROOT / "results" / "analysis" / "pocket_binding_signal.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved pair-level table -> {out}")
    print("\nИнтерпретация: structure 'необходима', если partial rho(d_pocket/d_ecl2) > 0")
    print("и заметно больше partial rho(d_bg) ~ 0.")


if __name__ == "__main__":
    main()
