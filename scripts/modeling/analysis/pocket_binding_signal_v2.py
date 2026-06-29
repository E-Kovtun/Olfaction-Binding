"""v2: all-pairs + phi binding metric, to fix the underpowered within-family test.

For every receptor pair compute:
  d_bind_jac = 1 - Jaccard of bound-odorant sets over commonly-tested molecules
  d_bind_phi = 1 - phi(label vectors) over commonly-tested molecules  (uses negatives too)
  d_pocket   = cosine dist of BW-22 pocket ESM embedding
  d_pocketH  = Hamming dist of the 22 BW pocket residues
  d_ecl2     = cosine dist of ECL2 (BW) embedding
  d_bg       = cosine dist of background embedding
  d_global   = cosine dist of mean (whole-seq) embedding   [control]
  same_family= bool

Then partial Spearman with each d_bind, controlling d_global, for (a) all pairs
and (b) within-family pairs. The all-pairs setting restores d_global's dynamic
range -> the control should now correlate clearly (sanity that the test has power).

    uv run python scripts/modeling/analysis/pocket_binding_signal_v2.py
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

K_SHARED = 5


def _family(slug: str) -> str:
    slug = slug.replace("_human", "")
    m = re.match(r"^(or?\d+)", slug)
    return m.group(1).upper() if m else slug.upper()


def partial_spearman(x, y, z):
    rxy = spearmanr(x, y).correlation
    rxz = spearmanr(x, z).correlation
    ryz = spearmanr(y, z).correlation
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return ((rxy - rxz * ryz) / den if den > 0 else np.nan), rxy


def cos_dist(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return np.nan if nu == 0 or nv == 0 else 1.0 - float(u @ v) / (nu * nv)


def report(df, label):
    print(f"\n========== {label}  (n={len(df)}) ==========")
    for metric in ["d_bind_jac", "d_bind_phi"]:
        d = df.dropna(subset=[metric, "d_global"])
        print(f"\n--- target = {metric}  (usable n={len(d)}, "
              f"mean={d[metric].mean():.3f}) ---")
        print(f"{'predictor':<12} {'raw rho':>9} {'partial rho':>13}")
        rg = spearmanr(d["d_global"], d[metric]).correlation
        for col in ["d_pocket", "d_pocketH", "d_ecl2", "d_bg"]:
            s = d.dropna(subset=[col])
            prho, raw = partial_spearman(s[col], s[metric], s["d_global"])
            print(f"{col:<12} {raw:>9.3f} {prho:>13.3f}")
        print(f"{'d_global':<12} {rg:>9.3f} {'(control)':>13}")


def main():
    pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
    prof = {r: dict(zip(g["inchikey"], g["label"])) for r, g in pairs.groupby("receptor")}
    posset = {r: {m for m, l in d.items() if l == 1} for r, d in prof.items()}

    V = build_variants()
    mean, pocket, bg, ecl2 = V["mean"], V["pocket22"], V["background"], V["ecl2_bw"]
    recs = list(mean.keys())
    del V; gc.collect()

    bwm = pd.read_csv(DATA / "processed" / "bw_pocket_matrix_curated.csv", index_col="receptor")
    ref = pd.read_csv(DATA / "processed" / "bw_ref_used_curated.csv")
    fam = {r: _family(s) for r, s in zip(ref["receptor"], ref["ref_entry"])}

    rows = []
    for a, b in itertools.combinations(recs, 2):
        da, db = prof[a], prof[b]
        shared = da.keys() & db.keys()
        if len(shared) < K_SHARED:
            continue
        shared = list(shared)
        la = np.fromiter((da[m] for m in shared), dtype=float, count=len(shared))
        lb = np.fromiter((db[m] for m in shared), dtype=float, count=len(shared))

        # Jaccard over positives
        pa, pb = posset[a] & set(shared), posset[b] & set(shared)
        union = pa | pb
        d_jac = (1.0 - len(pa & pb) / len(union)) if union else np.nan

        # phi (Pearson on binary), needs variance in both
        d_phi = (1.0 - np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else np.nan

        if a in bwm.index and b in bwm.index:
            ra, rb = bwm.loc[a].values, bwm.loc[b].values
            ok = pd.notna(ra) & pd.notna(rb)
            d_pH = float((ra[ok] != rb[ok]).mean()) if ok.sum() else np.nan
        else:
            d_pH = np.nan

        rows.append((d_jac, d_phi,
                     cos_dist(pocket[a], pocket[b]), d_pH,
                     cos_dist(ecl2[a], ecl2[b]),
                     cos_dist(bg[a], bg[b]),
                     cos_dist(mean[a], mean[b]),
                     fam.get(a) == fam.get(b), len(shared)))

    cols = ["d_bind_jac", "d_bind_phi", "d_pocket", "d_pocketH", "d_ecl2",
            "d_bg", "d_global", "same_family", "n_shared"]
    df = pd.DataFrame(rows, columns=cols)
    print(f"all pairs with >= {K_SHARED} shared molecules: {len(df)}")
    print(f"  median shared: {df['n_shared'].median():.0f}  "
          f"within-family: {int(df['same_family'].sum())}  "
          f"phi-usable: {df['d_bind_phi'].notna().sum()}  "
          f"jac-usable: {df['d_bind_jac'].notna().sum()}")

    out = ROOT / "results" / "analysis" / "pocket_binding_signal_v2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    report(df, "ALL PAIRS")
    report(df[df["same_family"]], "WITHIN FAMILY")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
