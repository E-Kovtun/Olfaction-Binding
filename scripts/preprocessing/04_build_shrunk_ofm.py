#!/usr/bin/env python
"""Cut a complete insect panel down to M2OR's sparsity profile.

WHY. M2OR and the insect panels differ in two ways at once: they are different assays,
and one is 6% measured while the others are complete. Every comparison between them
confounds the two, including the regime inversion in the main tables (our graph wins
cold-molecule on M2OR and transductive on the insects). This script removes the second
difference so the first can be read on its own: same receptors, same odorants, same
responses, only most cells declared NOT MEASURED.

WHAT IS MATCHED. The `marginal` mask reproduces M2OR's two count profiles -- how many
odorants a receptor was run against, and how many receptors an odorant was run on --
each normalised to the target panel's own dimensions, plus the overall density. It does
NOT reproduce M2OR's assay panels: there, 1237 receptors share only 214 distinct odorant
sets, the largest being 9 odorants across 159 receptors. That block structure is a third
difference and is deliberately left out, so a difference this experiment finds cannot be
attributed to it.

READ THE ARITHMETIC BEFORE RUNNING. M2OR's density is 6.3%. At that density Carey keeps
~348 of 5500 cells and Hallem ~167 of 2640 -- about 7 odorants per receptor. A graph
trained on that is being told about the amount of data, not the shape of it, so the
single honest reading needs a DENSITY SWEEP and a `--mask random` control at each
density. `--dry-run` prints the whole diagnostic without writing a file.

    python scripts/preprocessing/04_build_shrunk_ofm.py --dataset cc --dry-run
    python scripts/preprocessing/04_build_shrunk_ofm.py --dataset cc hc --density 0.25

Writes, per dataset:
    data/external/ofm/<DIR>/raw/<ds>_z.csv          the surviving rows, columns untouched
    data/external/ofm/<DIR>/rand_splits/...         upstream's folds, filtered
    data/processed/molecules/molecule_smiles_<ds>.csv
Then build the cold-molecule family with 03_build_ofm_our_inductive_splits.py --dataset <ds>.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "orbind").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind.regimes import LORAX_DIR                                   # noqa: E402
from orbind.regimes_ofm import DATASETS, MOL_DIR, OFM_DIR, ofm_pool    # noqa: E402

FOLDS = (1, 2, 3, 4, 5)
SHRUNK = [k for k, v in DATASETS.items() if "base" in v]


def m2or_profile():
    """M2OR's two count profiles, as FRACTIONS of what was available to each row/column.

    Fractions, not counts: the target panel has different dimensions (and the opposite
    aspect ratio -- M2OR is 1237x596, the insects are 50x110 and 24x110), so raw counts
    are not transferable and only the normalised shape is.
    """
    d = _root / LORAX_DIR / "rand_split_1"
    pool = pd.concat([pd.read_csv(d / f"{s}_df.csv") for s in ("train", "val", "test")],
                     ignore_index=True)
    rec, mol = pool["Protein sequence"], pool["SMILES"]
    cells = pool.groupby([rec, mol]).size()
    n_rec, n_mol = rec.nunique(), mol.nunique()
    return {
        "row": (cells.groupby(level=0).size() / n_mol).to_numpy(float),
        "col": (cells.groupby(level=1).size() / n_rec).to_numpy(float),
        "density": len(cells) / (n_rec * n_mol),
        "shape": (n_rec, n_mol),
    }


def _draw(frac, n, rng):
    """`n` fractions with the same shape as the empirical `frac`.

    The empirical quantile function sampled at n even points, then permuted: the
    staircase is reproduced exactly, and which receptor lands on which step is random
    rather than tied to file order (which on these panels is meaningful).
    """
    q = np.quantile(frac, (np.arange(n) + 0.5) / n)
    return q[rng.permutation(n)]


def _ipf(row_t, col_t, iters=200):
    """A probability per cell, entries in [0, 1], whose row and column sums APPROACH the
    targets. Plain iterative proportional fitting with a clip.

    The clip is what keeps a heavy-tailed margin from asking for more than one
    measurement in a cell, and it is also why the margins are approximate rather than
    exact: a row whose target approaches the number of columns can only be served by
    driving cells to 1, and once they saturate the remaining mass has nowhere to go. On
    these panels that bites at the top of the profile -- M2OR's busiest receptors were
    run against 318 of 596 odorants, and asking for the same FRACTION of a 110-odorant
    panel lands near saturation. The visible consequence is a head slightly heavier than
    the target (78 against 58.7 on Carey). The DENSITY is unaffected: it is fixed
    exactly by the top-k draw in `mask`, not by these margins.
    """
    R, M = len(row_t), len(col_t)
    P = np.full((R, M), row_t.sum() / (R * M))
    for _ in range(iters):
        P *= (row_t / np.maximum(P.sum(1), 1e-12))[:, None]
        P = np.clip(P, 0.0, 1.0)
        P *= (col_t / np.maximum(P.sum(0), 1e-12))[None, :]
        P = np.clip(P, 0.0, 1.0)
    return P


def _repair(keep, P, min_row, min_col):
    """No receptor and no odorant may vanish, WITHOUT moving the density.

    A row with no cells leaves a graph node with no edges: that entity is gone, not
    under-measured, which is a different experiment. But adding cells without taking any
    back raises the density above the one being matched, and at M2OR's density the floor
    is a large share of the whole budget -- an earlier version of this function
    overshot 0.063 by half on the fly panel and reported a profile that was mostly its
    own floor. So every addition is paid for by a removal, taken where P is lowest among
    the rows and columns that can spare it.
    """
    for _ in range(200):
        need = [(i, None) for i in np.where(keep.sum(1) < min_row)[0]] + \
               [(None, j) for j in np.where(keep.sum(0) < min_col)[0]]
        if not need:
            return keep
        for i, j in need:
            if i is not None:
                cand = np.where(~keep[i])[0]
                if not len(cand):
                    continue
                add = (i, int(cand[np.argmax(P[i, cand])]))
            else:
                cand = np.where(~keep[:, j])[0]
                if not len(cand):
                    continue
                add = (int(cand[np.argmax(P[cand, j])]), j)
            # whoever pays must stay above both floors afterwards
            spare = keep & (keep.sum(1) > min_row)[:, None] & (keep.sum(0) > min_col)[None, :]
            spare[add] = False
            if not spare.any():
                raise SystemExit(
                    "cannot satisfy the per-receptor/per-molecule floor at this density: "
                    "no cell can be freed. Lower --min-per-receptor/--min-per-molecule "
                    "or raise --density.")
            drop = np.unravel_index(int(np.argmin(np.where(spare, P, np.inf))), P.shape)
            keep[add] = True
            keep[drop] = False
    raise SystemExit("the floor repair did not converge -- lower the floors or raise "
                     "--density")


def mask(n_rec, n_mol, density, prof, kind, seed, min_row=1, min_col=1):
    rng = np.random.default_rng(seed)
    total = int(round(density * n_rec * n_mol))
    # The floor and the density are two demands on the same budget, and on these small
    # panels they collide: at M2OR's 6.3% the fly panel has 167 cells to spend while two
    # per odorant alone would cost 220. Say so instead of quietly keeping more cells.
    floor = max(min_row * n_rec, min_col * n_mol)
    if floor > total:
        raise SystemExit(
            f"density {density:.4f} leaves {total} cells on a {n_rec}x{n_mol} panel, but "
            f"the floor (>={min_row}/receptor, >={min_col}/odorant) needs at least "
            f"{floor}. Raise --density or lower the floor.")
    if kind == "random":
        P = np.full((n_rec, n_mol), density)
    else:
        row_t = _draw(prof["row"], n_rec, rng) * n_mol
        col_t = _draw(prof["col"], n_mol, rng) * n_rec
        row_t *= total / row_t.sum()
        col_t *= total / col_t.sum()
        P = _ipf(row_t, col_t)
    # Gumbel top-k: exactly `total` cells, each drawn in proportion to P, reproducible
    # from the seed. Bernoulli sampling would hit the density only on average.
    g = np.log(np.maximum(P, 1e-12)) + rng.gumbel(size=P.shape)
    keep = np.zeros(P.shape, bool)
    keep.flat[np.argsort(-g, axis=None)[:total]] = True
    keep = _repair(keep, P, min_row, min_col)
    assert int(keep.sum()) == total, "the repair changed the density"
    return keep


def report(name, keep, prof):
    n_rec, n_mol = keep.shape
    per_r, per_c = keep.sum(1), keep.sum(0)
    print(f"\n  {name}: {int(keep.sum())} of {n_rec * n_mol} cells "
          f"-> density {keep.mean():.4f}   (M2OR {prof['density']:.4f})")
    for lab, got, ref, avail in (("receptor", per_r, prof["row"] * n_mol, n_mol),
                                 ("molecule", per_c, prof["col"] * n_rec, n_rec)):
        qs = [0, 10, 25, 50, 75, 90, 100]
        a = np.percentile(got, qs).round(1)
        b = np.percentile(ref, qs).round(1)
        print(f"    per {lab:8} (of {avail:3}) min/p10/p25/med/p75/p90/max")
        print(f"        this  {list(a)}")
        print(f"        M2OR* {list(b)}    (*rescaled to this panel)")
    empty_folds = np.sum(per_r < len(FOLDS))
    if empty_folds:
        print(f"    NOTE {empty_folds}/{n_rec} receptors hold fewer cells than there are "
              f"folds -- some (receptor, fold) cells will be empty")


def build(ds, args, prof):
    spec = DATASETS[ds]
    base = spec["base"]
    pool = ofm_pool(base)
    recs = list(pd.unique(pool["Protein sequence"]))
    mols = list(pd.unique(pool["SMILES"]))
    ri = {r: i for i, r in enumerate(recs)}
    mi = {m: i for i, m in enumerate(mols)}

    keep = mask(len(recs), len(mols), args.density, prof, args.mask, args.seed,
                min_row=args.min_per_receptor, min_col=args.min_per_molecule)
    report(f"{ds} [{args.mask}, seed {args.seed}]", keep, prof)

    sel = keep[pool["Protein sequence"].map(ri).to_numpy(),
               pool["SMILES"].map(mi).to_numpy()]
    shrunk = pool[sel]
    surviving = set(shrunk["SMILES"])
    print(f"    rows {len(pool)} -> {len(shrunk)}; "
          f"odorants {len(mols)} -> {shrunk['SMILES'].nunique()}; "
          f"receptors {len(recs)} -> {shrunk['Protein sequence'].nunique()}")
    if args.dry_run:
        print("    (dry run -- nothing written)")
        return

    raw = OFM_DIR / spec["raw"]
    if raw.exists() and not args.overwrite:
        raise SystemExit(f"{raw} exists -- pass --overwrite")
    raw.parent.mkdir(parents=True, exist_ok=True)
    shrunk.to_csv(raw, index=False)
    print(f"    wrote {raw.relative_to(_root)}")

    # Upstream's own i.i.d. folds, restricted to the surviving rows. Filtering rather
    # than redrawing keeps `rand` meaning exactly what it means on the parent panel, and
    # the three parts still partition the shrunk pool, which is what ofm_indices checks.
    key = ["SMILES", "Protein sequence"]
    kept_keys = set(map(tuple, shrunk[key].to_numpy()))
    for f in FOLDS:
        src = OFM_DIR / DATASETS[base]["dir"] / "rand_splits" / f"rand_split_{f}"
        dst = OFM_DIR / spec["dir"] / "rand_splits" / f"rand_split_{f}"
        dst.mkdir(parents=True, exist_ok=True)
        sizes = {}
        for split in ("train", "val", "test"):
            df = pd.read_csv(src / f"{split}_df.csv")
            m = [tuple(t) in kept_keys for t in df[key].to_numpy()]
            df[m].to_csv(dst / f"{split}_df.csv", index=False)
            sizes[split] = int(np.sum(m))
        print(f"    rand_split_{f}: " + " ".join(f"{k}={v}" for k, v in sizes.items()))

    bridge = pd.read_csv(MOL_DIR / DATASETS[base]["molecules"])
    out = MOL_DIR / spec["molecules"]
    bridge[bridge["smiles"].isin(surviving)].to_csv(out, index=False)
    print(f"    wrote {out.relative_to(_root)}")
    print(f"    NEXT: python scripts/preprocessing/03_build_ofm_our_inductive_splits.py "
          f"--dataset {ds}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=SHRUNK, choices=SHRUNK)
    ap.add_argument("--density", type=float, default=None,
                    help="fraction of cells kept. Default: M2OR's own, measured here")
    ap.add_argument("--mask", default="marginal", choices=["marginal", "random"],
                    help="marginal: match M2OR's row/column count profiles. "
                         "random: uniform holes at the same density -- the control that "
                         "says whether the SHAPE mattered or only the amount")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-receptor", type=int, default=1,
                    help="cells every receptor keeps. A receptor at 0 is a graph node "
                         "with no edges -- absent, not under-measured (default 1)")
    ap.add_argument("--min-per-molecule", type=int, default=1,
                    help="cells every odorant keeps (default 1). Raising either floor "
                         "costs budget the density has to pay for, and on these panels "
                         "the two collide well before M2OR's density")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    prof = m2or_profile()
    print(f"M2OR profile: {prof['shape'][0]} receptors x {prof['shape'][1]} odorants, "
          f"density {prof['density']:.4f}")
    if a.density is None:
        a.density = prof["density"]
    for ds in a.dataset:
        build(ds, a, prof)


if __name__ == "__main__":
    main()
