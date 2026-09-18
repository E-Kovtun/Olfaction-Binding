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

WHAT THE MASK DOES AND DOES NOT TOUCH. It decides what may be TRAINED on. Scoring uses
everything else: the pool written here is the parent's, complete, and each fold's test
split is whatever train and val did not take. A masked-out cell's response was hidden
from the model, never from us, so it is legitimate to score on -- and it is the
difference between a ~70-row test and a ~5200-row one.

Writes, per dataset:
    data/external/ofm/<DIR>/raw/<ds>_z.csv           the parent pool, COMPLETE
    data/external/ofm/<DIR>/raw/<ds>_measured.csv    the cells training may see
    data/external/ofm/<DIR>/{rand,our_inductive}_splits/...
                                                     train/val masked, test = the rest,
                                                     plus test_origin.csv saying which
                                                     kind each test cell is
    data/processed/molecules/molecule_smiles_<ds>.csv
Both split families are built here, from the parent's own fold files, so the held-out
odorants match the parent fold for fold. 03_build_ofm_our_inductive_splits.py is NOT
run for these -- it would split the pool without regard to the mask.
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

    measured = keep[pool["Protein sequence"].map(ri).to_numpy(),
                    pool["SMILES"].map(mi).to_numpy()]
    key = ["SMILES", "Protein sequence"]
    pool_idx = pd.MultiIndex.from_frame(pool[key])
    print(f"    pool {len(pool)} rows; measured {int(measured.sum())}")

    # THE MASK CONSTRAINS TRAINING, NOT SCORING. train and val keep only measured cells
    # -- a sparse assay is all the model would have had -- and test is everything else,
    # labels included. Those labels were never hidden from us, only from the model, so
    # scoring on them is honest; it is also what turns a ~70-row test into a ~5200-row
    # one, and the fly panel's transductive error bar of +/-0.657 was the cost of not
    # doing it. The three parts still partition the pool exactly, so `ofm_indices`
    # needs no special case.
    plans = {}
    for family in spec["families"]:
        for f in FOLDS:
            src = (OFM_DIR / DATASETS[base]["dir"] / f"{family}_splits"
                   / f"{family}_split_{f}")
            up = {s: pool_idx.isin(set(map(tuple,
                                           pd.read_csv(src / f"{s}_df.csv")[key].to_numpy())))
                  for s in ("train", "val", "test")}
            tr = up["train"] & measured
            va = up["val"] & measured
            plans[(family, f)] = (tr, va, ~(tr | va), up)

    for family in spec["families"]:
        tr, va, te, _ = plans[(family, 1)]
        cold = int((~pool.loc[te, "SMILES"].isin(set(pool.loc[tr, "SMILES"]))).sum())
        print(f"    {family:14} fold 1: train={int(tr.sum()):5} val={int(va.sum()):4} "
              f"test={int(te.sum()):5}  (cold-molecule test cells {cold})")
    if args.dry_run:
        print("    (dry run -- nothing written)")
        return

    raw = OFM_DIR / spec["raw"]
    if raw.exists() and not args.overwrite:
        raise SystemExit(f"{raw} exists -- pass --overwrite")
    raw.parent.mkdir(parents=True, exist_ok=True)
    # The pool is the parent's, COMPLETE. The mask lives in the split files, not here:
    # every cell keeps its response, which is what makes the held-out ones scoreable.
    pool.to_csv(raw, index=False)
    pool[measured][key].to_csv(raw.parent / f"{ds}_measured.csv", index=False)
    print(f"    wrote {raw.relative_to(_root)} + {ds}_measured.csv")

    for (family, f), (tr, va, te, up) in sorted(plans.items()):
        dst = OFM_DIR / spec["dir"] / f"{family}_splits" / f"{family}_split_{f}"
        dst.mkdir(parents=True, exist_ok=True)
        for split, m in (("train", tr), ("val", va), ("test", te)):
            pool[m].to_csv(dst / f"{split}_df.csv", index=False)
        # What KIND of cell each test row is. Without this the big test can never be cut
        # back to the parent panel's own test rows, and the cold-molecule column would
        # quietly be mostly warm -- on our_inductive only about a fifth of the test
        # cells belong to an odorant the SPLIT held out.
        #
        # TWO DIFFERENT NOTIONS, do not conflate them:
        #   origin == "upstream_test"  the parent's own test block. Scoring on exactly
        #                              these rows is what makes shrunk-vs-parent a
        #                              paired comparison.
        #   cold_molecule / _receptor  no row in the MASKED train split -- either the
        #                              split held it out, or the mask deleted its every
        #                              measurement. The model cannot tell those apart,
        #                              so this is the wider set (on Carey 1933 cells
        #                              against the split's 1100) and it is the honest
        #                              "the model never saw this entity" flag.
        train_mols = set(pool.loc[tr, "SMILES"])
        train_recs = set(pool.loc[tr, "Protein sequence"])
        origin = np.where(up["test"], "upstream_test",
                          np.where(up["train"], "unmeasured_train", "unmeasured_val"))
        pool[te][key].assign(
            origin=origin[te],
            cold_molecule=~pool.loc[te, "SMILES"].isin(train_mols).to_numpy(),
            cold_receptor=~pool.loc[te, "Protein sequence"].isin(train_recs).to_numpy(),
        ).to_csv(dst / "test_origin.csv", index=False)
    print(f"    wrote {len(plans)} split folders under "
          f"{(OFM_DIR / spec['dir']).relative_to(_root)}")

    bridge = pd.read_csv(MOL_DIR / DATASETS[base]["molecules"])
    out = MOL_DIR / spec["molecules"]
    bridge.to_csv(out, index=False)
    print(f"    wrote {out.relative_to(_root)} ({len(bridge)} odorants, unchanged)")


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
