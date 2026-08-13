"""Build `our_inductive` -- our own cold-molecule 5-fold splits for CC / HC.

Why this exists
---------------
Upstream's `scaf` family is advertised as a Bemis-Murcko scaffold split, but on
Carey it degenerates. Only **16 distinct scaffolds** cover the 110 odorants and
one of them -- the empty scaffold shared by all 71 acyclic molecules -- is 65%
of the set. The folds are built by a deterministic rule (group by scaffold,
order the groups by descending size, keep file order inside a group, cut the
concatenation into five consecutive blocks of 22), so:

  * folds 1-3 are three arbitrary slices of the SAME (empty) scaffold group --
    there is no scaffold-holdout there at all, only a molecule holdout;
  * because the raw csv is ordered by chemical class, fold 1 lands exactly on
    the carboxylic-acid homologous series (C2..C18), which barely activates any
    AgOr. Its test sd is 0.215 against 0.75-1.23 elsewhere and its naive R2 is
    -4.92 against -0.02..-0.05. The published scaf mean of -1.016 is that one
    fold and nothing else.

So `scaf` cannot answer "how well do we generalise to an unseen odorant?" --
its five numbers differ by which chemical class happened to fall in the block,
not by sampling noise, and averaging them is dominated by a degenerate fold.

What `our_inductive` does instead
---------------------------------
Still a **cold-molecule** split (held-out molecules appear in no training row,
contribute no message-passing edge), same shape as upstream so results are
directly comparable: 22 test / 18 val / 70 train molecules, 1100 / 900 / 3500
rows on CC.

The difference is the assignment rule. Molecules are ordered by a **dynamic
range** score -- the sd of their z-response across all receptors, which is what
fold 1 lacked -- and dealt into the five folds by systematic sampling
(positions i, i+5, i+10, ... ). Every fold therefore takes exactly one molecule
from each consecutive block of five in the score order, so all five folds span
the full activity range by construction and no fold can be flat. Ties in the
score are broken by SMILES so the whole thing is deterministic; there is no
seed.

Val is drawn the same way from the 88 non-test molecules: 18 positions spread
evenly over the score-ordered remainder, so early stopping sees the same range
the test does.

Honest limitation
-----------------
This is **not** scaffold-disjoint, and it does not pretend to be. With 71 of
110 molecules sharing the empty scaffold, no 5-fold scheme on this dataset can
be; upstream's isn't either (its empty group is spread over four folds and its
thiazole group over two). The claim here is exactly "unseen molecule", which is
the claim `scaf` actually delivers too -- minus the degenerate fold.

Usage
-----
    python scripts/preprocessing/03_build_ofm_our_inductive_splits.py --dataset cc
    python scripts/preprocessing/03_build_ofm_our_inductive_splits.py --dataset hc

Writes data/external/ofm/{CC,HC}/our_inductive_splits/our_inductive_split_{1..5}/
{train,val,test}_df.csv in upstream's own format, so `orbind.regimes_ofm`
consumes them through the same code path as rand/cdhit/scaf.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from orbind.regimes_ofm import DATASETS, OFM_DIR, ofm_pool  # noqa: E402

FAMILY = "our_inductive"
N_FOLDS = 5
N_VAL_MOLS = 18


def molecule_scores(pool: pd.DataFrame) -> pd.Series:
    """Dynamic range of each molecule's response profile, high first.

    sd across receptors, not mean: a molecule that moves *some* receptor is
    informative even if its average is low, and it is precisely the flat
    molecules that made scaf fold 1 unscorable.
    """
    score = pool.groupby("SMILES")["output"].std(ddof=0)
    # descending score, SMILES as the tie-break -- no seed, no file-order dependence
    return score.iloc[np.lexsort((score.index.to_numpy(), -score.to_numpy()))]


def build(dataset: str, overwrite: bool) -> None:
    spec = DATASETS[dataset]
    pool = ofm_pool(dataset)
    score = molecule_scores(pool)
    order = list(score.index)  # molecules, most dynamic first
    n_mol = len(order)
    if n_mol % N_FOLDS:
        print(f"note: {n_mol} molecules is not divisible by {N_FOLDS}; "
              f"fold sizes will differ by one")

    # systematic sampling: fold f takes positions f, f+5, f+10, ...
    folds = {f: [order[i] for i in range(f - 1, n_mol, N_FOLDS)] for f in range(1, N_FOLDS + 1)}

    out_root = OFM_DIR / spec["dir"] / f"{FAMILY}_splits"
    if out_root.exists() and not overwrite:
        raise SystemExit(f"{out_root} already exists -- pass --overwrite to rebuild")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{dataset}: {len(pool)} rows, {n_mol} molecules, "
          f"{pool['Protein sequence'].nunique()} receptors")
    print(f"{'fold':>5} {'mols':>5} {'train':>6} {'val':>5} {'test':>5} "
          f"{'test mean':>10} {'test sd':>8} {'naive R2':>9}")

    for f in range(1, N_FOLDS + 1):
        test_mols = folds[f]
        rest = [m for m in order if m not in set(test_mols)]
        # 18 positions spread evenly over the score-ordered remainder
        val_pos = sorted(set(np.linspace(0, len(rest) - 1, N_VAL_MOLS).round().astype(int)))
        val_mols = [rest[i] for i in val_pos]
        train_mols = [m for m in rest if m not in set(val_mols)]

        assert not (set(test_mols) & set(val_mols) | set(test_mols) & set(train_mols)
                    | set(val_mols) & set(train_mols))
        assert len(test_mols) + len(val_mols) + len(train_mols) == n_mol

        d = out_root / f"{FAMILY}_split_{f}"
        d.mkdir(exist_ok=True)
        frames = {"train": train_mols, "val": val_mols, "test": test_mols}
        rows = {}
        for split, mols in frames.items():
            sub = pool[pool["SMILES"].isin(set(mols))]
            sub.to_csv(d / f"{split}_df.csv", index=False)
            rows[split] = len(sub)
        assert sum(rows.values()) == len(pool)

        y_te = pool.loc[pool["SMILES"].isin(set(test_mols)), "output"]
        y_tr = pool.loc[pool["SMILES"].isin(set(train_mols)), "output"]
        naive = 1 - ((y_te - y_tr.mean()) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
        print(f"{f:>5} {len(test_mols):>5} {rows['train']:>6} {rows['val']:>5} {rows['test']:>5} "
              f"{y_te.mean():>+10.3f} {y_te.std():>8.3f} {naive:>+9.3f}")

    print(f"\nwritten to {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cc", choices=sorted(DATASETS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    build(args.dataset, args.overwrite)


if __name__ == "__main__":
    main()
