"""Carey (CC) and Hallem-Carlson (HC) pools and split-index bookkeeping.

The sibling of `orbind.regimes`, for the two non-M2OR datasets shipped with the
olfactory foundation models release (https://zenodo.org/records/17228740).
Kept separate because almost nothing is shared: a different pool, a different
key, a *continuous* target, and upstream's splits arrive as three families
rather than one.

Simpler than the M2OR case in one respect
-----------------------------------------
`orbind.regimes` has to reconstruct its pool by concatenating fold 1's
train+val+test, because LoRaX ships no standalone pool file. Here the raw
response table *is* the pool, and every split family is an exact partition of
it -- verified: for CC all three families x 5 folds reunite to the same 5500
rows, for HC to the same 2640, with `(SMILES, Protein sequence)` unique
throughout. So "position i" is simply row i of the raw csv.

The three families are the three generalization scenarios
---------------------------------------------------------
    rand    i.i.d.               -- transductive; every receptor and every
                                    odorant is also in train
    cdhit   unseen receptors     -- CD-HIT sequence clusters held out whole;
                                    all 10 test receptors are novel
    scaf    unseen odorants      -- Bemis-Murcko scaffolds held out whole;
                                    all 22 test odorants are novel

CC has all three; **HC ships only `rand`**. That asymmetry is upstream's, not
ours -- see `available_families`.

A fourth family is ours, not upstream's
---------------------------------------
    our_inductive   unseen odorants, stratified -- built by
                    scripts/preprocessing/03_build_ofm_our_inductive_splits.py

Same shape as `scaf` (22 test / 18 val / 70 train molecules) and the same
claim -- cold molecule -- but the 22 are dealt by systematic sampling over the
molecules ordered by response dynamic range instead of by blocks of a
scaffold-sorted list. `scaf`'s rule is deterministic and its fold 1 lands on
the carboxylic-acid homologous series: test sd 0.215 and naive R2 -4.92, which
IS the published -1.016 average. Under `our_inductive` every CC fold has test
sd 0.97-1.04 and naive R2 -0.000, so R2 is readable fold by fold. It is not
scaffold-disjoint and does not claim to be -- with 71 of 110 odorants sharing
the empty Murcko scaffold, no 5-fold scheme here can be, upstream's included.
Both are built for HC too, which upstream left with `rand` alone.

The target is continuous
------------------------
`output` is a globally z-scored response, not a 0/1 flag. `ofm_pairs` puts it
in `label` unchanged; it is the caller's job to run with
`task="regression"` (see `orbind.tasks`). Binarising it would throw away the
very thing these datasets were fetched for.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve().parent.parent

OFM_DIR = _root / "data" / "external" / "ofm"
MOL_DIR = _root / "data" / "processed" / "molecules"

# (SMILES, Protein sequence) is unique in both raw tables -- both are complete
# receptor x odorant matrices (CC 50x110, HC 24x110).
_KEY_COLS = ["SMILES", "Protein sequence"]

DATASETS: dict[str, dict] = {
    "cc": {
        "dir": "CC",
        "raw": pathlib.Path("CC") / "raw" / "CC_reformat_z.csv",
        "families": ("rand", "cdhit", "scaf", "our_inductive"),
        "molecules": "molecule_smiles_cc.csv",
        "label": "Carey",
    },
    "hc": {
        "dir": "HC",
        "raw": pathlib.Path("HC") / "raw" / "hc_with_prot_seq_z.csv",
        "families": ("rand", "our_inductive"),
        "molecules": "molecule_smiles_hc.csv",
        "label": "Hallem-Carlson",
    },
    # The same two panels with most cells declared NOT MEASURED, so that what is left
    # carries M2OR's sparsity profile instead of a complete matrix. Built by
    # scripts/preprocessing/04_build_shrunk_ofm.py; the receptors, the odorants and the
    # responses are untouched, only which (receptor, odorant) cells survive. They exist
    # to separate "M2OR behaves differently because it is sparse" from "because it is a
    # different assay", which no comparison between the complete insect panels and M2OR
    # can do. `base` is the panel each one was cut from -- embeddings and the molecule
    # bridge are shared with it, since neither receptors nor odorants changed.
    "cc_shrinked": {
        "dir": "CC_shrinked",
        "raw": pathlib.Path("CC_shrinked") / "raw" / "cc_shrinked_z.csv",
        "families": ("rand", "our_inductive"),
        "molecules": "molecule_smiles_cc_shrinked.csv",
        "label": "Carey (shrunk)",
        "base": "cc",
    },
    "hc_shrinked": {
        "dir": "HC_shrinked",
        "raw": pathlib.Path("HC_shrinked") / "raw" / "hc_shrinked_z.csv",
        "families": ("rand", "our_inductive"),
        "molecules": "molecule_smiles_hc_shrinked.csv",
        "label": "Hallem-Carlson (shrunk)",
        "base": "hc",
    },
}
FOLDS = (1, 2, 3, 4, 5)


def _spec(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}, got {dataset!r}")
    return DATASETS[dataset]


def available_families(dataset: str) -> tuple[str, ...]:
    """Split families upstream actually ships for this dataset. HC has only
    `rand`; asking it for `cdhit`/`scaf` is a missing-data problem, not a
    code problem, so callers should check here rather than catch a
    FileNotFoundError."""
    return _spec(dataset)["families"]


def ofm_pool(dataset: str) -> pd.DataFrame:
    """The raw response table, in file order. Row i here is what every index
    array below refers to."""
    path = OFM_DIR / _spec(dataset)["raw"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- extract the dataset from data.zip "
            f"(https://zenodo.org/records/17228740) under data/external/ofm/")
    pool = pd.read_csv(path)
    if pool.duplicated(_KEY_COLS).any():
        raise ValueError(f"{path.name}: {_KEY_COLS} is not unique, positions would be ambiguous")
    return pool


def ofm_indices(dataset: str, family: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position arrays into `ofm_pool(dataset)` for upstream's own
    `{family}_split_{fold}` train/val/test partition."""
    spec = _spec(dataset)
    if family not in spec["families"]:
        raise ValueError(
            f"{dataset} ships no {family!r} splits (has {spec['families']}); "
            f"upstream released cdhit/scaf only for CC")
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}, got {fold}")

    pool = ofm_pool(dataset)
    pool_pos = pool[_KEY_COLS].reset_index().rename(columns={"index": "_pos"})
    split_dir = OFM_DIR / spec["dir"] / f"{family}_splits" / f"{family}_split_{fold}"

    def idx_for(split: str) -> np.ndarray:
        path = split_dir / f"{split}_df.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path)
        merged = df[_KEY_COLS].merge(pool_pos, on=_KEY_COLS, how="left")
        if merged["_pos"].isna().any():
            n = int(merged["_pos"].isna().sum())
            raise ValueError(f"{dataset}/{family}_{fold} {split}: {n}/{len(merged)} rows "
                             f"didn't match the pool")
        return merged["_pos"].to_numpy(dtype=np.int64)

    tr, va, te = idx_for("train"), idx_for("val"), idx_for("test")
    covered = len(tr) + len(va) + len(te)
    if covered != len(pool) or len(set(tr) | set(va) | set(te)) != len(pool):
        raise ValueError(f"{dataset}/{family}_{fold}: train+val+test cover {covered} of "
                         f"{len(pool)} pool rows and are not a clean partition")
    return tr, va, te


def ofm_pairs(dataset: str) -> pd.DataFrame:
    """The pool as a `pairs` table in this repo's convention (`receptor`,
    `inchikey`, `label`), row-for-row aligned with `ofm_pool`/`ofm_indices`.

    `label` stays the continuous z-scored response -- run these with
    `task="regression"`.
    """
    spec = _spec(dataset)
    pool = ofm_pool(dataset)

    mol_path = MOL_DIR / spec["molecules"]
    if not mol_path.exists():
        raise FileNotFoundError(
            f"{mol_path} not found -- run scripts/embedding_generation/molecules/"
            f"07_prepare_ofm_molecules.py --tag {dataset}")
    bridge = pd.read_csv(mol_path)
    smiles_to_inchikey = dict(zip(bridge["smiles"], bridge["inchikey"]))

    pairs = pool.rename(columns={"Protein sequence": "receptor", "output": "label",
                                 "SMILES": "smiles"})
    pairs["inchikey"] = pairs["smiles"].map(smiles_to_inchikey)
    if pairs["inchikey"].isna().any():
        missing = sorted(set(pairs.loc[pairs["inchikey"].isna(), "smiles"]))
        raise ValueError(f"{len(missing)} SMILES have no InChIKey in {mol_path.name}, "
                         f"e.g. {missing[:3]} -- rerun script 07")
    pairs["label"] = pairs["label"].astype(np.float32)
    return pairs
