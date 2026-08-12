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
        "families": ("rand", "cdhit", "scaf"),
        "molecules": "molecule_smiles_cc.csv",
        "label": "Carey",
    },
    "hc": {
        "dir": "HC",
        "raw": pathlib.Path("HC") / "raw" / "hc_with_prot_seq_z.csv",
        "families": ("rand",),
        "molecules": "molecule_smiles_hc.csv",
        "label": "Hallem-Carlson",
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
