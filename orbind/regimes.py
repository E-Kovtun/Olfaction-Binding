"""full_full split-index bookkeeping.

LoRaX's rand_split_1..5 are five different train/val/test *partitions of the
same underlying pool* (verified: all 5 folds contain exactly the same 46563
rows, same labels, just reshuffled across train/val/test). So we don't
duplicate that pool into our own processed/ tables -- we persist only the
*partition* (which pool positions are train/val/test, per fold or per
inductive-molecule seed) and reconstruct the pool itself deterministically
from LoRaX's own csv files whenever it's actually needed. The pool's row
order is fixed by a simple, reproducible rule (concat fold_1's train+val+test
in that order), so "position i" means the same thing every time without us
storing the pool contents anywhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LORAX_DIR = "data/splits_indexes/lorax_m2or"
_KEY_COLS = ["SMILES", "Protein sequence", "_DataQuality"]


def load_full_full_pool(lorax_dir: str = LORAX_DIR, pool_fold: int = 1) -> pd.DataFrame:
    """Deterministically reconstruct the canonical full_full pool: train+val+test
    of `pool_fold`, concatenated in that fixed order. Any fold works as the
    pool source (all 5 are the same rows reshuffled) -- fold 1 is just the
    convention so "position i" is stable across calls."""
    parts = [pd.read_csv(f"{lorax_dir}/rand_split_{pool_fold}/{s}_df.csv") for s in ("train", "val", "test")]
    return pd.concat(parts, ignore_index=True)


def transductive_indices(fold: int, lorax_dir: str = LORAX_DIR, pool_fold: int = 1):
    """Position arrays into load_full_full_pool(pool_fold) for LoRaX's own
    rand_split_{fold} train/val/test partition."""
    pool = load_full_full_pool(lorax_dir, pool_fold)
    pool_pos = pool[_KEY_COLS].reset_index().rename(columns={"index": "_pos"})

    def idx_for(split):
        df = pd.read_csv(f"{lorax_dir}/rand_split_{fold}/{split}_df.csv")
        merged = df[_KEY_COLS].merge(pool_pos, on=_KEY_COLS, how="left")
        if merged["_pos"].isna().any():
            n_missing = int(merged["_pos"].isna().sum())
            raise ValueError(f"fold {fold} {split}: {n_missing}/{len(merged)} rows didn't match the pool")
        return merged["_pos"].to_numpy(dtype=np.int64)

    return idx_for("train"), idx_for("val"), idx_for("test")


def inductive_molecule_indices(seed: int, lorax_dir: str = LORAX_DIR, pool_fold: int = 1,
                                holdout_fraction: float = 0.30,
                                test_fraction_within_holdout: float = 2.0 / 3.0):
    """Cold-molecule split of the pool: mirrors
    train_full_full_site_mil_attention.py's own inductive_molecule logic
    exactly (val/test drawn only from ec50-quality rows of held-out
    molecules; train is every pool row whose molecule isn't held out,
    regardless of quality)."""
    pool = load_full_full_pool(lorax_dir, pool_fold)
    ec50 = pool.loc[pool["_DataQuality"].eq("ec50")]
    mol_tbl = ec50.groupby("SMILES", sort=False)["output"].max().reset_index(name="has_pos")
    mols = mol_tbl["SMILES"].to_numpy()
    strat = mol_tbl["has_pos"].astype(int).to_numpy()
    try:
        _, held = train_test_split(mols, test_size=holdout_fraction, random_state=seed, stratify=strat)
        held_tbl = mol_tbl[mol_tbl["SMILES"].isin(held)]
        val_mols, test_mols = train_test_split(
            held_tbl["SMILES"].to_numpy(), test_size=test_fraction_within_holdout,
            random_state=seed + 17, stratify=held_tbl["has_pos"].astype(int).to_numpy())
    except ValueError:
        _, held = train_test_split(mols, test_size=holdout_fraction, random_state=seed)
        val_mols, test_mols = train_test_split(held, test_size=test_fraction_within_holdout, random_state=seed + 17)

    held_set = set(val_mols) | set(test_mols)
    train_idx = np.where(~pool["SMILES"].isin(held_set))[0].astype(np.int64)
    val_idx = np.where(pool["SMILES"].isin(val_mols) & pool["_DataQuality"].eq("ec50"))[0].astype(np.int64)
    test_idx = np.where(pool["SMILES"].isin(test_mols) & pool["_DataQuality"].eq("ec50"))[0].astype(np.int64)
    return train_idx, val_idx, test_idx


def inductive_molecule_v5_indices(seed: int, lorax_dir: str = LORAX_DIR, pool_fold: int = 1,
                                   test_frac: float = 0.2, val_frac: float = 0.1):
    """Cold-molecule split reproducing `orbind.lorax.build_inductive_molecule`
    *exactly* -- the split the v5 graph screen actually ran, as opposed to
    `inductive_molecule_indices` above (our own later variant: 30% holdout,
    stratified by whether a molecule has any positive, 2/3-1/3 test/val).

    v5's rule: molecules with EC50 data are the testable pool; an unstratified
    `np.random.default_rng(seed).permutation` puts the first 20% in test and the
    next 10% in val; train is every pool row (any quality) whose molecule is in
    neither. Val/test keep only EC50-quality rows of their molecules.

    Position-for-position identical to lorax's version despite lorax building
    its own frame: lorax first drops duplicate
    (SMILES, Protein sequence, _DataQuality) rows and rows lacking an
    embedding, but on this pool both are no-ops (46563 rows survive dedup, and
    ChemBERTa/ESM-1b cover every molecule/receptor), and the molecule-level
    draw only ever depends on the *set* of unique EC50 SMILES, which dedup
    cannot change."""
    pool = load_full_full_pool(lorax_dir, pool_fold)
    ec50 = pool.loc[pool["_DataQuality"].eq("ec50")]
    testable = np.array(sorted(ec50["SMILES"].unique()))
    perm = np.random.default_rng(seed).permutation(len(testable))
    n_te, n_va = int(test_frac * len(testable)), int(val_frac * len(testable))
    test_mols = set(testable[perm[:n_te]])
    val_mols = set(testable[perm[n_te:n_te + n_va]])

    smiles, is_ec50 = pool["SMILES"], pool["_DataQuality"].eq("ec50")
    train_idx = np.where(~smiles.isin(test_mols | val_mols))[0].astype(np.int64)
    val_idx = np.where(smiles.isin(val_mols) & is_ec50)[0].astype(np.int64)
    test_idx = np.where(smiles.isin(test_mols) & is_ec50)[0].astype(np.int64)
    return train_idx, val_idx, test_idx


def build_split_index_store(folds=(1, 2, 3, 4, 5), inductive_seeds=(42, 43, 44, 45, 46),
                             lorax_dir: str = LORAX_DIR, pool_fold: int = 1) -> dict[str, np.ndarray]:
    """All persisted index arrays, keyed `{regime}_{repeat}_{split}`."""
    store: dict[str, np.ndarray] = {}
    for fold in folds:
        tr, va, te = transductive_indices(fold, lorax_dir, pool_fold)
        store[f"transductive_{fold}_train"] = tr
        store[f"transductive_{fold}_val"] = va
        store[f"transductive_{fold}_test"] = te
    for seed in inductive_seeds:
        tr, va, te = inductive_molecule_indices(seed, lorax_dir, pool_fold)
        store[f"inductive_molecule_{seed}_train"] = tr
        store[f"inductive_molecule_{seed}_val"] = va
        store[f"inductive_molecule_{seed}_test"] = te
    for seed in inductive_seeds:
        tr, va, te = inductive_molecule_v5_indices(seed, lorax_dir, pool_fold)
        store[f"inductive_molecule_v5_{seed}_train"] = tr
        store[f"inductive_molecule_v5_{seed}_val"] = va
        store[f"inductive_molecule_v5_{seed}_test"] = te
    return store


def load_split(regime: str, repeat: int, path: str = "data/processed/full_full_split_indices.npz"):
    """Read back one (train_idx, val_idx, test_idx) triple for `regime`
    ("transductive", "inductive_molecule", or "inductive_molecule_v5") and
    `repeat` (fold 1-5, or seed) from the persisted store."""
    z = np.load(path)
    prefix = f"{regime}_{repeat}_"
    return z[f"{prefix}train"], z[f"{prefix}val"], z[f"{prefix}test"]


def full_full_pairs(pool_fold: int = 1,
                     bridge_path: str = "data/processed/molecules/lorax_smiles_to_inchikey.csv") -> pd.DataFrame:
    """The full_full pool as a `pairs` table matching curated/full's column
    convention (`receptor`, `inchikey`, `label`), for `orbind.ensemble.run_ensemble`
    and its entity extractors. Row order matches `load_full_full_pool`/`load_split`
    exactly -- this is the same pool, just renamed and with `inchikey` bridged in,
    nothing filtered or reordered."""
    pool = load_full_full_pool(lorax_dir=LORAX_DIR, pool_fold=pool_fold)
    bridge = pd.read_csv(bridge_path)
    smiles_to_inchikey = dict(zip(bridge["smiles_lorax"], bridge["inchikey"]))
    pairs = pool.rename(columns={"Protein sequence": "receptor", "output": "label", "SMILES": "smiles"})
    pairs["inchikey"] = pairs["smiles"].map(smiles_to_inchikey)
    pairs["label"] = pairs["label"].astype(np.float32)
    return pairs
