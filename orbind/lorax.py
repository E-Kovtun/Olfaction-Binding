"""Data helpers for the FULL_FULL dataset (LORAX / Hladis M2OR release).

full_full lives in data/external/lorax_m2or as 5 pre-defined random folds. Two
evaluation regimes are built here so the graph pipeline can mirror curated/full:

  transductive        — use the LORAX fold as-is. MP graph + supervision = train
                        (full noisy mix); test = the fold's EC50-only pairs. All
                        nodes are known; only edges are held out. (≈ stratified)

  inductive_molecule  — hold out whole MOLECULES (cold start, ≈ group_molecule).
                        We repartition the fold by molecule: a fraction of the
                        molecules that have EC50 data are removed from train
                        ENTIRELY (no MP edges, no supervision); the test is those
                        held-out molecules' EC50 pairs. Train keeps every pair
                        (any quality) of the remaining molecules. Test stays
                        EC50-only to match the transductive protocol.

Node features (paper: molecule encoder is interchangeable):
  proteins  — putative ESM-1b 650M mean-pooled, keyed by amino-acid sequence.
              The identity is a working hypothesis: these vectors do not match
              fresh ESM2-t33 embeddings for the same sequences.
  molecules — ChemBERTa-77M (384-d), keyed by SMILES (GIN covers only 64%).
"""
from __future__ import annotations
import pathlib, pickle
import numpy as np, pandas as pd, torch

from orbind.hetero import load_npz_dict, MOL, PROT  # noqa: F401  (re-export convenience)

_root = pathlib.Path(__file__).resolve().parent.parent
LORAX = _root / "data" / "external" / "lorax_m2or"
MOLECULE_EMBEDDINGS = _root / "data" / "embeddings" / "molecules"
CHEMBERTA = MOLECULE_EMBEDDINGS / "chemberta_77m_lorax.pkl"
ESM = LORAX / "esm1b_650m_mean_lorax.npz"


def _resolve_embedding_path(path, default):
    path = pathlib.Path(path) if path is not None else default
    return path if path.is_absolute() else _root / path


def _load_embedding_dict(path):
    if path.suffix == ".npz":
        return {k: np.asarray(v, dtype=np.float32)
                for k, v in load_npz_dict(path).items()}
    if path.suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return {k: np.asarray(v, dtype=np.float32) for k, v in pickle.load(f).items()}
    raise ValueError(f"Unsupported embedding file: {path}")


def load_embeddings(protein_path=None, molecule_path=None):
    """Load default LORAX features or explicitly selected NPZ/pickle features."""
    protein_path = _resolve_embedding_path(protein_path, ESM)
    molecule_path = _resolve_embedding_path(molecule_path, CHEMBERTA)
    return _load_embedding_dict(protein_path), _load_embedding_dict(molecule_path)


def _load_fold(fold, esm, chem):
    base = LORAX / f"rand_split_{fold}"
    dfs = {s: pd.read_csv(base / f"{s}_df.csv") for s in ("train", "val", "test")}
    keep = lambda d: d[d["SMILES"].isin(chem) & d["Protein sequence"].isin(esm)].reset_index(drop=True)
    return {s: keep(d) for s, d in dfs.items()}


def _node_universe(dfs, esm, chem):
    mol_ids = sorted(set().union(*[set(d["SMILES"]) for d in dfs.values()]))
    prot_ids = sorted(set().union(*[set(d["Protein sequence"]) for d in dfs.values()]))
    mi = {k: i for i, k in enumerate(mol_ids)}
    pi = {k: i for i, k in enumerate(prot_ids)}
    Xm = torch.tensor(np.stack([chem[k] for k in mol_ids]), dtype=torch.float)
    Xp = torch.tensor(np.stack([esm[k] for k in prot_ids]), dtype=torch.float)
    return Xm, Xp, mi, pi


def _pairs(df, mi, pi):
    m = df["SMILES"].map(mi).to_numpy()
    p = df["Protein sequence"].map(pi).to_numpy()
    y = df["output"].to_numpy()
    return {"pos": np.stack([m[y == 1], p[y == 1]], axis=1),
            "neg": np.stack([m[y == 0], p[y == 0]], axis=1)}


def _report(splits):
    for s in ("train", "val", "test"):
        npos, nneg = len(splits[s]["pos"]), len(splits[s]["neg"])
        print(f"  {s:5s}: {npos} pos / {nneg} neg (pos frac {npos/max(npos+nneg,1):.3f})")


def build_transductive(fold, esm, chem):
    """LORAX fold as-is: train/val = full mix, test = EC50-only edges."""
    dfs = _load_fold(fold, esm, chem)
    Xm, Xp, mi, pi = _node_universe(dfs, esm, chem)
    splits = {s: _pairs(dfs[s], mi, pi) for s in ("train", "val", "test")}
    print(f"  nodes: {Xm.shape[0]} molecules, {Xp.shape[0]} proteins")
    _report(splits)
    return Xm, Xp, splits


def build_inductive_molecule(fold, esm, chem, seed=42, test_frac=0.2, val_frac=0.1):
    """Cold-molecule split: hold out whole molecules, test on their EC50 pairs.

    The held-out molecules contribute NO message-passing edges and NO supervision
    (true cold start). Train keeps all pairs (any quality) of remaining molecules.
    """
    dfs = _load_fold(fold, esm, chem)
    pool = pd.concat(dfs.values(), ignore_index=True).drop_duplicates(
        subset=["SMILES", "Protein sequence", "_DataQuality"])
    Xm, Xp, mi, pi = _node_universe(dfs, esm, chem)

    # molecules eligible to be a cold test molecule = those with EC50 data
    ec50 = pool[pool["_DataQuality"] == "ec50"]
    testable = np.array(sorted(ec50["SMILES"].unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(testable))
    n_te = int(test_frac * len(testable)); n_va = int(val_frac * len(testable))
    M_test = set(testable[perm[:n_te]])
    M_val  = set(testable[perm[n_te:n_te + n_va]])
    held   = M_test | M_val

    test_df = ec50[ec50["SMILES"].isin(M_test)]
    val_df  = ec50[ec50["SMILES"].isin(M_val)]
    train_df = pool[~pool["SMILES"].isin(held)]     # all qualities, remaining mols
    splits = {"train": _pairs(train_df, mi, pi),
              "val":   _pairs(val_df, mi, pi),
              "test":  _pairs(test_df, mi, pi)}
    print(f"  nodes: {Xm.shape[0]} molecules, {Xp.shape[0]} proteins | "
          f"cold mols: {len(M_test)} test / {len(M_val)} val of {len(testable)} testable")
    _report(splits)
    return Xm, Xp, splits


def build(regime, fold, esm, chem, seed=42):
    if regime == "transductive":
        return build_transductive(fold, esm, chem)
    if regime == "inductive_molecule":
        return build_inductive_molecule(fold, esm, chem, seed=seed)
    raise ValueError(regime)
