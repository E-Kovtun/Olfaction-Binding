"""Script 07 — molecule tables and imported molecule embeddings for the three
benchmark datasets of the olfactory foundation models release: M2OR, Carey (CC)
and Hallem-Carlson (HC).

Two jobs, because they share the same SMILES -> InChIKey mapping and it must be
computed exactly once:

1. Write `data/processed/molecules/molecule_smiles_{tag}.csv` (columns
   `inchikey,smiles`) — the input every downstream molecule embedder wants,
   including `embed_molecules_gin.py`.

2. Convert upstream's featurized molecules to our npz convention. Their pickles
   are `{SMILES: float64[d]}`; ours are `ids=<inchikey>, emb=float32[n, d]`.
   The rekeying is the whole point: our pairs tables address molecules by
   InChIKey, theirs by the raw SMILES string.

Why InChIKey and not the SMILES string
--------------------------------------
It is the identifier the rest of this repo uses, and it is canonical — the same
molecule written two ways collapses to one key. That collapsing is also the one
thing that can go wrong here, so the script refuses to write a table where two
distinct SMILES map to one InChIKey rather than silently dropping a row.

Where the inputs come from
--------------------------
https://zenodo.org/records/17228740 -> data.zip, whose `data/` folder is unpacked
as `data/external/ofm/` (gitignored):
    {CC,HC}/raw/*.csv                              the insect response tables
    {CC,HC,M2OR}/embeddings/featurized_mols/*.pkl  upstream's molecule features
M2OR's molecule list is not read from a raw table but from the LORAX pool the
whole pipeline uses (`orbind.regimes.load_full_full_pool`, 596 molecules), so the
table and the npz cover exactly the molecules every M2OR run sees. Its release
folder is looked up as `M2OR` and then `M2OR_full`; `--release-dir` names it
explicitly.

Writing over an existing file
-----------------------------
An npz that already exists is replaced only if the new vectors are identical to it
(same keys, max |difference| below 1e-6). Otherwise the new file is written beside
it as `<name>.new.npz`, the largest difference is printed, and the old file is left
alone: every number computed on it would move with it. `--overwrite` replaces it
anyway.

Run
---
    .venv/bin/python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag m2or
    .venv/bin/python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag cc
    .venv/bin/python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag hc
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

OFM_DIR = _root / "data" / "external" / "ofm"
MOL_DIR = _root / "data" / "processed" / "molecules"
EMB_DIR = _root / "data" / "embeddings" / "molecules"

# Upstream's raw response table per insect dataset, and the SMILES column in it.
RAW = {
    "cc": pathlib.Path("CC") / "raw" / "CC_reformat_z.csv",
    "hc": pathlib.Path("HC") / "raw" / "hc_with_prot_seq_z.csv",
}
TAGS = ("m2or", "cc", "hc")
# Where each dataset's featurized_mols live inside the release, first match wins.
RELEASE_DIRS = {"m2or": ("M2OR", "M2OR_full"), "cc": ("CC",), "hc": ("HC",)}
# featurized_mols pickle -> our npz basename. Both are keyed by SMILES upstream.
IMPORTS = {
    "ChemBERTa-77M-MTR": "chemberta_77m",
    "gin_supervised_contextpred": "gin_supervised_contextpred_ofm",
}
SAME_TOL = 1e-6


def inchikey_of(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def dataset_smiles(tag: str) -> list[str]:
    """Every distinct SMILES of the dataset, from the table the pipeline itself reads."""
    if tag == "m2or":
        from orbind.regimes import load_full_full_pool
        df = load_full_full_pool()
        print(f"m2or: {len(df)} pool rows (LORAX rand_split_1 train+val+test)")
    else:
        raw = OFM_DIR / RAW[tag]
        if not raw.exists():
            raise FileNotFoundError(f"{raw} not found -- see this script's docstring")
        df = pd.read_csv(raw)
        print(f"{tag}: {len(df)} rows")
    return sorted(set(df["SMILES"].dropna().astype(str)))


def build_table(tag: str) -> pd.DataFrame:
    smiles = dataset_smiles(tag)
    print(f"{tag}: {len(smiles)} unique SMILES")

    keys = {s: inchikey_of(s) for s in smiles}
    unparsed = [s for s, k in keys.items() if k is None]
    if unparsed:
        raise ValueError(f"RDKit could not parse {len(unparsed)} SMILES, e.g. {unparsed[:3]}")

    out = pd.DataFrame({"inchikey": [keys[s] for s in smiles], "smiles": smiles})
    dup = out[out.duplicated("inchikey", keep=False)].sort_values("inchikey")
    if not dup.empty:
        raise ValueError(
            f"{dup['inchikey'].nunique()} InChIKeys are shared by several SMILES; "
            f"rekeying would lose rows:\n{dup.to_string(index=False)}"
        )
    return out


def release_dir(tag: str, override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    for name in RELEASE_DIRS[tag]:
        d = OFM_DIR / name
        if d.exists():
            return d
    raise FileNotFoundError(f"none of {RELEASE_DIRS[tag]} under {OFM_DIR} -- see this "
                            f"script's docstring, or pass --release-dir")


def save_checked(out: pathlib.Path, ids: np.ndarray, emb: np.ndarray, overwrite: bool) -> None:
    """Write `out`, unless it exists and the new vectors differ from it (see docstring)."""
    if out.exists() and not overwrite:
        old = np.load(out, allow_pickle=True)
        old_map = dict(zip(map(str, old["ids"]), old["emb"]))
        new_map = dict(zip(map(str, ids), emb))
        if set(old_map) == set(new_map):
            diff = max(float(np.abs(old_map[k] - new_map[k]).max()) for k in new_map)
            if diff <= SAME_TOL:
                print(f"  {out.name}: identical to the existing file "
                      f"(max |diff| {diff:.1e}), kept as it is")
                return
            why = f"same {len(new_map)} keys, max |diff| {diff:.3e}"
        else:
            why = (f"keys differ: {len(set(new_map) - set(old_map))} new, "
                   f"{len(set(old_map) - set(new_map))} missing")
        side = out.with_name(out.stem + ".new.npz")
        np.savez_compressed(side, ids=ids, emb=emb)
        print(f"  {out.name}: DIFFERS from the existing file ({why}). Left it alone and "
              f"wrote {side.name}; --overwrite replaces it")
        return
    np.savez_compressed(out, ids=ids, emb=emb)
    print(f"  -> {out.name}  ({emb.shape[0]} x {emb.shape[1]}-d)")


def import_embeddings(tag: str, table: pd.DataFrame, src_root: pathlib.Path,
                      overwrite: bool) -> None:
    smi2key = dict(zip(table["smiles"], table["inchikey"]))
    src_dir = src_root / "embeddings" / "featurized_mols"
    for stem, out_stem in IMPORTS.items():
        src = src_dir / f"{stem}.pkl"
        if not src.exists():
            print(f"  (no {src} , skipped)")
            continue
        with open(src, "rb") as fh:
            raw = pickle.load(fh)

        # Keys arrive as numpy str_; normalise before looking them up.
        raw = {str(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        missing = [s for s in table["smiles"] if s not in raw]
        extra = [s for s in raw if s not in smi2key]
        if missing:
            raise ValueError(f"{src.name}: {len(missing)} dataset SMILES absent from the "
                             f"pickle, e.g. {missing[:3]}")
        if extra:
            print(f"  {src.name}: ignoring {len(extra)} pickle entries not in the dataset")

        ids = table["inchikey"].to_numpy()
        emb = np.stack([raw[s] for s in table["smiles"]]).astype(np.float32)
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        save_checked(EMB_DIR / f"{out_stem}_{tag}.npz", ids, emb, overwrite)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=TAGS, help="dataset to prepare")
    ap.add_argument("--no-import", action="store_true",
                    help="only write the molecule table, do not convert the pickles")
    ap.add_argument("--release-dir", default=None,
                    help="the dataset's folder of the release, if not under "
                         "data/external/ofm/ (it must hold embeddings/featurized_mols/)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing npz even if the new vectors differ")
    args = ap.parse_args()

    table = build_table(args.tag)
    MOL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MOL_DIR / f"molecule_smiles_{args.tag}.csv"
    table.to_csv(out_csv, index=False)
    print(f"-> {out_csv}  ({len(table)} molecules)")

    if not args.no_import:
        src_root = release_dir(args.tag, args.release_dir)
        print(f"importing from {src_root}")
        import_embeddings(args.tag, table, src_root, args.overwrite)


if __name__ == "__main__":
    main()
