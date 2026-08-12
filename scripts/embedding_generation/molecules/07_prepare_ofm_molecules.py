"""Script 07 — molecule tables and imported molecule embeddings for the two
non-M2OR benchmark datasets shipped with the olfactory foundation models
release: Carey (CC) and Hallem-Carlson (HC).

Two jobs, because they share the same SMILES -> InChIKey mapping and it must be
computed exactly once:

1. Write `data/processed/molecules/molecule_smiles_{tag}.csv` (columns
   `inchikey,smiles`) — the input every downstream molecule embedder wants,
   including `embed_molecules_gin.py`.

2. Convert upstream's featurized molecules to our npz convention. Their pickles
   are `{SMILES: float64[d]}`; ours are `ids=<inchikey>, emb=float32[n, d]`
   (see `chemberta_77m_m2or.npz`). The rekeying is the whole point: our pairs
   tables address molecules by InChIKey, theirs by the raw SMILES string.

Why InChIKey and not the SMILES string
--------------------------------------
It is the identifier the rest of this repo uses, and it is canonical — the same
molecule written two ways collapses to one key. That collapsing is also the one
thing that can go wrong here, so the script refuses to write a table where two
distinct SMILES map to one InChIKey rather than silently dropping a row.

Where the inputs come from
--------------------------
https://zenodo.org/records/17228740 -> data.zip, members
    {CC,HC}/raw/*.csv
    {CC,HC}/embeddings/featurized_mols/*.pkl
extracted under `data/external/ofm/` (gitignored).

Run
---
    uv run python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag cc
    uv run python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag hc
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

# Upstream's raw response table per dataset, and the SMILES column in it.
RAW = {
    "cc": pathlib.Path("CC") / "raw" / "CC_reformat_z.csv",
    "hc": pathlib.Path("HC") / "raw" / "hc_with_prot_seq_z.csv",
}
# featurized_mols pickle -> our npz basename. Both are keyed by SMILES upstream.
IMPORTS = {
    "ChemBERTa-77M-MTR": "chemberta_77m",
    "gin_supervised_contextpred": "gin_supervised_contextpred_ofm",
}


def inchikey_of(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def build_table(tag: str) -> pd.DataFrame:
    raw = OFM_DIR / RAW[tag]
    if not raw.exists():
        raise FileNotFoundError(f"{raw} not found -- see this script's docstring")
    df = pd.read_csv(raw)
    smiles = sorted(set(df["SMILES"].dropna().astype(str)))
    print(f"{tag}: {len(df)} rows, {len(smiles)} unique SMILES")

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


def import_embeddings(tag: str, table: pd.DataFrame) -> None:
    smi2key = dict(zip(table["smiles"], table["inchikey"]))
    src_dir = OFM_DIR / tag.upper() / "embeddings" / "featurized_mols"
    for stem, out_stem in IMPORTS.items():
        src = src_dir / f"{stem}.pkl"
        if not src.exists():
            print(f"  (no {src.name} for {tag.upper()}, skipped)")
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
        out = EMB_DIR / f"{out_stem}_{tag}.npz"
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, ids=ids, emb=emb)
        print(f"  {src.name} -> {out.name}  ({emb.shape[0]} x {emb.shape[1]}-d)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(RAW), help="dataset to prepare")
    ap.add_argument("--no-import", action="store_true",
                    help="only write the molecule table, do not convert the pickles")
    args = ap.parse_args()

    table = build_table(args.tag)
    MOL_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = MOL_DIR / f"molecule_smiles_{args.tag}.csv"
    table.to_csv(out_csv, index=False)
    print(f"-> {out_csv}  ({len(table)} molecules)")

    if not args.no_import:
        import_embeddings(args.tag, table)


if __name__ == "__main__":
    main()
