"""Generate ECFP (Morgan) fingerprint embeddings for a dataset's molecules.

Output matches the ChemBERTa / GIN npz format exactly -- two arrays `ids`
(inchikeys) and `emb` (float32 [n, nbits]) -- so it drops straight into the
ensemble pipeline as another `mol=gin:<npz>` source and into
`orbind.dataset.load_npz_dict`.

By default the molecule set is the dataset's canonical pool (for m2or: the 596
unique molecules of `full_full_pairs`, i.e. the exact ChemBERTa coverage). The
script also writes that (inchikey, smiles) list to a CSV, which is the SAME input
`embed_molecules_gin.py --molecules` takes -- so GIN can be regenerated on the
identical set for full parity (see the audit script).

    # server, .venv (rdkit):
    python scripts/embedding_generation/molecules/embed_molecules_ecfp.py --dataset m2or
    # then, for GIN parity on the identical 596:
    python scripts/embedding_generation/molecules/embed_molecules_gin.py \
        --molecules data/processed/molecules/m2or_molecules.csv \
        --out data/embeddings/molecules/gin_supervised_contextpred_m2or.npz
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))


def canonical_molecules(dataset: str) -> pd.DataFrame:
    """(inchikey, smiles) for the dataset's pool -- the exact set the current
    M2OR embeddings must cover."""
    if dataset == "m2or":
        from orbind.regimes import full_full_pairs
        p = full_full_pairs(pool_fold=1)
        df = (p.drop_duplicates("inchikey")[["inchikey", "smiles"]]
              .dropna(subset=["smiles"]).reset_index(drop=True))
        return df
    if dataset in ("cc", "hc"):
        bridge = _root / "data" / "processed" / "molecules" / f"molecule_smiles_{dataset}.csv"
        df = pd.read_csv(bridge)
        return (df.dropna(subset=["smiles"]).drop_duplicates("inchikey")[["inchikey", "smiles"]]
                .reset_index(drop=True))
    raise SystemExit(f"canonical molecule set for dataset={dataset!r} not wired (m2or/cc/hc); "
                     f"pass --molecules <csv with inchikey,smiles> instead")


def morgan_fps(smiles, radius: int, nbits: int, counts: bool):
    """Morgan/ECFP fingerprints as float32 arrays. Returns (list_of_vec_or_None, ok_mask).
    Uses the modern MorganGenerator, falling back to the legacy API."""
    from rdkit import Chem
    try:
        from rdkit.Chem import rdFingerprintGenerator as rfg
        gen = rfg.GetMorganGenerator(radius=radius, fpSize=nbits)
        fp = (gen.GetCountFingerprintAsNumPy if counts else gen.GetFingerprintAsNumPy)
    except Exception:                                   # older rdkit
        from rdkit.Chem import AllChem
        from rdkit import DataStructs

        def fp(mol):
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            a = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(bv, a)
            return a

    out, ok = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append(None); ok.append(False); continue
        out.append(np.asarray(fp(m), dtype=np.float32)); ok.append(True)
    return out, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="m2or")
    ap.add_argument("--molecules", default=None, help="override: csv with columns inchikey, smiles")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (2 = ECFP4)")
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--counts", action="store_true", help="count fingerprint instead of binary")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mol-list-out", default=None,
                    help="where to write the (inchikey, smiles) csv (default data/processed/molecules/<dataset>_molecules.csv)")
    args = ap.parse_args()

    if args.molecules:
        df = pd.read_csv(args.molecules).dropna(subset=["smiles"]).drop_duplicates("inchikey").reset_index(drop=True)
    else:
        df = canonical_molecules(args.dataset)
        list_out = pathlib.Path(args.mol_list_out) if args.mol_list_out else (
            _root / "data" / "processed" / "molecules" / f"{args.dataset}_molecules.csv")
        list_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(list_out, index=False)
        print(f"canonical molecule list ({len(df)}) -> {list_out}", flush=True)

    print(f"featurizing {len(df)} molecules as ECFP (radius={args.radius}, nbits={args.nbits}, "
          f"{'counts' if args.counts else 'binary'})...", flush=True)
    fps, ok = morgan_fps(df["smiles"].tolist(), args.radius, args.nbits, args.counts)
    ok = np.asarray(ok)
    if not ok.all():
        bad = df.loc[~ok, "inchikey"].tolist()
        print(f"  WARNING: {len(bad)} SMILES failed to parse -> dropped: {bad[:10]}"
              f"{' ...' if len(bad) > 10 else ''}", flush=True)

    ids = df.loc[ok, "inchikey"].to_numpy()
    emb = np.stack([f for f, o in zip(fps, ok) if o]).astype(np.float32)

    out = pathlib.Path(args.out) if args.out else (
        _root / "data" / "embeddings" / "molecules" / f"ecfp_{args.dataset}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids, dtype=object), emb=emb)
    print(f"saved {emb.shape[0]} x {emb.shape[1]} float32 -> {out}", flush=True)
    print(f"next: audit parity with\n"
          f"  python scripts/embedding_generation/molecules/audit_molecule_npz.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
