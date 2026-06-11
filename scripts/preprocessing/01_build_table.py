"""Script 1 — curate the M2OR pairs from the full relational export (M2OR.zip).

Loads the zip, annotates the pair table with our filter markers, derives the
curated (receptor, molecule, label) pairs with SMILES inline, and writes the
join tables the embedding scripts consume. No network backfill needed.

Usage
-----
uv run python scripts/downloading/00_download_m2or.py     # fetch data/raw/M2OR.zip
uv run python scripts/preprocessing/01_build_table.py     # curate (mono compounds only)
uv run python scripts/preprocessing/01_build_table.py --mixture-policy mono+isomers
"""
import argparse, pathlib, sys
# locate repo root (depth-independent) so `orbind` imports regardless of script nesting
_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import filters as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/M2OR.zip", help="path to M2OR.zip")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--mixture-policy", default="mono", choices=list(F.MIXTURE_POLICY))
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    prot = out / "proteins"; mol = out / "molecules"
    for d in (out, prot, mol):
        d.mkdir(parents=True, exist_ok=True)

    tabs = F.load_zip(args.input)
    print(f"loaded M2OR.zip: pairs={len(tabs['pairs'])}, compounds={len(tabs['main_compounds'])}, "
          f"receptors={len(tabs['main_receptors'])}")

    pairs = F.annotate(tabs, mixture_policy=args.mixture_policy)
    print(f"kept_by_us rows: {int(pairs['kept_by_us'].sum())} / {len(pairs)} "
          f"(mixture-policy={args.mixture_policy})")

    curated = F.build_pairs(pairs)
    final = curated[curated["kept_final"]].copy()

    # shared pair tables at processed root
    pairs.to_csv(out / "pairs_annotated.csv.gz", index=False, compression="gzip")
    curated.to_csv(out / "pairs_all_flagged.csv", index=False)
    final[["receptor", "inchikey", "label", "smiles"]].to_csv(out / "pairs_curated.csv", index=False)

    # per-modality join tables the embedding scripts consume (both 100% populated)
    (final[["receptor"]].drop_duplicates()
        .rename(columns={"receptor": "sequence"}).assign(receptor_id=lambda d: d["sequence"])
        [["receptor_id", "sequence"]]).to_csv(prot / "receptor_sequences.csv", index=False)
    (final[["inchikey", "smiles"]].drop_duplicates("inchikey")
        ).to_csv(mol / "molecule_smiles.csv", index=False)

    print("\n=== curated summary ===")
    for k, v in F.summary(curated).items():
        print(f"  {k}: {v}")
    print(f"\nwrote -> {out}/pairs_curated.csv, {prot}/receptor_sequences.csv, "
          f"{mol}/molecule_smiles.csv")


if __name__ == "__main__":
    main()
