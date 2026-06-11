"""Script 1 — curate the M2OR pairs from the full relational export (M2OR.zip).

Loads the zip, annotates the pair table with our filter markers, derives the
curated (receptor, molecule, label) pairs with SMILES inline, and writes the
join tables the embedding scripts consume. No network backfill needed.

Usage
-----
uv run python scripts/00_download_m2or.py          # fetch data/raw/M2OR.zip
uv run python scripts/01_build_table.py            # curate (mono compounds only)
uv run python scripts/01_build_table.py --mixture-policy mono+isomers
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from orbind import filters as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/M2OR.zip", help="path to M2OR.zip")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--mixture-policy", default="mono", choices=list(F.MIXTURE_POLICY))
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    tabs = F.load_zip(args.input)
    print(f"loaded M2OR.zip: pairs={len(tabs['pairs'])}, compounds={len(tabs['main_compounds'])}, "
          f"receptors={len(tabs['main_receptors'])}")

    pairs = F.annotate(tabs, mixture_policy=args.mixture_policy)
    print(f"kept_by_us rows: {int(pairs['kept_by_us'].sum())} / {len(pairs)} "
          f"(mixture-policy={args.mixture_policy})")

    curated = F.build_pairs(pairs)
    final = curated[curated["kept_final"]].copy()

    pairs.to_csv(out / "pairs_annotated.csv.gz", index=False, compression="gzip")
    curated.to_csv(out / "pairs_all_flagged.csv", index=False)
    final[["receptor", "inchikey", "label", "smiles"]].to_csv(out / "pairs_curated.csv", index=False)

    # join tables for the embedding scripts (both 100% populated now)
    (final[["receptor"]].drop_duplicates()
        .rename(columns={"receptor": "sequence"}).assign(receptor_id=lambda d: d["sequence"])
        [["receptor_id", "sequence"]]).to_csv(out / "receptor_sequences.csv", index=False)
    (final[["inchikey", "smiles"]].drop_duplicates("inchikey")
        ).to_csv(out / "molecule_smiles.csv", index=False)

    print("\n=== curated summary ===")
    for k, v in F.summary(curated).items():
        print(f"  {k}: {v}")
    print(f"\nwrote -> {out}/ : pairs_curated.csv (+smiles), receptor_sequences.csv, "
          f"molecule_smiles.csv, pairs_annotated.csv.gz")


if __name__ == "__main__":
    main()
