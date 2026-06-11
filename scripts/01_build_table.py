"""Script 1 — load the M2OR table, annotate it with our filter columns, derive
the curated (receptor, molecule, label) pairs, and (optionally) backfill the
missing receptor sequences via Gene ID.

Examples
--------
# offline: annotate + curated pairs using the inline (~75%) sequences
python scripts/01_build_table.py --input ../m2or_official/M2OR_20230428.csv

# also backfill the missing 25% of sequences from UniProt (network)
python scripts/01_build_table.py --input ... --receptor-key "Gene ID" --backfill-seq
"""
import argparse, pathlib, sys
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from orbind import filters as F
from orbind.backfill import gene_to_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to M2OR_20230428.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--receptor-key", default="Gene ID", choices=["Gene ID", "Sequence"])
    ap.add_argument("--backfill-seq", action="store_true",
                    help="fill missing receptor sequences via Gene ID -> UniProt (network)")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep=";", low_memory=False)
    df = F.annotate(df)
    print(f"raw bioassays: {len(df)} | kept_by_us rows: {int(df['kept_by_us'].sum())}")

    # optional: backfill sequences for kept human WT rows lacking inline Sequence
    if args.backfill_seq:
        kept = df[df["kept_by_us"]]
        need = kept[~F._nonempty(kept[F.C_SEQ])][F.C_GENE].dropna().unique()
        print(f"backfilling sequences for {len(need)} genes via UniProt ...")
        g2s = gene_to_sequence(need)
        fill = df[F.C_GENE].map(g2s)
        df.loc[~F._nonempty(df[F.C_SEQ]) & df[F.C_GENE].notna(), F.C_SEQ] = \
            fill[~F._nonempty(df[F.C_SEQ]) & df[F.C_GENE].notna()]
        miss = sum(1 for g in need if not g2s.get(g))
        print(f"  unresolved genes: {miss}/{len(need)}")

    pairs = F.build_pairs(df, receptor_key=args.receptor_key)
    final = pairs[pairs["kept_final"]].copy()
    keep_recs = set(final["receptor"]); keep_mols = set(final["inchikey"])

    # annotated bioassay table (markers) + curated pairs
    df.to_csv(out / "m2or_annotated.csv.gz", index=False, compression="gzip")
    pairs.to_csv(out / "pairs_all_flagged.csv", index=False)
    final[["receptor", "inchikey", "label"]].to_csv(out / "pairs_curated.csv", index=False)

    # join tables the embedding scripts consume: receptor->sequence, molecule->smiles
    kept = df[df["kept_by_us"]]
    rec_seq = (kept[kept[args.receptor_key].isin(keep_recs)]
               .dropna(subset=[F.C_SEQ]).sort_values(F.C_SEQ)
               .groupby(args.receptor_key)[F.C_SEQ].first().reset_index()
               .rename(columns={args.receptor_key: "receptor_id", F.C_SEQ: "sequence"}))
    rec_seq.to_csv(out / "receptor_sequences.csv", index=False)
    mol_smi = (kept[kept[F.C_INCHI].isin(keep_mols)]
               .groupby(F.C_INCHI)[F.C_SMILES].first().reset_index()
               .rename(columns={F.C_INCHI: "inchikey", F.C_SMILES: "smiles"}))
    mol_smi.to_csv(out / "molecule_smiles.csv", index=False)
    print(f"\nreceptor_sequences.csv: {rec_seq['sequence'].notna().sum()}/{len(keep_recs)} have sequence")
    print(f"molecule_smiles.csv:    {mol_smi['smiles'].notna().sum()}/{len(keep_mols)} have SMILES")

    print("\n=== curated summary ===")
    for k, v in F.summary(pairs).items():
        print(f"  {k}: {v}")
    print(f"\nwrote -> {out}/ : pairs_curated.csv, pairs_all_flagged.csv,")
    print(f"         receptor_sequences.csv, molecule_smiles.csv, m2or_annotated.csv.gz")


if __name__ == "__main__":
    main()
