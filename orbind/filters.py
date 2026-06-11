"""M2OR cleaning logic — operates on the FULL relational export (M2OR.zip).

Unlike the GitHub flat dump, the web export ships SMILES inline and explicit
``mixture`` / ``mutation`` / ``species`` fields, so no PubChem/UniProt backfill
is needed. We annotate the pair table with marker columns (non-destructive),
then derive the curated unique (receptor, molecule) pairs.

Filter chain:
    1. human only              -> ``is_human``   (via experiments.species_id)
    2. drop engineered mutants -> ``is_mutant``  (main_receptors.mutation)
    3. keep pure compounds     -> ``is_pure``    (main_compounds.mixture policy)
    4. explicit binary label   -> ``is_binary``  (responsive in {0, 1})
    5. row passes 1-4          -> ``kept_by_us``
    6. dedup to (sequence, inchikey) pairs (label = max responsive)
    7. drop orphan receptors   -> receptor with 0 positive pairs
"""
from __future__ import annotations
import zipfile
import pandas as pd

# which main_compounds.mixture values count as a single "pure" structure
MIXTURE_POLICY = {
    "mono": {"mono"},
    "mono+isomers": {"mono", "sum of isomers"},
}


def _nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def load_zip(path) -> dict:
    z = zipfile.ZipFile(path)
    rd = lambda n: pd.read_csv(z.open(n), sep=";", low_memory=False)
    return {k: rd(f"{k}.csv") for k in
            ["pairs", "main_compounds", "main_receptors", "experiments", "species"]}


def annotate(tabs: dict, mixture_policy: str = "mono") -> pd.DataFrame:
    """Join the relational tables and add boolean marker columns to `pairs`."""
    pairs, mr, mc = tabs["pairs"].copy(), tabs["main_receptors"], tabs["main_compounds"]
    ex, sp = tabs["experiments"], tabs["species"]

    human_id = int(sp.loc[sp["name"].str.contains("homo", case=False), "id"].iloc[0])
    pair_species = ex.groupby("pairs_id")["species_id"].agg(lambda s: s.mode().iloc[0])
    pairs["species_id"] = pairs["id"].map(pair_species)

    pairs = pairs.merge(
        mr[["id", "mutation", "uniprot_id", "sequence"]].rename(columns={"id": "main_receptors_id"}),
        on="main_receptors_id", how="left")
    pairs = pairs.merge(
        mc[["id", "mixture"]].rename(columns={"id": "main_compounds_id"}),
        on="main_compounds_id", how="left")

    keep_mix = MIXTURE_POLICY[mixture_policy]
    pairs["is_human"]  = pairs["species_id"] == human_id
    pairs["is_mutant"] = _nonempty(pairs["mutation"])
    pairs["is_pure"]   = pairs["mixture"].astype(str).str.lower().isin(keep_mix)
    pairs["is_binary"] = pairs["responsive"].isin([0, 1])
    pairs["kept_by_us"] = (pairs["is_human"] & ~pairs["is_mutant"]
                           & pairs["is_pure"] & pairs["is_binary"])
    return pairs


def build_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Dedup kept rows to unique (receptor sequence, molecule) pairs; flag orphans."""
    kept = pairs[pairs["kept_by_us"]].dropna(subset=["sequence", "inchi_key"])
    out = (kept.groupby(["sequence", "inchi_key"], as_index=False)
           .agg(label=("responsive", "max"),
                smiles=("smiles", "first"),
                uniprot_id=("uniprot_id", "first"))
           .rename(columns={"sequence": "receptor", "inchi_key": "inchikey"}))
    pos_per_rec = out.groupby("receptor")["label"].transform("sum")
    out["is_orphan"] = pos_per_rec == 0
    out["kept_final"] = ~out["is_orphan"]
    return out


def summary(pairs: pd.DataFrame) -> dict:
    final = pairs[pairs["kept_final"]]
    n, npos = len(final), int(final["label"].sum())
    return {
        "receptors": int(final["receptor"].nunique()),
        "molecules": int(final["inchikey"].nunique()),
        "pairs": n,
        "positives": npos,
        "pos_rate_%": round(100 * npos / n, 2) if n else 0.0,
        "pos_to_neg": f"1:{(n - npos) / npos:.1f}" if npos else "n/a",
        "smiles_coverage_%": round(100 * final["smiles"].notna().mean(), 1),
    }
