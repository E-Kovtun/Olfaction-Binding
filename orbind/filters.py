"""Deterministic M2OR cleaning logic.

The official M2OR dump (``M2OR_20230428.csv``, ``;``-separated, one row per
*bioassay*) is annotated in place — we never silently drop rows, we add marker
columns so every decision is auditable. The curated *pairs* table is then
derived from the rows we kept.

Filter chain (matches the agreed spec):
    1. human only            -> ``is_human``
    2. drop engineered mutants -> ``is_mutant`` (official ``Mutation`` field)
    3. explicit binary label -> ``is_binary`` (``Responsive`` in {0, 1})
    4. row passes 1-3        -> ``kept_by_us``
    5. dedup to receptor x molecule pairs (label = max Responsive)
    6. drop orphan receptors -> receptor with 0 positive pairs (``is_orphan``)
"""
from __future__ import annotations
import pandas as pd

# --- column names in the official dump -------------------------------------
C_SPECIES = "species"
C_MUT     = "Mutation"
C_GENE    = "Gene ID"
C_UNIPROT = "Uniprot ID"
C_SEQ     = "Sequence"
C_INCHI   = "InChI Key"
C_SMILES  = "canonicalSMILES"
C_RESP    = "Responsive"


def _nonempty(s: pd.Series) -> pd.Series:
    return s.notna() & (s.astype(str).str.strip() != "")


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean marker columns to the raw bioassay table (non-destructive)."""
    df = df.copy()
    df["is_human"]  = df[C_SPECIES].astype(str).str.contains("homo", case=False, na=False)
    df["is_mutant"] = _nonempty(df[C_MUT])
    df["is_binary"] = df[C_RESP].isin([0, 1])
    df["kept_by_us"] = df["is_human"] & (~df["is_mutant"]) & df["is_binary"]
    return df


def build_pairs(df: pd.DataFrame, receptor_key: str = C_GENE) -> pd.DataFrame:
    """Collapse kept bioassays into unique (receptor, molecule) pairs.

    receptor_key: ``"Gene ID"`` (100% coverage) or ``"Sequence"`` (75%, ESM-ready).
    A pair is positive if ANY measurement was responsive (label = max).
    Orphan receptors (no positive pair) are flagged in ``is_orphan``.
    """
    kept = df[df["kept_by_us"]].dropna(subset=[receptor_key, C_INCHI]).copy()
    pairs = (
        kept.groupby([receptor_key, C_INCHI], as_index=False)
        .agg(label=(C_RESP, "max"))
        .rename(columns={receptor_key: "receptor", C_INCHI: "inchikey"})
    )
    pos_per_rec = pairs.groupby("receptor")["label"].transform("sum")
    pairs["is_orphan"] = pos_per_rec == 0
    pairs["kept_final"] = ~pairs["is_orphan"]
    return pairs


def summary(pairs: pd.DataFrame) -> dict:
    final = pairs[pairs["kept_final"]]
    n_pos = int(final["label"].sum())
    n = len(final)
    return {
        "receptors": int(final["receptor"].nunique()),
        "molecules": int(final["inchikey"].nunique()),
        "pairs": n,
        "positives": n_pos,
        "pos_rate_%": round(100 * n_pos / n, 2) if n else 0.0,
        "pos_to_neg": f"1:{(n - n_pos) / n_pos:.1f}" if n_pos else "n/a",
        "orphan_receptors_dropped": int(pairs["is_orphan"].groupby(pairs["receptor"]).first().sum()),
    }
