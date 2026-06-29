"""Script 02 — assign Ballesteros-Weinstein generic numbering to all curated receptors.

Strategy
--------
For each receptor, we pick the closest subfamily reference from GPCRDB (17 human OR
subfamily representatives, one per family) by alignment score, then transfer BW numbers
via column-wise mapping. This avoids the single-reference bias that caused gaps at TM5/TM7
boundaries for distant subfamilies.

Output
------
data/processed/bw_numbering_curated.csv
    long format: receptor, bw_number, amino_acid
    (all BW-annotated positions, not just the 22-pocket ones)

data/processed/bw_pocket_matrix_curated.csv
    wide format: receptor x 22 pocket BW positions -> amino acid (or NaN)

data/processed/bw_ref_used_curated.csv
    which subfamily reference was chosen for each receptor + alignment identity

Usage
-----
uv run python scripts/preprocessing/02_bw_numbering.py
"""
from __future__ import annotations
import pathlib, sys
import requests
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# 17 human OR subfamily representatives (one per GPCRDB subfamily slug)
# ---------------------------------------------------------------------------
OR_SUBFAMILY_ENTRIES = [
    "o51a2_human", "o52a1_human", "o56a1_human",  # class O1 (fish-like)
    "or1a1_human", "or2a1_human", "or3a1_human",   # class O2 families 1-3
    "or4a5_human", "or5a1_human", "or6a2_human",
    "or7a5_human", "or8a1_human", "or9a2_human",
    "o10a2_human", "o11a1_human", "o12d1_human",
    "o13a1_human", "o14a2_human",
]

# 22 canonical OR binding-pocket positions (matched by "helix.pos" prefix;
# the "x" suffix is OR-class specific and resolved from each reference)
POCKET_PREFIXES = [
    "2.53", "2.57",
    "3.28", "3.29", "3.32", "3.33", "3.36", "3.37",
    "4.57", "4.61",
    "5.38", "5.39", "5.42", "5.43", "5.46",
    "6.44", "6.48", "6.51", "6.52", "6.55",
    "7.35", "7.39",
]

GPCRDB = "https://gpcrdb.org/services"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_aligner() -> PairwiseAligner:
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score    =  2.0
    a.mismatch_score = -1.0
    a.open_gap_score   = -5.0
    a.extend_gap_score = -0.5
    return a


def fetch_reference(entry: str) -> tuple[str, dict[int, str], list[str]]:
    """Return (sequence, {seq_pos: bw_full}, pocket_bw_keys) for a GPCRDB entry."""
    prot = requests.get(f"{GPCRDB}/protein/{entry}/", timeout=30).json()
    seq  = prot["sequence"]

    res_data = requests.get(f"{GPCRDB}/residues/{entry}/", timeout=30).json()
    bw_map: dict[int, str] = {}
    prefix2full: dict[str, str] = {}
    for r in res_data:
        gn = r.get("display_generic_number")
        if gn:
            bw_map[r["sequence_number"]] = gn
            prefix2full[gn.split("x")[0]] = gn

    pocket_keys = [prefix2full[p] for p in POCKET_PREFIXES if p in prefix2full]
    missing = [p for p in POCKET_PREFIXES if p not in prefix2full]
    if missing:
        print(f"    [{entry}] missing pocket prefixes: {missing}")

    return seq, bw_map, pocket_keys


def alignment_score(aligner: PairwiseAligner, ref: str, target: str) -> float:
    """Raw alignment score (higher = better match)."""
    return aligner.score(ref, target)


def align_and_transfer(
    aligner: PairwiseAligner,
    ref_seq: str,
    bw_map: dict[int, str],
    target: str,
) -> tuple[dict[str, str], dict[str, int], float]:
    """Return ({bw: aa}, {bw: 0-based-pos}, pct_identity)."""
    aln = next(iter(aligner.align(ref_seq, target)))
    ref_blocks, tgt_blocks = aln.aligned

    # ref_1based -> tgt_1based
    ref2tgt: dict[int, int] = {}
    matches = 0
    aligned_len = 0
    for (rs, re), (ts, te) in zip(ref_blocks, tgt_blocks):
        for ri, ti in zip(range(rs, re), range(ts, te)):
            ref2tgt[ri + 1] = ti + 1
            aligned_len += 1
            if ref_seq[ri] == target[ti]:
                matches += 1

    pct_id = matches / aligned_len if aligned_len else 0.0

    result_aa:  dict[str, str] = {}
    result_pos: dict[str, int] = {}
    for ref_pos, bw in bw_map.items():
        tgt_pos = ref2tgt.get(ref_pos)
        if tgt_pos is not None and 1 <= tgt_pos <= len(target):
            result_aa[bw]  = target[tgt_pos - 1]
            result_pos[bw] = tgt_pos - 1   # 0-based

    return result_aa, result_pos, pct_id


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    seqs_df = pd.read_csv(OUT / "proteins" / "receptor_sequences.csv")
    sequences = seqs_df["sequence"].tolist()
    print(f"Receptors to process : {len(sequences)}")
    print(f"Subfamily references : {len(OR_SUBFAMILY_ENTRIES)}")

    # --- fetch all references -----------------------------------------------
    print("\nFetching subfamily references from GPCRDB...")
    refs: dict[str, tuple[str, dict[int, str], list[str]]] = {}
    for entry in OR_SUBFAMILY_ENTRIES:
        seq, bw_map, pocket_keys = fetch_reference(entry)
        refs[entry] = (seq, bw_map, pocket_keys)
        print(f"  {entry}: len={len(seq)} BW={len(bw_map)} pocket={len(pocket_keys)}/22")

    # unified pocket column order: use OR1A1 (most complete) as canonical
    aligner = _make_aligner()
    ref_seqs = {e: refs[e][0] for e in OR_SUBFAMILY_ENTRIES}

    # --- process each receptor ----------------------------------------------
    print("\nProcessing receptors...")
    rows_long:    list[dict] = []
    matrix_rows:  list[dict] = []
    pos_rows:     list[dict] = []
    ref_used_rows: list[dict] = []

    for i, seq in enumerate(sequences):
        # pick best-matching reference by alignment score
        scores = {e: alignment_score(aligner, refs[e][0], seq) for e in OR_SUBFAMILY_ENTRIES}
        best_entry = max(scores, key=lambda e: scores[e])
        best_ref_seq, best_bw_map, best_pocket_keys = refs[best_entry]

        bw_result, bw_positions, pct_id = align_and_transfer(
            aligner, best_ref_seq, best_bw_map, seq)

        # collect pocket column names from best reference
        for bw, aa in bw_result.items():
            rows_long.append({"receptor": seq, "bw_number": bw, "amino_acid": aa,
                               "seq_pos": bw_positions.get(bw, None)})

        mat_row = {"receptor": seq}
        mat_row.update({bw: bw_result.get(bw, np.nan) for bw in best_pocket_keys})
        matrix_rows.append(mat_row)

        pos_row = {"receptor": seq}
        pos_row.update({bw: bw_positions.get(bw, np.nan) for bw in best_pocket_keys})
        pos_rows.append(pos_row)

        ref_used_rows.append({
            "receptor": seq,
            "ref_entry": best_entry,
            "pct_identity": round(pct_id * 100, 1),
            "alignment_score": round(scores[best_entry], 1),
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(sequences):
            print(f"  {i+1}/{len(sequences)} done")

    # --- save ---------------------------------------------------------------
    df_long = pd.DataFrame(rows_long)
    df_long.to_csv(OUT / "bw_numbering_curated.csv", index=False)
    print(f"\nSaved bw_numbering_curated.csv  ({len(df_long):,} rows)")

    # pocket matrix: align all columns across all references
    # different refs may use slightly different x-suffixes, canonicalize to prefix
    def to_prefix(bw: str) -> str:
        return bw.split("x")[0]

    # build canonical column order by prefix
    all_pocket_prefixes_seen: list[str] = []
    seen_set: set[str] = set()
    for row in matrix_rows:
        for k in row:
            if k == "receptor":
                continue
            pfx = to_prefix(k)
            if pfx not in seen_set:
                all_pocket_prefixes_seen.append(pfx)
                seen_set.add(pfx)

    # re-index matrix using prefixes as canonical keys
    canonical_rows: list[dict] = []
    for row, mat_row in zip(ref_used_rows, matrix_rows):
        crow: dict = {"receptor": mat_row["receptor"]}
        pfx2aa: dict[str, str] = {to_prefix(k): v for k, v in mat_row.items() if k != "receptor"}
        for pfx in POCKET_PREFIXES:
            crow[pfx] = pfx2aa.get(pfx, np.nan)
        canonical_rows.append(crow)

    df_mat = pd.DataFrame(canonical_rows).set_index("receptor")
    df_mat.to_csv(OUT / "bw_pocket_matrix_curated.csv")
    print(f"Saved bw_pocket_matrix_curated.csv  ({df_mat.shape[0]} x {df_mat.shape[1]})")

    df_ref = pd.DataFrame(ref_used_rows)
    df_ref.to_csv(OUT / "bw_ref_used_curated.csv", index=False)
    print(f"Saved bw_ref_used_curated.csv")

    # pocket positions (0-based indices into the receptor sequence)
    pos_canonical: list[dict] = []
    for ref_row, pos_row in zip(ref_used_rows, pos_rows):
        crow: dict = {"receptor": pos_row["receptor"]}
        to_pfx = lambda k: k.split("x")[0]
        pfx2pos: dict[str, int] = {to_pfx(k): v for k, v in pos_row.items()
                                   if k != "receptor" and not pd.isna(v)}
        for pfx in POCKET_PREFIXES:
            crow[pfx] = pfx2pos.get(pfx, np.nan)
        pos_canonical.append(crow)
    df_pos = pd.DataFrame(pos_canonical).set_index("receptor")
    df_pos.to_csv(OUT / "bw_pocket_positions_curated.csv")
    print(f"Saved bw_pocket_positions_curated.csv  ({df_pos.shape[0]} x {df_pos.shape[1]})")

    # --- summary ------------------------------------------------------------
    print("\nCoverage per pocket position (fraction non-NaN):")
    print(df_mat.notna().mean().round(3).to_string())

    print("\nReference usage:")
    print(df_ref["ref_entry"].value_counts().to_string())

    print("\nAlignment identity stats:")
    print(df_ref["pct_identity"].describe().round(1).to_string())


if __name__ == "__main__":
    main()
