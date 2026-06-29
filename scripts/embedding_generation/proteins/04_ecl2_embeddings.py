"""Script 04 — ECL2-focused ESM-2 embeddings for olfactory receptors.

Olfactory receptors are class A GPCRs.  The ligand-binding pocket is located
in extracellular loop 2 (ECL2, between TM4 and TM5) — not spread over the
entire ~310-residue sequence.  Mean-pooling ESM-2 over ALL residues dilutes
this signal: the dominant component is the conserved GPCR backbone.

This script:
  1. Predicts TM-helix positions locally via a Kyte-Doolittle hydrophobicity
     sliding window (no network calls required).
  2. Extracts ECL2 = gap between TM4 and TM5.
  3. Runs ESM-2 650M per-residue (batched), mean-pools only over the ECL2
     slice, and saves one 1280-d vector per receptor.

Output: data/embeddings/proteins/esm2_650m_ecl2_mean_curated.npz
        (same format as esm2_650m_mean_curated.npz — keys ids / emb)

Usage
-----
    uv run python scripts/embedding_generation/proteins/04_ecl2_embeddings.py \\
        --sequences data/processed/proteins/receptor_sequences.csv \\
        --out       data/embeddings/proteins/esm2_650m_ecl2_mean_curated.npz
"""
import argparse, pathlib, sys
import numpy as np
import pandas as pd

# ── TM helix detection ────────────────────────────────────────────────────────

# Kyte-Doolittle hydrophobicity scale
_KD = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}


def _hydrophobicity_profile(seq: str, window: int = 19) -> np.ndarray:
    h = np.array([_KD.get(aa, 0.0) for aa in seq], dtype=np.float32)
    n = len(h)
    k = window // 2
    profile = np.convolve(h, np.ones(window) / window, mode='same')
    # zero out edge artefacts
    profile[:k] = 0.0
    profile[n - k:] = 0.0
    return profile


def _find_tm_helices(
    profile: np.ndarray,
    n_tm: int = 7,
    threshold: float = 1.3,
    min_width: int = 12,
    min_gap: int = 4,
) -> list[tuple[int, int]]:
    """Return list of (start, end) TM helix positions, 1-indexed inclusive.

    Strategy:
      1. Threshold the profile to find hydrophobic runs.
      2. Merge runs separated by < min_gap residues.
      3. Keep only runs of width >= min_width.
      4. If we get more than n_tm, keep the n_tm highest-scoring ones.
      5. If we get fewer, lower the threshold by 0.15 and retry (max 4 attempts).
    """
    for attempt in range(5):
        above = profile >= threshold
        runs: list[list] = []
        i = 0
        while i < len(profile):
            if above[i]:
                j = i
                while j < len(profile) and above[j]:
                    j += 1
                runs.append([i, j - 1, float(profile[i:j].mean())])
                i = j
            else:
                i += 1

        # merge close runs
        merged: list[list] = []
        for run in runs:
            if merged and run[0] - merged[-1][1] <= min_gap:
                merged[-1][1] = run[1]
                merged[-1][2] = max(merged[-1][2], run[2])
            else:
                merged.append(run)

        # drop too-narrow runs
        merged = [r for r in merged if r[1] - r[0] + 1 >= min_width]

        if len(merged) >= n_tm:
            # take the n_tm highest-scoring, re-sort by position
            merged.sort(key=lambda r: -r[2])
            merged = sorted(merged[:n_tm], key=lambda r: r[0])
            return [(r[0] + 1, r[1] + 1) for r in merged]  # → 1-indexed

        # too few helices: lower threshold and retry
        threshold -= 0.15

    # absolute fallback: return whatever we found, padded if necessary
    merged.sort(key=lambda r: r[0])
    return [(r[0] + 1, r[1] + 1) for r in merged]


def ecl2_from_sequence(seq: str) -> tuple[int, int]:
    """Predict ECL2 residue range (1-indexed, inclusive) for an OR sequence."""
    profile = _hydrophobicity_profile(seq, window=19)
    tms = _find_tm_helices(profile)

    if len(tms) >= 5:
        # ECL2 = gap between TM4 and TM5
        tm4_end = tms[3][1]
        tm5_start = tms[4][0]
        if tm5_start > tm4_end + 1:
            return (tm4_end + 1, tm5_start - 1)

    # fallback: 50–60 % of sequence length (reliable for ~310 aa ORs)
    n = len(seq)
    return (round(n * 0.50), round(n * 0.60))


# ── ESM-2 per-residue → ECL2 mean-pool ───────────────────────────────────────

def embed_ecl2(
    ids: list[str],
    seqs: list[str],
    ecl2_pos: dict[str, tuple[int, int]],
    version: str = "650m",
    batch: int = 4,
) -> list[np.ndarray]:
    """Run ESM-2 per-residue; return mean over ECL2 slice per sequence (1280-d)."""
    import esm, torch

    layer = 33 if version == "650m" else 36
    loader = {
        "650m": esm.pretrained.esm2_t33_650M_UR50D,
        "3B":   esm.pretrained.esm2_t36_3B_UR50D,
    }[version]

    print(f"loading ESM-2 {version} …")
    model, alphabet = loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    bc = alphabet.get_batch_converter()

    reps: list[np.ndarray] = []
    for i in range(0, len(ids), batch):
        chunk_ids = ids[i : i + batch]
        chunk_seqs = [s[:1600] for s in seqs[i : i + batch]]
        data = [(rid, seq) for rid, seq in zip(chunk_ids, chunk_seqs)]
        _, _, toks = bc(data)
        lens = (toks != alphabet.padding_idx).sum(1)

        with torch.inference_mode():
            out = model(toks.to(device), repr_layers=[layer])
        r = out["representations"][layer]   # [B, L+2, hidden]

        for k, rid in enumerate(chunk_ids):
            seq_len = int(lens[k]) - 2          # strip BOS + EOS
            ecl2_s, ecl2_e = ecl2_pos[rid]
            s = max(1, ecl2_s)
            e = min(ecl2_e, seq_len)
            if s > e:                           # safety: use full sequence
                s, e = 1, seq_len
            # r[k, pos] corresponds to 1-indexed position pos
            ecl2_rep = r[k, s : e + 1, :].mean(0).cpu().numpy()
            reps.append(ecl2_rep)

        print(f"  embedded {min(i + batch, len(ids))}/{len(ids)}", end="\r")
    print()
    return reps


# ── main ──────────────────────────────────────────────────────────────────────

def _ecl2_pos_from_bw(bw_csv: pathlib.Path, ids: list[str]) -> dict[str, tuple[int, int]]:
    """Return {receptor_id: (ecl2_start_1based, ecl2_end_1based)} from BW numbering.

    ECL2 = residues strictly between the last TM4 position and first TM5 position.
    seq_pos column in bw_numbering_curated.csv is 0-based; we convert to 1-based here.
    """
    bw = pd.read_csv(bw_csv)
    bw["helix"] = bw["bw_number"].str.split(".").str[0]
    bw = bw.dropna(subset=["seq_pos"])
    bw["seq_pos"] = bw["seq_pos"].astype(int)

    tm4_end  = bw[bw["helix"] == "4"].groupby("receptor")["seq_pos"].max()
    tm5_start = bw[bw["helix"] == "5"].groupby("receptor")["seq_pos"].min()

    result: dict[str, tuple[int, int]] = {}
    for rid in ids:
        if rid in tm4_end.index and rid in tm5_start.index:
            s = int(tm4_end[rid]) + 2   # 1-based, exclusive of last TM4 residue
            e = int(tm5_start[rid])     # 1-based, exclusive of first TM5 residue
            if e > s:
                result[rid] = (s, e)
    return result


def main():
    _root = pathlib.Path(__file__).resolve()
    while not (_root / "pyproject.toml").exists():
        _root = _root.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="data/processed/proteins/receptor_sequences.csv")
    ap.add_argument("--out",       default="data/embeddings/proteins/esm2_650m_ecl2_mean_curated.npz")
    ap.add_argument("--version",   default="650m", choices=["650m", "3B"])
    ap.add_argument(
        "--boundary-method", default="hydrophobicity",
        choices=["hydrophobicity", "bw"],
        help=(
            "hydrophobicity: predict TM4/TM5 boundaries via Kyte-Doolittle sliding window "
            "(no extra data needed, runs ESM from scratch). "
            "bw: use TM4/TM5 boundaries from bw_numbering_curated.csv and mean-pool "
            "directly from an existing per-residue npz (no ESM re-run)."
        ),
    )
    ap.add_argument("--bw-csv",
                    default="data/processed/bw_numbering_curated.csv",
                    help="path to bw_numbering_curated.csv (only used with --boundary-method bw)")
    ap.add_argument("--per-residue-npz",
                    default="data/embeddings/proteins/esm2_650m_per_residue_curated.npz",
                    help="per-residue embedding npz (only used with --boundary-method bw)")
    args = ap.parse_args()

    df = (
        pd.read_csv(_root / args.sequences)
        .dropna(subset=["sequence"])
        .drop_duplicates("receptor_id")
    )
    ids  = df["receptor_id"].tolist()
    seqs = df["sequence"].str.replace(" ", "", regex=False).tolist()
    print(f"{len(ids)} receptor sequences loaded")
    print(f"boundary-method: {args.boundary_method}")

    if args.boundary_method == "bw":
        # ── fast path: read per-residue npz, slice ECL2, mean-pool ──────────
        bw_csv  = _root / args.bw_csv
        pr_path = _root / args.per_residue_npz
        if not bw_csv.exists():
            raise FileNotFoundError(f"{bw_csv} not found — run 02_bw_numbering.py first")
        if not pr_path.exists():
            raise FileNotFoundError(f"{pr_path} not found — run 05_per_residue_embeddings.py first")

        print("loading BW boundaries …")
        ecl2_pos = _ecl2_pos_from_bw(bw_csv, ids)
        widths   = [e - s + 1 for s, e in ecl2_pos.values()]
        print(f"ECL2 width (BW): min={min(widths)}  mean={sum(widths)/len(widths):.1f}  max={max(widths)}")
        print(f"receptors with BW ECL2: {len(ecl2_pos)}/{len(ids)}")

        print("loading per-residue npz …")
        pr_npz = np.load(str(pr_path), allow_pickle=False)

        vecs, ids_out = [], []
        for rid in ids:
            mat = pr_npz[rid] if rid in pr_npz.files else None
            if mat is None:
                print(f"  WARNING: {rid[:30]}… not in per-residue npz, skipping")
                continue
            if rid in ecl2_pos:
                s, e = ecl2_pos[rid]
                s0, e0 = s - 1, e - 1          # 0-based
                s0 = max(0, s0)
                e0 = min(len(mat) - 1, e0)
                vec = mat[s0 : e0 + 1].mean(0) if e0 >= s0 else mat.mean(0)
            else:
                print(f"  WARNING: no BW ECL2 for {rid[:30]}…, falling back to full mean")
                vec = mat.mean(0)
            vecs.append(vec)
            ids_out.append(rid)

    else:
        # ── original path: hydrophobicity window → run ESM ──────────────────
        print("predicting ECL2 positions via hydrophobicity profile …")
        ecl2_pos_hydro: dict[str, tuple[int, int]] = {}
        widths = []
        for rid, seq in zip(ids, seqs):
            pos = ecl2_from_sequence(seq)
            ecl2_pos_hydro[rid] = pos
            widths.append(pos[1] - pos[0] + 1)
        print(f"ECL2 width (hydrophobicity): min={min(widths)}  mean={sum(widths)/len(widths):.1f}  max={max(widths)}")
        vecs    = embed_ecl2(ids, seqs, ecl2_pos_hydro, version=args.version)
        ids_out = ids

    out_path = _root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), ids=np.array(ids_out), emb=np.stack(vecs))
    print(f"wrote {len(ids_out)} x {vecs[0].shape[0]}-d -> {out_path}")


if __name__ == "__main__":
    main()
