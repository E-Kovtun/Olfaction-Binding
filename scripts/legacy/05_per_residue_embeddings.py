"""Script 05 — canonical per-residue ESM-2 embeddings (+ derived mean).

Per-residue is the SINGLE SOURCE OF TRUTH: the mean embedding is exactly
residue-axis mean of the per-residue matrix (verified: max|Δ| ~ 1e-6).
Both outputs are derived from a single ESM forward pass.

Outputs (under data/embeddings/proteins/)
------------------------------------------
esm2_650m_per_residue_{tag}.npz   — {sequence: float32[seq_len, 1280]}
esm2_650m_mean_{tag}.npz          — {sequence: float32[1280]}  (derived)

`full` (780 receptors, pairs_m2or_full.csv) is the single source of truth going
forward — curated (409) is a strict subset and is derived on demand by
filtering `full` against `pairs_curated.csv["receptor"]` (no separate curated
npz is persisted; see any consumer script for the one-line filter).

Default run:
    uv run python scripts/embedding_generation/proteins/05_per_residue_embeddings.py

Extend to an even larger corpus later without recomputing what's already done:
    uv run python scripts/embedding_generation/proteins/05_per_residue_embeddings.py \\
        --source <bigger_pairs.csv> --tag full --reuse data/embeddings/proteins/esm2_650m_per_residue_full.npz

Args
----
--source   CSV with a 'receptor' column (AA sequences). Default: pairs_m2or_full.csv
--tag      Output file suffix. Default: full
--reuse    Existing per-residue npz to copy already-computed receptors from
           (avoids re-running ESM on sequences already embedded).
           Default: esm2_650m_per_residue_full.npz. Pass "" to disable reuse.
--version  ESM-2 variant: 650m or 3B. Default: 650m
--batch    Sequences per forward pass. Default: 4
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

EMB_DIR = _root / "data" / "embeddings" / "proteins"


def embed_per_residue(
    ids: list[str],
    seqs: list[str],
    version: str = "650m",
    batch: int = 4,
) -> dict[str, np.ndarray]:
    """Return {sequence: float32[seq_len, 1280]} for all sequences."""
    import esm, torch

    layer = 33 if version == "650m" else 36
    loader = {
        "650m": esm.pretrained.esm2_t33_650M_UR50D,
        "3B":   esm.pretrained.esm2_t36_3B_UR50D,
    }[version]

    print(f"loading ESM-2 {version} ...")
    model, alphabet = loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    bc = alphabet.get_batch_converter()
    print(f"running on {device}")

    result: dict[str, np.ndarray] = {}
    for i in range(0, len(ids), batch):
        chunk_ids  = ids[i : i + batch]
        chunk_seqs = [s[:1600] for s in seqs[i : i + batch]]
        data = [(rid, seq) for rid, seq in zip(chunk_ids, chunk_seqs)]
        _, _, toks = bc(data)
        lens = (toks != alphabet.padding_idx).sum(1)

        with torch.inference_mode():
            out = model(toks.to(device), repr_layers=[layer])
        r = out["representations"][layer]   # [B, L+2, hidden]

        for k, rid in enumerate(chunk_ids):
            seq_len = int(lens[k]) - 2      # strip BOS + EOS tokens
            rep = r[k, 1 : seq_len + 1, :].cpu().numpy().astype(np.float32)
            result[rid] = rep               # [seq_len, 1280]

        print(f"  embedded {min(i + batch, len(ids))}/{len(ids)}", end="\r", flush=True)

    print()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="data/processed/pairs_m2or_full.csv",
                    help="CSV with a 'receptor' column (AA sequences)")
    ap.add_argument("--tag",     default="full",
                    help="Output file suffix (esm2_650m_per_residue_{tag}.npz)")
    ap.add_argument("--reuse",   default="data/embeddings/proteins/esm2_650m_per_residue_full.npz",
                    help="Existing per-residue npz to copy already-computed entries from. Pass '' to disable.")
    ap.add_argument("--version", default="650m", choices=["650m", "3B"])
    ap.add_argument("--batch",   default=4, type=int)
    args = ap.parse_args()

    # --- load target sequences ---
    df = pd.read_csv(_root / args.source)
    col = "receptor" if "receptor" in df.columns else "sequence"
    seqs_all = df[col].dropna().str.replace(" ", "", regex=False).drop_duplicates().tolist()
    print(f"{len(seqs_all)} unique receptor sequences in source")

    # --- load reuse cache ---
    cached: dict[str, np.ndarray] = {}
    reuse_path = args.reuse.strip()
    if reuse_path:
        rp = _root / reuse_path
        if rp.exists():
            d = np.load(str(rp), allow_pickle=False)
            cached = {k: d[k] for k in d.files}
            print(f"reusing {len(cached)} pre-computed receptors from {rp.name}")
        else:
            print(f"reuse file not found: {rp} — will compute all from scratch")

    # --- determine which sequences need ESM ---
    todo_seqs = [s for s in seqs_all if s not in cached]
    print(f"{len(todo_seqs)} sequences need ESM forward pass  "
          f"({len(seqs_all) - len(todo_seqs)} reused)")

    # --- run ESM on new sequences ---
    new_emb: dict[str, np.ndarray] = {}
    if todo_seqs:
        new_emb = embed_per_residue(
            todo_seqs, todo_seqs, version=args.version, batch=args.batch
        )

    # --- merge and write per-residue ---
    per_res = {**cached, **new_emb}
    # keep only sequences that appear in our source (drop any extras from reuse)
    per_res = {s: per_res[s] for s in seqs_all if s in per_res}

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out_pr = EMB_DIR / f"esm2_650m_per_residue_{args.tag}.npz"
    np.savez_compressed(str(out_pr), **per_res)

    lengths = [v.shape[0] for v in per_res.values()]
    size_mb = sum(v.nbytes for v in per_res.values()) / 1024 ** 2
    print(f"\nper-residue: {len(per_res)} receptors  "
          f"seq_len min={min(lengths)} mean={sum(lengths)/len(lengths):.0f} max={max(lengths)}  "
          f"uncompressed={size_mb:.0f} MB")
    print(f"-> {out_pr}")

    # --- derive mean (residue axis) and write ---
    mean_emb = {s: per_res[s].mean(0) for s in per_res}
    out_mean = EMB_DIR / f"esm2_650m_mean_{args.tag}.npz"
    np.savez_compressed(str(out_mean), **mean_emb)
    print(f"mean:        {len(mean_emb)} receptors  dim=1280")
    print(f"-> {out_mean}")


if __name__ == "__main__":
    main()
