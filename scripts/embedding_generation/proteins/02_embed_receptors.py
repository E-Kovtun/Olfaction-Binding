"""Script 2 — GENERATE ESM-2 embeddings for the receptors we need.

SUPERSEDED by 05_per_residue_embeddings.py, which generates per-residue
embeddings AND derives the mean from them in one ESM pass (single source of
truth — see that script's docstring). Kept for reference / mean-only reruns;
prefer 05 for anything new.

There is no public database of ESM-2 embeddings for olfactory receptors, so we
run ESM-2 (650M, esm2_t33_650M_UR50D — same as NOSE/MolOR) over the unique
sequences and cache one vector per sequence. Mean-pooled over residues -> 1280-d.

Input : data/processed/proteins/receptor_sequences.csv  (receptor_id, sequence),
        produced by the preprocessing step.

Examples
--------
uv run python scripts/embedding_generation/proteins/02_embed_receptors.py \
       --sequences data/processed/proteins/receptor_sequences.csv \
       --out data/embeddings/proteins/esm2_650m_mean.npz
"""
import argparse, pathlib
import numpy as np
import pandas as pd

esm_model = esm_alphabet = None


def setup_esm(version="650m", device=None):
    import esm, torch
    global esm_model, esm_alphabet
    if esm_model is None:
        loader = {"650m": esm.pretrained.esm2_t33_650M_UR50D,
                  "3B": esm.pretrained.esm2_t36_3B_UR50D}[version]
        esm_model, esm_alphabet = loader()
        esm_model.eval().to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return esm_model, esm_alphabet


def embed(sequences, version="650m", per_residue=False, batch=5):
    """Returns list of np.ndarray; mean-pooled (1280-d) unless per_residue."""
    import torch
    model, alphabet = setup_esm(version)
    device = next(model.parameters()).device
    bc = alphabet.get_batch_converter()
    layer = 33 if version == "650m" else 36
    reps = []
    for i in range(0, len(sequences), batch):
        chunk = [s[:1600] for s in sequences[i:i + batch]]
        data = [(f"p{j}", s) for j, s in enumerate(chunk)]
        _, _, toks = bc(data)
        lens = (toks != alphabet.padding_idx).sum(1)
        with torch.inference_mode():
            r = model(toks.to(device), repr_layers=[layer])["representations"][layer]
        for k, L in enumerate(lens):
            v = r[k, 1:L - 1] if not per_residue else r[k, 1:L - 1]
            reps.append((v.mean(0) if not per_residue else v).cpu().numpy())
        print(f"  embedded {min(i + batch, len(sequences))}/{len(sequences)}", end="\r")
    print()
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True,
                    help="csv with columns: receptor_id, sequence")
    ap.add_argument("--out", default="data/embeddings/proteins/esm2_650m_mean.npz")
    ap.add_argument("--version", default="650m", choices=["650m", "3B"])
    args = ap.parse_args()

    df = pd.read_csv(args.sequences).dropna(subset=["sequence"]).drop_duplicates("receptor_id")
    ids = df["receptor_id"].tolist()
    seqs = df["sequence"].str.replace(" ", "", regex=False).tolist()
    print(f"generating ESM-2 {args.version} embeddings for {len(seqs)} receptors ...")
    vecs = embed(seqs, version=args.version)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids), emb=np.stack(vecs))
    print(f"wrote {len(ids)} x {vecs[0].shape[0]}-d -> {args.out}")


if __name__ == "__main__":
    main()
