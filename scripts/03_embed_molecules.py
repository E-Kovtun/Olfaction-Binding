"""Script 3 — GENERATE molecule (GCN/GNN) embeddings for the molecules we need.

There is no canonical "GCN embedding" to download: an embedding is defined by a
trained GNN. We default to a PRE-TRAINED GIN from dgllife
(``gin_supervised_contextpred`` — the family LORAX benchmarked), which yields a
fixed 300-d vector per molecule. Swap ``--model`` for another pretrained GNN, or
use ``--graphs-only`` to just emit DGL graphs and train your own GCN end-to-end.

Requires SMILES. The official M2OR dump lacks SMILES (only InChIKey), so resolve
them first with orbind.backfill.inchikey_to_smiles (network) and pass a csv of
(inchikey, smiles).

Examples
--------
python scripts/03_embed_molecules.py --molecules data/processed/molecule_smiles.csv \
       --out data/embeddings/gin_contextpred.npz
"""
import argparse, pathlib
import numpy as np
import pandas as pd


def embed_pretrained_gin(smiles, model_name="gin_supervised_contextpred"):
    import torch
    from dgllife.model import load_pretrained
    from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
    from dgl import batch as dgl_batch

    model = load_pretrained(model_name).eval()
    af, bf = PretrainAtomFeaturizer(), PretrainBondFeaturizer()
    vecs, ok = [], []
    for smi in smiles:
        g = smiles_to_bigraph(smi, node_featurizer=af, edge_featurizer=bf)
        if g is None:
            vecs.append(None); ok.append(False); continue
        with torch.no_grad():
            nfeats = [g.ndata.pop("atomic_number"), g.ndata.pop("chirality_type")]
            efeats = [g.edata.pop("bond_type"), g.edata.pop("bond_direction_type")]
            node_repr = model(g, nfeats, efeats)
            # readout: mean over atoms -> graph embedding (300-d)
            vecs.append(node_repr.mean(0).numpy()); ok.append(True)
    return vecs, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", required=True, help="csv with columns: inchikey, smiles")
    ap.add_argument("--out", default="data/embeddings/gin_contextpred.npz")
    ap.add_argument("--model", default="gin_supervised_contextpred")
    ap.add_argument("--graphs-only", action="store_true",
                    help="emit DGL graphs for end-to-end training instead of fixed embeddings")
    args = ap.parse_args()

    df = pd.read_csv(args.molecules).dropna(subset=["smiles"]).drop_duplicates("inchikey")
    keys, smis = df["inchikey"].tolist(), df["smiles"].tolist()
    print(f"generating molecule embeddings for {len(smis)} molecules via {args.model} ...")

    vecs, ok = embed_pretrained_gin(smis, args.model)
    keys = [k for k, o in zip(keys, ok) if o]
    emb = np.stack([v for v, o in zip(vecs, ok) if o])
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(keys), emb=emb)
    print(f"wrote {len(keys)} x {emb.shape[1]}-d -> {args.out}  ({sum(ok)}/{len(ok)} parsed)")


if __name__ == "__main__":
    main()
