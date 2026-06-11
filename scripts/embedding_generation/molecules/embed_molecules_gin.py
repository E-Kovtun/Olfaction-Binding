"""Molecule embeddings via a PRETRAINED GIN from LORAX (Hu et al. 2020).

LORAX benchmarked the `gin_supervised_*` graph representations; these are the
dgllife / Hu-2020 self-supervised GIN models. This script loads one of them and
produces a fixed graph-level embedding per molecule (mean-pooled node reprs,
300-d) — a drop-in alternative to the raw PyG graphs from 03_embed_molecules.py.

Stack: dgl + dgllife (pinned torch 2.2 in pyproject). `molfeat` wraps the same
models if you want a unified hub API later.

Models (--model): gin_supervised_contextpred | gin_supervised_infomax
                  | gin_supervised_edgepred  | gin_supervised_masking

Example
-------
uv run python scripts/embedding_generation/molecules/embed_molecules_gin.py \
       --molecules data/processed/molecules/molecule_smiles.csv \
       --model gin_supervised_contextpred \
       --out data/embeddings/molecules/gin_contextpred.npz
"""
import argparse, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


def build_graphs(smiles):
    """SMILES -> dgl graphs with the Hu-2020 featurization (self-loops on)."""
    from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
    af, bf = PretrainAtomFeaturizer(), PretrainBondFeaturizer(self_loop=True)
    graphs, ok = [], []
    for smi in smiles:
        g = smiles_to_bigraph(smi, add_self_loop=True, canonical_atom_order=False,
                              node_featurizer=af, edge_featurizer=bf)
        graphs.append(g); ok.append(g is not None)
    return graphs, ok


def embed(graphs, model_name, batch=256):
    import dgl, torch
    from dgllife.model import load_pretrained
    model = load_pretrained(model_name).eval()
    out = []
    valid = [g for g in graphs if g is not None]
    for i in range(0, len(valid), batch):
        bg = dgl.batch(valid[i:i + batch])
        nf = [bg.ndata["atomic_number"], bg.ndata["chirality_type"]]
        ef = [bg.edata["bond_type"], bg.edata["bond_direction_type"]]
        with torch.no_grad():
            bg.ndata["h"] = model(bg, nf, ef)
            out.append(dgl.mean_nodes(bg, "h").cpu().numpy())   # graph-level, 300-d
        print(f"  embedded {min(i + batch, len(valid))}/{len(valid)}", end="\r")
    print()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", required=True, help="csv with columns: inchikey, smiles")
    ap.add_argument("--model", default="gin_supervised_contextpred")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = pathlib.Path(args.out or f"data/embeddings/molecules/{args.model}.npz")

    df = pd.read_csv(args.molecules).dropna(subset=["smiles"]).drop_duplicates("inchikey")
    print(f"generating {args.model} embeddings for {len(df)} molecules ...")
    graphs, ok = build_graphs(df["smiles"].tolist())
    ids = [k for k, o in zip(df["inchikey"], ok) if o]
    emb = embed(graphs, args.model)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), emb=emb)
    print(f"wrote {len(ids)} x {emb.shape[1]}-d -> {out}  ({sum(ok)}/{len(ok)} parsed)")


if __name__ == "__main__":
    main()
