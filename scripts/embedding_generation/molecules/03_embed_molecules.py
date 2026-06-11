"""Script 3 — molecule representations for the GCN side.

Stack: RDKit (SMILES -> graph) + torch_geometric (PyG). No dgl.

There is no public "GCN embedding" to download and no canonical pretrained GCN
here, so this script FEATURIZES each molecule into a PyG ``Data`` graph (atom /
bond features + connectivity). Those graphs are the reusable input for a GCN:

  * default            -> save graphs to a .pt list (train your GCN end-to-end);
  * --checkpoint M.pt  -> load a trained GCN and dump fixed embeddings (.npz).

Atom features (per node, 33-d one-hots + scalars): atomic number, degree,
formal charge, hybridization, aromaticity, H count. Mirrors the spirit of the
deepchem MolGraphConvFeaturizer used by the local GCN baseline.

Example
-------
uv run python scripts/embedding_generation/molecules/03_embed_molecules.py \
       --molecules data/processed/molecules/molecule_smiles.csv \
       --out data/embeddings/molecules/mol_graphs.pt
"""
import argparse, pathlib
import numpy as np
import pandas as pd

ATOM_LIST = list(range(1, 54))                 # H..I
HYBRID = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "UNSPECIFIED"]


def _onehot(x, choices):
    v = [0.0] * (len(choices) + 1)
    v[choices.index(x) if x in choices else -1] = 1.0
    return v


def atom_features(atom):
    from rdkit.Chem import rdchem
    return (
        _onehot(atom.GetAtomicNum(), ATOM_LIST)
        + _onehot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        + _onehot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + _onehot(str(atom.GetHybridization()), HYBRID)
        + [float(atom.GetIsAromatic()), float(atom.GetTotalNumHs())]
    )


def smiles_to_pyg(smiles):
    from rdkit import Chem
    import torch
    from torch_geometric.data import Data
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]                       # undirected
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", required=True, help="csv with columns: inchikey, smiles")
    ap.add_argument("--out", default="data/embeddings/molecules/mol_graphs.pt")
    ap.add_argument("--checkpoint", default=None,
                    help="optional trained GCN (.pt) to emit fixed embeddings instead of graphs")
    args = ap.parse_args()
    import torch

    df = pd.read_csv(args.molecules).dropna(subset=["smiles"]).drop_duplicates("inchikey")
    graphs, ids, bad = [], [], 0
    for ik, smi in zip(df["inchikey"], df["smiles"]):
        g = smiles_to_pyg(smi)
        if g is None:
            bad += 1; continue
        g.inchikey = ik
        graphs.append(g); ids.append(ik)
    print(f"featurized {len(graphs)}/{len(df)} molecules ({bad} unparseable) | node dim={graphs[0].num_node_features}")

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        from torch_geometric.loader import DataLoader
        model = torch.load(args.checkpoint, weights_only=False).eval()
        embs = []
        with torch.no_grad():
            for g in DataLoader(graphs, batch_size=64):
                embs.append(model(g).cpu().numpy())          # model must return graph-level vec
        emb = np.concatenate(embs)
        np.savez_compressed(out.with_suffix(".npz"), ids=np.array(ids), emb=emb)
        print(f"wrote {len(ids)} x {emb.shape[1]}-d -> {out.with_suffix('.npz')}")
    else:
        torch.save({"ids": ids, "graphs": graphs}, out)
        print(f"wrote {len(graphs)} PyG graphs -> {out}  (feed to your GCN end-to-end)")


if __name__ == "__main__":
    main()
