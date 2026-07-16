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
       --out data/embeddings/molecules/gin_contextpred.npz \
       --node-out data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz

The optional node output stores the same 300-d GIN representations before mean
pooling, one variable-length ``[n_atoms, 300]`` array per molecule.

Model weights are cached beside fair-esm in Torch Hub's checkpoint directory
(``~/.cache/torch/hub/checkpoints``), never in the repository working tree.
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


def load_pretrained_cached(model_name, cache_dir=None):
    """Load a dgllife GIN, downloading once into the shared Torch Hub cache."""
    import torch
    from dgl.data.utils import _get_dgl_url, download
    from dgllife.model.pretrain import create_property_model, url

    cache = (pathlib.Path(cache_dir).expanduser() if cache_dir else
             pathlib.Path(torch.hub.get_dir()) / "checkpoints")
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache / f"{model_name}_pre_trained.pth"
    if not checkpoint_path.exists():
        download(_get_dgl_url(url[model_name]), path=str(checkpoint_path),
                 overwrite=False, log=True)

    model = create_property_model(model_name)
    if model is None:
        raise ValueError(f"unsupported property model: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    print(f"pretrained model <- {checkpoint_path}")
    return model.eval()


def embed(graphs, model_name, batch=256, return_nodes=False, cache_dir=None):
    """Return graph means and, optionally, the pre-pooling atom representations."""
    import dgl, torch
    model = load_pretrained_cached(model_name, cache_dir=cache_dir)
    out, node_out = [], []
    valid = [g for g in graphs if g is not None]
    for i in range(0, len(valid), batch):
        bg = dgl.batch(valid[i:i + batch])
        nf = [bg.ndata["atomic_number"], bg.ndata["chirality_type"]]
        ef = [bg.edata["bond_type"], bg.edata["bond_direction_type"]]
        with torch.no_grad():
            h = model(bg, nf, ef)
            bg.ndata["h"] = h
            out.append(dgl.mean_nodes(bg, "h").cpu().numpy())   # graph-level, 300-d
            if return_nodes:
                node_out.extend(x.cpu().numpy().astype(np.float32)
                                for x in torch.split(h, bg.batch_num_nodes().tolist()))
        print(f"  embedded {min(i + batch, len(valid))}/{len(valid)}", end="\r")
    print()
    graph_out = np.concatenate(out).astype(np.float32)
    return (graph_out, node_out) if return_nodes else graph_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", required=True, help="csv with columns: inchikey, smiles")
    ap.add_argument("--model", default="gin_supervised_contextpred")
    ap.add_argument("--out", default=None)
    ap.add_argument("--node-out", default=None,
                    help="Optional npz output with one variable-length [atoms,300] array per inchikey")
    ap.add_argument("--model-cache", default=None,
                    help="Pretrained-weight cache (default: Torch Hub checkpoints, beside ESM weights)")
    args = ap.parse_args()
    out = pathlib.Path(args.out or f"data/embeddings/molecules/{args.model}.npz")

    df = pd.read_csv(args.molecules).dropna(subset=["smiles"]).drop_duplicates("inchikey")
    print(f"generating {args.model} embeddings for {len(df)} molecules ...")
    graphs, ok = build_graphs(df["smiles"].tolist())
    ids = [k for k, o in zip(df["inchikey"], ok) if o]
    embedded = embed(graphs, args.model, return_nodes=bool(args.node_out),
                     cache_dir=args.model_cache)
    if args.node_out:
        emb, node_emb = embedded
    else:
        emb = embedded

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), emb=emb)
    print(f"wrote {len(ids)} x {emb.shape[1]}-d -> {out}  ({sum(ok)}/{len(ok)} parsed)")
    if args.node_out:
        node_out = pathlib.Path(args.node_out)
        node_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(node_out, **dict(zip(ids, node_emb)))
        n_atoms = [x.shape[0] for x in node_emb]
        print(f"wrote atom embeddings for {len(ids)} molecules "
              f"(atoms min={min(n_atoms)}, mean={np.mean(n_atoms):.1f}, max={max(n_atoms)}) -> {node_out}")


if __name__ == "__main__":
    main()
