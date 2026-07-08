"""Feature-spectrum probe for the compressed full_full graph runs.

For each cached disjoint_raw GNN run (regime x seed) it reconstructs the graph
protein embedding z_prot from the saved encoder weights (NO retraining) and fits
an XGBoost probe on a ladder of feature sets:

  1 compressed      : [compressed mol | compressed prot]           (no graph)
  2 graph_zprot     : [compressed mol | z_prot]                    (= disjoint raw mol)
  3 graph+cprot     : [compressed mol | z_prot | compressed prot]
  4 graph+fullprot  : [compressed mol | z_prot | full ESM-1b 1280]

The 5th chart bar (full no-graph XGBoost) comes from results/full_full/tables
and is added in the notebook, not here.

Writes results/graph/full_full/compression/h64/tables/feature_spectrum_by_seed.csv
"""
import pathlib, sys
import numpy as np, pandas as pd, torch

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import lorax as L, hetero as H
from orbind.baselines import train_boost
from orbind.dataset import metrics

MOL, PROT = H.MOL, H.PROT
METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
HIDDEN = 64
PROT_C = "data/embeddings/proteins/esm1b_650m_mean_lorax_pca32.npz"
MOL_C = "data/embeddings/molecules/chemberta_77m_lorax_pca16.npz"
RUNS = _root / "results/graph/full_full/compression/h64/runs"
OUT = _root / "results/graph/full_full/compression/h64/tables/feature_spectrum_by_seed.csv"
SEEDS = [42, 43, 44]
REGIMES = ["transductive", "inductive_molecule"]
BAR_ORDER = ["compressed", "graph_zprot", "graph+cprot", "graph+fullprot"]

# disjoint_raw run directory names (with the pre-cleanup legacy fallbacks).
LEGACY = {"transductive": "transductive_disjoint", "inductive_molecule": "inductive_disjoint"}
def ckpt_path(regime, seed):
    fname = f"gnn_signed_disjoint_unentangled_boost_{regime}_fold1.pt"
    cfg = {"transductive": "transductive_disjoint_raw",
           "inductive_molecule": "inductive_disjoint_raw"}[regime]
    for name in (LEGACY[regime], cfg):                # legacy dir wins if it has the file
        p = RUNS / name / f"seed_{seed}" / "checkpoints" / fname
        if p.exists():
            return p
    return None


def edge_feats(mol, prot_blocks, idx):
    m = mol[idx[0].numpy()]
    p = np.concatenate([b[idx[1].numpy()] for b in prot_blocks], axis=1)
    return np.concatenate([m, p], axis=1).astype(np.float32)


def main():
    esm_c, chem_c = L.load_embeddings(PROT_C, MOL_C)
    esm_full, _ = L.load_embeddings()                 # defaults = full ESM-1b 1280
    dfs = L._load_fold(1, esm_c, chem_c)
    _, _, _, pi = L._node_universe(dfs, esm_c, chem_c)
    prot_ids = [None] * len(pi)
    for k, i in pi.items():
        prot_ids[i] = k
    assert all(k in esm_full for k in prot_ids), "some prot nodes missing from full ESM"
    Xp_full = np.stack([np.asarray(esm_full[k], dtype=np.float32) for k in prot_ids])
    print(f"prot nodes={len(prot_ids)} | full ESM dim={Xp_full.shape[1]}", flush=True)

    rows = []
    for regime in REGIMES:
        for seed in SEEDS:
            p = ckpt_path(regime, seed)
            if p is None:
                print(f"SKIP {regime} seed {seed}: no checkpoint", flush=True); continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            torch.manual_seed(seed)
            Xm, Xp, splits = L.build(regime, 1, esm_c, chem_c, seed=seed)
            gnn_train, probe_train = H.split_train_for_probe(
                splits["train"], probe_frac=ck["config"]["probe_train_frac"], seed=seed + 101)
            mp_pos = torch.tensor(gnn_train["pos"].T, dtype=torch.long)
            mp_neg = torch.tensor(gnn_train["neg"].T, dtype=torch.long)
            eidx = H.edge_index_dict(mp_pos, mp_neg, mode="signed")
            sup_probe = H.sup_edges(probe_train); sup_test = H.sup_edges(splits["test"])
            model = H.HeteroLink(hidden=HIDDEN, dropout=0.3, mp_mode="signed")
            with torch.no_grad():
                model.encode({MOL: Xm, PROT: Xp}, eidx)
            model.load_state_dict(ck["best_state"]); model.eval()
            with torch.no_grad():
                z = model.encode({MOL: Xm, PROT: Xp}, eidx)
            Zp = z[PROT].numpy(); Xm_raw = Xm.numpy(); Xp_raw = Xp.numpy()
            ip, it = sup_probe[0], sup_test[0]
            ytr = sup_probe[1].numpy(); yte = sup_test[1].numpy()
            ladders = {
                "compressed":     (Xm_raw, [Xp_raw]),
                "graph_zprot":    (Xm_raw, [Zp]),
                "graph+cprot":    (Xm_raw, [Zp, Xp_raw]),
                "graph+fullprot": (Xm_raw, [Zp, Xp_full]),
            }
            for bar, (mol, pb) in ladders.items():
                if bar == "graph_zprot":              # identical to the saved run; reuse it
                    m = metrics(yte, ck["test_scores"])
                else:
                    sc = train_boost(edge_feats(mol, pb, ip), ytr, edge_feats(mol, pb, it), seed=seed)
                    m = metrics(yte, sc)
                dim = edge_feats(mol, pb, it).shape[1]
                rows.append({"regime": regime, "seed": seed, "bar": bar, "dim": dim,
                             **{k: float(m[k]) for k in METRICS}})
                print(f"{regime:20s} seed {seed} {bar:16s} dim={dim:5d} "
                      + " ".join(f"{k}={m[k]:.3f}" for k in METRICS), flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nsaved {OUT.relative_to(_root)}  ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
