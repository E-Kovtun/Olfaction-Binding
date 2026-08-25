"""Repair/add the best-validation XGBoost probe inside a full_full v5 checkpoint.

This is intentionally in-place: old quantile-screen runs that only contain the
last-epoch probe can be upgraded without repeating GNN training. The MP graph is
reconstructed from the checkpoint config, including the molecule-coverage
quantile filter, so the encoded protein states match the original run protocol.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind.legacy import hetero as H
from orbind.legacy import lorax as L
from orbind.legacy.hetero_gat import HeteroGATLink
from orbind.baselines import train_boost
from orbind.dataset import metrics


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _make_model(ck, device):
    cfg = ck["config"]
    if ck["arch"] == "gnn":
        hidden = int(cfg.get("hidden", 256))
        model = H.HeteroLink(hidden=hidden, dropout=float(cfg.get("dropout", 0.3)),
                             mp_mode=cfg["mp_mode"])
    else:
        hidden = int(cfg.get("hidden", cfg.get("gat_hidden", 128)))
        model = HeteroGATLink(hidden=hidden, heads=int(cfg.get("heads", 4)),
                              dropout=float(cfg.get("dropout", 0.3)),
                              mp_mode=cfg["mp_mode"])
    return model.to(device)


def _split_train_for_probe_if_needed(ck, splits):
    cfg = ck["config"]
    if cfg.get("disjoint_probe_train", False):
        return H.split_train_for_probe(
            splits["train"],
            probe_frac=float(cfg.get("probe_train_frac", 0.5)),
            seed=int(ck.get("gnn_seed", ck.get("seed"))) + 101,
        )
    return splits["train"], splits["train"]


def _build_mp_edges(ck, Xm, splits):
    cfg = ck["config"]
    gnn_train, _ = _split_train_for_probe_if_needed(ck, splits)
    mp_pos_a, mp_neg_a = gnn_train["pos"], gnn_train["neg"]
    q = float(cfg.get("mol_quality_q", 0.0) or 0.0)
    if q > 0.0:
        mask = H.quality_mol_mask(
            splits["train"]["pos"], splits["train"]["neg"], Xm.shape[0], q)
        keep = set(int(i) for i in np.where(mask)[0])

        def filt(a):
            if len(a) == 0:
                return a
            return a[np.array([int(m) in keep for m in a[:, 0]], dtype=bool)]

        mp_pos_a, mp_neg_a = filt(mp_pos_a), filt(mp_neg_a)
    mp_pos = torch.tensor(mp_pos_a.T, dtype=torch.long)
    mp_neg = torch.tensor(mp_neg_a.T, dtype=torch.long)
    return H.edge_index_dict(mp_pos, mp_neg, mode=cfg["mp_mode"])


def _features(Xm, Zp, sup):
    idx = sup[0].detach().cpu().numpy()
    return np.concatenate([Xm[idx[0]], Zp[idx[1]]], axis=1)


def repair_checkpoint(path, device, force=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("test_scores") is not None and not force:
        print(f"SKIP_HAS_BEST {path.name}")
        return "cached"
    if ck.get("best_state") is None:
        raise RuntimeError(f"Checkpoint has no best_state: {path}")

    cfg = ck["config"]
    if cfg.get("protein_pca_dim", 0) or cfg.get("molecule_pca_dim", 0):
        raise RuntimeError(f"PCA-compressed checkpoints are not supported by this repair script: {path.name}")

    gnn_seed = int(ck.get("gnn_seed", ck.get("seed")))
    boost_seed = int(ck.get("boost_seed", gnn_seed))
    print(f"REPAIR_BEST {path.name} device={device} gnn_seed={gnn_seed} boost_seed={boost_seed}")

    esm, chem = L.load_embeddings(cfg.get("protein_embeddings"), cfg.get("molecule_embeddings"))
    Xm, Xp, splits = L.build(ck["regime"], int(ck["fold"]), esm, chem, seed=gnn_seed)
    eidx = _build_mp_edges(ck, Xm, splits)

    x_dict = {H.MOL: Xm.to(device), H.PROT: Xp.to(device)}
    eidx = {k: v.to(device) for k, v in eidx.items()}

    model = _make_model(ck, device)
    with torch.no_grad():
        model.encode(x_dict, eidx)
    model.load_state_dict({k: v.to(device) for k, v in ck["best_state"].items()})
    model.eval()
    with torch.no_grad():
        z = (model.encode_history(x_dict, eidx, depth=int(cfg.get("history_depth", 2)))
             if cfg.get("history", False) else model.encode(x_dict, eidx))

    Xm_probe = (z[H.MOL] if cfg.get("transductive_exp", False) else x_dict[H.MOL]).detach().cpu().numpy()
    Zp = z[H.PROT].detach().cpu().numpy()
    if cfg.get("concat_raw_prot", False):
        Zp = np.concatenate([x_dict[H.PROT].detach().cpu().numpy(), Zp], axis=1)

    ytr = ck["probe_train_sup"][1].detach().cpu().numpy()
    yte = ck["test_sup"][1].detach().cpu().numpy()
    scores = train_boost(
        _features(Xm_probe, Zp, ck["probe_train_sup"]), ytr,
        _features(Xm_probe, Zp, ck["test_sup"]), seed=boost_seed,
    )
    print("BEST_VAL_PROBE " + " ".join(
        f"{k}={v:.3f}" for k, v in metrics(yte, scores).items()))

    scores_raw = scores_enriched = None
    if ck["regime"] == "transductive":
        Xm_raw = x_dict[H.MOL].detach().cpu().numpy()
        Xm_enr = z[H.MOL].detach().cpu().numpy()
        if cfg.get("transductive_exp", False):
            scores_enriched = scores
            scores_raw = train_boost(
                _features(Xm_raw, Zp, ck["probe_train_sup"]), ytr,
                _features(Xm_raw, Zp, ck["test_sup"]), seed=boost_seed,
            )
        else:
            scores_raw = scores
            scores_enriched = train_boost(
                _features(Xm_enr, Zp, ck["probe_train_sup"]), ytr,
                _features(Xm_enr, Zp, ck["test_sup"]), seed=boost_seed,
            )

    ck["test_scores"] = scores
    ck["test_scores_raw"] = scores_raw
    ck["test_scores_enriched"] = scores_enriched
    if ck.get("test_scores_final") is not None:
        ck["config"]["probe_checkpoint"] = "both"
    else:
        ck["config"]["probe_checkpoint"] = "best"
    ck["best_val_probe_repaired"] = True

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ck, tmp)
    tmp.replace(path)
    print(f"UPDATED_CHECKPOINT {path}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    repair_checkpoint(args.checkpoint, _device(args.device), force=args.force)


if __name__ == "__main__":
    main()
