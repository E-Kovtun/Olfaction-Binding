"""Fit cached XGBoost probes on the best-validation encoder snapshots from full_full v5.

GNN training is never repeated. By default only the eight explicitly accepted
non-collapsed runs are evaluated; one small result bundle is written per probe.
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

from orbind import hetero as H
from orbind import lorax as L
from orbind.hetero_gat import HeteroGATLink
from orbind.baselines import train_boost
from orbind.dataset import metrics

HISTORICAL_COLLAPSE_KEYS = {
    ("transductive", "gnn", "pos_only", 1, 42),
    ("transductive", "gnn", "signed", 1, 42),
    ("transductive", "gnn", "signed", 3, 44),
    ("transductive", "gnn", "all_edges", 2, 43),
    ("inductive_molecule", "gnn", "pos_only", 1, 43),
    ("inductive_molecule", "gnn", "pos_only", 1, 44),
    ("inductive_molecule", "gnn", "signed", 1, 42),
    ("inductive_molecule", "gnn", "signed", 1, 43),
}

# Stable targeted reruns replaced every historically collapsed checkpoint.
COLLAPSE_MASK = set()


def run_key(ck):
    return (ck["regime"], ck["arch"], ck["config"]["mp_mode"],
            int(ck["fold"]), int(ck.get("gnn_seed", ck["seed"])))


def make_model(ck, device):
    cfg = ck["config"]
    if ck["arch"] == "gnn":
        model = H.HeteroLink(hidden=int(cfg["hidden"]), dropout=0.3,
                             mp_mode=cfg["mp_mode"])
    else:
        model = HeteroGATLink(hidden=int(cfg["hidden"]), heads=4, dropout=0.3,
                              mp_mode=cfg["mp_mode"])
    return model.to(device)


def evaluate(path, out_dir, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = run_key(ck)
    if key in COLLAPSE_MASK:
        print(f"SKIP collapsed: {path.name}")
        return "collapsed"
    out = out_dir / path.name
    if out.exists():
        print(f"SKIP cached: {out.name}")
        return "cached"
    cfg = ck["config"]
    if cfg.get("protein_pca_dim", 0) or cfg.get("molecule_pca_dim", 0):
        raise RuntimeError(f"Best-probe evaluator expects uncompressed v5 inputs: {path.name}")
    if cfg.get("history") or cfg.get("concat_raw_prot") or cfg.get("transductive_exp"):
        raise RuntimeError(f"Best-probe evaluator expects standard raw-molecule v5: {path.name}")

    esm, chem = L.load_embeddings(cfg.get("protein_embeddings"), cfg.get("molecule_embeddings"))
    Xm, Xp, _ = L.build(ck["regime"], int(ck["fold"]), esm, chem,
                        seed=int(ck.get("gnn_seed", ck["seed"])))
    x_dict = {H.MOL: Xm.to(device), H.PROT: Xp.to(device)}

    train_idx, train_y = ck["gnn_train_sup"]
    mp_pos = train_idx[:, train_y == 1].long().to(device)
    mp_neg = train_idx[:, train_y == 0].long().to(device)
    eidx = {k: v.to(device) for k, v in
            H.edge_index_dict(mp_pos, mp_neg, mode=cfg["mp_mode"]).items()}

    model = make_model(ck, device)
    with torch.no_grad():
        model.encode(x_dict, eidx)  # initialize lazy layers
    model.load_state_dict({k: v.to(device) for k, v in ck["best_state"].items()})
    model.eval()
    with torch.no_grad():
        z = model.encode(x_dict, eidx)
    Zm = x_dict[H.MOL].detach().cpu().numpy()
    Zp = z[H.PROT].detach().cpu().numpy()

    def features(sup):
        idx = sup[0].cpu().numpy()
        return np.concatenate([Zm[idx[0]], Zp[idx[1]]], axis=1)

    ytr = ck["probe_train_sup"][1].cpu().numpy()
    yte = ck["test_sup"][1].cpu().numpy()
    scores = train_boost(features(ck["probe_train_sup"]), ytr,
                         features(ck["test_sup"]), seed=int(ck["boost_seed"]))
    result = {
        "source_checkpoint": str(path.relative_to(ROOT)),
        "regime": ck["regime"], "arch": ck["arch"], "mp_mode": cfg["mp_mode"],
        "fold": int(ck["fold"]), "gnn_seed": int(ck["gnn_seed"]),
        "boost_seed": int(ck["boost_seed"]),
        "best_epoch": int(max(ck["decoder_history"], key=lambda r: r["val_AUPRC"])["epoch"]),
        "test_scores_best": scores, "test_metrics_best": metrics(yte, scores),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"DONE {out.name}: " + " ".join(
        f"{k}={result['test_metrics_best'][k]:.3f}" for k in ("AUROC", "AUPRC", "MCC", "F1")))
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/graph/full_full/v5/architecture_screen")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"best-checkpoint probe device: {device}")
    base = ROOT / args.results_dir
    paths = sorted((base / "training/checkpoints").glob("*.pt"))
    counts = {"done": 0, "cached": 0, "collapsed": 0}
    for path in paths:
        state = evaluate(path, base / "probes/best_checkpoint", device)
        counts[state] += 1
    print("best-probe queue complete:", counts)


if __name__ == "__main__":
    main()
