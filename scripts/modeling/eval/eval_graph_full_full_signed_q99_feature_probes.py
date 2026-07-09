"""Feature-probe XGBoost variants for full_full v5 GNN signed q99 checkpoints.

No GNN training is performed. For each existing signed_q99 checkpoint, the script
reconstructs the original q99 message-passing graph, restores the requested GNN
state, extracts protein representations, and fits several downstream XGBoost
probes on [raw ChemBERTa molecule || selected protein blocks].

Probe variants:
  z2_rawmol              : raw molecule + final protein encoder output z2
  z2_rawprot_rawmol      : raw molecule + raw protein + z2
  z2_x0_rawmol           : raw molecule + pre-MP projected protein x0 + z2
  z2_rawprot_x0_x1_rawmol: raw molecule + raw protein + x0 + x1 + z2
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import hetero as H
from orbind import lorax as L
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]
VARIANTS = [
    ("z2_rawmol", ["z2"]),
    ("z2_rawprot_rawmol", ["raw_prot", "z2"]),
    ("z2_x0_rawmol", ["x0", "z2"]),
    ("z2_rawprot_x0_x1_rawmol", ["raw_prot", "x0", "x1", "z2"]),
]


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _checkpoint_state(ck, checkpoint):
    if checkpoint == "last_epoch":
        state = ck.get("final_state")
        if state is None:
            raise RuntimeError("checkpoint has no final_state; cannot evaluate last_epoch")
        return state
    if checkpoint == "best_val":
        state = ck.get("best_state")
        if state is None:
            raise RuntimeError("checkpoint has no best_state; cannot evaluate best_val")
        return state
    raise ValueError(checkpoint)


def _make_model(ck, device):
    cfg = ck["config"]
    if ck["arch"] != "gnn":
        raise RuntimeError(f"This probe script expects GNN checkpoints, got arch={ck['arch']}")
    hidden = int(cfg.get("hidden", 256))
    return H.HeteroLink(hidden=hidden, dropout=float(cfg.get("dropout", 0.3)),
                        mp_mode=cfg["mp_mode"]).to(device)


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


def _encode_stages(model, x_dict, eidx):
    import torch.nn.functional as F
    x0 = {k: F.relu(model.proj[k](v)) for k, v in x_dict.items()}
    if model.mp_mode == "signed":
        pos_eidx = {H.ETYPE: eidx[H.ETYPE], H.RTYPE: eidx[H.RTYPE]}
        neg_eidx = {H.ETYPE_NEG: eidx[H.ETYPE_NEG], H.RTYPE_NEG: eidx[H.RTYPE_NEG]}
        xp = model.conv1(x0, pos_eidx)
        xn = model.conv1_neg(x0, neg_eidx)
        x1 = {k: F.relu(xp[k] - xn.get(k, torch.zeros_like(xp[k]))) for k in xp}
        xp = model.conv2(x1, pos_eidx)
        xn = model.conv2_neg(x1, neg_eidx)
        z2 = {k: xp[k] - xn.get(k, torch.zeros_like(xp[k])) for k in xp}
    else:
        x1 = {k: F.relu(v) for k, v in model.conv1(x0, eidx).items()}
        z2 = model.conv2(x1, eidx)
    return {"x0": x0, "x1": x1, "z2": z2}


def _edge_features(Xm_raw, protein_blocks, sup):
    idx = sup[0].detach().cpu().numpy()
    mol = Xm_raw[idx[0]]
    prot = np.concatenate([protein_blocks[name][idx[1]] for name in protein_blocks], axis=1)
    return np.concatenate([mol, prot], axis=1)


def evaluate_checkpoint(path, checkpoint, device, out_dir, force=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if ck.get("arch") != "gnn" or cfg.get("mp_mode") != "signed" or round(float(cfg.get("mol_quality_q", 0.0)), 2) != 0.99:
        print(f"SKIP_NOT_SIGNED_Q99 {path.name}")
        return []
    if cfg.get("protein_pca_dim", 0) or cfg.get("molecule_pca_dim", 0):
        raise RuntimeError(f"PCA-compressed checkpoints are not supported: {path.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}__{checkpoint}_feature_probes.pt"
    csv_path = out_dir / f"{path.stem}__{checkpoint}_feature_probes.csv"
    if out_path.exists() and csv_path.exists() and not force:
        print(f"SKIP_CACHED {out_path.name}")
        return pd.read_csv(csv_path).to_dict("records")

    gnn_seed = int(ck.get("gnn_seed", ck.get("seed")))
    boost_seed = int(ck.get("boost_seed", gnn_seed))
    state = _checkpoint_state(ck, checkpoint)
    print(f"FEATURE_PROBES {path.name} checkpoint={checkpoint} device={device}")

    esm, chem = L.load_embeddings(cfg.get("protein_embeddings"), cfg.get("molecule_embeddings"))
    Xm, Xp, splits = L.build(ck["regime"], int(ck["fold"]), esm, chem, seed=gnn_seed)
    eidx = _build_mp_edges(ck, Xm, splits)

    x_dict = {H.MOL: Xm.to(device), H.PROT: Xp.to(device)}
    eidx = {k: v.to(device) for k, v in eidx.items()}
    model = _make_model(ck, device)
    with torch.no_grad():
        model.encode(x_dict, eidx)  # initialize lazy layers
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    with torch.no_grad():
        stages = _encode_stages(model, x_dict, eidx)

    Xm_raw = _as_numpy(x_dict[H.MOL]).astype(np.float32)
    protein_sources = {
        "raw_prot": _as_numpy(x_dict[H.PROT]).astype(np.float32),
        "x0": _as_numpy(stages["x0"][H.PROT]).astype(np.float32),
        "x1": _as_numpy(stages["x1"][H.PROT]).astype(np.float32),
        "z2": _as_numpy(stages["z2"][H.PROT]).astype(np.float32),
    }

    ytr = _as_numpy(ck["probe_train_sup"][1]).astype(np.float32)
    yte = _as_numpy(ck["test_sup"][1]).astype(np.float32)
    rows = []
    scores_by_variant = {}
    for variant, blocks in VARIANTS:
        block_arrays = {name: protein_sources[name] for name in blocks}
        Xtr = _edge_features(Xm_raw, block_arrays, ck["probe_train_sup"])
        Xte = _edge_features(Xm_raw, block_arrays, ck["test_sup"])
        scores = train_boost(Xtr, ytr, Xte, seed=boost_seed)
        m = metrics(yte, scores)
        row = {
            "source_checkpoint": str(path.relative_to(ROOT)),
            "regime": ck["regime"],
            "fold": int(ck["fold"]),
            "repeat_id": int(ck["fold"]) if ck["regime"] == "transductive" else gnn_seed,
            "gnn_seed": gnn_seed,
            "boost_seed": boost_seed,
            "checkpoint": checkpoint,
            "variant": variant,
            "protein_blocks": "+".join(blocks),
            "feature_dim": int(Xtr.shape[1]),
            **{k: float(m[k]) for k in METRICS},
        }
        rows.append(row)
        scores_by_variant[variant] = scores
        print(f"  {variant:28s} dim={Xtr.shape[1]:4d} " + " ".join(f"{k}={m[k]:.3f}" for k in ("AUROC", "AUPRC", "MCC", "F1")))

    result = {
        "source_checkpoint": str(path.relative_to(ROOT)),
        "checkpoint": checkpoint,
        "rows": rows,
        "test_labels": yte,
        "scores": scores_by_variant,
    }
    torch.save(result, out_path)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return rows


def summarize(rows, out_dir, checkpoint):
    if not rows:
        print("No feature probe rows produced")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"signed_q99_{checkpoint}_feature_probe_runs.csv", index=False)
    summary_rows = []
    for keys, g in df.groupby(["regime", "checkpoint", "variant", "protein_blocks"], sort=False):
        for metric in ["AUROC", "AUPRC", "MCC", "F1"]:
            x = g[metric].astype(float)
            summary_rows.append({
                "regime": keys[0], "checkpoint": keys[1], "variant": keys[2],
                "protein_blocks": keys[3], "metric": metric,
                "mean": float(x.mean()),
                "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "n": int(g["repeat_id"].nunique()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"signed_q99_{checkpoint}_feature_probe_summary.csv", index=False)
    print("\nsummary")
    print(summary.pivot_table(index=["regime", "variant"], columns="metric", values="mean").round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/graph/full_full/v5/quantile_screen/training")
    ap.add_argument("--out-dir", default="results/graph/full_full/v5/signed_q99_feature_probes")
    ap.add_argument("--checkpoint", choices=["last_epoch", "best_val"], default="last_epoch")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--regime", choices=["both", "transductive", "inductive_molecule"], default="both")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = _device(args.device)
    ckpt_dir = ROOT / args.results_dir / "checkpoints"
    out_dir = ROOT / args.out_dir
    paths = sorted(ckpt_dir.glob("gnn_signed_q99_unentangled_boost_*_fold*_gnn*_boost*.pt"))
    if args.regime != "both":
        paths = [p for p in paths if f"_{args.regime}_" in p.name]
    print(f"signed q99 feature probes | checkpoint={args.checkpoint} | files={len(paths)} | device={device}")

    all_rows = []
    for path in paths:
        all_rows.extend(evaluate_checkpoint(path, args.checkpoint, device, out_dir, force=args.force))
    summarize(all_rows, out_dir, args.checkpoint)


if __name__ == "__main__":
    main()
