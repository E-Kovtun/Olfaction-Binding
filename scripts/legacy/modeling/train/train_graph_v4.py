"""v4 diagnostic trainer — multi-checkpoint probing.

A single GNN/GAT link-prediction run (default: gnn · signed · shared · transductive)
that keeps SEVERAL encoder checkpoints and probes every one of them:

  Always kept (3):
    * best_auprc — epoch with max val AUPRC
    * best_loss  — epoch with min validation loss
    * last       — final epoch

  With --detailed-diagnostics (+10):
    * periodic snapshots every epochs//10 epochs (ep 1/10 … 10/10)
    => 13 checkpoints total.

Per-epoch history logs train_loss AND val_loss (BCE on the val edges) plus the
usual validation metrics, so both losses can be plotted afterwards.

For EVERY kept checkpoint we fit an XGBoost probe on [raw ChemBERTa mol ||
graph-enriched prot] and compute train / val / test metrics. Everything (states,
scores, metrics, history) is saved into one v4 checkpoint file.

Isolated from the production trainer (train_graph_full_full.py) on purpose.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_root = Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind import hetero as H          # noqa: E402
from orbind import lorax as L           # noqa: E402
from orbind.hetero_gat import HeteroGATLink  # noqa: E402
from orbind.dataset import metrics      # noqa: E402

METRIC_KEYS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def make_model(arch, mp_mode, args):
    if arch == "gnn":
        return H.HeteroLink(hidden=args.hidden, dropout=args.dropout, mp_mode=mp_mode)
    return HeteroGATLink(hidden=args.gat_hidden, heads=args.heads,
                         dropout=args.dropout, mp_mode=mp_mode)


def fit_boost(Xtr, ytr, seed):
    """Same XGBoost config as orbind.baselines.train_boost, but returns the model
    so we can predict train/val/test from a single fit."""
    import xgboost as xgb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="aucpr", tree_method="hist", device=device,
                            n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf


def snapshot(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def run(args):
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)

    esm, chem = L.load_embeddings()
    print(f"embeddings: ESM proteins={len(esm)} | ChemBERTa mols={len(chem)}")
    print(f"\n[v4 | fold {args.fold} | {args.regime} | {args.arch} | {args.mp_mode} | shared]")
    Xm, Xp, splits = L.build(args.regime, args.fold, esm, chem, seed=args.seed)
    x_dict = {H.MOL: Xm, H.PROT: Xp}

    # Shared supervision: the GNN and the probe use the same train edges.
    train_split = splits["train"]
    mp_pos = torch.tensor(train_split["pos"].T, dtype=torch.long)
    mp_neg = torch.tensor(train_split["neg"].T, dtype=torch.long)
    eidx = H.edge_index_dict(mp_pos, mp_neg, mode=args.mp_mode)
    sup = {"train": H.sup_edges(train_split),
           "val": H.sup_edges(splits["val"]),
           "test": H.sup_edges(splits["test"])}
    print(f"  train pos/neg={len(train_split['pos'])}/{len(train_split['neg'])} | "
          f"val={splits['val']['pos'].shape[0]}/{splits['val']['neg'].shape[0]} | "
          f"test={splits['test']['pos'].shape[0]}/{splits['test']['neg'].shape[0]}")

    model = make_model(args.arch, args.mp_mode, args)
    with torch.no_grad():
        model.encode(x_dict, eidx)          # init lazy params
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
        if args.lr_scheduler else None)

    tr_idx, tr_y = sup["train"]
    val_idx, val_y = sup["val"]
    pw = torch.tensor([(tr_y == 0).sum() / max((tr_y == 1).sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    # Periodic snapshot cadence for detailed diagnostics.
    period = max(1, args.epochs // 10) if args.detailed_diagnostics else 0
    periodic = {}                            # epoch -> state_dict

    history = []
    best_auprc, best_auprc_state, best_auprc_ep = -1.0, None, 0
    best_loss, best_loss_state, best_loss_ep = float("inf"), None, 0

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        train_logits = model(x_dict, eidx, tr_idx)
        loss = loss_fn(train_logits, tr_y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(x_dict, eidx)
            val_logits = model.decode(z_eval, val_idx)
            val_loss = float(loss_fn(val_logits, val_y))
            val_m = metrics(val_y.numpy(), torch.sigmoid(val_logits).numpy())

        row = {"epoch": ep, "train_loss": float(loss.detach()), "val_loss": val_loss,
               "lr": float(opt.param_groups[0]["lr"])}
        row.update({f"val_{k}": float(v) for k, v in val_m.items()})
        history.append(row)

        if val_m["AUPRC"] > best_auprc:
            best_auprc, best_auprc_ep = val_m["AUPRC"], ep
            best_auprc_state = snapshot(model)
        if val_loss < best_loss:
            best_loss, best_loss_ep = val_loss, ep
            best_loss_state = snapshot(model)
        if period and (ep % period == 0):
            periodic[ep] = snapshot(model)

        if scheduler is not None:
            scheduler.step(val_m["AUPRC"])
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            print(f"    epoch {ep:4d} | train_loss={row['train_loss']:.4f} "
                  f"val_loss={val_loss:.4f} | val AUROC={val_m['AUROC']:.3f} "
                  f"AUPRC={val_m['AUPRC']:.3f} | lr={row['lr']:.2e}", flush=True)

    final_state = snapshot(model)
    print(f"  best val AUPRC={best_auprc:.4f} @ ep{best_auprc_ep} | "
          f"best val loss={best_loss:.4f} @ ep{best_loss_ep}")

    # ---- assemble the checkpoint set ----
    ckpt_states = {}
    if args.detailed_diagnostics:
        for ep in sorted(periodic):
            ckpt_states[f"ep{ep:04d}"] = (ep, periodic[ep])
    ckpt_states["best_auprc"] = (best_auprc_ep, best_auprc_state)
    ckpt_states["best_loss"] = (best_loss_ep, best_loss_state)
    ckpt_states["last"] = (args.epochs, final_state)
    print(f"  probing {len(ckpt_states)} checkpoints "
          f"({'detailed' if args.detailed_diagnostics else 'standard'} mode)")

    # ---- probe every checkpoint: XGBoost on [raw mol || graph prot] ----
    Xm_raw = x_dict[H.MOL].numpy()
    y_split = {s: sup[s][1].numpy() for s in ("train", "val", "test")}
    checkpoints = {}
    for tag, (ep, state) in ckpt_states.items():
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in state.items()})
        model.eval()
        with torch.no_grad():
            z = model.encode(x_dict, eidx)
        Zp = z[H.PROT].numpy()

        def feats(split):
            idx = sup[split][0]
            return np.concatenate([Xm_raw[idx[0].numpy()], Zp[idx[1].numpy()]], axis=1)

        clf = fit_boost(feats("train"), y_split["train"], seed=args.seed)
        scores, mets = {}, {}
        for s in ("train", "val", "test"):
            sc = clf.predict_proba(feats(s))[:, 1]
            scores[s], mets[s] = sc, metrics(y_split[s], sc)
        checkpoints[tag] = {"epoch": ep, "state": state, "scores": scores, "metrics": mets}
        print(f"    [{tag:>10} ep{ep:>4}] "
              + " | ".join(f"{s}: AUPRC={mets[s]['AUPRC']:.3f} AUROC={mets[s]['AUROC']:.3f}"
                           for s in ("train", "val", "test")))

    # ---- persist ----
    variant = f"{args.mp_mode}_shared"
    run_id = f"{args.arch}_{variant}_{args.regime}_fold{args.fold}"
    rd = _root / args.results_dir
    (rd / "history").mkdir(parents=True, exist_ok=True)
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_csv = rd / "history" / f"{run_id}.csv"
    hist_df.to_csv(hist_csv, index=False)

    out = {
        "label": run_id, "variant": variant, "regime": args.regime, "arch": args.arch,
        "fold": args.fold,
        "config": {"mp_mode": args.mp_mode, "shared": True, "lr": args.lr,
                   "grad_clip": args.grad_clip, "lr_scheduler": args.lr_scheduler,
                   "epochs": args.epochs, "detailed_diagnostics": args.detailed_diagnostics,
                   "hidden": args.hidden if args.arch == "gnn" else args.gat_hidden},
        "sup": {s: sup[s] for s in ("train", "val", "test")},
        "history": history,
        "history_csv": str(hist_csv.relative_to(_root)),
        "checkpoints": checkpoints,          # tag -> {epoch, state, scores{tr,va,te}, metrics{tr,va,te}}
        "checkpoint_tags": list(checkpoints),
    }
    ckpt_path = rd / "checkpoints" / f"{run_id}.pt"
    torch.save(out, ckpt_path)
    print(f"  saved -> {ckpt_path.relative_to(_root)}  ({len(checkpoints)} checkpoints)")
    print(f"  history -> {hist_csv.relative_to(_root)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gnn", choices=["gnn", "gat"])
    ap.add_argument("--mp_mode", default="signed", choices=list(H.MP_MODES))
    ap.add_argument("--regime", default="transductive",
                    choices=["transductive", "inductive_molecule"])
    ap.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--detailed-diagnostics", action="store_true",
                    help="Also snapshot a checkpoint every epochs//10 epochs (13 total).")
    ap.add_argument("--results-dir", default="results/graph/full_full/legacy/pre_v5/v4/full_full")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lr-scheduler", action="store_true", default=True)
    ap.add_argument("--no-lr-scheduler", dest="lr_scheduler", action="store_false")
    ap.add_argument("--scheduler-patience", type=int, default=100)
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--gat_hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.grad_clip < 0:
        ap.error("--grad-clip must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
