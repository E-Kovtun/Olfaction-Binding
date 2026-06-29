"""Train the bipartite link predictor (GNN or GAT) on the FULL_FULL dataset.

full_full = the LORAX / Hladis M2OR release (data/external/lorax_m2or), the most
complete variant we have. Unlike curated/full we do NOT make our own random
splits: we respect LORAX's pre-defined folds, where the test set is EC50-only
(highest-quality, dose-response, ~22% positive) while train/val are the full
noisy mix (primary + secondary + ec50, ~5.7% positive). See the notebook
graph_evaluation_full_full.ipynb for the protocol write-up.

Node features (the paper shows the molecule encoder is interchangeable):
  * proteins  — ESM2-650M mean-pooled (same encoder as curated/full), keyed by
                the raw amino-acid sequence.
  * molecules — ChemBERTa-77M (384-d), keyed by SMILES. GIN only covers 64% of
                LORAX molecules, so we use the LORAX-provided ChemBERTa here.

Message passing uses TRAIN edges only; supervision uses the LORAX train/val/test
splits directly. Only the unentangled XGBoost probe is reported
([raw_mol_emb || graph_prot_emb]); the MLP probe is dropped.

  uv run python scripts/modeling/train/train_graph_full_full.py --arch gnn --mp_mode signed
  uv run python scripts/modeling/train/train_graph_full_full.py --arch gat --mp_mode all_edges

Writes checkpoints to <results-dir>/checkpoints/ in the same shape the
graph_evaluation notebooks expect (regime == "ec50").
"""
import argparse, pathlib, pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import hetero as H
from orbind import lorax as L
from orbind.hetero_gat import HeteroGATLink
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def variant_name(args):
    q = args.mol_quality_q
    hist_tag = ("_history" if args.history_depth >= 2 else "_hist1") if args.history else ""
    return (args.mp_mode + (f"_q{int(q * 100)}" if q > 0 else "") + hist_tag
            + ("_rawp" if args.concat_raw_prot else ""))


def save_history(history, csv_path, plot_path):
    """Atomically persist exact per-epoch metrics and a compact diagnostic plot."""
    table = pd.DataFrame(history)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(".tmp.csv")
    table.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_test = "test_AUROC" in table.columns

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        (axes[0, 0], ["AUROC", "AUPRC"], "Ranking metrics"),
        (axes[0, 1], ["MCC", "F1"], "Threshold metrics @ 0.5"),
        (axes[1, 0], ["precision", "recall"], "Precision / recall @ 0.5"),
    ]
    for ax, names, title in panels:
        for metric in names:
            ax.plot(table.epoch, table[f"val_{metric}"], label=f"val {metric}")
            if has_test:
                ax.plot(table.epoch, table[f"test_{metric}"], "--", alpha=.75,
                        label=f"test {metric}")
        ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    axes[1, 1].plot(table.epoch, table.train_loss, color="black")
    axes[1, 1].set_title("Training loss"); axes[1, 1].grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    suptitle = ("Validation vs test history (--observe_test)" if has_test
                else "Validation history (test hidden — no --observe_test)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    tmp_plot = plot_path.with_suffix(".tmp.png")
    fig.savefig(tmp_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    tmp_plot.replace(plot_path)


def make_model(arch, mp_mode, args):
    if arch == "gnn":
        return H.HeteroLink(hidden=args.hidden, dropout=args.dropout, mp_mode=mp_mode)
    return HeteroGATLink(hidden=args.gat_hidden, heads=args.heads,
                         dropout=args.dropout, mp_mode=mp_mode)


def run(args):
    torch.manual_seed(args.seed)
    esm, chem = L.load_embeddings()
    print(f"embeddings: ESM proteins={len(esm)} | ChemBERTa mols={len(chem)}")
    print(f"\n[fold {args.fold} | {args.regime} | {args.arch} | {args.mp_mode}]")
    Xm, Xp, splits = L.build(args.regime, args.fold, esm, chem, seed=args.seed)
    x_dict = {H.MOL: Xm, H.PROT: Xp}

    # MP graph from TRAIN edges only; optional molecule-coverage quality filter
    mp_pos_a, mp_neg_a = splits["train"]["pos"], splits["train"]["neg"]
    if args.mol_quality_q > 0.0:
        mask = H.quality_mol_mask(mp_pos_a, mp_neg_a, Xm.shape[0], args.mol_quality_q)
        keep = set(int(i) for i in np.where(mask)[0])
        _f = lambda a: a[np.array([int(m) in keep for m in a[:, 0]], dtype=bool)] if len(a) else a
        mp_pos_a, mp_neg_a = _f(mp_pos_a), _f(mp_neg_a)
        print(f"  MP edges after quality filter: {len(mp_pos_a)} pos, {len(mp_neg_a)} neg")
    mp_pos = torch.tensor(mp_pos_a.T, dtype=torch.long)
    mp_neg = torch.tensor(mp_neg_a.T, dtype=torch.long)
    eidx = H.edge_index_dict(mp_pos, mp_neg, mode=args.mp_mode)
    sup = {s: H.sup_edges(splits[s]) for s in ("train", "val", "test")}

    model = make_model(args.arch, args.mp_mode, args)
    with torch.no_grad():
        model.encode(x_dict, eidx)              # init lazy params
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_idx, tr_y = sup["train"]
    pw = torch.tensor([(tr_y == 0).sum() / max((tr_y == 1).sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    if args.observe_test:
        print(
            "\n" + "!" * 70 + "\n"
            "!!  WARNING: --observe_test ENABLED                                 !!\n"
            "!!  Test-set metrics are computed and logged EVERY epoch.           !!\n"
            "!!  This is ONLY valid for gnn_training_diagnostics.ipynb.          !!\n"
            "!!  DO NOT use this flag for any hyperparameter sweep or tuning.    !!\n"
            "!!  DO NOT report results from a run that used this flag.           !!\n"
            + "!" * 70 + "\n"
        )

    variant = variant_name(args)
    history_dir = _root / args.results_dir / "history"
    run_id = f"{args.arch}_{variant}_{args.regime}_fold{args.fold}"
    history_csv = history_dir / f"{run_id}.csv"
    history_plot = history_dir / f"{run_id}.png"
    history = []

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(x_dict, eidx, tr_idx), tr_y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(x_dict, eidx)
            val_pred = torch.sigmoid(model.decode(z_eval, sup["val"][0])).numpy()
            if args.observe_test:
                test_pred = torch.sigmoid(model.decode(z_eval, sup["test"][0])).numpy()

        val_m = metrics(sup["val"][1].numpy(), val_pred)
        row = {"epoch": ep, "train_loss": float(loss.detach())}
        row.update({f"val_{k}": float(v) for k, v in val_m.items()})
        if args.observe_test:
            test_m = metrics(sup["test"][1].numpy(), test_pred)
            row.update({f"test_{k}": float(v) for k, v in test_m.items()})
        history.append(row)

        # CSV is updated every epoch so an interrupted 900-epoch run remains useful.
        save_table = pd.DataFrame(history)
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = history_csv.with_suffix(".tmp.csv")
        save_table.to_csv(tmp_csv, index=False)
        tmp_csv.replace(history_csv)
        if ep % args.plot_every == 0 or ep == args.epochs:
            save_history(history, history_csv, history_plot)
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            log = (f"    epoch {ep:3d} | loss={row['train_loss']:.4f} | "
                   f"val AUROC={val_m['AUROC']:.3f} AUPRC={val_m['AUPRC']:.3f}")
            if args.observe_test:
                log += f" | test AUROC={test_m['AUROC']:.3f} AUPRC={test_m['AUPRC']:.3f}"
            print(log, flush=True)

    save_history(history, history_csv, history_plot)
    hist_df = pd.DataFrame(history)
    best_i = int(hist_df["val_AUPRC"].idxmax())
    best = hist_df.loc[best_i]
    best_line = f"  best val AUPRC epoch={int(best.epoch)}: val={best.val_AUPRC:.3f}"
    if args.observe_test:
        best_line += f", test={best.test_AUPRC:.3f}"
    print(best_line)
    if args.observe_test:
        for metric in ("AUROC", "AUPRC", "MCC", "F1"):
            corr = hist_df[f"val_{metric}"].corr(hist_df[f"test_{metric}"])
            print(f"  epoch-wise val/test correlation {metric}: r={corr:.3f}")
    print(f"  history -> {history_csv.relative_to(_root)}")
    print(f"  plot    -> {history_plot.relative_to(_root)}")

    # ---- unentangled BOOST probe: [raw chemberta mol || graph-enriched esm prot] ----
    model.eval()
    with torch.no_grad():
        z = (model.encode_history(x_dict, eidx, depth=args.history_depth)
             if args.history else model.encode(x_dict, eidx))
    Xm_raw = x_dict[H.MOL].numpy()
    Zp = z[H.PROT].numpy()
    # The graph probe normally never sees the RAW ESM — only the trained 256-d
    # proj+ReLU bottleneck (even history concats stages of that bottleneck).
    # --concat_raw_prot puts the full raw ESM-1280 back alongside z_prot, so we
    # can tell "graph compresses/loses raw detail" from "graph adds new signal".
    prot_desc = f"{Zp.shape[1]} {args.arch.upper()}"
    if args.concat_raw_prot:
        Xp_raw = x_dict[H.PROT].numpy()
        Zp = np.concatenate([Xp_raw, Zp], axis=1)
        prot_desc = f"{Xp_raw.shape[1]} raw ESM + {prot_desc}"

    def feats(idx):
        return np.concatenate([Xm_raw[idx[0].numpy()], Zp[idx[1].numpy()]], axis=1)

    Xtr, ytr = feats(sup["train"][0]), sup["train"][1].numpy()
    Xte, yte = feats(sup["test"][0]),  sup["test"][1].numpy()
    print(f"  unentangled features: {Xtr.shape[1]}d "
          f"(mol {Xm_raw.shape[1]} ChemBERTa + prot {prot_desc})")
    scores = train_boost(Xtr, ytr, Xte, seed=args.seed)
    r = metrics(yte, scores)
    print("  unentangled_boost EC50-TEST " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))

    # ---- save in the shape graph_evaluation_full_full expects ----
    # variant fully identifies the run within (regime, arch): mp_mode [+q##] [+history]
    q = args.mol_quality_q
    prefix = args.arch                          # "gnn" or "gat"
    ckpt = {
        "label": f"{variant}_unentangled_boost", "variant": variant,
        "head": "unentangled_boost", "regime": args.regime,
        "arch": args.arch, "fold": args.fold,
        "config": {"mp_mode": args.mp_mode, "mol_quality_q": q, "history": args.history,
                   "history_depth": args.history_depth if args.history else 0,
                   "concat_raw_prot": args.concat_raw_prot,
                   "hidden": args.hidden if args.arch == "gnn" else args.gat_hidden},
        "test_sup": sup["test"], "test_scores": scores,
        "decoder_history": history,
        "decoder_history_csv": str(history_csv.relative_to(_root)),
    }
    rd = _root / args.results_dir / "checkpoints"; rd.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{variant}_unentangled_boost_{args.regime}_fold{args.fold}.pt"
    torch.save(ckpt, rd / fname)
    print(f"  checkpoint -> {args.results_dir}/checkpoints/{fname}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",    default="gnn", choices=["gnn", "gat"])
    ap.add_argument("--mp_mode", default="signed", choices=list(H.MP_MODES))
    ap.add_argument("--regime",  default="transductive",
                    choices=["transductive", "inductive_molecule"])
    ap.add_argument("--mol_quality_q", type=float, default=0.0,
                    help="Restrict MP edges to molecules above this coverage quantile (0=off)")
    ap.add_argument("--history", action="store_true",
                    help="Use protein history concat instead of final z_prot in the probe")
    ap.add_argument("--history_depth", type=int, default=2, choices=[1, 2],
                    help="History depth: 2=[x0||x1||x2] (last MP), 1=[x0||x1] (first MP)")
    ap.add_argument("--concat_raw_prot", action="store_true",
                    help="Probe on [raw ESM-1280 || z_prot] — gives the probe the full "
                         "raw protein embedding the graph bottleneck otherwise discards")
    ap.add_argument("--observe_test", action="store_true",
                    help="[DIAGNOSTICS ONLY] Log test-set metrics every epoch. "
                         "ONLY for gnn_training_diagnostics.ipynb. "
                         "NEVER use during HP sweeps — test leaks into your mental model.")
    ap.add_argument("--fold",    type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--results-dir", default="results/full_full/")
    ap.add_argument("--hidden",     type=int,   default=256)   # GNN width
    ap.add_argument("--gat_hidden", type=int,   default=128)   # GAT width
    ap.add_argument("--heads",      type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--lr",         type=float, default=5e-3)
    ap.add_argument("--epochs",     type=int,   default=900)
    ap.add_argument("--log-every",  type=int,   default=10,
                    help="Print compact validation/test status every N epochs")
    ap.add_argument("--plot-every", type=int,   default=25,
                    help="Refresh the history PNG every N epochs (CSV is saved every epoch)")
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
