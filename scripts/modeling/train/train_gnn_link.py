"""Train the heterogeneous bipartite link predictor (molecule <-> protein).

Two regimes (run both by default):
  transductive        — hold out edges (all nodes known); ≈ stratified.
  inductive_molecule  — hold out whole molecules; ≈ group_molecule (cold start).

Three message-passing modes (--mp_mode):
  pos_only  — only positive edges in the MP graph (default).
  all_edges — positive AND negative edges in the MP graph (sign-blind).
  signed    — negatives contribute with explicit minus sign via separate SAGEConv.

Head: always unentangled — encoder trained with MLP loss, then frozen;
  baseline MLP and XGBoost trained on [raw_mol_emb || gnn_prot_emb].
  Saves TWO checkpoints (unentangled_mlp, unentangled_boost) per regime.

  --history: use [x0||x1||x2] protein embedding (3× wider) instead of final z_prot.

Optional quality filter (--mol_quality_q):
  Restricts MP edges to molecules above a coverage quantile (0=off, rec: 0.87).

Examples:
  uv run python scripts/modeling/train/train_gnn_link.py --mp_mode signed
  uv run python scripts/modeling/train/train_gnn_link.py --mp_mode signed --history
"""
import argparse, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import hetero as H
from orbind.dataset import metrics

REGIMES = ["transductive", "inductive_molecule"]
METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def _apply_quality_filter(mp_pos, mp_neg, pos, neg, n_mol, quantile):
    mask = H.quality_mol_mask(pos, neg, n_mol, quantile)
    quality_idx = set(int(i) for i in np.where(mask)[0])
    def _filter(mp):
        keep = torch.tensor([int(mp[0, i]) in quality_idx for i in range(mp.shape[1])])
        return mp[:, keep]
    mp_pos_f, mp_neg_f = _filter(mp_pos), _filter(mp_neg)
    print(f"  MP edges after quality filter: {mp_pos_f.shape[1]} pos, {mp_neg_f.shape[1]} neg "
          f"(was {mp_pos.shape[1]} pos, {mp_neg.shape[1]} neg)")
    return mp_pos_f, mp_neg_f


def _unentangled_heads(model, x_dict, eidx, sup, seed, history=False):
    """Baseline MLP and XGBoost on [raw_mol_emb || gnn_prot_emb].

    Molecule embeddings: raw GIN features (x_dict[MOL]), NOT passed through GNN.
    Protein  embeddings: GNN-learned z[PROT] from the trained encoder.
      history=True: concatenate initial + post-conv1 + post-conv2 (3× wider).
    Splits:  exactly the train/test indices from GNN training (no leakage).
    """
    from orbind.baselines import train_mlp, train_boost

    model.eval()
    with torch.no_grad():
        z = (model.encode_history if history else model.encode)(x_dict, eidx)

    Xm_raw = x_dict[H.MOL].numpy()   # raw GIN molecule embeddings
    Zp_gnn = z[H.PROT].numpy()       # GNN protein embeddings (or history concat)

    def feats(idx):
        return np.concatenate([Xm_raw[idx[0].numpy()],
                                Zp_gnn[idx[1].numpy()]], axis=1)

    tr_idx, tr_y = sup["train"]
    te_idx, te_y = sup["test"]
    Xtr = feats(tr_idx)
    Xte = feats(te_idx)
    ytr = tr_y.numpy()
    yte = te_y.numpy()

    tag = "(history)" if history else ""
    print(f"  unentangled{tag} features: {Xtr.shape[1]}d "
          f"(mol {Xm_raw.shape[1]} raw + prot {Zp_gnn.shape[1]} GNN)")

    mlp_scores   = train_mlp(Xtr, ytr, Xte, seed=seed)
    boost_scores = train_boost(Xtr, ytr, Xte, seed=seed)

    return {"mlp": (mlp_scores, yte), "boost": (boost_scores, yte)}


def run_regime(regime, Xm, Xp, pos, neg, n_mol, args):
    torch.manual_seed(args.seed)
    splits, mp_pos, mp_neg = H.make_splits(pos, neg, n_mol, regime=regime, seed=args.seed)
    x_dict = {H.MOL: Xm, H.PROT: Xp}

    if args.mol_quality_q > 0.0:
        mp_pos, mp_neg = _apply_quality_filter(mp_pos, mp_neg, pos, neg, n_mol, args.mol_quality_q)

    eidx = H.edge_index_dict(mp_pos, mp_neg, mode=args.mp_mode)
    sup  = {s: H.sup_edges(splits[s]) for s in ("train", "val", "test")}
    print(f"  train pos/neg={len(splits['train']['pos'])}/{len(splits['train']['neg'])} | "
          f"test pos/neg={len(splits['test']['pos'])}/{len(splits['test']['neg'])}")

    model = H.HeteroLink(hidden=args.hidden, dropout=args.dropout, mp_mode=args.mp_mode)
    with torch.no_grad():
        model.encode(x_dict, eidx)          # initialize lazy params
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_idx, tr_y = sup["train"]
    pw      = torch.tensor([(tr_y == 0).sum() / max((tr_y == 1).sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        loss_fn(model(x_dict, eidx, tr_idx), tr_y).backward()
        opt.step()
        if ep % 50 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x_dict, eidx, sup["val"][0])).numpy()
            m = metrics(sup["val"][1].numpy(), p)
            print(f"    epoch {ep:3d} | val " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))

    q          = args.mol_quality_q
    mode_label = args.mp_mode + (f"_q{int(q * 100)}" if q > 0.0 else "")
    base_ckpt  = {
        "model_state": model.state_dict(),
        "config": {"hidden": args.hidden, "dropout": args.dropout,
                   "mp_mode": args.mp_mode, "dec_layers": 3},
        "mol_quality_q": q,
        "eidx":    eidx,
        "test_sup": sup["test"],
        "regime":  regime,
    }

    def _save(label, head, extra=None):
        ckpt = {**base_ckpt, "label": label, "head": head}
        if extra:
            ckpt.update(extra)
        fname = f"gnn_{label}_{regime}.pt"
        rd = _root / args.results_dir / "checkpoints"; rd.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, rd / fname)
        print(f"  checkpoint -> {args.results_dir}/checkpoints/{fname}")

    unent    = _unentangled_heads(model, x_dict, eidx, sup, args.seed, history=args.history)
    hist_tag = "_history" if args.history else ""
    for sub, (scores, yte) in unent.items():
        sub_label = mode_label + f"_unentangled{hist_tag}_{sub}"
        _save(sub_label, f"unentangled_{sub}", {"test_scores": scores})
        r = metrics(yte, scores)
        print(f"  unentangled_{sub} TEST " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))

    return metrics(unent["mlp"][1], unent["mlp"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime",        default="both",     choices=REGIMES + ["both"])
    ap.add_argument("--mp_mode",       default="pos_only", choices=list(H.MP_MODES),
                    help="Message-passing mode: pos_only, all_edges, signed")
    ap.add_argument("--mol_quality_q", type=float, default=0.0,
                    help="Restrict MP edges to molecules above this coverage quantile (0=off)")
    ap.add_argument("--history", action="store_true",
                    help="Use [x0||x1||x2] protein history instead of final z_prot")
    ap.add_argument("--pairs",       default="data/processed/pairs_curated.csv")
    ap.add_argument("--prot",        default="data/embeddings/proteins/esm2_650m_mean_curated.npz")
    ap.add_argument("--mol",         default="data/embeddings/molecules/gin_supervised_contextpred.npz")
    ap.add_argument("--results-dir", default="results/",
                    help="Dataset results root; checkpoints go to <dir>/checkpoints, "
                         "tables to <dir>/tables (use results/full/ for the full dataset)")
    ap.add_argument("--hidden",  type=int,   default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr",      type=float, default=5e-3)
    ap.add_argument("--epochs",  type=int,   default=300)
    ap.add_argument("--seed",    type=int,   default=42)
    args = ap.parse_args()

    Xm, Xp, pos, neg, n_mol = H.build_nodes(args.pairs, args.prot, args.mol)
    regimes = REGIMES if args.regime == "both" else [args.regime]
    cols = {}
    for r in regimes:
        print(f"\n[{r}]")
        cols[r] = run_regime(r, Xm, Xp, pos, neg, n_mol, args)
        print("  TEST " + " ".join(f"{k}={v:.3f}" for k, v in cols[r].items()))

    q          = args.mol_quality_q
    mode_label = args.mp_mode + (f"_q{int(q * 100)}" if q > 0.0 else "")
    hist_tag   = "_history" if args.history else ""
    csv_label  = mode_label + f"_unentangled{hist_tag}"
    tab  = pd.DataFrame({c: [cols[c][k] for k in METRICS] for c in cols}, index=METRICS).round(3)
    out  = _root / args.results_dir / "tables"; out.mkdir(parents=True, exist_ok=True)
    fname = f"gnn_link_results_{csv_label}.csv"
    tab.to_csv(out / fname)
    print(f"\n=== GNN [{csv_label}] (TEST) ===\n{tab.to_string()}\nsaved -> {args.results_dir}/tables/{fname}")


if __name__ == "__main__":
    main()
