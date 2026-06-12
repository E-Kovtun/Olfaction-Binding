"""Train the heterogeneous bipartite link predictor (molecule <-> protein).

Two regimes (run both by default):
  transductive        — hold out edges (all nodes known); ≈ stratified.
  inductive_molecule  — hold out whole molecules; ≈ group_molecule (cold start).

Message-passing graph = train positives only; supervision = M2OR pos + tested
neg; unmeasured pairs are ignored (no random negatives). Decoder is bipartite,
so molecules are only ever scored against proteins.

  uv run python scripts/modeling/train_gnn_link.py --regime both
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


def run_regime(regime, Xm, Xp, pos, neg, n_mol, args):
    torch.manual_seed(args.seed)
    splits, mp = H.make_splits(pos, neg, n_mol, regime=regime, seed=args.seed)
    x_dict = {H.MOL: Xm, H.PROT: Xp}
    eidx = H.edge_index_dict(mp)
    sup = {s: H.sup_edges(splits[s]) for s in ("train", "val", "test")}
    print(f"  train pos/neg={len(splits['train']['pos'])}/{len(splits['train']['neg'])} | "
          f"test pos/neg={len(splits['test']['pos'])}/{len(splits['test']['neg'])}")

    model = H.HeteroLink(hidden=args.hidden, dropout=args.dropout)
    with torch.no_grad():                       # initialize lazy params
        model.encode(x_dict, eidx)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    tr_idx, tr_y = sup["train"]
    pw = torch.tensor([(tr_y == 0).sum() / max((tr_y == 1).sum(), 1)])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(x_dict, eidx, tr_idx)
        loss = loss_fn(logits, tr_y)
        loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(x_dict, eidx, sup["val"][0])).numpy()
            m = metrics(sup["val"][1].numpy(), p)
            print(f"    epoch {ep:3d} | val " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x_dict, eidx, sup["test"][0])).numpy()
    return metrics(sup["test"][1].numpy(), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="both", choices=REGIMES + ["both"])
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--prot", default="data/embeddings/proteins/esm2_650m.npz")
    ap.add_argument("--mol", default="data/embeddings/molecules/gin_supervised_contextpred.npz")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Xm, Xp, pos, neg, n_mol = H.build_nodes(args.pairs, args.prot, args.mol)
    regimes = REGIMES if args.regime == "both" else [args.regime]
    cols = {}
    for r in regimes:
        print(f"\n[{r}]")
        cols[r] = run_regime(r, Xm, Xp, pos, neg, n_mol, args)
        print("  TEST " + " ".join(f"{k}={v:.3f}" for k, v in cols[r].items()))

    tab = pd.DataFrame({c: [cols[c][k] for k in METRICS] for c in cols}, index=METRICS).round(3)
    out = _root / "results"; out.mkdir(exist_ok=True)
    tab.to_csv(out / "gnn_link_results.csv")
    print(f"\n=== GNN link-prediction (TEST) ===\n{tab.to_string()}\nsaved -> results/gnn_link_results.csv")


if __name__ == "__main__":
    main()
