"""MP baseline (LORAX-style): concat[molecule || protein] -> MLP -> bind / no-bind.

Weighted in two senses:
  * the train/test split is stratified by label (positives present on both sides);
  * the loss uses pos_weight = #neg/#pos to counter the ~1:11 imbalance.

Run (after embeddings exist):
  uv run python scripts/modeling/train_mp.py \
      --pairs data/processed/pairs_curated.csv \
      --prot  data/embeddings/proteins/esm2_650m.npz \
      --mol   data/embeddings/molecules/gin_supervised_contextpred.npz \
      --split stratified            # or group_receptor
"""
import argparse, pathlib, sys
import numpy as np

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import dataset as D


def build_mlp(in_dim, hidden, p):
    import torch.nn as nn
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p)]
        d = h
    layers += [nn.Linear(d, 1)]
    return nn.Sequential(*layers)


def metrics(y, p):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 matthews_corrcoef, f1_score, precision_score, recall_score)
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--prot", default="data/embeddings/proteins/esm2_650m.npz")
    ap.add_argument("--mol", default="data/embeddings/molecules/gin_supervised_contextpred.npz")
    ap.add_argument("--split", default="stratified",
                    choices=["stratified", "group_receptor", "group_molecule"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", default="512,128")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--save-split", default="data/processed/mp_split.csv")
    args = ap.parse_args()

    import torch, torch.nn as nn
    torch.manual_seed(args.seed)

    print("loading data ...")
    X, y, pairs = D.assemble(args.pairs, args.prot, args.mol)
    tr, te = D.split(pairs, y, kind=args.split, test_size=args.test_size, seed=args.seed)
    print(f"  split={args.split}: train={tr.sum()} (pos {int(y[tr].sum())}) | "
          f"test={te.sum()} (pos {int(y[te].sum())})")

    # standardize on train statistics
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd

    Xtr = torch.tensor(Xs[tr]); ytr = torch.tensor(y[tr])
    Xte = torch.tensor(Xs[te]); yte = torch.tensor(y[te])
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    print(f"  pos_weight={pos_weight.item():.1f}")

    model = build_mlp(X.shape[1], [int(h) for h in args.hidden.split(",")], args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = Xtr.shape[0]
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]).squeeze(-1), ytr[b])
            loss.backward(); opt.step()
        if ep % 20 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(Xte).squeeze(-1)).numpy()
            m = metrics(yte.numpy(), p)
            print(f"  epoch {ep:3d} | " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))

    # final report + save split
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte).squeeze(-1)).numpy()
    print("\n=== TEST metrics ===")
    for k, v in metrics(yte.numpy(), p).items():
        print(f"  {k}: {v:.3f}")

    pairs = pairs.copy(); pairs["split"] = np.where(tr, "train", "test")
    pathlib.Path(args.save_split).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.save_split, index=False)
    print(f"\nsaved split -> {args.save_split}")


if __name__ == "__main__":
    main()
