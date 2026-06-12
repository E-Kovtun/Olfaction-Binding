"""Build the MP results table across 4 settings, for an MLP or a boosting head.

Settings (columns): protein in {ESM, random} x split in {stratified, molecule}.
Both heads consume the SAME concatenated [molecule || protein] features.

  uv run python scripts/modeling/eval_mp_table.py --head all     # mlp + boost
  uv run python scripts/modeling/eval_mp_table.py --head boost

Writes results/<head>_results.{md,csv}.
"""
import argparse, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import dataset as D

# column order = honest progression
CONFIGS = [
    ("Mock·strat",  True,  "stratified"),
    ("ESM·strat",   False, "stratified"),
    ("ESM·mol",     False, "group_molecule"),
    ("Mock·mol",    True,  "group_molecule"),
]
METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def train_mlp(Xtr, ytr, Xte, seed=42, hidden=(512, 128), dropout=0.3, lr=1e-3, epochs=100, batch=256):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr, Xte = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]; opt.zero_grad()
            loss_fn(model(Xtr[b]).squeeze(-1), ytr[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(-1)).numpy()


def train_boost(Xtr, ytr, Xte, seed=42):
    import xgboost as xgb
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def run_head(head, args):
    trainer = {"mlp": train_mlp, "boost": train_boost}[head]
    cols = {}
    for name, rand, kind in CONFIGS:
        print(f"\n[{head}] {name}")
        X, y, pairs = D.assemble(args.pairs, args.prot, args.mol, random_prot=rand, seed=args.seed)
        tr, te = D.split(pairs, y, kind=kind, test_size=args.test_size, seed=args.seed)
        print(f"  train={tr.sum()} (pos {int(y[tr].sum())}) | test={te.sum()} (pos {int(y[te].sum())})")
        p = trainer(X[tr], y[tr], X[te], seed=args.seed)
        m = D.metrics(y[te], p)
        cols[name] = m
        print("  " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))

    tab = pd.DataFrame({c: [cols[c][k] for k in METRICS] for c in cols}, index=METRICS).round(3)
    out = _root / "results"; out.mkdir(exist_ok=True)
    tab.to_csv(out / f"{head}_results.csv")
    # manual markdown (no tabulate dependency)
    hdr = "| Metric | " + " | ".join(tab.columns) + " |"
    sep = "|" + "---|" * (len(tab.columns) + 1)
    rows = [f"| {k} | " + " | ".join(f"{tab.loc[k, c]:.3f}" for c in tab.columns) + " |" for k in METRICS]
    (out / f"{head}_results.md").write_text(
        f"# MP results — {head} head\n\n" + "\n".join([hdr, sep, *rows]) + "\n", encoding="utf-8")
    print(f"\n=== {head} table ===\n{tab.to_string()}\nsaved -> results/{head}_results.md")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="all", choices=["mlp", "boost", "all"])
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--prot", default="data/embeddings/proteins/esm2_650m.npz")
    ap.add_argument("--mol", default="data/embeddings/molecules/gin_supervised_contextpred.npz")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    heads = ["mlp", "boost"] if args.head == "all" else [args.head]
    for h in heads:
        run_head(h, args)


if __name__ == "__main__":
    main()
