"""Build the MP results table across 4 settings, for an MLP or a boosting head.

Settings (columns): protein in {ESM, random} x split in {stratified, molecule}.
Both heads consume the SAME concatenated [molecule || protein] features.

  uv run python scripts/modeling/eval/eval_mp_table.py --head all     # mlp + boost
  uv run python scripts/modeling/eval/eval_mp_table.py --head boost

Writes results/tables/<head>_results.{md,csv}.
"""
import argparse, pathlib, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import dataset as D
from orbind.baselines import HEADS

# column order = honest progression
CONFIGS = [
    ("Mock·strat",  True,  "stratified"),
    ("ESM·strat",   False, "stratified"),
    ("ESM·mol",     False, "group_molecule"),
    ("Mock·mol",    True,  "group_molecule"),
]
METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def run_head(head, args):
    trainer = HEADS[head]
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
    out = _root / args.out_dir / "tables"; out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / f"{head}_results.csv")
    hdr = "| Metric | " + " | ".join(tab.columns) + " |"
    sep = "|" + "---|" * (len(tab.columns) + 1)
    rows = [f"| {k} | " + " | ".join(f"{tab.loc[k, c]:.3f}" for c in tab.columns) + " |" for k in METRICS]
    (out / f"{head}_results.md").write_text(
        f"# MP results — {head} head\n\n" + "\n".join([hdr, sep, *rows]) + "\n", encoding="utf-8")
    print(f"\n=== {head} table ===\n{tab.to_string()}\nsaved -> {args.out_dir}/tables/{head}_results.md")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="all", choices=["mlp", "boost", "all"])
    ap.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    ap.add_argument("--prot", default="data/embeddings/proteins/esm2_650m_mean.npz")
    ap.add_argument("--mol", default="data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz")
    ap.add_argument("--out-dir", default="results/curated/",
                    help="Dataset results root; tables go to <dir>/tables (use results/full/ for the full dataset)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    heads = ["mlp", "boost"] if args.head == "all" else [args.head]
    for h in heads:
        run_head(h, args)


if __name__ == "__main__":
    main()
