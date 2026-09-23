"""Two table runs side by side: value, place, and what moved between them.

Built to answer "did swapping the protein source lift anything, or just reshuffle
the order" without reading two printouts in two terminals. Takes the `main_long.csv`
that `m1_main_tables.py` writes for each run and joins them on
(dataset, regime, method, metric).

Both sides must have been produced with the SAME flags -- same metrics, same
baselines, same alpha -- or the comparison quietly compares two different tables.
The header prints each side's row count so a mismatch is visible.

    python scripts/legacy/04_compare_runs.py \
        --a results/article_tables/main --a-label ESM-1b \
        --b results/article_tables/esm3/main --b-label ESM3

    # only the cells that moved by more than noise:
    python scripts/legacy/04_compare_runs.py --a A --b B --min-delta 0.01
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

KEYS = ["dataset", "regime", "method", "metric"]

#: Lower is better for these, so a positive delta is a WORSENING. Everything else
#: (AUROC, AUPRC, MCC, F1, R2, Pearson, Spearman) is higher-is-better.
LOWER_IS_BETTER = {"RMSE", "MAE"}


def load(path: pathlib.Path) -> pd.DataFrame:
    csv = path if path.suffix == ".csv" else path / "main_long.csv"
    if not csv.exists():
        raise SystemExit(f"no main_long.csv at {csv} -- run m1_main_tables.py with "
                         f"--out {path} first")
    df = pd.read_csv(csv)
    missing = [k for k in KEYS if k not in df.columns]
    if missing:
        raise SystemExit(f"{csv} predates the dataset/regime columns ({missing}); "
                         f"regenerate it with the current m1_main_tables.py")
    return df


def improvement(row) -> float:
    """Signed 'did it get better', in the metric's own units."""
    d = row["mean_b"] - row["mean_a"]
    return -d if row["metric"] in LOWER_IS_BETTER else d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline run: its out dir or main_long.csv")
    ap.add_argument("--b", required=True, help="the run being compared against it")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--metric", nargs="+", default=None,
                    help="restrict to these metrics (default: all shared ones)")
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="hide cells whose value moved less than this")
    ap.add_argument("--out", default=None, help="also write the joined table here as csv")
    args = ap.parse_args(argv)

    a, b = load(pathlib.Path(args.a)), load(pathlib.Path(args.b))
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    print(f"{args.a_label}: {len(a)} rows | {args.b_label}: {len(b)} rows | "
          f"matched: {len(m)}")
    only_a = len(a) - len(m)
    if only_a:
        print(f"  note: {only_a} row(s) of {args.a_label} have no counterpart in "
              f"{args.b_label} -- the two runs do not cover the same cells")

    if args.metric:
        m = m[m["metric"].isin(args.metric)]
    if args.dataset:
        m = m[m["dataset"].isin(args.dataset)]
    m = m[(m["mean_b"] - m["mean_a"]).abs() >= args.min_delta]
    if m.empty:
        print("nothing to show")
        return 0

    m = m.assign(better=m.apply(improvement, axis=1),
                 d_rank=m["rank_b"] - m["rank_a"])
    m = m.sort_values(KEYS)

    width = max(len(s) for s in m["method"])
    for (ds, reg), g in m.groupby(["dataset", "regime"], sort=False):
        print(f"\n=== {ds} / {reg}")
        print(f"  {'method':<{width}}  {'metric':<9} "
              f"{args.a_label:>9} {args.b_label:>9} {'delta':>8}  "
              f"{'place':>11}  moved")
        for _, r in g.iterrows():
            # The place is what a reader of the paper actually sees change; a cell
            # can move 0.02 and keep its rank, or move 0.004 and lose two places.
            place = f"{r['rank_a']:.2f} -> {r['rank_b']:.2f}"
            arrow = ("better" if r["better"] > 0 else
                     "worse" if r["better"] < 0 else "same")
            print(f"  {r['method']:<{width}}  {r['metric']:<9} "
                  f"{r['mean_a']:9.3f} {r['mean_b']:9.3f} "
                  f"{r['mean_b'] - r['mean_a']:+8.3f}  {place:>11}  {arrow}")

    print(f"\nsummary over {len(m)} shared cells "
          f"({args.b_label} vs {args.a_label}, sign already corrected for "
          f"lower-is-better metrics):")
    for method, g in m.groupby("method"):
        up = int((g["better"] > 0).sum())
        print("  %-*s  better in %d/%d cells, mean move %+.4f, mean place %+.2f"
              % (width, method, up, len(g), g["better"].mean(), g["d_rank"].mean()))

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        m.to_csv(out, index=False)
        print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
