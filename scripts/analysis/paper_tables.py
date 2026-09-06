#!/usr/bin/env python
"""The three paper tables -- M2OR, Carey, Hallem -- printed from what is on disk.

    python scripts/analysis/paper_tables.py
    python scripts/analysis/paper_tables.py --alphas 0.4 1.0 --mol-source chemberta
    python scripts/analysis/paper_tables.py --latex
    python scripts/analysis/paper_tables.py --no-baselines      # v8 grid only

One table per dataset, one block per regime, one row per model:

    boost          XGBoost on [ raw ESM || molecule ]. The number a graph must beat.
    GNN alpha=..   the v8 gate at that alpha, one-hot receptor nodes throughout.
    legacy GNN     the pre-v8 model. It is alpha=1 up to one global scalar applied
                   inside the forward pass -- invisible to a tree head, not to the
                   gradient, so it is a separate row and not a synonym for alpha=1.
    Hladis         Hladis et al. (ICLR 2023), and any other `--baseline` asked for.

Every cell is mean +/- a Student-t 95% interval over the dataset's own five splits
(folds 1-5, or the cold-molecule seeds 42-46 on m2or/inductive). The last column is the
PAIRED difference against boost on the metric of record, with the count of splits in
favour: the arms ran on the same split, so differencing them first removes the split and
is far stronger than two overlapping intervals.

TWO RESULT TREES, AND THE REASON THIS SCRIPT IS CAREFUL
-------------------------------------------------------
boost / gate / legacy come from `results/graph/v8_alpha_gate/metrics_*.csv`, written by
one sweep: same folds, same head, same coverage mask, paired by construction.

Hladis does not live there. It is an ensembler run under
`results/ensemble_logs/<pool>/<run>/metrics.csv`, and putting a number from a different
tree into the same table is exactly how a comparison stops meaning anything. So a
baseline row is admitted only after its run is matched on

    dataset + split family      (ofm dataset/split_family, or full_full mode)
    task                        (so a regression table never quotes an AUROC)
    the SPLIT IDS themselves    (folds 1-5 vs seeds 42-46)

and it is PAIRED with boost only when the split ids match exactly. When they do not, the
row is still printed -- hiding it would be worse -- but the delta column says `unpaired`
and the footnote names the mismatch. What this cannot check is the head: an ensembler run
with `--tune-boost` is not comparable to the sweep's fixed head, and `config.json` is
read for it and reported in the footnote so the reader can see which it was.

If a baseline has no run for a (dataset, regime), the row is a dash. Per the project's
own notes Hladis was run on M2OR and deferred on the insect panels, so expect exactly
that, and expect the footnote to say so rather than the row to vanish.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.analysis import alpha_grid as ag                      # noqa: E402

DATASET_LABEL = {"m2or": "M2OR", "cc": "Carey", "hc": "Hallem-Carlson"}
DATASET_ORDER = ["m2or", "cc", "hc"]
REGIME_ORDER = ["transductive", "inductive"]
# The columns each table prints. Fewer than the battery on purpose: this is the table,
# not the archive -- `headline_table.py --all-metrics` is where everything lives.
SHOW = {"regression": ["R2", "RMSE", "Pearson", "Spearman"],
        "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
# How a (dataset, regime) of the v8 grid appears in an ensembler run's config.json.
ENSEMBLE_SCOPE = {
    ("cc", "transductive"): dict(regime="ofm", dataset="cc", split_family="rand"),
    ("cc", "inductive"): dict(regime="ofm", dataset="cc", split_family="our_inductive"),
    ("hc", "transductive"): dict(regime="ofm", dataset="hc", split_family="rand"),
    ("hc", "inductive"): dict(regime="ofm", dataset="hc", split_family="our_inductive"),
    ("m2or", "transductive"): dict(regime="full_full", full_full_mode="transductive"),
    ("m2or", "inductive"): dict(regime="full_full",
                                full_full_mode="inductive_molecule_v5"),
}
ENSEMBLE_ROOT = "results/ensemble_logs"
# How a baseline keyword is spelled in a table. Anything not here prints as given.
BASELINE_LABEL = {"hladis": "Hladis", "prosmith": "ProSmith", "lorax": "LORAX",
                  "molor": "MolOR"}


# ------------------------------------------------------------------ the v8 grid rows

def grid_rows(df, ds, regime, alphas, metrics):
    """boost / gate@alpha / legacy for one (dataset, regime), already paired.

    Returns (rows, per-split values of the metric of record keyed by arm) -- the second
    is what the delta column needs, and it is taken from the same frame so the pairing
    cannot drift from the means printed beside it."""
    g = df[(df.dataset == ds) & (df.regime == regime)]
    if g.empty:
        return [], {}
    rows, cells = [], {}

    def add(label, q):
        if q.empty:
            return
        vals = {m: ag.ci(q[m]) for m in metrics if m in q.columns}
        n = max((v[2] for v in vals.values()), default=0)
        rows.append(dict(model=label, n=n, source="v8 grid",
                         **{m: vals.get(m, (np.nan, np.nan, 0)) for m in metrics}))
        cells[label] = q.set_index(ag.SPLIT)

    add("boost", g[g.arm == "boost_full"])
    for a in alphas:
        q = g[(g.arm == "gate") & np.isclose(g.alpha.astype(float), float(a))]
        add(f"GNN alpha={a:g}", q)
    add("legacy GNN", g[g.arm == "graph_legacy"])
    return rows, cells


# --------------------------------------------------------------- the ensembler rows

def _scope_of(cfg):
    """The (dataset, regime) an ensembler run belongs to, or None if it is not one of
    the six cells these tables are about."""
    for key, want in ENSEMBLE_SCOPE.items():
        if all(str(cfg.get(k, "")) == v for k, v in want.items()):
            return key
    return None


def find_runs(root, keyword):
    """Every ensembler run whose `--source` spec mentions `keyword`.

    The COMBO NAME cannot be used for this: it is "cls+mol" whatever the cls source
    happens to be, which is the trap the project's own notes call "one combo name is
    not one construction". The source spec in config.json is the only place the
    extractor is actually named."""
    out = []
    root = pathlib.Path(root)
    if not root.exists():
        return out
    for cfg_path in sorted(root.glob("*/*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:                      # noqa: BLE001 -- a half-written run
            continue
        srcs = " ".join(str(s) for s in (cfg.get("sources") or []))
        if keyword not in srcs.lower():
            continue
        scope = _scope_of(cfg)
        m = cfg_path.parent / "metrics.csv"
        if scope is None or not m.exists():
            continue
        out.append(dict(run=cfg_path.parent, pool=cfg_path.parent.parent.name,
                        dataset=scope[0], regime=scope[1], cfg=cfg,
                        task=cfg.get("task") or ("regression"
                                                 if cfg.get("regime") == "ofm"
                                                 else "classification"),
                        tuned=bool(cfg.get("tune_boost", False))))
    return out


def baseline_row(runs, ds, regime, metrics, task, combo=None):
    """One baseline row: the newest matching run, its combo, its per-split values.

    Newest by mtime, because a rerun of the same pool is a correction of the earlier
    one -- and the footnote names the run, so the choice is visible rather than
    implied."""
    cand = [r for r in runs if r["dataset"] == ds and r["regime"] == regime]
    cand = [r for r in cand if r["task"] == task]
    if not cand:
        return None
    r = max(cand, key=lambda x: (x["run"] / "metrics.csv").stat().st_mtime)
    d = pd.read_csv(r["run"] / "metrics.csv")
    d = d[d["kind"] == "combo"] if "kind" in d.columns else d
    if d.empty:
        return None
    if combo and "name" in d.columns and combo in set(d["name"]):
        d = d[d["name"] == combo]
    elif "name" in d.columns and d["name"].nunique() > 1:
        # several heads in one run; take the one with the most splits, then the best
        # metric of record -- and say which, in the footnote
        key = ag.OF_RECORD[task]
        pick = (d.groupby("name")
                .agg(n=("repeat", "nunique"), v=(key, "mean"))
                .sort_values(["n", "v"], ascending=[False, False]))
        d = d[d["name"] == pick.index[0]]
    return dict(run=r, frame=d.set_index("repeat"),
                combo=str(d["name"].iloc[0]) if "name" in d.columns else "?")


# --------------------------------------------------------------------------- pairing

def _delta(x, y):
    """(mean, half-width, won, n) of two aligned Series."""
    common = x.index.intersection(y.index)
    if not len(common):
        return np.nan, np.nan, 0, 0
    d = (pd.to_numeric(x.loc[common], errors="coerce")
         - pd.to_numeric(y.loc[common], errors="coerce")).dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    mu, hw, n = ag.ci(d)
    return mu, hw, int((d > 0).sum()), n


def paired_delta(a, b, metric):
    """Within the v8 grid: paired on (fold, seed).

    Both the split AND the model draw are shared between two arms of the same sweep, so
    differencing removes both. This is the strongest form available and it is only
    available here, inside one tree."""
    if a is None or b is None or metric not in a or metric not in b:
        return np.nan, np.nan, 0, 0
    return _delta(a[metric], b[metric])


def fold_ids(frame):
    """The split ids a frame rests on, whichever index it carries."""
    if frame is None:
        return []
    if isinstance(frame.index, pd.MultiIndex) and "fold" in frame.index.names:
        return sorted(set(frame.index.get_level_values("fold")))
    return sorted(set(frame.index))


def cross_tree_delta(base_frame, boost_frame, metric):
    """Against an ensembler run: paired on the FOLD only.

    The sweep carries a second axis the ensembler run does not -- the model seed -- so
    the honest reduction is to average our side over seeds first and pair on the split
    that both sides actually share. With one model seed, which is the whole grid, this
    is identical to the (fold, seed) pairing above; with several it is the only
    definition that does not invent a correspondence.

    Returns n=0 when the two rest on different folds, which is the statement that these
    numbers were measured on different held-out rows and must not be differenced."""
    if base_frame is None or boost_frame is None:
        return np.nan, np.nan, 0, 0
    if metric not in base_frame.columns or metric not in boost_frame.columns:
        return np.nan, np.nan, 0, 0
    ours = (pd.to_numeric(boost_frame[metric], errors="coerce")
            .groupby(level="fold").mean())
    theirs = pd.to_numeric(base_frame[metric], errors="coerce")
    theirs = theirs.groupby(level=0).mean()
    return _delta(theirs, ours)


# ------------------------------------------------------------------------- rendering

def _cell(t, w=16):
    if t is None or not np.isfinite(t[0]):
        return "--".rjust(w)
    return (f"{t[0]:.3f}" + (f" +/-{t[1]:.3f}" if np.isfinite(t[1]) else "")).rjust(w)


def render(ds, blocks, metrics, task, notes, w_model=16):
    head = (f"  {'model':<{w_model}}{'n':>4}"
            + "".join(m.rjust(16) for m in metrics)
            + f"{'d vs boost':>22}")
    print()
    print("=" * len(head))
    print(f"=== {DATASET_LABEL.get(ds, ds)}   [{task}, metric of record "
          f"{ag.OF_RECORD[task]}]")
    print("=" * len(head))
    for regime, rows in blocks:
        print(f"\n  {regime.upper()}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in rows:
            line = f"  {r['model']:<{w_model}}{r['n']:>4}"
            line += "".join(_cell(r.get(m)) for m in metrics)
            d = r.get("delta")
            if d is None:
                line += f"{'':>22}"
            elif d[3] == 0:
                line += f"{'unpaired':>22}"
            else:
                txt = (f"{d[0]:+.3f}" + (f" +/-{d[1]:.3f}" if np.isfinite(d[1]) else "")
                       + f" [{d[2]}/{d[3]}]")
                line += txt.rjust(22)
            print(line + ("" if r.get("source") == "v8 grid" else "  *"))
    if notes:
        print()
        for n in notes:
            print(f"  * {n}")


def latex(ds, blocks, metrics, task):
    print()
    print(f"% ---- {DATASET_LABEL.get(ds, ds)} ----")
    print(r"\begin{tabular}{l" + "r" * (len(metrics) + 1) + "}")
    print(r"\toprule")
    print("model & " + " & ".join(metrics) + r" & $\Delta$ vs boost \\")
    for regime, rows in blocks:
        print(r"\midrule")
        print(rf"\multicolumn{{{len(metrics) + 2}}}{{l}}{{\textit{{{regime}}}}} \\")
        for r in rows:
            cells = []
            for m in metrics:
                t = r.get(m)
                cells.append("--" if t is None or not np.isfinite(t[0])
                             else (rf"${t[0]:.3f} \pm {t[1]:.3f}$"
                                   if np.isfinite(t[1]) else f"${t[0]:.3f}$"))
            d = r.get("delta")
            dt = ("" if d is None else "unpaired" if d[3] == 0
                  else rf"${d[0]:+.3f}$ [{d[2]}/{d[3]}]")
            print(f"{r['model']} & " + " & ".join(cells) + f" & {dt} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ------------------------------------------------------------------------------ main

def build(df, args):
    runs_by_baseline = {b: find_runs(args.ensemble_root, b.lower())
                        for b in (args.baseline or [])}
    out = []
    for ds in [d for d in DATASET_ORDER if d in set(df.dataset)]:
        task = ag.TASK[ds]
        metrics = [m for m in (args.metrics or SHOW[task])
                   if m in ag.metrics_available(df, ds)]
        of_record = ag.OF_RECORD[task]
        blocks, notes = [], []
        for regime in [r for r in REGIME_ORDER if r in set(df.regime)]:
            rows, cells = grid_rows(df, ds, regime, args.alphas, metrics)
            if not rows and not any(runs_by_baseline.values()):
                continue
            boost = cells.get("boost")
            for r in rows:
                if r["model"] != "boost":
                    r["delta"] = paired_delta(cells.get(r["model"]), boost, of_record)

            for name, runs in runs_by_baseline.items():
                got = baseline_row(runs, ds, regime, metrics, task, args.combo)
                if got is None:
                    rows.append(dict(model=BASELINE_LABEL.get(name, name), n=0,
                                     source="absent", **{m: None for m in metrics}))
                    notes.append(f"{BASELINE_LABEL.get(name, name)}: no run under "
                                 f"{args.ensemble_root} for {ds}/{regime} -- row left "
                                 f"blank, not omitted")
                    continue
                f = got["frame"]
                vals = {m: ag.ci(f[m]) if m in f.columns else None for m in metrics}
                d = cross_tree_delta(f, boost, of_record)
                rows.append(dict(model=BASELINE_LABEL.get(name, name),
                                 n=int(f.index.nunique()),
                                 source="ensemble_logs", delta=d, **vals))
                r = got["run"]
                theirs, ours = fold_ids(f), fold_ids(boost)
                if not d[3]:
                    mism = (f"; SPLIT IDS DIFFER -- {theirs[:6]} against the sweep's "
                            f"{ours[:6]}, so NOT paired: the two rows are not measured "
                            f"on the same held-out data")
                elif set(theirs) != set(ours):
                    # It paired, on the intersection -- but then the delta is over a
                    # DIFFERENT set of folds than the two means printed beside it, and
                    # nothing in the row says so. This is the quiet version of the trap.
                    mism = (f"; rests on {theirs} while the sweep rows rest on {ours} "
                            f"-- the delta is over the {d[3]} shared fold(s) only, so it "
                            f"is NOT the difference of the two means printed")
                else:
                    mism = ""
                notes.append(
                    f"{BASELINE_LABEL.get(name, name)} @ {ds}/{regime}: "
                    f"{r['pool']}/{r['run'].name}, combo {got['combo']}, head "
                    f"{'TUNED -- not comparable to the sweep fixed head' if r['tuned'] else 'fixed'}"
                    f"{mism}")
            blocks.append((regime, rows))
        if blocks:
            out.append((ds, blocks, metrics, task, notes))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ag.DEFAULT_ROOT)
    ap.add_argument("--ensemble-root", default=ENSEMBLE_ROOT)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.4, 1.0])
    ap.add_argument("--mol-source", default="chemberta",
                    help="ONE source per table -- the tables are per dataset, not per "
                         "source. Pass another to reprint them on it")
    ap.add_argument("--nodes", nargs="+", default=["onehot"],
                    choices=["esm", "onehot"])
    ap.add_argument("--dataset", nargs="+", default=None)
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--seed", type=int, nargs="+", default=None)
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--baseline", nargs="+", default=["hladis"],
                    help="extractor keywords matched against each ensembler run's "
                         "--source spec (not its combo name, which is 'cls+mol' "
                         "whatever the source is). e.g. hladis prosmith lorax molor")
    ap.add_argument("--no-baselines", dest="baseline", action="store_const", const=[],
                    help="v8 grid only -- one tree, everything paired by construction")
    ap.add_argument("--combo", default=None,
                    help="combo name to read from a baseline run (default: the one with "
                         "the most splits)")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()

    df = ag.load(root=a.root, mol_source=a.mol_source, nodes=a.nodes,
                 dataset=a.dataset, regime=a.regime, seed=a.seed)
    tables = build(df, a)
    if not tables:
        raise SystemExit("nothing to print")
    print(f"\nmolecule source: {a.mol_source}   nodes: {','.join(a.nodes)}   "
          f"cells: mean +/- 95% t-CI over the splits; "
          f"[won/n] = splits where the row beat boost")
    for ds, blocks, metrics, task, notes in tables:
        (latex if a.latex else render)(*( (ds, blocks, metrics, task) if a.latex
                                          else (ds, blocks, metrics, task, notes)))


if __name__ == "__main__":
    main()
