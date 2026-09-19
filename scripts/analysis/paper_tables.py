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
(folds 1-5, or the cold-molecule seeds 42-46 on m2or/inductive).

The right-hand block is the MEAN PLACE, ONE COLUMN PER METRIC: each model is ranked
among the others WITHIN each split and the ranks are averaged, 1 = best. Ranking inside
the split is what makes it readable at all -- folds differ enormously in difficulty (a
cold-molecule fold moves every model at once), so a comparison of means partly reports
which folds a model was measured on. The places are taken over the splits every row in
the block shares, and the block header says how many that is; a row that shares none is
excluded from the ranking rather than given a place against nobody.

A place per metric rather than one for the metric of record, because the four are not
redundant: on a regression table R2 and Pearson answer different questions (calibrated
error vs shape alone) and a model can lead one and trail the other, and on M2OR the
threshold-free pair (AUROC/AUPRC) routinely disagrees with the thresholded pair
(MCC/F1). One column collapsed exactly the disagreement worth seeing. Note RMSE is the
one metric where SMALLER is better; the ranking already accounts for it.

`--delta` swaps the whole block for a single column, the paired difference against boost
on the metric of record -- sharper, but it only exists where the splits match.

WHICH COMBO A BORROWED ROW MUST BE ON
-------------------------------------
The sweep's graph rows are `train_boost([ z_prot || raw molecule ])` -- the `cls+mol`
combo. An ensembler run often carries `cls+prot+mol` as well, and it usually scores
highest, because that head is handed the RAW ESM VECTOR on top of the baseline's own
feature: it is the baseline plus everything `boost` already has, so it outscores boost
partly by containing it. `COMBO_PREFERENCE` therefore matches the CONSTRUCTION rather
than maximising the score, the footnote lists what else the run offered, and a row that
can only be had with `prot` in it is printed with a warning attached.

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

and among the runs that match, the one with the MOST SPLITS wins -- newest only breaks a
tie. Newest-wins hands the row to whichever run was launched last, and the last launch is
routinely a one-fold probe. `_timing/` is skipped outright for the same reason: it holds
one-fold stopwatch runs made for the params/train-time table, run with
`--skip-checkpoints` and never intended as a score.

A row that survives all of that is still only PAIRED with boost where the split ids
match. When they do not, it is printed anyway -- hiding it would be worse -- and the
footnote names the mismatch. What none of this can check is the head: an ensembler run
with `--tune-boost` got a per-combo optuna search the sweep's rows never had, so
`config.json` is read for it and the footnote says which head trained the row.

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

DATASET_LABEL = {"m2or": "M2OR", "cc": "Carey", "hc": "Hallem-Carlson",
                 "cc_shrinked": "Carey (shrunk)",
                 "hc_shrinked": "Hallem-Carlson (shrunk)",
                 "cc_shrinked50": "Carey (shrunk 50%)",
                 "hc_shrinked50": "Hallem-Carlson (shrunk 50%)"}
# The shrunk panels sit LAST: every table's --dataset defaults to this list, so putting
# them anywhere else would silently reorder tables that already exist in the paper.
DATASET_ORDER = ["m2or", "cc", "hc", "cc_shrinked", "hc_shrinked",
                 "cc_shrinked50", "hc_shrinked50"]
REGIME_ORDER = ["transductive", "inductive"]
# The columns each table prints. Fewer than the battery on purpose: this is the table,
# not the archive -- `headline_table.py --all-metrics` is where everything lives.
SHOW = {"regression": ["R2", "RMSE", "Pearson", "Spearman"],
        "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
# The metrics a SMALLER value wins. Defined once in the base reader and imported, so
# this table and `alpha_grid.places` can never rank a column in opposite directions.
LOWER_IS_BETTER = ag.LOWER_IS_BETTER
# How a (dataset, regime) of the v8 grid appears in an ensembler run's config.json.
ENSEMBLE_SCOPE = {
    ("cc", "transductive"): dict(regime="ofm", dataset="cc", split_family="rand"),
    ("cc", "inductive"): dict(regime="ofm", dataset="cc", split_family="our_inductive"),
    ("hc", "transductive"): dict(regime="ofm", dataset="hc", split_family="rand"),
    ("hc", "inductive"): dict(regime="ofm", dataset="hc", split_family="our_inductive"),
    ("cc_shrinked", "transductive"): dict(regime="ofm", dataset="cc_shrinked",
                                          split_family="rand"),
    ("cc_shrinked", "inductive"): dict(regime="ofm", dataset="cc_shrinked",
                                       split_family="our_inductive"),
    ("hc_shrinked", "transductive"): dict(regime="ofm", dataset="hc_shrinked",
                                          split_family="rand"),
    ("hc_shrinked", "inductive"): dict(regime="ofm", dataset="hc_shrinked",
                                       split_family="our_inductive"),
    ("cc_shrinked50", "transductive"): dict(regime="ofm", dataset="cc_shrinked50",
                                            split_family="rand"),
    ("cc_shrinked50", "inductive"): dict(regime="ofm", dataset="cc_shrinked50",
                                         split_family="our_inductive"),
    ("hc_shrinked50", "transductive"): dict(regime="ofm", dataset="hc_shrinked50",
                                            split_family="rand"),
    ("hc_shrinked50", "inductive"): dict(regime="ofm", dataset="hc_shrinked50",
                                         split_family="our_inductive"),
    ("m2or", "transductive"): dict(regime="full_full", full_full_mode="transductive"),
    ("m2or", "inductive"): dict(regime="full_full",
                                full_full_mode="inductive_molecule_v5"),
}
ENSEMBLE_ROOT = "results/ensemble_logs"
# How a baseline keyword is spelled in a table. Anything not here prints as given.
BASELINE_LABEL = {"hladis": "Hladis", "prosmith": "ProSmith", "lorax": "LORAX",
                  "molor": "MolOR"}
# Pools that are bookkeeping, not results. `_timing/` holds one-fold runs made to
# measure training time for the paper's params/time table -- `--skip-checkpoints`, one
# repeat, no intention of being a score. Picking one as a baseline row is how a table
# ends up quoting a single fold of a stopwatch run.
SKIP_POOL_PREFIX = "_"
# Which combo of a baseline run belongs in THIS table, most preferred first.
#
# It has to match the construction of the row it sits beside. The sweep's graph rows are
# `train_boost([ z_prot || raw molecule ])` -- cls+mol. A `cls+prot+mol` combo hands the
# same head the RAW ESM VECTOR as well, so it is not the baseline architecture at all:
# it is that architecture plus everything `boost` already has, and it beats boost partly
# by containing it. Reading it as "Hladis" overstates Hladis and understates the graph.
COMBO_PREFERENCE = ["cls+mol", "cls"]
# What the PAPER's baseline rows actually stand on, per baseline. `--combo` overrides
# both this and the preference above.
#
# Hladis is reported at `cls+prot+mol`, and that is a deliberate convention, not a
# slip: `boost` in these tables IS prot+mol, so a baseline at cls+prot+mol answers
# "does this baseline's representation ADD anything to what boost already has", which
# is the question the cls-baseline head-to-head asks of every competitor at once. It is
# a DIFFERENT question from the one the graph rows answer -- those are cls+mol, i.e.
# "does the graph REPLACE ESM" -- so the two live in one table only with that said out
# loud, which is what `_baseline_notes` does.
PAPER_COMBO = {"hladis": "cls+prot+mol"}
# How to read a molecule source off an ensembler run. The FILE decides, not the source
# TYPE: a spec reads `mol=gin:data/embeddings/molecules/chemberta_77m_cc.npz`, where
# `gin` is only the npz loader and the actual embedding is ChemBERTa. Matching on the
# type would put a ChemBERTa row in the ECFP table and never say a word.
MOL_FINGERPRINT = {"chemberta": "chemberta",
                   "gin": "gin_supervised_contextpred",
                   "ecfp": "ecfp_"}


# ------------------------------------------------------------------ the v8 grid rows

def grid_rows(df, ds, regime, alphas, metrics):
    """boost / gate@alpha / legacy for one (dataset, regime), already paired.

    Returns (rows, per-split values of the metric of record keyed by arm) -- the second
    is what the delta column needs, and it is taken from the same frame so the pairing
    cannot drift from the means printed beside it."""
    g = df[(df.dataset == ds) & (df.regime == regime)]
    if g.empty:
        return [], {}
    if alphas is None:
        # `--alphas all`: whatever this cell actually holds. Resolved PER CELL rather
        # than once for the whole frame, because a sweep is usually finished unevenly
        # and a union would print empty rows for alphas this dataset never ran.
        alphas = sorted(pd.to_numeric(g.loc[g.arm == "gate", "alpha"],
                                      errors="coerce").dropna().unique())
    rows, cells = [], {}

    def add(label, q):
        if q.empty:
            return
        # seeds averaged inside each fold first: `n` is folds, the independent unit
        vals = {m: ag.ci_folds(q, m) for m in metrics if m in q.columns}
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

def mol_source_of(cfg):
    """Which molecule embedding an ensembler run was fed, or None if it is unreadable.

    This matters more than it looks. The insect Hladis pools carry three runs per cell
    -- concatCB, concatECFP, concatGIN -- identical in every other respect. Picking
    between them by file mtime, which is what any tiebreak on "newest" amounts to, puts
    whichever finished last into a table whose other rows are on one fixed source."""
    for spec in (cfg.get("sources") or []):
        spec = str(spec)
        if not spec.lower().startswith("mol="):
            continue
        for name, mark in MOL_FINGERPRINT.items():
            if mark in spec.lower():
                return name
    return None


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
        pool = cfg_path.parent.parent.name
        if scope is None or not m.exists() or pool.startswith(SKIP_POOL_PREFIX):
            continue
        out.append(dict(run=cfg_path.parent, pool=cfg_path.parent.parent.name,
                        dataset=scope[0], regime=scope[1], cfg=cfg,
                        mol_source=mol_source_of(cfg),
                        task=cfg.get("task") or ("regression"
                                                 if cfg.get("regime") == "ofm"
                                                 else "classification"),
                        tuned=bool(cfg.get("tune_boost", False))))
    return out


def baseline_row(runs, ds, regime, metrics, task, combo=None, mol_source=None):
    """One baseline row: the newest matching run, its combo, its per-split values.

    Newest by mtime, because a rerun of the same pool is a correction of the earlier
    one -- and the footnote names the run, so the choice is visible rather than
    implied."""
    cand = [r for r in runs if r["dataset"] == ds and r["regime"] == regime
            and r["task"] == task]
    if not cand:
        return None

    def _read(r):
        d = pd.read_csv(r["run"] / "metrics.csv")
        return d[d["kind"] == "combo"] if "kind" in d.columns else d

    # The run is chosen in this order, and the FIRST key is the point: when a combo has
    # been asked for, a run that actually contains it beats a fuller run that does not.
    # Without that, "the run with the most splits" can win and then silently fall back
    # to a different construction than the one requested.
    #
    #   1. same molecule embedding as the table  -- else the row is from another column
    #   2. does it offer the requested combo
    #   3. how many splits it has        -- a one-fold probe is not a table row
    #   4. fixed head over tuned         -- the sweep's rows are fixed-head
    #   5. newest, only to break a tie   -- otherwise the row depends on launch order
    scored = []
    for r in cand:
        d = _read(r)
        if d.empty:
            continue
        same_mol = bool(mol_source and r.get("mol_source") == mol_source)
        has = bool(combo and "name" in d.columns and combo in set(d["name"]))
        scored.append((1 if same_mol else 0,
                       1 if has else 0,
                       int(d["repeat"].nunique()),
                       0 if r["tuned"] else 1,
                       (r["run"] / "metrics.csv").stat().st_mtime, r, d))
    if not scored:
        return None
    *_, r, d = max(scored, key=lambda x: x[:5])

    offered = sorted(set(d["name"])) if "name" in d.columns else []
    requested = bool(combo)
    if combo and combo in offered:
        pick = combo
    else:
        # match the construction, do not maximise the score
        pick = next((c for c in COMBO_PREFERENCE if c in offered),
                    offered[0] if offered else None)
    if pick is not None:
        d = d[d["name"] == pick]
    return dict(run=r, frame=d.set_index("repeat"), combo=str(pick or "?"),
                offered=offered,
                n_runs=len(cand),
                requested=requested,
                missed=bool(combo and combo not in offered),
                wanted_mol=mol_source,
                mol_mismatch=bool(mol_source and r.get("mol_source") != mol_source),
                carries_prot="prot" in str(pick or ""))


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


def _to_folds(s):
    """Collapse a (fold, seed) series to one value per fold."""
    if isinstance(s.index, pd.MultiIndex) and "fold" in s.index.names:
        return pd.to_numeric(s, errors="coerce").groupby(level="fold").mean()
    return pd.to_numeric(s, errors="coerce")


def paired_delta(a, b, metric):
    """Within the v8 grid: paired on (fold, seed), then averaged over seeds per fold.

    Both the split AND the model draw are shared between two arms of the same sweep, so
    differencing removes both -- that is the strongest form available and it exists only
    here, inside one tree. Averaging the seeds afterwards is what keeps the interval
    honest: five seeds on one fold are one observation about generalisation, not five."""
    if a is None or b is None or metric not in a or metric not in b:
        return np.nan, np.nan, 0, 0
    common = a.index.intersection(b.index)
    if not len(common):
        return np.nan, np.nan, 0, 0
    d = (pd.to_numeric(a[metric].loc[common], errors="coerce")
         - pd.to_numeric(b[metric].loc[common], errors="coerce")).dropna()
    if d.empty:
        return np.nan, np.nan, 0, 0
    if isinstance(d.index, pd.MultiIndex) and "fold" in d.index.names:
        d = d.groupby(level="fold").mean()
    mu, hw, n = ag.ci(d)
    return mu, hw, int((d > 0).sum()), n


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

def mean_place(cells, baselines, metric):
    """Mean place of each model among the models in this block, 1 = best.

    Ranked WITHIN each split and then averaged, which is the only pooled comparison
    that survives the fact that folds differ wildly in difficulty: a cold-molecule fold
    can move every model by 0.1, and a mean-of-scores would report that rather than the
    ordering.

    The ranking runs on the folds SHARED by every model in the block. A model that
    overlaps nothing gets no place at all rather than a place computed on its own folds,
    which would not be a place in the same competition."""
    series = {}
    for label, frame in cells.items():
        if frame is None or metric not in frame.columns:
            continue
        v = pd.to_numeric(frame[metric], errors="coerce")
        v = v.groupby(level="fold").mean() if isinstance(frame.index, pd.MultiIndex) \
            else v.groupby(level=0).mean()
        series[label] = v.dropna()
    for label, frame in baselines.items():
        if frame is None or metric not in frame.columns:
            continue
        v = pd.to_numeric(frame[metric], errors="coerce").groupby(level=0).mean()
        series[label] = v.dropna()
    if not series:
        return {}, [], []
    folds = set.intersection(*(set(v.index) for v in series.values()))
    dropped = []
    if not folds:
        # one model is on a disjoint fold set (folds 1-5 against seeds 42-46, or a
        # one-fold probe). Drop the smallest offenders until the rest share something.
        for label in sorted(series, key=lambda k: len(series[k])):
            trial = {k: v for k, v in series.items() if k != label}
            if trial and set.intersection(*(set(v.index) for v in trial.values())):
                dropped.append(label)
                series = trial
                folds = set.intersection(*(set(v.index) for v in series.values()))
                break
    if not folds:
        return {}, sorted(series), []
    folds = sorted(folds)
    M = pd.DataFrame({k: v.reindex(folds) for k, v in series.items()})
    # direction from the metric NAME. It never mattered while only the metric of
    # record was ranked -- R2 and AUROC are both larger-is-better -- but a place per
    # metric puts RMSE in the block, where ranking it the same way would silently
    # award first place to the worst model.
    ranks = M.rank(axis=1, ascending=metric in LOWER_IS_BETTER, method="average")
    return ranks.mean().to_dict(), dropped, folds


def places_by_metric(cells, baselines, metrics):
    """`mean_place` for each metric: (places per metric, models dropped anywhere,
    the shared folds per metric).

    Per metric and not once, because a metric can be missing from a borrowed row while
    the others are present, and then that row is in one competition and not another --
    which is a fact about the run, and hiding it behind a single ranking would put a
    place next to a model that never ran that metric."""
    per, dropped, folds = {}, [], {}
    for m in metrics:
        p, d, f = mean_place(cells, baselines, m)
        per[m], folds[m] = p, f
        for lab in d:
            if lab not in dropped:
                dropped.append(lab)
    return per, dropped, folds


def _cell(t, w=16):
    if t is None or not np.isfinite(t[0]):
        return "--".rjust(w)
    return (f"{t[0]:.3f}" + (f" +/-{t[1]:.3f}" if np.isfinite(t[1]) else "")).rjust(w)


DIAL_NOTE = {
    "onehot": "   v8 gate: alpha 0 = structure alone -> 1 = the graph alone",
    "nodedial": "   v9 node dial: alpha 0 = receptor identity alone -> 1 = ESM nodes "
                "(legacy)",
}


def _place_widths(metrics):
    """One column per metric, each just wide enough for its own name."""
    return {m: max(len(m) + 2, 6) for m in metrics}


def _place_block(r, metrics, widths):
    pl = r.get("place")
    if not isinstance(pl, dict):
        return None
    out = []
    for m in metrics:
        v = pl.get(m)
        txt = "--" if v is None or not np.isfinite(v) else f"{v:.2f}"
        out.append(txt.rjust(widths[m]))
    return "".join(out)


def render(ds, blocks, metrics, task, notes, col="mean place", dial="", w_model=16):
    ranked = any(isinstance(r.get("place"), dict)
                 for _, rows, _ in blocks for r in rows)
    widths = _place_widths(metrics)
    lead = 2 + w_model + 4
    head = (f"  {'model':<{w_model}}{'n':>4}"
            + "".join(m.rjust(16) for m in metrics))
    if ranked:
        # the metric names appear twice, so the two blocks are named above them --
        # without that a reader cannot tell 0.894 from a place of 1.20 at a glance
        head += "  " + "".join(m.rjust(widths[m]) for m in metrics)
        group = (" " * lead + "value (mean +/- 95% t-CI)".center(16 * len(metrics))
                 + "  " + f"{col}, 1 = best".center(sum(widths.values())))
    else:
        head += f"{col:>22}"
        group = None
    print()
    print("=" * len(head))
    print(f"=== {DATASET_LABEL.get(ds, ds)}   [{task}, metric of record "
          f"{ag.OF_RECORD[task]}]{dial}")
    print("=" * len(head))
    for regime, rows, rank_note in blocks:
        print(f"\n  {regime.upper()}" + (f"   ({rank_note})" if rank_note else ""))
        if group:
            print(group)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for r in rows:
            line = f"  {r['model']:<{w_model}}{r['n']:>4}"
            line += "".join(_cell(r.get(m)) for m in metrics)
            block = _place_block(r, metrics, widths)
            if "delta" in r:
                d = r["delta"]
                txt = ("unpaired" if d is None or d[3] == 0 else
                       f"{d[0]:+.3f}"
                       + (f" +/-{d[1]:.3f}" if np.isfinite(d[1]) else "")
                       + f" [{d[2]}/{d[3]}]")
                line += txt.rjust(22)
            elif block is not None:
                line += "  " + block
            elif ranked:
                line += "  " + " " * sum(widths.values())
            else:
                line += "".rjust(22)   # boost under --delta: it IS the reference
            print(line + ("" if r.get("source") == "v8 grid" else "  *"))
    if notes:
        print()
        for n in notes:
            print(f"  * {n}")


def latex(ds, blocks, metrics, task, col="mean place", dial=""):
    ranked = any(isinstance(r.get("place"), dict)
                 for _, rows, _ in blocks for r in rows)
    extra = len(metrics) if ranked else 1
    print()
    print(f"% ---- {DATASET_LABEL.get(ds, ds)} ----")
    if dial:
        # which dial this is has to survive the copy-paste into the paper: alpha
        # means opposite things on the two of them
        print("%" + dial)
    print(r"\begin{tabular}{l" + "r" * len(metrics) + "|" + "r" * extra + "}")
    print(r"\toprule")
    if ranked:
        print(rf"& \multicolumn{{{len(metrics)}}}{{c}}{{value}} & "
              rf"\multicolumn{{{len(metrics)}}}{{c}}{{{col}, 1 = best}} \\")
        print("model & " + " & ".join(metrics) + " & " + " & ".join(metrics) + r" \\")
    else:
        print("model & " + " & ".join(metrics) + f" & {col} " + r"\\")
    for regime, rows, _ in blocks:
        print(r"\midrule")
        print(rf"\multicolumn{{{len(metrics) + extra + 1}}}{{l}}"
              rf"{{\textit{{{regime}}}}} \\")
        for r in rows:
            cells = []
            for m in metrics:
                t = r.get(m)
                cells.append("--" if t is None or not np.isfinite(t[0])
                             else (rf"${t[0]:.3f} \pm {t[1]:.3f}$"
                                   if np.isfinite(t[1]) else f"${t[0]:.3f}$"))
            if "delta" in r:
                d = r["delta"]
                tail = [("unpaired" if d is None or d[3] == 0
                         else rf"${d[0]:+.3f}$ [{d[2]}/{d[3]}]")]
            elif isinstance(r.get("place"), dict):
                tail = []
                for m in metrics:
                    v = r["place"].get(m)
                    tail.append("--" if v is None or not np.isfinite(v)
                                else f"{v:.2f}")
            else:
                tail = [""] * extra
            print(f"{r['model']} & " + " & ".join(cells + tail) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ------------------------------------------------------------------------------ main

def combo_for(name, args):
    """Which head this baseline is read at: the CLI first, then the paper's own
    convention, then the construction that matches the graph rows."""
    spec = getattr(args, "combo", None)
    if isinstance(spec, str):
        spec = [spec]
    for item in (spec or []):
        key, sep, val = str(item).partition("=")
        if not sep:
            return key                     # one name, applied to every baseline
        if key.lower() == name.lower():
            return val
    return PAPER_COMBO.get(name.lower())


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
            base_frames = {}

            for name, runs in runs_by_baseline.items():
                label = BASELINE_LABEL.get(name, name)
                got = baseline_row(runs, ds, regime, metrics, task,
                                   combo_for(name, args), args.mol_source)
                if got is None:
                    rows.append(dict(model=label, n=0, source="absent",
                                     **{m: None for m in metrics}))
                    notes.append(f"{label}: no run under {args.ensemble_root} for "
                                 f"{ds}/{regime} -- row left blank, not omitted")
                    continue
                f = got["frame"]
                base_frames[label] = f
                rows.append(dict(model=label, n=int(f.index.nunique()),
                                 source="ensemble_logs",
                                 **{m: ag.ci(f[m]) if m in f.columns else None
                                    for m in metrics}))
                notes.extend(_baseline_notes(label, ds, regime, got, boost, of_record))

            if getattr(args, "delta", False):
                for r in rows:
                    if r["model"] == "boost":
                        continue
                    src = cells.get(r["model"])
                    r["delta"] = (paired_delta(src, boost, of_record) if src is not None
                                  else cross_tree_delta(base_frames.get(r["model"]),
                                                        boost, of_record))
                note = ""
            else:
                per, dropped, folds = places_by_metric(cells, base_frames, metrics)
                for r in rows:
                    r["place"] = {m: per[m].get(r["model"]) for m in metrics}
                # the block header counts on the metric of record, or on the first
                # metric that ranked anything when --metrics leaves it out
                ref = folds.get(of_record) or next(
                    (f for f in folds.values() if f), [])
                k = len([r for r in rows
                         if any(v is not None for v in r["place"].values())])
                note = (f"places over {k} models on {len(ref)} shared split(s)"
                        if ref else "no shared splits -- no ranking possible")
                spans = {m: len(f) for m, f in folds.items()}
                if len(set(spans.values())) > 1:
                    # a borrowed row can carry AUROC and not MCC; then the two columns
                    # are two different competitions and the header must say so
                    note += ("; NOT all on the same splits -- "
                             + ", ".join(f"{m} {n}" for m, n in spans.items()))
                for label in dropped:
                    notes.append(
                        f"{label} is EXCLUDED from the ranking: it shares no split with "
                        f"the other rows, so it cannot hold a place in the same "
                        f"competition. Its metric cells above are still its own numbers.")
            blocks.append((regime, rows, note))
        if blocks:
            out.append((ds, blocks, metrics, task, notes))
    return out


def _baseline_notes(label, ds, regime, got, boost, of_record):
    """Everything about a borrowed row that the reader has to know and the row itself
    cannot show: which run it came from out of how many, which head trained it, which
    combo was taken out of what was offered, and whether its splits are ours."""
    r, f = got["run"], got["frame"]
    theirs, ours = fold_ids(f), fold_ids(boost)
    shared = len(set(theirs) & set(ours))
    bits = [f"{r['pool']}/{r['run'].name}"]
    if got["n_runs"] > 1:
        bits.append(f"1 of {got['n_runs']} candidate runs, chosen by "
                    f"molecule source, then combo, then splits")
    bits.append(f"combo {got['combo']}"
                + (f" (of {', '.join(got['offered'])})"
                   if len(got["offered"]) > 1 else ""))
    bits.append(f"mol {r.get('mol_source') or '?'}")
    bits.append("head TUNED -- not comparable to the sweep's fixed head"
                if r["tuned"] else "head fixed")
    if not shared:
        bits.append(f"splits {theirs[:6]} share NOTHING with the sweep's {ours[:6]}")
    elif set(theirs) != set(ours):
        bits.append(f"rests on {theirs} while the sweep rows rest on {ours} -- "
                    f"only {shared} split(s) in common")
    out = [f"{label} @ {ds}/{regime}: " + "; ".join(bits)]
    if got["mol_mismatch"]:
        theirs = got["run"].get("mol_source") or "an unreadable source"
        out.append(
            f"!! {label} @ {ds}/{regime} was fed {theirs} molecules while this table is "
            f"on {got['wanted_mol']} -- no run on {got['wanted_mol']} was found for "
            f"this cell, so the row comes from a different column of the study. "
            f"Run `--audit` to see what else is there.")
    if got["missed"]:
        out.append(
            f"!! {label} @ {ds}/{regime}: the combo you asked for is NOT in any matching "
            f"run -- this row fell back to {got['combo']}. Offered here: "
            f"{', '.join(got['offered']) or 'nothing'}. Run `--audit` to see every "
            f"candidate.")
    if got["carries_prot"]:
        out.append(
            f"   {label} @ {ds}/{regime} is on {got['combo']}: the head gets the RAW "
            f"PROTEIN VECTOR as well as this baseline's own feature. Since `boost` in "
            f"this table IS prot+mol, that row answers \"does {label} ADD to boost\", "
            f"while the graph rows (cls+mol) answer \"does the graph REPLACE ESM\". "
            f"Both are legitimate; they are not the same question, and the mean place "
            f"ranks them as if they were."
            + ("" if got["requested"] else
               f" Nothing asked for this combo -- it was the only one on offer."))
    return out


def audit(args):
    """Every candidate run for every baseline, and what each combo in it scores.

    The tables print ONE row per baseline, chosen by rules (skip `_timing`, most splits
    wins, match the construction). Those rules are the difference between a plausible
    number and the right one, so this prints what they chose FROM -- including the runs
    they deliberately refused, marked as such. Read it whenever a baseline row moves and
    you want to know which rule moved it."""
    root = pathlib.Path(args.ensemble_root)
    print(f"\n=== CANDIDATE RUNS under {root}")
    print("    (skipped rows are what the selection rules refuse; the chosen row is "
          "marked ->)")
    for name in (args.baseline or []):
        label = BASELINE_LABEL.get(name, name)
        # deliberately NOT find_runs: this listing must show what find_runs drops
        allruns = []
        for cfg_path in sorted(root.glob("*/*/config.json")):
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:                     # noqa: BLE001
                continue
            if name.lower() not in " ".join(str(x) for x in
                                            (cfg.get("sources") or [])).lower():
                continue
            m = cfg_path.parent / "metrics.csv"
            if not m.exists():
                continue
            allruns.append((cfg_path.parent, cfg, m))
        if not allruns:
            print(f"\n  {label}: no run mentions it anywhere under {root}")
            continue
        chosen = {}
        for ds in DATASET_ORDER:
            for regime in REGIME_ORDER:
                got = baseline_row(find_runs(root, name.lower()), ds, regime,
                                   [], ag.TASK[ds], combo_for(name, args),
                                   args.mol_source)
                if got:
                    chosen[(str(got["run"]["run"]), got["combo"])] = f"{ds}/{regime}"
        print(f"\n  {label}")
        for run, cfg, m in allruns:
            pool = run.parent.name
            scope = _scope_of(cfg)
            d = pd.read_csv(m)
            d = d[d["kind"] == "combo"] if "kind" in d.columns else d
            why = []
            if pool.startswith(SKIP_POOL_PREFIX):
                why.append("SKIPPED: bookkeeping pool, not results")
            if scope is None:
                why.append("SKIPPED: not one of the six table cells "
                           f"(regime={cfg.get('regime')}, "
                           f"{cfg.get('split_family') or cfg.get('full_full_mode') or cfg.get('split')})")
            head = ("TUNED" if cfg.get("tune_boost") else "fixed")
            print(f"    {pool}/{run.name}"
                  + (f"   [{scope[0]}/{scope[1]}]" if scope else "")
                  + f"   mol {mol_source_of(cfg) or '?'}   head {head}"
                  + ("   " + "; ".join(why) if why else ""))
            if d.empty:
                print("        (no combo rows)")
                continue
            task = ag.TASK[scope[0]] if scope else None
            key = ag.OF_RECORD[task] if task and ag.OF_RECORD[task] in d.columns else None
            for cname, g in d.groupby("name", sort=True):
                mark = "  ->" if (str(run), cname) in chosen else "    "
                val = (f"{ag.OF_RECORD[task]} {g[key].mean():.3f}" if key else "")
                flag = "   <-- carries the RAW PROTEIN VECTOR" if "prot" in cname else ""
                print(f"    {mark} combo {cname:<16} folds "
                      f"{[int(x) for x in sorted(g['repeat'].unique())]}  {val}{flag}")
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ag.DEFAULT_ROOT)
    ap.add_argument("--ensemble-root", default=ENSEMBLE_ROOT)
    ap.add_argument("--alphas", nargs="+", default=["0.4", "1.0"],
                    help="which dial positions get a row. `all` takes every alpha "
                         "present in the files, resolved per (dataset, regime), which "
                         "is the way to see the whole sweep rather than the two or "
                         "three positions you already suspect")
    ap.add_argument("--mol-source", default="chemberta",
                    help="ONE source per table -- the tables are per dataset, not per "
                         "source. Pass another to reprint them on it")
    ap.add_argument("--nodes", nargs="+", default=["onehot"],
                    choices=["esm", "onehot", "nodedial"],
                    help="which series to table. `onehot` is the v8 gate, `nodedial` the "
                         "v9 node dial. Never both: alpha runs structure -> function on "
                         "the first and function -> structure on the second")
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
    ap.add_argument("--combo", nargs="+", default=None,
                    help="which head to read from a baseline run. Either one name for "
                         "every baseline (`--combo cls+prot+mol`) or per baseline "
                         "(`--combo hladis=cls+prot+mol prosmith=cls+mol`). Default: "
                         f"{PAPER_COMBO} for those, else the combo matching the graph "
                         "rows' construction (cls+mol). A run offering the requested "
                         "combo is preferred over a fuller run that does not.")
    ap.add_argument("--delta", action="store_true",
                    help="print the paired difference against boost instead of the mean "
                         "place. Available only where the rows share splits; a borrowed "
                         "row on other folds shows `unpaired`")
    ap.add_argument("--audit", action="store_true",
                    help="list every candidate ensembler run and every combo in it, "
                         "with what the selection rules chose and what they refused. "
                         "Use it when a baseline row moves and you want to know why")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--graph-combo", default="cls+mol",
                    choices=["cls+mol", "cls+prot+mol"],
                    help="which boosting head the GNN rows are read from. cls+mol = "
                         "[z_prot || molecule] (the graph replacing ESM, what the tables "
                         "have always shown); cls+prot+mol = [z_prot || raw ESM || "
                         "molecule], the construction the borrowed baseline rows use")
    a = ap.parse_args()
    a.alphas = (None if any(str(x).lower() == "all" for x in a.alphas)
                else [float(x) for x in a.alphas])

    if a.audit:
        audit(a)
        return
    df = ag.load(root=a.root, mol_source=a.mol_source, nodes=a.nodes,
                 dataset=a.dataset, regime=a.regime, seed=a.seed, combo=a.graph_combo)
    tables = build(df, a)
    if not tables:
        raise SystemExit("nothing to print")
    print(f"\nmolecule source: {a.mol_source}   nodes: {','.join(a.nodes)}   "
          f"GNN head: {a.graph_combo}")
    print("cells: mean +/- 95% t-CI over the splits.  "
          + ("last column: paired difference vs boost [won/n]" if a.delta else
             "mean place: rank among the rows WITHIN each split, averaged (1 = best)"))
    col = "d vs boost" if a.delta else "mean place"
    dial = "".join(DIAL_NOTE.get(n, "") for n in a.nodes)
    for ds, blocks, metrics, task, notes in tables:
        if a.latex:
            latex(ds, blocks, metrics, task, col, dial)
        else:
            render(ds, blocks, metrics, task, notes, col, dial)


if __name__ == "__main__":
    main()
