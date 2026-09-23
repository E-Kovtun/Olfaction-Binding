"""The shared layer under every article table: where a number comes from, how it is
reduced to one value per held-out split, and how two rows are tested against each other.

Two result trees feed the tables. Nothing here re-derives a selection rule the older
readers already encode:

  sweep          results/graph/v9_seeded -- our graph at a dial position and the boosting
                 base, fitted in the same cells (same split, same model seed), read
                 through `scripts/analysis/alpha_grid.load`.
  ensemble_logs  results/ensemble_logs -- the external baselines (LORAX, ProSmith, MolOR,
                 Hladis), chosen by `scripts/analysis/paper_tables.baseline_row`, which
                 already knows the four ways a borrowed row goes wrong (`_timing` pools,
                 combo names, raw-ESM combos, the molecule source hidden in the file).

THE UNIT IS THE SPLIT. Sweep rows carry several model seeds per split; they are averaged
inside the split first, so every mean, std, rank and test below is over the (usually five)
held-out splits. External baselines carry one seed per split. See `alpha_grid.fold_means`
for why the (fold, seed) cell is not an observation.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis import alpha_grid as ag      # noqa: E402
from scripts.analysis import paper_tables as pt    # noqa: E402

#: NO DEFAULT, deliberately (23.09.2026). This used to name a specific run, and a
#: reader invoked without `--sweep-root` then rendered a table from whichever series
#: that happened to be -- a different protein source, a different encoder, silently.
#: A missing root is now an error with a sentence, which is the only safe behaviour
#: for a value that decides which model the table is about.
SWEEP_ROOT = None
ENSEMBLE_ROOT = pt.ENSEMBLE_ROOT
OUT_ROOT = "results/article_tables"

DATASETS = list(pt.DATASET_ORDER)
REGIMES = list(pt.REGIME_ORDER)
MOL_SOURCES = ["chemberta", "gin", "ecfp"]
TASK = ag.TASK
OF_RECORD = ag.OF_RECORD
LOWER_IS_BETTER = ag.LOWER_IS_BETTER
SHOW = pt.SHOW

# The two heads the sweep fits on one trained graph.
GRAPH_COMBOS = ("cls+mol", "cls+prot+mol")
BASELINES = ("lorax", "prosmith", "molor", "hladis")
# tab:t1full puts every competitor at its full feature set; the tables keep that footing.
BASELINE_COMBO = "cls+prot+mol"
ALPHA = 1.0
EXPECTED_SPLITS = 5

DATASET_LABEL = {"m2or": "M2OR", "cc": "Mosquito (Carey)", "hc": "Fly (Hallem-Carlson)",
                 "cc_shrinked": "Mosquito (Carey, shrunk)",
                 "hc_shrinked": "Fly (Hallem-Carlson, shrunk)",
                 "cc_shrinked50": "Mosquito (Carey, shrunk 50%)",
                 "hc_shrinked50": "Fly (Hallem-Carlson, shrunk 50%)"}
DATASET_TEX = {"m2or": "M2OR", "cc": "Mosquito (Carey)", "hc": r"Fly (Hallem--Carlson)",
               "cc_shrinked": "Mosquito (Carey, shrunk)",
               "hc_shrinked": r"Fly (Hallem--Carlson, shrunk)",
               "cc_shrinked50": r"Mosquito (Carey, shrunk 50\%)",
               "hc_shrinked50": r"Fly (Hallem--Carlson, shrunk 50\%)"}
REGIME_LABEL = {"transductive": "Transductive", "inductive": "Cold molecule"}
MOL_LABEL = {"chemberta": "ChemBERTa", "gin": "GIN", "ecfp": "ECFP"}
BASELINE_LABEL = {"lorax": "LORAX", "prosmith": "ProSmith", "molor": "MolOR",
                  "hladis": "Hladis"}
BASELINE_TEX = BASELINE_LABEL | {"hladis": r"Hladi\v{s}"}
METRIC_TEX = {"R2": "$R^2$"}
BOOST_LABEL = "Boosting base (prot+mol)"


def resolve(p):
    """A repo-relative path works from any working directory."""
    p = pathlib.Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def out_dir(p):
    p = pathlib.Path(p)
    p = p if p.is_absolute() else ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------ rows

@dataclass
class Row:
    """One method in one (dataset, regime, molecule) cell: its value per split."""
    key: str                      # stable id: "ours:cls+mol", "boost", "hladis"
    label: str
    tex: str
    kind: str                     # ours | boost | baseline
    combo: str
    source: str                   # sweep | ensemble_logs
    folds: pd.DataFrame | None = None   # index = split id, columns = metrics
    seeds: float = 1.0
    origin: str = ""
    flags: list = field(default_factory=list)
    usable: bool = True
    # where this row's PER-ROW scores live, if they were written at all. Filled by the
    # builders below and consumed only by the threshold machinery.
    locator: dict | None = None

    @property
    def present(self):
        return self.folds is not None and not self.folds.empty

    def values(self, metric):
        if not self.present or metric not in self.folds.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(self.folds[metric], errors="coerce").dropna()

    def splits(self):
        return sorted(self.folds.index) if self.present else []


_SWEEP = {}
_SWEEP_WHY = {}
_RUNS = {}


def clear_cache():
    _SWEEP.clear()
    _SWEEP_WHY.clear()
    _RUNS.clear()


def require_root(root):
    """A sweep root the caller actually chose. See SWEEP_ROOT for why there is no
    default to fall back on."""
    if not root:
        raise SystemExit(
            "--sweep-root is required: it names the run this table is rendered from, "
            "and there is no default because two runs of this project are two "
            "different models. Pass the directory the sweep wrote, e.g.\n"
            "    --sweep-root results/graph/main")
    return root


def sweep_frame(root=SWEEP_ROOT, combo="cls+mol"):
    """Every nodedial row under `root` holding ONE graph head, or an empty frame."""
    root = require_root(root)
    key = (str(root), combo)
    if key not in _SWEEP:
        try:
            _SWEEP[key] = ag.load(root=str(resolve(root)), nodes="nodedial", combo=combo)
            _SWEEP_WHY[key] = ""
        except SystemExit as exc:
            # `ag.load` exits when its filters leave nothing -- which includes the case
            # where every file under the root was DROPPED as unparseable, a very
            # different thing from "not run yet". Discarding that message is how a
            # finished sweep read as an empty directory for a whole debugging round:
            # the table printed `--` for every graph row and volunteered no reason.
            _SWEEP[key] = pd.DataFrame()
            _SWEEP_WHY[key] = str(exc).strip() or "alpha_grid.load returned nothing"
    return _SWEEP[key]


def sweep_reason(root=SWEEP_ROOT, combo="cls+mol"):
    """Why `sweep_frame` came back empty, verbatim from the reader that gave up."""
    return _SWEEP_WHY.get((str(root), combo), "")


def _per_split(q, metrics):
    cols = [m for m in metrics if m in q.columns]
    q = q.assign(**{m: pd.to_numeric(q[m], errors="coerce") for m in cols})
    g = q.groupby("fold", sort=True)
    folds = g[cols].mean()
    folds.index = folds.index.astype(int)
    seeds = float(g["seed"].nunique().mean()) if "seed" in q.columns else 1.0
    return folds, seeds


def _cell(df, ds, regime, mol, seeds):
    if df.empty:
        return df
    g = df[(df.dataset == ds) & (df.regime == regime) & (df.mol_source == mol)]
    return g[g.seed.isin(seeds)] if seeds else g


def ours_row(ds, regime, mol, metrics, combo="cls+mol", alpha=ALPHA, seeds=None,
             root=SWEEP_ROOT):
    label = f"Our graph ({combo})"
    row = Row(f"ours:{combo}", label, label, "ours", combo, "sweep")
    frame = sweep_frame(root, combo)
    g = _cell(frame, ds, regime, mol, seeds)
    if g.empty:
        row.flags.append(sweep_reason(root, combo) if frame.empty
                         else "no rows for this dataset/regime/molecule source")
        return row
    q = g[(g.arm == "gate")
          & np.isclose(pd.to_numeric(g.alpha, errors="coerce").astype(float), alpha)]
    if q.empty:
        row.flags.append(f"no alpha={alpha:g} rows on head {combo}")
        return row
    row.folds, row.seeds = _per_split(q, metrics)
    row.origin = f"{root} alpha={alpha:g} head {combo}"
    row.locator = dict(kind="sweep", root=root, ds=ds, regime=regime, mol=mol,
                       arm="gate", alpha=alpha, combo=combo)
    return row


def boost_row(ds, regime, mol, metrics, seeds=None, root=SWEEP_ROOT):
    row = Row("boost", BOOST_LABEL, BOOST_LABEL, "boost", "prot+mol", "sweep")
    for combo in GRAPH_COMBOS:        # the reference arm is kept in either head's frame
        g = _cell(sweep_frame(root, combo), ds, regime, mol, seeds)
        q = g[g.arm == "boost_full"] if not g.empty else g
        if not q.empty:
            row.folds, row.seeds = _per_split(q, metrics)
            row.origin = f"{root} boost_full"
            row.locator = dict(kind="sweep", root=root, ds=ds, regime=regime, mol=mol,
                               arm="boost_full", alpha=None, combo="cls+mol")
            return row
    row.flags.append(sweep_reason(root, GRAPH_COMBOS[0]) or "no boost_full rows")
    return row


def baseline_row(name, ds, regime, mol, metrics, combo=BASELINE_COMBO,
                 root=ENSEMBLE_ROOT, allow_mol_mismatch=False):
    key = (str(root), name)
    if key not in _RUNS:
        _RUNS[key] = pt.find_runs(resolve(root), name.lower())
    lab = BASELINE_LABEL.get(name, name)
    row = Row(name, f"{lab} ({combo})", f"{BASELINE_TEX.get(name, lab)} ({combo})",
              "baseline", combo, "ensemble_logs")
    got = pt.baseline_row(_RUNS[key], ds, regime, metrics, TASK[ds], combo, mol)
    if got is None:
        row.flags.append("no run")
        return row
    f = got["frame"]
    cols = [m for m in metrics if m in f.columns]
    folds = f[cols].apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
    folds.index = folds.index.astype(int)
    row.folds, row.combo = folds, got["combo"]
    row.label = f"{lab} ({row.combo})"
    row.tex = f"{BASELINE_TEX.get(name, lab)} ({row.combo})"
    r = got["run"]
    row.origin = f"{r['pool']}/{r['run'].name}"
    row.locator = dict(kind="ensemble", run=r["run"], combo=row.combo)
    if got["missed"]:
        row.flags.append(f"asked for {combo}, the run offers {', '.join(got['offered'])}")
    if got["mol_mismatch"]:
        row.flags.append(f"molecules are {r.get('mol_source') or '?'}, not {mol}")
        row.usable = bool(allow_mol_mismatch)
    if r["tuned"]:
        row.flags.append("TUNED head -- the sweep's heads are fixed")
    return row


def flag_split_mismatch(rows):
    """A borrowed row on other splits than the sweep's cannot be paired or ranked with
    it; say so on the row rather than let the intersection silently shrink."""
    ref = next((r.splits() for r in rows if r.source == "sweep" and r.present), None)
    if not ref:
        return
    for r in rows:
        if r.present and r.source != "sweep" and r.splits() != ref:
            r.flags.append(f"splits {r.splits()} vs the sweep's {ref}")


def cell_rows(ds, regime, mol, metrics, ours=GRAPH_COMBOS, baselines=BASELINES,
              baseline_combo=BASELINE_COMBO, alpha=ALPHA, seeds=None,
              sweep_root=SWEEP_ROOT, ensemble_root=ENSEMBLE_ROOT,
              allow_mol_mismatch=False):
    """Baselines, then the boosting base, then our graph's heads -- the table's order.

    `baseline_combo` is one combo for every baseline or a {name: combo} dict."""
    def combo(b):
        if isinstance(baseline_combo, dict):
            return baseline_combo.get(b, BASELINE_COMBO)
        return baseline_combo

    rows = [baseline_row(b, ds, regime, mol, metrics, combo(b), ensemble_root,
                         allow_mol_mismatch) for b in baselines]
    rows.append(boost_row(ds, regime, mol, metrics, seeds, sweep_root))
    rows += [ours_row(ds, regime, mol, metrics, c, alpha, seeds, sweep_root) for c in ours]
    flag_split_mismatch(rows)
    return rows


# ------------------------------------------------------------------ the decision cut

# The metrics that depend on WHERE the decision boundary falls. AUROC and AUPRC do not
# and are never touched here.
THRESHOLDED = ("MCC", "F1")
# What a column scored at a validation-chosen cut is called, next to the 0.5 one.
VAL_SUFFIX = "@val"
# How the sweep's dump names each head's scores (`run_alpha_gate_sweep.PRED_KEY` and
# `VAL_PRED_KEY`). Spelled out rather than imported: importing the sweep pulls in
# torch_geometric, and this module is read by notebooks and a laptop.
DUMP_TEST_KEY = {"cls+mol": "pred", "cls+prot+mol": "pred_prot"}
DUMP_VAL_KEY = {"cls+mol": "pred_va", "cls+prot+mol": "pred_va_prot"}


def _ht():
    """headline_table, for its filename parser -- no heavy dependencies."""
    global _HT
    try:
        return _HT
    except NameError:
        _HT = ag._parser()
        return _HT


def _dump_dir(root, ds, regime, mol):
    """The sweep's dump folder for one cell, found by parsing folder names the same way
    the readers parse the CSV names."""
    d = resolve(root) / "dumps"
    if not d.exists():
        return None
    ht = _ht()
    for p in sorted(d.iterdir()):
        if not p.is_dir():
            continue
        ds_, fam, nodes, mol_, var = ht.parse_name(p.name)
        if (ds_ == ds and ht.REGIME_OF.get(fam) == regime and mol_ == mol
                and nodes == "nodedial"
                and (not var or var == ht.CANONICAL_VARIANT.get(ds_, var))):
            return p
    return None


def _sweep_cells(loc, splits):
    """{(split, seed): npz path} for a sweep row, by globbing the cells on disk."""
    d = _dump_dir(loc["root"], loc["ds"], loc["regime"], loc["mol"])
    if d is None:
        return {}
    a = "None" if loc["alpha"] is None else f"{float(loc['alpha']):g}"
    out = {}
    for f in sorted(d.glob(f"{loc['arm']}_a{a}_f*_s*.npz")):
        stem = f.stem.rsplit("_f", 1)[1]
        fold, _, seed = stem.partition("_s")
        try:
            fold, seed = int(fold), int(seed)
        except ValueError:
            continue
        if fold in splits:
            out[(fold, seed)] = f
    return out


def _read_sweep_cell(path, combo):
    vk, tkey = DUMP_VAL_KEY.get(combo), DUMP_TEST_KEY.get(combo)
    if vk is None:
        return None
    with np.load(path, allow_pickle=False) as z:
        if not {vk, tkey, "y_val", "y_true"} <= set(z.files):
            return None
        return dict(val=z[vk], y_val=z["y_val"], test=z[tkey], y_test=z["y_true"])


def _read_ensemble_cell(run, combo, repeat):
    f = pathlib.Path(run) / "scores" / f"repeat_{repeat}.npz"
    if not f.exists():
        return None
    with np.load(f, allow_pickle=False) as z:
        vk, tkey = f"val__{combo}", f"test__{combo}"
        if not {vk, tkey, "y_val", "y_test"} <= set(z.files):
            return None
        return dict(val=z[vk], y_val=z["y_val"], test=z[tkey], y_test=z["y_test"])


def row_scores(row):
    """{(split, seed): {val, y_val, test, y_test}} for one row, or None if this row's
    per-row scores were not written. A run made before the producers saved them returns
    None, which is what makes the fallback below all-or-nothing rather than silent."""
    loc, splits = row.locator, set(row.splits())
    if not loc or not splits:
        return None
    out = {}
    if loc["kind"] == "sweep":
        for (fold, seed), f in _sweep_cells(loc, splits).items():
            got = _read_sweep_cell(f, loc["combo"])
            if got is not None:
                out[(fold, seed)] = got
    else:
        for fold in splits:
            got = _read_ensemble_cell(loc["run"], loc["combo"], fold)
            if got is not None:
                out[(fold, 0)] = got
    return out or None


def _hard_metric(metric, y, p, t):
    from sklearn.metrics import f1_score, matthews_corrcoef
    yb = (np.asarray(y) > 0.5).astype(int)
    hard = (np.asarray(p) >= t).astype(int)
    if metric == "MCC":
        return float(matthews_corrcoef(yb, hard))
    return float(f1_score(yb, hard, zero_division=0))


def _metric_curve(metric, y, p, cands):
    """`_hard_metric` at every cut in `cands` at once.

    One sort, then the confusion counts at each cut by `searchsorted` -- the same
    numbers sklearn would build per call, without its per-call input validation. That
    validation is what made the per-cut loop cost minutes on M2OR: ~200 cuts x 25
    (fold, seed) cells x every row x both metrics is tens of thousands of calls.
    Degenerate cuts score 0, as sklearn scores them (MCC with a zero denominator,
    F1 with `zero_division=0`)."""
    yb = np.asarray(y) > 0.5
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    ps, pos_sorted = p[order], yb[order].astype(float)
    cum_pos = np.concatenate([[0.0], np.cumsum(pos_sorted)])
    n, n_pos = float(len(p)), float(yb.sum())
    idx = np.searchsorted(ps, np.asarray(cands, float), side="left")   # first p >= t
    tp = n_pos - cum_pos[idx]
    fp = (n - idx) - tp
    fn = n_pos - tp
    tn = (n - n_pos) - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "MCC":
            den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            out = (tp * tn - fp * fn) / den
        else:
            den = 2 * tp + fp + fn
            out = 2 * tp / den
    return np.where(den > 0, out, 0.0)


def _best_threshold(metric, y, p, grid=200):
    """The cut that maximises `metric` on the VALIDATION rows.

    Searched over quantiles of the scores rather than every midpoint: the winner is
    indistinguishable and the cost stops depending on how many rows the split has.
    Ties go to the LOWEST cut, as `max` over the ascending candidates always did."""
    p = np.asarray(p, float)
    if p.size == 0:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.005, 0.995, grid)))
    cands = np.unique(np.concatenate([[0.5], qs]))
    return float(cands[int(np.argmax(_metric_curve(metric, y, p, cands)))])


def add_val_threshold_metrics(rows, metrics, task):
    """Add `MCC@val` / `F1@val` BESIDE MCC and F1 -- the same metric at a cut chosen on
    the validation rows of that fold, for that method, instead of the fixed 0.5.

    An addition, never a replacement, and that is the whole design. Every method we
    compare against reports its thresholded metrics at 0.5 -- LORAX (`train_lorax.py`,
    `train_GB.py`), ProSmith (`training_GB.py`, via `np.round`), Hladis
    (`make_compute_metrics.py`, likewise `round`; his 200-threshold sweep feeds only the
    PR/ROC curves) -- so the 0.5 column is what keeps this table comparable with their
    published numbers. The val-chosen column answers the other question, which is who
    wins once the operating point is picked honestly. Printing both is the only way to
    say both things without one quietly standing in for the other.

    ALL OR NOTHING: unless EVERY usable row can be re-scored, no row gets the extra
    columns. A column mixing the two cuts would rank models by their operating points.

    Returns (names added, mode), mode being "val", "fixed" or "n/a".
    """
    base = [m for m in metrics if m in THRESHOLDED]
    if task != "classification" or not base:
        return [], "n/a"
    live = [r for r in rows if r.present and r.usable]
    if not live:
        return [], "fixed"
    scores = {}
    for r in live:
        got = row_scores(r)
        if got is None:
            return [], "fixed"
        scores[r.key] = got
    for r in live:
        per = scores[r.key]
        for m in base:
            col = f"{m}{VAL_SUFFIX}"
            vals = {}
            for (fold, seed), s in per.items():
                t = _best_threshold(m, s["y_val"], s["val"])
                vals.setdefault(fold, []).append(_hard_metric(m, s["y_test"], s["test"], t))
            if col not in r.folds.columns:
                r.folds[col] = np.nan
            for fold, v in vals.items():
                if fold in r.folds.index:
                    r.folds.loc[fold, col] = float(np.mean(v))
    return [f"{m}{VAL_SUFFIX}" for m in base], "val"


def with_val_columns(metrics, extra):
    """The column order: each val-cut metric immediately after the 0.5 one it mirrors."""
    out = []
    for m in metrics:
        out.append(m)
        if f"{m}{VAL_SUFFIX}" in extra:
            out.append(f"{m}{VAL_SUFFIX}")
    return out


# ----------------------------------------------------------------------------- stats

def matrix(rows, metric):
    """split x method, on the splits every usable row with this metric shares."""
    s = {r.key: r.values(metric) for r in rows if r.usable and len(r.values(metric))}
    if not s:
        return pd.DataFrame()
    common = sorted(set.intersection(*(set(v.index) for v in s.values())))
    return pd.DataFrame({k: v.reindex(common) for k, v in s.items()}, index=common)


def split_ranks(M, metric):
    """Place within each split, 1 = best; RMSE/MAE ranked the other way round."""
    if M.empty or M.shape[1] < 2:
        return pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    return M.rank(axis=1, ascending=metric in LOWER_IS_BETTER, method="average")


def paired_test(x, y, metric):
    """x against y on their shared splits: mean difference, how often each leads, and a
    two-sided paired t-test. Not Wilcoxon: at five splits its smallest attainable
    two-sided p is 0.0625, so it could never reject anything."""
    common = x.index.intersection(y.index)
    out = dict(n_pair=len(common), delta=np.nan, p=np.nan, ahead=0, behind=0)
    if not len(common):
        return out
    a = x.loc[common].astype(float)
    b = y.loc[common].astype(float)
    d = a - b
    lead = -d if metric in LOWER_IS_BETTER else d
    out.update(delta=float(d.mean()), ahead=int((lead > 0).sum()),
               behind=int((lead < 0).sum()))
    if len(common) >= 3 and float(d.std(ddof=1)) > 0:
        from scipy.stats import ttest_rel
        out["p"] = float(ttest_rel(a, b).pvalue)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values; NaN stays NaN and does not count."""
    p = pd.Series(pvals, dtype=float)
    ok = p.dropna().sort_values()
    m, run, adj = len(ok), 0.0, {}
    for i, (k, v) in enumerate(ok.items()):
        run = max(run, min(1.0, (m - i) * v))
        adj[k] = run
    return pd.Series({k: adj.get(k, np.nan) for k in p.index}, dtype=float)


def friedman_p(M):
    M = M.dropna()
    if M.shape[1] < 3 or M.shape[0] < 2:
        return np.nan
    from scipy.stats import friedmanchisquare
    try:
        return float(friedmanchisquare(*[M[c] for c in M.columns]).pvalue)
    except ValueError:
        return np.nan


def best_other(rows, metric, exclude=("ours",)):
    """The key of the strongest row in this column that is NOT ours -- the opponent our
    models are tested against. Per metric, because the leader changes between them."""
    cand = [(r.key, float(r.values(metric).mean())) for r in rows
            if r.usable and r.kind not in exclude and len(r.values(metric))]
    if not cand:
        return None
    pick = min if metric in LOWER_IS_BETTER else max
    return pick(cand, key=lambda kv: kv[1])[0]


def block_stats(rows, metrics, ref_key=None, ref_mode="fixed", test_kinds=("ours",)):
    """One record per (method, metric): mean/std over splits, place, and a paired test.

    `ref_mode="fixed"` tests every row against `ref_key` -- the older tables' shape.
    `ref_mode="best_other"` is what the main table does: the reference is chosen PER
    METRIC as the best row that is not ours (`best_other`), and only rows whose kind is
    in `test_kinds` are tested at all. A baseline is then never tested against another
    baseline, which is not a claim this paper makes and would only spend the correction.

    The test is Holm-corrected across whatever was tested in that metric -- one family
    per table column, so with two of our rows the correction is over two p-values."""
    recs = []
    for m in metrics:
        M = matrix(rows, m)
        R = split_ranks(M, m)
        if ref_mode == "best_other":
            rk = best_other(rows, m, exclude=tuple(test_kinds))
            targets = [r for r in rows if r.kind in test_kinds]
        else:
            rk = ref_key
            targets = [r for r in rows if r.key != rk]
        ref = next((r for r in rows if r.key == rk), None)
        tests = {}
        if ref is not None and len(ref.values(m)):
            for r in targets:
                if r.key != rk and r.usable and len(r.values(m)):
                    tests[r.key] = paired_test(r.values(m), ref.values(m), m)
        adj = holm({k: t["p"] for k, t in tests.items()})
        fp = friedman_p(M)
        for r in rows:
            v = r.values(m)
            t = tests.get(r.key, {})
            recs.append(dict(
                key=r.key, method=r.label, kind=r.kind, combo=r.combo, source=r.source,
                usable=r.usable, seeds=r.seeds, metric=m, n=len(v),
                mean=float(v.mean()) if len(v) else np.nan,
                std=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                rank=(float(R[r.key].mean()) if r.key in R.columns and len(R)
                      else np.nan),
                n_ranked=len(R), k_ranked=R.shape[1], friedman_p=fp, ref=rk or "",
                delta_vs_ref=t.get("delta", np.nan), p_vs_ref=t.get("p", np.nan),
                p_holm=float(adj.get(r.key, np.nan)) if r.key in adj else np.nan,
                ahead_of_ref=t.get("ahead", np.nan), behind_ref=t.get("behind", np.nan),
                n_pair=t.get("n_pair", np.nan),
                origin=r.origin, flags="; ".join(r.flags)))
    return pd.DataFrame(recs)


def top_two(st_m, metric):
    """Keys of the best and second-best usable row in one column."""
    s = st_m[st_m.usable.astype(bool) & np.isfinite(st_m["mean"].astype(float))]
    if s.empty:
        return None, None
    keys = list(s.sort_values("mean", ascending=metric in LOWER_IS_BETTER).key)
    return keys[0], (keys[1] if len(keys) > 1 else None)


# ------------------------------------------------------------------------ formatting

def fnum(v, nd=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def tex_num(mean, std, nd=3):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{nd}f}" + (rf"$\pm${std:.{nd}f}" if np.isfinite(std) else "")


def txt_num(mean, std, nd=3):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{nd}f}" + (f"+/-{std:.{nd}f}" if np.isfinite(std) else "")


def metric_tex(m):
    if m.endswith(VAL_SUFFIX):
        base = m[:-len(VAL_SUFFIX)]
        return METRIC_TEX.get(base, base) + r"$^{\mathrm{val}}$"
    return METRIC_TEX.get(m, m)


def pstr(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def tex_p(p):
    if p is None or not np.isfinite(p):
        return "--"
    return r"$<$0.001" if p < 1e-3 else f"{p:.3f}"


def tex_p_pair(raw, adj):
    """The two p-values of one cell, raw/Holm, as the table prints them."""
    if (raw is None or not np.isfinite(raw)) and (adj is None or not np.isfinite(adj)):
        return ""
    return f"{tex_p(raw)}/{tex_p(adj)}"


def text_block(st, metrics, title, sig=0.05, w=22, wp=14):
    """The block as a console table.

    Two columns per metric -- the value with its place, and (for the rows that were
    tested) the pair of p-values against that column's own reference, raw/Holm. They are
    separate columns on purpose: glued into one, a long cell runs into its neighbour and
    the table stops being readable exactly where the numbers matter."""
    keys = list(dict.fromkeys(st.key))
    tested = bool(st["p_vs_ref"].notna().any() or st["p_holm"].notna().any())
    wm = max([len(str(x)) for x in st.method] + [16]) + 2
    head = f"  {'method':<{wm}}"
    for m in metrics:
        head += f"{m:>{w}}" + (f"{'p t/Holm':>{wp}}" if tested else "")
    head += f"{'rank':>7}"
    lines = [title, head, "  " + "-" * (len(head) - 2)]
    for k in keys:
        s = st[st.key == k].set_index("metric")
        cells = ""
        for m in metrics:
            if m not in s.index or not np.isfinite(s.loc[m, "mean"]):
                cells += f"{'--':>{w}}" + (f"{'':>{wp}}" if tested else "")
                continue
            r = s.loc[m]
            txt = txt_num(r["mean"], r["std"])
            if np.isfinite(r["rank"]):
                txt += f" ({r['rank']:.2f})"
            cells += f"{txt:>{w}}"
            if not tested:
                continue
            if np.isfinite(r["p_vs_ref"]) or np.isfinite(r["p_holm"]):
                p = f"{pstr(r['p_vs_ref'])}/{pstr(r['p_holm'])}"
                if np.isfinite(r["p_holm"]) and r["p_holm"] < sig:
                    p += "*"
            else:
                p = "ref" if k == r["ref"] else ""
            cells += f"{p:>{wp}}"
        mr = s["rank"].mean()
        use = "" if bool(s["usable"].iloc[0]) else "  [not usable]"
        lines.append(f"  {s['method'].iloc[0]:<{wm}}{cells}{fnum(mr, 2):>7}{use}")
    return "\n".join(lines)
