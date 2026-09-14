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

SWEEP_ROOT = "results/graph/v9_seeded"
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

DATASET_LABEL = {"m2or": "M2OR", "cc": "Mosquito (Carey)", "hc": "Fly (Hallem-Carlson)"}
DATASET_TEX = {"m2or": "M2OR", "cc": "Mosquito (Carey)", "hc": r"Fly (Hallem--Carlson)"}
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
_RUNS = {}


def clear_cache():
    _SWEEP.clear()
    _RUNS.clear()


def sweep_frame(root=SWEEP_ROOT, combo="cls+mol"):
    """Every nodedial row under `root` holding ONE graph head, or an empty frame."""
    key = (str(root), combo)
    if key not in _SWEEP:
        try:
            _SWEEP[key] = ag.load(root=str(resolve(root)), nodes="nodedial", combo=combo)
        except SystemExit:
            _SWEEP[key] = pd.DataFrame()
    return _SWEEP[key]


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
    g = _cell(sweep_frame(root, combo), ds, regime, mol, seeds)
    if g.empty:
        row.flags.append("no sweep file for this cell")
        return row
    q = g[(g.arm == "gate")
          & np.isclose(pd.to_numeric(g.alpha, errors="coerce").astype(float), alpha)]
    if q.empty:
        row.flags.append(f"no alpha={alpha:g} rows on head {combo}")
        return row
    row.folds, row.seeds = _per_split(q, metrics)
    row.origin = f"{root} alpha={alpha:g} head {combo}"
    return row


def boost_row(ds, regime, mol, metrics, seeds=None, root=SWEEP_ROOT):
    row = Row("boost", BOOST_LABEL, BOOST_LABEL, "boost", "prot+mol", "sweep")
    for combo in GRAPH_COMBOS:        # the reference arm is kept in either head's frame
        g = _cell(sweep_frame(root, combo), ds, regime, mol, seeds)
        q = g[g.arm == "boost_full"] if not g.empty else g
        if not q.empty:
            row.folds, row.seeds = _per_split(q, metrics)
            row.origin = f"{root} boost_full"
            return row
    row.flags.append("no boost_full rows")
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


def text_block(st, metrics, title, sig=0.05, w=34):
    """The block as a console table: value +/- std (place), then the tested rows' two
    p-values against that column's reference, raw/Holm."""
    keys = list(dict.fromkeys(st.key))
    head = f"  {'method':<28}" + "".join(f"{m:>{w}}" for m in metrics) + f"{'rank':>7}"
    lines = [title, head, "  " + "-" * (len(head) - 2)]
    for k in keys:
        s = st[st.key == k].set_index("metric")
        cells = ""
        for m in metrics:
            if m not in s.index or not np.isfinite(s.loc[m, "mean"]):
                cells += f"{'--':>{w}}"
                continue
            r = s.loc[m]
            txt = txt_num(r["mean"], r["std"])
            if np.isfinite(r["rank"]):
                txt += f" ({r['rank']:.2f})"
            if np.isfinite(r["p_vs_ref"]) or np.isfinite(r["p_holm"]):
                txt += f" p{pstr(r['p_vs_ref'])}/{pstr(r['p_holm'])}"
                if np.isfinite(r["p_holm"]) and r["p_holm"] < sig:
                    txt += "*"
            elif k == r["ref"]:
                txt += " ref"
            cells += f"{txt:>{w}}"
        mr = s["rank"].mean()
        use = "" if bool(s["usable"].iloc[0]) else "  [not usable]"
        lines.append(f"  {s['method'].iloc[0]:<28}{cells}{fnum(mr, 2):>7}{use}")
    return "\n".join(lines)
