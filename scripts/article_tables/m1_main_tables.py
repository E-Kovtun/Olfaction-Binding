#!/usr/bin/env python
"""Main tables: every method on every (dataset, regime), our graph at one dial position.

    python scripts/article_tables/m1_main_tables.py                    # M1, all datasets
    python scripts/article_tables/m1_main_tables.py --dataset cc hc    # M1, two of them
    python scripts/article_tables/m1_main_tables.py --baseline-combo cls --no-ours  # A1

`--baseline-combo` takes EITHER one combo for every baseline OR `name=combo` items, never
a mixture: a bare item cannot be read as "the default for the rest", because a name absent
from the dict already falls back to `tablekit.BASELINE_COMBO`. To put three baselines on
one combo and one on another, spell all four out.

One table per dataset, the two regimes side by side (transductive | cold molecule):

    LORAX / ProSmith / MolOR / Hladis   external baselines, results/ensemble_logs, at
                                        --baseline-combo (default cls+prot+mol, the
                                        footing of tab:t1full)
    Boosting base (prot+mol)            XGBoost on [raw ESM || molecule], from the sweep
    Our graph (cls+mol)                 [z_prot || molecule]             } one trained
    Our graph (cls+prot+mol)            [z_prot || raw ESM || molecule]  } graph, two heads

A value cell is mean +/- std over the held-out splits; sweep rows average their model
seeds inside each split first.

EVERY METRIC IS FOLLOWED BY A `p` COLUMN, and only OUR two rows carry one. The opponent
is the best NON-OURS row in that same column -- per metric, since the leader changes
between them -- and the two numbers are the two-sided paired t-test over splits, raw and
Holm-corrected within the column (a family of two). Nothing is tested baseline against
baseline: that is not a claim this paper makes, and it would only spend the correction.

The Friedman omnibus (are the methods distinguishable at all in that regime) is one line
under the table, per metric.

Writes to results/article_tables/main/:
    main_long.csv       one row per (dataset, regime, method, metric): every number above
    <ds>.tex            the LaTeX table, caption generated from what was actually read
    <ds>_summary.txt    the facts a paragraph about the table would state, and every flag
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

TEST_KINDS = ("ours",)


def combo_spec(items):
    """`cls+prot+mol` for every baseline, or `name=combo` items for some of them."""
    if len(items) == 1 and "=" not in items[0]:
        return items[0]
    spec = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"--baseline-combo: mix of plain and name=combo in {items}")
        spec[k.lower()] = v
    return spec


def build(ds, a):
    base = list(a.metrics or tk.SHOW[tk.TASK[ds]])
    per, extras, cuts = {}, {}, {}
    for regime in tk.REGIMES:
        rows = tk.cell_rows(ds, regime, a.mol_source, base, ours=a.ours,
                            baselines=a.baselines, baseline_combo=a.baseline_combo,
                            alpha=a.alpha, seeds=a.seeds, sweep_root=a.sweep_root,
                            ensemble_root=a.ensemble_root,
                            allow_mol_mismatch=a.allow_mol_mismatch)
        if not any(r.present for r in rows):
            continue
        per[regime] = rows
        # MCC and F1 at a cut chosen on validation, ADDED beside the 0.5 ones that every
        # published baseline reports -- see `tablekit.add_val_threshold_metrics`.
        extras[regime], cuts[regime] = tk.add_val_threshold_metrics(rows, base,
                                                                   tk.TASK[ds])
    if not per:
        return {}, base
    if not a.val_cut or not all(extras.values()):
        # one block could not be re-scored, so no block gets the extra columns: two
        # regimes of one table must not be read at different operating points
        extras = {r: [] for r in per}
    metrics = tk.with_val_columns(base, sorted({c for v in extras.values() for c in v}))
    blocks = {}
    for regime, rows in per.items():
        st = tk.block_stats(rows, metrics, ref_mode="best_other", test_kinds=TEST_KINDS)
        blocks[regime] = (rows, st.assign(dataset=ds, regime=regime,
                                          mol_source=a.mol_source, alpha=a.alpha,
                                          cut=(cuts[regime] if extras[regime] else
                                               ("fixed" if cuts[regime] != "n/a" else "n/a"))))
    return blocks, metrics


def _mean_ranks(st):
    """Averaged over the 0.5 columns only: a val-cut column is the SAME metric seen
    a second way, and counting both would weight MCC and F1 twice."""
    q = st[st.usable.astype(bool) & ~st.metric.str.endswith(tk.VAL_SUFFIX)]
    return q.groupby("key")["rank"].mean()


def _combo_split(blocks):
    """Keys whose combo differs ACROSS regimes (a fallback in one of them)."""
    seen = {}
    for rows, _ in blocks.values():
        for r in rows:
            if r.present:
                seen.setdefault(r.key, set()).add(r.combo)
    return {k for k, v in seen.items() if len(v) > 1}


def friedman_line(blocks, metrics):
    parts = []
    for reg, (_, st) in blocks.items():
        bits = []
        for m in metrics:
            q = st[st.metric == m]
            if len(q) and np.isfinite(q["friedman_p"].iloc[0]):
                bits.append(f"{tk.metric_tex(m)} {tk.tex_p(q['friedman_p'].iloc[0])}")
        if bits:
            k = int(st.k_ranked.max())
            n = int(st.n_ranked.max())
            parts.append(f"{tk.REGIME_LABEL[reg].lower()} ({k} methods, {n} splits): "
                         + ", ".join(bits))
    return "Friedman -- " + "; ".join(parts) if parts else ""


def caption(ds, blocks, metrics, a):
    sts = pd.concat([b[1] for b in blocks.values()], ignore_index=True)
    use = sts[sts.usable.astype(bool) & (sts.n > 0)]
    n = int(use.n.max()) if len(use) else 0
    sw = use[use.source == "sweep"]
    seeds = f"{sw.seeds.max():g}" if len(sw) else "the"
    splits = ("the LORAX folds and the cold-molecule seeds 42--46" if ds == "m2or"
              else "the upstream random folds and our cold-molecule folds")
    text = (f"{tk.DATASET_TEX[ds]}, molecule embedding {tk.MOL_LABEL[a.mol_source]}, "
            f"our graph at $\\alpha={a.alpha:g}$. Mean $\\pm$ std over {n} held-out splits "
            f"({splits}); rows from our sweep are first averaged over {seeds} model seeds "
            f"within each split, external baselines have one per split. "
            r"\textbf{Bold} = best in column, \underline{underline} = second. "
            r"The $p$ column after each metric is given for OUR rows only: a two-sided "
            r"paired $t$-test over splits against the best non-ours row in that same "
            r"column, printed raw/Holm (Holm corrects within the column, over our two "
            r"rows). Rank = place within each split among all rows, averaged over splits "
            f"and the {len(metrics)} metrics. -- = not available.")
    cuts = {b[1]["cut"].iloc[0] for b in blocks.values() if len(b[1])}
    thresholded = [m for m in metrics if m in tk.THRESHOLDED]
    if thresholded and cuts != {"n/a"}:
        names = " and ".join(tk.metric_tex(m) for m in thresholded)
        text += (f" {names} are scored at the fixed 0.5 cut, which is what every method "
                 r"compared here reports (LORAX, ProSmith and Hladi\v{s} all threshold "
                 r"their own probabilities at 0.5), so those columns are comparable with "
                 r"the published numbers.")
        if cuts == {"val"}:
            text += (r" The $^{\mathrm{val}}$ columns are the same two metrics at a cut "
                     r"chosen on the VALIDATION rows of each fold, separately for every "
                     r"method -- an additional view, not a replacement; the rank column "
                     r"ignores them. AUROC and AUPRC depend on no threshold.")
        else:
            text += (r" The validation-chosen cut is not shown: per-row scores are not on "
                     r"disk for at least one row here, and giving that cut to some methods "
                     r"and not others would favour them for a reason unrelated to the "
                     r"model. AUROC and AUPRC depend on no threshold.")
    if _combo_split(blocks):
        text += r" $^{\ddagger}$ = the feature set differs between the two regimes."
    return text


def latex(ds, blocks, metrics, a):
    regs = [r for r in tk.REGIMES if r in blocks]
    ncol = 2 * len(metrics) + 1                  # value + p per metric, then Rank
    total = 1 + ncol * len(regs)
    keys = list(dict.fromkeys(k for r in regs for k in blocks[r][1].key))
    split_combo = _combo_split(blocks)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{" + caption(ds, blocks, metrics, a) + "}",
           rf"\label{{tab:main_{ds}}}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{@{}l " + " ".join("c" * ncol for _ in regs) + "@{}}",
           r"\toprule",
           "& " + " & ".join(rf"\multicolumn{{{ncol}}}{{c}}{{\textbf{{{tk.REGIME_LABEL[r]}}}}}"
                             for r in regs) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + i * ncol}-{1 + (i + 1) * ncol}}}"
                   for i in range(len(regs)))]
    sub = " & ".join(sum([[tk.metric_tex(m), "$p$"] for m in metrics], []) + ["Rank"])
    out.append(r"\textbf{Method (features)} & " + " & ".join(sub for _ in regs) + r" \\")
    prev = None
    for key in keys:
        rows_k = [r for reg in regs for r in blocks[reg][0] if r.key == key]
        present = [r for r in rows_k if r.present]
        label = (present or rows_k)[0].tex
        if key in split_combo:
            label += r"$^{\ddagger}$"
        if rows_k[0].kind != prev:
            out.append(r"\midrule")
            prev = rows_k[0].kind
        cells = []
        for reg in regs:
            st = blocks[reg][1]
            for m in metrics:
                col = st[st.metric == m]
                r = col[col.key == key]
                if r.empty or not bool(r.usable.iloc[0]) or not np.isfinite(r["mean"].iloc[0]):
                    cells += ["--", ""]
                    continue
                r = r.iloc[0]
                txt = tk.tex_num(r["mean"], r["std"])
                best, second = tk.top_two(col, m)
                if key == best:
                    txt = rf"\cbest{{{txt}}}"
                elif key == second:
                    txt = rf"\gbest{{{txt}}}"
                cells += [txt, tk.tex_p_pair(r["p_vs_ref"], r["p_holm"])]
            ranks = _mean_ranks(st)
            mr = ranks.get(key, np.nan)
            if not np.isfinite(mr):
                cells.append("--")
            elif np.isclose(mr, ranks.min()):
                cells.append(rf"\textbf{{{mr:.2f}}}")
            else:
                cells.append(f"{mr:.2f}")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    fried = friedman_line(blocks, metrics)
    if fried:
        out.append(r"\midrule")
        out.append(rf"\multicolumn{{{total}}}{{@{{}}l}}{{\footnotesize {fried}}} \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def summary(ds, blocks, metrics, a):
    rec = tk.OF_RECORD[tk.TASK[ds]]
    L = [f"{tk.DATASET_LABEL[ds]} | molecules {a.mol_source} | graph alpha={a.alpha:g} | "
         f"our rows tested against the best non-ours row in each column"]
    for reg, (rows, st) in blocks.items():
        L.append(f"\n[{tk.REGIME_LABEL[reg]}]")
        use = st[st.usable.astype(bool)]
        mr = use.groupby("method")["rank"].mean().dropna().sort_values()
        if len(mr):
            L.append(f"  mean rank over {len(metrics)} metrics (1 = best): "
                     + ", ".join(f"{k} {v:.2f}" for k, v in mr.items()))
        label_of = {r.key: r.label for r in rows}
        for m in metrics:
            col = st[st.metric == m]
            if col.empty:
                continue
            ref_key = col["ref"].iloc[0]
            L.append(f"  {m}: opponent = {label_of.get(ref_key, ref_key or 'none')}, "
                     f"Friedman p={tk.pstr(col['friedman_p'].iloc[0])}")
            for _, o in col[col.kind.isin(TEST_KINDS) & col.usable.astype(bool)].iterrows():
                if not np.isfinite(o["delta_vs_ref"]):
                    continue
                L.append(f"    {o['method']}: {o['delta_vs_ref']:+.3f} vs opponent, "
                         f"ahead on {int(o['ahead_of_ref'])}/{int(o['n_pair'])} splits, "
                         f"t-test p={tk.pstr(o['p_vs_ref'])} (Holm {tk.pstr(o['p_holm'])}), "
                         f"own value {tk.txt_num(o['mean'], o['std'])}")
        if rec in set(st.metric):
            q = st[st.metric == rec]
            L.append(f"  metric of record {rec}: Friedman p={tk.pstr(q['friedman_p'].iloc[0])} "
                     f"over {int(q.k_ranked.max())} methods and {int(q.n_ranked.max())} splits")
        for r in rows:
            if not r.present:
                L.append(f"  ! {r.label}: not available ({'; '.join(r.flags)})")
            elif r.flags:
                L.append(f"  ! {r.label}: {'; '.join(r.flags)}")
    return "\n".join(L)


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=tk.DATASETS, choices=tk.DATASETS)
    ap.add_argument("--mol-source", default="chemberta", choices=tk.MOL_SOURCES)
    ap.add_argument("--alpha", type=float, default=tk.ALPHA)
    ap.add_argument("--ours", nargs="+", default=list(tk.GRAPH_COMBOS),
                    choices=tk.GRAPH_COMBOS, help="which heads of our graph get a row")
    ap.add_argument("--no-ours", dest="ours", action="store_const", const=[],
                    help="drop our graph entirely: baselines against the boosting base "
                         "and nothing else. That is tab:t1's shape -- with "
                         "--baseline-combo cls it asks whether a learned pair "
                         "representation beats two frozen embeddings, a question our "
                         "own rows are not part of. No row is then tested, since every "
                         "test in this table is ours-against-the-best-other.")
    ap.add_argument("--baselines", nargs="+", default=list(tk.BASELINES))
    ap.add_argument("--baseline-combo", nargs="+", default=[tk.BASELINE_COMBO],
                    help="one combo for all, or name=combo items")
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict the sweep rows to these model seeds")
    ap.add_argument("--allow-mol-mismatch", action="store_true",
                    help="keep a baseline row fed another molecule embedding")
    ap.add_argument("--sig", type=float, default=0.05,
                    help="only marks the console view; the table prints both p-values")
    ap.add_argument("--no-val-cut", dest="val_cut", action="store_false",
                    help="do not add the MCC/F1 columns at a validation-chosen "
                         "threshold, even where the per-row scores are on disk")
    ap.add_argument("--sweep-root", default=tk.SWEEP_ROOT)
    ap.add_argument("--ensemble-root", default=tk.ENSEMBLE_ROOT)
    ap.add_argument("--out", default=f"{tk.OUT_ROOT}/main")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    a.baseline_combo = combo_spec(a.baseline_combo)
    a.ours = [c for c in tk.GRAPH_COMBOS if c in set(a.ours)]
    out = tk.out_dir(a.out)
    longs = []
    for ds in a.dataset:
        blocks, metrics = build(ds, a)
        if not blocks:
            print(f"\n=== {tk.DATASET_LABEL[ds]}: nothing on disk")
            continue
        for reg, (_, st) in blocks.items():
            print("\n" + tk.text_block(st, metrics, f"=== {tk.DATASET_LABEL[ds]} / "
                                                     f"{tk.REGIME_LABEL[reg]}", a.sig))
            # block_stats knows nothing about which panel it came from, and
            # main_long.csv is the file you join two runs on -- without these two
            # columns the rows of six panels are indistinguishable.
            longs.append(st.assign(dataset=ds, regime=reg))
        summ = summary(ds, blocks, metrics, a)
        (out / f"{ds}.tex").write_text(latex(ds, blocks, metrics, a) + "\n", encoding="utf-8")
        (out / f"{ds}_summary.txt").write_text(summ + "\n", encoding="utf-8")
        print("\n" + summ)
    if longs:
        pd.concat(longs, ignore_index=True).to_csv(out / "main_long.csv", index=False)
    print(f"\nwritten to {out}")
    return longs


if __name__ == "__main__":
    main()
