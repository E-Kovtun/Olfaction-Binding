#!/usr/bin/env python
"""Main tables: every method on every (dataset, regime), our graph at one dial position.

    python scripts/article_tables/01_main_tables.py
    python scripts/article_tables/01_main_tables.py --dataset cc hc
    python scripts/article_tables/01_main_tables.py --baseline-combo cls hladis=cls+prot+mol

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
    metrics = a.metrics or tk.SHOW[tk.TASK[ds]]
    blocks = {}
    for regime in tk.REGIMES:
        rows = tk.cell_rows(ds, regime, a.mol_source, metrics, ours=a.ours,
                            baselines=a.baselines, baseline_combo=a.baseline_combo,
                            alpha=a.alpha, seeds=a.seeds, sweep_root=a.sweep_root,
                            ensemble_root=a.ensemble_root,
                            allow_mol_mismatch=a.allow_mol_mismatch)
        if not any(r.present for r in rows):
            continue
        # MCC and F1 depend on where the decision boundary falls. Prefer a cut chosen on
        # validation; fall back to the fixed 0.5 for EVERY row if even one of them has no
        # per-row scores on disk -- see `tablekit.apply_val_threshold`.
        cut = tk.apply_val_threshold(rows, metrics, tk.TASK[ds])
        st = tk.block_stats(rows, metrics, ref_mode="best_other", test_kinds=TEST_KINDS)
        blocks[regime] = (rows, st.assign(dataset=ds, regime=regime,
                                          mol_source=a.mol_source, alpha=a.alpha,
                                          cut=cut))
    return blocks, metrics


def _mean_ranks(st):
    return st[st.usable.astype(bool)].groupby("key")["rank"].mean()


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
        if cuts == {"val"}:
            text += (f" {names} are scored at a decision threshold chosen on the "
                     r"VALIDATION split of each fold, separately for every method.")
        else:
            text += (f" {names} are scored at the FIXED 0.5 threshold for every method: "
                     r"per-row validation scores are not on disk for at least one row "
                     r"here, and a threshold given to some methods and not others would "
                     r"favour them for a reason unrelated to the model. AUROC and AUPRC "
                     r"do not depend on a threshold.")
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
            longs.append(st)
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
