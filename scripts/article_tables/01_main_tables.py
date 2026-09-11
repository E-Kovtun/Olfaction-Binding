#!/usr/bin/env python
"""Main tables: every method on every (dataset, regime), our graph at one dial position.

    python scripts/article_tables/01_main_tables.py
    python scripts/article_tables/01_main_tables.py --dataset cc hc --ref cls+prot+mol
    python scripts/article_tables/01_main_tables.py --baseline-combo cls hladis=cls+prot+mol

One table per dataset, the two regimes side by side (transductive | cold molecule):

    LORAX / ProSmith / MolOR / Hladis   external baselines, results/ensemble_logs, at
                                        --baseline-combo (default cls+prot+mol, the
                                        footing of tab:t1full)
    Boosting base (prot+mol)            XGBoost on [raw ESM || molecule], from the sweep
    Our graph (cls+mol)                 [z_prot || molecule]             } one trained
    Our graph (cls+prot+mol)            [z_prot || raw ESM || molecule]  } graph, two heads

A cell is mean +/- std over the held-out splits; sweep rows average their model seeds
inside each split first. `*` = a paired two-sided t-test over splits against `--ref`,
Holm-corrected within the column. Rank = place within each split among all rows,
averaged over splits and the metrics shown. The Friedman p per metric is in the long CSV
and in the caption for the metric of record.

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
        st = tk.block_stats(rows, metrics, f"ours:{a.ref}")
        blocks[regime] = (rows, st.assign(dataset=ds, regime=regime,
                                          mol_source=a.mol_source, alpha=a.alpha))
    return blocks, metrics


def _mean_ranks(st):
    return st[st.usable.astype(bool)].groupby("key")["rank"].mean()


def caption(ds, blocks, metrics, a):
    sts = pd.concat([b[1] for b in blocks.values()], ignore_index=True)
    use = sts[sts.usable.astype(bool) & (sts.n > 0)]
    n = int(use.n.max()) if len(use) else 0
    sw = use[use.source == "sweep"]
    seeds = f"{sw.seeds.max():g}" if len(sw) else "the"
    rec = tk.OF_RECORD[tk.TASK[ds]]
    fried = "; ".join(
        f"{tk.REGIME_LABEL[r].lower()} $p={tk.pstr(st.loc[st.metric == rec, 'friedman_p'].iloc[0])}$"
        for r, (_, st) in blocks.items() if (st.metric == rec).any())
    splits = ("the LORAX folds and the cold-molecule seeds 42--46" if ds == "m2or"
              else "the upstream random folds and our cold-molecule folds")
    text = (f"{tk.DATASET_TEX[ds]}, molecule embedding {tk.MOL_LABEL[a.mol_source]}, "
            f"our graph at $\\alpha={a.alpha:g}$. Mean $\\pm$ std over {n} held-out splits "
            f"({splits}); rows from our sweep are first averaged over {seeds} model seeds "
            f"within each split, external baselines have one per split. "
            r"\textbf{Bold} = best in column, \underline{underline} = second. "
            f"$^{{*}}$ = differs from Our graph ({a.ref}) at $p<{a.sig:g}$, paired two-sided "
            f"$t$-test over splits, Holm-corrected within the column. Rank = place within "
            f"each split among all rows, averaged over splits and the {len(metrics)} "
            f"metrics. Friedman test on {tk.metric_tex(rec)}: {fried}. "
            r"-- = not available.")
    if _combo_split(blocks):
        text += r" $^{\ddagger}$ = the feature set differs between the two regimes."
    return text


def _combo_split(blocks):
    """Keys whose combo differs ACROSS regimes (a fallback in one of them)."""
    seen = {}
    for rows, _ in blocks.values():
        for r in rows:
            if r.present:
                seen.setdefault(r.key, set()).add(r.combo)
    return {k for k, v in seen.items() if len(v) > 1}


def latex(ds, blocks, metrics, a):
    regs = [r for r in tk.REGIMES if r in blocks]
    ncol = len(metrics) + 1
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
    sub = " & ".join(tk.metric_tex(m) for m in metrics) + " & Rank"
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
                    cells.append("--")
                    continue
                r = r.iloc[0]
                txt = tk.tex_num(r["mean"], r["std"])
                if key != r["ref"] and np.isfinite(r["p_holm"]) and r["p_holm"] < a.sig:
                    txt += r"$^{*}$"
                best, second = tk.top_two(col, m)
                if key == best:
                    txt = rf"\cbest{{{txt}}}"
                elif key == second:
                    txt = rf"\gbest{{{txt}}}"
                cells.append(txt)
            ranks = _mean_ranks(st)
            mr = ranks.get(key, np.nan)
            if not np.isfinite(mr):
                cells.append("--")
            elif np.isclose(mr, ranks.min()):
                cells.append(rf"\textbf{{{mr:.2f}}}")
            else:
                cells.append(f"{mr:.2f}")
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def summary(ds, blocks, metrics, a):
    rec = tk.OF_RECORD[tk.TASK[ds]]
    ref_key = f"ours:{a.ref}"
    L = [f"{tk.DATASET_LABEL[ds]} | molecules {a.mol_source} | graph alpha={a.alpha:g} | "
         f"reference Our graph ({a.ref})"]
    for reg, (rows, st) in blocks.items():
        L.append(f"\n[{tk.REGIME_LABEL[reg]}]")
        use = st[st.usable.astype(bool)]
        mr = use.groupby("method")["rank"].mean().dropna().sort_values()
        if len(mr):
            L.append(f"  mean rank over {len(metrics)} metrics (1 = best): "
                     + ", ".join(f"{k} {v:.2f}" for k, v in mr.items()))
        col = st[st.metric == rec]
        ref = col[col.key == ref_key]
        if not ref.empty and np.isfinite(ref["mean"].iloc[0]):
            r = ref.iloc[0]
            L.append(f"  Our graph ({a.ref}) {rec} {tk.txt_num(r['mean'], r['std'])} on "
                     f"{int(r['n'])} splits x {r['seeds']:g} seeds")
            for _, o in col[(col.key != ref_key) & col.usable.astype(bool)].iterrows():
                if not np.isfinite(o["delta_vs_ref"]):
                    continue
                L.append(f"    vs {o['method']}: ours {-o['delta_vs_ref']:+.3f} {rec}, "
                         f"ahead on {int(o['behind_ref'])}/{int(o['n_pair'])} splits, "
                         f"t-test p={tk.pstr(o['p_vs_ref'])} (Holm {tk.pstr(o['p_holm'])})")
        if len(col):
            L.append(f"  Friedman on {rec}: p={tk.pstr(col['friedman_p'].iloc[0])} over "
                     f"{int(col.k_ranked.max())} methods and {int(col.n_ranked.max())} splits")
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
    ap.add_argument("--ref", default="cls+mol", choices=tk.GRAPH_COMBOS,
                    help="the row every other row is tested against")
    ap.add_argument("--baselines", nargs="+", default=list(tk.BASELINES))
    ap.add_argument("--baseline-combo", nargs="+", default=[tk.BASELINE_COMBO],
                    help="one combo for all, or name=combo items")
    ap.add_argument("--metrics", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="restrict the sweep rows to these model seeds")
    ap.add_argument("--allow-mol-mismatch", action="store_true",
                    help="keep a baseline row fed another molecule embedding")
    ap.add_argument("--sig", type=float, default=0.05)
    ap.add_argument("--sweep-root", default=tk.SWEEP_ROOT)
    ap.add_argument("--ensemble-root", default=tk.ENSEMBLE_ROOT)
    ap.add_argument("--out", default=f"{tk.OUT_ROOT}/main")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    a.baseline_combo = combo_spec(a.baseline_combo)
    a.ours = [c for c in tk.GRAPH_COMBOS if c in set(a.ours) | {a.ref}]
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
