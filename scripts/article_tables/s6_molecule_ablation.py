#!/usr/bin/env python
"""Molecule-embedding ablation: our graph, the boosting base and Hladis on every molecule
embedding -- the successor of tab:t2m2or / tab:t2cc / tab:t2hc, now at the seeded dial.

    python scripts/article_tables/s6_molecule_ablation.py
    python scripts/article_tables/s6_molecule_ablation.py --ours cls+prot+mol

One table per dataset: rows = ChemBERTa / GIN / ECFP, columns = regime x method. A cell is
the metric of record, mean +/- std over splits, and in parentheses the place among the
methods within each split, averaged over splits. `*` = paired t-test against our graph,
Holm within the cell group. The last row is the mean place over the embeddings.

Graph and base come from ONE sweep cell, so they are paired by split and seed; Hladis is
borrowed from results/ensemble_logs and matched on the molecule embedding FILE (the insect
pools hold concatCB / concatGIN / concatECFP side by side).

Writes to results/article_tables/molecule/:
    molecule_long.csv, <ds>.tex, <ds>_summary.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

ORDER = {"ours": 0, "boost": 1, "baseline": 2}


def build(ds, a):
    metric = a.metric or tk.OF_RECORD[tk.TASK[ds]]
    cells = {}
    for regime in tk.REGIMES:
        for mol in a.mol_sources:
            rows = tk.cell_rows(ds, regime, mol, [metric], ours=[a.ours],
                                baselines=a.baselines, baseline_combo=a.baseline_combo,
                                alpha=a.alpha, seeds=a.seeds, sweep_root=a.sweep_root,
                                ensemble_root=a.ensemble_root)
            rows = sorted(rows, key=lambda r: ORDER[r.kind])
            if any(r.present for r in rows):
                st = tk.block_stats(rows, [metric], f"ours:{a.ours}")
                cells[(regime, mol)] = (rows, st.assign(dataset=ds, regime=regime,
                                                        mol_source=mol, alpha=a.alpha))
    return cells, metric


def _keys(a):
    return [f"ours:{a.ours}", "boost"] + list(a.baselines)


def _head(key):
    if key.startswith("ours:"):
        return "Graph"
    if key == "boost":
        return "Base"
    return tk.BASELINE_TEX.get(key, key)


def latex(ds, cells, metric, a):
    regs = [r for r in tk.REGIMES if any(k[0] == r for k in cells)]
    keys = _keys(a)
    k = len(keys)
    combo = (a.baseline_combo if isinstance(a.baseline_combo, str)
             else ", ".join(f"{n} {c}" for n, c in a.baseline_combo.items()))
    cap = (f"{tk.DATASET_TEX[ds]}: our graph ({a.ours}, $\\alpha={a.alpha:g}$), the boosting "
           f"base (prot+mol) and "
           + ", ".join(tk.BASELINE_TEX.get(b, b) for b in a.baselines)
           + f" ({combo}) across molecule embeddings, {tk.metric_tex(metric)}, mean $\\pm$ "
           f"std over held-out splits (our sweep's rows averaged over model seeds within each "
           f"split). In parentheses: place among the {k} methods within each split, "
           r"averaged over splits. \textbf{Bold} = best value in the group. "
           f"$^{{*}}$ = differs from our graph at $p<{a.sig:g}$, paired two-sided $t$-test "
           r"over splits, Holm-corrected within the group. Last row: mean place over the "
           r"embeddings. -- = not available.")
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + cap + "}",
           rf"\label{{tab:mol_{ds}}}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{@{}l " + " ".join("c" * k for _ in regs) + "@{}}", r"\toprule",
           "& " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{\textbf{{{tk.REGIME_LABEL[r]}}}}}"
                             for r in regs) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(len(regs))),
           r"\textbf{Molecule} & " + " & ".join(" & ".join(_head(x) for x in keys)
                                                for _ in regs) + r" \\",
           r"\midrule"]
    for mol in a.mol_sources:
        row = []
        for reg in regs:
            got = cells.get((reg, mol))
            best = tk.top_two(got[1], metric)[0] if got else None
            for key in keys:
                r = got[1][got[1].key == key] if got else pd.DataFrame()
                if r.empty or not bool(r.usable.iloc[0]) or not np.isfinite(r["mean"].iloc[0]):
                    row.append("--")
                    continue
                r = r.iloc[0]
                txt = tk.tex_num(r["mean"], r["std"])
                if np.isfinite(r["rank"]):
                    txt += f" ({r['rank']:.2f})"
                if key != r["ref"] and np.isfinite(r["p_holm"]) and r["p_holm"] < a.sig:
                    txt += r"$^{*}$"
                row.append(rf"\cbest{{{txt}}}" if key == best else txt)
        out.append(f"{tk.MOL_LABEL[mol]} & " + " & ".join(row) + r" \\")
    out.append(r"\midrule")
    row = []
    for reg in regs:
        mr = mean_place(cells, reg, keys)
        low = np.nanmin(list(mr.values())) if any(np.isfinite(v) for v in mr.values()) else np.nan
        for key in keys:
            v = mr[key]
            row.append("--" if not np.isfinite(v) else
                       (rf"\cbest{{{v:.2f}}}" if np.isclose(v, low) else f"{v:.2f}"))
    out.append(r"\textit{Mean rank} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def mean_place(cells, reg, keys):
    """Mean over the embeddings of each method's per-split place."""
    vals = {k: [] for k in keys}
    for (r, _), (_, st) in cells.items():
        if r != reg:
            continue
        for key in keys:
            s = st[(st.key == key) & st.usable.astype(bool)]
            if len(s) and np.isfinite(s["rank"].iloc[0]):
                vals[key].append(float(s["rank"].iloc[0]))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in vals.items()}


def summary(ds, cells, metric, a):
    keys = _keys(a)
    L = [f"{tk.DATASET_LABEL[ds]} | {metric} | graph {a.ours} at alpha={a.alpha:g}"]
    for reg in tk.REGIMES:
        here = [(m, cells[(reg, m)]) for m in a.mol_sources if (reg, m) in cells]
        if not here:
            continue
        L.append(f"\n[{tk.REGIME_LABEL[reg]}]")
        mp = mean_place(cells, reg, keys)
        L.append("  mean place over embeddings: "
                 + ", ".join(f"{tk.BASELINE_LABEL.get(k, _head(k))} {tk.fnum(v, 2)}"
                             for k, v in mp.items()))
        for mol, (rows, st) in here:
            o = st[st.key == "boost"]
            ours = st[st.key == f"ours:{a.ours}"]
            line = f"  {tk.MOL_LABEL[mol]:<10}"
            if len(ours) and np.isfinite(ours["mean"].iloc[0]):
                line += f" graph {tk.txt_num(ours['mean'].iloc[0], ours['std'].iloc[0])}"
            if len(o) and np.isfinite(o["delta_vs_ref"].iloc[0]):
                r = o.iloc[0]
                line += (f" | graph - base {-r['delta_vs_ref']:+.3f}, graph ahead "
                         f"{int(r['behind_ref'])}/{int(r['n_pair'])}, p={tk.pstr(r['p_vs_ref'])}")
            L.append(line)
            for r in rows:
                if not r.present:
                    L.append(f"      ! {r.label}: not available ({'; '.join(r.flags)})")
                elif r.flags:
                    L.append(f"      ! {r.label}: {'; '.join(r.flags)}")
    return "\n".join(L)


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=tk.DATASETS, choices=tk.DATASETS)
    ap.add_argument("--mol-sources", nargs="+", default=tk.MOL_SOURCES, choices=tk.MOL_SOURCES)
    ap.add_argument("--metric", default=None, help="default: R2 on insects, AUROC on M2OR")
    ap.add_argument("--alpha", type=float, default=tk.ALPHA)
    ap.add_argument("--ours", default="cls+mol", choices=tk.GRAPH_COMBOS)
    ap.add_argument("--baselines", nargs="+", default=["hladis"])
    ap.add_argument("--baseline-combo", default=tk.BASELINE_COMBO)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--sig", type=float, default=0.05)
    ap.add_argument("--sweep-root", default=tk.SWEEP_ROOT)
    ap.add_argument("--ensemble-root", default=tk.ENSEMBLE_ROOT)
    ap.add_argument("--out", default=f"{tk.OUT_ROOT}/molecule")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = tk.out_dir(a.out)
    longs = []
    for ds in a.dataset:
        if a.metric and a.metric not in tk.SHOW[tk.TASK[ds]] + ["MAE", "Pearson", "Spearman"]:
            print(f"  {ds}: {a.metric} is not a {tk.TASK[ds]} metric, skipped")
            continue
        cells, metric = build(ds, a)
        if not cells:
            print(f"\n=== {tk.DATASET_LABEL[ds]}: nothing on disk")
            continue
        for (reg, mol), (_, st) in cells.items():
            print("\n" + tk.text_block(st, [metric], f"=== {tk.DATASET_LABEL[ds]} / "
                                                     f"{tk.REGIME_LABEL[reg]} / {mol}", a.sig))
            longs.append(st)
        summ = summary(ds, cells, metric, a)
        (out / f"{ds}.tex").write_text(latex(ds, cells, metric, a) + "\n", encoding="utf-8")
        (out / f"{ds}_summary.txt").write_text(summ + "\n", encoding="utf-8")
        print("\n" + summ)
    if longs:
        pd.concat(longs, ignore_index=True).to_csv(out / "molecule_long.csv", index=False)
    print(f"\nwritten to {out}")
    return longs


if __name__ == "__main__":
    main()
