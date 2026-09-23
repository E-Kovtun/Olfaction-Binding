#!/usr/bin/env python
"""Geometry table: how well each receptor embedding lines up with the functional response
profile -- RSA, CCA and Procrustes -- for our graph at a fixed dial position and for the
frozen protein features of tab:t4.

    python scripts/legacy/02_geometry_table.py
    python scripts/legacy/02_geometry_table.py --regime inductive --alphas 1.0 0.0

Rows:
    Our graph (alpha=...)   the sweep's own geometry columns `{rsa,cca,procrustes}_fun`
                            at that alpha (a property of the trained graph, identical for
                            both heads)
    ESM-1b, ProtT5, ...     results/article_tables/protein_geometry/<ds>_<regime>.csv,
                            written by 02a_protein_geometry.py on the same receptors and
                            the same train profile

A cell is mean +/- std over splits (sweep rows averaged over seeds within each split).
`o` = the mean z against the row-permutation null is below --z (not distinguishable from
chance). `*` = differs from the reference graph row, paired t-test over splits, Holm
within the column.

Insects only by default: on M2OR the sparse profile is largely assay design.

Writes to results/article_tables/geometry/: geometry_long.csv, geometry_<regime>.tex,
geometry_<regime>_summary.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

GEOMS = ["rsa", "cca", "procrustes"]
GEOM_LABEL = {"rsa": "RSA", "cca": "CCA", "procrustes": "Procrustes"}
EMB_LABEL = {"esm1b": "ESM-1b", "prott5": "ProtT5", "esm2": "ESM-2", "esm3": "ESM3", "kmer2": "kmer2",
             "ctd": "CTD", "pseaac": "PseAAC", "blosum": "BLOSUM", "aac": "AAC",
             "aaindex": "AAindex", "onehot": "one-hot"}
COLS = [f"{g}_fun" for g in GEOMS] + [f"{g}_fun_z" for g in GEOMS]


def graph_rows(ds, regime, a):
    rows = []
    for alpha in a.alphas:
        r = tk.ours_row(ds, regime, a.mol_source, COLS, "cls+mol", alpha, a.seeds,
                        a.sweep_root)
        r.key = f"ours:a={alpha:g}"
        r.label = f"Our graph (alpha={alpha:g})"
        r.tex = rf"Our graph ($\alpha={alpha:g}$)"
        rows.append(r)
    return rows


def embedding_rows(ds, regime, a):
    path = tk.resolve(a.protein_geometry) / f"{ds}_{regime}.csv"
    if not path.exists():
        r = tk.Row("esm1b", "ESM-1b", "ESM-1b", "embedding", "", "protein_geometry")
        r.flags.append(f"{path} missing -- run 02a_protein_geometry.py")
        return [r]
    df = pd.read_csv(path)
    rows = []
    names = [e for e in EMB_LABEL if e in set(df.embedding)] + \
            sorted(set(df.embedding) - set(EMB_LABEL))
    for e in names:
        q = df[df.embedding == e]
        cols = [c for c in COLS if c in q.columns]
        folds = q.groupby("fold")[cols].mean()
        folds.index = folds.index.astype(int)
        rows.append(tk.Row(e, EMB_LABEL.get(e, e), EMB_LABEL.get(e, e), "embedding", "",
                           "protein_geometry", folds=folds, origin=str(path)))
    return rows


def build(a):
    cells = {}
    for ds in a.dataset:
        rows = graph_rows(ds, a.regime, a) + embedding_rows(ds, a.regime, a)
        if not any(r.present for r in rows):
            continue
        st = tk.block_stats(rows, [f"{g}_fun" for g in GEOMS], f"ours:a={a.alphas[0]:g}")
        z = {(r.key, g): (float(r.values(f"{g}_fun_z").mean())
                          if len(r.values(f"{g}_fun_z")) else np.nan)
             for r in rows for g in GEOMS}
        st = st.assign(dataset=ds, regime=a.regime,
                       z=[z.get((k, m.replace("_fun", "")), np.nan)
                          for k, m in zip(st.key, st.metric)])
        cells[ds] = (rows, st)
    return cells


def latex(cells, a):
    dss = list(cells)
    k = len(GEOMS)
    keys = list(dict.fromkeys(key for ds in dss for key in cells[ds][1].key))
    cap = ("Alignment of receptor embeddings with the functional response profile "
           f"({tk.REGIME_LABEL[a.regime].lower()} splits): RSA = Spearman correlation between "
           "the off-diagonals of receptor cosine similarity and response-profile similarity; "
           "CCA = mean canonical correlation and Procrustes = $1-$disparity, both on the top-3 "
           "principal components. Larger = more aligned. Profile = train-split receptor "
           r"$\times$ odorant responses, row-centred. Mean $\pm$ std over splits (our graph "
           "averaged over model seeds within each split first). Our graph = the refined "
           "receptor vector; the other rows are frozen protein features. "
           r"\textbf{Bold} = best in column, \underline{underline} = second. "
           f"$^{{\\circ}}$ = mean $z$ against a row-permutation null below {a.z:g}.")
    if a.tests:
        cap += (f" $^{{*}}$ = differs from our graph ($\\alpha={a.alphas[0]:g}$) at "
                f"$p<{a.sig:g}$, paired two-sided $t$-test over splits, Holm-corrected "
                "within the column.")
    if "m2or" in dss:
        cap += (" On M2OR the profile is sparse and the assayed-pair mask alone reproduces "
                "most of the graph's alignment; read that block with caution.")
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + cap + "}",
           rf"\label{{tab:geometry_{a.regime}}}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{@{}l " + " ".join("c" * k for _ in dss) + "@{}}", r"\toprule",
           "& " + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{\textbf{{{tk.DATASET_TEX[d]}}}}}"
                             for d in dss) + r" \\",
           "".join(rf"\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(len(dss))),
           r"\textbf{Receptor embedding} & " + " & ".join(" & ".join(GEOM_LABEL[g] for g in GEOMS)
                                                          for _ in dss) + r" \\"]
    prev = None
    for key in keys:
        rows_k = [r for d in dss for r in cells[d][0] if r.key == key]
        kind = "ours" if key.startswith("ours") else ("plm" if key in ("esm1b", "prott5", "esm2", "esm3")
                                                      else ("id" if key == "onehot" else "desc"))
        if kind != prev:
            out.append(r"\midrule")
            prev = kind
        row = []
        for d in dss:
            st = cells[d][1]
            for g in GEOMS:
                col = st[st.metric == f"{g}_fun"]
                r = col[col.key == key]
                if r.empty or not np.isfinite(r["mean"].iloc[0]):
                    row.append("--")
                    continue
                r = r.iloc[0]
                txt = tk.tex_num(r["mean"], r["std"])
                if np.isfinite(r["z"]) and r["z"] < a.z:
                    txt += r"$^{\circ}$"
                if (a.tests and key != r["ref"] and np.isfinite(r["p_holm"])
                        and r["p_holm"] < a.sig):
                    txt += r"$^{*}$"
                best, second = tk.top_two(col, f"{g}_fun")
                txt = (rf"\cbest{{{txt}}}" if key == best else
                       rf"\gbest{{{txt}}}" if key == second else txt)
                row.append(txt)
        out.append(f"{rows_k[0].tex} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(out)


def summary(cells, a):
    L = [f"geometry vs functional profile | {a.regime} | reference our graph alpha={a.alphas[0]:g}"]
    for ds, (rows, st) in cells.items():
        L.append(f"\n[{tk.DATASET_LABEL[ds]}]")
        for g in GEOMS:
            col = st[st.metric == f"{g}_fun"].dropna(subset=["mean"])
            if col.empty:
                continue
            col = col.sort_values("mean", ascending=False)
            L.append(f"  {GEOM_LABEL[g]}: " + ", ".join(
                f"{r['method']} {r['mean']:.3f} (z {tk.fnum(r['z'], 1)})"
                for _, r in col.iterrows()))
            ref = col[col.key == col.ref.iloc[0]]
            esm = col[col.key == "esm1b"]
            if len(ref) and len(esm):
                e = esm.iloc[0]
                L.append(f"    our graph - ESM-1b {-e['delta_vs_ref']:+.3f}, graph ahead "
                         f"{int(e['behind_ref'])}/{int(e['n_pair'])}, p={tk.pstr(e['p_vs_ref'])}"
                         f" (Holm {tk.pstr(e['p_holm'])})")
        for r in rows:
            if not r.present or r.flags:
                L.append(f"  ! {r.label}: {'; '.join(r.flags) or 'not available'}")
    return "\n".join(L)


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["cc", "hc"], choices=tk.DATASETS)
    ap.add_argument("--regime", default="transductive", choices=tk.REGIMES)
    ap.add_argument("--alphas", type=float, nargs="+", default=[tk.ALPHA],
                    help="dial positions that get a row; the first is the test reference")
    ap.add_argument("--mol-source", default="chemberta", choices=tk.MOL_SOURCES)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--sig", type=float, default=0.05)
    ap.add_argument("--no-tests", dest="tests", action="store_false",
                    help="drop the paired-t/Holm markers from the LaTeX table. The console "
                         "view and geometry_long.csv keep every p-value either way, so this "
                         "only changes what the paper prints. The permutation-null mark "
                         "($^\\circ$) is a different statement and stays.")
    ap.add_argument("--sweep-root", default=tk.SWEEP_ROOT)
    ap.add_argument("--protein-geometry", default=f"{tk.OUT_ROOT}/protein_geometry")
    ap.add_argument("--out", default=f"{tk.OUT_ROOT}/geometry")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    out = tk.out_dir(a.out)
    cells = build(a)
    if not cells:
        print("nothing on disk")
        return None
    for ds, (_, st) in cells.items():
        print("\n" + tk.text_block(st, [f"{g}_fun" for g in GEOMS],
                                   f"=== {tk.DATASET_LABEL[ds]} / {a.regime}", a.sig))
    summ = summary(cells, a)
    (out / f"geometry_{a.regime}.tex").write_text(latex(cells, a) + "\n", encoding="utf-8")
    (out / f"geometry_{a.regime}_summary.txt").write_text(summ + "\n", encoding="utf-8")
    long = pd.concat([st for _, st in cells.values()], ignore_index=True)
    path = out / "geometry_long.csv"
    if path.exists():
        old = pd.read_csv(path)
        long = pd.concat([old[old.regime != a.regime], long], ignore_index=True)
    long.to_csv(path, index=False)
    print("\n" + summ + f"\n\nwritten to {out}")
    return cells


if __name__ == "__main__":
    main()
