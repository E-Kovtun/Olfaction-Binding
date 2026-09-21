#!/usr/bin/env python
"""What every article table can be built from RIGHT NOW -- read-only, trains nothing.

    python scripts/article_tables/00_inventory.py
    python scripts/article_tables/00_inventory.py --only main molecule

For each table it lists every input cell and gives it a status:

    READY     present on all expected splits (and seeds, for sweep rows), no flags
    PARTIAL   present, but short of splits/seeds, on another combo or molecule
              embedding than asked, or on a tuned head -- the detail says which
    MISSING   nothing on disk; the detail says what would produce it

Tables:
    main        01_main_tables.py     6 cells x (4 baselines + boost + 2 graph heads)
    molecule    03_molecule_ablation  18 cells x (graph, boost, Hladis)
    geometry    02 / 02a              sweep geometry at alpha + frozen-embedding CSVs
    protein     tab:t4 as it stands   results/tables/prot_floor_<ds>_<regime>.csv
    figures     node_dial notebook    alphas on disk per cell (+ the Hladis line)
    construction / architecture       graph ablations: what exists, what does not

Also writes results/article_tables/inventory.csv.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tablekit as tk  # noqa: E402

TABLES = ["main", "molecule", "geometry", "protein", "figures", "construction",
          "architecture"]


def status(row, metric, a):
    v = row.values(metric)
    if not len(v):
        return "MISSING", "; ".join(row.flags) or "absent"
    issues = []
    if len(v) < a.expect_splits:
        issues.append(f"{len(v)}/{a.expect_splits} splits")
    if row.source == "sweep" and row.seeds < a.expect_seeds:
        issues.append(f"{row.seeds:g}/{a.expect_seeds} seeds")
    if not row.usable:
        issues.append("NOT USABLE")
    issues += row.flags
    detail = (f"{len(v)} splits" + (f" x {row.seeds:g} seeds" if row.source == "sweep" else "")
              + (f" | {row.origin}" if row.origin else ""))
    return ("PARTIAL" if issues else "READY"), detail + ("" if not issues
                                                           else " | " + "; ".join(issues))


def rec(table, ds, regime, mol, item, st_detail):
    return dict(table=table, dataset=ds, regime=regime, mol_source=mol, item=item,
                status=st_detail[0], detail=st_detail[1])


def main_cells(a):
    out = []
    for ds in tk.DATASETS:
        m = tk.OF_RECORD[tk.TASK[ds]]
        for reg in tk.REGIMES:
            for r in tk.cell_rows(ds, reg, "chemberta", [m], sweep_root=a.sweep_root,
                                  ensemble_root=a.ensemble_root):
                name = r.label if r.kind != "baseline" else tk.BASELINE_LABEL[r.key]
                out.append(rec("main", ds, reg, "chemberta", name, status(r, m, a)))
    return out


def molecule_cells(a):
    out = []
    for ds in tk.DATASETS:
        m = tk.OF_RECORD[tk.TASK[ds]]
        for reg in tk.REGIMES:
            for mol in tk.MOL_SOURCES:
                for r in tk.cell_rows(ds, reg, mol, [m], baselines=("hladis",),
                                      sweep_root=a.sweep_root, ensemble_root=a.ensemble_root):
                    name = r.label if r.kind != "baseline" else "Hladis"
                    out.append(rec("molecule", ds, reg, mol, name, status(r, m, a)))
    return out


def geometry_cells(a):
    out = []
    pg = tk.resolve(f"{tk.OUT_ROOT}/protein_geometry")
    for ds in tk.DATASETS:
        for reg in tk.REGIMES:
            r = tk.ours_row(ds, reg, "chemberta", ["rsa_fun"], "cls+mol", tk.ALPHA,
                            root=a.sweep_root)
            out.append(rec("geometry", ds, reg, "chemberta", "sweep geometry alpha=1",
                           status(r, "rsa_fun", a)))
            p = pg / f"{ds}_{reg}.csv"
            if p.exists():
                df = pd.read_csv(p)
                out.append(rec("geometry", ds, reg, "", "frozen embeddings",
                               ("READY", f"{df.embedding.nunique()} embeddings x "
                                         f"{df.fold.nunique()} splits | {p}")))
            else:
                out.append(rec("geometry", ds, reg, "", "frozen embeddings",
                               ("MISSING", "run 02a_protein_geometry.py (CPU, no training)")))
    return out


def protein_cells(a):
    out = []
    for ds in tk.DATASETS:
        for reg in tk.REGIMES:
            p = tk.resolve(f"results/tables/prot_floor_{ds}_{reg}.csv")
            out.append(rec("protein", ds, reg, "chemberta", "prot_floor_sweep",
                           ("READY", str(p)) if p.exists() else
                           ("MISSING", "prot_floor_sweep.py output not on this box")))
    return out


def figure_cells(a):
    out = []
    df = tk.sweep_frame(a.sweep_root, "cls+mol")
    for ds in tk.DATASETS:
        for reg in tk.REGIMES:
            for mol in tk.MOL_SOURCES:
                g = (df[(df.dataset == ds) & (df.regime == reg) & (df.mol_source == mol)
                        & (df.arm == "gate")] if not df.empty else df)
                n = int(g.alpha.nunique()) if len(g) else 0
                s = ("MISSING" if n == 0 else "READY" if n >= a.expect_alphas else "PARTIAL")
                out.append(rec("figures", ds, reg, mol, "dial curve",
                               (s, f"{n}/{a.expect_alphas} alphas")))
            h = tk.baseline_row("hladis", ds, reg, "chemberta", [tk.OF_RECORD[tk.TASK[ds]]],
                                root=a.ensemble_root)
            out.append(rec("figures", ds, reg, "chemberta", "Hladis reference line",
                           status(h, tk.OF_RECORD[tk.TASK[ds]], a)))
    return out


def construction_cells(a):
    out = []
    # The article's own construction sweep (scripts/article_sweeps/), which shares the
    # alpha sweep's folds and metric battery. The v7 study below predates it: same
    # question, older seeding protocol, not paired with anything the tables read.
    art = sorted(tk.resolve("results/article_sweeps/quantile_criteria").glob("metrics_*.csv"))
    for f in art:
        out.append(rec("construction", f.stem.split("_")[1], "", "",
                       "criterion x quantile (article_sweeps)", ("READY", f.name)))
    if not art:
        out.append(rec("construction", "", "", "", "criterion x quantile (article_sweeps)",
                       ("MISSING", "scripts/article_sweeps/run_quantile_criteria.py "
                                   "output not on this box")))
    v7 = sorted(tk.resolve("results/graph/full_full/v7/protein_based_graph").glob("metrics_*.csv"))
    out.append(rec("construction", "m2or", "", "", "quantile x criterion sweep (v7, superseded)",
                   ("PARTIAL", f"{len(v7)} file(s); older seeding protocol, not paired with "
                               f"v9_seeded -- use article_sweeps for the paper") if v7 else
                   ("MISSING", "the superseded v7 study is not on this box either")))
    alt = sorted(tk.resolve(a.sweep_root).glob("metrics_m2or_*_q0cov_*nodedial.csv"))
    out.append(rec("construction", "m2or", "", "", "q0/coverage variant in the sweep",
                   ("PARTIAL", ", ".join(p.name for p in alt)) if alt else
                   ("MISSING", "sweep --variant q0cov not run")))
    out.append(rec("construction", "", "", "", "unsigned / all-edges graph",
                   ("MISSING", "no switch in orbind/gnn_extractor.py; legacy v5 numbers only")))
    return out


def architecture_cells(a):
    return [rec("architecture", "", "", "", "SAGE vs GAT / GCN / depth / width",
                ("MISSING", "_SignedSage is SAGEConv-only; needs a conv switch + runs"))]


BUILDERS = {"main": main_cells, "molecule": molecule_cells, "geometry": geometry_cells,
            "protein": protein_cells, "figures": figure_cells,
            "construction": construction_cells, "architecture": architecture_cells}


def parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=TABLES, choices=TABLES)
    ap.add_argument("--expect-splits", type=int, default=tk.EXPECTED_SPLITS)
    ap.add_argument("--expect-seeds", type=int, default=5)
    ap.add_argument("--expect-alphas", type=int, default=11)
    ap.add_argument("--sweep-root", default=tk.SWEEP_ROOT)
    ap.add_argument("--ensemble-root", default=tk.ENSEMBLE_ROOT)
    ap.add_argument("--out", default=tk.OUT_ROOT)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    recs = []
    for t in a.only:
        recs += BUILDERS[t](a)
    df = pd.DataFrame(recs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    for t in a.only:
        q = df[df.table == t]
        counts = q.status.value_counts()
        print(f"\n=== {t.upper()}   " + "  ".join(f"{k} {counts.get(k, 0)}"
                                                 for k in ("READY", "PARTIAL", "MISSING")))
        print(q.drop(columns="table").to_string(index=False))
    path = tk.out_dir(a.out) / "inventory.csv"
    df.to_csv(path, index=False)
    print(f"\nwritten to {path}")
    return df


if __name__ == "__main__":
    main()
