"""One table over every ensemble run: the fitted ensemble next to raw boost.

Two numbers per metric per run, side by side:
  * `ensemble[simplex]` -- the simplex-weighted combination of that run's
    per-combo heads, i.e. what the whole ensembling machinery produces;
  * combo `prot+mol` -- the plain XGBoost over [protein || molecule]
    embeddings, the reference every graph/attention idea in this project is
    ultimately judged against.

Columns interleave the two sources per metric (AUROC simplex, AUROC
prot+mol, AUPRC simplex, ...) so the comparison that matters is always
adjacent rather than at opposite ends of the row.

Each cell is `mean ± half-width of the 95% t-interval` over that run's
repeats (LoRaX folds, or cold-molecule seeds) -- not a standard deviation.
With n=5 the interval is wide by construction; that is the point, given how
much cold-molecule numbers move between seeds.

A run that never computed `prot+mol` shows "-" in that column rather than a
blank, so a missing reference is not mistaken for a missing metric.

Usage
-----
uv run python scripts/modeling/eval/ensemble_summary_table.py
uv run python scripts/modeling/eval/ensemble_summary_table.py --csv summary.csv
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

METRICS = ["AUROC", "AUPRC", "F1", "MCC"]
SOURCES = [("simplex", "ensemble", "ensemble[simplex]"),
           ("prot+mol", "combo", "prot+mol")]
ENCODER_TAGS = ("chemberta", "gin", "esm1b", "esm2")


def molecule_encoder(cfg: dict) -> str:
    """Which molecule embedding a run actually used. Not readable from the
    source's CLI type -- `gin` there just means "npz keyed by inchikey" and
    is routinely handed ChemBERTa -- so it comes from the file name."""
    for src in cfg.get("sources", []):
        name, _, rest = src.partition("=")
        if name != "mol":
            continue
        parts = rest.split(":")
        stem = pathlib.Path(parts[1]).stem.lower() if len(parts) > 1 and parts[1] else ""
        return next((t.upper() if t == "gin" else t for t in ENCODER_TAGS if t in stem), stem or "?")
    return "?"


def ci95(values: pd.Series) -> tuple[float, float, int]:
    """(mean, half-width, n). Half-width is nan below two repeats -- one
    number carries no interval, and pretending otherwise would read as a
    suspiciously tight result."""
    x = values.dropna().astype(float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return float(x.iloc[0]), np.nan, 1
    half = student_t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(half), n


def cell(mean: float, half: float) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.3f}" if not np.isfinite(half) else f"{mean:.3f} ± {half:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/ensemble_logs")
    ap.add_argument("--csv", default=None, help="also write the table here")
    ap.add_argument("--raw", action="store_true",
                     help="emit separate mean/ci columns instead of 'mean ± ci' strings")
    args = ap.parse_args()

    rows = []
    for d in sorted((_root / args.out_dir).iterdir()):
        met, cfg_path = d / "metrics.csv", d / "config.json"
        if not d.is_dir() or not met.exists():
            continue
        m = pd.read_csv(met)
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        mode = cfg.get("full_full_mode") if cfg.get("regime") == "full_full" else cfg.get("split")

        row = {"run": d.name, "mode": mode or "?", "mol": molecule_encoder(cfg)}
        n_seen = set()
        for metric in METRICS:
            for label, kind, name in SOURCES:
                sub = m[(m["kind"] == kind) & (m["name"] == name)]
                mean, half, n = ci95(sub[metric]) if metric in sub else (np.nan, np.nan, 0)
                if n:
                    n_seen.add(n)
                if args.raw:
                    row[f"{metric} {label} mean"] = mean
                    row[f"{metric} {label} ci95"] = half
                else:
                    row[f"{metric} {label}"] = cell(mean, half)
        row["n"] = max(n_seen) if n_seen else 0
        rows.append(row)

    if not rows:
        print(f"no runs with metrics.csv under {_root / args.out_dir}")
        return

    table = pd.DataFrame(rows)
    meta = ["run", "mode", "mol", "n"]
    table = table[meta + [c for c in table.columns if c not in meta]]
    # transductive first: it is the easier regime and reads as the reference
    # point for the cold-molecule numbers underneath it.
    table = table.sort_values(
        ["mode", "mol", "run"],
        key=lambda s: s.map({"transductive": ""}).fillna(s) if s.name == "mode" else s)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 40)

    for mode, by_mode in table.groupby("mode", sort=False):
        print(f"\n{'=' * 100}\n{mode}  ({len(by_mode)} run(s))\n{'=' * 100}")
        for mol, by_mol in by_mode.groupby("mol", sort=False):
            print(f"\n  molecules: {mol}")
            block = by_mol.drop(columns=["mode", "mol"])
            print("\n".join("  " + line for line in block.to_string(index=False).splitlines()))

    print("\ncells: mean ± half-width of the 95% t-interval over n repeats; "
          "'-' = that combo was never computed in that run")

    if args.csv:
        out = pathlib.Path(args.csv)
        table.to_csv(out, index=False)
        print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
