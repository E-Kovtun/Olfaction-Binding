#!/usr/bin/env python
"""Score the sweep's OWN trained models on the VALIDATION split.

    python scripts/analysis/val_rescore.py --root results/graph/v9_node_dial
    python scripts/analysis/val_rescore.py --root ... --dataset hc --dry-run

WHY THIS EXISTS. `run_alpha_gate_sweep.py` computes `Zp_va` and then uses it only to
collect receptor vectors: every metric it writes is on TEST. So the grid on disk cannot
answer "which alpha should we report" without choosing alpha on the very rows the choice
is later defended with. That is the one thing a hyperparameter must never be chosen on,
and alpha is a hyperparameter -- it is fixed before training and used unchanged at
inference, exactly like a learning rate.

WHAT IT DOES NOT DO: retrain anything on the GPU. The sweep dumps `z_prot` -- the
refined receptor cloud in universe order -- for every graph cell, and `fit_boost` is
deterministic given its seed. So the head is refit on the SAME train rows with the SAME
seed, which reproduces the estimator whose test score is already in the CSV, and is then
asked for the val rows instead. One XGBoost fit per cell, no message passing.

WHY THAT IS EXACT, AND HOW IT IS CHECKED. With `emit="prot"` the per-pair feature IS the
receptor's row of `z_prot`, so collapsing to one row per receptor and expanding again is
lossless. That is an argument, not a proof, so `--verify` (on by default) also predicts
TEST from the refit model and compares against the number the sweep recorded. A
reconstruction that is right agrees to ~1e-6; anything larger is printed loudly and the
val numbers from that cell should not be believed.

WHAT MAKES VAL A LEGITIMATE PLACE TO CHOOSE FROM. The graph never sees it: message
passing edges are built from train rows only, training runs a fixed epoch count with no
early stopping and never consults val, and the v9 mixture statistics are fit on train
receptors. And the val split has the same SHAPE as test in every cell that matters --
`inductive_molecule_v5` draws val molecules disjoint from train and test, and
`our_inductive` spreads val over the same dynamic-range order as test, so a cold-molecule
choice is made on cold-molecule evidence.

OUTPUT. `val_metrics_<same tag>.csv` next to each `metrics_<tag>.csv`, same columns, same
(arm, alpha, fold, seed) identity, plus `split="val"`. Read it with
`alpha_grid.load(..., split="val")`. The glob prefixes differ, so no reader can pick up
one thinking it is the other.

Resumable: a cell already in the val file is skipped, and the file is rewritten after
every fold.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from orbind.baselines import fit_boost, predict_scores                # noqa: E402


def _module(rel, name):
    """Import a script by path -- these live under scripts/, which is not a package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _root / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sweep = _module("scripts/modeling/train/run_alpha_gate_sweep.py", "_sweep_for_val")
ht = _module("scripts/analysis/headline_table.py", "_ht_for_val")

VAL_PREFIX = "val_metrics_"


def _fmt(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def dumps_dir(csv_path):
    """Where `sweep._dump` put this file's cells."""
    return csv_path.parent / "dumps" / csv_path.stem.replace("metrics_", "")


def dump_path(d, arm, alpha, fold, seed):
    a = "None" if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)) \
        else f"{float(alpha):g}"
    return d / f"{arm}_a{a}_f{fold}_s{seed}.npz"


def _ns(ds, mol_source, variant, pool_fold=1):
    """The subset of the sweep's argparse namespace its data helpers actually read."""
    return argparse.Namespace(mol_source=mol_source, mol_embeddings=None,
                              prot_embeddings=None, pool_fold=pool_fold,
                              variant=variant, _ds=ds, _variant=variant)


def val_arrays(P, pairs, mol, prot):
    """The val half of what `_fold_prep` builds for test, in the same shapes."""
    va = P["va"]
    ik = pairs["inchikey"].to_numpy()[va]
    rc = pairs["receptor"].to_numpy()[va]
    return dict(y=pairs["label"].to_numpy(np.float32)[va], rec=rc, mol=ik,
                Xm=sweep._mat(mol, ik), Xp=sweep._mat(prot, rc))


def features(Z, receptors, rec_ids, Xm):
    """[ z_prot(receptor) || raw molecule ] -- the sweep's own `cls+mol` construction.

    `Z` is one row per receptor in universe order; the boost input repeats that row for
    every pair the receptor appears in. Missing receptor -> None, and the caller skips
    the cell rather than filling a zero vector nobody would notice."""
    row = {str(r): i for i, r in enumerate(receptors)}
    idx = [row.get(str(r)) for r in rec_ids]
    if any(i is None for i in idx):
        return None
    return np.concatenate([Z[np.asarray(idx, dtype=np.int64)], Xm], axis=1)


def score_cell(row, P, V, dumps, task, verify):
    """(val metrics, test metrics or None, note). `row` is one line of the metrics CSV."""
    arm = str(row["arm"])
    alpha = None if pd.isna(row["alpha"]) else float(row["alpha"])
    fold, seed = int(row["fold"]), int(row["seed"])

    Pv = dict(P)
    Pv["y_te"], Pv["rec_te"], Pv["mol_te"] = V["y"], V["rec"], V["mol"]

    if arm == "naive":
        # a constant has no model to refit; the honest val number is the same constant
        const = np.full(len(V["y"]), float(P["y_tr"].mean()), dtype=np.float32)
        return sweep._score(Pv, const, task), None, ""

    if arm == "boost_full":
        Xtr = np.concatenate([P["Xp_tr"], P["Xm_tr"]], 1)
        Xva = np.concatenate([V["Xp"], V["Xm"]], 1)
        Xte = np.concatenate([P["Xp_te"], P["Xm_te"]], 1)
    else:
        f = dump_path(dumps, arm, alpha, fold, seed)
        if not f.exists():
            return None, None, f"no embedding dump at {f.name}"
        z = np.load(f, allow_pickle=False)
        if "z_prot" not in z:
            return None, None, f"{f.name} holds no z_prot (run without --dump-embeddings?)"
        Z, recs = z["z_prot"], z["receptors"]
        Xtr = features(Z, recs, P["rec_tr"], P["Xm_tr"])
        Xva = features(Z, recs, V["rec"], V["Xm"])
        Xte = features(Z, recs, P["rec_te"], P["Xm_te"]) if verify else None
        if Xtr is None or Xva is None:
            return None, None, f"{f.name}: a receptor in this fold is not in the dump"

    est = fit_boost(Xtr, P["y_tr"], seed=seed, task=task)
    val = sweep._score(Pv, predict_scores(est, Xva, task), task)
    test = (sweep._score(P, predict_scores(est, Xte, task), task)
            if verify and Xte is not None else None)
    return val, test, ""


def rescore_file(csv_path, args):
    """One metrics_*.csv -> its val_metrics_*.csv. Returns (rows written, verify gaps)."""
    ds, family, nodes, mol, variant = ht.parse_name(csv_path.stem)
    if family not in ht.REGIME_OF:
        return 0, []
    regime = ht.REGIME_OF[family]
    if args.dataset and ds not in args.dataset:
        return 0, []
    if args.regime and regime not in args.regime:
        return 0, []
    task = sweep.TASK[ds]
    src = pd.read_csv(csv_path)
    if "seed" not in src.columns:
        src = src.assign(seed=42)
    src = src[src["status"].astype(str).eq("ok")] if "status" in src.columns else src
    if args.alphas is not None:
        keep = src["alpha"].isna() | src["alpha"].astype(float).round(6).isin(
            [round(float(a), 6) for a in args.alphas])
        src = src[keep]
    if src.empty:
        return 0, []

    out = csv_path.parent / (VAL_PREFIX + csv_path.stem.replace("metrics_", "") + ".csv")
    done, rows = set(), []
    if out.exists() and not args.force:
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["arm"], "" if pd.isna(r["alpha"]) else round(float(r["alpha"]), 6),
                 int(r["fold"]), int(r["seed"])) for _, r in prev.iterrows()}
    todo = [r for _, r in src.iterrows()
            if (r["arm"], "" if pd.isna(r["alpha"]) else round(float(r["alpha"]), 6),
                int(r["fold"]), int(r["seed"])) not in done]
    print(f"\n{csv_path.name}  [{ds}/{regime}, mol {mol}, nodes {nodes}]  "
          f"{len(todo)} cell(s) to score ({len(done)} already in {out.name})")
    if not todo or args.dry_run:
        return 0, []

    variant = variant or sweep.DEFAULT_VARIANT[ds]
    ns = _ns(ds, mol, variant, args.pool_fold)
    data = sweep._prepare(ds, ns)
    pairs, molemb, protemb, _ = data
    dumps = dumps_dir(csv_path)
    of_record = sweep.TASK[ds]
    key_metric = "R2" if of_record == "regression" else "AUROC"

    written, gaps, cache = 0, [], {}
    t0 = time.time()
    for fold in sorted({int(r["fold"]) for r in todo}):
        if fold not in cache:
            cache[fold] = sweep._fold_prep(ds, regime, fold, ns, data)
        P = cache[fold]
        V = val_arrays(P, pairs, molemb, protemb)
        if not len(V["y"]):
            print(f"  fold {fold}: the val split is EMPTY -- nothing to choose on")
            continue
        for r in [x for x in todo if int(x["fold"]) == fold]:
            val, test, note = score_cell(r, P, V, dumps, task, args.verify)
            if val is None:
                print(f"  skip {r['arm']} a={r['alpha']} f{fold} s{r['seed']}: {note}")
                continue
            rec = dict(r)
            rec.update(val)
            rec["split"] = "val"
            rec["n_val"] = int(len(V["y"]))
            rows.append(rec)
            written += 1
            if test is not None and key_metric in test and key_metric in r.index:
                d = abs(float(test[key_metric]) - float(r[key_metric]))
                gaps.append(d)
                if d > 1e-4:
                    print(f"  !! {r['arm']} a={r['alpha']} f{fold} s{r['seed']}: refit "
                          f"reproduces test {key_metric} to {d:.2e} -- the "
                          f"reconstruction is NOT exact, do not trust this val number")
            tmp = out.with_suffix(".tmp.csv")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            tmp.replace(out)
            # per CELL, not per fold: a fold is fourteen XGBoost fits and on m2or that
            # is minutes of silence, which reads exactly like a hang
            el = time.time() - t0
            a = "" if pd.isna(r["alpha"]) else f" a={float(r['alpha']):.2f}"
            print(f"  [{written}/{len(todo)} {_fmt(el)} "
                  f"ETA {_fmt(el / written * (len(todo) - written))}] "
                  f"f{fold} {r['arm']}{a}  val n={len(V['y'])}", flush=True)
    return written, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/graph/v9_node_dial",
                    help="the sweep directory holding metrics_*.csv and dumps/")
    ap.add_argument("--dataset", nargs="+", default=None, choices=["cc", "hc", "m2or"])
    ap.add_argument("--regime", nargs="+", default=None,
                    choices=["transductive", "inductive"])
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="only these dial positions (the reference arms are always "
                         "included -- a choice needs something to be measured against)")
    ap.add_argument("--pool-fold", type=int, default=1)
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip predicting test as well. Verification is what turns "
                         "'the refit should reproduce the sweep' into a number, and it "
                         "costs one predict per cell -- keep it on unless you are "
                         "rerunning something already verified")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be scored and stop. Run this FIRST: it is how "
                         "you find out whether the embedding dumps are actually there")
    ap.add_argument("--force", action="store_true", help="rescore cells already written")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.is_absolute() and not root.exists():
        root = _root / args.root
    files = sorted(root.glob("metrics_*.csv"))
    if not files:
        raise SystemExit(f"no metrics_*.csv under {root}")

    total, gaps = 0, []
    for p in files:
        n, g = rescore_file(p, args)
        total += n
        gaps += g
    print(f"\n{total} val cell(s) written under {root}")
    if gaps:
        print(f"verification: refit reproduces the recorded test metric to "
              f"max {max(gaps):.2e} over {len(gaps)} cell(s)"
              + ("  -- exact" if max(gaps) <= 1e-4 else
                 "  -- NOT exact, read the !! lines above"))
    if total:
        print("\nread with:\n"
              "  from scripts.analysis.alpha_grid import load\n"
              "  load(root=..., nodes='nodedial', split='val')\n"
              "  python scripts/analysis/alpha_choice.py --select-on val")


if __name__ == "__main__":
    main()
