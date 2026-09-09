#!/usr/bin/env python
"""v8: sweep the alpha gate on Carey / Hallem-Carlson, both primary regimes.

WHAT THE GATE IS. The receptor vector the boost eventually reads is

    z_prot = (1 - alpha) * frozen_SVD(ESM)  +  alpha * signed_graph(...)

with both branches RMS-normalised, the structural branch fit on TRAIN receptors and
frozen, and alpha fixed before training and used unchanged at inference. See
`orbind.gnn_extractor.GnnSignedExtractor.alpha`.

WHY IT EXISTS. In the historical graph the only path from ESM to the receptor
embedding is trainable, and 900 epochs of binding loss empty it: measured on the
(k, phi) grid, the refined receptor cloud sits AT the permutation null against ESM
(z = +0.4 on CC) while scoring z = +11 against the response profile, in every cell of
a two-axis ablation. Both of those axes removed information; neither could pull back
toward structure, because nothing in the architecture ever pulled that way. The
structural end was reachable only by not training. A frozen branch is a path the
optimizer cannot drain, which is what turns "structure vs function" from a pair of
ablations into one dial with two known ends:

    alpha = 0   the receptor cloud IS ESM's geometry -- exactly, on the train span,
                since the branch is an isometry there (uncentered, so cosines and
                not merely distances are preserved)
    alpha = 1   the historical graph, up to one global scalar no geometry sees

WHAT IS MEASURED. Every cell reports both halves at once:

  PREDICTION  the pipeline's own head -- `train_boost` on [ z_prot || raw molecule ],
              the `cls+mol` combo, R2/RMSE/MAE/Pearson/Spearman on the fold's test
              rows. Alongside it two references that are not the graph at all:
              `boost_full` = [ raw ESM || molecule ] (what the graph must beat) and
              `naive` = the constant train mean (R2's honest zero).
  GEOMETRY    the three second-order readouts (RSA, CCA, Procrustes) of the receptor
              cloud against raw ESM and against the TRAIN response profile, each with
              its own permutation null, so the dial can be watched moving. Train
              columns only -- the diagnostic must not see test odorants.

`graph_legacy` (alpha=None, the pre-v8 model) is run as its own arm so the
comparison to the previous implementation is a row in the same table under the same
folds, not a number quoted from another run.

DATASETS. cc/hc are the complete continuous insect panels: regression, R2/RMSE/MAE/
Pearson/Spearman, upstream's `rand` and our stratified `our_inductive` folds. m2or is
the sparse binary pool: classification, AUROC/AUPRC/MCC/F1, LORaX folds 1-5 for
transductive and cold-molecule seeds 42-46 for `inductive_molecule_v5`. The MP-edge
variant defaults per dataset -- q99/greedy on m2or, where coverage is heavy-tailed and
the quantile picks a hub core, q0/coverage on the insects, whose complete matrix leaves
the quantile nothing to cut -- and on m2or it is part of the filename, so both coexist.

REPEATS ARE FOLDS x SEEDS. A fold changes WHICH ROWS are held out; a seed changes the
model's own draw -- graph init, the bag, and the head's subsample/colsample. Five folds
at one seed bound the first and say nothing about the second, and the second was
measured at +/-0.007 (graph, GPU scatter) to ~0.02 (head lottery), which is the size of
the effects being claimed. So an error bar meant for print needs both axes. Seed 42 is
the ensembler's own default and the seed every reported number stands on; extra seeds
extend that row rather than replacing it.

    # the headline table: no gate, both node kinds, 5 seeds x 5 folds
    python scripts/modeling/train/run_alpha_gate_sweep.py --dataset cc hc m2or \\
        --mol-source chemberta --seeds 42 43 44 45 46 --no-gate \\
        --max-parallel 4 --gpus 0 1 2 3
    # the dial: one seed, dense alpha, one-hot nodes (where alpha is honest)
    python scripts/modeling/train/run_alpha_gate_sweep.py --dataset cc hc m2or \\
        --mol-source chemberta --nodes onehot --no-legacy \\
        --alphas 0 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

WHAT ONE RUN WRITES -- three files, one computation:

    metrics_*.csv       the TEST rows, in the schema every existing reader parses.
    val_metrics_*.csv   the same fitted head scored on VALIDATION. This is the only
                        split alpha may be CHOSEN on: alpha is a hyperparameter, fixed
                        before training and used unchanged at inference, so picking it
                        by the score on the folds the paper reports is selection on the
                        test set. See scripts/analysis/alpha_choice.py --select-on val.
    records_*.csv       EVERY row -- train, val and test -- with per-cell wall clock
                        (graph and head separately), split sizes, the geometry readouts,
                        and provenance (git commit, host, run start). The full dump; the
                        other two are views of it, so they cannot disagree.

Scoring all three splits costs two extra predicts per cell and no extra training: the
head is fit ONCE and asked three times. Refitting per split would make each column a
different model and the val number would stop being about the reported one.

SEEDING THE GRAPH (`--seed-graph`). Off by default, and the default is the historical
behaviour: `_train_one` never applied `hp["seed"]`, so the graph's weights came from
torch's global RNG -- drawn from OS entropy at first use, then advanced by whatever else
that worker process trained first. That is why the `gate alpha=1` and `graph_legacy`
arms, which are provably the same computation on the same input (rho=1 returns the
embedding file unchanged; legacy switches the dial off), do not land on the same number:
their gap is a free measurement of the init lottery, pooled at 0.006 on both metric
scales. With the flag the lottery is gone -- though not bit-determinism, since the
message passing's scatter-add on CUDA is atomic and reorders between runs.

It is opt-in because turning it on changes every number already in results/graph/. A run
with it is a NEW series, not more folds of an old one, and every row carries a
`seeded_graph` column saying which kind it is.

Resumable: a finished (arm, alpha, fold, seed) is skipped -- the test row is written
last, so a cell killed mid-way is recomputed rather than left with train and val only.
The parent process is the sole writer of all three files. A records file written before
seeds existed is read as seed 42, so an old series is extended, not recomputed. A
directory holding a metrics file but NO records file is refused: it predates the dump,
and resuming into it would produce a records file covering only the cells that happened
to be missing. Point --out somewhere new, --force to recompute in place, or run
scripts/analysis/val_rescore.py to add val to the old run without retraining.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind.baselines import fit_boost, predict_scores            # noqa: E402
from orbind.dataset import (                                       # noqa: E402
    METRICS as METRIC_FNS, METRICS_FULL, load_npz_dict)
from orbind.gnn_extractor import GnnSignedExtractor                # noqa: E402
from orbind.regimes import full_full_pairs, load_split           # noqa: E402
from orbind.regimes_ofm import ofm_indices, ofm_pairs              # noqa: E402
from orbind.gpu_dashboard import (                                 # noqa: E402
    Dashboard, plan_placement, visible_gpus)
from scripts.modeling.analysis.mechanism_holdout import (          # noqa: E402
    GEOMETRY, geometry_nulls)

# What each regime name means per dataset. On the insects `rand` is i.i.d. and
# `our_inductive` is the stratified cold-molecule family -- upstream's `scaf` is
# cold-molecule too but its fold 1 is degenerate (test sd 0.215), so a sweep must not
# be read off it. On M2OR they are the LORaX folds and the v5 cold-molecule split, the
# two the paper's tables use. The ligand-class holdout ("special-inductive") is a
# different script: scripts/modeling/analysis/mechanism_holdout.py
FAMILY = {"cc": {"transductive": "rand", "inductive": "our_inductive"},
          "hc": {"transductive": "rand", "inductive": "our_inductive"},
          "m2or": {"transductive": "transductive",
                   "inductive": "inductive_molecule_v5"}}
# M2OR's repeats are cold-molecule SEEDS, not folds, for the inductive regime.
REPEATS = {"cc": {}, "hc": {},
           "m2or": {"transductive": [1, 2, 3, 4, 5],
                    "inductive": [42, 43, 44, 45, 46]}}
TASK = {"cc": "regression", "hc": "regression", "m2or": "classification"}
TASK_METRICS = {"regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
                "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
# The MP-edge variants. On M2OR coverage is heavy-tailed and q99+greedy picks a hub
# core -- that is the graph every M2OR number in the paper stands on. On the complete
# insect matrices the quantile has nothing to cut, so q0/coverage is theirs.
VARIANTS = {"q99greedy": dict(q=0.99, criterion="greedy_pair_cover",
                              k_mode="coverage_quantile"),
            "q0cov": dict(q=0.0, criterion="coverage", k_mode="coverage_quantile")}
DEFAULT_VARIANT = {"cc": "q0cov", "hc": "q0cov", "m2or": "q99greedy"}
GEOMS = ["rsa", "cca", "procrustes"]
REFS = ["esm", "fun"]
# The molecule source is BOTH the boost's molecular half and the graph's molecule node
# features, so switching it moves every arm at once -- which is the point: the paper
# reports GNN-vs-boost on more than one source. ChemBERTa is the primary one; GIN stays
# because the earlier series was run on it. Only `gin` is untagged in the filename, and
# only because the files already on disk were written before this flag existed; every
# row also carries a `mol_source` column, so nothing has to be inferred from a name.
MOL_SOURCES = {"chemberta": {None: "data/embeddings/molecules/chemberta_77m_{ds}.npz"},
               "gin": {None: "data/embeddings/molecules/gin_supervised_contextpred_{ds}.npz",
                       "m2or": "data/embeddings/molecules/"
                               "gin_supervised_contextpred_all_m2or.npz"},
               # ECFP4, 2048 binary bits from `embed_molecules_ecfp.py`. It is the
               # third source the paper reports GNN-vs-boost on, and the only one
               # that is not learned -- so a conclusion that holds here does not
               # depend on any pretrained molecular model. No per-dataset special
               # case: one file per dataset, all written by the same script.
               "ecfp": {None: "data/embeddings/molecules/ecfp_{ds}.npz"}}
# M2OR's protein file carries no dataset suffix; cc/hc have one each.
PROT_SOURCE = {None: "data/embeddings/proteins/esm1b_650m_mean_{ds}.npz",
               "m2or": "data/embeddings/proteins/esm1b_650m_mean.npz"}
UNTAGGED_MOL = "gin"


# Which held-out set a row is scored on. `train` is in the dump because an overfit
# head is invisible in a test column and obvious next to a train one; it costs one
# predict. `val` is what alpha may be CHOSEN on -- see scripts/analysis/alpha_choice.py.
SPLITS = ["train", "val", "test"]
# (labels, receptor ids, molecule ids) per split, as `_fold_prep` names them.
SPLIT_KEYS = {"train": ("y_tr", "rec_tr", "mol_tr"),
              "val": ("y_va", "rec_va", "mol_va"),
              "test": ("y_te", "rec_te", "mol_te")}


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_root,
                              capture_output=True, text=True, timeout=5,
                              check=True).stdout.strip()
    except Exception:                       # noqa: BLE001 -- provenance is not the job
        return ""


_PROV = None


def provenance(args):
    """Stamped on every record row. A dump that cannot say which code and which
    embedding files produced it is a table of numbers, not evidence -- and six months
    from now the difference matters more than the numbers do."""
    global _PROV
    if _PROV is None:
        _PROV = dict(commit=_git_commit(), host=socket.gethostname(),
                     started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return dict(_PROV)


def splits_wanted(args, P):
    """The splits to score. An empty val is normal on some split families and must not
    become a row of NaN that later reads as a failure."""
    out = ["test"]
    if len(P["va"]):
        out.insert(0, "val")
    if getattr(args, "score_train", True):
        out.insert(0, "train")
    return out


def paths(ds, args):
    src = MOL_SOURCES[args.mol_source]
    mp = args.mol_embeddings or src.get(ds, src[None])
    pp = args.prot_embeddings or PROT_SOURCE.get(ds, PROT_SOURCE[None])
    return pp.format(ds=ds), mp.format(ds=ds)


def repeats(ds, regime, args):
    """Which repeats to run. `--folds` names them literally and therefore cannot be
    mixed across datasets -- m2or/inductive's repeats are the cold-molecule SEEDS
    42-46, so `--folds 1 2 3` would ask it for splits that do not exist. `--n-repeats`
    is the portable way to say "a cheaper slice": it takes the first N of whatever this
    (dataset, regime) actually uses."""
    if args.folds:
        return list(args.folds)
    reps = REPEATS[ds].get(regime, [1, 2, 3, 4, 5])
    return reps[:args.n_repeats] if args.n_repeats else reps


def _fmt(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


def _mat(emb, keys):
    if not len(keys):
        # `np.stack([])` raises. An empty split is legal -- some families ship no val --
        # and it must arrive downstream as a zero-row matrix of the right width, not as
        # an exception thrown from fold preparation for every arm at once.
        d = int(np.asarray(next(iter(emb.values()))).shape[-1])
        return np.empty((0, d), dtype=np.float32)
    return np.stack([emb[k] for k in keys]).astype(np.float32)


def _prepare(ds, args):
    """Pool + embeddings + coverage mask, once per process."""
    pairs = ofm_pairs(ds) if ds != "m2or" else full_full_pairs(pool_fold=args.pool_fold)
    pp, mp = paths(ds, args)
    mol = load_npz_dict(str(_root / mp))
    prot = load_npz_dict(str(_root / pp))
    cov = pairs["inchikey"].isin(mol).to_numpy() & pairs["receptor"].isin(prot).to_numpy()
    return pairs, mol, prot, cov


def _split(ds, regime, rep):
    """(train, val, test) row indices. M2OR keeps its splits in a persisted npz keyed
    by regime+repeat; the insects derive theirs from upstream's fold files."""
    if ds == "m2or":
        return load_split(FAMILY[ds][regime], int(rep))
    return ofm_indices(ds, FAMILY[ds][regime], int(rep))


def _fold_prep(ds, regime, fold, args, data):
    """Everything a fold needs, shared by every arm run on it."""
    pairs, mol, prot, cov = data
    ik, rc = pairs["inchikey"].to_numpy(), pairs["receptor"].to_numpy()
    lab = pairs["label"].to_numpy(np.float32)
    tr, va, te = (np.asarray(a)[cov[np.asarray(a)]]
                  for a in _split(ds, regime, fold))
    # Receptor order for the geometry readouts: the graph's own universe order, so
    # z_prot rows line up with the reference clouds without a second lookup.
    order = list(pd.unique(np.concatenate([rc[tr], rc[va], rc[te]])))
    rank = {r: i for i, r in enumerate(order)}
    # The functional reference: the TRAIN response matrix, receptors x train
    # odorants, row-centred. No test odorant enters it -- a diagnostic that peeked
    # would report the leak as a success.
    #
    # CAUTION on M2OR. There the matrix is SPARSE, so most cells are "not assayed"
    # rather than "no response", and the imputation below writes the global mean into
    # them. What is left is partly WHICH PAIRS WERE MEASURED -- assay design, not
    # binding. The `tested mask` control in mechanism_holdout measured that share at
    # 76-103% of the trained graph's own gain on M2OR. So on M2OR read the `fun`
    # geometry columns as an upper bound contaminated by design, and lean on the
    # prediction columns instead. On the complete insect matrices this does not arise.
    tr_mols = list(pd.unique(ik[tr]))
    mrank = {m: i for i, m in enumerate(tr_mols)}
    R = np.full((len(order), len(tr_mols)), np.nan)
    R[[rank[r] for r in rc[tr]], [mrank[m] for m in ik[tr]]] = lab[tr]
    R = np.nan_to_num(R, nan=float(np.nanmean(R)) if np.isfinite(R).any() else 0.0)
    return {"pairs": pairs, "tr": tr, "va": va, "te": te,
            "y_tr": lab[tr], "y_va": lab[va], "y_te": lab[te],
            # group ids for the within-receptor / within-molecule scores, and the
            # TRAIN response for the binarisation cuts -- which must never be fit
            # on test rows, or the label definition itself sees the held-out data
            "rec_tr": rc[tr], "mol_tr": ik[tr],
            "rec_va": rc[va], "mol_va": ik[va],
            "rec_te": rc[te], "mol_te": ik[te],
            "Xm_tr": _mat(mol, ik[tr]), "Xm_va": _mat(mol, ik[va]),
            "Xm_te": _mat(mol, ik[te]),
            "Xp_tr": _mat(prot, rc[tr]), "Xp_va": _mat(prot, rc[va]),
            "Xp_te": _mat(prot, rc[te]),
            "order": order, "prank": rank,
            "ref": {"esm": _mat(prot, order).astype(np.float64),
                    "fun": R - R.mean(1, keepdims=True)}}


def _geometry(Z, P, n_perm):
    """RSA / CCA / Procrustes of the receptor cloud against both reference clouds,
    each as a raw value and as z against its own permutation null. The nulls are
    what make three measures on different scales comparable at all."""
    out = {}
    for ref, M in P["ref"].items():
        stats = geometry_nulls(Z, M, n_perm) if n_perm else {}
        for g in GEOMS:
            v = float(GEOMETRY[g](Z, M))
            out[f"{g}_{ref}"] = v
            if g in stats:
                mu, sd = stats[g]
                out[f"{g}_{ref}_z"] = float((v - mu) / sd) if sd > 1e-12 else np.nan
    return out


def _row(arm, alpha, fold, seed, **rest):
    return {"arm": arm, "alpha": np.nan if alpha is None else float(alpha),
            "fold": int(fold), "seed": int(seed), "status": "ok", **rest}


def _score(P, pred, task, full=True):
    """Every metric the task admits, not just the headline four or five.

    On the continuous insect panels that means the pooled regression scores plus
    the binary battery after binarising the response at `rec0` -- above each
    receptor's own TRAIN centre, the same convention the graph's edge signs use.
    Twelve columns instead of five. On m2or it adds precision and recall, which
    were always computed and then dropped on the way into the row.

    Which of them a table reads is a decision for the reader; recomputing any of
    them is a decision for a GPU, so they are all written now. `TASK_METRICS`
    stays the short list the progress line shows."""
    if not full:
        return {k: float(v) for k, v in METRIC_FNS[task](P["y_te"], pred).items()}
    if task == "regression":
        m = METRICS_FULL[task](P["y_te"], pred, rec_te=P["rec_te"], mol_te=P["mol_te"],
                               y_tr=P["y_tr"], rec_tr=P["rec_tr"])
    else:
        m = METRICS_FULL[task](P["y_te"], pred, rec_te=P["rec_te"], mol_te=P["mol_te"])
    return {k: float(v) for k, v in m.items()}


def _score_split(P, pred, task, split):
    """`_score` on one split. The metric battery is written against the "test" keys, so
    the split's own labels and group ids are swapped in rather than the battery being
    duplicated -- one definition of every metric, for every split."""
    yk, rk, mk = SPLIT_KEYS[split]
    Q = dict(P)
    Q["y_te"], Q["rec_te"], Q["mol_te"] = P[yk], P[rk], P[mk]
    return _score(Q, pred, task)


def _dump(args, ds, regime, arm, alpha, fold, seed, **arrays):
    """One npz per cell, keyed by everything that changes what is in it.

    The key matters more than it looks. The extractor's own checkpointing writes
    `gnn_{name}_model{m}.pt`, and `name` is "cls" in every cell of this grid --
    one directory would have the first cell's weights silently reused by the
    other 719. Keying on (arm, alpha, fold, seed) under the run's own stem makes
    that class of mistake impossible here."""
    if not arrays:
        return
    a = "None" if alpha is None else f"{float(alpha):g}"
    d = (out_path(ds, regime, args).parent / "dumps"
         / out_path(ds, regime, args).stem.replace("metrics_", ""))
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{arm}_a{a}_f{fold}_s{seed}.npz", **arrays)


def _baseline_rows(fold, seed, P, args):
    """The two references the graph is judged against, on this fold's own rows.

    `naive` is the constant train mean: under regression that is upstream's own naive
    baseline and the only thing that makes an R2 near zero readable; under
    classification the same constant is the class prevalence, i.e. AUROC 0.5 and the
    AUPRC floor. `naive` ignores the seed -- a constant has no randomness -- but is
    still written per seed so every arm has the same row count.

    On the seed itself: 42 is the default because under `--regime ofm` (and full_full)
    `train_ensemble_boost.py` calls `run_ensemble` WITHOUT a `seed=`, so it keeps that
    function's own default of 42 and hands the same 42 to the extractor and to
    `fit_boost` on every fold -- only the split moves. Every cc/hc number in the
    tables was produced that way, and `fit_boost` draws subsample=0.8/colsample=0.8
    from `random_state`, so a different seed shifts R2 by up to 0.045 on one fold.
    That is what made this sweep's first `boost_full` read 0.501 against the table's
    0.516. Extra seeds therefore ADD to the seed-42 row, they do not replace it."""
    task = TASK[args._ds]
    rows = []
    want = splits_wanted(args, P)
    X = {s: np.concatenate([P[f"Xp_{k}"], P[f"Xm_{k}"]], 1)
         for s, k in (("train", "tr"), ("val", "va"), ("test", "te")) if s in want}
    t0 = time.time()
    est = fit_boost(X["train"] if "train" in X
                    else np.concatenate([P["Xp_tr"], P["Xm_tr"]], 1),
                    P["y_tr"], seed=seed, task=task)
    t_head = time.time() - t0
    base = dict(n_receptors=len(P["order"]), mol_source=args.mol_source,
                t_graph=0.0, t_head=t_head, **provenance(args))
    for s in want:
        rows.append(_row("boost_full", None, fold, seed, split=s,
                         n_rows=len(P[SPLIT_KEYS[s][0]]), **base,
                         **_score_split(P, predict_scores(est, X[s], task), task, s)))
    for s in want:
        y = P[SPLIT_KEYS[s][0]]
        const = np.full(len(y), float(P["y_tr"].mean()), dtype=np.float32)
        rows.append(_row("naive", None, fold, seed, split=s, n_rows=len(y),
                         **(base | dict(t_head=0.0)),
                         **_score_split(P, const, task, s)))
    pred = predict_scores(est, X["test"], task)
    # boost has no receptor cloud, but its per-pair predictions are half of every
    # stratified comparison against the graph, so they are dumped alongside
    if args.dump_predictions:
        _dump(args, args._ds, args._regime, "boost_full", None, fold, seed,
              pred=np.asarray(pred, np.float32), y_true=np.asarray(P["y_te"], np.float32),
              receptor=np.asarray(P["rec_te"], dtype=object).astype("U"),
              inchikey=np.asarray(P["mol_te"], dtype=object).astype("U"))
    return rows


def _graph_row(arm, alpha, fold, seed, ds, P, args):
    """One trained graph -> the cls+mol boost feature -> metrics + geometry."""
    task = TASK[ds]
    pp, mp = paths(ds, args)
    dial = getattr(args, "dial", "gate")
    knob = ({"alpha": alpha} if dial == "gate" else
            {"prot_mix": alpha, "mix_seed": args.mix_seed,
             "mix_renorm": not args.no_mix_renorm})
    ext = GnnSignedExtractor(
        name="cls", protein_path=pp, molecule_path=mp, **VARIANTS[args._variant],
        task=task, n_models=args.n_models, epochs=args.epochs, emit="prot",
        # the v9 dial replaces the node features itself, so `onehot_nodes` must be off
        # there -- its rho=0 end IS the one-hot arm, and the extractor refuses both
        onehot_nodes=(args.nodes == "onehot" and dial == "gate"),
        deterministic_init=args.seed_graph, **knob)
    t0 = time.time()
    Zp_tr, Zp_va, Zp_te = ext.fit_transform(P["pairs"], P["tr"], P["va"], P["te"], seed)
    t_graph = time.time() - t0
    want = splits_wanted(args, P)
    Z = {"train": Zp_tr, "val": Zp_va, "test": Zp_te}
    X = {s: np.concatenate([Z[s], P[f"Xm_{k}"]], 1)
         for s, k in (("train", "tr"), ("val", "va"), ("test", "te")) if s in want}
    t1 = time.time()
    # ONE fit, scored on every split. Refitting per split would be a different model
    # per column and the val number would stop being about the reported one.
    est = fit_boost(np.concatenate([Zp_tr, P["Xm_tr"]], 1), P["y_tr"],
                    seed=seed, task=task)
    t_head = time.time() - t1
    pred = predict_scores(est, X["test"], task)
    # One receptor, one row: the per-pair features repeat the receptor vector, so
    # collapse back to the universe order the reference clouds are in. All three
    # splits are walked -- a receptor that appears only in val would otherwise be
    # missing and silently shift every row below it.
    seen = {}
    for idx, part in ((P["tr"], Zp_tr), (P["va"], Zp_va), (P["te"], Zp_te)):
        rc = P["pairs"]["receptor"].to_numpy()[idx]
        for i, r in enumerate(rc):
            seen.setdefault(r, part[i])
    missing = [r for r in P["order"] if r not in seen]
    Z = np.stack([seen[r] for r in P["order"] if r in seen]).astype(np.float64)
    geo = {} if missing else _geometry(Z, P, args.n_perm)
    dump = {}
    if args.dump_embeddings and not missing:
        # the receptor cloud the geometry was measured on, in universe order. 1.3 MB
        # on m2or, 50 KB on the insects -- against ~6 MB for the weights, and this is
        # what every downstream geometry question actually consumes
        dump |= dict(z_prot=Z.astype(np.float32),
                     receptors=np.asarray(P["order"], dtype=object).astype("U"))
    if args.dump_predictions:
        dump |= dict(pred=np.asarray(pred, np.float32),
                     y_true=np.asarray(P["y_te"], np.float32),
                     receptor=np.asarray(P["rec_te"], dtype=object).astype("U"),
                     inchikey=np.asarray(P["mol_te"], dtype=object).astype("U"))
    _dump(args, ds, args._regime, arm, alpha, fold, seed, **dump)
    base = dict(n_receptors=len(P["order"]),
                k_pca=int(getattr(ext, "_k_pca", 0)), variant=args._variant,
                mol_source=args.mol_source, nodes=args.nodes, dial=dial,
                mix_seed=(args.mix_seed if dial == "nodes" else ""),
                seeded_graph=bool(args.seed_graph),
                t_graph=t_graph, t_head=t_head, **provenance(args), **geo)
    # The geometry is a property of the CELL, not of a split -- it is measured on the
    # receptor cloud, which no held-out set changes. Repeated on every row so one line
    # of the dump is self-contained.
    return [_row(arm, alpha, fold, seed, split=s, n_rows=len(P[SPLIT_KEYS[s][0]]),
                 **base, **_score_split(P, predict_scores(est, X[s], task), task, s))
            for s in want]


def _failed(job, err):
    arm, alpha, fold, seed = job
    r = _row(arm, alpha, fold, seed, split="test",
             **{k: float("nan") for k in TASK_METRICS["regression"]},
             **{k: float("nan") for k in TASK_METRICS["classification"]})
    r["status"] = f"failed: {err}"
    return r


def _run_job(job, ds, regime, args, cache, data):
    arm, alpha, fold, seed = job
    # The fold prep is seed-independent (splits, embeddings, the response matrix), so
    # it is cached by fold alone and shared by every seed a worker happens to draw.
    if fold not in cache:
        cache[fold] = _fold_prep(ds, regime, fold, args, data)
    P = cache[fold]
    if arm == "baselines":
        return _baseline_rows(fold, seed, P, args)
    return _graph_row(arm, alpha, fold, seed, ds, P, args)


def _worker(job_q, res_q, ds, regime, args, wid=0, log_dir=None):
    """Persistent GPU-pinned worker; `worker_done` from `finally` so a crash cannot
    leave the parent blocked forever waiting for a row that will never come.

    The extractor prints several lines per cell -- edge counts, the gate's rank, the
    node swap. With a dozen workers on one terminal that is the wall of log the
    dashboard exists to replace, so it is redirected to one file per worker: still
    there for a post-mortem, no longer between the reader and the numbers."""
    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(log_dir / f"worker{wid}.log", "a", buffering=1,
                          encoding="utf-8")
        sys.stderr = sys.stdout
    try:
        data, cache = _prepare(ds, args), {}
        while True:
            job = job_q.get()
            if job is None:
                break
            res_q.put(("start", (wid, job)))
            try:
                for row in _run_job(job, ds, regime, args, cache, data):
                    res_q.put(("row", (wid, row)))
            except Exception as e:                  # noqa: BLE001 -- one cell must not kill the sweep
                res_q.put(("row", (wid, _failed(job, e))))
    finally:
        res_q.put(("worker_done", wid))


def key(arm, alpha, fold, seed):
    """The resume identity of one cell. Everything that changes what is computed and
    is NOT already fixed by the filename has to be in here, or a second invocation
    reads a finished file and silently reports the first one's numbers."""
    return (arm, "" if alpha is None else round(float(alpha), 6), int(fold), int(seed))


def out_path(ds, regime, args):
    """What goes in the FILENAME and what goes in a COLUMN. Filename: only the axes a
    reader must be able to pick a series by without opening it -- the edge variant
    (m2or only, where both are live), the node features, and the molecule source. GIN
    is untagged so the files written before that flag existed stay addressable.
    Column: everything else, the seed included, so one file accumulates the whole seed
    grid and stays resumable across separate invocations."""
    variant = args.variant or DEFAULT_VARIANT[ds]
    # The v9 dial gets its own tag, and it is not optional: `alpha` means the OPPOSITE
    # thing on the two dials (v8 runs structure -> function as it rises, v9 runs
    # function -> structure), so a file that mixed them would be unreadable and would
    # look fine.
    tag = "" if getattr(args, "dial", "gate") == "gate" else "_nodedial"
    tag += "" if args.nodes == "esm" or getattr(args, "dial", "gate") == "nodes" \
        else f"_{args.nodes}"
    if args.mol_source != UNTAGGED_MOL:
        tag = f"_{args.mol_source}{tag}"
    if ds == "m2or":
        tag = f"_{variant}{tag}"
    return pathlib.Path(args.out or (_root / "results/graph/v8_alpha_gate")) / \
        f"metrics_{ds}_{FAMILY[ds][regime]}{tag}.csv"


def sibling_paths(ds, regime, args):
    """The three files one run writes, from ONE computation:

      metrics_*.csv       TEST rows, the schema every existing reader already parses
      val_metrics_*.csv   the same rows scored on VALIDATION -- what alpha may be
                          chosen on, and the same filename the old rescore produced,
                          so `alpha_grid.load(split="val")` needs no change
      records_*.csv       EVERY row: train, val and test, with timings, sizes,
                          geometry and provenance. The full dump; the other two are
                          views of it.

    Three files rather than one long frame because the first two are contracts with
    code that already exists, and the third is a contract with analysis that does not
    exist yet -- which is exactly why it carries everything.
    """
    m = out_path(ds, regime, args)
    stem = m.stem.replace("metrics_", "")
    return m, m.parent / f"val_metrics_{stem}.csv", m.parent / f"records_{stem}.csv"


def load_done(rec_out, out, args):
    """(record rows already on disk, the CELLS they cover).

    Resume reads the RECORDS file, because that is the one holding every split: a cell
    present in metrics_*.csv but absent from records_*.csv has no val row and no train
    row, so treating it as done would leave the dump permanently ragged.

    That is also why an old directory is refused rather than silently extended. A sweep
    run before this file existed has test rows and nothing else; continuing into it
    would produce a records file covering only the cells that happened to be missing,
    which is worse than either recomputing or starting clean.

    A file written before seeds existed holds exactly one seed, 42 -- the ensembler's
    default, which is what those rows were produced at. Reading them as 42 is what lets
    an old series be EXTENDED with more seeds instead of recomputed.
    """
    if args.force:
        return [], set()
    if not rec_out.exists():
        if out.exists():
            raise SystemExit(
                f"{out.name} exists but {rec_out.name} does not: this directory holds a "
                f"sweep from before the full dump, so its cells have no val or train "
                f"rows.\n  Point --out at a new directory to build the dump cleanly, or "
                f"pass --force to recompute in place (which OVERWRITES {out.name}).\n"
                f"  To add val to the old run without recomputing anything:\n"
                f"    python scripts/analysis/val_rescore.py --root {out.parent}")
        return [], set()
    prev = pd.read_csv(rec_out)
    if "seed" not in prev.columns:
        prev["seed"] = 42
    prev = prev.assign(seed=prev["seed"].fillna(42).astype(int))
    if "split" not in prev.columns:
        prev = prev.assign(split="test")
    # `--seed-graph` is the one axis that is NOT in the filename, so nothing but this
    # check stands between a resume and a records file holding two different kinds of
    # row under one name. Half a grid with a seeded graph init and half without is not
    # a grid: the seed column would mean something different depending on when the row
    # was computed, and no reader could tell.
    if "seeded_graph" in prev.columns:
        was = set(prev["seeded_graph"].dropna().astype(bool))
        now = bool(args.seed_graph)
        if was and was != {now}:
            raise SystemExit(
                f"{rec_out.name} holds rows with seeded_graph={sorted(was)} and this "
                f"run has --seed-graph {'on' if now else 'off'}.\n"
                f"  Those are two different series: with the flag the graph's init "
                f"comes from --seeds, without it from torch's global RNG.\n"
                f"  Add or drop --seed-graph to match, or point --out at a new "
                f"directory.")
    cells = {key(r["arm"], None if pd.isna(r["alpha"]) else r["alpha"],
                 r["fold"], r["seed"])
             for _, r in prev[prev["split"].astype(str) == "test"].iterrows()}
    return prev.to_dict("records"), cells


def plan(reps, args, done):
    """The cells still to run, seed-major so a partial sweep is a whole seed rather
    than a ragged slice of every one."""
    jobs = []
    for s in args.seeds:
        for f in reps:
            if not all(key(a, None, f, s) in done for a in ("boost_full", "naive")):
                jobs.append(("baselines", None, f, s))
            if args.baselines_only:
                continue
            if args.legacy and key("graph_legacy", None, f, s) not in done:
                jobs.append(("graph_legacy", None, f, s))
            for a in (args.alphas if args.gate else []):
                if key("gate", a, f, s) not in done:
                    jobs.append(("gate", float(a), f, s))
    return jobs


def workers_wanted(args, gpus):
    """How many trainings run at once.

    `--per-gpu` is the flag to reach for: a single graph training leaves most of a
    modern card idle, both in memory and in occupancy, so the useful question is how
    many fit on one card rather than how many cards there are. `--max-parallel` still
    works and still means a total, for the runs already written against it."""
    if args.per_gpu:
        return max(1, args.per_gpu * max(len(gpus), 1))
    return max(1, args.max_parallel)


def sweep(ds, regime, args, dash=None):
    args._ds, args._variant = ds, (args.variant or DEFAULT_VARIANT[ds])
    args._regime = regime
    reps = repeats(ds, regime, args)
    out, val_out, rec_out = sibling_paths(ds, regime, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, done = load_done(rec_out, out, args)
    jobs = plan(reps, args, done)

    def flush():
        """One stream, three views. Written whenever a cell completes, so a kill loses
        at most the cell in flight and never leaves the three disagreeing.

        Defined before the early return on purpose: reopening a finished directory then
        rebuilds metrics_/val_metrics_ from the records file, which is the repair for a
        run killed between the two writes."""
        if not rows:
            return
        rec = pd.DataFrame(rows)
        if "split" not in rec.columns:
            rec = rec.assign(split="test")
        for path, want in ((out, "test"), (val_out, "val"), (rec_out, None)):
            d = rec if want is None else rec[rec["split"].astype(str) == want]
            if want is not None:
                d = d.drop(columns=["split"])
            if d.empty:
                continue
            tmp = path.with_suffix(".tmp.csv")
            d.to_csv(tmp, index=False)
            tmp.replace(path)

    pp, mp = paths(ds, args)
    ends = ("a=0 structure alone -> a=1 the graph alone" if args.dial == "gate"
            else "a=0 receptor identity alone -> a=1 the legacy graph (ESM nodes)")
    print(f"\n=== {ds.upper()} / {regime} ({FAMILY[ds][regime]}) ===\n"
          f"    dial {args.dial}: {ends}\n"
          f"    repeats {reps}  seeds {args.seeds}  alphas {args.alphas}  "
          f"legacy {args.legacy}\n"
          f"    task {TASK[ds]}  nodes {args.nodes}  variant {args._variant} "
          f"{VARIANTS[args._variant]}\n"
          f"    mol {args.mol_source}: {mp.rsplit('/', 1)[-1]}   "
          f"prot: {pp.rsplit('/', 1)[-1]}\n"
          f"    graph init {'SEEDED from --seeds' if args.seed_graph else 'unseeded (global RNG)'}"
          f"   splits scored: {'train+' if args.score_train else ''}val+test\n"
          f"    {len(jobs)} jobs ({len(done)} cells already done)  ->  {out.name}, "
          f"{val_out.name}, {rec_out.name}", flush=True)
    if not jobs:
        flush()
        return out

    heavy = sum(1 for j in jobs if j[0] != "baselines")
    t0, state = time.time(), {"n": 0}
    seen_rows = {(key(r["arm"], None if pd.isna(r.get("alpha")) else r.get("alpha"),
                     r["fold"], r["seed"]), str(r.get("split", "test")))
                 for r in rows}
    if dash is not None:
        dash.set_stage(f"{ds}/{regime} {args.mol_source}/{args.nodes} -- "
                       f"{heavy} cells, {len(done)} already on disk")

    def record(row):
        k = key(row["arm"], None if pd.isna(row["alpha"]) else row["alpha"],
                row["fold"], row["seed"])
        split = str(row.get("split", "test"))
        if (k, split) in seen_rows:
            return
        rows.append(row); seen_rows.add((k, split))
        # a CELL is finished when its test row lands: that is when the three files are
        # consistent, and it is the granularity `plan` resumes at
        if split != "test":
            return
        done.add(k)
        flush()
        if row["arm"] in ("boost_full", "naive"):
            tag = f"[{row['arm']}]"
        else:
            state["n"] += 1
            el = time.time() - t0
            tag = (f"[{state['n']}/{heavy} {_fmt(el)} "
                   f"ETA {_fmt(el / state['n'] * (heavy - state['n']))}]")
        shown = ("R2", "Spearman") if TASK[ds] == "regression" else ("AUROC", "AUPRC")
        failed = str(row["status"]).startswith("failed")
        body = (row["status"] if failed
                else " ".join(f"{m}={row[m]:+.3f}" for m in shown)
                + "".join(f"  {g}/{r}={row.get(f'{g}_{r}_z', float('nan')):+.1f}"
                          for r in REFS for g in ("rsa",)))
        a = "" if pd.isna(row["alpha"]) else f"a={row['alpha']:.2f}"
        label = f"s{row['seed']} f{row['fold']} {row['arm']}{(' ' + a) if a else ''}"
        if dash is None:
            print(f"  {label:<32} {body}  {tag}", flush=True)
        elif failed:
            dash.note(f"{ds}/{regime} {label}: {row['status'][:70]}")
        else:
            dash.last = f"{label}  {body}"
            dash.render()
        if dash is not None and row["arm"] not in ("boost_full", "naive"):
            dash.done += 1              # owned here, so the serial path counts too

    if workers_wanted(args, args.gpus or visible_gpus() or [0]) <= 1:
        data, cache = _prepare(ds, args), {}
        for job in jobs:
            try:
                for row in _run_job(job, ds, regime, args, cache, data):
                    record(row)
            except Exception as e:                  # noqa: BLE001
                record(_failed(job, e))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        job_q, res_q = ctx.Queue(), ctx.Queue()
        gpus = args.gpus or visible_gpus() or [0]
        n = min(workers_wanted(args, gpus), len(jobs))
        for j in jobs:
            job_q.put(j)
        for _ in range(n):
            job_q.put(None)
        placement = plan_placement(gpus, n)
        log_dir = out.parent / "logs" if dash is not None else None
        procs = []
        for i in range(n):
            # CUDA_VISIBLE_DEVICES is read by the child at import; the parent must set
            # it before start() and must never touch CUDA itself
            os.environ["CUDA_VISIBLE_DEVICES"] = str(placement[i])
            p = ctx.Process(target=_worker,
                            args=(job_q, res_q, ds, regime, args, i, log_dir))
            p.start(); procs.append(p)
            if dash is not None:
                dash.bind(i, placement[i])
            else:
                print(f"    worker {i} -> GPU {placement[i]}", flush=True)
        import queue as _queue
        alive, seen_done = n, set()
        while alive:
            try:
                kind, payload = res_q.get(timeout=30)
            except _queue.Empty:
                # `finally` in the worker covers an exception; it does NOT cover a
                # process killed outright. Raising --per-gpu makes that a live
                # possibility, and a blocking get() would hang the sweep on it.
                dead = [i for i, pr in enumerate(procs)
                        if not pr.is_alive() and i not in seen_done]
                for i in dead:
                    seen_done.add(i)
                    alive -= 1
                    if dash is not None:
                        dash.running.pop(i, None)
                        dash.note(f"worker {i} died without finishing "
                                  f"(exit {procs[i].exitcode}) -- its cells stay "
                                  f"unwritten and a rerun will pick them up")
                    else:
                        print(f"  !! worker {i} died (exit {procs[i].exitcode})",
                              flush=True)
                continue
            if kind == "worker_done":
                seen_done.add(payload)
                alive -= 1
            elif kind == "start":
                wid, job = payload
                arm, alpha, fold, seed = job
                lab = ("base" if arm == "baselines" else
                       (f"a{alpha:.2f}" if alpha is not None else "legacy"))
                if dash is not None:
                    dash.start_job(wid, f"{lab} f{fold}")
            else:
                wid, row = payload
                if dash is not None:
                    dash.running.pop(wid, None)   # the counter lives in record()
                record(row)
        for p in procs:
            p.join()
    flush()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["hc", "cc"],
                    choices=["hc", "cc", "m2or"],
                    help="cc/hc are the continuous insect panels (regression); m2or is "
                         "the sparse binary pool (classification, AUROC/AUPRC/MCC/F1)")
    ap.add_argument("--variant", choices=list(VARIANTS), default=None,
                    help="MP-edge variant. Default per dataset: q99greedy on m2or (the "
                         "graph its tables stand on), q0cov on the insects (their complete "
                         "matrix leaves the quantile nothing to cut). On m2or the variant "
                         "is part of the output filename, so both can coexist")
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"],
                    choices=["transductive", "inductive"],
                    help="the two PRIMARY regimes. The ligand-class holdout "
                         "(special-inductive) lives in mechanism_holdout.py")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="repeats to run. Default: folds 1-5, except m2or/inductive "
                         "whose repeats are the cold-molecule seeds 42-46")
    ap.add_argument("--n-repeats", type=int, default=None,
                    help="run only the first N repeats of whatever this (dataset, "
                         "regime) uses -- folds 1..N on most, cold-molecule seeds "
                         "42..42+N-1 on m2or/inductive. Prefer this over --folds when "
                         "several datasets are in one invocation")
    ap.add_argument("--pool-fold", type=int, default=1,
                    help="m2or only: which LORaX fold reconstructs the pool")
    ap.add_argument("--no-legacy", dest="legacy", action="store_false",
                    help="skip the pre-v8 graph arm (alpha=None)")
    ap.add_argument("--no-gate", dest="gate", action="store_false",
                    help="skip every gate arm, leaving baselines + legacy. The cheap "
                         "shape for the headline table, where the only graph rows "
                         "wanted are the pre-v8 model and (with --nodes onehot) a "
                         "single alpha")
    ap.add_argument("--mol-source", nargs="+", dest="mol_sources",
                    choices=list(MOL_SOURCES), default=["chemberta"],
                    help="molecule embeddings -- the boost's molecular half AND the "
                         "graph's molecule node features. chemberta is the primary "
                         "source; gin is kept because the earlier series ran on it and "
                         "the paper reports both. Only gin is untagged in the filename. "
                         "Several may be given: each gets its own files, so one "
                         "invocation covers the whole grid")
    ap.add_argument("--prot-embeddings", default=None,
                    help="override; {ds} is filled in per dataset")
    ap.add_argument("--mol-embeddings", default=None,
                    help="override; also the graph's MP node features. Overrides "
                         "--mol-source for the PATH but not for the filename tag")
    ap.add_argument("--no-dump-embeddings", dest="dump_embeddings",
                    action="store_false",
                    help="skip the receptor cloud dump. It is ~1.3 MB per m2or cell "
                         "and 50 KB per insect one -- ~320 MB for the whole grid -- "
                         "and it is what every geometry question asked later needs, "
                         "so the default is to keep it")
    ap.add_argument("--no-dump-predictions", dest="dump_predictions",
                    action="store_false",
                    help="skip the per-pair prediction dump. ~50 MB for the grid, and "
                         "without it no analysis can be stratified by receptor or by "
                         "odorant after the fact -- only a rerun can")
    ap.add_argument("--n-models", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations per geometry null; 0 skips the nulls")
    ap.add_argument("--dial", choices=["gate", "nodes"], default="gate",
                    help="WHICH KNOB --alphas moves.\n"
                         "  gate  (v8) the graph's OUTPUT against a frozen ESM branch: "
                         "z_prot = (1-a)*frozen_SVD(ESM) + a*graph. a=0 is structure "
                         "alone, a=1 the graph alone. Use with --nodes onehot, which is "
                         "what makes a an honest fraction of structure.\n"
                         "  nodes (v9) the graph's INPUT: x_prot = mu + a*centred(ESM) "
                         "+ (1-a)*centred(one fixed near-orthogonal vector per "
                         "receptor). a=1 IS the legacy graph, a=0 is a graph over "
                         "receptor identity alone. NOTE THE DIRECTION IS REVERSED "
                         "relative to the gate, and --nodes is ignored because this "
                         "dial sets the node features itself")
    ap.add_argument("--mix-seed", type=int, default=0,
                    help="--dial nodes: which draw of the identity vectors. Not the "
                         "model seed -- who a receptor is should not change with the "
                         "training run -- but sweeping it checks that no result rests "
                         "on one lucky set of directions")
    ap.add_argument("--no-mix-renorm", action="store_true",
                    help="--dial nodes: skip the rescaling that holds the mixture's "
                         "centred spread constant. Off, the middle of the dial is "
                         "quieter than both ends by sqrt(a^2+(1-a)^2) and any dip there "
                         "is an artefact of the parameterisation -- which is what this "
                         "flag exists to demonstrate")
    ap.add_argument("--nodes", choices=["esm", "onehot"], default="esm",
                    help="receptor NODE features. `onehot` removes ESM from the graph "
                         "entirely, so with the gate on it reaches the receptor vector "
                         "ONLY through the frozen branch at weight (1-alpha) -- the "
                         "decomposition in which alpha is an honest fraction of "
                         "structure. Writes its own metrics_*_onehot.csv")
    ap.add_argument("--seed-graph", action="store_true",
                    help="SEED THE GRAPH'S OWN INITIALISATION from --seeds. Off (the "
                         "default) the weights come from torch's global RNG, which is "
                         "drawn from OS entropy and then advanced by whatever else that "
                         "worker trained first -- which is why the alpha=1 and "
                         "graph_legacy arms, provably the same computation on the same "
                         "input, do not land on the same number. On, that lottery is "
                         "gone (the scatter-add on CUDA is still atomic, so this is not "
                         "bit-determinism). It is OFF by default because turning it on "
                         "changes every number already in results/graph/ -- a run with "
                         "it is a NEW series, not more folds of an old one, and the "
                         "`seeded_graph` column records which kind each row is")
    ap.add_argument("--no-score-train", dest="score_train", action="store_false",
                    help="skip scoring the fitted head on its own train rows. They cost "
                         "one predict and are the only column in which an overfit head "
                         "is visible at all")
    ap.add_argument("--baselines-only", action="store_true",
                    help="only boost_full + naive (no graph, so seconds not hours) -- the "
                         "cheap way to check this sweep reproduces the table of record "
                         "before spending a GPU on the rest")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="seeds for the graph AND the boosting head. Each is run on "
                         "every fold, so the repeats become folds x seeds: folds vary "
                         "WHICH ROWS are held out, seeds vary the model's own draw "
                         "(graph init, bagging, the head's subsample/colsample), and "
                         "only the two together bound the error bar. 42 is the "
                         "ensembler's own default and the seed every reported number "
                         "was produced at, so keep it in the list -- extra seeds add "
                         "to that row rather than replacing it")
    ap.add_argument("--max-parallel", type=int, default=1,
                    help="total concurrent trainings. Ignored when --per-gpu is given")
    ap.add_argument("--per-gpu", type=int, default=None,
                    help="concurrent trainings PER GPU -- the knob to tune. One graph "
                         "training leaves most of a modern card idle, so raise this "
                         "while watching the memory and util rows of the dashboard "
                         "until either stops improving")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="print one line per finished cell instead of the live block, "
                         "and let the workers' own output through")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU indices to use. Default: every card nvidia-smi reports. "
                         "Workers are dealt out in proportion to the memory FREE on "
                         "each, so a card someone else is using gets fewer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="recompute cells already in the CSV")
    args = ap.parse_args()

    args.mol_source = args.mol_sources[0]        # so paths()/out_path() have one
    if args.variant and any(d != "m2or" for d in args.dataset):
        print("NOTE: --variant is being applied to an insect dataset too; its canonical "
              "graph is q0cov and the filename will NOT record the variant there.",
              flush=True)
    # The molecule source is an outer loop rather than a flag on one run: it changes
    # both the boost's molecular half and the graph's molecule nodes, so each value is
    # its own set of files -- and looping here is what lets the whole grid be one
    # resumable command instead of a shell loop that forgets where it stopped.
    todo = [(mol, ds, regime) for mol in args.mol_sources
            for ds in args.dataset for regime in args.regime]

    # Count the whole grid before starting anything, so the progress bar and the ETA
    # are about the RUN and not about whichever file happens to be open. Cheap: it
    # only reads the CSVs already on disk.
    total = 0
    for mol, ds, regime in todo:
        args.mol_source = mol
        args._variant = args.variant or DEFAULT_VARIANT[ds]
        m, _, rec = sibling_paths(ds, regime, args)
        _, done = load_done(rec, m, args)
        total += sum(1 for j in plan(repeats(ds, regime, args), args, done)
                     if j[0] != "baselines")

    gpus = args.gpus or visible_gpus() or [0]
    n_workers = workers_wanted(args, gpus)
    dash = None
    if not args.no_dashboard and total:
        dash = Dashboard(total, gpus if gpus != [0] or visible_gpus() else [],
                         title=f"v8 alpha grid -- {total} cells, {n_workers} workers "
                               f"over {len(gpus)} gpu(s)")
    print(f"\n{total} cells to run, {n_workers} concurrent "
          f"({'--per-gpu ' + str(args.per_gpu) if args.per_gpu else '--max-parallel ' + str(args.max_parallel)})"
          f" on gpu(s) {gpus}", flush=True)

    written = []
    for i, (mol, ds, regime) in enumerate(todo, 1):
        args.mol_source = mol
        out = sweep(ds, regime, args, dash=dash)
        written.append(out)
        if dash is not None:
            dash.note(f"[{i}/{len(todo)}] finished {out.name}")
    if dash is not None:
        dash.close()
    print("\nwrote:")
    for w in written:
        print(f"  {w}")
    print("\nread with:\n"
          "  python scripts/analysis/headline_table.py    # the numbers, every run\n"
          "  python scripts/analysis/alpha_grid.py        # the dial: geometry + prediction\n"
          "  notebooks/graph/alpha_gate/alpha_gate.ipynb   # the same, as figures")


if __name__ == "__main__":
    main()
