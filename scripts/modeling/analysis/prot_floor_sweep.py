"""Protein-representation FLOOR sweep: does a real pLM beat amino-acid counts?

For one dataset x one regime, boosts `[protein_feature || ChemBERTa]` with the
pipeline's own fit_boost/predict_scores (auto-GPU) and reports the fold-mean
metric for each protein feature side by side:

  * real pLMs      -- esm1b, esm2, prott5, esm3  (loaded from npz, keyed by SEQUENCE;
                      each is included only if its npz exists AND covers every
                      receptor, so a box that is missing one just skips it)
  * classical floor -- AAC, kmer2, AAindex, CTD, PseAAC, BLOSUM (computed here
                      from the receptor sequence, no learning)
  * controls       -- onehot (identity), onehot_only (no molecule), mol_only.
  * OUR graph      -- `--gnn esm3@1 esm3@0 prott5@1`: the pipeline's signed GNN,
                      emit=prot, boosted as [refined receptor || ChemBERTa] (that is
                      our `cls+mol`). Trained HERE, per fold, on THIS script's splits:
                      the refined receptor sees its fold's train pairs, so a vector
                      borrowed from another run's folds would be a leak, not a
                      shortcut. `@1` is the plain graph on that protein source, `@0`
                      is the node dial at zero -- receptor identity and nothing else,
                      so its protein file only decides the mask, not the features.

Regimes match the rest of the pipeline's axis (COLD MOLECULE = inductive), so
this table sits next to every other table in the paper:
  transductive   m2or = LORAX `rand` folds; cc/hc = upstream `rand` (receptors
                 AND molecules seen).
  inductive      COLD MOLECULE -- m2or = `inductive_molecule_v5` (folds 1-5 ->
                 seeds 42-46); cc/hc = `our_inductive`. This is where the graph
                 wins on M2OR, so the floor here says exactly the useful thing:
                 in that regime, swapping ESM for AAC (or even onehot) barely
                 moves the score -> the graph's gain is data-driven receptor
                 refinement, not a better encoder. NB receptors are all SEEN, so
                 the protein-feature spread compresses -- that IS the point.
  cold_receptor  OPTIONAL appendix -- hold whole receptors out (m2or =
                 group_receptor seeds; cc = upstream `cdhit`; HC has none). The
                 protein-transfer axis; not the project's inductive.

RESUMABLE. A cell already in the CSV is kept, not refitted, so adding `--gnn` to a
table whose descriptor and pLM rows are already there costs only the graphs. What may
be reused is decided by the sidecar `prot_floor_<ds>_<regime>.json`, written beside the
CSV since Sep 2026: it records the folds, the seeds and the interpreter. A CSV with no
sidecar predates that and is NOT reused by default -- this script once folded the
insects' val rows into train (see `make_splits`), so an old file can hold rows fitted
on more data than the paper's other methods ever saw. `--trust-existing` overrides,
`--force` refits everything.

    # one cell:
    .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py --dataset m2or --regime transductive
    # everything (skips cells a dataset can't do):
    for d in m2or cc hc; do for r in transductive inductive; do \
      .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py --dataset $d --regime $r; done; done
"""
import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind import dataset as D                                 # noqa: E402
from orbind.baselines import fit_boost, predict_scores          # noqa: E402

AA = "ARNDCQEGHILKMFPSTWYV"
AAi = {a: i for i, a in enumerate(AA)}

# ---------------- physicochemical scales (order ARNDCQEGHILKMFPSTWYV) ----------
KD   = [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]
VOL  = [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]
POL  = [8.1,10.5,11.6,13.0,5.5,10.5,12.3,9.0,10.4,5.2,4.9,11.3,5.7,5.2,8.0,9.2,8.6,5.4,6.2,5.9]
PI   = [6.0,10.76,5.41,2.77,5.07,5.65,3.22,5.97,7.59,6.02,5.98,9.74,5.74,5.48,6.30,5.68,5.60,5.89,5.66,5.96]
PA   = [1.42,0.98,0.67,1.01,0.70,1.11,1.51,0.57,1.00,1.08,1.21,1.16,1.45,1.13,0.57,0.77,0.83,1.08,0.69,1.06]
PB   = [0.83,0.93,0.89,0.54,1.19,1.10,0.37,0.75,0.87,1.60,1.30,0.74,1.05,1.38,0.55,0.75,1.19,1.37,1.47,1.70]
MW   = [89,174,132,133,121,146,147,75,155,131,131,146,149,165,115,105,119,204,181,117]
AAINDEX = np.array([KD, VOL, POL, PI, PA, PB, MW], dtype=np.float64)   # 7 x 20

# PseAAC properties (Chou 2001): hydrophobicity, hydrophilicity, side-chain mass
H1 = [0.62,-2.53,-0.78,-0.90,0.29,-0.85,-0.74,0.48,-0.40,1.38,1.06,-1.50,0.64,1.19,0.12,-0.18,-0.05,0.81,0.26,1.08]
H2 = [-0.5,3.0,0.2,3.0,-1.0,0.2,3.0,0.0,-0.5,-1.8,-1.8,3.0,-1.3,-2.5,0.0,0.3,-0.4,-3.4,-2.3,-1.5]
SM = [15,101,58,59,47,72,73,1,82,57,57,73,75,91,42,31,45,130,107,43]
def _z(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()
PSE = np.stack([_z(H1), _z(H2), _z(SM)])                              # 3 x 20

# BLOSUM62 (order ARNDCQEGHILKMFPSTWYV)
BL = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
BLOSUM = np.array([[float(x) for x in r.split()] for r in BL.splitlines()])   # 20x20

# CTD groupings (7 attributes x 3 groups), standard PROFEAT/iFeature
CTD_GROUPS = {
 "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
 "vdw_volume":     ["GASTPDC", "NVEQIL", "MHKFRYW"],
 "polarity":       ["LIFWCMVY", "PATGS", "HQRKNED"],
 "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
 "charge":         ["KR", "ANCQGHILMFPSTWYV", "DE"],
 "secondary_str":  ["EALMQKRH", "VIYCWFT", "GNPSD"],
 "solvent_acc":    ["ALFCGIVW", "RKQEND", "MPSTHY"],
}


#: The message-passing edge variant each dataset's reported numbers stand on --
#: the same table `run_alpha_gate_sweep.VARIANTS`/`DEFAULT_VARIANT` encodes. Spelled
#: out rather than imported: importing the sweep pulls in its whole world, and a
#: construction that silently differed from the paper's would be invisible here.
GNN_VARIANT = {
    "m2or": dict(q=0.99, criterion="greedy_pair_cover", k_mode="coverage_quantile"),
    "cc":   dict(q=0.0,  criterion="coverage",          k_mode="coverage_quantile"),
    "hc":   dict(q=0.0,  criterion="coverage",          k_mode="coverage_quantile"),
}
#: What `--gnn` means with no arguments: the three rows the protein table asks for.
DEFAULT_GNN = ["esm3@1", "esm3@0", "prott5@1"]


# ---------------- descriptor functions: sequence -> vector --------------------
def _clean(seq):
    return [c for c in seq.upper() if c in AAi]

def d_aac(seq):
    s = _clean(seq); v = np.zeros(20)
    for c in s: v[AAi[c]] += 1
    return v / max(len(s), 1)

def d_kmer2(seq):
    s = _clean(seq); v = np.zeros(400)
    for a, b in zip(s, s[1:]): v[AAi[a] * 20 + AAi[b]] += 1
    return v / max(len(s) - 1, 1)

def d_aaindex(seq):
    s = _clean(seq)
    if not s: return np.zeros(14)
    idx = [AAi[c] for c in s]; vals = AAINDEX[:, idx]
    return np.concatenate([vals.mean(1), vals.std(1)])

def d_blosum(seq):
    s = _clean(seq)
    if not s: return np.zeros(20)
    return BLOSUM[[AAi[c] for c in s]].mean(0)

def _grp_string(seq, groups):
    m = {}
    for gi, g in enumerate(groups):
        for a in g: m[a] = gi
    return [m[c] for c in seq if c in m]

def d_ctd(seq):
    s = _clean(seq); out = []
    for groups in CTD_GROUPS.values():
        gs = _grp_string(s, groups); n = len(gs)
        comp = np.zeros(3)
        for g in gs: comp[g] += 1
        comp = comp / max(n, 1)
        trans = np.zeros(3)
        for x, y in zip(gs, gs[1:]):
            if x != y:
                trans[{(0, 1): 0, (0, 2): 1, (1, 2): 2}[tuple(sorted((x, y)))]] += 1
        trans = trans / max(n - 1, 1)
        dist = []
        for gi in range(3):
            pos = [i + 1 for i, g in enumerate(gs) if g == gi]
            if not pos:
                dist += [0, 0, 0, 0, 0]
            else:
                k = len(pos)
                for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                    j = 0 if q == 0 else int(np.ceil(q * k)) - 1
                    dist.append(pos[j] / n)
        out += list(comp) + list(trans) + dist
    return np.array(out)   # 147

def d_pseaac(seq, lam=5, w=0.05):
    s = _clean(seq)
    if len(s) <= lam: s = s + s[:lam]
    idx = [AAi[c] for c in s]
    f = np.zeros(20)
    for i in idx: f[i] += 1
    f = f / len(idx)
    P = PSE[:, idx]
    tau = np.array([((P[:, :-k] - P[:, k:]) ** 2).mean(0).mean() for k in range(1, lam + 1)])
    denom = f.sum() + w * tau.sum()
    return np.concatenate([f / denom, w * tau / denom])   # 25

DESC = {"aac": d_aac, "kmer2": d_kmer2, "aaindex": d_aaindex,
        "ctd": d_ctd, "pseaac": d_pseaac, "blosum": d_blosum}

# ---------------- per-dataset config ------------------------------------------
# The pLM order here is the order they appear in the table. A file that is
# absent (or doesn't cover every receptor) is skipped with a warning, so the
# same command runs on any box -- ProtT5 shows up wherever prott5_<ds>.npz was
# built, esm2 only on M2OR until insect esm2 exists.
EMB = "data/embeddings"
DATASETS = {
    "m2or": {"task": "classification", "primary": "AUROC",
             "mol": f"{EMB}/molecules/chemberta_77m_m2or.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean.npz",
                      "prott5": f"{EMB}/proteins/prott5_m2or.npz",
                      "esm3":  f"{EMB}/proteins/esm3_m2or.npz"}},
    "cc":   {"task": "regression", "primary": "R2",
             "mol": f"{EMB}/molecules/chemberta_77m_cc.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean_cc.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean_cc.npz",
                      "prott5": f"{EMB}/proteins/prott5_cc.npz",
                      "esm3":  f"{EMB}/proteins/esm3_cc.npz"}},
    "hc":   {"task": "regression", "primary": "R2",
             "mol": f"{EMB}/molecules/chemberta_77m_hc.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean_hc.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean_hc.npz",
                      "prott5": f"{EMB}/proteins/prott5_hc.npz",
                      "esm3":  f"{EMB}/proteins/esm3_hc.npz"}},
}


def load_pairs(dataset):
    """(pairs, task) for the dataset, with `receptor`/`inchikey`/`label` cols."""
    if dataset == "m2or":
        from orbind.regimes import full_full_pairs
        return full_full_pairs(pool_fold=1), "classification"
    from orbind.regimes_ofm import ofm_pairs
    return ofm_pairs(dataset), "regression"


def make_splits(dataset, pairs, y_all, regime, folds):
    """{fold: (train_idx, test_idx)} for the (dataset, regime) cell.

    inductive == COLD MOLECULE (the project's axis): m2or =
    `inductive_molecule_v5` seeds 42-46 (folds 1-5 -> seed 41+fold); cc/hc =
    upstream `our_inductive`. transductive = `rand` folds. cold_receptor
    (optional) holds whole receptors out: m2or = group_receptor seeds; cc =
    upstream `cdhit`; HC ships neither, so it has no cold_receptor cell.

    TRAIN ONLY, on every dataset -- val is never fitted on. This script used to
    fold the insects' val into train ("no tuning here, so val is just more rows"),
    which gave its boost 11-26% more rows than every other method in the paper --
    and on `our_inductive` 18 odorants the others never saw -- so its ESM-1b row
    read 0.537 where the main tables' identical boost read 0.486. The rule the
    rest of the paper follows is: parameters are fitted on train only; val exists
    for decisions (a baseline's best epoch, the alpha, a threshold).
    """
    splits = {}
    if dataset == "m2or":
        from orbind.regimes import load_split
        for f in folds:
            if regime == "transductive":
                tr, va, te = load_split("transductive", f)
                splits[f] = (tr, te)
            elif regime == "inductive":               # cold MOLECULE (project axis)
                tr, va, te = load_split("inductive_molecule_v5", 41 + f)   # folds 1-5 -> seeds 42-46
                splits[f] = (tr, te)
            else:                                     # cold_receptor (optional appendix)
                trm, tem = D.split(pairs, y_all, kind="group_receptor", seed=f)
                splits[f] = (np.where(trm)[0], np.where(tem)[0])
                print(f"  cold_receptor fold{f}: train {trm.sum()} / test {tem.sum()} rows "
                      f"({pairs['receptor'].iloc[np.where(tem)[0]].nunique()} unseen receptors)",
                      flush=True)
        return splits
    # insects
    from orbind.regimes_ofm import ofm_indices, available_families
    family = {"transductive": "rand", "inductive": "our_inductive",
              "cold_receptor": "cdhit"}[regime]
    if family not in available_families(dataset):
        raise SystemExit(f"{dataset} ships no {family!r} split -> no {regime} cell "
                         f"(has {available_families(dataset)})")
    for f in folds:
        tr, va, te = ofm_indices(dataset, family, f)
        splits[f] = (tr, te)
    return splits


def parse_gnn_spec(spec, plm_specs):
    """`name@alpha` -> (row name, npz path, alpha). Fails loudly on a typo.

    The alpha is the v9 NODE dial (`prot_mix`): 1 is the plain graph on that protein
    source, 0 is a graph over receptor identity alone. It is NOT the v8 gate, which
    mixes the graph's output against a frozen branch and runs the other way.
    """
    if "@" not in spec:
        raise SystemExit(f"--gnn takes name@alpha (e.g. esm3@1), got {spec!r}")
    name, a = spec.split("@", 1)
    try:
        alpha = float(a)
    except ValueError:
        raise SystemExit(f"--gnn: {a!r} is not a number in {spec!r}")
    if not 0.0 <= alpha <= 1.0:
        raise SystemExit(f"--gnn: alpha must be in [0, 1], got {alpha} in {spec!r}")
    if name not in plm_specs:
        raise SystemExit(f"--gnn: unknown protein source {name!r}; "
                         f"have {sorted(plm_specs)} (add one with --extra-prot)")
    return f"GNN[{name}]@a{alpha:g}", plm_specs[name], alpha


def gnn_rows(args, dataset, spec, pairs, splits, y_all, Xmol, row_m, task, metric_fn,
             mol_path, done=frozenset()):
    """One `--gnn` spec, every fold: train the graph, boost [refined || molecule].

    Everything here is deliberate and worth stating once:

    * the graph is trained INSIDE this script's fold, so the row is comparable with
      the descriptor rows above it by construction rather than by inspection;
    * `val` is empty -- this table fits on train only (see `make_splits`), and the
      extractor never consults val anyway: fixed epoch count, no early stopping;
    * the molecule embedding is the SAME npz the boost's molecule half uses, so the
      only thing that changes between this row and `esm3` above is the receptor side;
    * `deterministic_init=True` removes the initialisation lottery, which is what the
      paper's own runs do (`--seed-graph`);
    * the training REGIME -- neighbour sampling and per-layer L2 normalisation -- is
      passed explicitly rather than inherited from the extractor's default, because
      that default moved on 23.09.2026 and a table whose rows silently changed model
      between two runs is worse than one that fails.
    """
    from orbind.gnn_extractor import GnnSignedExtractor

    row_name, prot_path, alpha = spec
    variant = GNN_VARIANT[dataset]
    rows = []
    for f in args.folds:
        tr, te = splits[f]
        empty = np.empty(0, dtype=np.int64)
        for gseed in args.gnn_seeds:
            if all(cell_key({"prot": row_name, "fold": f, "seed": sd,
                             "gnn_seed": gseed}) in done for sd in args.seeds):
                continue          # this graph is already fitted for every boost seed
            ext = GnnSignedExtractor(
                name="cls", protein_path=prot_path, molecule_path=mol_path,
                task=task, emit="prot", prot_mix=alpha,
                n_models=args.gnn_n_models, epochs=args.gnn_epochs,
                deterministic_init=True, fanout=gnn_regime(args)[0],
                normalize_layers=gnn_regime(args)[1], **variant)
            t0 = time.time()
            Zp_tr, _Zp_va, Zp_te = ext.fit_transform(pairs, tr, empty, te, gseed)
            t_graph = time.time() - t0
            Xtr = np.concatenate([Zp_tr, Xmol[row_m[tr]]], 1).astype(np.float32)
            Xte = np.concatenate([Zp_te, Xmol[row_m[te]]], 1).astype(np.float32)
            for seed in args.seeds:
                if cell_key({"prot": row_name, "fold": f, "seed": seed,
                             "gnn_seed": gseed}) in done:
                    continue
                model = fit_boost(Xtr, y_all[tr], seed=seed, task=task)
                m = metric_fn(y_all[te], predict_scores(model, Xte, task=task))
                fan, norm = gnn_regime(args)
                rows.append({"prot": row_name, "fold": f, "seed": seed,
                             "gnn_seed": gseed, "pdim": int(Zp_tr.shape[1]),
                             "dim": int(Xtr.shape[1]), "n_train": len(tr),
                             "t_graph": t_graph,
                             "gnn_regime": regime_tag(fan, norm), **m})
                print(f"{row_name:22} fold{f} gseed{gseed} seed{seed} "
                      f"pdim={Zp_tr.shape[1]:5} " +
                      "  ".join(f"{k}={m[k]:.3f}" for k in list(m)[:2]), flush=True)
    return rows


def gnn_regime(args):
    """(fanout, normalize_layers) for this run's graphs. `--gnn-fanout 0 0` is how
    sampling is switched off, because a zero entry is refused by the extractor."""
    fan = tuple(f for f in (getattr(args, "gnn_fanout", None) or ()) if f > 0)
    return fan, bool(getattr(args, "gnn_normalize_layers", True))


def regime_tag(fanout, normalize):
    """One short string per encoder regime, stamped on every GNN row.

    It exists so that "which model is this row" is answerable from the CSV alone.
    A row written before 23.09.2026 has no such column; that silence means the
    historical encoder, and `HISTORICAL_TAG` is what it is read as.
    """
    return ("sampled" + "-".join(map(str, fanout)) if fanout else "full") + \
           ("+norm" if normalize else "")


#: What a missing `gnn_regime` column means: full neighbourhood, un-normalised.
HISTORICAL_TAG = "full"


def gnn_regimes_on_disk(rows):
    """Every regime tag among the GNN rows already in the CSV."""
    seen = set()
    for r in rows:
        g = r.get("gnn_seed", None)
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue                      # a descriptor row: no graph, no regime
        tag = r.get("gnn_regime", None)
        seen.add(HISTORICAL_TAG if tag is None or
                 (isinstance(tag, float) and np.isnan(tag)) else str(tag))
    return seen


def cell_key(row):
    """What makes a fitted cell unique: representation x fold x boost seed x graph seed.

    The graph seed is in the key because two GNN rows differing only by it are two
    trainings; `-1` marks the rows that have no graph at all, so a descriptor row and
    a GNN row can never collide.
    """
    g = row.get("gnn_seed", None)
    g = -1 if g is None or (isinstance(g, float) and np.isnan(g)) else int(g)
    return (str(row["prot"]), int(row["fold"]), int(row["seed"]), g)


def load_existing(out, args, dataset, regime):
    """(rows kept from disk, their keys). Empty unless a resume is allowed.

    The sidecar is the whole safety mechanism here. Rows written before it existed may
    have been fitted under the val-in-train convention this script used to have, which
    gave its boost 11-26% more rows than every other method in the paper -- silently
    mixing those with fresh ones would put two different experiments in one column.
    """
    if args.force or not out.exists():
        return [], set()
    prev = pd.read_csv(out)
    need = {"prot", "fold", "seed"}
    if not need <= set(prev.columns):
        raise SystemExit(
            f"{out.name} is missing {sorted(need - set(prev.columns))} -- it predates "
            f"the current row format and cannot be resumed from. Refit it with --force.")
    side = out.with_suffix(".json")
    if not side.exists() and not args.trust_existing:
        raise SystemExit(
            f"{out.name} has no sidecar {side.name}, so it predates the provenance "
            f"record and may have been fitted under the old val-in-train convention.\n"
            f"Refit it with --force, or keep it with --trust-existing if you know it "
            f"was produced after that fix.")
    if side.exists():
        old = json.loads(side.read_text(encoding="utf-8"))
        want = {"dataset": dataset, "regime": regime, "mol": args.mol}
        moved = [k for k, v in want.items() if str(old.get(k)) != str(v)]
        if moved:
            raise SystemExit(f"{side.name} says {moved} differ from this run -- that is "
                             f"a different table, not more rows of this one. Use --out.")
    rows = prev.to_dict("records")
    if getattr(args, "gnn", None) is not None:
        want = regime_tag(*gnn_regime(args))
        stale = gnn_regimes_on_disk(rows) - {want}
        if stale:
            raise SystemExit(
                f"{out.name} already holds GNN rows trained in regime(s) "
                f"{sorted(stale)}, and this run would add {want!r}. Two encoders in "
                f"one column is not a table.\n"
                f"Drop the graph rows and keep everything else -- the descriptor rows "
                f"cost nothing to keep and are unaffected:\n"
                f"    python -c \"import pandas as pd; d=pd.read_csv(r'{out}'); "
                f"d[d.gnn_seed.isna()].to_csv(r'{out}', index=False)\"\n"
                f"then re-run this command. `--force` instead refits the whole table.")
    print(f"resuming: {len(rows)} row(s) already in {out.name}", flush=True)
    return rows, {cell_key(r) for r in rows}


def write_sidecar(out, args, rows, dataset, regime):
    """Provenance beside the CSV: what produced THESE rows, and with what.

    `dataset`/`regime` are passed, never read off `args`: after the run learned to
    walk every cell those are the whole LIST, and a sidecar claiming all three would
    pass the very check it exists to fail.
    """
    import xgboost
    body = {**{k: v for k, v in vars(args).items()},
            "dataset": dataset, "regime": regime,
            "python": sys.executable, "xgboost": xgboost.__version__,
            "n_rows": len(rows),
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".json").write_text(json.dumps(body, indent=2, default=str),
                                        encoding="utf-8")


def parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Lists, not single values: the table is six cells and they must be fitted the
    # same way. A cell a dataset cannot do (HC ships no cold_receptor) is skipped with
    # a note rather than killing the run.
    ap.add_argument("--dataset", nargs="+", default=["m2or", "cc", "hc"],
                    choices=sorted(DATASETS))
    ap.add_argument("--regime", nargs="+", default=["transductive", "inductive"],
                    choices=["transductive", "inductive", "cold_receptor"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    # The boost's own seed (subsample/colsample draws), NOT the split: the main
    # tables' boost row is the alpha sweep's `boost_full`, fitted at seeds 42-46 and
    # averaged within each split, so the same seeds here are what make the two rows
    # the same number. The old default was the fold number, which moves R2 by up to
    # 0.045 on a single insect fold.
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="boost seeds, averaged within each fold (default 42-46)")
    ap.add_argument("--mol", default=None, help="override molecule npz")
    ap.add_argument("--extra-prot", action="append", default=[], metavar="name=path",
                    help="add a protein npz (keyed by sequence); repeatable")
    ap.add_argument("--force", action="store_true",
                    help="refit every cell, ignoring what is already in the CSV")
    ap.add_argument("--trust-existing", action="store_true",
                    help="reuse a CSV that has no provenance sidecar (see RESUMABLE)")
    ap.add_argument("--gnn", nargs="*", default=None, metavar="name@alpha",
                    help="add OUR graph as rows: name@alpha, alpha on the v9 NODE "
                         f"dial (1 = plain graph on that source, 0 = receptor identity "
                         f"alone). Bare --gnn means {' '.join(DEFAULT_GNN)}")
    ap.add_argument("--gnn-seeds", type=int, nargs="+", default=[42],
                    help="GRAPH seeds (each trains a graph). One by default: a graph "
                         "per fold per seed per spec is the expensive part of this "
                         "table, and the boost seeds below already average inside "
                         "each fold")
    ap.add_argument("--gnn-fanout", type=int, nargs=2, default=[25, 10],
                    metavar=("L1", "L2"),
                    help="neighbour sampling for OUR rows, layer 1 then layer 2, "
                         "redrawn every epoch; inference stays full-neighbourhood. "
                         "GraphSAGE's own 25 10 is the default since 23.09.2026; "
                         "`--gnn-fanout 0 0` restores the historical encoder")
    ap.add_argument("--gnn-no-normalize-layers", dest="gnn_normalize_layers",
                    action="store_false",
                    help="do NOT L2-normalise after each layer (on by default since "
                         "23.09.2026, together with --gnn-fanout)")
    ap.add_argument("--gnn-epochs", type=int, default=900)
    ap.add_argument("--gnn-n-models", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="default: results/tables/prot_floor_<dataset>_<regime>.csv")
    return ap


def run_cell(args, dataset, regime):
    """One (dataset, regime) cell, end to end. Returns the CSV it wrote, or None."""

    cfg = DATASETS[dataset]
    task = cfg["task"]; primary = cfg["primary"]
    metric_fn = D.METRICS[task]

    print(f"[{dataset} / {regime}] loading pairs + embeddings ...", flush=True)
    pairs, _ = load_pairs(dataset)
    recs = pd.unique(pairs["receptor"]); mols = pd.unique(pairs["inchikey"])
    rec_i = {r: i for i, r in enumerate(recs)}; mol_i = {m: i for i, m in enumerate(mols)}
    row_r = pairs["receptor"].map(rec_i).to_numpy()
    row_m = pairs["inchikey"].map(mol_i).to_numpy()
    y_all = pairs["label"].to_numpy().astype(np.float32)

    mol_path = args.mol or cfg["mol"]
    chem = D.load_npz_dict((_root / mol_path) if not pathlib.Path(mol_path).is_absolute() else mol_path)
    missing_mol = [m for m in mols if m not in chem]
    if missing_mol:
        raise SystemExit(f"{len(missing_mol)} molecules missing from {mol_path}, e.g. {missing_mol[:3]}")
    Xmol = np.stack([chem[m] for m in mols]).astype(np.float32)

    # protein matrices: classical descriptors (always) + onehot + available pLMs
    protmats = {}
    for name, fn in DESC.items():
        protmats[name] = np.stack([fn(r) for r in recs]).astype(np.float32)
    protmats["onehot"] = np.eye(len(recs), dtype=np.float32)

    plm_specs = dict(cfg["plms"])
    for kv in args.extra_prot:                          # ad-hoc additions/overrides
        name, path = kv.split("=", 1)
        plm_specs[name] = path
    plm_names = []
    for name, path in plm_specs.items():
        p = (_root / path) if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
        if not p.exists():
            print(f"  [skip pLM {name}] {path} not found", flush=True)
            continue
        emb = D.load_npz_dict(p)
        miss = [r for r in recs if r not in emb]
        if miss:
            print(f"  [skip pLM {name}] covers {len(recs)-len(miss)}/{len(recs)} receptors", flush=True)
            continue
        protmats[name] = np.stack([emb[r] for r in recs]).astype(np.float32)
        plm_names.append(name)
        print(f"  loaded pLM {name}: {protmats[name].shape}", flush=True)

    print(f"receptors={len(recs)} mols={len(mols)} chem_dim={Xmol.shape[1]} "
          f"pLMs={plm_names} task={task}", flush=True)

    # spec = (row_name, protein_matrix_key or None, use_molecule)
    specs  = [(n, n, True) for n in plm_names]           # real pLMs first
    specs += [("onehot", "onehot", True)]
    specs += [(n, n, True) for n in DESC]                # classical floor
    specs += [("onehot_only", "onehot", False), ("mol_only", None, True)]

    splits = make_splits(dataset, pairs, y_all, regime, args.folds)

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"results/tables/prot_floor_{dataset}_{regime}.csv")
    if not out.is_absolute():
        out = _root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, done = load_existing(out, args, dataset, regime)
    skipped = 0
    for name, pm, use_mol in specs:
        pdim = protmats[pm].shape[1] if pm is not None else 0
        for f in args.folds:
            if all(cell_key({"prot": name, "fold": f, "seed": sd}) in done
                   for sd in args.seeds):
                skipped += len(args.seeds)
                continue                      # every seed of this fold is already fitted
            tr, te = splits[f]

            def build(idx):
                # [protein || molecule], the alpha sweep's own column order: with
                # colsample_bytree the seed picks columns BY POSITION, so the same seed
                # on the other order is a different forest
                parts = []
                if pm is not None: parts.append(protmats[pm][row_r[idx]])
                if use_mol: parts.append(Xmol[row_m[idx]])
                return np.concatenate(parts, axis=1).astype(np.float32)

            Xtr, Xte = build(tr), build(te)
            for seed in args.seeds:
                if cell_key({"prot": name, "fold": f, "seed": seed}) in done:
                    skipped += 1
                    continue
                model = fit_boost(Xtr, y_all[tr], seed=seed, task=task)
                p = predict_scores(model, Xte, task=task)
                m = metric_fn(y_all[te], p)
                rows.append({"prot": name, "fold": f, "seed": seed, "pdim": pdim,
                             "dim": Xtr.shape[1], "n_train": len(tr), **m})
                done.add(cell_key(rows[-1]))
                print(f"{name:12} fold{f} seed{seed} pdim={pdim:5} " +
                      "  ".join(f"{k}={m[k]:.3f}" for k in list(m)[:2]), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    if args.gnn is not None:
        if dataset not in GNN_VARIANT:
            raise SystemExit(f"no edge variant recorded for {dataset}; add one to "
                             f"GNN_VARIANT before asking for GNN rows")
        specs = [parse_gnn_spec(x, plm_specs) for x in (args.gnn or DEFAULT_GNN)]
        missing = [(n, p) for n, p, _ in specs
                   if not ((_root / p) if not pathlib.Path(p).is_absolute()
                           else pathlib.Path(p)).exists()]
        if missing:
            raise SystemExit("--gnn: missing npz -> " +
                             ", ".join(f"{n} ({p})" for n, p in missing))
        print(f"\nGNN rows: {[n for n, _, _ in specs]}  variant={GNN_VARIANT[dataset]}"
              f"  graph seeds={args.gnn_seeds}", flush=True)
        for spec in specs:
            new = gnn_rows(args, dataset, spec, pairs, splits, y_all, Xmol, row_m,
                           task, metric_fn, mol_path, done)
            rows += new
            done |= {cell_key(r) for r in new}
            pd.DataFrame(rows).to_csv(out, index=False)

    if skipped:
        print(f"\nreused {skipped} cell(s) already on disk (--force refits them)",
              flush=True)
    write_sidecar(out, args, rows, dataset, regime)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns
                   if c not in ("prot", "fold", "seed", "gnn_seed", "pdim", "dim",
                                "n_train", "t_graph")]
    # the split is the unit: seeds are averaged inside a fold first, as the main tables do
    per_fold = df.groupby(["prot", "fold"], as_index=False)[metric_cols + ["pdim"]].mean()
    # a GNN row carries extra keys (gnn_seed, t_graph); averaging them away here is
    # right -- the unit of evidence is the fold, whatever produced the features
    g = per_fold.groupby("prot")
    print("\n" + "=" * 96)
    print(f"{dataset.upper()} / {regime.upper()} -- protein floor "
          f"(mol=ChemBERTa), train only, seeds {args.seeds} averaged per fold, "
          f"{len(args.folds)} folds, sorted by {primary}")
    print("mean ± std over folds (the main tables' convention)   [ci95 = 1.96*std/sqrt(n)]")
    print("=" * 96)
    for name in g[primary].mean().sort_values(ascending=False).index:
        s = g.get_group(name)
        def mc(c):
            x = s[c].to_numpy()
            return f"{x.mean():.3f}±{x.std(ddof=1):.3f}"
        x = s[primary].to_numpy()
        cells = "  ".join(f"{c} {mc(c)}" for c in metric_cols[:4])
        print(f"{name:12} pdim={int(s['pdim'].iloc[0]):5}  {cells}   "
              f"[ci95 {primary} ±{1.96 * x.std(ddof=1) / len(x) ** 0.5:.3f}]")
    print(f"\nwrote -> {out}")


    return out


def main():
    args = parser().parse_args()
    if args.out and (len(args.dataset) > 1 or len(args.regime) > 1):
        raise SystemExit("--out names ONE file; drop it to write the default path per "
                         "cell, or ask for a single --dataset/--regime")
    done, skipped = [], []
    for dataset in args.dataset:
        for regime in args.regime:
            print(f"\n{'#' * 78}\n# {dataset} / {regime}\n{'#' * 78}", flush=True)
            try:
                done.append(run_cell(args, dataset, regime))
            except SystemExit as e:
                # A dataset that ships no such split family is not a failure of the
                # run -- HC has no cold_receptor at all. Say so and keep going, or a
                # six-cell command dies on the one cell that was never possible.
                print(f"  [skip {dataset}/{regime}] {e}", flush=True)
                skipped.append((dataset, regime, str(e)))
    print(f"\n{'=' * 78}\nwrote {len(done)} cell(s):")
    for o in done:
        print(f"  {o}")
    for d, r, why in skipped:
        print(f"  [skipped] {d}/{r}: {why}")
    print("\nNow render the table:\n"
          "  .venv/bin/python scripts/article_tables/s2_protein_sources.py "
          f"--dataset {' '.join(args.dataset)} --regime {' '.join(args.regime)}")


if __name__ == "__main__":
    main()
