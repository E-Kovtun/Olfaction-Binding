"""Interaction-profile collaborative filtering over the molecule similarity graph.

A DIFFERENT architecture from the similarity-graph MP run: instead of propagating
a molecule's *structural embedding* over its chemical neighbours, we propagate the
neighbours' *binding profiles* (their rows of the train interaction matrix). The
query molecule borrows "how do chemically-similar molecules bind" and the XGBoost
probe turns that into a prediction for the target protein.

Parameter-free (no neural training) — the only learner is the boost. Bars:

  raw            : [ mol | prot ]                                    (baseline)
  profile_A      : [ mol | prot | nbr_frac_pos(P) | nbr_frac_neg(P) ]
                   FAT variant — the whole neighbour binding profile over all P
                   proteins; the boost discovers protein-protein correlations.
  profile_B      : [ mol | prot | cf_pos, cf_neg, cf_pos_smooth, cf_neg_smooth ]
                   ALIGNED variant — neighbour profile read only at the target
                   protein X, plus a version smoothed over X's protein neighbours
                   ("protein Y correlated with X"). 4 scalars.
  profile_A_own  : profile_A + the molecule's OWN train row (target col masked)   [transductive only]
  profile_B_own  : profile_B + own row smoothed to X's protein neighbours          [transductive only]

Aggregation weights are BINARY (plain mean over the k nearest neighbours) — useful
signal often sits with a neighbour that is not the single closest one. Missing
(untested) entries are encoded as separate pos/neg fraction channels (indicator).

Leakage: the interaction matrix is built from TRAIN edges ONLY, so a query's held-out
test pair is never in it. NEIGHBOUR features carry no leak (a neighbour's binding to
X is a *different* datapoint, legitimate CF signal). OWN features could leak the label
via the target column, so it is masked: profile_A_own zeroes the target column per
pair; profile_B_own only reads OTHER proteins (protein-neighbour smoothing excludes
self). In inductive_molecule the query is cold (empty own row) → own bars ~= neighbour bars.

Repeat semantics (matches the full_full v5 convention):
  transductive       -> genuine LoRaX folds 1..5
  inductive_molecule -> fold 1 with cold-split seeds 42..46 (stratified prevalence)

Writes one tidy CSV of per-repeat metrics; the notebook only reads + plots it.
Usage:
  uv run python scripts/modeling/eval/run_molecule_profile_cf.py           # all units
  uv run python scripts/modeling/eval/run_molecule_profile_cf.py --regime inductive_molecule --repeat 42
"""
import argparse, pathlib, sys
import numpy as np, pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import lorax as L, hetero as H
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
CORE_BARS = ["raw", "profile_A", "profile_B"]
OWN_BARS = ["profile_A_own", "profile_B_own"]          # transductive only (warm query has an own row)
REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def standardize(d):
    keys = list(d)
    X = np.stack([np.asarray(d[k], np.float64) for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((np.asarray(d[k], np.float64) - mu) / sd).astype(np.float32) for k in keys}


def knn_idx(X, k):
    """Indices of each row's k nearest neighbours by cosine (self excluded)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, X.shape[0] - 1)
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]          # [N, k], unordered (mean is order-free)


def mean_rows(S, nbr):
    """out[i] = mean over j of S[nbr[i, j]]  -> neighbour-aggregated row profile [N, P]."""
    out = np.zeros_like(S)
    for j in range(nbr.shape[1]):
        out += S[nbr[:, j]]
    return out / nbr.shape[1]


def mean_cols(S, nbr_p):
    """out[:, p] = mean over j of S[:, nbr_p[p, j]]  -> smooth each column over protein neighbours."""
    out = np.zeros_like(S)
    for j in range(nbr_p.shape[1]):
        out += S[:, nbr_p[:, j]]
    return out / nbr_p.shape[1]


def stratified_cold_split(base, val_every=3, seed=42):
    """Rebalance held-out (cold) molecules into val/test with matched prevalence."""
    hp = np.concatenate([base["val"]["pos"], base["test"]["pos"]], 0)
    hn = np.concatenate([base["val"]["neg"], base["test"]["neg"]], 0)
    mols = sorted(set(hp[:, 0]).union(hn[:, 0]))
    pos_by = {m: hp[hp[:, 0] == m] for m in mols}; neg_by = {m: hn[hn[:, 0] == m] for m in mols}
    pf = lambda m: len(pos_by[m]) / max(len(pos_by[m]) + len(neg_by[m]), 1)
    rng = np.random.default_rng(seed)
    order = sorted(mols, key=lambda m: (pf(m), rng.random()))
    vmol = [m for i, m in enumerate(order) if i % val_every == 0]
    tmol = [m for i, m in enumerate(order) if i % val_every != 0]
    def cat(ms, by):
        a = [by[m] for m in ms if len(by[m])]
        return np.concatenate(a, 0) if a else np.zeros((0, 2), int)
    return {"train": base["train"],
            "val":  {"pos": cat(vmol, pos_by), "neg": cat(vmol, neg_by)},
            "test": {"pos": cat(tmol, pos_by), "neg": cat(tmol, neg_by)}}


def _col(a):
    a = np.asarray(a, np.float32)
    return a.reshape(a.shape[0], -1)


def run_unit(regime, rep, esm, chem, args):
    """One (regime, repeat) unit -> rows for each bar."""
    fold = rep if regime == "transductive" else 1
    split_seed = 42 if regime == "transductive" else rep
    tag = f"[{regime} rep{rep}]"
    print(f"{tag} building profiles (k={args.k}, kp={args.kp})...", flush=True)
    Xm_t, Xp_t, splits = L.build(regime, fold, esm, chem, seed=split_seed)
    if regime == "inductive_molecule":
        splits = stratified_cold_split(splits, seed=split_seed)         # matched prevalence
    Xm, Xp = Xm_t.numpy(), Xp_t.numpy()
    M, P = Xm.shape[0], Xp.shape[0]

    tr_idx, ytr = H.sup_edges(splits["train"]); te_idx, yte = H.sup_edges(splits["test"])
    m_tr, p_tr = tr_idx[0].numpy(), tr_idx[1].numpy(); ytr_np = ytr.numpy()
    m_te, p_te = te_idx[0].numpy(), te_idx[1].numpy(); yte_np = yte.numpy()
    prev = float(yte_np.mean())

    # interaction matrix from TRAIN edges only (pos / neg indicator channels)
    S_pos = np.zeros((M, P), np.float32); S_neg = np.zeros((M, P), np.float32)
    S_pos[m_tr[ytr_np == 1], p_tr[ytr_np == 1]] = 1.0
    S_neg[m_tr[ytr_np == 0], p_tr[ytr_np == 0]] = 1.0

    nbr_m = knn_idx(Xm, args.k)                 # molecule chemical neighbours
    nbr_p = knn_idx(Xp, args.kp)                # protein sequence neighbours
    FP = mean_rows(S_pos, nbr_m)                # neighbour frac-pos profile [M, P]
    FN = mean_rows(S_neg, nbr_m)
    FP_sm = mean_cols(FP, nbr_p)               # + smoothed over protein neighbours
    FN_sm = mean_cols(FN, nbr_p)
    OWNP_sm = mean_cols(S_pos, nbr_p)          # own row smoothed to protein neighbours (self-prot excluded)
    OWNN_sm = mean_cols(S_neg, nbr_p)

    def build(bar, m, p):
        cols = [Xm[m], Xp[p]]
        if bar == "profile_A":
            cols += [FP[m], FN[m]]
        elif bar == "profile_B":
            cols += [FP[m, p], FN[m, p], FP_sm[m, p], FN_sm[m, p]]
        elif bar == "profile_A_own":
            op, on = S_pos[m].copy(), S_neg[m].copy()
            op[np.arange(len(m)), p] = 0.0; on[np.arange(len(m)), p] = 0.0   # mask target (=label)
            cols += [FP[m], FN[m], op, on]
        elif bar == "profile_B_own":
            cols += [FP[m, p], FN[m, p], FP_sm[m, p], FN_sm[m, p], OWNP_sm[m, p], OWNN_sm[m, p]]
        return np.concatenate([_col(c) for c in cols], axis=1).astype(np.float32)

    bars = CORE_BARS + (OWN_BARS if regime == "transductive" else [])
    rows = []
    for bar in bars:
        print(f"{tag} probe {bar}...", flush=True)
        Xtr, Xte = build(bar, m_tr, p_tr), build(bar, m_te, p_te)
        sc = train_boost(Xtr, ytr_np, Xte, seed=args.boost_seed)
        mm = metrics(yte_np, sc)
        rows.append({"regime": regime, "repeat": rep, "fold": fold, "bar": bar,
                     "n_features": Xtr.shape[1], "test_prevalence": round(prev, 4),
                     **{k: float(mm[k]) for k in METRICS}})
        print(f"{tag} {bar:14s} prev={prev:.3f} " + " ".join(f"{k}={mm[k]:.3f}" for k in METRICS), flush=True)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="molecule neighbours (chemical kNN)")
    ap.add_argument("--kp", type=int, default=10, help="protein neighbours (sequence kNN, for smoothing)")
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None, help="single-unit mode: one regime")
    ap.add_argument("--repeat", type=int, default=None, help="single-unit mode: one repeat id")
    ap.add_argument("--out", default="results/graph/mol_profile_cf/tables/molecule_profile_cf_runs.csv")
    args = ap.parse_args()
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"profile-CF | k={args.k} | kp={args.kp} | boost_seed={args.boost_seed}", flush=True)

    esm, chem = L.load_embeddings()
    esm, chem = standardize(esm), standardize(chem)

    if args.regime is not None and args.repeat is not None:
        units = [(args.regime, args.repeat)]                # one unit (launcher schedules these)
    else:
        units = [(r, rep) for r in args.regimes for rep in REPEATS[r]]

    rows = []
    for regime, rep in units:
        rows += run_unit(regime, rep, esm, chem, args)
        pd.DataFrame(rows).to_csv(out, index=False)         # incremental: survive interruption
    print(f"\nsaved {out.relative_to(_root)} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
