"""Collaborative factorization (architecture B) — external compute.

Isolates the "y-ceiling": factorize ONLY the train interaction matrix (no embeddings),
A_signed = S_pos - S_neg  ~=  U V^T (truncated SVD, rank r), giving a molecule latent
factor u_m and a receptor latent factor v_p that are learned PURELY from who-binds-what.
Then measure how much this pure-y representation adds on top of (or instead of) the raw
embeddings the boost already reads.

Bars (both regimes):
  raw      : [ mol | prot ]                          baseline (x only)
  mf_prot  : + v_p    (receptor collaborative factor; available in both regimes)
  mf_mol   : + u_m    (molecule collaborative factor)
  mf_both  : + u_m + v_p
  mf_only  : [ u_m | v_p ]   NO embeddings -> pure collaborative signal (the y-ceiling in isolation)

Cold-start is the whole point of the diagnostic: in inductive_molecule the held-out
molecule contributes NO train edges, so its matrix row is all-zero and u_m ~= 0 -> the
molecule factor collapses and mf_mol / mf_both / mf_only degrade to protein-only. That
visible collapse IS the result: pure-y molecule factors cannot generalise to a cold
molecule (classic CF cold-start), which is exactly why the interaction graph needs
node features. In transductive both factors are informative.

Leakage: the factorized matrix contains TRAIN edges only; a test pair's entry is absent,
so u_m . v_p at a test pair is a genuine collaborative prediction, never the held-out label
(standard transductive-MF evaluation / stacked generalisation).

Repeat semantics: transductive = LoRaX folds 1..5; inductive_molecule = fold 1, cold-split
seeds 42..46. Inductive cold val/test split is STRATIFIED to matched prevalence. Embeddings
are standardised (the MF factors keep their own SVD scale; the boost is scale-invariant).

Usage:
  uv run python scripts/modeling/eval/run_collaborative_factorization.py
  uv run python scripts/modeling/eval/run_collaborative_factorization.py --regime transductive --repeat 1 --rank 32
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
BARS = ["raw", "mf_prot", "mf_mol", "mf_both", "mf_only"]
REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def standardize(d):
    keys = list(d)
    X = np.stack([np.asarray(d[k], np.float64) for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((np.asarray(d[k], np.float64) - mu) / sd).astype(np.float32) for k in keys}


def factorize(S_pos, S_neg, rank):
    """Truncated SVD of the signed train interaction matrix -> (u_m [M,r], v_p [P,r], sv_energy).

    Cold molecules have all-zero rows -> their u_m rows come out ~0 (the cold-start collapse).
    Singular values are split sqrt/sqrt across the two factors so u_m . v_p reconstructs A.
    """
    A = (S_pos - S_neg).astype(np.float32)                 # [M, P] signed (untested = 0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = int(min(rank, len(s)))
    sr = np.sqrt(s[:r])
    Um = (U[:, :r] * sr).astype(np.float32)                # [M, r]
    Vp = (Vt[:r].T * sr).astype(np.float32)                # [P, r]
    energy = float((s[:r] ** 2).sum() / max((s ** 2).sum(), 1e-12))
    return Um, Vp, energy


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
    print(f"{tag} factorizing interaction matrix (rank={args.rank})...", flush=True)
    Xm_t, Xp_t, splits = L.build(regime, fold, esm, chem, seed=split_seed)
    if regime == "inductive_molecule":
        splits = stratified_cold_split(splits, seed=split_seed)         # matched prevalence
    Xm, Xp = Xm_t.numpy(), Xp_t.numpy()
    M, P = Xm.shape[0], Xp.shape[0]

    tr_idx, ytr = H.sup_edges(splits["train"]); te_idx, yte = H.sup_edges(splits["test"])
    m_tr, p_tr = tr_idx[0].numpy(), tr_idx[1].numpy(); ytr_np = ytr.numpy()
    m_te, p_te = te_idx[0].numpy(), te_idx[1].numpy(); yte_np = yte.numpy()
    prev = float(yte_np.mean())

    # interaction matrix from TRAIN edges only
    S_pos = np.zeros((M, P), np.float32); S_neg = np.zeros((M, P), np.float32)
    S_pos[m_tr[ytr_np == 1], p_tr[ytr_np == 1]] = 1.0
    S_neg[m_tr[ytr_np == 0], p_tr[ytr_np == 0]] = 1.0

    Um, Vp, energy = factorize(S_pos, S_neg, args.rank)
    # fraction of molecules whose factor is ~zero (cold-start collapse), for the diagnostic
    cold_frac = float((np.linalg.norm(Um, axis=1) < 1e-6).mean())
    print(f"{tag} SVD: rank={Um.shape[1]} | sv_energy={energy:.3f} | zero-factor mols={cold_frac:.3f}", flush=True)

    def build(bar, m, p):
        if bar == "raw":
            cols = [Xm[m], Xp[p]]
        elif bar == "mf_prot":
            cols = [Xm[m], Xp[p], Vp[p]]
        elif bar == "mf_mol":
            cols = [Xm[m], Xp[p], Um[m]]
        elif bar == "mf_both":
            cols = [Xm[m], Xp[p], Um[m], Vp[p]]
        elif bar == "mf_only":
            cols = [Um[m], Vp[p]]
        return np.concatenate([_col(c) for c in cols], axis=1).astype(np.float32)

    rows = []
    for bar in BARS:
        print(f"{tag} probe {bar}...", flush=True)
        Xtr, Xte = build(bar, m_tr, p_tr), build(bar, m_te, p_te)
        sc = train_boost(Xtr, ytr_np, Xte, seed=args.boost_seed)
        mm = metrics(yte_np, sc)
        rows.append({"regime": regime, "repeat": rep, "fold": fold, "bar": bar,
                     "n_features": Xtr.shape[1], "test_prevalence": round(prev, 4),
                     "mf_rank": Um.shape[1], "sv_energy": round(energy, 4),
                     "zero_factor_mols": round(cold_frac, 4),
                     **{k: float(mm[k]) for k in METRICS}})
        print(f"{tag} {bar:8s} prev={prev:.3f} " + " ".join(f"{k}={mm[k]:.3f}" for k in METRICS), flush=True)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=32, help="truncated-SVD rank (latent dimension)")
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None, help="single-unit mode: one regime")
    ap.add_argument("--repeat", type=int, default=None, help="single-unit mode: one repeat id")
    ap.add_argument("--out", default="results/graph/collab_factorization/tables/collab_factorization_runs.csv")
    args = ap.parse_args()
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"collaborative factorization | rank={args.rank} | boost_seed={args.boost_seed}", flush=True)

    esm, chem = L.load_embeddings()
    esm, chem = standardize(esm), standardize(chem)

    if args.regime is not None and args.repeat is not None:
        units = [(args.regime, args.repeat)]
    else:
        units = [(r, rep) for r in args.regimes for rep in REPEATS[r]]

    rows = []
    for regime, rep in units:
        rows += run_unit(regime, rep, esm, chem, args)
        pd.DataFrame(rows).to_csv(out, index=False)         # incremental: survive interruption
    print(f"\nsaved {out.relative_to(_root)} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
