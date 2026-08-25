"""Metapath M-P-M-P collaborative filtering (architecture C) — external compute.

profile-CF (3b) borrowed from CHEMICAL neighbours (x). Here neighbours are defined by
CO-BINDING (y): "molecules that share receptors with me". For a query (m, X) we ask
whether the molecules that co-bind with m (share receptor targets) also bind X — the
classic M-P-M-P metapath / item-based CF where molecule similarity is measured from the
interaction matrix, NOT from ChemBERTa. This is the y-native cousin of profile-CF.

Bars (both regimes):
  raw       : [ mol | prot ]                                   baseline
  meta      : + co-binding-neighbour CF read at the target protein   (scalars)
  meta_full : + the whole co-binding-neighbour profile over all P proteins (fat; boost finds structure)

Cold-start collapse (the honest result in inductive_molecule): a held-out molecule has an
EMPTY train row → no co-binding signature → its metapath features are zeroed → meta ~= raw.
Pure-y molecule neighbourhoods cannot reach a cold molecule (same wall as MF, architecture B).
In transductive both are live.

Aggregation weights are BINARY (mean over k co-binding neighbours). Missing entries -> separate
pos/neg fraction channels.

Leakage: co-binding neighbours exclude self (m' != m) and the interaction matrix is TRAIN-only,
so a test pair (m,X) is absent (S[m,X]=0) → the neighbour selection for a test query does not see
the label and the metapath read at X uses only other molecules' train bindings. (For TRAIN pairs the
feature is mildly optimistic because m's signature contains the target column while picking
neighbours — standard transductive-CF stacking; test metrics stay honest.)

Repeat semantics: transductive = LoRaX folds 1..5; inductive_molecule = fold 1, cold seeds 42..46.
Inductive cold val/test split is STRATIFIED to matched prevalence. Embeddings are standardised.

Usage:
  uv run python scripts/modeling/eval/run_metapath_cf.py
  uv run python scripts/modeling/eval/run_metapath_cf.py --regime transductive --repeat 1
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
BARS = ["raw", "meta", "meta_full"]
REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def standardize(d):
    keys = list(d)
    X = np.stack([np.asarray(d[k], np.float64) for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((np.asarray(d[k], np.float64) - mu) / sd).astype(np.float32) for k in keys}


def cobinding_knn(R, k):
    """Molecule neighbours by CO-BINDING: cosine over signed interaction rows.

    Returns (nbr [M, k], mean cosine to neighbours, valid mask [M]).
    Molecules with an all-zero row (cold, no train edges) get valid=False -> their metapath
    features are zeroed downstream (cold-start collapse).
    """
    n = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / (n + 1e-8)
    sim = Rn @ Rn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, R.shape[0] - 1)
    nbr = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    valid = (n[:, 0] > 1e-8)
    row = np.arange(R.shape[0])[:, None]
    ms = sim[row, nbr]
    mean_sim = float(np.mean(np.clip(ms[valid], -1.0, 1.0))) if valid.any() else 0.0
    return nbr, mean_sim, valid


def mean_rows(S, nbr):
    out = np.zeros_like(S)
    for j in range(nbr.shape[1]):
        out += S[nbr[:, j]]
    return out / nbr.shape[1]


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
    fold = rep if regime == "transductive" else 1
    split_seed = 42 if regime == "transductive" else rep
    tag = f"[{regime} rep{rep}]"
    print(f"{tag} building co-binding metapath (k={args.k})...", flush=True)
    Xm_t, Xp_t, splits = L.build(regime, fold, esm, chem, seed=split_seed)
    if regime == "inductive_molecule":
        splits = stratified_cold_split(splits, seed=split_seed)          # matched prevalence
    Xm, Xp = Xm_t.numpy(), Xp_t.numpy()
    M, P = Xm.shape[0], Xp.shape[0]

    tr_idx, ytr = H.sup_edges(splits["train"]); te_idx, yte = H.sup_edges(splits["test"])
    m_tr, p_tr = tr_idx[0].numpy(), tr_idx[1].numpy(); ytr_np = ytr.numpy()
    m_te, p_te = te_idx[0].numpy(), te_idx[1].numpy(); yte_np = yte.numpy()
    prev = float(yte_np.mean())

    S_pos = np.zeros((M, P), np.float32); S_neg = np.zeros((M, P), np.float32)
    S_pos[m_tr[ytr_np == 1], p_tr[ytr_np == 1]] = 1.0
    S_neg[m_tr[ytr_np == 0], p_tr[ytr_np == 0]] = 1.0

    R = S_pos - S_neg                                    # signed molecule signatures
    nbr_m, cobind_sim, valid = cobinding_knn(R, args.k)  # co-binding molecule neighbours
    cold_frac = float((~valid).mean())
    print(f"{tag} metapath: co-binding mean-sim={cobind_sim:.3f} | zero-signature mols={cold_frac:.3f}", flush=True)

    FPy = mean_rows(S_pos, nbr_m); FNy = mean_rows(S_neg, nbr_m)   # co-binding-neighbour profile [M, P]
    FPy[~valid] = 0.0; FNy[~valid] = 0.0                            # cold molecules: collapse to zero

    def build(bar, m, p):
        cols = [Xm[m], Xp[p]]
        if bar == "meta":
            cols += [FPy[m, p], FNy[m, p]]
        elif bar == "meta_full":
            cols += [FPy[m], FNy[m]]
        return np.concatenate([_col(c) for c in cols], axis=1).astype(np.float32)

    rows = []
    for bar in BARS:
        print(f"{tag} probe {bar}...", flush=True)
        Xtr, Xte = build(bar, m_tr, p_tr), build(bar, m_te, p_te)
        sc = train_boost(Xtr, ytr_np, Xte, seed=args.boost_seed)
        mm = metrics(yte_np, sc)
        rows.append({"regime": regime, "repeat": rep, "fold": fold, "bar": bar,
                     "n_features": Xtr.shape[1], "test_prevalence": round(prev, 4),
                     "cobind_mean_sim": round(cobind_sim, 4), "zero_signature_mols": round(cold_frac, 4),
                     **{k: float(mm[k]) for k in METRICS}})
        print(f"{tag} {bar:9s} prev={prev:.3f} " + " ".join(f"{k}={mm[k]:.3f}" for k in METRICS), flush=True)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="co-binding molecule neighbours")
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--out", default="results/graph/metapath_cf/tables/metapath_cf_runs.csv")
    args = ap.parse_args()
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"metapath M-P-M-P | k={args.k} | boost_seed={args.boost_seed}", flush=True)

    esm, chem = L.load_embeddings()
    esm, chem = standardize(esm), standardize(chem)

    if args.regime is not None and args.repeat is not None:
        units = [(args.regime, args.repeat)]
    else:
        units = [(r, rep) for r in args.regimes for rep in REPEATS[r]]

    rows = []
    for regime, rep in units:
        rows += run_unit(regime, rep, esm, chem, args)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out.relative_to(_root)} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
