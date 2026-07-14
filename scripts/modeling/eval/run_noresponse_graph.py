"""Co-NON-response receptor graph (architecture D) — external compute.

Twin of architecture A (receptor co-response), but the receptor graph is built from the
TESTED NEGATIVES we usually ignore: two receptors are neighbours if the SAME molecules FAIL
to activate both (cosine over their S_neg columns). Question: does "receptors rejected by the
same molecules" carry signal complementary to co-response (A) and to ESM sequence?

All three receptor graphs are run side by side so the only thing that changes is the neighbour
source used to smooth the molecule-neighbour binding profile:

Bars (both regimes):
  raw       : [ mol | prot ]
  nbr       : + molecule (chemical) neighbour CF at the target protein
  nbr_esm   : nbr + profile smoothed over ESM receptor neighbours          (x / sequence)
  nbr_cor   : nbr + profile smoothed over CO-RESPONSE receptor neighbours  (y / signed, = arch A)
  nbr_nor   : nbr + profile smoothed over CO-NON-response receptor neighbours (y / negatives, = arch D)

Parameter-free (numpy + XGBoost). Binary aggregation weights. Missing entries -> pos/neg channels.

Leakage: all receptor graphs and the interaction matrix are built from TRAIN edges only; molecule-
neighbour features are a different datapoint; receptor smoothing excludes self. Clean in both regimes.

Repeat semantics: transductive = LoRaX folds 1..5; inductive_molecule = fold 1, cold seeds 42..46.
Inductive cold val/test split is STRATIFIED to matched prevalence. Embeddings are standardised.

Usage:
  uv run python scripts/modeling/eval/run_noresponse_graph.py
  uv run python scripts/modeling/eval/run_noresponse_graph.py --regime inductive_molecule --repeat 42
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
BARS = ["raw", "nbr", "nbr_esm", "nbr_cor", "nbr_nor"]
REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def standardize(d):
    keys = list(d)
    X = np.stack([np.asarray(d[k], np.float64) for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((np.asarray(d[k], np.float64) - mu) / sd).astype(np.float32) for k in keys}


def knn_idx(X, k):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, X.shape[0] - 1)
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def column_knn(Cols, k):
    """Receptor neighbours by cosine over a [P, M] column-signature matrix (self excluded).

    Returns (nbr [P, k], mean cosine to neighbours). Used for both co-response (signed columns)
    and co-non-response (S_neg columns) receptor graphs.
    """
    Cn = Cols / (np.linalg.norm(Cols, axis=1, keepdims=True) + 1e-8)
    sim = Cn @ Cn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, Cols.shape[0] - 1)
    nbr = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    row = np.arange(Cols.shape[0])[:, None]
    mean_sim = float(np.mean(np.clip(sim[row, nbr], -1.0, 1.0)))
    return nbr, mean_sim


def mean_rows(S, nbr):
    out = np.zeros_like(S)
    for j in range(nbr.shape[1]):
        out += S[nbr[:, j]]
    return out / nbr.shape[1]


def mean_cols(S, nbr_p):
    out = np.zeros_like(S)
    for j in range(nbr_p.shape[1]):
        out += S[:, nbr_p[:, j]]
    return out / nbr_p.shape[1]


def neighbour_overlap(nbr_a, nbr_b):
    j = []
    for a, b in zip(nbr_a, nbr_b):
        sa, sb = set(a.tolist()), set(b.tolist())
        u = len(sa | sb)
        j.append(len(sa & sb) / u if u else 0.0)
    return float(np.mean(j))


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
    print(f"{tag} building receptor graphs (k={args.k}, kp={args.kp})...", flush=True)
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

    nbr_m = knn_idx(Xm, args.k)                                   # molecule chemical neighbours (base profile)
    nbr_p_esm = knn_idx(Xp, args.kp)                             # receptor sequence neighbours (x)
    nbr_p_cor, cor_sim = column_knn((S_pos - S_neg).T, args.kp)  # co-response (signed columns, y)
    nbr_p_nor, nor_sim = column_knn(S_neg.T, args.kp)            # co-NON-response (negative columns, y)
    ov_esm_nor = neighbour_overlap(nbr_p_esm, nbr_p_nor)
    ov_cor_nor = neighbour_overlap(nbr_p_cor, nbr_p_nor)
    print(f"{tag} receptor graphs: co-response sim={cor_sim:.3f} | co-non-response sim={nor_sim:.3f} | "
          f"Jaccard nor-vs-esm={ov_esm_nor:.3f} nor-vs-cor={ov_cor_nor:.3f}", flush=True)

    FP = mean_rows(S_pos, nbr_m); FN = mean_rows(S_neg, nbr_m)            # molecule-neighbour profile
    FP_esm = mean_cols(FP, nbr_p_esm); FN_esm = mean_cols(FN, nbr_p_esm)
    FP_cor = mean_cols(FP, nbr_p_cor); FN_cor = mean_cols(FN, nbr_p_cor)
    FP_nor = mean_cols(FP, nbr_p_nor); FN_nor = mean_cols(FN, nbr_p_nor)

    def build(bar, m, p):
        cols = [Xm[m], Xp[p]]
        if bar == "nbr":
            cols += [FP[m, p], FN[m, p]]
        elif bar == "nbr_esm":
            cols += [FP[m, p], FN[m, p], FP_esm[m, p], FN_esm[m, p]]
        elif bar == "nbr_cor":
            cols += [FP[m, p], FN[m, p], FP_cor[m, p], FN_cor[m, p]]
        elif bar == "nbr_nor":
            cols += [FP[m, p], FN[m, p], FP_nor[m, p], FN_nor[m, p]]
        return np.concatenate([_col(c) for c in cols], axis=1).astype(np.float32)

    rows = []
    for bar in BARS:
        print(f"{tag} probe {bar}...", flush=True)
        Xtr, Xte = build(bar, m_tr, p_tr), build(bar, m_te, p_te)
        sc = train_boost(Xtr, ytr_np, Xte, seed=args.boost_seed)
        mm = metrics(yte_np, sc)
        rows.append({"regime": regime, "repeat": rep, "fold": fold, "bar": bar,
                     "n_features": Xtr.shape[1], "test_prevalence": round(prev, 4),
                     "coresp_sim": round(cor_sim, 4), "nonresp_sim": round(nor_sim, 4),
                     "jaccard_nor_esm": round(ov_esm_nor, 4), "jaccard_nor_cor": round(ov_cor_nor, 4),
                     **{k: float(mm[k]) for k in METRICS}})
        print(f"{tag} {bar:8s} prev={prev:.3f} " + " ".join(f"{k}={mm[k]:.3f}" for k in METRICS), flush=True)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="molecule neighbours (chemical kNN, base profile)")
    ap.add_argument("--kp", type=int, default=10, help="receptor neighbours (esm / co-response / co-non-response)")
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None)
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--out", default="results/graph/noresponse/tables/noresponse_runs.csv")
    args = ap.parse_args()
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"co-non-response receptor graph | k={args.k} | kp={args.kp} | boost_seed={args.boost_seed}", flush=True)

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
