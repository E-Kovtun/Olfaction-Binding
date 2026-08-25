"""Receptor co-response graph (architecture A) — external compute.

The similarity-graph / profile-CF runs built neighbour graphs in EMBEDDING space
(x), which is the boost's own territory -> null. Here the graph topology comes from
y: two RECEPTORS are neighbours if they are activated by SIMILAR MOLECULE SETS
(cosine over their signed columns of the train interaction matrix). This is a
FUNCTIONAL (pharmacological) similarity that ESM sequence embeddings need not
capture -- and it is the one y-structure a cold molecule can reach, because
receptors stay warm in inductive_molecule.

The experiment isolates one question: does smoothing a molecule's binding profile
over CO-RESPONSE receptor neighbours (y) beat smoothing over ESM receptor
neighbours (x)? Same smoothing mechanism, only the neighbour source differs.

Bars (both regimes):
  raw      : [ mol | prot ]                                             baseline
  nbr      : + molecule-neighbour CF read at the target protein          (does a profile help at all)
  nbr_esm  : nbr + profile smoothed over ESM receptor neighbours         (x-space control)
  nbr_cor  : nbr + profile smoothed over CO-RESPONSE receptor neighbours (y-space = architecture A)
Transductive-only (the query's OWN binding row exists -> purest functional-vs-sequence test):
  own_esm  : [ mol | prot | own row smoothed over ESM receptor neighbours ]
  own_cor  : [ mol | prot | own row smoothed over CO-RESPONSE receptor neighbours ]

Everything is parameter-free (numpy + XGBoost). Aggregation weights are BINARY (plain
mean over k neighbours). Missing entries -> separate pos/neg fraction channels.

Leakage: the interaction matrix (and thus the co-response graph) is built from TRAIN
edges ONLY. Neighbour features carry no leak (a neighbour's binding is a different
datapoint). OWN features exclude the target receptor (kNN excludes self), so the label
never enters. In inductive_molecule the query is cold (empty own row) -> own bars are
omitted; the co-response graph is still fully available (receptors are warm).

Repeat semantics: transductive = LoRaX folds 1..5; inductive_molecule = fold 1 with
cold-split seeds 42..46. The inductive cold val/test split is STRATIFIED to matched
prevalence. Embeddings are standardised.

Usage:
  uv run python legacy/scripts/modeling/eval/run_receptor_coresponse_graph.py
  uv run python legacy/scripts/modeling/eval/run_receptor_coresponse_graph.py --regime inductive_molecule --repeat 42
"""
import argparse, pathlib, sys
import numpy as np, pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind.legacy import lorax as L, hetero as H
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
CORE_BARS = ["raw", "nbr", "nbr_esm", "nbr_cor"]
OWN_BARS = ["own_esm", "own_cor"]                       # transductive only (warm query has an own row)
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
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def coresponse_knn(S_pos, S_neg, k):
    """Receptor neighbours by co-response: cosine over signed molecule-response columns.

    Each receptor -> signed vector over molecules (+1 bind / -1 non-bind / 0 untested).
    Two receptors are similar if the molecules agree on binding AND non-binding.
    Returns (nbr [P, k], mean cosine to those neighbours) — the mean is a degeneracy diagnostic.
    """
    C = (S_pos - S_neg).T                              # [P, M] signed response signature per receptor
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
    sim = Cn @ Cn.T
    np.fill_diagonal(sim, -np.inf)
    k = min(k, C.shape[0] - 1)
    nbr = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    row = np.arange(C.shape[0])[:, None]
    mean_sim = float(np.mean(np.clip(sim[row, nbr], -1.0, 1.0)))
    return nbr, mean_sim


def mean_rows(S, nbr):
    """out[i] = mean over j of S[nbr[i, j]] — neighbour-aggregated row profile [N, P]."""
    out = np.zeros_like(S)
    for j in range(nbr.shape[1]):
        out += S[nbr[:, j]]
    return out / nbr.shape[1]


def mean_cols(S, nbr_p):
    """out[:, p] = mean over j of S[:, nbr_p[p, j]] — smooth each column over receptor neighbours."""
    out = np.zeros_like(S)
    for j in range(nbr_p.shape[1]):
        out += S[:, nbr_p[:, j]]
    return out / nbr_p.shape[1]


def neighbour_overlap(nbr_a, nbr_b):
    """Mean Jaccard overlap between two per-node neighbour sets (functional vs sequence graph)."""
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
    """One (regime, repeat) unit -> rows for each bar."""
    fold = rep if regime == "transductive" else 1
    split_seed = 42 if regime == "transductive" else rep
    tag = f"[{regime} rep{rep}]"
    print(f"{tag} building co-response graph (k={args.k}, kp={args.kp})...", flush=True)
    Xm_t, Xp_t, splits = L.build(regime, fold, esm, chem, seed=split_seed)
    if regime == "inductive_molecule":
        splits = stratified_cold_split(splits, seed=split_seed)          # matched prevalence
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

    nbr_m = knn_idx(Xm, args.k)                          # molecule chemical neighbours (for the base profile)
    nbr_p_esm = knn_idx(Xp, args.kp)                     # receptor SEQUENCE neighbours (x-space control)
    nbr_p_cor, cor_sim = coresponse_knn(S_pos, S_neg, args.kp)   # receptor CO-RESPONSE neighbours (y-space)
    overlap = neighbour_overlap(nbr_p_esm, nbr_p_cor)
    print(f"{tag} receptor graph: co-response mean-sim={cor_sim:.3f} | "
          f"ESM/co-response neighbour Jaccard={overlap:.3f}", flush=True)

    FP = mean_rows(S_pos, nbr_m); FN = mean_rows(S_neg, nbr_m)            # molecule-neighbour profile [M, P]
    FP_esm = mean_cols(FP, nbr_p_esm); FN_esm = mean_cols(FN, nbr_p_esm)  # + smoothed over ESM receptors
    FP_cor = mean_cols(FP, nbr_p_cor); FN_cor = mean_cols(FN, nbr_p_cor)  # + smoothed over co-response receptors
    OWN_esm_p = mean_cols(S_pos, nbr_p_esm); OWN_esm_n = mean_cols(S_neg, nbr_p_esm)
    OWN_cor_p = mean_cols(S_pos, nbr_p_cor); OWN_cor_n = mean_cols(S_neg, nbr_p_cor)

    def build(bar, m, p):
        cols = [Xm[m], Xp[p]]
        if bar == "nbr":
            cols += [FP[m, p], FN[m, p]]
        elif bar == "nbr_esm":
            cols += [FP[m, p], FN[m, p], FP_esm[m, p], FN_esm[m, p]]
        elif bar == "nbr_cor":
            cols += [FP[m, p], FN[m, p], FP_cor[m, p], FN_cor[m, p]]
        elif bar == "own_esm":
            cols += [OWN_esm_p[m, p], OWN_esm_n[m, p]]
        elif bar == "own_cor":
            cols += [OWN_cor_p[m, p], OWN_cor_n[m, p]]
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
                     "coresp_mean_sim": round(cor_sim, 4), "esm_coresp_jaccard": round(overlap, 4),
                     **{k: float(mm[k]) for k in METRICS}})
        print(f"{tag} {bar:9s} prev={prev:.3f} " + " ".join(f"{k}={mm[k]:.3f}" for k in METRICS), flush=True)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="molecule neighbours (chemical kNN, base profile)")
    ap.add_argument("--kp", type=int, default=10, help="receptor neighbours (both ESM and co-response)")
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None, help="single-unit mode: one regime")
    ap.add_argument("--repeat", type=int, default=None, help="single-unit mode: one repeat id")
    ap.add_argument("--out", default="results/graph/receptor_coresponse/tables/receptor_coresponse_runs.csv")
    args = ap.parse_args()
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"receptor co-response | k={args.k} | kp={args.kp} | boost_seed={args.boost_seed}", flush=True)

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
