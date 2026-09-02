"""Assemble the MP (molecule + protein) dataset and make weighted splits.

Pairs come from preprocessing (`pairs_curated.csv`: receptor = sequence, inchikey,
label). Embeddings come from the two generation scripts (`.npz` with `ids`,`emb`).
We keep only pairs whose BOTH sides have an embedding, concatenate
[molecule || protein], and split train/test either:

  * "stratified"      — random split stratified by label (as in LORAX's MP);
  * "group_receptor"  — all pairs of a receptor stay on one side (no leakage from
                        near-identical paralogs/sequences).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def load_npz_dict(path) -> dict:
    d = np.load(path, allow_pickle=True)
    if "ids" in d.files:
        return {k: v for k, v in zip(d["ids"].tolist(), d["emb"])}
    return {k: d[k] for k in d.files}


def assemble(pairs_csv, prot_npz, mol_npz, random_prot=False, seed=0):
    """Return (X, y, pairs) where X = [molecule_emb || protein_emb].

    random_prot: replace the protein block with independent seeded noise PER
    ROW (the "mock protein" floor). Per-row (not per-receptor) noise is the
    correct control — a consistent per-receptor vector would leak receptor
    identity to tree models on splits where receptors are seen.
    """
    pairs = pd.read_csv(pairs_csv)
    prot = load_npz_dict(prot_npz)   # sequence -> protein vector
    mol = load_npz_dict(mol_npz)     # inchikey -> molecule vector

    mask = pairs["receptor"].isin(prot) & pairs["inchikey"].isin(mol)
    dropped = int((~mask).sum())
    pairs = pairs[mask].reset_index(drop=True)
    if dropped:
        print(f"  dropped {dropped} pairs lacking an embedding")

    Xm = np.stack([mol[i] for i in pairs["inchikey"]]).astype(np.float32)
    if random_prot:
        dim = next(iter(prot.values())).shape[0]
        rng = np.random.default_rng(seed)
        Xp = rng.standard_normal((len(pairs), dim)).astype(np.float32)   # independent noise per row
    else:
        Xp = np.stack([prot[r] for r in pairs["receptor"]]).astype(np.float32)
    X = np.concatenate([Xm, Xp], axis=1)
    y = pairs["label"].to_numpy().astype(np.float32)
    print(f"  assembled X={X.shape} (mol {Xm.shape[1]} + prot {Xp.shape[1]}{', RANDOM' if random_prot else ''}), "
          f"positives={int(y.sum())}/{len(y)}")
    return X, y, pairs


def metrics(y, p):
    """Imbalance-aware binary metrics from labels y and scores p."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 matthews_corrcoef, f1_score, precision_score, recall_score)
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }


def regression_metrics(y, p):
    """Continuous-target metrics, for the Carey/Hallem response magnitude.

    `R2` is sklearn's: the reference is the *test* set's own mean. That is the
    same definition upstream reports, and it is why their "naive" row (predict
    the TRAIN mean) can be strongly negative -- the two means differ. Read a
    model's R2 against that naive row, never against 0.

    A constant predictor has no defined correlation with anything; rather than
    emit a nan that poisons every downstream mean, Pearson/Spearman are
    reported as 0.0 in that case (it is exactly the no-information value).
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    constant = len(p) == 0 or bool(np.allclose(p, p[0]))
    return {
        "R2": float(r2_score(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "Pearson": 0.0 if constant else float(pearsonr(y, p)[0]),
        "Spearman": 0.0 if constant else float(spearmanr(y, p).statistic),
    }


# The task axis (see orbind/tasks.py): which metric family a run reports.
METRICS = {"classification": metrics, "regression": regression_metrics}


def split(pairs, y, kind="stratified", test_size=0.2, seed=42):
    """Return boolean train/test masks over the rows of `pairs`."""
    n = len(y)
    rng = np.random.default_rng(seed)
    if kind == "stratified":
        from sklearn.model_selection import train_test_split
        idx_tr, idx_te = train_test_split(
            np.arange(n), test_size=test_size, random_state=seed, stratify=y)
    elif kind in ("group_receptor", "group_molecule"):
        # hold out whole receptors (or molecules): no entity appears on both sides
        col = "receptor" if kind == "group_receptor" else "inchikey"
        vals = pairs[col].to_numpy()
        uniq = rng.permutation(np.unique(vals))
        te_grp, n_te = set(), 0
        for g in uniq:
            if n_te >= test_size * n:
                break
            te_grp.add(g); n_te += int((vals == g).sum())
        idx_te = np.where(pairs[col].isin(te_grp))[0]
        idx_tr = np.where(~pairs[col].isin(te_grp))[0]
    else:
        raise ValueError(kind)

    tr = np.zeros(n, bool); te = np.zeros(n, bool)
    tr[idx_tr] = True; te[idx_te] = True
    return tr, te


# ---------------------------------------------------------------------------
# The full metric batteries. `metrics` / `regression_metrics` above stay exactly
# as they were -- every number already reported came out of them, and what follows
# is a superset that adds columns without moving one.

def _binary_block(y_bin, score, hard):
    """AUROC/AUPRC from a continuous score, MCC/F1/precision/recall from a hard call.

    The hard call is NOT `score >= 0.5`: on a z-scored response the prediction is in
    the same units as the truth, so it is thresholded at the very same cut the truth
    was binarised with. Anything else would score the model against a boundary it was
    never asked to respect.

    AUROC and AUPRC are NaN when the fold has one class only -- undefined, not zero,
    and a sweep must not die on it.
    """
    import numpy as np
    from sklearn.metrics import (average_precision_score, f1_score,
                                 matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    y_bin, hard = np.asarray(y_bin).astype(int), np.asarray(hard).astype(int)
    both = 0 < y_bin.sum() < len(y_bin)
    return {
        "AUROC": float(roc_auc_score(y_bin, score)) if both else float("nan"),
        "AUPRC": float(average_precision_score(y_bin, score)) if both else float("nan"),
        "MCC": float(matthews_corrcoef(y_bin, hard)) if both else float("nan"),
        "F1": float(f1_score(y_bin, hard, zero_division=0)),
        "precision": float(precision_score(y_bin, hard, zero_division=0)),
        "recall": float(recall_score(y_bin, hard, zero_division=0)),
    }


def binarisations(y_te, pred, rec_te=None, y_tr=None, rec_tr=None, which=("rec0",)):
    """Ways of calling a continuous response "a response".

    `rec0` is the reference of record: above that receptor's own TRAIN centre. It is
    the convention the graph's own edge signs already use (`_mp_edges` with
    `edge_center="per_receptor"`), and it removes the between-receptor baseline and
    dynamic-range heterogeneity that a fixed cut inherits -- without it a discrete
    metric largely reports which receptors are active rather than whether the odorants
    were called right. `glob0` (y > 0, "above the pool average" on a z-score) is kept
    because it needs no fitting at all, which makes it the fallback when there is no
    train split to centre on.

    THE CENTRE IS FIT ON TRAIN. Taken from the test rows it would let the label
    definition itself see the held-out data -- a leak that flatters whichever arm is
    being scored, in a column that looks like every other.

    Yields (name, y_bin, score, hard); `score` stays continuous for the ranking
    metrics, `hard` is the same cut applied to the prediction.
    """
    import numpy as np
    y_te, pred = np.asarray(y_te, float), np.asarray(pred, float)
    cuts = {"glob0": np.zeros(len(y_te))}          # needs no fitting
    if rec_te is not None and rec_tr is not None and y_tr is not None:
        import pandas as pd
        y_tr, rec_te = np.asarray(y_tr, float), np.asarray(rec_te)
        mean_r = pd.Series(y_tr).groupby(pd.Series(np.asarray(rec_tr))).mean()
        glob = float(np.mean(y_tr))
        cuts["rec0"] = np.array([mean_r.get(r, glob) for r in rec_te], float)
    for name, c in cuts.items():
        if which is None or name in which:
            yield name, (y_te > c), (pred - c), (pred > c)


def regression_metrics_full(y, p, rec_te=None, mol_te=None, y_tr=None, rec_tr=None,
                            which=("rec0",)):
    """Everything scoreable on a continuous response, in one row.

    Two families: the pooled regression scores upstream reports, and the binary
    battery after binarising the response at one reference (see `binarisations`).
    Discrete columns take a `{reference}_` prefix, so `rec0_AUROC` and friends.

    `mol_te` is accepted and ignored -- callers pass it, and dropping it from the
    signature would be a silent behaviour change at the call site.
    """
    import numpy as np
    from scipy.stats import kendalltau
    out = dict(regression_metrics(y, p))
    y, p = np.asarray(y, float), np.asarray(p, float)
    constant = len(p) == 0 or bool(np.allclose(p, p[0]))
    out["Kendall"] = 0.0 if constant else float(kendalltau(y, p).statistic)
    for name, y_bin, score, hard in binarisations(y, p, rec_te, y_tr, rec_tr, which):
        for k, v in _binary_block(y_bin, score, hard).items():
            out[f"{name}_{k}"] = v
    return out


def classification_metrics_full(y, p, rec_te=None, mol_te=None, **_ignored):
    """`metrics` in full: the four headline scores plus precision and recall, which
    were always computed and then dropped on the way into the row.

    The group arguments are accepted and ignored so both batteries take the same call
    from the sweep.
    """
    return dict(metrics(y, p))


# The column names the batteries emit, so a reader never hardcodes them and drifts.
BINARISATIONS = ("rec0",)
_DISCRETE = ("AUROC", "AUPRC", "MCC", "F1", "precision", "recall")
METRIC_NAMES = {
    "regression": (["R2", "RMSE", "MAE", "Pearson", "Spearman", "Kendall"]
                   + [f"{c}_{m}" for c in BINARISATIONS for m in _DISCRETE]),
    "classification": list(_DISCRETE),
}
# The short list a progress line or a compact dump shows.
METRIC_HEADLINE = {"regression": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
                   "classification": ["AUROC", "AUPRC", "MCC", "F1"]}
# The wide batteries, for runs that want every column and will decide what to read
# later. Same call as METRICS plus optional context (group ids, the train split).
METRICS_FULL = {"classification": classification_metrics_full,
                "regression": regression_metrics_full}
