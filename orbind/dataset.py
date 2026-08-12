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
