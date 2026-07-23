"""Configurable multi-source boosting ensemble (ProSmith/LORAX-style),
generalized to an arbitrary combination of pluggable embedding extractors.

Design
------
Every source is an *extractor*: given row-positions into a shared `pairs`
table (columns at least `receptor`, `inchikey`, `label`), it hands back
row-aligned feature matrices for train/val/test. What happens inside is the
extractor's business -- a static npz lookup for label-independent sources
(ESM, GIN), or a freshly-trained model producing leakage-free out-of-fold
vectors for label-dependent sources (e.g. a future attention/cls extractor).
The engine below never sees embeddings directly; it only asks each extractor
for (Xtr, Xval, Xte) and concatenates whichever subset a given combo needs.

    class Extractor(Protocol):
        name: str
        def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed,
                           checkpoint_dir=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

`checkpoint_dir`, if given, is where an extractor may persist whatever it
trained (e.g. a pair-level source's whole-train-fit model) -- entity-level
extractors have nothing to train and just ignore it.

One combo = one boosting head, trained on the concatenation of its sources'
features. The `combo_spec` mini-language names combos by digit strings over
the 1-based position of each extractor as given in the `extractors` dict:
`"1 23 123"` with `extractors={"cls":..., "prot":..., "mol":...}` means
[solo(cls), pair(prot, mol), triplet(cls, prot, mol)].

Final prediction is a weighted combination of the per-combo boosting heads,
fit on validation predictions (`fit_ensemble_weights`), evaluated once on
test. Extractor outputs are cached per name so a source shared by several
combos (e.g. cls appearing in both "1" and "123") is computed once per run.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import dataset as D
from .baselines import fit_boost, tune_boost


# --------------------------------------------------------------------------- extractors

@dataclass
class EntityExtractor:
    """Static, label-independent embedding lookup by a single id column
    (`receptor` sequence for proteins, `inchikey` for molecules).

    `model_name`/`pooling` are pure provenance/logging metadata -- swapping
    the underlying embedding (e.g. ESM-2 -> ESM-1b, or a different GIN
    pretraining objective) means constructing a new extractor with a
    different `path`/`model_name`, no code change here.
    """
    name: str
    path: str
    key_col: str
    model_name: str
    pooling: str = "mean"
    dim: int = field(init=False)

    def __post_init__(self):
        self._vecs = D.load_npz_dict(self.path)
        self.dim = int(next(iter(self._vecs.values())).shape[0])

    def covered(self, pairs: pd.DataFrame, idx: np.ndarray) -> np.ndarray:
        """Boolean mask over `idx`: True where this extractor has an embedding."""
        keys = pairs[self.key_col].to_numpy()[idx]
        return np.fromiter((k in self._vecs for k in keys), dtype=bool, count=len(keys))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        # nothing to train, nothing to checkpoint -- static lookup only.
        def lookup(idx):
            keys = pairs[self.key_col].to_numpy()[idx]
            missing = sorted({k for k in keys if k not in self._vecs})
            if missing:
                raise KeyError(
                    f"{self.name}: {len(missing)} {self.key_col}(s) missing an embedding "
                    f"in {self.path}, e.g. {missing[0]!r}")
            return np.stack([self._vecs[k] for k in keys]).astype(np.float32)
        return lookup(train_idx), lookup(val_idx), lookup(test_idx)


def EsmExtractor(name, path, model_name="esm2_t33_650M_UR50D", pooling="mean"):
    """Protein embedding extractor, keyed by `receptor` (amino-acid sequence).
    Default model_name records the exact checkpoint this repo uses
    (scripts/embedding_generation/proteins/02_embed_receptors.py)."""
    return EntityExtractor(name=name, path=path, key_col="receptor",
                            model_name=model_name, pooling=pooling)


def GinExtractor(name, path, model_name="gin_supervised_contextpred", pooling="mean"):
    """Molecule embedding extractor, keyed by `inchikey`. Default model_name
    records the exact pretrained GIN variant this repo uses
    (scripts/embedding_generation/molecules/embed_molecules_gin.py)."""
    return EntityExtractor(name=name, path=path, key_col="inchikey",
                            model_name=model_name, pooling=pooling)


# --------------------------------------------------------------------------- combo spec

def parse_combo_spec(spec: str, names: list[str]) -> list[tuple[str, ...]]:
    """"1 23 123" with names=["cls","prot","mol"] ->
    [("cls",), ("prot","mol"), ("cls","prot","mol")]. Digits are 1-based
    positions into `names`, matching the order extractors were given."""
    combos: list[tuple[str, ...]] = []
    for token in spec.split():
        if not token.isdigit():
            raise ValueError(f"combo token {token!r} must be a digit string, e.g. '123'")
        idxs = [int(ch) for ch in token]
        if any(i < 1 or i > len(names) for i in idxs):
            raise ValueError(
                f"combo token {token!r} references an out-of-range source index "
                f"(have {len(names)} sources: {names})")
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"combo token {token!r} repeats a source index")
        combos.append(tuple(names[i - 1] for i in idxs))
    if len(set(combos)) != len(combos):
        raise ValueError(f"combo spec {spec!r} has duplicate combos after parsing")
    if not combos:
        raise ValueError(f"combo spec {spec!r} parsed to zero combos")
    return combos


# --------------------------------------------------------------------------- splitting

def three_way_split(pairs: pd.DataFrame, y: np.ndarray, kind: str, seed: int,
                     test_size: float = 0.2, val_size: float = 0.2):
    """Train/val/test index arrays (positions into `pairs`), reusing
    `orbind.dataset.split` twice so group-based splits (group_molecule /
    group_receptor) stay leak-free at both cuts."""
    tr_va_mask, te_mask = D.split(pairs, y, kind=kind, test_size=test_size, seed=seed)
    sub_pairs = pairs.loc[tr_va_mask].reset_index(drop=True)
    sub_y = y[tr_va_mask]
    rel_val = val_size / (1.0 - test_size)
    tr_mask, va_mask = D.split(sub_pairs, sub_y, kind=kind, test_size=rel_val, seed=seed + 1)

    orig_idx = np.where(tr_va_mask)[0]
    train_idx = orig_idx[tr_mask]
    val_idx = orig_idx[va_mask]
    test_idx = np.where(te_mask)[0]
    assert len(set(train_idx) & set(val_idx) & set(test_idx)) == 0
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- ensembling

@dataclass
class EnsembleCombiner:
    combos: list[tuple[str, ...]]
    method: str
    weights: dict[tuple[str, ...], float]
    predict_fn: Callable[[np.ndarray], np.ndarray]
    intercept: float | None = None

    def predict(self, preds: dict[tuple[str, ...], np.ndarray]) -> np.ndarray:
        P = np.column_stack([preds[c] for c in self.combos])
        return self.predict_fn(P)


def _fit_simplex_weights(P: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]]) -> EnsembleCombiner:
    """Non-negative weights summing to 1, minimizing validation log-loss.
    Directly comparable to the ProSmith/LORAX weighting scheme."""
    from scipy.optimize import minimize
    K = P.shape[1]
    eps = 1e-6

    def neg_log_loss(w):
        p = np.clip(P @ w, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    w0 = np.full(K, 1.0 / K)
    res = minimize(neg_log_loss, w0, method="SLSQP", bounds=[(0.0, 1.0)] * K,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    w = res.x
    return EnsembleCombiner(combos=combos, method="simplex",
                             weights=dict(zip(combos, w.tolist())),
                             predict_fn=lambda P_, w=w: P_ @ w)


def _fit_logreg_weights(P: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]]) -> EnsembleCombiner:
    """Logistic-regression stacker over the K combo predictions. Coefficients
    are unconstrained (not a simplex) -- read them as relative importance,
    not literal mixing weights."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(P, y)
    return EnsembleCombiner(combos=combos, method="logreg",
                             weights=dict(zip(combos, model.coef_[0].tolist())),
                             intercept=float(model.intercept_[0]),
                             predict_fn=lambda P_: model.predict_proba(P_)[:, 1])


_WEIGHT_FITTERS = {"simplex": _fit_simplex_weights, "logreg": _fit_logreg_weights}


def fit_ensemble_weights(val_preds: dict[tuple[str, ...], np.ndarray], y_val: np.ndarray,
                          method: str = "both") -> dict[str, EnsembleCombiner]:
    combos = list(val_preds.keys())
    P = np.column_stack([val_preds[c] for c in combos])
    methods = list(_WEIGHT_FITTERS) if method == "both" else [method]
    return {m: _WEIGHT_FITTERS[m](P, y_val, combos) for m in methods}


def _coverage_intersection(pairs: pd.DataFrame, extractors: dict[str, object],
                            names: list[str], idx: np.ndarray) -> np.ndarray:
    """Boolean mask over `idx`: True where every extractor in `names` has an
    embedding for that row. Used to drop rows lacking coverage -- callers
    must pass the union of extractor names across ALL combos in the run, not
    just one combo's, so every combo (and the final ensemble) is fit/scored
    on the exact same rows."""
    mask = np.ones(len(idx), dtype=bool)
    for name in names:
        mask &= extractors[name].covered(pairs, idx)
    return mask


def run_ensemble(pairs: pd.DataFrame, extractors: dict[str, object], combo_spec: str,
                  split_kind: str = "stratified", seed: int = 42,
                  test_size: float = 0.2, val_size: float = 0.2,
                  weight_method: str = "both",
                  train_idx: np.ndarray | None = None, val_idx: np.ndarray | None = None,
                  test_idx: np.ndarray | None = None,
                  on_missing: str = "raise",
                  checkpoint_dir: "pathlib.Path | str | None" = None,
                  tune_boost_hp: bool = False, n_trials: int = 30,
                  optuna_storage: str | None = None, run_id: str | None = None) -> dict:
    """Fit one boosting head per combo (concatenating its extractors'
    features), fit ensemble weights on validation, evaluate on test.

    `tune_boost_hp`: if True, each combo's boosting head is tuned independently
    via `orbind.baselines.tune_boost` (optuna, `n_trials` per combo) instead of
    the fixed-hyperparameter `fit_boost` -- mirrors ProSmith/LORAX's own
    per-head hyperopt search (see `orbind/baselines.py::tune_boost` for the
    exact space), just with optuna's TPE sampler instead of their random
    search. Off by default -- fixed hyperparameters are far cheaper to
    iterate with, and tuning multiplies runtime by roughly `n_trials` per
    combo per repeat.

    `optuna_storage`/`run_id`: only used when `tune_boost_hp` is set. Pass a
    SQLAlchemy storage URL (e.g. "sqlite:///run_dir/optuna_studies.db") to
    persist every combo's study there under a
    `repeat{run_id}_combo{combo}` name (falls back to `seed` if `run_id` is
    omitted) instead of keeping it in-process only -- lets `optuna-dashboard`
    show live progress across every repeat/combo, even with several repeats
    running concurrently in separate processes and writing to the same file.

    `checkpoint_dir`, if given, gets one XGBoost booster per combo
    (`boost_{combo}.json`, via the sklearn wrapper's own `save_model`) plus
    whatever each extractor chooses to persist (e.g. a pair-level source's
    whole-train-fit torch model) -- see `Extractor.fit_transform`.

    Pass `train_idx`/`val_idx`/`test_idx` directly (e.g. from
    `orbind.regimes.load_split` for full_full) to bypass `split_kind`/
    `three_way_split` entirely -- whichever split "tradition" produced them,
    the rest of this function doesn't care.

    `on_missing`: "raise" (default) lets an extractor's own KeyError surface
    the first time a row it can't cover is requested -- a loud, precise
    signal that the embedding base is short of what this regime/combo needs.
    "drop" instead computes, once per run, the intersection of coverage
    across every extractor referenced by ANY combo in `combo_spec`, logs what
    fraction of train/val/test rows survive, and trains/evaluates every combo
    (and the final ensemble) on that same reduced set -- so combos stay
    directly comparable and the ensemble's per-combo val predictions stay
    row-aligned.

    Returns
    -------
    {
      "combos":   {combo_tuple: metrics_dict},   # solo performance per combo
      "ensemble": {weight_method: metrics_dict},
      "weights":  {weight_method: {combo_tuple: weight}},
      "n": {"train": int, "val": int, "test": int, "test_pos": int},
    }
    """
    if on_missing not in ("raise", "drop"):
        raise ValueError(f"on_missing must be 'raise' or 'drop', got {on_missing!r}")
    names = list(extractors.keys())
    combos = parse_combo_spec(combo_spec, names)

    y = pairs["label"].to_numpy(dtype=np.float32)
    if train_idx is None:
        assert val_idx is None and test_idx is None, "supply all three of train/val/test_idx or none"
        train_idx, val_idx, test_idx = three_way_split(pairs, y, split_kind, seed, test_size, val_size)
        print(f"  split={split_kind} seed={seed}: ", end="", flush=True)
    else:
        assert val_idx is not None and test_idx is not None, "supply all three of train/val/test_idx or none"
        print(f"  split=<externally supplied>: ", end="", flush=True)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    if on_missing == "drop":
        names_used = sorted({n for combo in combos for n in combo})
        for split_name, idx_name in (("train", "train_idx"), ("val", "val_idx"), ("test", "test_idx")):
            idx = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}[idx_name]
            mask = _coverage_intersection(pairs, extractors, names_used, idx)
            n_drop = int((~mask).sum())
            if n_drop:
                print(f"  WARNING coverage[{split_name}]: {int(mask.sum())}/{len(idx)} "
                      f"({100 * mask.mean():.1f}%) rows covered by all of {names_used} -- "
                      f"dropping {n_drop} row(s) lacking an embedding", flush=True)
            if idx_name == "train_idx":
                train_idx = idx[mask]
            elif idx_name == "val_idx":
                val_idx = idx[mask]
            else:
                test_idx = idx[mask]

    y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
    print(f"  after coverage: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} test_pos={int(y_te.sum())}", flush=True)

    if checkpoint_dir is not None:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get(name):
        if name not in cache:
            cache[name] = extractors[name].fit_transform(pairs, train_idx, val_idx, test_idx, seed,
                                                           checkpoint_dir=checkpoint_dir)
        return cache[name]

    repeat_tag = run_id if run_id is not None else seed
    combo_metrics, val_preds, test_preds = {}, {}, {}
    for combo in combos:
        Xtr = np.concatenate([get(n)[0] for n in combo], axis=1)
        Xva = np.concatenate([get(n)[1] for n in combo], axis=1)
        Xte = np.concatenate([get(n)[2] for n in combo], axis=1)
        if tune_boost_hp:
            study_name = f"repeat{repeat_tag}_combo{'+'.join(combo)}"
            clf, study = tune_boost(Xtr, y_tr, Xva, y_va, seed=seed, n_trials=n_trials,
                                     storage=optuna_storage, study_name=study_name)
            print(f"  [repeat {repeat_tag}] tune[{'+'.join(combo):>20}]: {n_trials} trials, "
                  f"best val AUPRC={study.best_value:.3f}, params={study.best_params}", flush=True)
            if checkpoint_dir is not None:
                study.trials_dataframe().to_csv(
                    checkpoint_dir / f"optuna_{'+'.join(combo)}.csv", index=False)
        else:
            clf = fit_boost(Xtr, y_tr, seed=seed)
        if checkpoint_dir is not None:
            # save the underlying Booster directly -- this xgboost version's
            # sklearn-wrapper .save_model() needs `_estimator_type`, which
            # isn't set on a bare XGBClassifier built the way fit_boost does.
            clf.get_booster().save_model(str(checkpoint_dir / f"boost_{'+'.join(combo)}.json"))
        p_va = clf.predict_proba(Xva)[:, 1]
        p_te = clf.predict_proba(Xte)[:, 1]
        val_preds[combo], test_preds[combo] = p_va, p_te
        combo_metrics[combo] = D.metrics(y_te, p_te)
        print(f"  [repeat {repeat_tag}] combo {'+'.join(combo):>20s}: dim={Xtr.shape[1]:4d} "
              + " ".join(f"{k}={v:.3f}" for k, v in combo_metrics[combo].items()), flush=True)

    combiners = fit_ensemble_weights(val_preds, y_va, method=weight_method)
    ensemble_metrics, weights = {}, {}
    for m, combiner in combiners.items():
        p_final = combiner.predict(test_preds)
        ensemble_metrics[m] = D.metrics(y_te, p_final)
        weights[m] = combiner.weights
        print(f"  [repeat {repeat_tag}] ensemble[{m}]: "
              + " ".join(f"{k}={v:.3f}" for k, v in ensemble_metrics[m].items()), flush=True)

    return {
        "combos": combo_metrics,
        "ensemble": ensemble_metrics,
        "weights": weights,
        "n": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
              "test_pos": int(y_te.sum())},
    }
