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
from .baselines import fit_boost, predict_scores, tune_boost
from .tasks import check_task


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


def _simplex(objective, K: int) -> np.ndarray:
    """Non-negative weights summing to 1, minimizing `objective(w)`."""
    from scipy.optimize import minimize
    res = minimize(objective, np.full(K, 1.0 / K), method="SLSQP",
                    bounds=[(0.0, 1.0)] * K,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    return res.x


def _fit_simplex_weights(P: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]]) -> EnsembleCombiner:
    """Non-negative weights summing to 1, minimizing validation log-loss.
    Directly comparable to the ProSmith/LORAX weighting scheme."""
    eps = 1e-6

    def neg_log_loss(w):
        p = np.clip(P @ w, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    w = _simplex(neg_log_loss, P.shape[1])
    return EnsembleCombiner(combos=combos, method="simplex",
                             weights=dict(zip(combos, w.tolist())),
                             predict_fn=lambda P_, w=w: P_ @ w)


def _fit_simplex_weights_mse(P: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]]) -> EnsembleCombiner:
    """The regression counterpart: the same convex combination of the combo
    predictions, chosen to minimize validation MSE instead of log-loss. Still
    a simplex, so the weights stay readable as mixing proportions."""
    w = _simplex(lambda w: float(((P @ w - y) ** 2).mean()), P.shape[1])
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


def _fit_linreg_weights(P: np.ndarray, y: np.ndarray, combos: list[tuple[str, ...]]) -> EnsembleCombiner:
    """The regression counterpart of the logreg stacker: ordinary least
    squares over the K combo predictions, coefficients unconstrained."""
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(P, y)
    return EnsembleCombiner(combos=combos, method="linreg",
                             weights=dict(zip(combos, model.coef_.tolist())),
                             intercept=float(model.intercept_),
                             predict_fn=lambda P_: model.predict(P_))


# One fitter table per task: the method NAMES differ on purpose, so a metrics
# file never leaves you guessing whether "logreg" meant a logistic stacker or
# a least-squares one.
_WEIGHT_FITTERS = {
    "classification": {"simplex": _fit_simplex_weights, "logreg": _fit_logreg_weights},
    "regression": {"simplex": _fit_simplex_weights_mse, "linreg": _fit_linreg_weights},
}


def fit_ensemble_weights(val_preds: dict[tuple[str, ...], np.ndarray], y_val: np.ndarray,
                          method: str = "both",
                          task: str = "classification") -> dict[str, EnsembleCombiner]:
    fitters = _WEIGHT_FITTERS[check_task(task)]
    combos = list(val_preds.keys())
    P = np.column_stack([val_preds[c] for c in combos])
    if method == "both":
        methods = list(fitters)
    elif method in fitters:
        methods = [method]
    else:
        raise ValueError(f"weight method {method!r} is not defined for task {task!r} "
                         f"(have {sorted(fitters)})")
    return {m: fitters[m](P, y_val, combos) for m in methods}


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
                  optuna_storage: str | None = None, run_id: str | None = None,
                  task: str = "classification") -> dict:
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

    `task`: "classification" (default, M2OR's 0/1 Responsive) or "regression"
    (the continuous z-scored response of the Carey/Hallem datasets -- see
    orbind/tasks.py and orbind/regimes_ofm.py). It switches the boosting head
    (XGBRegressor), the metric family (`orbind.dataset.METRICS`), and the
    ensemble weight fitters, and it is passed on to every extractor that
    declares a `task` attribute so their own criterion matches.

    Returns
    -------
    {
      "combos":   {combo_tuple: metrics_dict},   # solo performance per combo
      "ensemble": {weight_method: metrics_dict},
      "weights":  {weight_method: {combo_tuple: weight}},
      "naive":    metrics_dict,                  # constant train-mean predictor
      "n": {"train": int, "val": int, "test": int, ...},
    }

    The "naive" entry is the no-information floor: predict the TRAIN mean for
    every test row. Under regression that is upstream's own naive row, and it
    is the only thing that makes an R^2 near zero interpretable. Under
    classification the same constant is the class prevalence, which is exactly
    the AUPRC baseline (and gives AUROC 0.5 by construction).
    """
    if on_missing not in ("raise", "drop"):
        raise ValueError(f"on_missing must be 'raise' or 'drop', got {on_missing!r}")
    check_task(task)
    metric_fn = D.METRICS[task]
    names = list(extractors.keys())
    combos = parse_combo_spec(combo_spec, names)

    # Pair-level extractors train their own head and need the same criterion
    # as the boosting head downstream; entity-level lookups have no task.
    #
    # A pair-level source that does NOT declare `task` has not been ported to
    # the second task, and would quietly keep minimising BCE against a
    # continuous target -- a wrong answer rather than an error. Refuse instead.
    # (As of writing that is the attention-MIL and GNN sources: noisy-OR/LSE
    # pooling and the signed bipartite graph are both built out of binary
    # edges, so porting them is a design question, not a criterion swap.)
    for name, ex in extractors.items():
        if hasattr(ex, "task"):
            ex.task = task
        elif not isinstance(ex, EntityExtractor) and task != "classification":
            raise NotImplementedError(
                f"source {name!r} ({type(ex).__name__}) trains its own head but declares no "
                f"`task`, so it only supports classification; asked for {task!r}")

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
    # test_pos is kept on both tasks so the log line stays greppable; under
    # regression it counts test rows above the pool's zero (the response is
    # already z-scored), which is descriptive only, never a metric.
    n_info = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
              "test_pos": int((y_te > 0).sum())}
    if task == "regression":
        n_info |= {"test_mean": float(y_te.mean()), "test_std": float(y_te.std()),
                   "train_mean": float(y_tr.mean())}
    print(f"  after coverage: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)} test_pos={n_info['test_pos']}"
          + (f" test_mean={n_info['test_mean']:.3f} train_mean={n_info['train_mean']:.3f}"
             if task == "regression" else ""), flush=True)

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
                                     storage=optuna_storage, study_name=study_name, task=task)
            # Report the trials actually behind best_value, not the requested
            # budget: on a resumed (or study-imported) run the search is
            # already satisfied and this call runs none of its own, so
            # printing n_trials would claim work that never happened.
            import optuna as _optuna
            done = sum(1 for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE)
            print(f"  [repeat {repeat_tag}] tune[{'+'.join(combo):>20}]: "
                  f"{done} trials done (target {n_trials}), "
                  f"best val {'R2' if task == 'regression' else 'AUPRC'}={study.best_value:.3f}, "
                  f"params={study.best_params}", flush=True)
            if checkpoint_dir is not None:
                study.trials_dataframe().to_csv(
                    checkpoint_dir / f"optuna_{'+'.join(combo)}.csv", index=False)
        else:
            clf = fit_boost(Xtr, y_tr, seed=seed, task=task)
        if checkpoint_dir is not None:
            # save the underlying Booster directly -- this xgboost version's
            # sklearn-wrapper .save_model() needs `_estimator_type`, which
            # isn't set on a bare XGBClassifier built the way fit_boost does.
            clf.get_booster().save_model(str(checkpoint_dir / f"boost_{'+'.join(combo)}.json"))
        p_va = predict_scores(clf, Xva, task)
        p_te = predict_scores(clf, Xte, task)
        val_preds[combo], test_preds[combo] = p_va, p_te
        combo_metrics[combo] = metric_fn(y_te, p_te)
        print(f"  [repeat {repeat_tag}] combo {'+'.join(combo):>20s}: dim={Xtr.shape[1]:4d} "
              + " ".join(f"{k}={v:.3f}" for k, v in combo_metrics[combo].items()), flush=True)

    combiners = fit_ensemble_weights(val_preds, y_va, method=weight_method, task=task)
    ensemble_metrics, weights = {}, {}
    for m, combiner in combiners.items():
        p_final = combiner.predict(test_preds)
        ensemble_metrics[m] = metric_fn(y_te, p_final)
        weights[m] = combiner.weights
        print(f"  [repeat {repeat_tag}] ensemble[{m}]: "
              + " ".join(f"{k}={v:.3f}" for k, v in ensemble_metrics[m].items()), flush=True)

    # The no-information floor, on the same rows and the same metrics.
    naive_metrics = metric_fn(y_te, np.full(len(y_te), float(y_tr.mean())))
    print(f"  [repeat {repeat_tag}] naive[train-mean]: "
          + " ".join(f"{k}={v:.3f}" for k, v in naive_metrics.items()), flush=True)

    return {
        "combos": combo_metrics,
        "ensemble": ensemble_metrics,
        "weights": weights,
        "naive": naive_metrics,
        "n": n_info,
    }
