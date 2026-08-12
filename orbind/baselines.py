"""Reusable MP baselines (MLP + XGBoost) over in-memory embedding dicts.

Shared by scripts/modeling/eval/eval_mp_table.py and notebooks/. Works on dicts so a
notebook can pass *modified* protein embeddings (e.g. transformed ESM-2) without
touching files.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from . import dataset as D


def make_xy(pairs, prot, mol, random_prot=False, seed=0):
    """X = [molecule || protein] for the pairs that have both embeddings."""
    mask = pairs["receptor"].isin(prot) & pairs["inchikey"].isin(mol)
    p = pairs[mask].reset_index(drop=True)
    Xm = np.stack([mol[i] for i in p["inchikey"]]).astype(np.float32)
    if random_prot:
        dim = next(iter(prot.values())).shape[0]
        rng = np.random.default_rng(seed)
        Xp = rng.standard_normal((len(p), dim)).astype(np.float32)  # per-row noise floor
    else:
        Xp = np.stack([prot[r] for r in p["receptor"]]).astype(np.float32)
    X = np.concatenate([Xm, Xp], axis=1)
    y = p["label"].to_numpy().astype(np.float32)
    return X, y, p


def train_mlp(Xtr, ytr, Xte, seed=42, hidden=(512, 128), dropout=0.3, lr=1e-3, epochs=100, batch=256):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr, Xte = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]; d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]; opt.zero_grad()
            loss_fn(model(Xtr[b]).squeeze(-1), ytr[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte).squeeze(-1)).numpy()


def fit_boost(Xtr, ytr, seed=42, task="classification"):
    """Fit and return the estimator (not just its predictions), so callers
    that need scores on more than one held-out set (e.g. val AND test) don't
    have to refit.

    `task="regression"` swaps XGBClassifier for XGBRegressor on the same
    hyperparameters (400 trees, depth 6, lr 0.1, subsample/colsample 0.8) so
    the two tasks stay comparable head-to-head. `scale_pos_weight` has no
    meaning without classes and is dropped rather than neutralised.
    """
    import xgboost as xgb
    import torch
    from .tasks import check_task
    check_task(task)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    common = dict(n_estimators=400, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8,
                  tree_method="hist", n_jobs=-1, random_state=seed)

    def make_estimator(target_device):
        if task == "regression":
            return xgb.XGBRegressor(**common, objective="reg:squarederror",
                                     eval_metric="rmse", device=target_device)
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        return xgb.XGBClassifier(**common, scale_pos_weight=spw,
                                  eval_metric="aucpr", device=target_device)

    clf = make_estimator(device)
    try:
        clf.fit(Xtr, ytr)
    except xgb.core.XGBoostError:
        if device != "cuda":
            raise
        print("  XGBoost CUDA unavailable; retrying boost head on CPU", flush=True)
        clf = make_estimator("cpu")
        clf.fit(Xtr, ytr)
    return clf


def predict_scores(model, X, task="classification"):
    """The one number per row a downstream ensemble/metric wants: a positive-
    class probability for classification, the predicted value for regression."""
    return model.predict(X) if task == "regression" else model.predict_proba(X)[:, 1]


def train_boost(Xtr, ytr, Xte, seed=42, task="classification"):
    return predict_scores(fit_boost(Xtr, ytr, seed=seed, task=task), Xte, task)


def tune_boost(Xtr, ytr, Xva, yva, seed=42, n_trials=30, storage=None, study_name=None,
                task="classification"):
    """Per-combo XGBoost hyperparameter search (optuna, TPE sampler -- same
    idea as ProSmith/LORAX's own hyperopt random search over a near-identical
    space, just with a smarter sampler): each trial fits on train, scores
    val AUPRC, the best trial's params get one final train-only refit (so
    the returned classifier's val predictions stay honest for downstream
    ensemble-weight fitting, same contract as the fixed-hyperparameter
    `fit_boost`).

    `storage`/`study_name`: if given (a SQLAlchemy storage URL, e.g.
    "sqlite:///run_dir/optuna_studies.db"), the study is persisted there
    instead of living only in-process -- lets `optuna-dashboard` show every
    combo's/repeat's progress live while several repeats run concurrently in
    separate processes, all writing to the same file.

    Resumable: if `storage` already holds a study named `study_name` with
    `n_done` completed trials, only `max(0, n_trials - n_done)` more trials
    run -- so re-running with the same run folder (same sqlite file) and
    the same or a higher `n_trials` picks up mid-tuning instead of starting
    over or padding trials on top of an already-finished budget.

    `task="regression"` searches the same space minus `scale_pos_weight_mult`
    (meaningless without classes) and maximises val R^2 instead of val AUPRC.

    Returns `(estimator, study)` -- the caller decides what to do with the
    optuna `study` (e.g. persist `study.trials_dataframe()`, the full
    per-trial hyperparameters + val-score history, or just read
    `study.best_value`/`study.best_params`)."""
    import optuna
    import torch
    import xgboost as xgb
    from .tasks import check_task
    check_task(task)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    objective_metric = "R2" if task == "regression" else "AUPRC"

    def make_classifier(params, device):
        if task == "regression":
            return xgb.XGBRegressor(
                **params, objective="reg:squarederror",
                eval_metric="rmse", tree_method="hist", device=device,
                n_jobs=-1, random_state=seed)
        spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        weight_mult = params.pop("scale_pos_weight_mult")
        return xgb.XGBClassifier(
            **params, scale_pos_weight=spw * weight_mult,
            eval_metric="aucpr", tree_method="hist", device=device,
            n_jobs=-1, random_state=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 30, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 14),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.1, 15.0),
            max_delta_step=trial.suggest_float("max_delta_step", 0.0, 5.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        if task == "classification":
            params["scale_pos_weight_mult"] = trial.suggest_float("scale_pos_weight_mult", 0.5, 1.5)
        try:
            clf = make_classifier(dict(params), device)
            clf.fit(Xtr, ytr)
        except xgb.core.XGBoostError:
            clf = make_classifier(dict(params), "cpu")
            clf.fit(Xtr, ytr)
        p_va = predict_scores(clf, Xva, task)
        return D.METRICS[task](yva, p_va)[objective_metric]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                 storage=storage, study_name=study_name, load_if_exists=True)
    n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, n_trials - n_done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    clf = make_classifier(dict(study.best_params), device)
    try:
        clf.fit(Xtr, ytr)
    except xgb.core.XGBoostError:
        clf = make_classifier(dict(study.best_params), "cpu")
        clf.fit(Xtr, ytr)
    return clf, study


HEADS = {"mlp": train_mlp, "boost": train_boost}


def run_one(pairs, prot, mol, head, split, random_prot=False, seed=42, test_size=0.2):
    """assemble -> split -> train head -> metrics. Returns (metrics, info)."""
    X, y, p = make_xy(pairs, prot, mol, random_prot=random_prot, seed=seed)
    tr, te = D.split(p, y, kind=split, test_size=test_size, seed=seed)
    pred = HEADS[head](X[tr], y[tr], X[te], seed=seed)
    info = {"train": int(tr.sum()), "test": int(te.sum()), "test_pos": int(y[te].sum())}
    return D.metrics(y[te], pred), info
