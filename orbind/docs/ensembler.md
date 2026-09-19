# The ensembler

`orbind/ensemble.py` plus the `--source` / `--combos` front end in
`scripts/modeling/train/train_ensemble_boost.py`. This is the largest single
mechanism in the repository, and the only entry point that produces a paper
number — so it is worth knowing which half of it we actually use.

---

## The convention, first

The machine was built as a generalization of what ProSmith and LORAX do in their
second stage: build several feature combinations, fit a boosting head on each,
then **combine the heads** by weights fit on validation, optionally **tuning**
each head's hyperparameters first. All of that is implemented here and works.

**We deliberately do not report any of it.**

Every number in the paper is:

* **one combo, one head** — no weighted combination of combos;
* **fixed hyperparameters** — `fit_boost`'s 400 trees / depth 6 / lr 0.1 /
  subsample 0.8, identical for every method, every dataset, every fold;
* the same for the baselines as for us.

The reason is that the upper two layers are unfair in a specific way. Both fit on
the validation split, so both convert *more search* into *a better test number*
without changing the representation being tested — and the claim under test is
about what a receptor embedding carries, not about how hard we tuned a
gradient-boosted head on top of it. A stacker over `{cls, prot+mol, cls+prot+mol}`
will beat each of its inputs; that tells you nothing about whether `cls` is a good
receptor representation. Tuning per combo does the same thing more quietly: the
combo with the most searched head wins, and the search budget is not part of the
method being compared.

So the layers are kept, documented, and left switched off:

| layer | flag | in the paper |
|---|---|---|
| extractors → features | `--source` | **yes** |
| combos = feature concatenations | `--combos` | **yes**, one per reported row |
| combo weighting / stacking | `--weight-method` | no |
| per-head hyperparameter search | `--tune-boost` | no |

Runs are named `*-fixed` for exactly this reason. `--weight-method` cannot be
turned off from the CLI (it defaults to `both`), so ensemble rows *do* appear in
`metrics.csv` next to the combo rows — `summarize_runs.py` prints them, and they
are simply not what we read. With a single combo the simplex is degenerate and its
row equals that combo; the `logreg`/`linreg` row is a refit on top of it and can
drift, which is one more reason to read the combo row by name.

If you ever want the tuned/stacked variant for a rebuttal, run it as its own run
and label it; do not mix it into a `*-fixed` pool.

---

## Anatomy

### 1. Extractors

Everything is a source, and every source satisfies one protocol:

```python
class Extractor(Protocol):
    name: str
    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed,
                      checkpoint_dir=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def covered(self, pairs, idx) -> np.ndarray: ...   # optional
```

Given row-positions into a shared `pairs` frame (`receptor`, `inchikey`, `label`),
it returns row-aligned feature matrices. What happens inside is its own business.
The engine never sees an embedding file.

Two families:

* **entity-level** (`EntityExtractor` → `esm`, `gin`): a static npz lookup keyed by
  receptor sequence or InChIKey. Label-independent, nothing to train, ignores
  `checkpoint_dir`. Its `covered()` is what makes `--on-missing drop` possible.
* **pair-level** (`gnn_signed`, `lorax`, `prosmith`, `molor`, `hladis`, `attn_*`):
  trains its own model on the train rows of this fold and emits a per-pair vector,
  conventionally named `cls`. These persist to `checkpoint_dir`.

Pair-level sources embed train, val and test with the same fitted model. That
accepts mild train-row leakage; it is the same tradeoff ProSmith's own cls stage
makes, and keeping it is what makes the comparison a comparison. `n_models > 1`
bags that many independently-seeded models and concatenates their outputs;
`n_models=1` is ProSmith's exact scheme and is what our graph uses.

### 2. The combo mini-language

`--combos "1 23 123"` — each token is a string of **1-based positions into the
`--source` flags of that command line**, in the order they were given.

```
--source cls=... --source prot=... --source mol=...
   1 = cls        23 = prot+mol        123 = cls+prot+mol
```

One combo = one boosting head over the concatenation of its sources' features.
Extractor outputs are cached by name, so a source appearing in several combos is
computed once per fold.

> **The trap.** Adding or reordering a `--source` flag silently changes what every
> digit means. `13` meant `cls+mol` under one source order and something else under
> another — this has already produced a wrong comparison once. `metrics.csv` records
> combos by *name* (`cls+mol`), so compare names, never digits.

`parse_combo_spec` rejects non-digit tokens, out-of-range indices, a repeated index
inside one token, duplicate combos, and an empty spec.

### 3. Coverage

With `--on-missing raise` (default) an extractor raises the first time a row it
cannot cover is requested — loud and precise.

With `--on-missing drop` the engine computes, **once per run**, the intersection of
coverage over every extractor named by *any* combo, and trains and scores every
combo on that same reduced row set. Doing it per combo would leave the rows
unaligned and the combos incomparable.

### 4. The head

`orbind/baselines.py`. `fit_boost` is the fixed-hyperparameter XGBoost head;
`predict_scores` returns a positive-class probability (classification) or the
predicted value (regression). CUDA if available, with an automatic CPU retry on
`XGBoostError`.

`tune_boost` is the optuna/TPE search over the same space ProSmith and LORAX search
randomly — resumable, persistable to sqlite for `optuna-dashboard`. Off by default,
and per the convention above, off in everything reported.

### 5. Weighting (present, unused)

`fit_ensemble_weights` fits combiners on the validation predictions:

| task | methods |
|---|---|
| classification | `simplex` (non-negative, sums to 1, minimises val log-loss — the ProSmith/LORAX scheme) and `logreg` (unconstrained stacker) |
| regression | `simplex` (same, minimising val MSE) and `linreg` (OLS stacker) |

The method names differ per task on purpose, so a metrics file never leaves you
guessing whether `logreg` meant a logistic stacker or a least-squares one.

### 6. The naive row

Every run also scores the constant **train-mean** predictor on the same test rows
with the same metrics. Under regression that is upstream's own naive baseline;
under classification the same constant is the class prevalence, i.e. the AUPRC
floor with AUROC 0.5 by construction.

It is not decoration: R² is measured against the **test** mean while the naive
predictor uses the **train** mean, so a model can beat naive and still score below
zero. On a cold-molecule split that gap is the whole story.

### 7. What a run writes

```
results/ensemble_logs/<run_id>/
  config.json          every CLI arg (written at start-up — see the gotcha below)
  log.txt              this process's stdout
  logs/repeat_{R}.log  one per repeat, even when --max-parallel forks it
  metrics.csv          one row per (repeat, combo) and per (repeat, ensemble method)
  checkpoints/repeat_{R}/
     boost_{combo}.json          one booster per combo
     {type}_{source}_model{k}.pt  one state_dict per pair-level model
```

---

## Fine print

**Source dispatch is lazy.** `_FACTORY_SPEC` maps a type name to a module path, and
the module is imported only when a source of that type is requested. That is what
lets ProSmith/LORAX runs live in a PyG-free environment and MolOR in its own
dgl environment, without the graph pipeline's torch↔PyG pin.

**Field order in a `--source` string is positional and long.** `gnn_signed` takes
`name=gnn_signed[:protein_path:molecule_path[:n_models[:emit[:q[:criterion[:edge_threshold[:k_mode[:edge_center[:edge_weight_mode[:dummy_compression]]]]]]]]]]`.
A blank field keeps the default, so `cls=gnn_signed:::1` sets only `n_models`.
Defaults are the headline configuration: `q=0.99`, `criterion=greedy_pair_cover`,
`emit=prot`, `n_models=1`, `k_mode=coverage_quantile`.

**Paths inside a `--source` string must be relative.** The field separator is
`:`, so a Windows absolute path (`C:/...`) splits into two fields and the extractor
is handed `C` as its npz. Everything here uses repo-relative paths, which is why
this has never bitten a run -- but it is a real constraint of the syntax.

**`emit` decides what the graph hands back.** `prot` (default) emits the refined
receptor vector — the v5 probe shape, and what `cls` means in every reported row.
`both` also concatenates the molecule side. A `cls` from `emit=prot` is a receptor
representation, not a binding model; that distinction is the point of the whole
comparison.

**`task` propagates.** `--task` switches the head, the metric family and the weight
fitters, and is passed to every extractor that declares a `task` attribute so its
internal training criterion matches. `--regime ofm` defaults it to regression.

**Splits can come from anywhere.** Passing `train_idx`/`val_idx`/`test_idx`
directly bypasses `split_kind`/`three_way_split` entirely, which is how the
`full_full` and `ofm` regimes hand in their upstream folds. When the engine does
split, `three_way_split` calls `dataset.split` twice so group-based splits stay
leak-free at both cuts.

**Checkpoint reuse is silent.** An extractor that finds its `.pt` under
`checkpoints/repeat_{R}/` loads it and skips training. That is the point for
incremental runs — adding a `cls+prot+mol` row to a finished cls-only run costs
one boosting fit. It is also a trap for any timing or from-scratch measurement:
pass `--skip-checkpoints`, which sets `checkpoint_dir = None` and so disables
loading as well as saving.

**`config.json` is written at start-up.** It records what was *asked for*, not what
completed. To find out which sources a run actually produced, read `metrics.csv`.
`extend_runs_with_combo.py` was wrong until it was changed to do exactly that.

**Parallelism.** `--max-parallel N` runs repeats as separate OS processes, each
with its own CUDA context; with more than one GPU visible they are pinned
round-robin via `CUDA_VISIBLE_DEVICES`. A pair-level source's `n_models` are always
trained concurrently regardless. Two XGBoost processes pinned to the *same* GPU
will abort in `cuMemUnmap` — give each worker its own device, or use
`--max-parallel 1`.

The same abort hit torch and XGBoost inside ONE process: torch's caching allocator
keeps freed blocks reserved, XGBoost's `device="cuda"` then fails, and a CUDA error
thrown from an XGBoost destructor calls `std::terminate` (SIGABRT — no Python
`except` sees it). `run_ensemble` therefore runs every extractor first, calls
`_release_gpu()` (gc + `torch.cuda.empty_cache()`), and only then fits the boosting
heads. A repeat that dies that way used to leave the parent blocked forever on
`q.get()`; the parent now polls every child, reports `repeat N died (exit code -6)`
naming its log, lets the other repeats finish and write their rows, and exits
non-zero at the end. Rerunning the same command redoes only what is missing in
substance: trained models reload from `checkpoints/`.

**Repeats run as a queue, not in chunks.** `--max-parallel N` keeps N worker slots,
each bound to one GPU, and refills a slot the moment its repeat finishes
(`_run_repeat_pool`). The earlier scheme started N repeats and waited for all of
them before starting the next N, so a checkpoint-loaded repeat (minutes) paired with
one training from scratch (hours) left a card idle for the difference.
