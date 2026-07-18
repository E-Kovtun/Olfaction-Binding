# Multi-source boosting ensemble

Generalized, symmetric replacement for the ProSmith/LORAX-style boosting
ensemble: instead of a fixed solo/pair/triplet scheme over exactly three
models, any number of feature sources can be combined via a small
combo mini-language, each combo gets its own XGBoost head, and the heads
are combined via a fitted ensemble weighting.

Code: `orbind/ensemble.py` (engine), `orbind/attention_extractor.py` (cls
sources), `orbind/regimes.py` (full_full split bookkeeping),
`scripts/modeling/train/train_ensemble_boost.py` (CLI).

## Combo mini-language

Sources are registered in a fixed order (1-based). `--combos "1 2 12"` means:
solo source 1, solo source 2, and the concatenation of both. Any subset in
any digit string works (`"1 23 123"`, `"14"`, ...) — this is the symmetric
generalization of ProSmith/LORAX's fixed 3-source solo/pair/triplet scheme.

For each combo, `run_ensemble` concatenates that combo's sources' features
and fits one `XGBClassifier` head on train, producing val/test probabilities.

## Two source families (`Extractor` protocol)

Both share the contract `fit_transform(pairs, train_idx, val_idx, test_idx,
seed, checkpoint_dir=None) -> (Xtr, Xval, Xte)`.

- **Entity-level** (`EsmExtractor`, `GinExtractor`, `orbind/ensemble.py`):
  label-independent static `.npz` lookups (protein or molecule embeddings).
  No training, nothing to leak, `checkpoint_dir` is a no-op.
- **Pair-level / "cls"** (`MilNoisyOrExtractor`, `MilLseExtractor`,
  `orbind/attention_extractor.py`): supervised, trained fresh every run on
  `train_idx`'s own labels. These are the reimplementations of the two
  best-performing site-MIL pooling rules from the project's own screen
  (noisy-OR and log-sum-exp over per-site logits), rewritten as two fully
  independent classes with no import from the older
  `train_full_full_site_mil_attention.py` script.

## Why cls sources need out-of-fold (OOF) train features

Every combo — including a solo cls combo — gets re-boosted by its own
XGBoost head. If a cls source's train-row "feature" came from a model that
was fit on those same train rows' labels, the second-level booster would
be training on leaked information. So:

- `val`/`test` features come from **one** whole-train fit (never touches
  val/test, no leakage there regardless).
- `train` features come from a **stratified 5-fold OOF refit**: each fold
  is held out, a fresh model is trained on the other 4 folds (with its own
  inner train/val split for early stopping), and predicts only on the held
  out fold. After 5 folds every train row has an honest, leak-free feature.

The whole-train fit and all 5 OOF folds are independent trainings, always
run concurrently via a thread pool (`ThreadPoolExecutor(max_workers=n_folds+1)`)
— not a configurable option, since any future cls-style source will need
the same shape.

## Recent change: cls sources now emit an embedding, not a scalar

Originally each cls source's `forward()` collapsed all the way to one
pair-level probability (dim_out=1) — literally the model's own trained
prediction, handed to the booster as a single number. Per plan, this was
changed to strip the final `Linear(2*dim, 1)` layer and instead pool the
**pre-head hidden representation** (`model.embed()`, 2*dim = 128-dim by
default), using the same pooling weights the original scalar rules imply:

- noisy-OR: per-site sigmoid probabilities, normalized to sum to 1, used to
  weight-average the per-site hidden vectors.
- LSE: per-site `softmax(logit / temperature)`, same weighted average.

Training itself (BCE loss on the scalar `forward()`, early stopping on
validation AUPRC) is unchanged — only what gets handed to the boosting
stage changed, from a 1-d prediction to a 128-d "cls token" embedding, so
the second-level booster can pick up more than the model's own final
verdict (and, combined with prot/mol, more than just a restatement of
`prot+mol`'s own signal).

## Ensemble weighting (two methods, `--weight-method both|simplex|logreg`)

Both fit on the combos' validation-set predictions, evaluate once on test:

- **`simplex`**: non-negative weights summing to 1, minimizing validation
  log-loss (`scipy.optimize.minimize`, SLSQP). Directly comparable to the
  ProSmith/LORAX weighting scheme, just over an arbitrary combo set.
- **`logreg`**: unconstrained logistic-regression stacker over the combo
  predictions (coefficients can be negative, don't sum to 1 — read as
  relative importance, not literal mixing weights).

## Two split regimes (kept deliberately separate, not backported into each other)

- **curated_full** (`orbind/dataset.py::split`): pairs from a csv, split
  `stratified` / `group_molecule` / `group_receptor`, fresh random split
  per `--seeds`.
- **full_full** (`orbind/regimes.py`): pairs reconstructed from LoRaX's own
  data. `transductive` = LoRaX's own 5 `rand_split_{1..5}` folds (we only
  persist the *index partition* into a shared pool, not LoRaX's tables —
  `data/splits_indexes/lorax_m2or/`). `inductive_molecule` = our own
  cold-molecule split (30% of EC50-quality molecules held out, 2/3 test /
  1/3 val within the holdout), seeds 42-46 by default.

Transductive's train/val/test aren't just different sizes — they're
different *data*: train/val are the full noisy M2OR (~5-6% positive), test
is LoRaX's own EC50-only curated subset (~22% positive). This asymmetry
(also the basis of the LORAX-paper critique discussed earlier) turns out to
matter a lot for the bug below.

## Storage / logging (`results/ensemble_logs/<run_id>/`)

Every CLI invocation creates one timestamped run folder:

- `config.json` — every CLI arg, for reproducibility.
- `log.txt` — the process's own stdout (setup + final summary).
- `logs/repeat_{R}.log` — one per repeat, captures its combo/ensemble lines
  even when `--max-parallel` runs it in another process.
- `metrics.csv` — one row per (repeat, combo) and per (repeat, ensemble
  method) — not just the final ensemble score. Ensemble rows carry a
  `weights` column (JSON dict of combo -> weight) for `simplex`/`logreg`.
- `checkpoints/repeat_{R}/`: `boost_{combo}.json` (one XGBoost booster per
  combo, via `clf.get_booster().save_model()` — the sklearn wrapper's own
  `.save_model()` needs `_estimator_type`, which a bare `fit_boost`-built
  classifier doesn't set) and `attn_{source}.pt` (whole-train-fit torch
  `state_dict` per cls source).

## Parallelism

- OOF folds inside one cls source: always concurrent (hardcoded), see above.
- Repeats (`--max-parallel N`): each in its own OS process
  (`ProcessPoolExecutor`), own CUDA context.
- Multi-GPU: with `--gpus 0 1` (or auto-detected when >1 GPU visible),
  worker processes are pinned round-robin to a GPU via
  `CUDA_VISIBLE_DEVICES`, set in the pool's `initializer` before any
  CUDA-touching code runs in that process — no changes needed to the
  `"cuda"` device strings already in `baselines.py`/`attention_extractor.py`,
  since each pinned process only ever sees one (relabeled) GPU. A worker
  keeps its GPU for every repeat it picks up. Verified on a real 2xA100
  box: two PIDs, one per GPU, ~800MiB each at idle/startup.

## Open problem: embedding-based cls features break under the real OOF+boost pipeline

After switching cls sources to emit embeddings, a full server run on
`full_full/transductive` with `attn_lse` produced catastrophically bad
scores for every combo containing `cls` — not just weak, but **worse than
random**:

```
combo    cls                0.218  (AUROC, mean over 5 folds)
         cls+mol            0.240
         cls+prot           0.378
         cls+prot+mol       0.735
         prot               0.794
         prot+mol           0.895
```

Diagnosis so far (three isolated tests on real full_full fold-1 data,
`/tmp/diag_lse*.py`, not yet cleaned up as this is unresolved):

1. **Model training + `embed()` pooling in isolation are fine.** Training
   one whole-train model and evaluating its own scalar `forward()`
   prediction gives AUROC 0.84-0.90 (val/test). Fitting a plain
   `LogisticRegression` directly on that *same* model's val/test embeddings
   gives AUROC 0.83-0.84. So the pooling formula itself is not buggy, and
   a single model's embedding space is genuinely informative.

2. **The real OOF + XGBoost path reproduces the failure exactly**, in a
   single process, no multi-GPU/multiprocessing involved: solo `cls`
   (`MilLseExtractor.fit_transform` with default `n_folds=5`, then
   `fit_boost` on top) gives AUROC 0.329 (val) / **0.224 (test)** — matching
   the server's catastrophic numbers almost exactly. No NaNs, embeddings
   have sane-looking scale (mean ~0.5-0.6, std ~0.5-0.7, max ~4-9).

Working hypothesis: **the train-side OOF embeddings and the val/test
whole-train embeddings come from six *different, independently trained*
model instances** (5 OOF folds + 1 whole-train fit). Unlike a calibrated
scalar probability (a canonical 0-1 number, meaningful regardless of which
model instance produced it), an arbitrary hidden-layer embedding is *not*
identifiable across independently retrained networks — different random
inits/trajectories can rotate/rescale the same semantic content into
different bases. A booster trained on the geometry of 5 OOF models'
embedding spaces has no reason to transfer to a 6th, differently-trained
model's embedding space for val/test — this could easily manifest as
noise, or (given the double asymmetry of transductive's noisy-train/
curated-test split) as the systematic anti-correlation actually observed.

Not yet tested: whether this reproduces on `curated_full` at real OOF scale
(only a small, weakened-but-not-inverted smoke test was run there before
the bug was noticed on the server); whether some form of embedding
alignment/anchoring across the 6 models (e.g. a shared frozen backbone, or
whitening each model's embeddings against its own val fold before handing
them to the booster) fixes it; or whether cls sources should keep emitting
the scalar prediction *alongside* the embedding rather than instead of it.
