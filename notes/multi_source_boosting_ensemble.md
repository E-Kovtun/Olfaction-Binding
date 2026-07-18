# Multi-source boosting ensemble

Generalized, symmetric replacement for the ProSmith/LORAX-style boosting
ensemble: instead of a fixed solo/pair/triplet scheme over exactly three
models, any number of feature sources can be combined via a small
combo mini-language, each combo gets its own XGBoost head, and the heads
are combined via a fitted ensemble weighting.

Code: `orbind/ensemble.py` (engine), `orbind/attention_extractor.py` (MIL
cls sources), `orbind/gnn_extractor.py` (graph cls source), `orbind/regimes.py`
(full_full split bookkeeping), `orbind/baselines.py` (boosting head, fixed or
tuned), `scripts/modeling/train/train_ensemble_boost.py` (CLI).

## Combo mini-language

Sources are registered in a fixed order (1-based). `--combos "1 2 12"` means:
solo source 1, solo source 2, and the concatenation of both. Any subset in
any digit string works (`"1 23 123"`, `"14"`, ...) — this is the symmetric
generalization of ProSmith/LORAX's fixed 3-source solo/pair/triplet scheme.

For each combo, `run_ensemble` concatenates that combo's sources' features
and fits one boosting head on train, producing val/test probabilities.

## Three source families (`Extractor` protocol)

All share the contract `fit_transform(pairs, train_idx, val_idx, test_idx,
seed, checkpoint_dir=None) -> (Xtr, Xval, Xte)`.

- **Entity-level** (`EsmExtractor`, `GinExtractor`, `orbind/ensemble.py`):
  label-independent static `.npz` lookups (protein or molecule embeddings).
  No training, nothing to leak, `checkpoint_dir` is a no-op.
- **Pair-level MIL / "cls"** (`MilNoisyOrExtractor`, `MilLseExtractor`,
  `orbind/attention_extractor.py`): supervised, trained fresh every run on
  `train_idx`'s own labels. Reimplementations of the two best-performing
  site-MIL pooling rules from the project's own screen (noisy-OR and
  log-sum-exp over per-site logits), rewritten as two fully independent
  classes with no import from the older
  `train_full_full_site_mil_attention.py` script.
- **Pair-level graph / "cls"** (`GnnSignedExtractor`, `orbind/gnn_extractor.py`):
  a standalone reimplementation of this project's own "signed" GraphSAGE
  architecture (see `orbind/hetero.py`), with no import from that module or
  its training scripts. Node features start as mean-pooled GIN (molecule) /
  mean-pooled ESM (protein); message-passing edges are the train split's own
  pairs, with positive/negative edges flowing through separate SAGEConv
  stacks (subtracted per layer — "signed"), plus an optional per-molecule
  quantile filter (`q`, default 0.99, "q99") that keeps only the
  most-measured molecules as message-passing participants (supervision
  still covers every row regardless of the filter).

## Why cls sources need honest features, and how that evolved

Every combo — including a solo cls combo — gets re-boosted by its own
boosting head. If a cls source's train-row "feature" came from a model that
was fit on those same train rows' labels, the second-level booster would be
training on leaked information.

**Original scheme (retired): true out-of-fold (OOF) fold-holdout.** `val`/
`test` came from one whole-train fit; `train` came from a stratified 5-fold
refit (each fold held out, a fresh model trained on the other 4, predicting
only the held-out fold). Fine as long as each cls source emitted a
*calibrated scalar prediction* (a canonical 0–1 number, meaningful
regardless of which model instance produced it).

**When cls sources were changed to emit a pooled embedding instead of a
scalar** (see below), this scheme broke badly: a boosting head trained on
5 *different* fold-models' embedding geometries had no reason to transfer
to a 6th, differently-trained whole-train model's geometry for val/test —
random inits/trajectories rotate/rescale an otherwise-equivalent latent
basis arbitrarily across independently-trained instances. A real server run
on `full_full/transductive` with embedding-valued `attn_lse` produced
catastrophically anti-correlated scores for every `cls`-containing combo
(AUROC ~0.22–0.38, worse than random) — confirmed reproducible in isolation
(single process, no multi-GPU) down to a minimal repro script; a plain
scalar-valued whole-train-only test on the same data gave a sane
LogisticRegression AUROC of 0.83–0.84, proving the embedding itself and its
pooling formula weren't buggy — only the *cross-model* combination was.

Digging into ProSmith/LORAX's own actual code (`olfactory_foundation_models`,
`lorax` sibling repos) confirmed the root cause and the fix: **they never
have more than one instance of the embedding-producing model per split** —
one frozen whole-train-fit transformer embeds train/val/test alike, openly
accepting mild train-row leakage (bounded by early stopping on real val)
rather than cross-fitting. There is never a second, differently-trained
model instance to misalign against.

**Current scheme: `n_models` bagging (`_run_models` in both
`attention_extractor.py` and `gnn_extractor.py`).** Every one of `n_models`
independently-seeded whole-train model instances sees ALL of train_idx as
its own training signal (MP edges for the GNN; the whole labeled train set
for MIL) and embeds every row (train **and** val/test) through itself —
accepting the same mild leakage ProSmith/LORAX accept. The N models'
embeddings are concatenated (`N * per-model-dim` columns). `n_models=1`
reduces exactly to ProSmith's own single-model scheme; `n_models>1` (default
5) is a bagging ensemble of it. Because every model embeds every split,
there's no missing/misaligned block to reconcile — verified this actually
fixes the bug: the same `full_full/transductive` real-data test that gave
AUROC 0.22 under fold-holdout OOF gives **AUROC 0.885 (test)** under
`n_models=5` bagging (dim=32 → 64-d per model, 320-d total for `attn`)
— matching the raw-boost ceiling (~0.89).

## cls sources emit a pooled embedding, not a scalar

Each MIL/graph cls model's own training objective (BCE loss on a scalar
noisy-OR/LSE-pooled or GNN-decoder logit, early stopping on val AUPRC) is
unchanged. What's handed to the boosting stage is not that scalar
prediction, but a pooled *pre-head hidden representation*:

- MIL noisy-OR: per-site sigmoid probabilities, normalized to sum to 1,
  weight-average the per-site hidden vectors (2*dim, default dim=32 → 64-d).
- MIL LSE: per-site `softmax(logit / temperature)`, same weighted average.
- GNN: no separate pooling step needed — `[z_mol || z_prot]` (2*hidden,
  default hidden=256 → 512-d per model) straight from the encoder.

This mirrors ProSmith's own cls-token design (a real learned embedding fed
to boosting, not a scalar), which is directly what let their design (and now
ours) work once combined with the `n_models` fix above.

## Ensemble weighting (two methods, `--weight-method both|simplex|logreg`)

Both fit on the combos' validation-set predictions, evaluate once on test:

- **`simplex`**: non-negative weights summing to 1, minimizing validation
  log-loss (`scipy.optimize.minimize`, SLSQP). Directly comparable to the
  ProSmith/LORAX weighting scheme (they use a brute-force grid search over
  the same 2-simplex at 0.01 resolution, ~5000 points, maximizing val MCC —
  same idea, continuous optimizer instead of a grid).
- **`logreg`**: unconstrained logistic-regression stacker over the combo
  predictions (coefficients can be negative, don't sum to 1 — read as
  relative importance, not literal mixing weights). No analogue of this in
  ProSmith/LORAX.

**Real weights from three server runs** (full_full/transductive, mean
simplex weights over 5 folds, `cls` = attn_noisy_or / attn_lse / gnn_signed
respectively): `prot+mol` is always the single heaviest combo (39–51%), but
cls-containing combos collectively take a comparable share (46–60% combined)
— the ensemble does *not* collapse to ignoring cls, contrary to an earlier,
too-strong reading of the numbers. Despite that real weight, no combo beats
solo `prot+mol`'s own test AUROC (0.895) in any of the three runs — cls adds
weight on validation without a confirmed net test win; still open whether
that's fold noise or a real (if small) generalization gap. GNN's
`cls+prot+mol` got the largest cls-involving weight of the three (0.308,
vs. attn's 0.07–0.15), i.e. GNN's embedding seems to carry more information
*complementary* to prot+mol even though its solo AUROC (0.863) is the
weakest of the three cls variants (attn_noisy_or 0.886, attn_lse 0.881).

## Optional: per-combo XGBoost hyperparameter tuning (`--tune-boost`)

Off by default (fixed hyperparameters, `orbind/baselines.py::fit_boost`).
With `--tune-boost --n-trials N`, each combo's boosting head is tuned
independently via `orbind.baselines.tune_boost` (optuna, TPE sampler,
default space closely mirrors ProSmith/LORAX's own hyperopt space:
`learning_rate`, `max_depth`, `reg_lambda`, `reg_alpha`, `min_child_weight`,
`max_delta_step`, `subsample`, `colsample_bytree`, `n_estimators`, plus a
`scale_pos_weight` multiplier). Objective = val AUPRC (matching this
project's own early-stopping metric convention elsewhere). ProSmith/LORAX
default to 2000 hyperopt trials per head (500 in their own README example);
importantly, in both their pipeline and ours, tuning only retrains the cheap
boosting head per trial — the expensive cls model (transformer / MIL /
GNN) is fit once and cached, not retrained per trial — so `n_trials` in the
several-hundreds range is realistic cost-wise, same as theirs.

Per-combo tuning is logged: a `tune[combo]: N trials, best val AUPRC=...,
params=...` line in `logs/repeat_{R}.log`, plus the full per-trial history
(`study.trials_dataframe()`) saved to
`checkpoints/repeat_{R}/optuna_{combo}.csv` when checkpoints are enabled.

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
(also the basis of the LORAX-paper critique discussed earlier) is why cls
sources' train-row leakage matters more here than it might elsewhere.

## Storage / logging (`results/ensemble_logs/<run_id>/`)

Every CLI invocation creates one timestamped run folder:

- `config.json` — every CLI arg, for reproducibility.
- `log.txt` — the process's own stdout (setup + final summary).
- `logs/repeat_{R}.log` — one per repeat, captures its combo/ensemble/tuning
  lines even when `--max-parallel` runs it in another process.
- `metrics.csv` — one row per (repeat, combo) and per (repeat, ensemble
  method) — not just the final ensemble score. Ensemble rows carry a
  `weights` column (JSON dict of combo -> weight) for `simplex`/`logreg`.
- `checkpoints/repeat_{R}/`: `boost_{combo}.json` (one XGBoost booster per
  combo, via `clf.get_booster().save_model()`), `attn_{source}_model{k}.pt` /
  `gnn_{source}_model{k}.pt` (one torch state_dict per model, per pair-level
  source, `k` in `0..n_models-1`), `optuna_{combo}.csv` (full per-trial
  hyperparameter history, only with `--tune-boost`).

## Parallelism

- A pair-level source's `n_models`: always trained concurrently (hardcoded),
  see above.
- Repeats (`--max-parallel N`): each in its own OS process
  (`ProcessPoolExecutor`), own CUDA context. **Must use the `"spawn"` start
  method explicitly** (`multiprocessing.get_context("spawn")`, not the
  platform default) — on Linux the default is `"fork"`, and GPU
  auto-detection (`_detect_gpus`) already touches `torch.cuda` in the parent
  process before the pool is created; forking a child that inherits an
  initialized CUDA context fails with `CUDA error: initialization error`.
  Windows already defaults to spawn, which is why this only surfaced on the
  actual (Linux) server, not in any local testing.
- Multi-GPU: with `--gpus 0 1` (or auto-detected when >1 GPU visible),
  worker processes are pinned round-robin to a GPU via
  `CUDA_VISIBLE_DEVICES`, set in the pool's `initializer` before any
  CUDA-touching code runs in that process — no changes needed to the
  `"cuda"` device strings already in `baselines.py`/`attention_extractor.py`/
  `gnn_extractor.py`, since each pinned process only ever sees one
  (relabeled) GPU. A worker keeps its GPU for every repeat it picks up.
  Verified on a real 2xA100 box.

## Comparison to the actual ProSmith/LORAX codebases (read directly, not just the papers)

- **`olfactory_foundation_models`** (the paper's own ProSmith reproduction,
  not LORAX itself): fixed 3-model ensemble (embeds-only ESM1b+ChemBERTa
  mean-pooled / embeds+cls / cls-only), a real BERT-style transformer whose
  CLS hidden state (768-d, pre-final-head) feeds boosting, ProSmith's own
  molecule encoder for that transformer is a pretrained SMILES token
  sequence (ChemBERTa/MolT5/etc — sub-word tokens of the SMILES string, not
  atoms/graph nodes), GIN only used in simpler non-transformer baselines.
- **`lorax`** (the actual paper repo): fine-tunes the real pretrained
  ESM2-650M + ChemBERTa-77M-MTR checkpoints via LoRA adapters
  (`peft.get_peft_model`), fuses them with bidirectional cross-attention,
  trains end-to-end on task labels — then still runs the same
  ProSmith-style 3-model XGBoost/simplex-grid ensembling machinery on top
  ("Adapted from prosmith" in their own code comments), just fed by the
  fine-tuned representations instead of frozen ones. No graph/GIN model
  anywhere in this repo.
- Neither repo uses honest OOF cross-fitting for their cls features — both
  accept the same train-leakage tradeoff our `n_models` scheme now also
  accepts, for the same reason (avoiding cross-model geometry mismatch).

## ESM2 mean-embedding coverage gap (found while extending prot embeddings)

`data/embeddings/proteins/esm2_650m_mean.npz` only covered 780 receptors
(from `pairs_m2or_full.csv`), while the full_full/LoRaX pool needs 1237
unique receptor sequences — 464 missing. `esm1b_650m_mean.npz` was already
complete (0 missing) against the same 1237. Canonical generator:
`scripts/embedding_generation/proteins/05_per_residue_embeddings.py` (derives
mean from a fresh per-residue ESM-2 forward pass; supports `--reuse` to copy
already-computed receptors instead of recomputing). Gap-filled locally by
reusing the existing 780 and running ESM-2 only on the missing 464, so
`esm2_650m_mean.npz` reaches the same full coverage `esm1b_650m_mean.npz`
already had.
