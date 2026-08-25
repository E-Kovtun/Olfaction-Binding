# Full_full LoRaX site-MIL / attention screen and ensemble feature mode

This note explains the full_full mixed-granularity attention/MIL experiment added in July 2026, how it is run, what it saves, and how another agent can reuse its outputs as ensemble features.

The short version: the experiment trains several molecule-site-level models on the honest LoRaX `full_full` setting. Each model receives one protein-level ESM vector and a variable-length set of molecule-site GIN vectors. The saved test/validation predictions are intended to be compared directly with ordinary boosting and can later be used as extra features in an ensemble/stacking setup.

## Files

Main training script:

```text
legacy/scripts/modeling/train/train_full_full_site_mil_attention.py
```

Queue / dashboard launcher:

```text
legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

Plot-only notebook:

```text
legacy/notebooks/baselines_research/attention_full_full_site_mil.ipynb
```

Default output directory:

```text
results/attention/full_full/site_mil/
```

## Data dependencies

The experiment is intentionally not based on the older curated pair file. It works on the LoRaX `full_full` data and uses the revised data inventory.

Required LoRaX folds:

```text
data/external/lorax_m2or/rand_split_1/
data/external/lorax_m2or/rand_split_2/
data/external/lorax_m2or/rand_split_3/
data/external/lorax_m2or/rand_split_4/
data/external/lorax_m2or/rand_split_5/
```

Each fold is expected to contain:

```text
train_df.csv
val_df.csv
test_df.csv
```

Required molecule bridge:

```text
data/processed/molecules/lorax_smiles_to_inchikey.csv
```

This is the canonical bridge for this experiment. Do not use `molecule_smiles_all_m2or.csv` for LoRaX here: that older path led to a false low-coverage estimate.

Required molecule embeddings:

```text
data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz
data/embeddings/molecules/gin_supervised_contextpred_all_m2or_per_atom.npz
```

The mean GIN file is compact `.npz` with `ids` and `emb`. The per-atom/per-site file is old-style `.npz` with one InChIKey per key. The loader supports both formats.

Required protein embedding:

```text
data/external/lorax_m2or/esm1b_650m_mean_lorax.npz
```

This is keyed by protein sequence and is the protein embedding used by LoRaX-style full_full experiments. The receptor inventory lives at:

```text
data/processed/proteins/receptor_master.csv
```

but the training script does not need to read it directly; it only needs the ESM `.npz`.

## Coverage

With the revised bridge, coverage is effectively full:

- LoRaX molecules covered by GIN: `595/596`;
- LoRaX receptors covered by ESM1b: `1237/1237`;
- total covered rows per LoRaX fold: `46176/46563`;
- LoRaX test rows covered: `1565/1565`.

The one missing molecule is dropped. This is preferable to adding an artificial fallback, because all compared methods then operate on the same real-embedding subset.

The script records coverage fields in each run's `metrics.json`/`metrics.csv`, so this can be audited after a server run.

## Split protocols

Two regimes are implemented.

### `transductive`

Uses the genuine LoRaX random folds:

```text
repeat = 1..5
```

For example:

```bash
uv run python legacy/scripts/modeling/train/train_full_full_site_mil_attention.py \
  --regime transductive \
  --repeat 1 \
  --model attention_excess_max
```

Here `repeat` means LoRaX fold number. The fold's own `train_df.csv`, `val_df.csv`, and `test_df.csv` are used as-is, after the embedding-coverage filter.

### `inductive_molecule`

Uses the fold-1 full pair pool as the source container, then constructs cold-molecule splits:

```text
repeat = 42, 43, 44, 45, 46
```

Here `repeat` means the cold split seed.

Implementation details:

1. Concatenate fold-1 train/val/test.
2. Drop duplicate rows by `smiles`, `protein`, `_DataQuality`.
3. Filter to rows with GIN molecule embeddings and ESM protein embeddings.
4. Use EC50 rows to define held-out cold molecules for validation/test.
5. Train uses all rows whose molecule is not held out.
6. Validation/test use EC50 rows for held-out molecules.

This mirrors the project's full_full baseline logic: changing the nominal LoRaX fold does not create independent cold-molecule splits; changing the cold split seed does.

## Models

The queue runs six methods by default:

```text
boost_gin_mean
mil_max
mil_noisy_or
mil_lse
attention_mil
attention_excess_max
```

All non-boosting methods use:

- protein input: one `1280`-d ESM1b mean vector;
- molecule input: variable-length site/atom tokens, each `300`-d GIN;
- hidden dimension default: `64`;
- positive oversampling default: `POS_FRACTION=0.5`;
- early stopping by validation AUPRC.

### `boost_gin_mean`

Control model. It ignores per-site structure and trains the usual XGBoost head on:

```text
[GIN_mean_300 || ESM1b_mean_1280]
```

This is the local baseline inside the same filtered full_full universe. It is not the old ChemBERTa+ESM full_full baseline.

The boost model is trained with `orbind.baselines.train_boost`. Validation predictions are computed for threshold selection; test predictions are saved for plotting/ensemble use.

### `mil_max`

The model scores every molecule site independently in the protein context, then the pair logit is:

```text
max(site_logit_i)
```

This is the hardest "at least one site is enough" MIL assumption.

Interpretation:

```text
The pair is positive if one site looks strongly positive.
```

### `mil_noisy_or`

The model scores every site, converts site logits to site probabilities, and pools them as:

```text
p_pair = 1 - product_i(1 - p_site_i)
```

This is the most literal probabilistic form of the "at least one site is active" assumption.

Important caveat: `noisy_or` can saturate near `1.0` if a molecule has many sites with even moderately positive site probabilities. This makes its output useful as a ranking feature but often poorly calibrated as a probability.

### `mil_lse`

Smooth max pooling. The pair logit is:

```text
tau * (logsumexp(site_logit_i / tau) - log(N_sites))
```

Small `tau` behaves closer to max. Larger `tau` allows several moderate site signals to accumulate.

Interpretation:

```text
The pair is positive if one or several sites look good.
```

### `attention_mil`

This is a genuine normalized attention MIL model.

It projects the protein and sites, computes gated attention weights over molecule sites, forms an attention-weighted molecule bag representation, and predicts binding with a learned head.

The final prediction is not itself the max attention weight. Attention concentration can be inspected separately from the saved checkpoint in the notebook.

### `attention_excess_max`

This is the project's custom constrained-attention idea.

The model computes normalized attention weights over molecule sites:

```text
softmax(protein_query dot site_key)
```

Then it uses the maximum attention weight, corrected for the uniform baseline:

```text
raw_max = max_i attention_i
uniform = 1 / N_sites
score = (raw_max - uniform) / (1 - uniform)
```

The score is clipped to `[0, 1]` and treated as the final prediction.

Training uses BCE plus optional margin pressure:

```text
negative pairs: score should be below NEG_THRESHOLD
positive pairs: score should be above POS_THRESHOLD
```

Defaults:

```text
NEG_THRESHOLD=0.2
POS_THRESHOLD=0.4
MARGIN_WEIGHT=1.0
```

Important caveat: this score measures "excess attention concentration", not a calibrated binding probability. A high score can mean "the model confidently focuses on one site", which is not always the same as "the pair binds".

## Metrics and thresholds

The script computes:

```text
AUROC
AUPRC
MCC
F1
precision
recall
```

AUROC and AUPRC are threshold-independent.

MCC, F1, precision and recall are computed on the test set using a threshold selected on validation.

Default threshold logic:

```text
threshold = best validation F1 threshold
```

Unless explicitly overridden with:

```bash
--decision-threshold ...
```

or, for `attention_excess_max`, unless the launcher is run with:

```bash
USE_MARGIN_THRESHOLD=1
```

In the default queue, `USE_MARGIN_THRESHOLD=0`, so even `attention_excess_max` uses a validation-tuned threshold for reported F1/MCC/precision/recall.

This matters: the F1 shown in the notebook is not necessarily F1 at `0.5`.

## Queue behavior

Default full queue:

```bash
bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

Default grid size:

```text
2 regimes * 5 repeats * 6 models = 60 run units
```

The queue supports:

```bash
MAX_PARALLEL=2 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

Useful examples:

```bash
MODEL=attention_excess_max MAX_PARALLEL=1 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

```bash
REGIME=transductive MODEL=all MAX_PARALLEL=2 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

```bash
REGIME=inductive_molecule MODEL=mil_noisy_or MAX_PARALLEL=1 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

The dashboard reports:

- total/done/failed/running/pending;
- current stage per run;
- latest important log line;
- output log directory.

Default output paths:

```text
results/attention/full_full/site_mil/logs/
results/attention/full_full/site_mil/run_status/
results/attention/full_full/site_mil/runs/
results/attention/full_full/site_mil/predictions/
results/attention/full_full/site_mil/checkpoints/
results/attention/full_full/site_mil/metrics.csv
```

## Run IDs and saved files

Run IDs have this form:

```text
{regime}_rep{repeat}_{model}_seed{seed}_boost{boost_seed}
```

Examples:

```text
transductive_rep1_attention_excess_max_seed42_boost1042
inductive_molecule_rep42_mil_noisy_or_seed42_boost1042
```

For each run:

```text
runs/{run_id}/metrics.json
runs/{run_id}/metrics.csv
predictions/test_{run_id}.csv
predictions/val_{run_id}.csv
```

Torch models additionally save:

```text
checkpoints/{run_id}.pt
```

Boosting runs currently save predictions and metrics, not a reusable XGBoost model artifact.

## Prediction files as ensemble inputs

The prediction CSVs are the intended interface for future ensemble/stacking work.

Columns include:

```text
smiles
inchikey
protein
label
split
y
p
```

Where:

- `label`/`y` is the binary target;
- `p` is the model score;
- `split` is `val` or `test`;
- `(smiles, inchikey, protein)` identifies the pair.

To build an ensemble table, join prediction files by:

```text
regime
repeat
split
smiles
inchikey
protein
label
```

Then use each model's `p` as a feature, for example:

```text
p_boost_gin_mean
p_mil_max
p_mil_noisy_or
p_mil_lse
p_attention_mil
p_attention_excess_max
```

Important: validation predictions can be used to fit a second-level blender for a given repeat. Test predictions should only be used for final evaluation. Do not train an ensemble directly on test predictions.

Also remember that the saved `p` values are not equally calibrated:

- `boost_gin_mean` is closest to an ordinary probability score;
- `mil_noisy_or`, `mil_max`, and `mil_lse` may saturate near 1;
- `attention_excess_max` is a concentration score, not a probability.

For a robust ensemble, consider adding transformed features:

```text
p
rank(p) within repeat/model
logit(clipped p)
quantile(p) within repeat/model
```

Rank/quantile features may be especially useful because AUROC/AUPRC care about ordering and several MIL models are poorly calibrated but still informative.

## Plot-only notebook

The notebook:

```text
legacy/notebooks/baselines_research/attention_full_full_site_mil.ipynb
```

does not train anything. It reads:

```text
results/attention/full_full/site_mil/metrics.csv
results/attention/full_full/site_mil/predictions/*.csv
results/attention/full_full/site_mil/checkpoints/*.pt
```

It currently contains:

1. experiment description;
2. raw run table;
3. mean summary by `(regime, model)`;
4. barplots with 95% CI over repeats;
5. coverage table;
6. score-distribution plots;
7. attention-weight diagnostics for `attention_mil` and `attention_excess_max`.

The attention-weight diagnostics recompute max attention weights from checkpoints. Nothing is retrained.

## Known interpretation caveats

### Saturated histograms are expected

Several MIL methods produce score distributions piled up near 0 and/or 1. This does not automatically contradict AUROC or AUPRC.

AUROC measures ranking:

```text
P(score_positive > score_negative)
```

So two classes can both be near `1.0` and still have nontrivial AUROC if positives are slightly more extreme.

AUPRC is harsher under class imbalance: a relatively small number of high-scoring negatives can strongly reduce precision in the high-score region.

### `noisy_or` is sensitive to molecule size

Because:

```text
1 - product(1 - p_site_i)
```

increases with the number of sites, larger molecules can receive higher bag scores even when individual site probabilities are modest. This is a standard noisy-OR MIL issue.

### `attention_excess_max` is not a conventional probability

Its score is the normalized excess of the maximum attention weight over uniform attention. It is useful as a structured signal, but should not be interpreted as calibrated binding probability.

## Practical server workflow

After pushing code and transferring data to the server:

```bash
MAX_PARALLEL=1 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

On a single GPU, increasing parallelism can help until memory or scheduling overhead dominates:

```bash
MAX_PARALLEL=2 bash legacy/scripts/queues/run_full_full_site_mil_attention.sh
```

The earlier successful run used `MAX_PARALLEL=10`, but that is aggressive for one GPU and should be treated as opportunistic rather than safe default.

To inspect completion:

```bash
ls results/attention/full_full/site_mil/runs/*/metrics.json | wc -l
```

Expected full count:

```text
60
```

To inspect failures:

```bash
grep -R "Traceback\\|RuntimeError\\|ValueError\\|XGBoostError" results/attention/full_full/site_mil/logs/
```

## Current intended use

This experiment is not yet the final ensemble itself. It is a structured generator of additional model scores:

1. compare the MIL/attention methods against `boost_gin_mean`;
2. identify which scores carry signal in transductive and inductive regimes;
3. use validation/test prediction CSVs as inputs to a future ensemble/blender;
4. possibly include attention diagnostics as interpretability/sanity-check material.

The most promising downstream path is to treat the site-MIL/attention scores as extra features alongside ordinary molecule/protein embedding features in a carefully split stacking setup.
