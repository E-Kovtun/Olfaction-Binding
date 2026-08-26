# orbind

Receptor-side modelling of olfactory receptor–odorant binding.

The question the code is built around is **what a protein representation has to
carry** for a binding model to generalize. A plain gradient-boosted head over
frozen embeddings (`ESM ‖ ChemBERTa → XGBoost`) is a very strong baseline that
mostly reads *receptor identity*; this repository contains the experiments that
measure that, the refinement that adds function-derived structure to the
receptor vector (a signed bipartite receptor↔odorant graph), and the head-to-head
against four published interaction models re-implemented on our splits.

Three datasets, one pipeline: **M2OR** (human ORs, binary), **Carey** (`cc`,
mosquito *AgOr*, 50×110 continuous) and **Hallem–Carlson** (`hc`, fly, 24×110
continuous).

---

## Layout

```text
orbind/       the library: datasets, splits, extractors, the ensembler
scripts/      entry points (preprocessing, embeddings, training, analysis)
notebooks/    display/analysis notebooks; the models they read come from scripts
notes/        protocol decisions and the paper's storyline
legacy/       every closed line — scripts, notebooks, notes, experiments
data/         external datasets and embeddings (not versioned — see data/README.md)
results/      run outputs: metrics, logs, checkpoints (not versioned)
```

Two rules make the tree readable:

* **Nothing in `legacy/` feeds a paper table**, and nothing live imports from it.
  Most of it is a negative result kept so nobody re-runs it. Archived *library*
  modules are the one exception to the location: they sit in `orbind/legacy/`
  because they must stay on the import path for archived consumers to run.
* **Notebooks never train the models they display.** Every number in the paper
  comes from a script writing to `results/`.

---

## Environments

Four, on purpose — the source dispatch in the trainer imports each method's deps
lazily, so a ProSmith/LORAX run needs no PyG and the fragile torch↔PyG pin stays
confined to the graph pipeline.

| env | path | holds |
|---|---|---|
| project | `.venv` | graph + ensemble pipeline (torch, torch-geometric, xgboost) |
| controls | `.venv-controls` | ProSmith / LORAX baselines, PyG-free |
| embeddings | `.venv-embeddings` | run-once embedding generation (fair-esm, deepchem, rdkit) |
| molor | `.venv-molor` | MolOR only (dgl 2.4 + dgllife — install `dgl` **before** `dgllife`) |

```bash
uv python install 3.11
uv sync --frozen                 # the project env (.venv)
bash scripts/setup_envs.sh       # controls + embeddings
```

Scripts add the repo root to `sys.path` themselves, so `orbind` imports without
being installed — call an env's interpreter directly:
`.venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py ...`.

**Running a method in the wrong env is the failure mode to watch for.** It does
not always crash: it can silently fall back to an existing checkpoint. A MolOR
"training run" that finished in 5 s was exactly this.

---

## Data

`data/` is not versioned. **[`data/README.md`](data/README.md)** is the manifest:
what each file is, which script produces it, and which experiment needs it.

---

## The pipeline: one entry point

Every table in the paper comes from
[`scripts/modeling/train/train_ensemble_boost.py`](scripts/modeling/train/train_ensemble_boost.py).
It fits one boosting head per *combination of sources* and writes a timestamped
run folder under `results/ensemble_logs/`.

**Sources.** Each `--source name=type:...` registers one embedding extractor.
Entity-level ones (`esm`, `gin`) are static npz lookups; the rest train their own
model per fold and emit a pair-level `cls` vector:

| type | what it is |
|---|---|
| `esm`, `gin` | frozen protein / molecule embeddings from an npz |
| `gnn_signed` | **ours** — signed bipartite receptor↔odorant graph, refines the receptor vector |
| `lorax` | LoRA-ChemBERTa + cross-attention over frozen per-residue ESM-1b |
| `prosmith` | ProSmith/MPP transformer over per-residue protein + pooled molecule |
| `molor` | dgllife GCN cross-attending frozen per-residue ESM-1b |
| `hladis` | Receptor2Odorant (ICLR 2023): MPNN-attention, receptor broadcast onto every atom |

**Combos.** `--combos "1 2 12"` is a digit-string mini-language: each digit is the
**1-based position of a `--source` flag on that command line**. With
`--source cls=... --source prot=... --source mol=...`, `1` = cls alone, `23` =
prot+mol (the boosting baseline), `123` = cls+prot+mol. One combo = one boosting
head over the concatenation of its sources' features.

> Adding or reordering a `--source` flag silently changes what every digit means.
> `metrics.csv` records combos by *name*, so compare names, never digits.

**What we report.** One combo, one head, fixed hyperparameters. The machine can
also weight several combos into an ensemble and tune each head by hyperparameter
search; both are implemented, and both are deliberately switched off in everything
reported — see [`orbind/docs/ensembler.md`](orbind/docs/ensembler.md) for the whole
mechanism and the reasoning.

**Regimes and splits.**

| `--regime` | splits | flag |
|---|---|---|
| `curated_full` | stratified / group_molecule / group_receptor over a pairs csv | `--split`, `--seeds` |
| `full_full` | M2OR on LORAX's own pool: `transductive`, `inductive_molecule`, `inductive_molecule_v5` | `--full-full-mode`, `--repeats` |
| `ofm` | Carey / Hallem: `rand`, `cdhit`, `scaf`, `our_inductive` | `--dataset`, `--split-family`, `--repeats` |

`inductive_molecule_v5` is the cold-molecule split every M2OR baseline is compared
on. On the insect datasets the cold-molecule split of record is **`our_inductive`**,
ours, not upstream's `scaf` — see [Split validity](#split-validity) below.

**Task.** `--task {classification,regression}`; `--regime ofm` defaults to
regression, which swaps the head to `XGBRegressor` and the metrics to
R²/RMSE/MAE/Pearson/Spearman.

---

## Reproducing the paper

All runs go to `results/ensemble_logs/<pool>/<run>/metrics.csv`, one row per
(fold, combo) plus a `naive[train-mean]` row. Pool names below are the ones the
recorded results use.

Shared paths (M2OR):

```bash
PROT=data/embeddings/proteins/esm1b_650m_mean.npz
PRES=data/embeddings/proteins/esm1b_650m_per_residue_full_full.npz
MOL=data/embeddings/molecules/chemberta_77m_m2or.npz
```

### T1 / T1b — competitors head-to-head on M2OR

Every method as a `cls` source, scored both alone (T1) and concatenated with the
raw protein and molecule embeddings (T1b), against the boosting base.
Pools: `m2or-{transductive,inductive}-chemberta-fixed`.

```bash
.venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime full_full --full-full-mode inductive_molecule_v5 \
    --run-name inductive_lorax_chemberta \
    --source cls=lorax \
    --source prot=esm:$PROT:esm1b_t33_650M_UR50S \
    --source mol=gin:$MOL:chemberta_77m \
    --combos "1 123" --on-missing drop --max-parallel 1 --repeats 42 43 44 45 46
```

Swap `cls=lorax` for `cls=prosmith::::data/external/ofm/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl`,
`cls=hladis`, or (in `.venv-molor`) `cls=molor`. Our graph runs in the project env:

```bash
.venv/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime full_full --full-full-mode inductive_molecule_v5 \
    --run-name inductive_gnn99signed_chemberta \
    --source cls=gnn_signed:$PROT:$MOL \
    --source prot=esm:$PROT:esm1b_t33_650M_UR50S \
    --source mol=gin:$MOL:chemberta_77m \
    --combos "13 123" --on-missing drop --repeats 42 43 44 45 46
```

`gnn_signed` defaults are the headline configuration: `q=0.99`,
`criterion=greedy_pair_cover`, `emit=prot`, `n_models=1`. The graph's headline row
is `cls+mol` (`13`) — it deliberately excludes raw ESM. Replace
`--full-full-mode inductive_molecule_v5` with `transductive` and `--repeats 1 2 3 4 5`
for the other regime.

### T2 — molecule-source robustness (M2OR)

The same graph-vs-base comparison with `mol` set to each of ChemBERTa, GIN and
ECFP, both regimes: six pools `m2or-{inductive,transductive}-{chemberta,gin,ecfp}-fixed`.
Only the `mol=` path changes (`gin_supervised_contextpred_all_m2or.npz`,
`ecfp_m2or.npz`), and `gnn_signed`'s own molecule field with it.

### T4 / T5 — transfer to insects (Carey, Hallem–Carlson)

Pools `{cc,hc}-{rand,ourind}-molcross-fixed`; `rand` is the transductive column,
`our_inductive` the cold-molecule one. Regression throughout.

```bash
.venv/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime ofm --dataset cc --split-family our_inductive \
    --run-name cc_ourind_gnn_chemberta \
    --source cls=gnn_signed:data/embeddings/proteins/esm1b_650m_mean_cc.npz:data/embeddings/molecules/chemberta_77m_cc.npz \
    --source prot=esm:data/embeddings/proteins/esm1b_650m_mean_cc.npz:esm1b_t33_650M_UR50S \
    --source mol=gin:data/embeddings/molecules/chemberta_77m_cc.npz:chemberta_77m \
    --combos "13 23" --on-missing drop --repeats 1 2 3 4 5
```

A `_pm` run-name suffix marks the variant that also adds raw ESM (`cls+prot+mol`);
it is null everywhere on these datasets. Only the `*-molcross-fixed` complexes feed
the paper tables — the other cc/hc complexes on disk are older exploratory grids.

### Tp — protein-source floor

Separate script, not the ensembler: real pLMs (ESM-1b, ProtT5) against a classical
amino-acid floor (kmer2, CTD, PseAAC, BLOSUM, AAC, AAIndex) plus `onehot`,
`onehot_only` and `mol_only` controls.

```bash
.venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py --help
```

### T6 — mechanism holdout (`tab:t6`)

Hold out every odorant of a chemical class, train the graph without it, then ask whether the
receptor embedding still says something true about that class. Three receptor representations:
raw ESM (structure), GNN+ESM (both), GNN one-hot (function only).

Each dataset writes its own directory and shares nothing with the others, so the three run
side by side, one per GPU -- no merge step, the artifacts land exactly where the serial run
puts them:

```sh
i=0
for d in m2or cc hc; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset $d > mh_$d.log 2>&1 &
  i=$((i + 1))
done; wait
```

`CUDA_VISIBLE_DEVICES` rather than `--device cuda:N`: it also pins whatever the boosting head
and the extractor pick up on their own. A fourth GPU has nothing to do here -- M2OR is the long
pole and stays one process. Serially, on one GPU, it is the same command with `--dataset all`.

Writes `results/mechanism_holdout/<ds>/`; `notebooks/graph/mechanism_holdout/mechanism_holdout.ipynb`
reads those artifacts and draws them (set `DATASET` in its first cell). The metrics of record
are three geometric ones — RSA, CCA, Procrustes — none with a head or a hyperparameter; `tab:t6`
reports RSA. A predictive-OOD boosting readout runs on the same masks as a differently-shaped
check, and `--hladis` scores a competitor on those masks too. The notebook closes on one number per
representation: each class scored against its own null in units of that null's spread, then
averaged with weights `trust = (1 - struct_leak)(1 - func_redund)` — how isolated the holdout
actually was — printed beside the equal-weight mean. The run also dumps the receptor
embeddings, so any further second-order metric costs no retraining. The two insect matrices are
the stand; M2OR is illustrative.

### Appendix — quantile × criterion sweep

```bash
.venv/bin/python scripts/modeling/train/run_quantile_criteria_sweep.py --dataset cc
```

Read by `notebooks/graph/alternatives/protein_based_graph{,_carey}.ipynb`.
Note the two datasets need different readings of `q`: M2OR's coverage quantile
cuts a long-tailed distribution, while the insect matrices are complete, so
coverage is constant and the quantile is a no-op — `--k-mode fraction` is what
makes the axis mean anything there (`orbind/mol_selection.resolve_K`).

---

## Reading results

```bash
python scripts/analysis/summarize_runs.py                      # everything
python scripts/analysis/summarize_runs.py --pool cc-ourind      # substring filter
python scripts/analysis/summarize_runs.py --pool m2or --folds   # per-fold values
```

One row per (run, combo), mean ± 95% t-CI, columns chosen by task. Fold counts are
printed rather than filtered, so a half-finished run is visible instead of silently
averaged. `scripts/analysis/extend_runs_with_combo.py` prints the commands that add
a `cls+prot+mol` row to an existing cls-only run by reusing its checkpoints.

---

## Before you read a number

Four things decide whether a result means what it looks like. The full list, with
the failures that produced each one, is in
[`orbind/docs/gotchas.md`](orbind/docs/gotchas.md).

* **One combo, one head, fixed hyperparameters** — no combo stacking, no per-head
  tuning, identical settings for us and for every baseline. Why:
  [`orbind/docs/ensembler.md`](orbind/docs/ensembler.md#the-convention-first).
* **Read the `naive[train-mean]` row next to every R².** R² is measured against the
  **test** mean while naive predicts the **train** mean, so a model can beat naive
  and still score below zero.
* **Five seeds minimum on cold-molecule regimes.** Per-seed swings there reach
  ±0.1 AUROC; a one-seed win in this project has already turned out to be noise.
* <a name="split-validity"></a>**On Carey, `our_inductive` is the cold-molecule
  split, not upstream's `scaf`.** `scaf`'s fold 1 lands on the carboxylic-acid
  homologous series (test sd 0.215, naive R² −4.92, which *is* the published
  −1.016 average); `our_inductive` holds naive R² ≈ 0 on every fold.

Two more that bite during a run rather than after it: a method run in the wrong
environment can silently load a checkpoint instead of training (pass
`--skip-checkpoints` for anything timed), and `config.json` records what a run was
*asked* to do — read `metrics.csv` for what it actually produced.

## Tests

```bash
uv sync --frozen --group dev     # once, for pytest
uv run pytest
```

They cover the pure, semantics-carrying functions -- the two readings of the
quantile (`resolve_K`), what "positive" means on a continuous target
(`pos_threshold`), the `--combos` digit language and the `--source` field order.
No data, no GPU, ~10 s. This is deliberately not a test suite for the models: it
pins the plumbing whose meaning can shift without anything crashing.

## Notebooks

See [`notebooks/README.md`](notebooks/README.md). All of them locate the repo root
by walking up to `pyproject.toml`, so they run from any depth.

## Archive

[`legacy/README.md`](legacy/README.md) indexes every closed line and says what each
one showed — the graph line v3–v6, the attention/site-MIL branch, the molecule-side
graph nulls, the pocket/protein-variant cluster, and the exploratory embedding
notebooks. Kept rather than deleted because most of them are negative results.
