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

Five, on purpose — the source dispatch in the trainer imports each method's deps
lazily, so a ProSmith/LORAX run needs no PyG and the fragile torch↔PyG pin stays
confined to the graph pipeline.

| env | path | holds |
|---|---|---|
| project | `.venv` | graph + ensemble pipeline (torch, torch-geometric, xgboost, rdkit) — also Hladiš |
| controls | `.venv-controls` | ProSmith / LORAX baselines, PyG-free, **no rdkit** |
| embeddings | `.venv-embeddings` | run-once embedding generation (fair-esm, deepchem, rdkit) |
| molor | `.venv-molor` | MolOR only (dgl 2.4 + dgllife — install `dgl` **before** `dgllife`) |
| esm | `.venv-esm` | ESM3 / ESM-C embeddings only (EvolutionaryScale `esm` SDK) |

```bash
uv python install 3.11
uv sync --frozen                 # the project env (.venv)
bash scripts/setup_envs.sh       # controls + embeddings
```

`.venv-esm` is separate because the SDK's package is also named `esm` and clashes
with fair-esm. Its install lines are in the docstring of
`scripts/embedding_generation/proteins/embed_proteins_plm.py`; add `httpx` by hand
(the SDK imports it without declaring it), and expect it to pull its own torch
(2.14+cu130 on the server — it works on the A100s there).

**Every env that fits a boosting head must hold `xgboost>=2.0,<3.0`** — the same
head is what all reported numbers share, and 3.x also aborts on some of the
server's GPUs (see [`orbind/docs/gotchas.md`](orbind/docs/gotchas.md)). A run now
refuses to start on the wrong major. `.venv-molor` is built by hand, so pin it
there explicitly:

```bash
uv pip install --python .venv-molor/bin/python "xgboost>=2.0,<3.0"
uv pip install --python .venv-controls/bin/python "xgboost>=2.0,<3.0"
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
`cls=hladis` (in the project `.venv` — it needs rdkit, which `.venv-controls` does
not have), or (in `.venv-molor`) `cls=molor`. Our graph runs in the project env:

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

**The external `cls` baselines must be given the insect protein file explicitly.**
Bare `cls=lorax` (likewise prosmith, molor, hladis) loads its M2OR default, which
holds none of the insect receptors: coverage drops every row and the run dies with
`num_samples=0`. The specs of record, with `{ds}` = `cc` or `hc`:

```
cls=lorax:data/embeddings/proteins/esm1b_650m_per_residue_{ds}.npz
cls=prosmith:data/embeddings/proteins/esm1b_650m_per_residue_{ds}.npz:data/embeddings/molecules/chemberta_77m_{ds}.npz::data/external/ofm/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl
cls=molor:data/embeddings/proteins/esm1b_650m_per_residue_{ds}.npz:1
cls=hladis:data/embeddings/proteins/esm1b_650m_mean_{ds}.npz:1:2000:1200:100
```

with `--combos "1 12 123" --max-parallel 2 --gpus 0 1`. Hladiš's `2000:1200:100` is its
step budget rescaled to the insect panels (see `orbind/docs/gotchas.md`).

### Shrunk insect panels (`cc_shrinked`, `hc_shrinked`, `*_shrinked50`)

The insect panels with most cells declared *not measured*, so that what is left has
M2OR's sparsity profile — to separate "M2OR behaves differently because it is sparse"
from "because it is a different assay". Built by
`scripts/preprocessing/04_build_shrunk_ofm.py`; registered in
`orbind/regimes_ofm.DATASETS` with a `base` key naming the parent panel, whose
embeddings they share (no npz of their own).

| tag | density | mask |
|---|---|---|
| `{cc,hc}_shrinked` | 0.063 (M2OR's own) | `marginal`: M2OR's row/column count profile, IPF + Gumbel top-k |
| `{cc,hc}_shrinked50` | 0.5 | the same profile rescaled; its head saturates against the panel width |

**The mask constrains training, not scoring.** train/val = the parent's split ∩
measured; test = *everything else*, with its labels. Each fold's `test_origin.csv`
says which test rows were the parent's own test block (`origin == "upstream_test"`,
the anchor for a paired comparison with the complete panel) and which entities the
masked train never saw (`cold_molecule`/`cold_receptor` — a wider set). Read R²
against the `naive` row, never against 0: the masked train mean and the test mean differ.

Baseline pools are `{ds}-{rand,ourind}-fulltest`; graph sweeps are
`results/graph/v11_shrunk` (0.063) and `results/graph/v12_shrunk50` (0.5), both
`--dial nodes --seed-graph`.

### Tp — protein-source floor

Separate script, not the ensembler: real pLMs (ESM-1b, ProtT5, ESM3; ESM-2 on M2OR
only) against a classical amino-acid floor (kmer2, CTD, PseAAC, BLOSUM, AAC,
AAIndex) plus `onehot`, `onehot_only` and `mol_only` controls. A pLM whose npz is
absent is skipped with a warning. The boost is fitted on **train only** at seeds
42–46 (averaged within each fold) with `[prot ‖ mol]` columns — the alpha sweep's
`boost_full` exactly, so its ESM-1b row equals the main tables' boosting row. (Until
2026-09-19 it fitted the insects on train+val with the fold number as seed, which put
its insect numbers 0.007–0.051 R² above the same boost everywhere else.)

Protein sources and where they come from:

| source | files | how |
|---|---|---|
| ESM-1b | `esm1b_650m_mean*.npz`, `esm1b_650m_per_residue_*.npz` | **imported**, not computed: M2OR mean from LoRaX, per-residue from the OFM zenodo release (`06_import_ofm_esm1b.py`) |
| ESM-2 | `esm2_650m_*` | `05_per_residue_embeddings.py` |
| ProtT5 | `prott5_{ds}.npz` | `embed_proteins_plm.py --model prott5` |
| ESM3 | `esm3_{ds}.npz`, `esm3_per_residue_{ds}.npz` | `embed_proteins_plm.py --model esm3 --dataset all --per-residue` in `.venv-esm` |

ESM3 is `esm3-sm-open-v1` (1.4B, the only open-weight ESM3; MIT licence), fed the
sequence track alone. Any of these drops in by file name: `prot=esm:<npz>:<label>`
(the third field is provenance only), `--prot-embeddings …/esm3_{ds}.npz` for the
sweep, the per-residue file for LORAX/ProSmith/MolOR's `cls=` spec.

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

Which refinement graph is a flag: `--variant q99greedy` (M2OR's default -- the q99 + greedy
pair cover the rest of the M2OR paper uses) or `--variant q0cov` (the insects' default -- full
coverage, no quantile cut stacked on the class removal). A non-legacy variant writes to
`<dataset>__<variant>/`, so the two coexist and the notebook's `VARIANT` flag selects one.

`CUDA_VISIBLE_DEVICES` rather than `--device cuda:N`: it also pins whatever the boosting head
and the extractor pick up on their own. A fourth GPU has nothing to do here -- M2OR is the long
pole and stays one process. Serially, on one GPU, it is the same command with `--dataset all`.

Three post-hoc passes bring an older run up to date without retraining anything, all reading
its own `embeddings.npz`: `--derive` adds the two derived representations and the per-model
nulls (and runs automatically after a fresh run), `--backfill` adds the isolation controls
and null spreads to `nulls.csv`, and `--rescore-ood` refits the boosting head for BOTH target
series and rewrites `ood.csv`:

```sh
.venv/bin/python scripts/modeling/analysis/mechanism_holdout.py --dataset all --backfill --derive --rescore-ood
```

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

## The runbook: what to run, in order

The sections above explain *why* each run exists. This one is what to type. It has
two stages and the order between them is fixed: **stage 1 computes and names things**,
stage 2 turns those names into tables. Nothing in stage 2 fits a model except where it
says so; nothing in stage 1 needs a table to exist.

Every command runs from the repo root. Which interpreter matters: `.venv-controls`
for LORAX/ProSmith, `.venv-molor` for MolOR, `.venv` for Hladiš, the graph, the
fitters and every reader.

**The three names you choose, and what they mean.**

| name | chosen by | what it is | ours |
|---|---|---|---|
| `<run>` | `--out results/graph/<run>` (1.1) | one sweep grid = one protein source + one set of dial flags + one encoder regime | `v9_seeded` (ESM-1b), `v14_esm3_paper` (ESM3) |
| `<pool>` | `--out-dir results/ensemble_logs*/<pool>` (1.2) | one (dataset, regime, molecule source, protein source) of baselines | `m2or-transductive-chemberta-esm3`, `cc-ourind-esm3`, … |
| `--out` | every reader | where a rendered table lands | `results/article_tables/esm3/…` |

A third root, `v13_esm3`, is the same ESM3 grid trained **before 23.09.2026**, when
neighbour sampling and per-layer normalisation were off. It is kept for provenance and
must not be mixed with `v14_esm3_paper` in one table -- the two are different models
under one name. `--fanout 0 0 --no-normalize-layers` reproduces it.

`<run>` and the ensemble root must be a **matched pair**: `v14_esm3_paper` goes with
`results/ensemble_logs_esm3`, `v9_seeded` with `results/ensemble_logs`. Crossing them
does not fail — it prints a table that compares a graph on one protein source against
baselines on another.

---

### Stage 1 — the producers

#### 1.1 The sweep — our graph and the boosting base

The run name is the thing every later command points back at, and every command writing
into one root must carry the **same** dial flags or its cells are not comparable:
`--dial nodes` is the v9 parameterisation (alpha moves the graph's *input*), and
`--seed-graph` ties the initialisation to the row's seed. Turning either on or off makes
a new series, not more folds of an old one.

Two commands, in this order. The first is what five artifacts wait on; the second is
small and feeds only A6, so it can be left until just before that table.

```bash
# (a) THE DIAL, chemberta. Long: 11 positions x 6 cells x 5 folds x 5 seeds of graph.
#     Feeds M1 and A1 (the alpha=1 end), A3.1 (every position, incl. the val
#     files), A3.2 (the alpha=0 end). This is the command currently defining the root.
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source chemberta \
    --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/v14_esm3_paper \
    --max-parallel 4 --gpus 0 1 2 3

# (b) THE OTHER MOLECULE SOURCES, one position each. Feeds A6 and nothing else.
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source gin ecfp --alphas 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/v14_esm3_paper \
    --max-parallel 4 --gpus 0 1 2 3
```

Both go into ONE root and cannot collide: the molecule source is part of each cell's
filename, the dial position is part of the cell key inside it. (a) therefore subsumes any
earlier `--alphas 1.0` chemberta run — those rows count as done.

Both train the encoder in **GraphSAGE's own regime** — neighbour sampling (fan-out 25
then 10, redrawn every epoch, inference still full-neighbourhood) and per-layer L2
normalisation, the default since 23.09.2026. `--fanout 0 0 --no-normalize-layers` gives
the historical encoder back. **Consequence for the ESM-1b root `v9_seeded`:** re-running
either command against it tops it up with rows from a different model than the ones
already there. Give the new regime its own `--out`, or pass those two flags.

Quote `'…{ds}.npz'`: the placeholder belongs to the script and an unquoted brace belongs
to the shell. ESM3's files are named uniformly (`esm3_m2or.npz`, `esm3_cc.npz`,
`esm3_hc.npz`), so one template covers all three.

The sweep is resumable — it reads what is already in the CSV and fits only the missing
heads — so the commands above are also the commands that top a series up, and the dense
grid extends the same root rather than needing its own.

One run writes three files per cell: `metrics_*.csv` (TEST), `val_metrics_*.csv` (the
same fitted head scored on VALIDATION — the only split alpha may be chosen on) and
`records_*.csv` (everything, with wall clock and provenance). Runs made before Sep 2026
have no val file; see 1.4.

To check that an existing root was produced the way the tables assume, read the flags
back out of the CSV rather than trusting shell history:

```bash
.venv/bin/python scripts/analysis/sweep_provenance.py --root results/graph/v14_esm3_paper
```

#### 1.2 The baselines — LORAX, ProSmith, MolOR, Hladiš

Four methods × 3 datasets × 2 regimes, one pool directory per cell. The ESM-1b M2OR
form is under **T1 / T1b** above; this is its ESM3 counterpart.

```bash
PRES3=data/embeddings/proteins/esm3_per_residue_m2or.npz
PROT3=data/embeddings/proteins/esm3_m2or.npz
MOL=data/embeddings/molecules/chemberta_77m_m2or.npz

.venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime full_full --full-full-mode transductive \
    --out-dir results/ensemble_logs_esm3/m2or-transductive-chemberta-esm3 \
    --run-name transductive_lorax_esm3 \
    --source cls=lorax:${PRES3} \
    --source prot=esm:${PROT3}:esm3-sm-open-v1 \
    --source mol=gin:${MOL}:chemberta_77m \
    --combos "1 123" --on-missing drop \
    --max-parallel 2 --gpus 0 1 --repeats 1 2 3 4 5
```

For cold molecule: `--full-full-mode inductive_molecule_v5 --repeats 42 43 44 45 46`
and the `m2or-inductive-chemberta-esm3` pool.

Two things that silently break a run rather than failing it:

* **`--out-dir` is not optional.** The readers glob `<root>/<pool>/<run>/config.json`
  at exactly that depth; a run written to the default root is invisible to them.
* **`cls=` goes first.** Combos are named after the `--source` order, so a reordered
  command line produces `prot+mol+cls` where the reader is looking for
  `cls+prot+mol`, and the row is simply not found.

The other three `cls=` specs, with `{ds}` = `m2or`, `cc` or `hc`:

```
cls=prosmith:esm3_per_residue_{ds}.npz:chemberta_77m_{ds}.npz::<prosmith .pkl>
cls=molor:esm3_per_residue_{ds}.npz:1
cls=hladis:esm3_{ds}.npz
```

Hladiš takes the **mean** file — it has no per-residue path, and its molecule side is
built from SMILES, so it has no molecule npz either.

Its budget is counted in **optimizer steps**, and since 24.09.2026 every ESM3 run takes
upstream's defaults — 1 model, `max_steps=10000`, `warmup_steps=6000`, `eval_every=500` —
on all three panels. Hence the bare spec above: no budget fields anywhere.

That is deliberately generous on the insects. The defaults are sized for M2OR's 41k train
rows, about 24 epochs at batch 100; the same step count on CC (5500 rows) is ~180 epochs
and on HC (2640) ~380. We accept the overshoot rather than calibrate per panel, because
one spec across panels is worth more here than a tuned step count: what the extra steps
buy Hladiš is compute, not an unfair advantage — the weights kept are the best of 20
validation checkpoints (`eval_every` divides the budget into twenty either way), so a
longer run cannot score worse than a shorter one by overtraining past its own optimum.
Reading it the other way round: a competitor given more training than it needs is a
*conservative* comparison for us.

The three numbers are coupled and must not be changed one at a time. The LR is
`init·min(step^-0.5, step·warmup^-1.5)`, so `warmup_steps` fixes where the peak falls and
a `max_steps` below it never leaves the ramp.

**The earlier ESM3 insect runs used a rescaled `1:2000:1200:100` and are superseded** —
see 1.3 for which ones and what retyping that costs. The ESM-1b section above keeps the
rescaled spec because that is what those runs actually did; it is a record of the old
series, not a recipe.

Insects: `--regime ofm --dataset {cc,hc} --split-family {rand,our_inductive}
--task regression --combos "1 12 123" --repeats 1 2 3 4 5`, pools
`{cc,hc}-{rand,ourind}-esm3`. The insect protein file must be named explicitly — a
bare `cls=lorax` loads its M2OR default, which holds none of these receptors, and the
run dies with `num_samples=0`.

#### 1.3 Hladiš on the other molecule sources

Sweep 1.1 already covers the graph and the base on all three molecule sources. Only
Hladiš needs extra runs, because its row in the molecule ablation (A6) moves with the
*boost's* molecular half rather than with its own input:

```bash
for DS in cc hc; do
  for FAM in rand our_inductive; do
    if [[ ${FAM} == rand ]]; then POOL=${DS}-rand-esm3; TAG=${DS}_rand
    else POOL=${DS}-ourind-esm3; TAG=${DS}_our_inductive; fi
    for MOL in gin ecfp; do
      if [[ ${MOL} == gin ]]; then MF=data/embeddings/molecules/gin_supervised_contextpred_${DS}.npz
      else MF=data/embeddings/molecules/ecfp_${DS}.npz; fi
      .venv/bin/python scripts/modeling/train/train_ensemble_boost.py \
        --regime ofm --dataset ${DS} --split-family ${FAM} --task regression \
        --out-dir results/ensemble_logs_esm3/${POOL} \
        --run-name ${TAG}_hladis_esm3_${MOL} \
        --source cls=hladis:data/embeddings/proteins/esm3_${DS}.npz \
        --source prot=esm:data/embeddings/proteins/esm3_${DS}.npz:esm3-sm-open-v1 \
        --source mol=gin:${MF}:${MOL} \
        --combos "1 12 123" --on-missing drop \
        --max-parallel 2 --gpus 0 1 --repeats 1 2 3 4 5
    done
  done
done
```

And the same on M2OR, where the bare spec is the budget of record — these four runs did
not exist before 24.09.2026:

```bash
P3=data/embeddings/proteins/esm3_m2or.npz

for MOL in gin ecfp; do
  case ${MOL} in
    gin)  MF=data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz ;;
    ecfp) MF=data/embeddings/molecules/ecfp_m2or.npz ;;
  esac
  for MODE in transductive inductive_molecule_v5; do
    if [[ ${MODE} == transductive ]]; then
      POOL=m2or-transductive-${MOL}-esm3; TAG=transductive; REP=(1 2 3 4 5)
    else
      POOL=m2or-inductive-${MOL}-esm3;    TAG=inductive;    REP=(42 43 44 45 46)
    fi
    .venv/bin/python scripts/modeling/train/train_ensemble_boost.py \
        --regime full_full --full-full-mode ${MODE} \
        --out-dir results/ensemble_logs_esm3/${POOL} \
        --run-name ${TAG}_hladis_esm3_${MOL} \
        --source cls=hladis:${P3} \
        --source prot=esm:${P3}:esm3-sm-open-v1 \
        --source mol=gin:${MF}:${MOL} \
        --combos "1 123" --on-missing drop \
        --max-parallel 2 --gpus 0 1 --repeats ${REP}
  done
done
```

M2OR's GIN file is the one special case in the project: `gin_supervised_contextpred_all_m2or.npz`,
without the `_{ds}` suffix the other panels use.

`mol=gin:` is the generic entity extractor, not the GIN model: the file decides what
the embedding is and the third field is provenance only.

**The spec is bare on purpose** — one budget on every panel and every molecule source, as
1.2 explains. Two consequences for what is already on disk:

* the eight gin/ecfp insect runs (`{cc,hc}_{rand,our_inductive}_hladis_esm3_{gin,ecfp}`)
  were made with a bare spec and therefore already satisfy this. Nothing to redo;
* the four **ChemBERTa** insect runs (`{cc,hc}_{rand,our_inductive}_hladis_esm3`) were made
  with `1:2000:1200:100` and no longer match. They have to be refitted.

Refitting them is not a plain rerun: the checkpoints are named
`hladis_{name}_model{m}.pt` with **no budget in the name**, so a rerun that finds them
skips training entirely and reloads the old weights. Delete the four run directories
first, then refit them at ChemBERTa:

```bash
for DS in cc hc; do
  for FAM in rand our_inductive; do
    if [[ ${FAM} == rand ]]; then POOL=${DS}-rand-esm3; TAG=${DS}_rand
    else POOL=${DS}-ourind-esm3; TAG=${DS}_our_inductive; fi
    rm -rf results/ensemble_logs_esm3/${POOL}/${TAG}_hladis_esm3
    .venv/bin/python scripts/modeling/train/train_ensemble_boost.py \
      --regime ofm --dataset ${DS} --split-family ${FAM} --task regression \
      --out-dir results/ensemble_logs_esm3/${POOL} \
      --run-name ${TAG}_hladis_esm3 \
      --source cls=hladis:data/embeddings/proteins/esm3_${DS}.npz \
      --source prot=esm:data/embeddings/proteins/esm3_${DS}.npz:esm3-sm-open-v1 \
      --source mol=gin:data/embeddings/molecules/chemberta_77m_${DS}.npz:chemberta_77m \
      --combos "1 12 123" --on-missing drop \
      --max-parallel 2 --gpus 0 1 --repeats 1 2 3 4 5
  done
done
```

**These four feed M1, not only A6.** The Hladiš row of `tab:esm3cc` and `tab:esm3hc` comes
from exactly these runs, so after refitting, both insect M1 tables have to be regenerated
and retyped — and not just that one row: the rank column and the bold marks are computed
across the rows of the table, so every row's rank moves when Hladiš's values do. M2OR's
M1 table and A1 are untouched (M2OR was always on the defaults).

`transductive_hladis_esm3` on M2OR spells the same defaults out and adds a sixth field,
`report_own_head=1`, which only prints Hladiš's own scalar head on test and changes no
number in any table — it does not need refitting.

#### 1.4 The fitters that belong to one table each

These do not feed the main tables and do not read `<run>`; each is the compute half of
one supplementary artefact, and each is listed again beside its reader in stage 2. They
can run while 1.1 is still going.

```bash
# for A2 (tab:t4): fits every row of the protein-representation table, ours included
.venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --gnn esm3@1 esm3@0 prott5@1 --seeds 42 43 44 45 46

# for A3.2 (tab:alpha0): the one-hot boosting heads. Pass the SAME protein npz the
# sweep used -- it decides the coverage mask even though one-hot replaces ESM
.venv/bin/python scripts/article_tables/s3_onehot_boost.py \
    --dataset m2or cc hc \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz'

# for A4: the construction sweep, criterion x quantile. M2OR only (decided 25.09)
.venv/bin/python scripts/article_sweeps/s4_run_quantile_criteria.py \
    --dataset m2or --regime inductive transductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seed-graph --seeds 42 43 --max-parallel 4 --gpus 0 1 2 3

# for A5: the architecture sweep. The graph is PINNED per dataset at the paper's
# construction -- this moves the operator and nothing else
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seeds 42 43 44 45 46 --seed-graph --max-parallel 4 --gpus 0 1 2 3

# for A3.1, ONLY for a root made before Sep 2026: score that root's validation split.
# The head is refit on the same train rows with the same seed and asked for the val
# rows instead -- one XGBoost fit per cell, no message passing, no GPU, resumable,
# and self-checking (it re-predicts test and compares against the recorded number)
.venv/bin/python scripts/analysis/val_rescore.py --root results/graph/v14_esm3_paper
```

#### 1.5 Before reading anything

Inventory reports READY / PARTIAL / MISSING per input cell, which is the difference
between a table that is complete and one that merely printed:

```bash
.venv/bin/python scripts/article_tables/inventory.py \
    --sweep-root results/graph/v14_esm3_paper --ensemble-root results/ensemble_logs_esm3
```

---

### Stage 2 — the tables, in the order they stand in the paper

The order below is the registry's order (`paper/PLAN.md`): the main table first, then
the supplementary ablations as the argument needs them.

| # | artefact | reader | needs |
|---|---|---|---|
| M1 | main battery | `m1_main_tables.py` | 1.1 + 1.2 |
| A1 | baselines as `cls` vs the boosting base | `m1_main_tables.py --baseline-combo cls --no-ours` | 1.1 + 1.2 |
| A2 | protein representations + our rows (`tab:t4`) | `s2_protein_sources.py` | 1.4 (`prot_floor_sweep`) |
| A3.1 | the dial as advantage over the base, 3x2 battery | `notebooks/article_figures/prediction_dial.ipynb` | 1.1, dense grid + its `val_metrics_*` if `SPLIT="val"` |
| A3.2 | identity control (`tab:alpha0`) | `s3_alpha0_vs_boost.py` | 1.1 + 1.4 (`s3_onehot_boost`) |
| A4 | criterion × quantile, + the random control | `notebooks/article_figures/quantile_criteria.ipynb` | 1.4 (`run_quantile_criteria`) |
| A5 | architecture: which operator | `s5_architecture.py` | 1.4 (`run_architecture`) |
| A6 | molecule ablation | `s6_molecule_ablation.py` | 1.1 (all three `--mol-source`) + 1.3 |

#### M1 — the main battery

```bash
# one table per dataset, both regimes, one run per protein source
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --sweep-root results/graph/v9_seeded --ensemble-root results/ensemble_logs \
    --out results/article_tables/esm1b
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --sweep-root results/graph/v14_esm3_paper --ensemble-root results/ensemble_logs_esm3 \
    --out results/article_tables/esm3
```

`--no-val-cut` drops the extra column whose threshold is chosen on validation and
leaves the 0.5 cut alone. Give each protein source its own `--out`, or the second run
overwrites the first and there is nothing left to compare.

#### A1 — every competitor in its `cls` form, against the boosting base

The same reader, told to take each baseline in its own learned pair representation and
to leave our graph out entirely:

```bash
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --dataset m2or --baseline-combo cls --no-ours \
    --sweep-root results/graph/v14_esm3_paper --ensemble-root results/ensemble_logs_esm3 \
    --out results/article_tables/esm3/t1
```

#### A2 — the protein-representation table (`tab:t4`)

Two commands: the fitter from 1.4, then the reader. `--dataset` and `--regime` take
lists, so the whole six-cell table is one invocation of each.

```bash
# fits: classical amino-acid descriptors, the one-hot controls, each pLM whose npz
# covers the pool, and -- with --gnn -- our graph, boosted as
# [refined receptor || ChemBERTa], which is our cls+mol.
# Five seeds, as everywhere: one seed initialises the graph AND seeds the head,
# so a row is (fold, seed) exactly as in 1.1. There is no separate graph-seed axis.
.venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --gnn esm3@1 esm3@0 prott5@1 --seeds 42 43 44 45 46

# renders: one combined table, a column per cell, the metric of record only
.venv/bin/python scripts/article_tables/s2_protein_sources.py \
    --dataset m2or cc hc --regime transductive inductive
```

**Swapping only the graph rows.** Our three rows train in the sampled + normalised
regime (the default since 23.09.2026) and each one stamps `gnn_regime` into the CSV. A
resume that would mix two regimes in one table is refused by name. To replace the graph
rows of a table already on disk while keeping every descriptor row -- no pLM boosting is
refitted, and those are most of the table:

```bash
for f in results/tables/prot_floor_*.csv(N); do
  python - "${f}" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
d = pd.read_csv(p)
keep = d["gnn_seed"].isna() if "gnn_seed" in d.columns else d.index == d.index
print(f"{p}: {len(d)} rows -> {int(keep.sum())} kept")
d[keep].to_csv(p, index=False)
PY
done
```

`s2_protein_sources.py` also needs `--root results/tables` only if you moved the
fitter's output; the default is that path.

**Keeping what is still valid.** The resume key is `(representation, fold, seed,
gnn_seed)` and `gnn_seed == seed` now, so a graph row from a run that predates the
single-seed rule is reusable exactly when it sits on that diagonal. This keeps those and
drops the rest, including anything whose regime stamp is not the current one:

```bash
for f in results/tables/prot_floor_*.csv(N); do
  python - "${f}" <<'PY'
import sys, pandas as pd
p = sys.argv[1]
d = pd.read_csv(p)
if "gnn_seed" not in d.columns:
    print(f"{p}: no graph rows, left alone"); raise SystemExit
g = d["gnn_seed"].notna()
tag = d.get("gnn_regime", pd.Series(index=d.index, dtype=object)) == "sampled25-10+norm"
keep = (~g) | (g & (d["gnn_seed"] == d["seed"]) & tag)
print(f"{p}: {len(d)} -> {int(keep.sum())}   graph {int(g.sum())} -> "
      f"{int((keep & g).sum())}")
d[keep].to_csv(p, index=False)
PY
done
```

then re-run the fitter above; it refits the graphs alone. `--gnn-fanout 0 0
--gnn-no-normalize-layers` trains the historical encoder instead.

**What comes out.** The fitter writes one CSV per (dataset, regime) under
`results/tables/`, plus a provenance sidecar `prot_floor_<ds>_<regime>.json`. The reader
writes `results/article_tables/protein_sources/`: `protein_long.csv` (every metric,
every row), `protein_sources.tex` (the combined table) and the same table as text.
`--which headline|all` widens the rendered metrics; the long CSV always holds them all.

**Our rows are fitted here, not imported from the sweep.** The refined receptor vector
is trained on its fold's training pairs, so a vector lifted from another run's folds is
a leak, not a cached feature. `name@alpha` names the row: `esm3@1` is the plain graph on
ESM3 nodes, `esm3@0` is the v9 node dial at zero — receptor identity alone, so there the
protein file only decides the coverage mask. The edge variant follows the dataset
(M2OR's hub core, the insects' complete matrix).

**Resuming.** Rows already in the CSV are kept and only the missing ones are fitted, so
adding `--gnn` to a cell that already has its descriptor and pLM rows costs the graphs
and nothing else. What may be reused is decided by the sidecar; a CSV without one is
refused by default, since it may predate the fix that stopped this script folding the
insects' validation rows into train. `--trust-existing` accepts such a file, `--force`
refits everything. A cell a dataset cannot do (HC ships no `cold_receptor`) is skipped
with a note.

#### A3.1 — the dial as advantage over the base (one figure)

A 3×2 battery: columns are datasets, rows are regimes. Two curves per panel, each the
**paired difference of one boosting head against the base**, computed fold by fold on the
same held-out rows and then averaged over folds — `cls+mol`, where the refined receptor
replaces the raw protein vector, and `cls+prot+mol`, where it is added beside it. The base
is the zero line and is deliberately not a curve: every point is already measured against
it, so drawing it would be drawing zero twice.

```bash
jupyter lab notebooks/article_figures/prediction_dial.ipynb
```

Each curve carries a least-squares line, and the box inside each panel gives the
zero-slope test for both. **The slope is fitted per fold**, and the five slopes are the
sample the t-test is over — a slope fitted on the (fold, seed) rows would count five seeds
that share a held-out set as five observations. The per-fold advantages come from
`alpha_grid.delta_folds`, which shares its pairing with `delta_vs`, so the band and the
slope cannot rest on different reductions.

The question is not which $\alpha$ wins — an argmax over a flat surface is noise — but
whether the surface is flat at all. `SPLIT` defaults to `test`, which is honest here
precisely because nothing is chosen on this page: $\alpha = 1$ is reported for what it
means, not for being the argmax. Set `SPLIT = "val"` to see the same panels on the split a
choice would have to be made on; a root made before Sep 2026 gets validation rows from
`val_rescore.py` (1.4).

This figure replaced two (levels along the dial, and mean rank across cells) on
24.09.2026: they asked one question in two idioms. `alpha_rank_dial.ipynb` is still in the
tree and still runs, but it is not part of the paper.

`MOL_SOURCE` stays `chemberta`: the dense grid exists only there, and on the other
molecule sources only $\alpha \in \{0, 1\}$ was run.

#### A3.2 — the identity control (`tab:alpha0`)

Our graph with the receptor's sequence removed, against the boost over ESM and over a
one-hot receptor. The `s3_onehot_boost` half from 1.4 fits heads — minutes per fold on M2OR — and
caches; `05` only reads, so it is safe to re-run while tweaking a label.

```bash
.venv/bin/python scripts/article_tables/s3_alpha0_vs_boost.py \
    --sweep-root results/graph/v14_esm3_paper
```

#### A4 — the construction ablation, criterion × quantile

A different knob from the dial: not what the receptor vector is mixed from, but which
molecules carry the messages at all. `scripts/article_sweeps/` owns it, imports the
alpha sweep as a module for the folds and the metric battery, and touches none of it.

**This is the whole construction ablation, and there is no second one.** Our edges
*are* the measured pairs, so the only knob over that graph is which molecules may carry
messages: the full graph is `q = 0`, a point in this grid, and an IDF-weighted
construction is the `idf_coverage` / `composite` criterion, a curve in it. The grid
also carries `random` — K molecules drawn uniformly from the same eligible set, at the
same K — which is the control the criteria are read against: if a ranked criterion does
not beat a random draw, what the graph buys is message passing and not the choice of
hubs. Its draw follows the cell's seed, so what the figure shows is its spread and not
one lucky set.

**Scope, decided 25.09: M2OR only, both regimes, two seeds.** The insects are left
out on purpose. Their matrices are complete, so a coverage quantile cuts nothing there
and the criterion axis collapses to tie-breaks; what is left, `--k-mode fraction`, is a
different knob from the one M2OR is reported on, so the two would not belong on one
figure. M2OR is also the only dataset whose reported construction is non-trivial
(`greedy_pair_cover` at `q = 0.99`) and therefore the only one that owes a defence.

```bash
# the producer. Quantiles are FRACTIONS, and the cell the paper reports
# (greedy_pair_cover at 0.99) must be in the grid or there is nothing to compare against
.venv/bin/python scripts/article_sweeps/s4_run_quantile_criteria.py \
    --dataset m2or --regime inductive transductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seed-graph --seeds 42 43 --max-parallel 4 --gpus 0 1 2 3

# the figures. This sweep is read by eye and there is no text reader for it:
# quantile_grid.py is the aggregation layer the notebook imports, not a command
jupyter lab notebooks/article_figures/quantile_criteria.ipynb
```

That is 8 criteria x 6 quantiles x 5 folds x 2 seeds x 2 regimes, minus the `q = 0`
column: nothing is cut there, so all eight criteria keep the same molecules and train
the same graph. The sweep computes that cell once per (fold, seed) and writes the other
seven rows from it with `shared_from` naming the fit that ran — one experiment drawn
through eight curves rather than eight identical fits. That is 7 x 5 x 2 x 2 = 140 fits
not done: **820 graphs**, not 960, plus 20 boosting reference fits (one per fold, seed
and regime), which are cheap.

**The quantile is chosen on VALIDATION.** Reading the best `q` off these curves and then
defending it with the same curves takes the number and its defence from one set of rows,
so the notebook's `SPLIT` defaults to `"val"`; `test` is read once, at the end, for the
cell already picked. The claim the figure is meant to support is the weak one — that
over the range where `sep < 1` the construction does not matter, and the cell we report
is not behind the best one by more than the grid can resolve. The ablation's job is to
show the choice was not load-bearing, not to win a fourth decimal.

`--seed-graph` for the same reason the dial passes it: without a seeded graph
initialisation two cells differ by their init as well as by their construction, and the
knob's own effect is the smaller of the two. The `random` control is unaffected -- it
draws its hubs from `--seeds`, not from the init. Everything else is left at the
default, and the defaults here are the alpha sweep's own (`--fanout 25 10`, layer norm
on, `--n-models 1`, `--epochs 900`, chemberta), so this runs in the paper's
`v14_esm3_paper` encoder regime.

The folds are the paper's own: M2OR's splits come from
`data/processed/full_full_split_indices.npz` through `orbind.regimes.load_split`, the
same store M1, A1 and the dial read, so a row here is on the same held-out set as a row
there — including its `val` third.

**Before the long run, a smoke test.** One fold, two quantiles, two criteria — a few
minutes, and its rows count toward the full run, which writes into the same file:

```bash
.venv/bin/python scripts/article_sweeps/s4_run_quantile_criteria.py     --dataset m2or --regime inductive     --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz'     --seed-graph --seeds 42 --n-folds 1     --quantiles 0 0.99 --criteria coverage random --max-parallel 2 --gpus 0 1
```

**`--max-parallel` above the number of cards is allowed.** Workers are assigned
round-robin (`gpus[i % len(gpus)]`), so `--max-parallel 8 --gpus 0 1 2 3` puts two on
each card. It usually does help: a 900-epoch GNN on a graph this small is bound by
kernel-launch latency rather than by arithmetic, which is exactly why the card reads as
under-used. Two things bound it, and both are worth a look before committing 820 cells
to it:

* **Host RAM, which is the real limit.** Each worker is a separate process that caches
  `_fold_prep` for every fold it touches, and on M2OR with ESM3 that is roughly half a
  gigabyte per fold (the dense `Xp`/`Xm` blocks for train, val and test), so a worker
  that has seen all five holds ~2.5 GB, plus its CUDA context. Eight of those is ~25 GB.
  Check with `free -g` while it runs.
* **Per-card memory**, since XGBoost trains on the device too (`device="cuda"`,
  `tree_method="hist"`) in the same process as the graph. Two workers per card double
  it. `nvidia-smi` during the smoke test answers this in one line.

**Adding a seed later costs only the new seed.** The seed is part of the cell key, so
repeating the command with `--seeds 42 43 44` trains 205 graphs per regime and touches
nothing that is already there; the `q = 0` column is shared for the new seed exactly as
for the old ones. The sharing does not depend on which criteria a later run asks for
either: the source of a copy is any `q <= 0` row that was actually fitted (`shared_from`
empty), not a fixed criterion name, so a run with a shorter `--criteria` list cannot
refit that cell under a second name.

Two things to know about a re-run. `config_*.json` records the LAST invocation, not the
union of them, which is the same convention every producer here follows — the CSV is the
record of what was computed. And a seed that is added but not finished leaves cells
averaging different numbers of draws, which moves a curve by the imbalance rather than
by the knob: the notebook's second guard prints `qg.seed_balance` and says so, and
`SEEDS` cuts back to the seeds that are complete.

Resumable: re-running the same command continues it, and a resume also fills the shared
`q = 0` rows if an older run left them out. See `scripts/article_sweeps/README.md` for
what `--k-mode` changes and why a failed cell is written down rather than dropped.

#### A5 — the architecture table (`tab:arch`)

One row per message-passing operator, one column per (dataset, regime), each column that
panel's metric of record — six numbers per row, which is the whole table. The boosting
base is the anchor row, because "our graph against the base" is the comparison every
other table here makes.

```bash
# trains: the four operators, five folds, five seeds, on the pinned graph
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --max-parallel 4 --gpus 0 1 2 3

# renders: one combined table, six columns, the metric of record only
.venv/bin/python scripts/article_tables/s5_architecture.py \
    --dataset m2or cc hc --regime transductive inductive
```

**The graph does not move when the operator does.** The construction is pinned per
dataset in `VARIANT` (M2OR's hub core, the insects' complete matrix) and is deliberately
not a command-line flag: an operator comparison run on two different graphs is not one.
Everything else is held too — the signed two-stack structure, the subtraction, the
decoder, the epoch budget, the folds, the head, and since 23.09.2026 the encoder regime.

**The four operators** (`orbind.gnn_extractor.CONVS`): `sage` is ours; `gat` is the
attention epoch of this project, ported from `orbind/legacy/hetero_gat.py` with the two
details that make attention work on a bipartite graph (`add_self_loops=False`,
multi-head concat on layer 1 and a single head on layer 2); `graphconv` is the GCN-shaped
operator that is actually defined on two node sets — plain `GCNConv` is not here and
cannot be, since its symmetric normalisation and mandatory self-loops assume one node
set; `gin` is the molecular domain's standard and the most expressive of the four.

So the table is five rows: four operators and the boosting anchor.

**Why `DEFAULT_SPECS` carries a `:paper` suffix on all four.** That suffix names the
encoder regime on disk — neighbour sampling plus per-layer L2 normalisation — and it is
now the only regime the table reports, so the reader prints a suffixed row as the plain
operator name. It is kept rather than removed because every cell already computed carries
it in its `arch` column, and renaming would make each of them look missing. The
un-suffixed specs still train the historical encoder and still render, marked
`(full neighbourhood, un-normalised)`; they are what measured the regime in the first
place and are worth keeping on disk, but they are not part of the table. `--historical`
does not exist: pass the un-suffixed spec if you want those rows back and read them
knowing the label.

**The randomness is M1's, by construction.** One seed per row: it initialises the graph
AND seeds the boosting head, exactly as `--seed-graph` does in the main sweep, and the
reader averages seeds inside each fold before taking any interval. `--seeds` defaults to
`42 43 44 45 46` and graph seeding is ON by default — `--no-seed-graph` turns it off, and
a row produced that way is a different experiment from the main tables. `prot_mix=1.0` is
passed explicitly for the same reason: the main sweep always passes it, and rho=1
short-circuits to the embedding file, so the two scripts make the same call and not
merely the same model.

**Width is free, depth is not.** `--hidden 128 256 512` adds one row per width with no
new code. Depth is not a knob: the encoder is two layers by construction, and making that
variable is a refactor of the module rather than a flag — it is out of this sweep
deliberately.

The reader marks the best operator per column, never the anchor, and names in words any
column where our interval overlaps the marked one.

#### A6 — the molecule ablation

ChemBERTa / GIN / ECFP × {graph, base, Hladiš}:

```bash
.venv/bin/python scripts/article_tables/s6_molecule_ablation.py \
    --sweep-root results/graph/v14_esm3_paper --ensemble-root results/ensemble_logs_esm3 \
    --out results/article_tables/esm3/molecule
```

#### Beside the tables

Two runs side by side — value, place, and what moved between two `main_long.csv`:

```bash
.venv/bin/python scripts/legacy/04_compare_runs.py \
    --a results/article_tables/esm1b --a-label ESM-1b \
    --b results/article_tables/esm3  --b-label ESM3
```

The geometry pair is **not in the paper** (the whole geometry line is parked), but it
still runs, and `02a` skips a CSV that already exists — `--force` is what adds a
protein source generated after those files were written:

```bash
.venv/bin/python scripts/legacy/02a_protein_geometry.py --dataset cc hc --force
.venv/bin/python scripts/legacy/02_geometry_table.py \
    --sweep-root results/graph/v14_esm3_paper --out results/article_tables/esm3/geometry
```

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
