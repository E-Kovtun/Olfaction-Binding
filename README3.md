# OlfaGraph: reproducing the paper

This document takes you from an empty checkout to every table and figure of the paper.
It is organised as a two-stage pipeline:

1. **Producers** train models and write *artifacts* — directories of per-split results.
   They are slow (most of them want a GPU), resumable and never render anything.
2. **Readers** turn artifacts into the paper's tables and figures. They fit nothing, so
   they are cheap to re-run.

Every artifact in the paper is one reader applied to a fixed subset of the producers'
output. Section 5 lists those subsets. Each producer below is given as the **one
invocation** that yields exactly what the paper needs, even where the script accepts many
more flags.

| Paper artifact | Label | Reader | Needs producers |
|---|---|---|---|
| Main results (M2OR, Mosquito, Fly; seen and cold molecules) | `tab:main_results` | `m1_main_tables.py` | P1a, P2 |
| Comparison of receptor representations | `tab:protein_representations` | `s2_protein_sources.py` | P3 |
| Message-passing ablations and encoder comparison | `tab:graph_ablation` | `s5_architecture.py` | P5 |
| App. A: interaction baselines with `cls` only | `tab:esm3t1` | `m1_main_tables.py` | P1a, P2 |
| App. B.1: the α dial | `fig:dial` | `prediction_dial.ipynb` | P1a |
| App. B.2: the identity-only end | `tab:alpha0` | `s3_alpha0_vs_boost.py` | P1a, P4 |
| App. C: graph construction | `fig:construction` | `quantile_criteria.ipynb` | P6 |
| App. D: molecular representation | `tab:mol` | `s6_molecule_ablation.py` | P1a, P1b, P2 |

---

## 0. Vocabulary: paper names and code names

The code predates some of the paper's terminology. Flags and CSV columns use the left-hand
column below.

| code | paper |
|---|---|
| `m2or` / `cc` / `hc` | M2OR / Mosquito (Carey et al.) / Fly (Hallem & Carlson) |
| `transductive` | seen molecules |
| `inductive` (M2OR: `inductive_molecule_v5`; insects: split family `our_inductive`) | cold molecules |
| `boost_full`, "boosting base", `prot+mol` | XGBoost-base, $[\mathbf{x}_{\mathrm{prot}}\|\mathbf{x}_{\mathrm{mol}}]$ |
| our `cls+mol` | OlfaGraph, reduced form $[\mathbf{z}_{\mathrm{prot}}\|\mathbf{x}_{\mathrm{mol}}]$ |
| our `cls+prot+mol` | OlfaGraph, full form $[\mathbf{x}_{\mathrm{prot}}\|\mathbf{x}_{\mathrm{mol}}\|\mathbf{z}_{\mathrm{prot}}]$ |
| a baseline's `cls` / `cls+prot+mol` | its interaction representation alone / with both original embeddings |
| `--dial nodes`, `alpha` | the α dial of Appendix B (α=1: ESM3 input, α=0: receptor identity only) |
| `greedy_pair_cover`, `q=0.99` | the odorant selection of Section 3.2 / Appendix C |

Every model in the paper, OlfaGraph and the baselines alike, is read out by the same
XGBoost head with fixed hyperparameters (400 trees, depth 6, learning rate 0.1, row and
column subsampling 0.8), fitted on the training split only.

---

## 1. Environments

Five virtual environments, because the baselines' dependencies conflict with each other.

| env | holds | used for |
|---|---|---|
| `.venv` | torch, torch-geometric, xgboost, rdkit, dgl + dgllife (the `gin` extra) | OlfaGraph, XGBoost-base, Hladiš, every fitter and reader; ChemBERTa import, ESM-1b import, GIN and ECFP embeddings |
| `.venv-controls` | torch, transformers, peft, xgboost (no PyG, no rdkit) | LORAX, ProSmith |
| `.venv-molor` | torch 2.4, dgl 2.4 + dgllife, rdkit, xgboost | MolOR |
| `.venv-embeddings` | torch, transformers, sentencepiece | ProtT5 embeddings |
| `.venv-esm` | EvolutionaryScale `esm` SDK | ESM3 embeddings |

```bash
uv python install 3.11
uv sync --frozen --extra gin     # .venv
bash scripts/setup_envs.sh       # the other four
```

`.venv-esm` is separate because the SDK's package is also called `esm` and clashes with
fair-esm. ESM3's weights download from HuggingFace on first use; if that answers 401,
`export HF_TOKEN=<read token>` and rerun.

**Every environment that fits an XGBoost head holds `xgboost>=2.0,<3.0`.** The same head
is what all reported numbers share, and the producers refuse to start on another major.
`setup_envs.sh` pins it everywhere.

Scripts put the repository root on `sys.path` themselves. Run every command from the
repository root and call the interpreter of the environment named in it. A method run in
the wrong environment does not always crash: it can silently load a stale checkpoint
instead of training.

---

## 2. Data

[`data/README.md`](data/README.md) is the manifest of `data/`. The splits are versioned
with the repository; everything else (the benchmark release and the embeddings) is
produced by the commands below.

### 2.1 What is in the repository

The held-out splits every number stands on:

| path | what |
|---|---|
| `data/splits_indexes/lorax_m2or/rand_split_{1..5}/{train,val,test}_df.csv` | M2OR: LORAX's five folds of its 46,563-pair pool (seen molecules). Every fold holds the same rows, so the pool itself is reconstructed from these files |
| `data/processed/full_full_split_indices.npz` | M2OR: the positions of those folds in the pool, and five molecule-disjoint splits, seeds 42–46 (cold molecules) |
| `data/external/ofm/{CC,HC}/rand_splits/` | Mosquito and Fly: the released five folds (seen molecules) |
| `data/external/ofm/{CC,HC}/our_inductive_splits/` | Mosquito and Fly: our five molecule-disjoint folds (cold molecules) |

They can be rebuilt with `scripts/preprocessing/02_build_full_full_split_indices.py` and
`scripts/preprocessing/03_build_ofm_our_inductive_splits.py`; both are deterministic.

### 2.2 Download

All three benchmarks are the versions released with LORAX, in the olfactory
foundation-models data release on Zenodo (<https://zenodo.org/records/17228740>).
Download its archive from that page and unpack it anywhere. It holds two folders:

* `data/`, with one folder per dataset (`CC`, `HC` and M2OR's), each with its response
  table, its splits and the authors' precomputed features under
  `embeddings/featurized_mols/` and `embeddings/featurized_proteins/`;
* `BindingDB/`, whose `saved_model/` holds the BindingDB-pretrained ProSmith checkpoint
  the ProSmith paper uses and we initialise ProSmith from.

Then:

1. Copy the dataset folders from `data/` into `data/external/ofm/`, without overwriting
   anything already there: the splits of `CC` and `HC` are versioned in this repository
   and must stay as they are.
2. Copy `pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl` from `BindingDB/saved_model/`
   into `data/external/ofm/saved_model/`.

The layout the scripts expect afterwards:

```text
data/external/ofm/
  CC/  HC/        raw/, rand_splits/, our_inductive_splits/, embeddings/featurized_{mols,proteins}/
  M2OR/ or M2OR_full/   embeddings/featurized_{mols,proteins}/
  saved_model/    pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl
```

### 2.3 Molecule tables and embeddings

Every embedding is a flat `.npz` of `{key: vector}`. Proteins are keyed by amino-acid
sequence, molecules by InChIKey. No producer computes an embedding during a run.

| file (under `data/embeddings/`) | what | produced by | env | needed by |
|---|---|---|---|---|
| `proteins/esm3_{m2or,cc,hc}.npz` | ESM3 (`esm3-sm-open-v1`), mean over residues, 1536-d | `embed_proteins_plm.py --model esm3 --per-residue` | `.venv-esm` | everything |
| `proteins/esm3_per_residue_{m2or,cc,hc}.npz` | ESM3 per residue | the same command | `.venv-esm` | P2 (LORAX, ProSmith, MolOR) |
| `proteins/prott5_{m2or,cc,hc}.npz` | ProtT5, mean | `embed_proteins_plm.py --model prott5` | `.venv-embeddings` | P3 |
| `proteins/esm1b_650m_mean_{full_full,cc,hc}.npz` | ESM-1b, mean, as released | `06_import_ofm_esm1b.py` | `.venv` | P3 |
| `molecules/chemberta_77m_{m2or,cc,hc}.npz` | ChemBERTa-77M-MTR, as released | `07_prepare_ofm_molecules.py` | `.venv` | everything |
| `molecules/gin_supervised_contextpred_all_m2or.npz`, `…_{cc,hc}.npz` | pretrained GIN (Hu et al.), 300-d | `embed_molecules_gin.py` | `.venv` | P1b, P2 (Hladiš) |
| `molecules/ecfp_{m2or,cc,hc}.npz` | ECFP4, 2048 bits | `embed_molecules_ecfp.py` | `.venv` | P1b, P2 (Hladiš) |

```bash
# molecule tables (SMILES -> InChIKey) and the released ChemBERTa vectors, per dataset.
# M2OR's molecules are read from the LORAX pool itself (596), the insects' from their
# response tables. An existing npz is replaced only if the new vectors are identical.
for DS in m2or cc hc; do
  .venv/bin/python scripts/embedding_generation/molecules/07_prepare_ofm_molecules.py --tag ${DS}
done

# ESM3: mean and per-residue in one pass (the mean is derived from the per-residue file)
.venv-esm/bin/python scripts/embedding_generation/proteins/embed_proteins_plm.py \
    --model esm3 --dataset all --per-residue

# ProtT5
.venv-embeddings/bin/python scripts/embedding_generation/proteins/embed_proteins_plm.py \
    --model prott5 --dataset all

# ESM-1b, imported from the release rather than recomputed
.venv/bin/python scripts/embedding_generation/proteins/06_import_ofm_esm1b.py \
    --prots-pt data/external/ofm/M2OR_full/embeddings/featurized_proteins/prots.pt
for DS in cc hc; do
  UP=$(echo ${DS} | tr a-z A-Z)
  .venv/bin/python scripts/embedding_generation/proteins/06_import_ofm_esm1b.py \
      --prots-pt data/external/ofm/${UP}/embeddings/featurized_proteins/prots.pt \
      --tag ${DS} --pool none --compare-mean ""
done

# ECFP
for DS in m2or cc hc; do
  .venv/bin/python scripts/embedding_generation/molecules/embed_molecules_ecfp.py --dataset ${DS}
done

# GIN. M2OR's file is computed on the 770 molecules of the M2OR export listed in
# molecule_smiles_all_m2or.csv (versioned); it covers 595 of the pool's 596, and the
# pairs of the missing one are dropped in every M2OR run on GIN.
.venv/bin/python scripts/embedding_generation/molecules/embed_molecules_gin.py \
    --molecules data/processed/molecules/molecule_smiles_all_m2or.csv \
    --out data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz
for DS in cc hc; do
  .venv/bin/python scripts/embedding_generation/molecules/embed_molecules_gin.py \
      --molecules data/processed/molecules/molecule_smiles_${DS}.csv \
      --out data/embeddings/molecules/gin_supervised_contextpred_${DS}.npz
done
```

ESM3 and ChemBERTa are the embeddings of record throughout. ProtT5 and ESM-1b appear only
in the receptor-representation table, GIN and ECFP only in Appendix D.

---

## 3. Stage 1: producers

| # | producer | writes | feeds | cost |
|---|---|---|---|---|
| P1a | `run_alpha_gate_sweep.py`, ChemBERTa, the whole α dial | `results/graph/olfagraph/` | main table, App. A, B, D | 11 α × 6 cells × 5 splits × 5 seeds of graph training |
| P1b | the same, GIN and ECFP, α=1 only | `results/graph/olfagraph/` | App. D | 2 × 6 × 5 × 5 graphs |
| P2 | `train_ensemble_boost.py`, the four interaction baselines | `results/baselines/<cell>/<run>/` | main table, App. A, D | 4 methods × 6 cells × 5 splits, plus Hladiš on GIN/ECFP |
| P3 | `prot_floor_sweep.py` | `results/tables/` | receptor-representation table | descriptors are minutes, the three graph rows are 3 × 6 × 5 × 5 graphs |
| P4 | `s3_onehot_boost.py` | `results/article_tables/onehot_boost/` | App. B.2 | XGBoost only, minutes per split on M2OR |
| P5 | `s5_run_architecture.py` | `results/article_sweeps/architecture/` | architecture table | 8 rows × 6 cells × 5 splits × 5 seeds |
| P6 | `s4_run_quantile_criteria.py` | `results/article_sweeps/quantile_criteria/` | App. C | 41 grid cells × 2 settings × 5 splits × 5 seeds = 2,050 graphs, M2OR only |

P1 and P2 are the backbone. P3–P6 each feed a single artifact and can run in parallel
with them. Every producer is **resumable at cell granularity**: re-running the same
command continues it and skips what is already on disk.

Two rules hold for every root:

* **A root is one configuration.** The resume key does not include every flag, so topping
  up a root with different flags silently mixes two models under one name. If you change
  a flag, change the output directory with it.
* **One receptor embedding per pair of roots.** A reader takes OlfaGraph from the sweep
  root and the baselines from the baseline root, and cannot tell if they were trained on
  different protein embeddings.

### P1: OlfaGraph and XGBoost-base

One script produces both OlfaGraph and XGBoost-base, on the same splits and with the same
seeds, so every OlfaGraph row is paired with its reference.

```bash
# P1a: ChemBERTa, the full alpha dial (alpha=1 is the model the paper reports)
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source chemberta \
    --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/olfagraph \
    --max-parallel 4 --gpus 0 1 2 3

# P1b: the other molecular embeddings, at the reported alpha only
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source gin ecfp --alphas 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/olfagraph \
    --max-parallel 4 --gpus 0 1 2 3
```

What the flags fix:

* `--prot-embeddings` is ESM3 by default and is spelled out anyway, because the root
  is only meaningful together with the file it was built on. Quote the template: `{ds}`
  is filled in by the script, not by the shell.
* `--dial nodes` makes α move the receptor *input* of the graph, as in Appendix B. The
  default (`gate`) is a different, earlier parameterisation. Required.
* `--seed-graph` seeds the graph's initialisation from the model seed. Without it a rerun
  draws a new initialisation. Required.
* `--seeds 42 43 44 45 46` are the five model seeds. The default is a single seed.
  Required.

The remaining defaults are the paper's configuration and need no flag:

* the encoder: two signed GraphSAGE layers, hidden size 256, neighbour sampling 25/10,
  L2 normalisation after each layer, 900 epochs, Adam with learning rate 3·10⁻³, weight
  decay 10⁻⁴, gradient clipping at 1.0;
* the graph: on M2OR the message-passing odorants are chosen by greedy pair cover at the
  0.99 coverage quantile; on Mosquito and Fly the graph is the complete panel;
* both forms of OlfaGraph (`cls+mol` and `cls+prot+mol`) and XGBoost-base in every cell.

P1b can share P1a's root because the molecular embedding is part of every cell's file
name. Per cell the sweep writes `metrics_*.csv` (test), `val_metrics_*.csv` (validation,
the same fitted heads) and `records_*.csv` (provenance and wall clock).

To confirm that an existing root was produced with these flags, read them back from the
data rather than from shell history:

```bash
.venv/bin/python scripts/analysis/sweep_provenance.py --root results/graph/olfagraph
```

### P2: interaction baselines

Four methods (LORAX, ProSmith, MolOR, Hladiš) × three datasets × two settings, each run
in its own directory `results/baselines/<cell>/<run>/`. Both directory names are free:
readers classify a run by the `config.json` it writes, not by its path, but the depth is
fixed. Every method receives the same frozen ESM3 embeddings in place of its original
protein encoder.

Each run fits two XGBoost heads per split: `cls` (the method's interaction representation
alone, App. A) and `cls+prot+mol` (the main table).

```bash
E=data/embeddings
PROSMITH_CKPT=data/external/ofm/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl
run () {   # run <env-python> <dataset> <setting> <run-name> <cls-spec> <mol-npz> <mol-tag>
  local PY=$1 DS=$2 SET=$3 NAME=$4 CLS=$5 MOL=$6 MTAG=$7 SPLIT REP COMBOS
  if [[ ${DS} == m2or ]]; then
    COMBOS="1 123"
    if [[ ${SET} == seen ]]; then SPLIT=(--regime full_full --full-full-mode transductive);          REP=(1 2 3 4 5)
    else                          SPLIT=(--regime full_full --full-full-mode inductive_molecule_v5); REP=(42 43 44 45 46); fi
  else
    COMBOS="1 12 123"
    if [[ ${SET} == seen ]]; then SPLIT=(--regime ofm --dataset ${DS} --split-family rand          --task regression)
    else                          SPLIT=(--regime ofm --dataset ${DS} --split-family our_inductive --task regression); fi
    REP=(1 2 3 4 5)
  fi
  ${PY} scripts/modeling/train/train_ensemble_boost.py "${SPLIT[@]}" \
      --out-dir results/baselines/${DS}-${SET} --run-name ${NAME} \
      --source cls=${CLS} \
      --source prot=esm:${E}/proteins/esm3_${DS}.npz:esm3-sm-open-v1 \
      --source mol=gin:${MOL}:${MTAG} \
      --combos "${COMBOS}" --on-missing drop \
      --max-parallel 2 --gpus 0 1 --repeats "${REP[@]}"
}

for DS in m2or cc hc; do
  PRES=${E}/proteins/esm3_per_residue_${DS}.npz
  CB=${E}/molecules/chemberta_77m_${DS}.npz
  for SET in seen cold; do
    run .venv-controls/bin/python ${DS} ${SET} lorax    "lorax:${PRES}"                                 ${CB} chemberta_77m
    run .venv-controls/bin/python ${DS} ${SET} prosmith "prosmith:${PRES}:${CB}::${PROSMITH_CKPT}"      ${CB} chemberta_77m
    run .venv-molor/bin/python    ${DS} ${SET} molor    "molor:${PRES}:1"                               ${CB} chemberta_77m
    run .venv/bin/python          ${DS} ${SET} hladis   "hladis:${E}/proteins/esm3_${DS}.npz"           ${CB} chemberta_77m
  done
done
```

ProSmith is initialised from `${PROSMITH_CKPT}`, the BindingDB-pretrained checkpoint of the
ProSmith paper, shipped in the same release (§2.2). The `mol=gin:` source type is the
generic loader for any molecular `.npz`; its third field is a label only, and the file
decides the embedding. On the insect datasets a third head, `cls+prot` (`12`), is fitted as
well; no table reads it.

**Hladiš on the other molecular embeddings.** Appendix D compares Hladiš with each
molecular embedding. Hladiš builds its own molecular side from SMILES, but its row is
`cls+prot+mol`, whose molecular half is the embedding under test. It therefore needs one
run per embedding:

```bash
for DS in m2or cc hc; do
  for SET in seen cold; do
    GIN=${E}/molecules/gin_supervised_contextpred_${DS}.npz
    [[ ${DS} == m2or ]] && GIN=${E}/molecules/gin_supervised_contextpred_all_m2or.npz
    run .venv/bin/python ${DS} ${SET} hladis_gin  "hladis:${E}/proteins/esm3_${DS}.npz" ${GIN}                      gin
    run .venv/bin/python ${DS} ${SET} hladis_ecfp "hladis:${E}/proteins/esm3_${DS}.npz" ${E}/molecules/ecfp_${DS}.npz ecfp
  done
done
```

Hladiš's training budget is counted in optimizer steps and left at the published defaults
on every dataset: 10,000 steps, 6,000 warm-up steps, evaluation every 500 steps, keeping
the best of the twenty validation checkpoints. That is why its spec carries no budget
fields. Its checkpoints are named without the budget, so changing the budget requires
deleting the run directory first; otherwise the old weights are reloaded.

Two mistakes break a baseline run silently rather than loudly:

* **`--out-dir` is required.** Readers look for `<root>/<cell>/<run>/config.json` at
  exactly that depth.
* **`cls=` must be the first `--source`.** Combos are named by source order; a reordered
  command writes `prot+mol+cls`, and the reader does not find the row.

Every run records its sources, feature sets and repeats in its own `config.json`. For a
root that stopped partway, `relaunch_incomplete.py` rebuilds the command of every
unfinished run from that file:

```bash
.venv/bin/python scripts/modeling/train/relaunch_incomplete.py --root results/baselines
```

### P3: receptor representations

One script fits every row of the receptor-representation table under the same XGBoost
head: the protein language models, the sequence descriptors, the one-hot and
single-input controls, and, with `--gnn`, OlfaGraph itself. OlfaGraph's rows are trained
here, inside this script's own folds. The response-aware embedding of a receptor depends
on the training pairs of its fold, so importing it from another run would leak test
pairs.

```bash
.venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --gnn esm3@1 esm3@0 prott5@1 --seeds 42 43 44 45 46 \
    --root results/tables
```

Every flag above is the script's default, so a bare call does the same. `--no-gnn` leaves
out OlfaGraph's rows: the rest of the table then takes minutes and needs no GPU.
`name@alpha` names an OlfaGraph row: `esm3@1` is ESM3 initialisation, `prott5@1` ProtT5
initialisation, and `esm3@0` is α=0, the table's "random initialization" row (receptor
identity only). The protein language models are ESM3, ESM-1b and ProtT5, each
included only if its `.npz` exists and covers every receptor of the dataset. Output: one
CSV per cell under `--root`, each with a provenance sidecar `prot_floor_<ds>_<regime>.json`.

### P4: the one-hot boosting heads

The identity control of Appendix B.2 compares OlfaGraph at α=0 with XGBoost over a
one-hot receptor block. No sweep writes that head, so it is fitted and cached once:

```bash
.venv/bin/python scripts/article_tables/s3_onehot_boost.py \
    --dataset m2or cc hc \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz'
```

The protein file is passed even though one-hot replaces it: it decides which pairs are
covered, and the control must see exactly the rows of the model it controls.

### P5: architecture

The graph is fixed per dataset inside the script; only the message-passing operator or
the ablated component changes. Two commands, the operators and the encoder ablations:

```bash
# GraphSAGE (ours), GAT, GraphConv, GIN
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --max-parallel 4 --gpus 0 1 2 3

# one layer, positive edges only, unsigned edges, no message passing
.venv/bin/python scripts/article_sweeps/s5_run_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --conv sage:paper:1layer sage:paper:pos sage:paper:unsigned none \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --max-parallel 4 --gpus 0 1 2 3
```

Seeds (42–46) and graph seeding are on by default here. The suffix `:paper` names the
encoder configuration of the paper (neighbour sampling and per-layer normalisation) and
is printed as the plain operator name.

### P6: graph construction

The odorant-selection sweep of Appendix C: eight criteria (seven rankings and a random
control) × six coverage quantiles, on M2OR in both settings. The cell the paper reports,
greedy pair cover at q=0.99, is a point of the grid. At q=0 nothing is cut, so that
column is fitted once and shared by all criteria.

```bash
.venv/bin/python scripts/article_sweeps/s4_run_quantile_criteria.py \
    --dataset m2or --regime inductive transductive \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --seed-graph --seeds 42 43 44 45 46 --max-parallel 4 --gpus 0 1 2 3
```

Mosquito and Fly are not swept: every odorant there is measured against every receptor,
so a coverage cut removes nothing and their graph is always complete.

---

## 4. Checking the artifacts before reading them

```bash
# READY / PARTIAL / MISSING for every input cell of every table
.venv/bin/python scripts/article_tables/inventory.py \
    --sweep-root results/graph/olfagraph --ensemble-root results/baselines
```

A table rendered from a partial artifact still prints, but averages fewer splits than it
claims.

---

## 5. Stage 2: readers, in the order of the paper

| artifact | P1a | P1b | P2 | P3 | P4 | P5 | P6 |
|---|---|---|---|---|---|---|---|
| `tab:main_results` | α=1, both forms, XGBoost-base | | ChemBERTa runs, `cls+prot+mol` | | | | |
| `tab:protein_representations` | | | | ● | | | |
| `tab:graph_ablation` | | | | | | ● | |
| `tab:esm3t1` | XGBoost-base | | M2OR ChemBERTa runs, `cls` | | | | |
| `fig:dial` | every α, both forms, XGBoost-base | | | | | | |
| `tab:alpha0` | α=0 reduced form, XGBoost-base | | | | ● | | |
| `fig:construction` | | | | | | | ● |
| `tab:mol` | α=1 reduced form, XGBoost-base | ● | Hladiš, all three embeddings | | | | |

Readers write LaTeX, a long CSV and a plain-text rendering under
`results/article_tables/`. The paper's tables are typeset from them. Where the paper
stacks or relabels a reader's output, this is noted below.

### Main text

**`tab:main_results`**: every method on all four metrics, both settings, per dataset.

```bash
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --sweep-root results/graph/olfagraph --ensemble-root results/baselines \
    --out results/article_tables/main
```

It writes one table per dataset; the paper stacks the three. Baselines are read as
`cls+prot+mol` and OlfaGraph in its full form. The reader also prints OlfaGraph's reduced
form, which the paper's main table does not show. Cells are mean ± standard deviation over
the five splits. `--no-val-cut` omits the extra MCC/F1 columns at a validation-chosen
threshold, leaving the 0.5 threshold.

**`tab:protein_representations`**: one column per dataset and setting, the metric of
record (AUROC on M2OR, R² on the insects).

```bash
.venv/bin/python scripts/article_tables/s2_protein_sources.py \
    --dataset m2or cc hc --regime transductive inductive \
    --root results/tables --out results/article_tables/protein_representations
```

**`tab:graph_ablation`**: the operators and the encoder ablations in the reduced form,
with XGBoost-base as the unranked anchor.

```bash
.venv/bin/python scripts/article_tables/s5_architecture.py \
    --dataset m2or cc hc --regime transductive inductive \
    --root results/article_sweeps/architecture --out results/article_tables/architecture
```

### Appendix

**`tab:esm3t1` (A)**: each baseline with its interaction representation alone, on M2OR.
It is the main-table reader told to read the baselines as `cls` and to leave OlfaGraph out.

```bash
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --dataset m2or --baseline-combo cls --no-ours \
    --sweep-root results/graph/olfagraph --ensemble-root results/baselines \
    --out results/article_tables/baselines_cls
```

**`fig:dial` (B.1)**: the paired difference between OlfaGraph and XGBoost-base along α,
both forms, with per-panel slope tests.

```bash
jupyter lab notebooks/article_figures/prediction_dial.ipynb
```

Set the first knob, `ROOT_DIR`, to `results/graph/olfagraph` and keep `SPLIT = "test"`.
Nothing is chosen on this figure: α=1 is reported because it is the model, not because
it is the best point. With `SAVE_FIGS = True` the figure lands in
`results/article_figures/alpha_dial/alpha_dial_delta.pdf`.

**`tab:alpha0` (B.2)**: OlfaGraph at α=0 and XGBoost over one-hot receptors, as paired
differences from XGBoost-base.

```bash
.venv/bin/python scripts/article_tables/s3_alpha0_vs_boost.py \
    --sweep-root results/graph/olfagraph
```

**`fig:construction` (C)**: the construction sweep on the validation split.

```bash
jupyter lab notebooks/article_figures/quantile_criteria.ipynb
```

Its defaults already point at P6's output and at `SPLIT = "val"`, where a construction
may be judged. The cell after the figure prints the numbers the appendix text quotes. The
figure lands in `results/article_figures/quantile_criteria/quantile_pair.pdf`.

**`tab:mol` (D)**: OlfaGraph (reduced form), XGBoost-base and Hladiš with each molecular
embedding.

```bash
.venv/bin/python scripts/article_tables/s6_molecule_ablation.py --spread std \
    --sweep-root results/graph/olfagraph --ensemble-root results/baselines \
    --out results/article_tables/molecule
```

`--spread std` reproduces the table as printed (mean ± standard deviation over splits).
Without it the reader quotes the 95% interval.

---

## 6. How the numbers are computed

* **The unit of evidence is the held-out split.** Model seeds are averaged inside each
  split first; intervals and tests are taken over the five splits (Student-t, n=5).
  Treating (split, seed) pairs as independent would roughly halve every interval.
* **Paired comparisons are paired within the split.** OlfaGraph and XGBoost-base come
  from the same sweep cell, on the same held-out rows and seeds, so their difference is
  taken split by split before averaging. The figures of Appendices B and C and the
  differences in `tab:alpha0` are built this way.
* **Tests** are paired two-sided t-tests over splits, Holm-corrected within the family
  each caption names. Wilcoxon is not used: at five splits its smallest attainable
  two-sided p is 0.0625.
* **RMSE is the only metric where lower is better**; every ranking, bold mark and test
  direction flips for it, and the readers handle this themselves.
* **Seen molecules** hold out individual measured pairs (M2OR: LORAX's five folds;
  insects: the released `rand` folds). **Cold molecules** hold out whole odorants
  (M2OR: five molecule-disjoint splits, seeds 42–46; insects: `our_inductive`, built by
  `03_build_ofm_our_inductive_splits.py`). Validation rows are never used for training
  by any model in the paper.

Further detail on the pipeline's pitfalls is in
[`orbind/docs/gotchas.md`](orbind/docs/gotchas.md), and on the boosting ensembler, of
which the paper uses one head per feature set, in
[`orbind/docs/ensembler.md`](orbind/docs/ensembler.md).
