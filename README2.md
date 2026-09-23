# Reproducing the tables

This document is written for someone who has the repository, the data and a GPU, and
wants the paper's tables back. It assumes nothing about the project's history: every
directory it asks you to create is named here, and every command names its inputs and
outputs explicitly rather than relying on a default.

**Scope.** It currently covers the three tables listed below. The remaining ones are
being converted to the same form and will appear here as they are; the first command in
§[3.1](#31-our-graph--run-this-first-before-anything-else) already produces what they
need, and that section says which artifact reads what.

| Table | What it shows | Section |
|---|---|---|
| Main battery | every competitor, the boosting base and our graph, per dataset and regime | [4.1](#41-main-battery) |
| Baselines alone | each competitor in its own learned pair representation, against the base | [4.2](#42-baselines-in-their-own-representation) |
| Receptor representations | our graph, protein language models, classical descriptors, identity controls | [4.3](#43-receptor-representations) |

---

## 1. Vocabulary

Three **datasets**, named in the code by short keys you will pass to `--dataset`:

| key | what it is | target |
|---|---|---|
| `m2or` | human olfactory receptors, binding measurements | binary |
| `cc` | mosquito receptors (Carey et al.), 50 receptors × 110 odorants | continuous |
| `hc` | fly receptors (Hallem & Carlson), 24 × 110 | continuous |

Two **regimes**, the axis every table is split along:

* **transductive** — the test pairs' molecules were seen during training, in other
  combinations. This is the easy end.
* **cold molecule** — no test molecule appears in training at all. In the code this
  regime is called `inductive`; on the insect datasets it is a split family named
  `our_inductive`, and the transductive one is `rand`.

Two **feature sets** a method can be scored on, called *combos*:

* `cls+mol` — the method's own learned representation beside the raw molecule vector;
* `cls+prot+mol` — the same, plus the raw receptor vector.

Everything is scored with the same boosting head, fitted on the training split only, so
the rows differ in their features and in nothing else.

---

## 2. Prerequisites

**Environments.** Two are used below: the project environment (`.venv`) and a second one
for the competitor models that pull heavier dependencies (`.venv-controls`). One
competitor, MolOR, needs its own (`.venv-molor`), and Hladiš needs RDKit, which lives in
the project environment. Each command below names the interpreter it must be run with.

**Embeddings.** All of them are files on disk, keyed by sequence or by InChIKey, and none
is computed during a run:

| what | file | produced by |
|---|---|---|
| receptor, pooled | `data/embeddings/proteins/esm3_{m2or,cc,hc}.npz` | `scripts/embedding_generation/proteins/embed_proteins_plm.py --model esm3` |
| receptor, per residue | `data/embeddings/proteins/esm3_per_residue_{...}.npz` | the same, `--per-residue` |
| molecule | `data/embeddings/molecules/chemberta_77m_{...}.npz` | `scripts/embedding_generation/molecules/03_embed_molecules.py` |

The per-residue file is needed only by the three competitors that cross-attend over the
sequence. Classical protein descriptors (§4.3) are computed inside their own script and
need no file.

**A GPU.** The graph sweep in §3.1 is the expensive step; the rest is boosting.

---

## 3. Producing the numbers

Everything below writes into three roots. Create them wherever you like and keep the
names consistent between the producing and the reading commands — the scripts take them
as flags and parse nothing out of the path:

```
results/graph/main        our graph, every dataset x regime        (§3.1)
results/baselines/        the competitors, one directory per cell   (§3.2)
results/tables/           the receptor-representation fits          (§3.3)
```

### 3.1 Our graph — run this first, before anything else

Two commands, and they are not symmetric. The first is what almost every table waits
on; the second is small, feeds exactly one table, and can be left until just before you
render it.

**(a) The receptor dial, on the reference molecule embedding.** This is the long one.

```bash
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source chemberta \
    --alphas 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/main \
    --max-parallel 4 --gpus 0 1 2 3
```

`--alphas` are positions on the receptor **node dial**. At `1` the receptor nodes carry
the sequence embedding and the graph is the model the tables report. At `0` they carry
only *which receptor this is* — one fixed near-orthogonal vector each, no sequence at
all — so the refined receptor holds nothing but what the response profile put there.
The positions in between are what make the reported end a measured choice rather than
an inherited one.

Running the whole dial rather than only `1` costs about eleven times the graph training
and nothing else; the boosting on top is minutes. It is the first command because five
separate artifacts read it, and because a dial run interrupted halfway is still useful
— the sweep resumes at cell granularity.

What this one run feeds:

| Artifact | What it takes from this run | Anything else needed |
|---|---|---|
| Main battery (§[4.1](#41-main-battery)) | the `1` end, both heads, all six cells | the competitors (§3.2) |
| Baselines alone (§[4.2](#42-baselines-in-their-own-representation)) | the boosting base row | the competitors (§3.2) |
| Dial figure | every position, test and validation | — |
| Mean-rank-against-dial figure | every position, validation | — |
| Identity control | the `0` end against the base | one extra fit, below |

The receptor-representation table (§4.3) is deliberately absent from that list: it
trains its own graphs inside its own folds, so that its row sits beside the descriptor
rows as a like-for-like comparison rather than as an import from elsewhere. It reads
nothing from this root.

The identity control is the only one with a genuine dependency outside this run: it
compares the dial's `0` end against a boosting head over a **one-hot** receptor block,
which no sweep writes. That fit trains no graph, so it is unaffected by anything here
and is cached once:

```bash
.venv/bin/python scripts/article_tables/s3_onehot_boost.py \
    --dataset m2or cc hc \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz'
```

Pass it the same receptor file the sweep used even though one-hot replaces the
embedding: that file decides which receptors are covered, and a different one would
give the control a different set of rows than the thing it controls.

**(b) The other molecule embeddings, at one position.** Small, and it feeds one table.

```bash
.venv/bin/python scripts/modeling/train/run_alpha_gate_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --mol-source gin ecfp --alphas 1.0 \
    --dial nodes --seed-graph --seeds 42 43 44 45 46 \
    --prot-embeddings 'data/embeddings/proteins/esm3_{ds}.npz' \
    --out results/graph/main \
    --max-parallel 4 --gpus 0 1 2 3
```

The molecule-source table asks whether the conclusion survives replacing the molecule
half, and it asks that at the position the paper reports — so there is no dial here,
one point per source is the whole requirement. Nothing else reads these rows, which is
why this command can wait until you are about to render that table.

**Both write into the same root, safely.** The molecule source is part of each cell's
filename and the dial position is part of the cell key inside it, so (b) cannot collide
with (a), and re-running either skips what is already on disk. That also means command
(a) subsumes a plain `--alphas 1.0` run: if you have one, its rows count as done.

Quote `'…{ds}.npz'` — the placeholder is expanded by the script, not by the shell.

`--seeds` are model seeds; five of them are averaged inside each fold before any
interval is taken. `--seed-graph` ties the graph's initialisation to the model seed, so
a rerun of these commands reproduces itself rather than drawing a new lottery ticket.

Each run writes, per cell, `metrics_*.csv` (test), `val_metrics_*.csv` (the same fitted
heads scored on validation — the only split a dial position may be chosen on) and
`records_*.csv` (everything, with wall clock and provenance).

To check that a finished root was produced by the flags you think it was, read them back
out of the data rather than out of your shell history:

```bash
.venv/bin/python scripts/analysis/sweep_provenance.py --root results/graph/main
```

### 3.2 The competitors

Four methods × 3 datasets × 2 regimes, each in its own directory. The layout is
`<root>/<cell>/<run>/`, and both level names are yours to choose — nothing is parsed out
of them, every property of a run is read from the `config.json` it writes.

M2OR, transductive, one competitor:

```bash
PROT=data/embeddings/proteins/esm3_m2or.npz
PRES=data/embeddings/proteins/esm3_per_residue_m2or.npz
MOL=data/embeddings/molecules/chemberta_77m_m2or.npz

.venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime full_full --full-full-mode transductive \
    --out-dir results/baselines/human-transductive \
    --run-name lorax \
    --source cls=lorax:${PRES} \
    --source prot=esm:${PROT}:esm3-sm-open-v1 \
    --source mol=gin:${MOL}:chemberta_77m \
    --combos "1 123" --on-missing drop \
    --max-parallel 2 --gpus 0 1 --repeats 1 2 3 4 5
```

For the cold-molecule cell: `--full-full-mode inductive_molecule_v5`,
`--repeats 42 43 44 45 46`, and `--out-dir results/baselines/human-cold-molecule`.

The other three `cls=` sources, with `{ds}` one of `m2or`, `cc`, `hc`:

```
cls=prosmith:esm3_per_residue_{ds}.npz:chemberta_77m_{ds}.npz::<prosmith checkpoint .pkl>
cls=molor:esm3_per_residue_{ds}.npz:1                      # in .venv-molor
cls=hladis:esm3_{ds}.npz:1:2000:1200:100                   # in .venv (needs rdkit)
```

Hladiš takes the pooled file: it has no per-residue path, and it builds its molecule side
from SMILES, so it takes no molecule npz either. `2000:1200:100` is its step budget.

The insect datasets use a different regime flag and a continuous target:

```bash
.venv-controls/bin/python scripts/modeling/train/train_ensemble_boost.py \
    --regime ofm --dataset cc --split-family rand --task regression \
    --out-dir results/baselines/mosquito-transductive \
    --run-name lorax \
    --source cls=lorax:data/embeddings/proteins/esm3_per_residue_cc.npz \
    --source prot=esm:data/embeddings/proteins/esm3_cc.npz:esm3-sm-open-v1 \
    --source mol=gin:data/embeddings/molecules/chemberta_77m_cc.npz:chemberta_77m \
    --combos "1 12 123" --on-missing drop \
    --max-parallel 2 --gpus 0 1 --repeats 1 2 3 4 5
```

`--split-family our_inductive` is the cold-molecule cell; `--dataset hc` is the fly. The
insect protein file must be named explicitly — a bare `cls=lorax` would load its M2OR
default, which holds none of these receptors, and the run would die with
`num_samples=0`.

Twelve directories result, one per (dataset, regime), four runs in each:

```
results/baselines/
    human-transductive/{lorax,prosmith,molor,hladis}/
    human-cold-molecule/...
    mosquito-transductive/...      mosquito-cold-molecule/...
    fly-transductive/...           fly-cold-molecule/...
```

**Two things that break a run silently rather than loudly.**

`--out-dir` is not optional. The readers look for `<root>/<cell>/<run>/config.json` at
exactly that depth; a run written to the default root is invisible to them.

`cls=` must come first. Combos are named after the order the `--source` flags appear in,
so a reordered command line produces `prot+mol+cls` where the reader looks for
`cls+prot+mol`, and the row is simply not found.

### 3.3 Receptor representations

One script fits every row of the receptor table: classical amino-acid descriptors, the
identity controls, each protein language model, and — with `--gnn` — our graph, trained
inside this script's own folds so that its row is comparable with the others by
construction:

```bash
.venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py \
    --dataset m2or cc hc --regime transductive inductive \
    --gnn esm3@1 esm3@0 prott5@1 --seeds 42 43 44 45 46
```

`name@alpha` is a protein source and a position on the node dial: `1` is the ordinary
graph over that source, `0` is a graph told only which receptor this is and nothing about
its sequence. It writes one CSV per cell under `results/tables/`, plus a provenance
sidecar beside each.

`--seeds` means the same thing here as everywhere: **one** seed per row, initialising the
graph and seeding the boosting head, so a row is (split, seed) and our rows are averaged
over exactly as many draws as every row they are compared with.

Our rows are fitted on this script's own folds, which are the project's folds with the
validation split left unscored -- so they land within about a thousandth of the same
graph's row in the main battery (§4.1). That agreement is worth checking when you have
both: it is the cheapest evidence that the two chains are reading the same model.

This run is resumable per cell. It refuses to resume a table whose graph rows were
trained under a different encoder configuration than the one you are asking for now, and
says so by name — two encoders in one column is not a table.

---

## 4. Rendering the tables

Readers only read. None of them fits anything, so they are cheap to re-run, and each one
takes the roots from §3 explicitly.

### 4.1 Main battery

```bash
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --sweep-root results/graph/main \
    --ensemble-root results/baselines \
    --out results/paper/main-battery
```

One table per dataset, both regimes side by side: the four competitors on
`cls+prot+mol`, the boosting base, and our graph in both of its forms. `--no-val-cut`
drops the extra column whose decision threshold is chosen on validation and reports the
plain 0.5 cut.

### 4.2 Baselines in their own representation

The same reader, told to take each competitor in its `cls` form alone — its own learned
pair representation, with no raw vectors beside it — and to leave our graph out:

```bash
.venv/bin/python scripts/article_tables/m1_main_tables.py --no-val-cut \
    --dataset m2or --baseline-combo cls --no-ours \
    --sweep-root results/graph/main \
    --ensemble-root results/baselines \
    --out results/paper/baselines-cls
```

This is the table that motivates the previous one: it shows what each competitor's
representation carries on its own, against a boosting base that has no learned
representation at all.

### 4.3 Receptor representations

```bash
.venv/bin/python scripts/article_tables/s2_protein_sources.py \
    --dataset m2or cc hc --regime transductive inductive \
    --root results/tables \
    --out results/paper/receptor-representations
```

One combined table, a column per (dataset, regime), the metric of record only.

---

## 5. Two rules that hold everywhere

**Keep a graph root and a competitor root together.** A table renders the graph from
`--sweep-root` and the competitors from `--ensemble-root`, and it cannot tell that the
two were produced on different receptor embeddings. Crossing them does not fail; it
prints a table comparing a graph on one protein representation against competitors on
another. One receptor embedding, one pair of roots.

**A root is one configuration.** The sweeps resume by skipping cells already on disk, and
what identifies a cell does not include every flag. Topping an existing root up with a
command whose flags differ therefore adds rows from a different model under the same
name. If you change a flag, change `--out` with it.

---

## 6. What the numbers mean

The unit of evidence is the **held-out split**, not the (split, seed) pair. Model seeds
are averaged inside each split first, and intervals are taken across the five splits.
Counting seeds as independent observations would halve every interval in these tables
and manufacture differences that are not there.

`RMSE` is the only metric here where lower is better; every ranking, bolding and
significance mark flips for it, and the readers handle that themselves.
