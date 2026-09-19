# Gotchas

Things that have already cost time here, in the order you are likely to hit them.
Ensembler-specific ones are in [`ensembler.md`](ensembler.md); these are
pipeline-wide.

## Environments

Running a method in the wrong environment does not always crash. MolOR imported
from `.venv` (no dgl) once "trained" a fold in 5 s — it had found an existing
checkpoint and taken the load-and-embed path. Any result that looks impossibly
fast is an environment or checkpoint artifact until proven otherwise; cross-check
against the method's own step budget before quoting a wall-clock number.

The server shell is **zsh**, where `$VAR` is *not* word-split. A variable holding
several flags expands as one argument and argparse fails with an unrelated-looking
error. Only single-token values go in variables there. Forcing the split with
`${=VAR}` is no fix: it also splits a quoted multi-word value such as
`--combos "1 12 123"`. Two more zsh-only traps: `$VAR:e` (a colon followed by a
letter) is a history modifier, so brace every variable that touches a colon —
`${base}:…`; and a glob with no match is an *error*, not a literal, so
`rm -rf pool-*` aborts an `&&` chain when nothing matches — write `pool-*(N)`.

Hladiš runs in the project `.venv`, not `.venv-controls`: it needs rdkit, and the
controls env does not install it.

## Metrics

**R² is measured against the test mean; naive predicts the train mean.** A model
can beat the naive row and still score below zero. Always read them together —
`summarize_runs.py` prints naive for regression runs whenever it was recorded.

**Five seeds is the minimum on cold-molecule regimes.** Per-seed swings there reach
±0.1 AUROC. A one-seed "+0.045 win" in this project turned out to be pure seed
noise, which is why the molecule-side-graph line was closed on 5 seeds and not on
the first result.

**Rank statistics are per fold, not rank-of-means.** Where the paper reports mean
ranks, methods are ranked within each fold and the ranks averaged. Ranking the
5-fold means instead hides fold-level disagreement and is a different (weaker)
claim.

## Splits

**Upstream's `scaf` family is an unusable instrument on Carey.** Its rule is
deterministic; with 71 of 110 odorants sharing the empty Bemis–Murcko scaffold,
folds 1–3 are three slices of that one group, and fold 1 lands on the
carboxylic-acid homologous series — test sd 0.215, naive R² −4.92. The published
−1.016 average *is* that one fold. Its 5-fold mean carries a ±7 CI and cannot be
read; even the honest median barely clears naive.

`our_inductive` (`scripts/preprocessing/03_build_ofm_our_inductive_splits.py`)
makes the same cold-molecule claim with test sd 0.97–1.04 and naive R² ≈ 0 on
every fold. It is seedless and deterministic: molecules ordered by response
dynamic range, dealt by systematic sampling. It is **not** scaffold-disjoint and
does not claim to be — with that scaffold distribution, no 5-fold scheme here can
be, upstream's included.

**`inductive_molecule` vs `inductive_molecule_v5`** are two different cold-molecule
splits of M2OR: the former is ours (30% holdout, stratified), the latter reproduces
the v5 graph screen's own split (20% test / 10% val molecules, unstratified) so
those numbers stay comparable head-on. Reported baselines all use `_v5`.

## The quantile axis

On M2OR the coverage distribution is long-tailed and the coverage quantile
isolates a hub core. On Carey and Hallem **every** molecule is measured against
**every** receptor, so the coverage vector is constant, `np.quantile` returns that
same value, and `cov >= threshold` keeps everything: q becomes a no-op and a whole
sweep collapses to one point (verified: q=0.99 and q=0 both give K=70 of 70).
`--k-mode fraction` is the tie-free reading of the same intent and is what those
datasets must use.

The criterion axis degenerates there too: on a complete matrix protein IDF is
`log2(m/m) = 0`, so `idf_coverage` and `composite` are identically zero, and
several of the remaining criteria rank identically. Four of the seven carry no
information on those datasets.

**Label-based criteria need a threshold on a continuous target.** `coverage`,
`balance_bits`, `entropy_bits` and `disc_pairs` count positives and negatives.
Left with the historical `y == 1` / `y == 0` rule they are all empty on a z-scored
response. Pass `pos_threshold` — the same value the graph uses for edge signs — so
"positive mix" means the same thing to the ranking and to the message passing.

## Re-implemented baselines

**Hladiš's budget is in passes, not steps.** Upstream's `train_epoch` iterates the
full loader and the outer loop is `while epoch <= N_EPOCH`, so the paper's "10 000
epochs" is ≈4.09M optimizer steps. The released `config_train.yml` is a debug
config — its curriculum ratio is the inverse of the paper's. Our re-implementation
runs a small fraction of that budget, which is why its own scalar head reaches
0.600 AUPRC against a published 0.765. That is a compute gap, not a port bug, and
it does not affect the ensemble rows built on its features. Transductively our port
reaches 0.729 AUPRC against his 0.765, inside his own 0.07–0.08 spread.

**External `cls` sources default to M2OR protein files.** LORAX, ProSmith and MolOR
read `esm1b_650m_per_residue_full_full.npz`, Hladiš `esm1b_650m_mean.npz`. On the
insect panels a bare `cls=lorax` therefore covers no receptor: `run_ensemble` drops
every row as uncovered and the run dies inside the DataLoader with `num_samples=0`,
far from the cause. The `WARNING coverage[train]: 0/…` line just above it is the
tell. Always name the insect file in the spec — the full set is in the root README.

**Hladiš's step budget is sized for M2OR.** 10000/6000/500 steps on M2OR's ~41k
train rows; on Carey's 3.5k that is ~285 epochs. The insect runs use
`1:2000:1200:100` (n_models:max_steps:warmup:eval_every), same 0.6 warmup ratio, and
the shrunk panels keep it for comparability even though their train sets differ.

**A method's name is not a configuration.** "LORAX" as a `cls` source and "LORAX
cls+prot+mol" are different rows, and our graph's headline `cls+mol` deliberately
excludes raw ESM while every external baseline's best row includes it. Always read
the combo name alongside the method name.

## Shrunk insect panels

**Two "cold" notions, not one.** `test_origin.csv` flags `origin == "upstream_test"`
(the parent panel's own test block — the rows to use for a paired shrunk-vs-complete
comparison) and `cold_molecule` / `cold_receptor` (no row in the *masked* train). The
second is wider — on Carey 1933 cells against the split's 1100 — because the mask can
delete every measurement of an entity, which the model cannot tell from a held-out one.

**A dataset tag must survive the filename round trip.** Sweep files are
`metrics_{ds}_{family}…csv`, and `cc_shrinked` contains the separator: split naively
it reads as dataset `cc`, family `shrinked_rand`, which is no family, and every file
was dropped as unparseable — the tables printed `--` on a finished sweep.
`headline_table.parse_name` now matches the longest known dataset first, and
`test_sweep_planning` round-trips every tag the writer knows. A new tag needs an entry
in `CANONICAL_VARIANT`, or it is not "known".

## Counting parameters

`torch.load(state_dict)` + `requires_grad` reports **0** for everything — loaded
tensors are not live `nn.Parameter`s. Count `numel` over the saved tensors and
subtract frozen backbones by name (LORAX freezes its ChemBERTa base via peft;
ProSmith trains its whole 42.5M `main_bert`).
