# legacy/

Every closed line in one place. **Nothing here feeds a paper table**, and no live
module imports from it.

```text
legacy/
  scripts/     mirrors scripts/ one level down    -> scripts/README.md
  notebooks/   mirrors notebooks/ one level down  -> notebooks/README.md
  notes/       the notes belonging to those lines
  experiments/ isolated side directions with their own deps and binaries
  config.yaml  the pre-`regimes.py` data descriptor (read by nothing since)
```

The mirrored layout means a file's original location is still readable from its
path: `legacy/scripts/modeling/train/train_mp.py` was
`scripts/modeling/train/train_mp.py`. Directory depth is preserved too, so the
`parents[k]` root-walk inside these scripts still resolves.

**One exception to "in one place".** Four archived *library* modules live in
[`../orbind/legacy/`](../orbind/legacy/) — `hetero`, `hetero_gat`, `lorax`,
`attention` — because the scripts and notebooks archived here import them, and
library code has to stay on the import path to be importable. They are imported as
`orbind.legacy.<module>` and nothing live touches them.

---

## Why keep any of it

Most of these are **negative results**. Deleting them makes it likely somebody
re-runs the same idea in six months and re-learns the same thing. Each README
below says what its line showed. The umbrella scoreboard for graph variants is the
Obsidian ledger — consult that before starting any new one.

Nothing here is maintained. Expect drift against the current `orbind` API: several
of these predate the ensembler, the task axis (`orbind/tasks.py`) and `k_mode`.
They are runnable, not supported — if one breaks, it is an archive artifact, not a
regression.

---

## The lines, in one paragraph each

**The graph line, v3–v6** (`scripts/modeling/train/run_graph_*`, `train_gnn_link`,
`train_gat_link`, `train_graph_*`; `notebooks/graph/benchmarks/`) — standalone
trainers and their benchmark notebooks from before the ensembler. Each carried its
own split handling and its own head, which is the duplication
`train_ensemble_boost.py` exists to remove. None of the v6 auxiliary objectives
(BPR, DGI, SimGCL) beat the plain signed graph. The surviving descendant is
`GnnSignedExtractor`. `notebooks/graph/benchmarks/curated.ipynb` is the
GNN-vs-GAT comparison from when the architecture was still undecided — the reason
`orbind/legacy/hetero_gat.py` is kept importable.

**Attention / site-MIL** (`train_attention`, `train_*_site_*`, the matching queues,
`notebooks/baselines_research/attention_*`, `notes/full_full_site_mil_attention_ensemble.md`)
— noisy-OR and LSE site-MIL heads over per-atom molecule features. Still reachable
from the ensembler as `attn_noisy_or` / `attn_lse` sources, but unused.

**Alternative graph formulations, all null** (`scripts/modeling/eval/run_*_cf`,
`run_molecule_graph_mp`, `run_receptor_coresponse_graph`, `run_noresponse_graph`;
`notebooks/graph/alternatives/molecule_side_graphs`) — molecule-side neighbour
aggregation, whether by structure message passing or by binding-profile
collaborative filtering, does **not** beat raw boosting over 5 seeds. An earlier
single-seed "+0.045 win" was seed noise; cold inductive swings ±0.1 per seed, which
is the standing reason five seeds is the minimum there.

**The pocket / protein-variant cluster** (`eval_protein_variants`,
`_append_concat22*`, `build_pocket_variants_cache`, `04_ecl2_embeddings`,
`pocket_binding_signal*`) — every ESM variant (mean, pocket-22, ECL2, background,
random-MLP) scores about the same under boosting. The pocket gives no edge over a
background stretch of the same length, which is the evidence that boosting reads
receptor *identity* rather than the binding site. `scripts/preprocessing/02_bw_numbering.py`
stayed live: its `bw_ref_used_curated.csv` labels receptors by OR family for the
refinement-geometry notebooks.

**Receptor-representation exploration** (`notebooks/protein_embeddings/`, 6) —
ready-made protein encoders are interchangeable here; ESM carries identity plus a
weak neighbourhood prior rather than ligand specificity; learning a receptor
embedding from scratch does not beat frozen ESM transductively. The paper's
protein-side table comes from `scripts/modeling/analysis/prot_floor_sweep.py`, not
from these.

**Molecule-source screening** (`notebooks/molecule_embeddings/`, 2) — superseded by
the ECFP/GIN/ChemBERTa columns now run through the ensembler proper.

**Per-receptor models** (`scripts/modeling/per_receptor_*`) — one model per
receptor, from before the pair-level formulation settled.

**Dead evaluators and one-off generators** (`eval_graph_full_full_*`, `eval_mp_table`,
`eval_full_full_baseline`, `eval_on_lorax_splits`, `gen_feature_spectrum`,
`gen_inductive_enrichment`, `ensemble_run_status`) — evaluators for graph versions
that no longer exist, plus the inductive-enrichment study (closed: pseudo-edges do
not help cold molecules). `ensemble_run_status` was replaced by
`scripts/analysis/summarize_runs.py`.

**`run_quantile_sweep.py`** — superseded by
`scripts/modeling/train/run_quantile_criteria_sweep.py`, which is live and produces
the appendix's quantile × criterion sweeps.

**Structure-based interaction** (`experiments/struct_interaction/`) — AutoDock Vina
docking of odorants into AF2 OR pockets, to test whether a physics-based teacher
could pretrain an interaction model that M2OR then merely validates. Paused at the
positive control: over 8 well-characterised ORs the per-receptor AUROC of
−affinity against the M2OR label is **~0.57 (Stouffer p ≈ 0.012)**, with only OR1A1
individually clear — an honest but far too noisy teacher to pretrain on.

This one is **not** dead weight: §01 and §02 of the paper assert that docking and
structural approaches are out for this family because the structures are weakly
characterised, and this is the measurement behind that sentence. If a reviewer
challenges it, the numbers are in `struct_interaction/data/dock_results.csv`
(480 receptor×odorant dockings) and the log at the bottom of its README.

It kept its own directory (rather than being folded into `legacy/scripts/`) because
the isolation was the point: its own venv, its own `vina.exe`, its own data tree,
importing nothing from `orbind`. Its scripts walk up to `pyproject.toml` like every
other, so the move did not touch them.

**`config.yaml`** — the original data/filter descriptor. Every path in it moved into
`orbind/regimes*.py` and every filter into `orbind/filters.py`; by the time it was
archived nothing read it. Kept as provenance for the early curation decisions.

---

## Running something from here

Same environments as the live tree (`.venv` unless the file says otherwise), same
root-walk, so a notebook or script runs where it always did:

```bash
.venv/bin/python legacy/scripts/modeling/train/train_mp.py --split group_molecule
```

Paths *inside* these files were rewritten when the tree moved, so cross-references
between archived files still point at real locations. References to live files
(`train_ensemble_boost.py`, `orbind.dataset`, …) were left alone and still resolve.

## A note on size

Roughly half of the archived notebooks carry saved outputs, which is most of the
~12 MB the notebooks add to the repository. Stripping outputs here is safe — this is
an archive, and its numbers of record live in the notes and in `results/`.
