# scripts/

Entry points. Each script finds the repo root by walking up to `pyproject.toml`, so it
can be invoked from anywhere; call an environment's interpreter directly
(`.venv/bin/python scripts/...`), since `orbind` is put on `sys.path` by the script itself.
Which environment each one needs is in [`README.md`](../README.md) §1.

**To reproduce the paper, follow [`README.md`](../README.md).** It names the exact
invocation of every script the paper depends on. This file is the map of the folder: what
each script is, and whether the paper uses it.

## Stages

| stage | folder | role in the paper |
|-------|--------|-------------------|
| 0. download   | `downloading/`          | not used (the benchmarks come from the LORAX release) |
| 1. preprocess | `preprocessing/`        | split indices for M2OR, cold-molecule splits for the insects |
| 2. embeddings | `embedding_generation/` | ESM3, ProtT5, ESM-1b; ChemBERTa, GIN, ECFP |
| 3. producers  | `modeling/`, `article_sweeps/` | train OlfaGraph, the baselines and the ablations; write results |
| 4. readers    | `article_tables/`, `analysis/` | turn results into the paper's tables |

Figures are drawn by notebooks in `../notebooks/article_figures/`.

## What is where

`P1`–`P6` refer to the producers of [`README.md`](../README.md) §3. Scripts without a
mark are not on the paper's path.

```
preprocessing/
  02_build_full_full_split_indices.py   [paper] M2OR split indices: LORAX's five folds
                                        (seen molecules) and seeds 42-46 (cold molecules)
  03_build_ofm_our_inductive_splits.py  [paper] cold-molecule splits for Mosquito and Fly
  01_build_table.py                     M2OR export -> pairs_curated.csv (curated regime)
  02_bw_numbering.py                    Ballesteros-Weinstein numbering
  04_build_shrunk_ofm.py                sparsified insect panels (closed experiment)

embedding_generation/
  proteins/
    embed_proteins_plm.py               [paper] ESM3 (mean + per-residue) and ProtT5
    06_import_ofm_esm1b.py              [paper] ESM-1b, imported from the benchmark release
  molecules/
    07_prepare_ofm_molecules.py         [paper] SMILES->InChIKey tables and the released
                                        ChemBERTa vectors, all three datasets
    embed_molecules_gin.py              [paper] pretrained GIN (Hu et al.)
    embed_molecules_ecfp.py             [paper] ECFP4
    audit_molecule_npz.py               coverage and key-collision check for a molecule npz
    03_embed_molecules.py               PyG molecule graphs (not used)

modeling/
  train/
    run_alpha_gate_sweep.py             [paper, P1] OlfaGraph and XGBoost-base on the same
                                        splits, over the alpha dial of Appendix B
    train_ensemble_boost.py             [paper, P2] the multi-source boosting pipeline: the
                                        four interaction baselines, one head per feature set
    relaunch_incomplete.py              prints the command of every run under a baseline root
                                        from its own config.json (--show-complete: all runs)
    run_quantile_criteria_sweep.py      the first construction sweep; superseded by
                                        article_sweeps/s4_run_quantile_criteria.py
  analysis/
    prot_floor_sweep.py                 [paper, P3] every row of the receptor-representation
                                        table, OlfaGraph's rows trained in its own folds
    mechanism_holdout.py, concat_diagnostic.py, c9_protein_repr_analysis.py
                                        parked analyses, not in the paper

article_sweeps/                         producers that exist only for the ablations
  s5_run_architecture.py                [paper, P5] operators and encoder ablations
  s4_run_quantile_criteria.py           [paper, P6] odorant-selection criterion x quantile
  s4_quantile_grid.py                   the aggregation layer the construction notebook
                                        imports (no command line)

article_tables/                         the readers; see article_tables/README.md
  m1_main_tables.py                     [paper] main table, and App. A with --baseline-combo cls
  s2_protein_sources.py                 [paper] receptor-representation table
  s3_onehot_boost.py                    [paper, P4] fits the one-hot XGBoost heads (cached)
  s3_alpha0_vs_boost.py                 [paper] App. B.2
  s5_architecture.py                    [paper] architecture table
  s6_molecule_ablation.py               [paper] App. D
  inventory.py                          READY / PARTIAL / MISSING per input cell of every table
  tablekit.py                           shared conventions of every table

analysis/
  alpha_grid.py                         the sweep melted: fold means, intervals, paired
                                        differences. Readers and figure notebooks import it
  paper_tables.py                       locates a baseline run by its config.json; used by
                                        the readers
  sweep_provenance.py                   which flags a sweep root was built with, read back
                                        from its CSVs; also diffs two roots
  val_rescore.py                        adds validation scores to a sweep root made before
                                        validation files were written
  summarize_runs.py, run_dates.py, extend_runs_with_combo.py
                                        dashboards over baseline roots
  alpha_choice.py, headline_table.py, mechanism_summary.py
                                        earlier readers, not in the paper

legacy/                                 retired scripts: the parked geometry line, ESM-2
                                        generation, and others (see legacy/README.md)
```

`setup_envs.sh` creates `.venv-controls` and `.venv-embeddings`. `.venv-molor` and
`.venv-esm` are set up by hand (see [`README.md`](../README.md) §1).

## Archive

Closed lines are in [`../legacy/scripts/`](../legacy/scripts/), mirroring this tree one
level down. Nothing there feeds the paper; most of it is a negative result worth not
repeating.
