# scripts/

Entry points. Each script finds the repo root by walking up to `pyproject.toml`, so
it can be invoked from anywhere; call an environment's interpreter directly
(`.venv/bin/python scripts/...`) since `orbind` is put on `sys.path` by the script
itself.

Which environment to use is in the root [README](../README.md#environments). Most of
this runs in the project `.venv`; ProSmith/LORAX need `.venv-controls`, MolOR needs
`.venv-molor`, and embedding generation needs `.venv-embeddings`.

Self-contained side directions with their own dependencies live in
[`../legacy/experiments/`](../legacy/experiments/) instead.

## Stages

| stage | folder | what it does |
|-------|--------|--------------|
| 0. download   | `downloading/`          | fetch the raw M2OR export |
| 1. preprocess | `preprocessing/`        | pair tables, split indices, our cold-molecule splits, BW numbering |
| 2. embeddings | `embedding_generation/` | protein (ESM-2 / ESM-1b / other pLMs) and molecule (ChemBERTa / GIN / ECFP) caches |
| 3. modeling   | `modeling/`             | train, evaluate, analyse |
| 4. reading    | `analysis/`             | dashboards over `results/ensemble_logs/` |

Number prefixes (`00_`, `01_`, …) record the original global order within a stage.

```
preprocessing/
  01_build_table.py                     M2OR export -> pairs_curated.csv
  02_build_full_full_split_indices.py   persist LORAX's fold indices
  02_bw_numbering.py                    Ballesteros-Weinstein numbering; its
                                        bw_ref_used_curated.csv labels receptors by
                                        OR family for the refinement notebooks
  03_build_ofm_our_inductive_splits.py  our stratified cold-molecule splits for
                                        Carey and Hallem (seedless, deterministic)

embedding_generation/
  proteins/    02_embed_receptors, 05_per_residue_embeddings, 06_import_ofm_esm1b,
               embed_proteins_plm (other pLMs + the classical amino-acid floor)
  molecules/   03_embed_molecules, 07_prepare_ofm_molecules, embed_molecules_gin,
               embed_molecules_ecfp, audit_molecule_npz

modeling/
  train/
    train_ensemble_boost.py         THE entry point: the multi-source boosting
                                    ensemble. Every paper table on M2OR / Carey /
                                    Hallem comes from here.
    run_quantile_criteria_sweep.py  quantile x criterion sweep of the pipeline GNN
                                    (appendix); read by
                                    notebooks/graph/alternatives/protein_based_graph*
  eval/
    eval_onehot_protein.py          control: one-hot protein blocks, no ESM. Largely
                                    subsumed by prot_floor_sweep's own controls.
  analysis/
    prot_floor_sweep.py             the protein-source table: real pLMs vs a
                                    classical amino-acid floor, plus onehot /
                                    onehot_only / mol_only controls
    c9_protein_repr_analysis.py     information-criteria battery (paper section 8)
    mechanism_holdout.py            ligand-class holdout: RSA + predictive OOD over three
                                    receptor representations, all three datasets. Writes the
                                    artifacts that notebooks/graph/mechanism_holdout/ draws.

analysis/
  summarize_runs.py                 one row per (run, combo), mean +- 95% CI,
                                    task-aware columns
  extend_runs_with_combo.py         prints the commands that add a cls+prot+mol row
                                    to an existing cls-only run, reusing checkpoints
```

`setup_envs.sh` creates `.venv-controls` and `.venv-embeddings`.

## Archive

Closed lines are in [`../legacy/scripts/`](../legacy/scripts/), mirroring this tree
one level down. Nothing there feeds a paper table; most of it is a negative result
worth not repeating. The unattended server queues went with them — the multi-GPU
dispatch that replaced them is built into `train_ensemble_boost.py`
(`--max-parallel` / `--gpus`) and `run_quantile_criteria_sweep.py`.
