# scripts/

Entry points. Each script finds the repo root by walking up to `pyproject.toml`, so
it can be invoked from anywhere; call an environment's interpreter directly
(`.venv/bin/python scripts/...`) since `orbind` is put on `sys.path` by the script
itself.

Which environment to use is in the root [README](../README.md#environments). Most of
this runs in the project `.venv` (Hladiš included — it needs rdkit); ProSmith/LORAX need
`.venv-controls`, MolOR needs `.venv-molor`, embedding generation needs
`.venv-embeddings`, and ESM3/ESM-C embeddings need `.venv-esm`.

Self-contained side directions with their own dependencies live in
[`../legacy/experiments/`](../legacy/experiments/) instead.

## Stages

| stage | folder | what it does |
|-------|--------|--------------|
| 0. download   | `downloading/`          | fetch the raw M2OR export |
| 1. preprocess | `preprocessing/`        | pair tables, split indices, our cold-molecule splits, BW numbering |
| 2. embeddings | `embedding_generation/` | protein (ESM-2 / ESM-1b / ProtT5 / ESM3) and molecule (ChemBERTa / GIN / ECFP) caches |
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
  04_build_shrunk_ofm.py                the shrunk insect panels (cc/hc_shrinked,
                                        *_shrinked50): M2OR-shaped "measured" mask
                                        over the complete panel; train/val = split ∩
                                        measured, test = everything else

embedding_generation/
  proteins/    02_embed_receptors, 05_per_residue_embeddings, 06_import_ofm_esm1b,
               embed_proteins_plm (ProtT5 / ESM-C / ESM3, mean + optional
               per-residue)
  molecules/   03_embed_molecules, 07_prepare_ofm_molecules, embed_molecules_gin,
               embed_molecules_ecfp, audit_molecule_npz

modeling/
  train/
    train_ensemble_boost.py         THE entry point: the multi-source boosting
                                    ensemble. Every paper table on M2OR / Carey /
                                    Hallem comes from here.
    relaunch_incomplete.py          rebuilds the command line of every run under an
                                    ensemble_logs root that has fewer than N repeats,
                                    from its own config.json. Prints, never executes.
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
    run_alpha_gate_sweep.py         V8: sweeps the alpha gate on cc/hc x
                                    {transductive, inductive}. Each cell reports the
                                    cls+mol boost metrics AND the receptor cloud's
                                    geometry against ESM and against the response
                                    profile, next to boost_full / naive / the pre-v8
                                    graph on the same folds. --nodes onehot removes ESM
                                    from the graph so alpha is an honest fraction of
                                    structure. (Listed here though it lives in
                                    modeling/train/.)
    concat_diagnostic.py            why GNN + PCA(ESM) scores below the GNN alone: an alpha
                                    sweep, an ESM-width sweep and a same-width random
                                    control, read from an existing run's embeddings.npz
    c9_protein_repr_analysis.py     information-criteria battery (paper section 8)
    mechanism_holdout.py            ligand-class holdout: RSA + predictive OOD over three
                                    receptor representations, all three datasets. Writes the
                                    artifacts that notebooks/graph/mechanism_holdout/ draws.

analysis/
  alpha_grid.py                     the alpha sweep melted: curves, geometry, paired
                                    deltas, mean PLACES. The layer the dial notebooks
                                    import so they compute nothing themselves
  alpha_choice.py                   WHICH alpha to report. --select-on val is the
                                    honest form: choose on validation, then read test
                                    once and print the optimism that choosing on test
                                    would have added
  val_rescore.py                    the sweep scores TEST only. This refits the head
                                    from each cell's dumped receptor cloud, same seed,
                                    and scores VALIDATION -- no graph retraining. Run
                                    it before alpha_choice --select-on val
  paper_tables.py                   the three paper tables, one place column per metric
  headline_table.py                 the scoreboard of record at one alpha
  summarize_runs.py                 one row per (run, combo), mean +- 95% CI,
                                    task-aware columns
  run_dates.py                      one compact line per run: finish time, env,
                                    method, repeats, recorded xgboost. Answers
                                    "which runs did THIS environment produce"
  sweep_provenance.py               which dial/alphas/seeds a sweep root was built
                                    with, read back out of its CSVs; two roots are
                                    also diffed field by field
  extend_runs_with_combo.py         prints the commands that add a cls+prot+mol row
                                    to an existing cls-only run, reusing checkpoints
  mechanism_summary.py              reads a mechanism-holdout run: geometry in raw units
                                    AND as z, the paired GNN+ESM vs one-hot comparison,
                                    and the sparse-matrix assay-design check

`setup_envs.sh` creates `.venv-controls` and `.venv-embeddings`. `.venv-molor` and
`.venv-esm` are set up by hand (see the root README).

## Archive

Closed lines are in [`../legacy/scripts/`](../legacy/scripts/), mirroring this tree
one level down. Nothing there feeds a paper table; most of it is a negative result
worth not repeating. The unattended server queues went with them — the multi-GPU
dispatch that replaced them is built into `train_ensemble_boost.py`
(`--max-parallel` / `--gpus`) and `run_quantile_criteria_sweep.py`.
