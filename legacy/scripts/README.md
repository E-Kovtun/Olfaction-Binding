# legacy/scripts/

Per-file detail for the archived scripts. The narrative index -- what each closed line
showed, and why it is kept -- is [`../README.md`](../README.md); this file says which file
was which.

The tree mirrors the live `scripts/` one level down, so a file's original location is
readable from its path. Nothing here is maintained.

## modeling/train/

**The graph line, v3–v6.** `run_graph_grid_v3.ps1`, `run_graph_rerun_v2.ps1`,
`run_graph_rerun_v3.ps1`, `run_graph_full_full_v5.ps1`,
`run_graph_full_full_v5_quantile_screen.sh`, and the four v6 objectives
(`_bpr`, `_dgi`, `_dgi_q99`, `_simgcl`). None of the auxiliary objectives beat the plain
signed graph; DGI in particular was dropped by decision, not by accident. The surviving
descendant of this whole line is `GnnSignedExtractor` inside the ensembler.

**Pre-ensembler trainers.** `train_gnn_link.py`, `train_gat_link.py`,
`train_graph_full_full.py`, `train_graph_v4.py`, `train_mp.py` — each was a standalone
script with its own split handling and its own head, which is exactly the duplication
`train_ensemble_boost.py` exists to remove.

**Attention / site-MIL.** `train_attention.py`, `train_curated_site_attention_max.py`,
`train_curated_site_global_attention.py`, `train_full_full_site_mil_attention.py`. The
noisy-OR and LSE site-MIL heads are reachable from the ensembler as `attn_noisy_or` /
`attn_lse` sources, but are not used anywhere any more.

**`run_quantile_sweep.py`** — superseded by `modeling/train/run_quantile_criteria_sweep.py`
(still live: it produces the appendix's quantile x criterion sweeps).

## modeling/eval/

**Alternative graph formulations, all null.** `run_collaborative_factorization`,
`run_metapath_cf`, `run_molecule_profile_cf`, `run_molecule_graph_mp`,
`run_receptor_coresponse_graph`, `run_noresponse_graph`. Molecule-side neighbour
aggregation — whether by structure message passing or by binding-profile collaborative
filtering — does **not** beat raw boosting over 5 seeds. An earlier single-seed "+0.045
win" turned out to be seed noise; cold inductive swings +-0.1 per seed, which is the
standing reason five seeds is the minimum on that regime.

**Evaluators for graph versions that no longer exist.** `eval_graph_full_full_v5_best.py`,
`eval_graph_full_full_quantile_best.py`, `eval_graph_full_full_signed_q99_feature_probes.py`
(+ its `.sh`), `eval_mp_table.py`, `eval_full_full_baseline.py` (+
`run_full_full_baseline_5repeat.sh`), `eval_on_lorax_splits.py`, `gen_feature_spectrum.py`,
`gen_inductive_enrichment.py`. The inductive-enrichment study is closed: pseudo-edges do
not help cold molecules.

**`ensemble_run_status.py`** — replaced by `scripts/analysis/summarize_runs.py`.

## modeling/ (root)

`per_receptor_gin_cv.py`, `per_receptor_gin_oof.py`, `per_receptor_unified_kfold.py` — the
one-model-per-receptor family, from before the pair-level formulation settled.

## modeling/analysis/

`pocket_binding_signal.py` — v1; `pocket_binding_signal_v2.py` is live.

## queues/

`quantile_screen.sh` drove the v5 screen. The three `run_curated_site_*` /
`run_full_full_site_mil_attention.sh` queues drove the attention line above.
