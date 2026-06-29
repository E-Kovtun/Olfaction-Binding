# Full-full GNN: 900-epoch training study

Date: 2026-06-29

## Question

The original full-full graph runs stopped at 300 epochs while validation metrics
were still noisy and, for some configurations, still improving. This experiment
records the decoder metrics at every epoch and tests whether a substantially longer
encoder training run improves the final unentangled XGBoost probe.

## Configuration

- dataset: `full_full` / LORAX fold 1
- regime: `inductive_molecule`
- architecture: GraphSAGE GNN
- message passing: `signed`
- molecule coverage filter: `q95`
- protein representation for the final probe: graph-enriched protein only (no raw ESM concat)
- seed: 42
- epochs: 900
- final head: XGBoost on `[raw ChemBERTa molecule || graph-enriched protein]`

The train script now writes exact per-epoch neural-decoder metrics for both validation
and test. Test history is diagnostic only: it is not used for optimization, stopping,
or checkpoint selection.

## Final downstream probe: 300 vs 900 epochs

| Metric | Previous 300-epoch run | New 900-epoch run | Delta |
|---|---:|---:|---:|
| AUROC | 0.7833 | 0.8072 | +0.0239 |
| AUPRC | 0.6409 | 0.6669 | +0.0261 |
| MCC | 0.4865 | 0.5460 | +0.0595 |
| F1 | 0.5606 | 0.6139 | +0.0533 |
| precision | 0.7208 | 0.7654 | +0.0447 |
| recall | 0.4587 | 0.5124 | +0.0537 |

The pre-900 result was preserved locally as
`gnn_signed_q95_unentangled_boost_inductive_molecule_fold1_pre900.pt`.

## Decoder dynamics

- Best validation AUPRC: **0.7983 at epoch 245**.
- Decoder test AUPRC at that epoch: **0.5261**.
- Best diagnostic test AUPRC: **0.5947 at epoch 543**.
- At epoch 900: validation AUPRC **0.7626**, test AUPRC **0.5225**.
- Epoch-wise validation/test correlations:
  - AUROC: 0.799
  - AUPRC: 0.760
  - MCC: 0.729
  - F1: 0.798

Loss and metrics are non-monotonic: there are substantial temporary regressions,
followed by slow recovery. A fixed 300-epoch stop can therefore miss useful later
representations. At the same time, the best neural-decoder validation epoch is not
necessarily the best epoch for the downstream XGBoost probe: the 900-epoch encoder
improved the final probe even though decoder validation had already peaked.

## Recommended next experiment

Train for a large epoch budget and save encoder checkpoints periodically. Select an
epoch using **validation only**. To align selection with the reported downstream task,
fit/evaluate the XGBoost probe on train/validation at saved checkpoints; touch the test
set only once after choosing the checkpoint. Do not select epoch 543 merely because its
diagnostic test curve is highest.

## Artifacts

- `results/full_full/history/gnn_signed_q95_inductive_molecule_fold1.csv`
- `results/full_full/history/gnn_signed_q95_inductive_molecule_fold1.png`
- `scripts/modeling/train/train_graph_full_full.py`
