# legacy/scripts/queues/

Unattended experiment queues, all from closed lines: `quantile_screen.sh` drove the
v5 quantile screen; the three `run_curated_site_*` / `run_full_full_site_mil_attention.sh`
drove the attention / site-MIL line.

They were versioned with the code on purpose -- a queue defines an experiment and
was reviewed like any other source file. Server-only edits were kept as `*.local.sh`
copies, which git ignores.

Nothing replaced this folder: the multi-GPU dispatch that made these unnecessary is
built into `scripts/modeling/train/train_ensemble_boost.py` (`--max-parallel` /
`--gpus`) and `run_quantile_criteria_sweep.py`.
