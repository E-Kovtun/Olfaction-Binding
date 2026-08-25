# Server queues

Canonical unattended experiment queues belong here and are versioned with the code. They define the experiment and should be reviewed like any other source file.

For temporary server-only edits, copy a queue to `*.local.sh`; local queue variants are ignored by Git.

The queues that lived here (`quantile_screen.sh`, the site-attention / site-MIL runners)
belong to closed lines and moved to [`../legacy/queues/`](../legacy/queues/). This folder
is currently empty of live queues -- the multi-GPU dispatch that replaced them is built
into `train_ensemble_boost.py` (`--max-parallel` / `--gpus`) and
`run_quantile_criteria_sweep.py`.
