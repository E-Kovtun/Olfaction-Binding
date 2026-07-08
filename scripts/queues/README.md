# Server queues

Canonical unattended experiment queues belong here and are versioned with the code. They define the experiment and should be reviewed like any other source file.

For temporary server-only edits, copy a queue to `*.local.sh`; local queue variants are ignored by Git.

`quantile_screen.sh` runs the complete three-repeat quantile screen sequentially on one GPU. `GPU_ID` selects the device; `REPEAT_INDEX=0`, `1`, or `2` can restrict execution to one repeat, while the default `all` runs all three. Existing checkpoints are skipped when the queue is restarted.
