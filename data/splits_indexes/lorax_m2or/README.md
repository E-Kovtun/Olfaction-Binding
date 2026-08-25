# LoRaX transductive splits (5 folds)

These 5 folds (`rand_split_1` .. `rand_split_5`, each with `train_df.csv`/
`val_df.csv`/`test_df.csv`) are **borrowed as-is from LoRaX**
(McConachie et al. 2025, ICLR2026) — they define the `transductive` regime
for our full_full pipeline.

This is a deliberate methodological borrowing, not an embedding source: we
reuse LoRaX's own train/val/test partitioning of the M2OR pool so our
`transductive` numbers are directly comparable to theirs. It is intentionally
**not** unified/renamed to hide its origin the way embedding files are (see
`data/embeddings/`) — the point here is exactly that it *is* LoRaX's split,
and that should stay visible.

All 5 folds are the same underlying ~46563-row pool, just reshuffled
differently across train/val/test each time (verified: identical rows and
labels across folds). We don't duplicate this data anywhere else -- only the
resulting *indices* are persisted, in `data/processed/full_full_split_indices.npz`
(see `orbind/regimes.py`). These csv files are the "common source" those
indices are reloaded against.
