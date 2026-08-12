"""The task axis: binary binding vs continuous response.

Everything in this repo was built for M2OR, where the target is a 0/1
`Responsive` flag. The Carey and Hallem-Carlson datasets that ship with the
olfactory foundation models release keep the *magnitude* of the response
(globally z-scored), and upstream scores them with R^2 -- so a second task has
to exist alongside the first rather than replacing it.

This module holds the two things every consumer needs to switch on, so the
switch is written once instead of five times:

  * `batch_loss_fn(task)` -- the criterion the pair-level "cls" extractors
    (ProSmith, LORAX, MolOR, Hladis) minimise. All four already share one
    shape: a scalar head emitting a raw value, an optionally weighted loss,
    and best-validation-loss weight selection. Only the criterion differs.
  * `TASKS` -- the canonical spelling, so CLI choices and asserts agree.

Metrics live in `orbind.dataset` (`METRICS[task]`) and the boosting head's own
switch lives in `orbind.baselines.fit_boost`, both next to their classification
counterparts.

A note on the sample weights
----------------------------
`_M2ORWeights` (quality x class-imbalance x pair-imbalance, Hladis eq. 3-6)
is a *classification* device: two of its three factors are about the
positive/negative mix. It disables itself automatically on these datasets --
`maybe_build` returns None when the pool has no `_DataQuality` column, and
neither CC nor HC has one -- so no explicit guard is needed. The weighted
branch below stays for the M2OR regression case, should one ever be built.
"""
from __future__ import annotations

TASKS = ("classification", "regression")


def check_task(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    return task


def batch_loss_fn(task: str):
    """Return `f(pred, y, w) -> scalar loss`, where `pred` is the head's raw
    scalar output and `w` is an optional per-row weight tensor (or None).

    Classification reads `pred` as a logit; regression reads it as the value
    itself. Both use `sum(w_i * l_i) / N` (weight= with reduction='mean'),
    which is upstream ProSmith's convention -- not `/ sum(w)`.
    """
    check_task(task)
    import torch.nn.functional as F

    if task == "classification":
        def loss(pred, y, w):
            return F.binary_cross_entropy_with_logits(pred, y, weight=w, reduction="mean")
        return loss

    def loss(pred, y, w):
        se = F.mse_loss(pred, y, reduction="none")
        return se.mean() if w is None else (w * se).mean()
    return loss
