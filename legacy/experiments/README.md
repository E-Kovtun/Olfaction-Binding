# legacy/experiments/

> **ARCHIVED.** Both the direction and the isolation policy below are historical.
> Nothing here has been touched since June 2026, and nothing in the live pipeline
> depends on it. The narrative index is [`../README.md`](../README.md).

Self-contained research sub-projects that were **deliberately isolated** from the main
[`../../scripts/`](../../scripts/) pipeline: they have their own dependencies, binaries, and data,
and would only pollute the root environment if merged.

> **Why not merge into `scripts/`?** The main pipeline runs in one shared `uv` venv. An
> experiment here may need an incompatible dependency set or a native binary. Keeping it
> as an island means the main pipeline stays reproducible and the experiment can be run
> (or abandoned) without touching it.

## Sub-projects

### `struct_interaction/` — structure / docking direction (paused June 2026)
AutoDock Vina docking of odorants into AF2 / active-state OR pockets, to test whether
structural binding signal predicts M2OR activation.

- own **`.venv`** + **`bin/vina.exe`** (native binary) — the reason this is not in `scripts/`
- numbered pipeline `01_prep_ligands` … `05_dock`, results in `data/dock_results.csv`
- status: paused at the positive control (~0.57 per-receptor AUROC, Stouffer p≈0.012)
- see `struct_interaction/README.md` for details

Conceptually related CPU-only analyses that *do* run in the main venv (e.g. whether pocket
divergence predicts binding divergence) live in
[`../../scripts/modeling/analysis/`](../../scripts/modeling/analysis/), not here.
