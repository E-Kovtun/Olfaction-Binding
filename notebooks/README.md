# notebooks/

**Display and analysis only.** Nothing here trains a model that a paper number
depends on — those come from `scripts/`, and land in `results/`. A notebook reads
CSVs and checkpoints, computes derived statistics, and draws.

All of them locate the repo root by walking up to `pyproject.toml`, so they run
from any depth:

```python
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
```

```bash
uv run python -m ipykernel install --user --name orbind --display-name "orbind (uv)"
uv run jupyter lab
```

## What is here

```text
notebooks/
  datasets/                carey_hallem_carlson_overview  -- the dataset description
                           behind the paper's data section
  graph/
    mechanism_holdout/     {M2OR, CC, HC} -- ligand-class holdout: does a refined
                           receptor carry a MECHANISM to a chemistry it never saw?
                           RSA/Mantel is the metric of record; a predictive-OOD
                           boosting readout runs on the same masks.
    refinement_geometry/   {M2OR, CC, HC} -- what refinement moves in the receptor
                           geometry; the companion to the same study.
    alternatives/          display-only readers for the quantile x criterion sweeps
```

See [`graph/README.md`](graph/README.md) for the graph notebooks in detail.

Outputs are committed when they are scientific figures or short summaries.
Tracebacks, run inventories and training logs are not.

## Archive

Closed-line notebooks are in [`../legacy/notebooks/`](../legacy/notebooks/) — the
benchmark sweeps from before `results/ensemble_logs/`, the receptor- and
molecule-representation screening, the attention/site-MIL line, and the
molecule-side graph nulls. They still run.
