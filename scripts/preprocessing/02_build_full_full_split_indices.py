"""Script 2 -- persist full_full split *indices* (not the LoRaX tables themselves).

LoRaX's rand_split_1..5 are five reshufflings of the same 46563-row pool
(verified identical rows/labels across folds). We store only which pool
positions are train/val/test for each fold, plus for our own cold-molecule
inductive_molecule seeds -- the actual receptor/inchikey/label data is
reloaded from data/splits_indexes/lorax_m2or/ (and our own embeddings) at use time
via orbind.regimes.load_full_full_pool, keeping this artifact tiny.

Usage
-----
uv run python scripts/preprocessing/02_build_full_full_split_indices.py
"""
import pathlib
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind.regimes import build_split_index_store


def main():
    out = _root / "data/processed/full_full_split_indices.npz"
    store = build_split_index_store()
    np.savez(out, **store)
    n_bytes = out.stat().st_size
    print(f"wrote -> {out} ({n_bytes / 1024:.1f} KiB, {len(store)} arrays)")
    for k, v in store.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
