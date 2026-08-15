"""Audit molecule-embedding npz files against the canonical dataset molecule set.

Checks that every source npz (ChemBERTa / GIN / ECFP / ...) is in the SAME format
and covers the SAME molecule set as the dataset's pool, so the molecule-source
sweep compares methods on identical rows. Reports, per npz: key count, embedding
dim/dtype, pool coverage (missing / extra), NaN/inf, all-zero rows, duplicate ids;
and, across all npz, the common intersection and whether every file exactly equals
the pool set.

    python scripts/embedding_generation/molecules/audit_molecule_npz.py --dataset m2or
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))


def _mols(dataset):
    return [f"data/embeddings/molecules/chemberta_77m_{dataset}.npz",
            f"data/embeddings/molecules/gin_supervised_contextpred_{dataset}.npz",
            f"data/embeddings/molecules/ecfp_{dataset}.npz"]


DEFAULTS = {"m2or": _mols("m2or"), "cc": _mols("cc"), "hc": _mols("hc")}


def pool_molecules(dataset: str) -> set:
    if dataset == "m2or":
        from orbind.regimes import full_full_pairs
        return set(full_full_pairs(pool_fold=1)["inchikey"].unique())
    if dataset in ("cc", "hc"):
        bridge = _root / "data" / "processed" / "molecules" / f"molecule_smiles_{dataset}.csv"
        return set(pd.read_csv(bridge)["inchikey"].unique())
    raise SystemExit(f"pool for dataset={dataset!r} not wired (m2or/cc/hc)")


def audit_one(path: pathlib.Path, pool: set):
    if not path.exists():
        return {"file": path.name, "status": "MISSING FILE"}
    d = np.load(path, allow_pickle=True)
    if "ids" not in d.files or "emb" not in d.files:
        return {"file": path.name, "status": f"BAD FORMAT (keys={d.files}, expected ids+emb)"}
    ids = d["ids"].tolist(); emb = d["emb"]
    keyset = set(ids)
    finite = np.isfinite(emb).all()
    allzero = int((~emb.any(axis=1)).sum())
    return {"file": path.name, "status": "ok", "keys": len(ids), "unique": len(keyset),
            "dup": len(ids) - len(keyset), "dim": emb.shape[1], "dtype": str(emb.dtype),
            "in_pool": len(keyset & pool), "missing_from_pool": len(pool - keyset),
            "extra_not_in_pool": len(keyset - pool), "finite": bool(finite),
            "allzero_rows": allzero, "_keyset": keyset}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="m2or")
    ap.add_argument("--npz", nargs="+", default=None, help="npz paths (default: dataset's standard three)")
    args = ap.parse_args()

    pool = pool_molecules(args.dataset)
    paths = [pathlib.Path(p) for p in (args.npz or DEFAULTS[args.dataset])]
    paths = [p if p.is_absolute() else _root / p for p in paths]

    print(f"dataset={args.dataset}  pool molecules={len(pool)}\n" + "=" * 96)
    rows, keysets = [], {}
    for p in paths:
        r = audit_one(p, pool)
        if "_keyset" in r:
            keysets[r["file"]] = r.pop("_keyset")
        rows.append(r)

    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['file']:<48} {r['status']}"); continue
        flag = "OK " if (r["missing_from_pool"] == 0 and r["extra_not_in_pool"] == 0
                         and r["dup"] == 0 and r["finite"] and r["allzero_rows"] == 0) else "!! "
        print(f"{flag}{r['file']:<46} keys={r['keys']:<5} dim={r['dim']:<5} {r['dtype']:<8} "
              f"pool:{r['in_pool']}/{len(pool)}  missing={r['missing_from_pool']} "
              f"extra={r['extra_not_in_pool']} dup={r['dup']} finite={r['finite']} zero={r['allzero_rows']}")

    if len(keysets) >= 2:
        common = set.intersection(*keysets.values())
        union = set.union(*keysets.values())
        print("=" * 96)
        print(f"common to ALL present npz: {len(common)}   |   union: {len(union)}")
        print(f"all present npz == pool exactly: "
              f"{all(ks == pool for ks in keysets.values())}")
        for f, ks in keysets.items():
            miss, extra = len(pool - ks), len(ks - pool)
            if miss or extra:
                print(f"   {f}: {miss} missing / {extra} extra vs pool")
        # rows any sweep would actually share (drop-intersection)
        print(f"\nfair-sweep molecule set (intersection across sources) = {len(common)} molecules")
        if len(common) < len(pool):
            print(f"   -> {len(pool) - len(common)} pool molecule(s) absent from >=1 source; "
                  f"a fair CB/GIN/ECFP comparison runs on these {len(common)}.")


if __name__ == "__main__":
    main()
