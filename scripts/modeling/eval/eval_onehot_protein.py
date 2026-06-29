"""Control: XGBoost with ONE-HOT protein blocks (no ESM).

Two variants:
  onehot      — per-receptor identity one-hot (409-d)
  onehot_fam  — receptor one-hot ⊕ family one-hot (409 + 17 = 426-d)

Logic of the test:
  * stratified / group_molecule (receptor seen): identity one-hot already
    suffices -> should match ESM. The extra family block adds nothing.
  * group_receptor (receptor UNSEEN): receptor one-hot collapses (its columns
    are all-zero in train), so `onehot` drops to the molecule-only floor.
    The family one-hot still fires (test receptors belong to seen families),
    so `onehot_fam` recovers part of the gap -> shows how much of ESM's
    transfer to new receptors is merely family-level.

Compare against ESM/random already in results/protein_variants_boost.csv.

    uv run python scripts/modeling/eval/eval_onehot_protein.py
"""
from __future__ import annotations
import pathlib, re, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from orbind import dataset as D
from orbind.baselines import run_one

DATA = ROOT / "data"
SPLITS = ["stratified", "group_molecule", "group_receptor"]
SEED = 42


def _family(slug: str) -> str:
    slug = slug.replace("_human", "")
    m = re.match(r"^(or?\d+)", slug)
    return m.group(1).upper() if m else slug.upper()


def main():
    pairs = pd.read_csv(DATA / "processed" / "pairs_curated.csv")
    gin   = D.load_npz_dict(DATA / "embeddings" / "molecules" / "gin_supervised_contextpred.npz")
    esm   = D.load_npz_dict(DATA / "embeddings" / "proteins" / "esm2_650m_mean_curated.npz")
    recs  = list(esm.keys())
    print(f"pairs={len(pairs)}  molecules={len(gin)}  receptors={len(recs)}")

    # family per receptor from bw_ref_used_curated.csv
    ref = pd.read_csv(DATA / "processed" / "bw_ref_used_curated.csv")
    seq2ref = ref.set_index("receptor")["ref_entry"].to_dict()
    fam_of  = {r: _family(seq2ref.get(r, "unk_human")) for r in recs}
    fams    = sorted(set(fam_of.values()))
    fidx    = {f: i for i, f in enumerate(fams)}
    ridx    = {r: i for i, r in enumerate(recs)}
    print(f"receptors={len(recs)}  families={len(fams)}  -> onehot={len(recs)}d  onehot_fam={len(recs)+len(fams)}d")

    eye_r = np.eye(len(recs), dtype=np.float32)
    eye_f = np.eye(len(fams), dtype=np.float32)
    onehot     = {r: eye_r[ridx[r]] for r in recs}
    onehot_fam = {r: np.concatenate([eye_r[ridx[r]], eye_f[fidx[fam_of[r]]]]) for r in recs}

    variants = {"onehot": onehot, "onehot_fam": onehot_fam}

    rows = []
    for name, prot in variants.items():
        for split in SPLITS:
            m, info = run_one(pairs, prot, gin, "boost", split, seed=SEED)
            rows.append({"variant": name, "split": split,
                         **{k: round(float(v), 4) for k, v in m.items()}, **info})
            print(f"  [{name:<11} | {split:<15}] "
                  f"AUROC={m['AUROC']:.3f} AUPRC={m['AUPRC']:.3f} MCC={m['MCC']:.3f}")

    res = pd.DataFrame(rows)
    out = ROOT / "results" / "analysis" / "onehot_protein_boost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)

    print("\n" + "=" * 60)
    for metric in ["AUROC", "AUPRC", "MCC"]:
        piv = res.pivot_table(index="variant", columns="split", values=metric)
        piv = piv.reindex(["onehot", "onehot_fam"])[SPLITS].round(3)
        print(f"\n### {metric}")
        print(piv.to_string())
    print(f"\nsaved -> {out}")
    print("\n(сравни с results/protein_variants_boost.csv: mean/random/...)")


if __name__ == "__main__":
    main()
