"""Stage-0-pre: dock each positive-control receptor's M2OR-tested odorants and
measure whether docking ENRICHES the M2OR-positive odorants over negatives.

Per receptor: predictor = -affinity (lower kcal/mol = stronger predicted binding);
AUROC(-affinity, M2OR_label). If docking is a faithful teacher, AUROC > 0.5 and
the known agonists score among the best. This is the positive control that makes
a later global null interpretable.

Incremental + resumable: appends to data/dock_results.csv, skips done pairs.

    experiments/struct_interaction/.venv/Scripts/python 05_dock.py \
        --max-pos 25 --max-neg 50 --exh 8
"""
from __future__ import annotations
import argparse, pathlib, re, subprocess, sys
import numpy as np
import pandas as pd
import gemmi
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
RECEP = HERE / "data" / "receptors"
LIGD = HERE / "data" / "ligands_pdbqt"
STRUCT = HERE / "data" / "structures"
VINA = HERE / "bin" / "vina.exe"
RESULTS = HERE / "data" / "dock_results.csv"
_MODE1 = re.compile(r"^\s*1\s+(-?\d+\.\d+)")


def pdb_seq(acc):
    st = gemmi.read_structure(str(STRUCT / f"AF-{acc}-F1.pdb"))
    out = []
    for res in st[0][0]:
        info = gemmi.find_tabulated_residue(res.name)
        out.append(info.one_letter_code.upper() if info else "X")
    return "".join(out)


def dock(receptor, ligand, box, exh, cpu=4):
    cmd = [str(VINA), "--receptor", str(receptor), "--ligand", str(ligand),
           "--config", str(box), "--exhaustiveness", str(exh), "--cpu", str(cpu)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        m = _MODE1.match(line)
        if m:
            return float(m.group(1))
    return np.nan


def auroc(label, score):
    label = np.asarray(label); score = np.asarray(score)
    pos, neg = label == 1, label == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = rankdata(score)
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", nargs="*", default=None)
    ap.add_argument("--max-pos", type=int, default=25)
    ap.add_argument("--max-neg", type=int, default=50)
    ap.add_argument("--exh", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    man = pd.read_csv(HERE / "data" / "structures_manifest.csv")
    if args.genes:
        man = man[man["gene"].isin(args.genes)]
    pairs = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    rng = np.random.RandomState(args.seed)

    done = set()
    if RESULTS.exists():
        d = pd.read_csv(RESULTS)
        done = set(zip(d["gene"], d["inchikey"]))

    for _, m in man.iterrows():
        gene, acc = m["gene"], m["acc"]
        rec_pdbqt = RECEP / f"{gene}.pdbqt"
        box = RECEP / f"{gene}.box.txt"
        if not rec_pdbqt.exists() or not box.exists():
            print(f"{gene}: receptor not prepped, skip"); continue
        seq = pdb_seq(acc)
        sub = pairs[pairs["receptor"] == seq]
        pos = sub[sub["label"] == 1]["inchikey"].unique().tolist()
        neg = sub[sub["label"] == 0]["inchikey"].unique().tolist()
        rng.shuffle(pos); rng.shuffle(neg)
        pos = pos[: args.max_pos]; neg = neg[: args.max_neg]
        todo = [(ik, 1) for ik in pos] + [(ik, 0) for ik in neg]
        print(f"\n=== {gene} ({acc})  pos={len(pos)} neg={len(neg)} ===", flush=True)

        for ik, lab in todo:
            if (gene, ik) in done:
                continue
            lig = LIGD / f"{ik}.pdbqt"
            if not lig.exists():
                continue
            score = dock(rec_pdbqt, lig, box, args.exh)
            row = pd.DataFrame([{"gene": gene, "acc": acc, "inchikey": ik,
                                 "label": lab, "affinity": score}])
            row.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
            done.add((gene, ik))

    # summary
    if RESULTS.exists():
        d = pd.read_csv(RESULTS).dropna(subset=["affinity"])
        print("\n========== positive-control enrichment ==========")
        print(f"{'gene':<8} {'n_pos':>5} {'n_neg':>5} {'AUROC(-aff,label)':>18}")
        aurocs = []
        for gene, g in d.groupby("gene"):
            a = auroc(g["label"], -g["affinity"])
            aurocs.append(a)
            print(f"{gene:<8} {int((g.label==1).sum()):>5} {int((g.label==0).sum()):>5} {a:>18.3f}")
        print(f"\nmean per-receptor AUROC: {np.nanmean(aurocs):.3f}  "
              f"(>0.5 = docking enriches M2OR positives)")


if __name__ == "__main__":
    main()
