"""Stage-0-pre: AF2 pdb -> docking-ready receptor pdbqt + pocket box.

For each fetched AF2 structure:
  - read the pdb, derive its one-letter sequence,
  - look up the 22 BW pocket positions (0-based seq idx) for that exact sequence
    in bw_pocket_positions_curated.csv (AF2 residue number == idx+1, identity=1.0),
  - box center = centroid of those CA atoms; box size = extent + padding (clamped),
  - run meeko mk_prepare_receptor -> <gene>.pdbqt + <gene>.box.txt (Vina config).

    experiments/struct_interaction/.venv/Scripts/python 03_prep_receptor.py
"""
from __future__ import annotations
import pathlib, sys, subprocess
import numpy as np
import pandas as pd
import gemmi

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
STRUCT = HERE / "data" / "structures"
RECEP = HERE / "data" / "receptors"
RECEP.mkdir(parents=True, exist_ok=True)
MK = HERE / ".venv" / "Scripts" / "mk_prepare_receptor.exe"

PAD, BMIN, BMAX = 3.0, 16.0, 26.0
_three2one = gemmi.ResidueInfo


def pdb_seq_and_ca(pdb_path):
    st = gemmi.read_structure(str(pdb_path))
    chain = st[0][0]
    seq, ca = [], {}
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        seq.append(info.one_letter_code.upper() if info else "X")
        a = res.find_atom("CA", "*")
        if a is not None:
            ca[res.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
    return "".join(seq), ca


def main():
    man = pd.read_csv(HERE / "data" / "structures_manifest.csv")
    bwp = pd.read_csv(ROOT / "data" / "processed" / "bw_pocket_positions_curated.csv",
                      index_col="receptor")
    rows = []
    for _, m in man.iterrows():
        gene, acc = m["gene"], m["acc"]
        pdb = STRUCT / f"AF-{acc}-F1.pdb"
        if not pdb.exists():
            print(f"{gene}: pdb missing"); continue
        seq, ca = pdb_seq_and_ca(pdb)

        if seq not in bwp.index:
            print(f"{gene}: sequence not in bw_pocket_positions (id<1?)"); continue
        pos = bwp.loc[seq].dropna().astype(int).values          # 0-based seq idx
        coords = [ca[p + 1] for p in pos if (p + 1) in ca]
        if len(coords) < 5:
            print(f"{gene}: too few pocket CA atoms ({len(coords)})"); continue
        coords = np.array(coords)
        center = coords.mean(0)
        extent = coords.max(0) - coords.min(0)
        size = np.clip(extent + 2 * PAD, BMIN, BMAX)

        base = RECEP / gene
        cmd = [str(MK), "--read_pdb", str(pdb), "-o", str(base),
               "-p", str(base) + ".pdbqt", "-v", str(base) + ".box.txt",
               "--charge_model", "gasteiger",
               "--box_center", *[f"{c:.3f}" for c in center],
               "--box_size", *[f"{s:.3f}" for s in size]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = (pathlib.Path(str(base) + ".pdbqt").exists()
              or (RECEP / f"{gene}_rigid.pdbqt").exists())
        status = "ok" if ok else "FAIL"
        if not ok:
            print(f"{gene}: mk_prepare_receptor FAIL\n  {r.stderr.strip()[:300]}")
        else:
            print(f"{gene:<8} pocket_CA={len(coords)}  center=({center[0]:.1f},{center[1]:.1f},"
                  f"{center[2]:.1f})  size=({size[0]:.0f},{size[1]:.0f},{size[2]:.0f})  {status}")
        rows.append({"gene": gene, "acc": acc, "n_pocket_ca": len(coords),
                     "cx": center[0], "cy": center[1], "cz": center[2],
                     "sx": size[0], "sy": size[1], "sz": size[2], "status": status})

    pd.DataFrame(rows).to_csv(HERE / "data" / "receptors_manifest.csv", index=False)
    print(f"\n-> {RECEP}")
    print(f"-> {HERE / 'data' / 'receptors_manifest.csv'}")


if __name__ == "__main__":
    main()
