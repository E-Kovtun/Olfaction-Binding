"""Stage-0-pre: convert the 3D ligand SDF -> per-odorant pdbqt (meeko).

One pdbqt per odorant (named by inchikey) for Vina docking.

    experiments/struct_interaction/.venv/Scripts/python 04_prep_ligands_pdbqt.py
"""
from __future__ import annotations
import pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
SDF = HERE / "data" / "ligands.sdf"
OUTD = HERE / "data" / "ligands_pdbqt"
OUTD.mkdir(parents=True, exist_ok=True)
MK = HERE / ".venv" / "Scripts" / "mk_prepare_ligand.exe"

cmd = [str(MK), "-i", str(SDF), "--multimol_outdir", str(OUTD)]
print("running:", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
n = len(list(OUTD.glob("*.pdbqt")))
print(r.stdout[-500:])
if r.returncode != 0:
    print("STDERR:", r.stderr[-800:])
print(f"ligand pdbqt files written: {n}  -> {OUTD}")
