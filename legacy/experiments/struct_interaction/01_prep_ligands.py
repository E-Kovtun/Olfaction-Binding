"""Stage-0 prerequisite (fully local): SMILES -> 3D ligand conformers.

Reads the curated odorant SMILES (read-only from data/processed), generates a
single low-energy 3D conformer per molecule with RDKit (ETKDGv3 + MMFF94), and
writes SDF + a manifest into THIS experiment's own data dir. Needed for any
docking; isolated from the main pipeline.

Output (under experiments/struct_interaction/data/):
    ligands.sdf          one 3D conformer per odorant, title = inchikey
    ligands_manifest.csv inchikey, smiles, n_atoms, mmff_energy, status

    python experiments/struct_interaction/01_prep_ligands.py
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
OUT = HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    cur = pd.read_csv(ROOT / "data" / "processed" / "pairs_curated.csv")
    mols = (cur.dropna(subset=["smiles"])
               .drop_duplicates("inchikey")[["inchikey", "smiles"]]
               .reset_index(drop=True))
    print(f"odorants to prepare: {len(mols)}")

    writer = Chem.SDWriter(str(OUT / "ligands.sdf"))
    rows, ok, fail = [], 0, 0
    for _, r in mols.iterrows():
        ik, smi = r["inchikey"], r["smiles"]
        status, n_atoms, energy = "ok", np.nan, np.nan
        m = Chem.MolFromSmiles(smi)
        if m is None:
            status = "parse_fail"
        else:
            m = Chem.AddHs(m)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(m, params) != 0:
                status = "embed_fail"
            else:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        m, AllChem.MMFFGetMoleculeProperties(m))
                    if ff is not None:
                        ff.Minimize(maxIts=500)
                        energy = float(ff.CalcEnergy())
                except Exception:
                    pass
                n_atoms = m.GetNumAtoms()
                m.SetProp("_Name", ik)
                writer.write(m)
        rows.append({"inchikey": ik, "smiles": smi, "n_atoms": n_atoms,
                     "mmff_energy": energy, "status": status})
        ok += status == "ok"; fail += status != "ok"
    writer.close()

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "ligands_manifest.csv", index=False)
    print(f"prepared {ok}  failed {fail}")
    if fail:
        print(man[man.status != "ok"][["inchikey", "status"]].to_string(index=False))
    print(f"-> {OUT / 'ligands.sdf'}")
    print(f"-> {OUT / 'ligands_manifest.csv'}")


if __name__ == "__main__":
    main()
