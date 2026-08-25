# LEGACY — structure-based interaction exploration

> **Status: legacy.** This exploratory direction is retained for provenance and is not part of the active notebook or modelling pipeline.
>
> Everything for this direction lives under
> `experiments/struct_interaction/` — its own scripts, data, results, logs.
> It does **not** import from or write to the main `scripts/` or `results/`.
> Read-only reuse of `data/processed/*` (pairs, sequences) is allowed; nothing
> here modifies shared files. This keeps the new direction trivially separable
> from the ESM/concat work that is already concluded.

## Hypothesis

The ESM-embedding + concat direction is closed: the bottleneck is the M2OR
target itself (sparse, missing-not-at-random, narrowly-tuned ORs → pairwise
binding signal is saturated). So instead of *learning* the receptor↔ligand
interaction from M2OR, **pretrain a geometry-aware interaction model on a dense
physics-based signal (docking / MD), then come to M2OR as a ready tool** —
M2OR becomes external validation, not training data.

## Why this is different from everything before

- Dense, non-missing supervision (docking score over arbitrary pocket×ligand).
- Geometry used where it lives (3D pocket from AlphaFold/ESMFold) + an explicit
  interaction module (residue↔atom), the one model class we never tested.
- M2OR used only to validate transfer → no marginal leakage, no biased negatives.

## Staged plan (each stage gated by a cheap decisive check — kill early)

- **Stage 0 — does the teacher carry signal?** Dock a subset (~30–50 receptors)
  of M2OR odorants against their AF/ESMFold pockets with an off-the-shelf docker;
  correlate docking score ↔ M2OR activation label.
  - **KILL CRITERION:** if docking↔activation ρ ≈ 0, the surrogate cannot transfer
    a signal that the teacher does not have → stop. (Direct analogue of the
    partial-correlation discipline that closed the ESM direction.)
  - Risk being tested: target mismatch (docking = binding affinity; M2OR =
    functional **activation**) and AF pocket quality for ORs.
- **Stage 1 — geometry sanity.** pLDDT in pocket residues; compare AF/ESMFold OR
  pockets to known class-A GPCR structures.
- **Stage 2 — surrogate.** If Stage 0/1 pass: train an E(3)-equivariant GNN /
  cross-attention (pocket residues × ligand atoms) to reproduce docking densely.
- **Stage 3 — transfer.** Apply to M2OR zero-shot; then light calibration
  (linear head on frozen interaction features). **Success = beats ESM-mean on
  group_receptor / low-data**, not on stratified.

## Better-than-docking supervisory sources to consider (Stage 0 alternatives)

- **OR mutagenesis** (single residue → activation change): the ideal
  phylogeny-decoupled signal, directly probes pocket sensitivity ESM blurs.
- Pooled deorphanization datasets (more receptors/molecules, denser overlap).
- Class-A GPCR ligand data (ChEMBL/GLASS) to pretrain a generic interaction prior.

## Environment / blockers (as of setup)

- RDKit, torch, BioPython: available.
- Internet to AlphaFold DB: works.
- `uniprot_id` in `pairs_m2or_full.csv` is **unreliable** (409 seqs → 97 UniProt) →
  resolve structures via ESMFold (sequence) or exact seq→UniProt match, not that column.
- **No docking engine installed** (vina/smina/gnina/obabel absent) → Stage 0's
  decisive docking check needs an install (`pip install vina meeko` + ligand/receptor
  prep) or a remote run. Ligand 3D prep and structure acquisition are done first.

## Decisions taken

- Teacher for Stage 0 = **docking** (then reassess if ρ≈0).
- Guard against false-negative = **positive control first**: dock known agonists
  for well-characterized ORs; only trust a global null if the pipeline ranks
  known agonists high.
- Structures = **download AF2 from AFDB** (local folding ruled out: no openfold,
  no CUDA). Resolve via UniProt gene name, not the unreliable uniprot_id column.
- Environment: no separate virtual environment. If this legacy experiment is revisited, add its missing docking dependencies to the root uv project before running it.

## Log

- setup: scaffold + hypothesis/plan written; 488 odorants/409 receptors confirmed.
- `01_prep_ligands.py`: 487/488 odorants -> 3D (ETKDGv3+MMFF) in data/ligands.sdf
  (1 embed fail: KGEKLUUHTZCSIP-JFGNBEQYSA-N).
- Historical isolated venv used meeko 0.7.1 + gemmi + rdkit + biopython; it has since been removed.
- `02_fetch_structures.py`: 8 positive-control ORs (OR2W1/OR1A1/OR1G1/OR51E2/
  OR51E1/OR2J2/OR7D4/OR10G4) -> AF2 models via AFDB API; ALL match their M2OR
  receptor at identity=1.000 (AF residue numbering == ours, BW maps directly).
  Positives in M2OR: OR2W1=115, OR1A1=52, OR1G1=34, OR51E2=28, ...
- Vina engine: official AutoDock Vina v1.2.7 Windows binary -> `bin/vina.exe` (verified).
- `03_prep_receptor.py`: AF2 pdb -> receptor pdbqt + pocket box (BW-22 CA centroid,
  ~24-26 A) via meeko mk_prepare_receptor (gasteiger). 8/8 ok -> data/receptors/.
- `04_prep_ligands_pdbqt.py`: ligands.sdf -> 487 per-odorant pdbqt (mk_prepare_ligand).
- `05_dock.py`: Vina dock of each receptor's M2OR-tested odorants; predictor=-affinity;
  per-receptor AUROC(-aff, M2OR_label) = positive-control enrichment. Incremental +
  resumable (data/dock_results.csv). Pilot OR51E2 4+4 ok (chain validated).
- RUNNING: full positive control, 8 receptors x (<=25 pos + <=50 neg), exhaustiveness 8.
  **READ CRITERION:** mean per-receptor AUROC clearly > 0.5 (docking enriches known/
  M2OR agonists) => pipeline is a faithful teacher, proceed to global Stage-0 ρ.
  AUROC ~ 0.5 => docking can't even rank known agonists => teacher unfaithful, a later
  null would be uninterpretable; would need better structures/endpoint before trusting.
