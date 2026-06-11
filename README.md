# orbind  *(template name — rename freely)*

Curated **olfactory receptor ⇄ ligand binding** dataset + embedding pipeline,
built on the **full official M2OR** database export (Lalis et al. 2024).

## Pipeline

```
M2OR.zip ──00 download──▶ relational export (pairs/compounds/receptors, SMILES inline)
         ──01 curate────▶ pairs_curated.csv (receptor, molecule, label, SMILES)
                          receptor_sequences.csv ──02──▶ esm2_650m.npz (1280-d/receptor)
                          molecule_smiles.csv     ──03──▶ mol_graphs.pt (PyG → your GCN)
```

## Data source — use the FULL export, not the GitHub dump

The GitHub flat dump (`M2OR_20230428.csv`) ships **without SMILES**. Script 00
pulls the real database export from the M2OR web app (`/export-db`), a relational
ZIP where **SMILES are inline** and `mixture` / `mutation` / `species` are explicit
fields — so no PubChem/UniProt backfill is ever needed.

## Cleaning policy (script 01)

Human-only · drop engineered **mutants** (`main_receptors.mutation`) · keep **pure**
compounds (`main_compounds.mixture == mono`; `--mixture-policy mono+isomers` to also
keep isomer-sums) · explicit **binary** label · dedup to unique `(sequence, molecule)`
pairs (label = max `responsive`) · drop **orphan** receptors (zero positives).
Nothing is deleted silently — the pair table keeps marker columns
(`is_human`, `is_mutant`, `is_pure`, `is_binary`, `kept_by_us`, `is_orphan`).

Curated counts (human, no-mutants, mono, no-orphans):
**409 receptors · 488 molecules · 21 384 pairs · 1 760 positives (≈1:11) · SMILES 100%.**

## ⚠️ Embeddings are GENERATED, not downloaded

No public database of ESM-2 or GNN embeddings exists for these receptors/molecules:

| Script | Produces | How |
|---|---|---|
| `02_embed_receptors.py` | ESM-2 650M, mean-pooled 1280-d | run ESM-2 over the receptor sequences |
| `03_embed_molecules.py`  | PyG graphs (RDKit) | featurize SMILES; embeddings come from a GCN you train (`--checkpoint`) |

## Environment — uv only (local `.venv`, no global installs)

```bash
uv sync            # installs everything into ./.venv (pandas, torch, fair-esm, rdkit, torch_geometric)
```

## Usage  (always via `uv run`)

```bash
uv run python scripts/00_download_m2or.py                # -> data/raw/M2OR.zip
uv run python scripts/01_build_table.py                  # -> data/processed/pairs_curated.csv (+SMILES)
uv run python scripts/02_embed_receptors.py --sequences data/processed/receptor_sequences.csv
uv run python scripts/03_embed_molecules.py --molecules data/processed/molecule_smiles.csv
```

## Layout

```
orbind/
├── orbind/        filters.py (relational M2OR cleaning)
├── scripts/       00_download · 01_build_table · 02_embed_receptors · 03_embed_molecules
├── data/          raw / processed / embeddings   (gitignored)
├── pyproject.toml · config.yaml
```

## Open design choices

- Mixture policy: `mono` (strict, pure substances) vs `mono+isomers`.
- Molecule model: graphs for an end-to-end GCN (no canonical pretrained GCN here).
- Splits: build **group-aware** (by receptor) to avoid the leakage LORAX's random CV has.
