# orbind  *(template name — rename freely)*

Curated **olfactory receptor ⇄ ligand binding** dataset + embedding pipeline,
built on the official **M2OR** database (Lalis et al. 2024).

## What this repo gives you

A reproducible path from the raw M2OR dump to model-ready tensors:

```
M2OR_20230428.csv ──01──▶ pairs_curated.csv (receptor, molecule, binary label)
                         receptor_sequences.csv ──02──▶ esm2_650m.npz  (1280-d / receptor)
                         molecule_smiles.csv     ──03──▶ gin_contextpred.npz (300-d / molecule)
```

## Cleaning policy (script 01)

Human-only · drop engineered **mutants** · explicit **binary** label · dedup to
unique `(receptor, molecule)` pairs (label = max `Responsive`) · drop **orphan**
receptors (zero positives). Nothing is deleted silently — every row keeps marker
columns (`is_human`, `is_mutant`, `is_binary`, `kept_by_us`, `is_orphan`).

Curated counts (human, no-mutants, no-orphans, key = Gene ID):
**560 receptors · 663 molecules · 31 398 pairs · 2 397 positives (≈1:12).**

## ⚠️ Embeddings are GENERATED, not downloaded

There is **no public database** of ESM-2 or GNN embeddings for these
receptors/molecules — both scripts compute them:

| Script | Produces | Source reality |
|---|---|---|
| `02_embed_receptors.py` | ESM-2 650M, mean-pooled 1280-d | run ESM-2 over sequences (weights ~2.5 GB, once) |
| `03_embed_molecules.py` | PyG graphs (RDKit) → your GCN | featurize SMILES; embeddings come from a GCN you train (or `--checkpoint`) |

The raw M2OR dump also lacks SMILES (only InChIKey) and ~25% of receptor
sequences; both are backfilled online (`orbind/backfill.py`): InChIKey → PubChem,
Gene ID → UniProt. Both cache to `data/processed/`.

## Environment (uv only — no global installs)

The whole project runs through [uv](https://docs.astral.sh/uv/) with a local `.venv`:

```bash
uv sync                       # core (pandas, numpy)
uv sync --extra receptors     # + fair-esm, torch   (script 02)
uv sync --extra molecules     # + rdkit, torch, torch_geometric  (script 03)
```

## Usage  (always via `uv run`)

```bash
# 1. curate the table (offline; add --backfill-seq for the missing 25%)
uv run python scripts/01_build_table.py --input ../m2or_official/M2OR_20230428.csv

# 2. receptor embeddings (ESM-2)
uv run python scripts/02_embed_receptors.py --sequences data/processed/receptor_sequences.csv

# 3. molecule graphs (resolve SMILES first via backfill)
uv run python scripts/03_embed_molecules.py --molecules data/processed/molecule_smiles.csv
```

## Layout

```
orbind/
├── orbind/        filters.py (cleaning) · backfill.py (UniProt/PubChem)
├── scripts/       01_build_table · 02_embed_receptors · 03_embed_molecules
├── data/          raw / processed / embeddings   (gitignored)
└── config.yaml
```

## Open design choices

- Receptor key: `Gene ID` (full, needs seq backfill) vs `Sequence` (ESM-ready, fewer molecules).
- Molecule model: pretrained GIN (fixed vectors) vs graphs for end-to-end GCN.
- Splits: build **group-aware** (by receptor) to avoid the leakage LORAX's random CV has.
