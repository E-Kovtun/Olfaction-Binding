# orbind  *(template name — rename freely)*

Curated **olfactory receptor ⇄ ligand binding** dataset + embedding pipeline,
built on the **full official M2OR** database export (Lalis et al. 2024).

## Pipeline

```
M2OR.zip ──00 download──▶ relational export (pairs/compounds/receptors, SMILES inline)
         ──01 curate────▶ pairs_curated.csv (receptor, molecule, label, SMILES)
                          receptor_sequences.csv ──02──▶ esm2_650m_mean_curated.npz (1280-d/receptor)
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
| `03_embed_molecules.py`  | PyG graphs (RDKit) | featurize SMILES; embeddings from a GCN you train (`--checkpoint`) |
| `embed_molecules_gin.py` | pretrained GIN, 300-d | LORAX's `gin_supervised_*` (Hu 2020) via dgllife — fixed vectors |

> The GIN path pins **torch 2.2 + dgl 2.2.1** (dgl ships graphbolt only for torch ≤2.2
> and needs the old `setuptools.extern`); these constraints live in `pyproject.toml`.

## Environment — uv only (local `.venv`, no global installs)

```bash
uv sync            # installs everything into ./.venv (pandas, torch, fair-esm, rdkit, torch_geometric)
```

## Usage  (always via `uv run`)

```bash
uv run python scripts/downloading/00_download_m2or.py            # -> data/raw/M2OR.zip
uv run python scripts/preprocessing/01_build_table.py            # -> data/processed/pairs_curated.csv (+SMILES)
uv run python scripts/embedding_generation/proteins/02_embed_receptors.py \
       --sequences data/processed/proteins/receptor_sequences.csv
uv run python scripts/embedding_generation/molecules/03_embed_molecules.py \
       --molecules data/processed/molecules/molecule_smiles.csv

# MP baseline: concat[molecule || protein] -> head (XGBoost | MLP) -> bind/no-bind
uv run python scripts/modeling/train/train_mp.py --split group_molecule   # our main mode
```

Goal: a competent **protein–molecule interaction** model for olfaction — not a single
target number. The reference baseline is **XGBoost** over `[GIN molecule ‖ ESM-2 mean
receptor]` (an MLP head also exists but is less maintained). Because the human OR
repertoire is **fixed and known** (~400 receptors), the relevant evaluation is
matrix-completion over known receptors: **group_molecule** (new odorants — our main
bet) and **stratified** (matrix fill, often used by other papers). The cold-**receptor**
split (`group_receptor`) is **de-emphasized** — generalizing to unseen receptors is not
really the olfactory task. Class imbalance (~1:11) is handled by
`scale_pos_weight` / `pos_weight`; metrics are imbalance-aware (AUROC, AUPRC, MCC, F1).

## Layout

```
orbind/
├── orbind/                 filters.py (relational M2OR cleaning)
├── scripts/
│   ├── downloading/        00_download_m2or.py
│   ├── preprocessing/      01_build_table.py
│   └── embedding_generation/
│       ├── proteins/       02_embed_receptors.py
│       └── molecules/      03_embed_molecules.py
├── data/                   (gitignored content; structure kept via .gitkeep)
│   ├── raw/                {proteins, molecules}   + M2OR.zip
│   ├── processed/          {proteins, molecules}   + pairs_curated.csv
│   └── embeddings/         {proteins, molecules}
├── pyproject.toml · config.yaml
```

## Open design choices

- Mixture policy: `mono` (strict, pure substances) vs `mono+isomers`.
- Molecule model: graphs for an end-to-end GCN (no canonical pretrained GCN here).
- Splits: **group_molecule** (new odorants, known receptors) is the practical target;
  **stratified** for matrix completion; **group_receptor** de-emphasized (OR repertoire
  is fixed/known, so cold-receptor isn't really the olfactory task).
