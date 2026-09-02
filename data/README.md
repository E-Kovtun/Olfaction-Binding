# data/

**Not versioned.** Only this file, the per-folder READMEs and the `.gitkeep`
skeleton are tracked; everything else is distributed as a separate bundle.
Unpack it so that paths begin at the repository root — all code uses
repository-relative paths, never machine-specific absolute ones.

```text
data/
  raw/          upstream archives as downloaded
  processed/    our tables: pairs, sequences, SMILES, split indices
  embeddings/   cached protein and molecule vectors (.npz)
  external/     third-party releases kept in their own shape (OFM: Carey, Hallem)
  splits_indexes/  borrowed split definitions (LORAX's M2OR folds)
```

---

## Naming conventions

Embeddings are `<model>_<dataset>.npz`, where the dataset suffix is one of
`m2or` / `cc` / `hc` (absent on the oldest M2OR files, which predate the
convention). Two orthogonal markers:

* `_per_residue` / `_per_atom` — the un-pooled variant. Needed only by the
  methods that cross-attend over residues (`prosmith`, `lorax`, `molor`) or
  atoms; everything else uses the mean-pooled file.
* `_mean` — explicitly mean-pooled protein embedding.

Every `.npz` is a flat `{key: vector}` mapping. **Proteins are keyed by
amino-acid sequence, molecules by InChIKey** — not by any upstream id, which is
what lets the same file serve datasets that name their entities differently.

---

## What each file is

### `raw/`

| file | what |
|---|---|
| `M2OR.zip` | the full M2OR relational export, fetched by `scripts/downloading/00_download_m2or.py` |

### `processed/` — produced by `scripts/preprocessing/`

| file | produced by | used by |
|---|---|---|
| `pairs_curated.csv` | `01_build_table.py` | the `curated_full` regime; most notebooks |
| `pairs_m2or_full.csv` | `01_build_table.py` | the wider (uncurated) M2OR pool |
| `full_full_split_indices.npz` | `02_build_full_full_split_indices.py` | `orbind/regimes.py` — the `full_full` regime |
| `proteins/receptor_sequences.csv` | `01_build_table.py` | protein embedding scripts |
| `molecules/molecule_smiles.csv` | `01_build_table.py` | molecule embedding scripts |
| `molecules/molecule_smiles_{cc,hc}.csv` | `07_prepare_ofm_molecules.py` | SMILES→InChIKey bridge for the insect datasets (`orbind/regimes_ofm.py`) |
| `molecules/lorax_smiles_to_inchikey.csv` | `07_prepare_ofm_molecules.py` | reconciles LORAX's SMILES with our InChIKey keying |
| `bw_*_curated.csv` | `02_bw_numbering.py` | Ballesteros–Weinstein residue numbering; `bw_ref_used_curated.csv` labels receptors by OR family in `notebooks/legacy/refinement_geometry/` |

The curation filters (human only, no mutants, mono-molecular, binary response,
no orphan receptors) live in `orbind/filters.py`; `01_build_table.py` applies them.

### `embeddings/proteins/` — produced by `scripts/embedding_generation/proteins/`

| file | produced by | used by |
|---|---|---|
| `esm2_650m_mean.npz` | `02_embed_receptors.py` | the `curated_full` regime |
| `esm1b_650m_mean{,_cc,_hc}.npz` | `06_import_ofm_esm1b.py` | **the protein source of record** — ESM-1b keeps us comparable to ProSmith / LORAX / MolOR, which all use it |
| `esm1b_650m_per_residue_{full_full,cc,hc}.npz` | `06_import_ofm_esm1b.py` | `prosmith`, `lorax`, `molor` cross-attention |
| `esm2_650m_per_residue_full.npz` | `05_per_residue_embeddings.py` | per-residue analyses |
| other pLMs (ProtT5) and classical descriptors | `embed_proteins_plm.py` | the protein-source floor table (`prot_floor_sweep.py`) |

### `embeddings/molecules/` — produced by `scripts/embedding_generation/molecules/`

| file | produced by | used by |
|---|---|---|
| `chemberta_77m_{m2or,cc,hc}.npz` | `07_prepare_ofm_molecules.py` | the molecule source of record |
| `gin_supervised_contextpred_*{,_per_atom}.npz` | `embed_molecules_gin.py` | the GIN column of the molecule-source table; `_per_atom` for site-level heads |
| `ecfp_{m2or,cc,hc}.npz` | `embed_molecules_ecfp.py` | the ECFP column |
| `mol_graphs.pt` | `03_embed_molecules.py` | PyG molecule graphs |

`audit_molecule_npz.py` checks a molecule npz against a pairs table (coverage,
key collisions) before it is used in a run.

### `external/ofm/` — the olfactory-foundation-models release

From <https://zenodo.org/records/17228740>, kept in upstream's own shape rather
than folded into `processed/`:

```text
external/ofm/
  CC/   raw/CC_reformat_z.csv            Carey: mosquito AgOr, 50 x 110, z-scored
        rand_splits/  cdhit_splits/  scaf_splits/     upstream's 5-fold families
        our_inductive_splits/                          ours (see below)
        embeddings/featurized_{mols,proteins}          upstream's own features
  HC/   raw/hc_with_prot_seq_z.csv       Hallem-Carlson: fly, 24 x 110, z-scored
        rand_splits/  our_inductive_splits/            HC ships only `rand`
```

The target is a **continuous** z-scored response, not a 0/1 flag. `orbind/regimes_ofm.py`
puts it in `label` unchanged and the caller runs with `--task regression`.

`our_inductive_splits/` is ours, built by
`scripts/preprocessing/03_build_ofm_our_inductive_splits.py` — seedless and
deterministic, and the cold-molecule split of record for these two datasets.
Upstream's `scaf` is kept for the appendix comparison only; see the root README
for why it is not readable fold by fold.

### `splits_indexes/lorax_m2or/`

LORAX's own 5-fold partition of the M2OR pool, borrowed as-is to make our
`transductive` numbers directly comparable. See that folder's README for why it
is deliberately not renamed or merged into our conventions.

---

## Minimum set per experiment

| to run | you need |
|---|---|
| M2OR, any regime | `processed/pairs_curated.csv`, `splits_indexes/lorax_m2or/`, `embeddings/proteins/esm1b_650m_mean.npz`, one molecule npz |
| ProSmith / LORAX / MolOR on M2OR | the above **plus** `embeddings/proteins/esm1b_650m_per_residue_full_full.npz` (and, for ProSmith's published numbers, the BindingDB checkpoint under `external/ofm/saved_model/`) |
| Carey / Hallem | `external/ofm/{CC,HC}/`, `processed/molecules/molecule_smiles_{cc,hc}.csv`, `embeddings/*_{cc,hc}.npz` |
| protein-source floor | `processed/pairs_curated.csv` + whatever `embed_proteins_plm.py` was asked to cache |

A missing embedding fails loudly: extractors raise on an unknown key unless the
run passes `--on-missing drop`, which drops the uncovered pairs and logs how many.
