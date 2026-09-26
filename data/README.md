# data/

**Mostly not versioned.** Tracked are this file, the per-folder READMEs, the
`.gitkeep` skeleton and **the held-out splits** every result stands on:
`splits_indexes/lorax_m2or/`, `processed/full_full_split_indices.npz`, and
`external/ofm/{CC,HC}/{rand_splits,our_inductive_splits}/`. Everything else (the
benchmark release, the embeddings) is produced by the commands in
[`README3.md`](../README3.md) §2. All code uses repository-relative paths.

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
| `molecules/molecule_smiles_{m2or,cc,hc}.csv` | `07_prepare_ofm_molecules.py` | SMILES→InChIKey table of each dataset's molecules (M2OR: the 596 of the LORAX pool); the input of the GIN embedder, and the insects' bridge in `orbind/regimes_ofm.py` |
| `molecules/lorax_smiles_to_inchikey.csv` | `07_prepare_ofm_molecules.py` | reconciles LORAX's SMILES with our InChIKey keying |
| `bw_*_curated.csv` | `02_bw_numbering.py` | Ballesteros–Weinstein residue numbering; `bw_ref_used_curated.csv` labels receptors by OR family in `notebooks/legacy/refinement_geometry/` |

The curation filters (human only, no mutants, mono-molecular, binary response,
no orphan receptors) live in `orbind/filters.py`; `01_build_table.py` applies them.

### `embeddings/proteins/` — produced by `scripts/embedding_generation/proteins/`

| file | produced by | used by |
|---|---|---|
| `esm3_{m2or,cc,hc}.npz` | `embed_proteins_plm.py --model esm3 --per-residue` | **the protein embedding of record** — every table in the paper: OlfaGraph's receptor nodes, XGBoost-base, Hladiš |
| `esm3_per_residue_{m2or,cc,hc}.npz` | the same command | LORAX, ProSmith and MolOR cross-attention |
| `prott5_{m2or,cc,hc}.npz` | `embed_proteins_plm.py --model prott5` | the receptor-representation table (ProtT5 row and ProtT5-initialised OlfaGraph) |
| `esm1b_650m_mean_{full_full,cc,hc}.npz` | `06_import_ofm_esm1b.py` | the receptor-representation table (ESM-1b row). `full_full` is M2OR |
| `esm1b_650m_per_residue_{full_full,cc,hc}.npz` | `06_import_ofm_esm1b.py` | the earlier ESM-1b baseline series only |
| `esm1b_650m_mean.npz` | LORAX's own file | superseded by `esm1b_650m_mean_full_full.npz`, which `06` checks against it |
| `esm2_650m_*` | `scripts/legacy/` | not in the paper |

Classical sequence descriptors (AAC, k-mer, CTD, PseAAC, BLOSUM, AAindex) need no file:
`prot_floor_sweep.py` computes them from the sequence.

### `embeddings/molecules/` — produced by `scripts/embedding_generation/molecules/`

| file | produced by | used by |
|---|---|---|
| `chemberta_77m_{m2or,cc,hc}.npz` | `07_prepare_ofm_molecules.py`, imported from the release | **the molecular embedding of record** |
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
  M2OR*/ embeddings/featurized_{mols,proteins}         M2OR's features (ChemBERTa, ESM-1b)
  saved_model/                           the BindingDB-pretrained ProSmith checkpoint
```

`rand_splits/` and `our_inductive_splits/` of CC and HC are versioned with the
repository; the rest arrives with the release.

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

The paper's full list, with the commands that produce each file, is in
[`README3.md`](../README3.md) §2. In short:

| to run | you need |
|---|---|
| anything on M2OR | `splits_indexes/lorax_m2or/`, `processed/full_full_split_indices.npz`, `embeddings/proteins/esm3_m2or.npz`, `embeddings/molecules/chemberta_77m_m2or.npz` |
| anything on Mosquito / Fly | `external/ofm/{CC,HC}/` (raw table and splits), `processed/molecules/molecule_smiles_{cc,hc}.csv`, `embeddings/proteins/esm3_{cc,hc}.npz`, `embeddings/molecules/chemberta_77m_{cc,hc}.npz` |
| LORAX / ProSmith / MolOR | the above **plus** `embeddings/proteins/esm3_per_residue_{ds}.npz` (and, for ProSmith, upstream's pretrained checkpoint under `external/ofm/saved_model/`) |
| the receptor-representation table | the above **plus** `prott5_{ds}.npz` and `esm1b_650m_mean_{full_full,cc,hc}.npz` |
| the molecular-representation appendix | the above **plus** the GIN and ECFP files |

A missing embedding fails loudly: extractors raise on an unknown key unless the
run passes `--on-missing drop`, which drops the uncovered pairs and logs how many.
