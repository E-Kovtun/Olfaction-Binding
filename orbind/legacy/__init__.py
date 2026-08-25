"""Archived library modules, kept importable so archived scripts and notebooks still run.

Nothing in the live pipeline imports from here. These four modules belong to the
standalone graph line that `orbind.gnn_extractor` (reached through the ensembler)
replaced:

* `hetero`     -- the heterogeneous bipartite molecule<->protein graph and its GraphSAGE
                  link predictor; the direct ancestor of `_SignedSage`.
* `hetero_gat` -- the GAT variant, from when the architecture was still undecided.
* `lorax`      -- our re-derivation of LORAX's cold-molecule split; `orbind.regimes.
                  inductive_molecule_v5_indices` is defined to reproduce it position for
                  position, so this file is the reference for that definition.
* `attention`  -- the site-MIL / attention head of the interaction branch.

Their consumers live under the repository-level `legacy/` tree. Code stays here rather
than in `legacy/` because it must remain on the import path for those consumers to work.
"""
