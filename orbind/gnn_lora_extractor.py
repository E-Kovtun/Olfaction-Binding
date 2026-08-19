"""Pair-level "cls" extractor for orbind.ensemble.run_ensemble: GnnLoraExtractor
-- our signed bipartite GNN (orbind.gnn_extractor) whose molecule node features
come from a ChemBERTa the LORAX way, but in TWO DECOUPLED STAGES rather than
one joint end-to-end optimisation.

Two stages, one extractor
-------------------------
Stage 1 -- fine-tune the molecule encoder EXACTLY as LORAX does. The full LORAX
    model (orbind.lorax_extractor._LoraxMM: LoRA-ChemBERTa molecule branch,
    frozen per-residue ESM-1b protein branch, cross-attention, weighted-BCE
    head) is trained on the train split with LORAX's own protocol -- lr 1e-4,
    15 epochs, batch 21, M2OR/Hladis sample weights, best-validation-loss
    checkpoint. This is literally `lorax_extractor._train_one`, so "the LoRA is
    trained the way it is in LORAX" holds by construction, not by imitation.

    The encoder is then FROZEN and run over every molecule in the universe
    (train+val+test) to produce ONE static 384-d vector per molecule -- the
    masked mean of LoRA-ChemBERTa's `last_hidden_state`, WITHOUT cross-attention,
    so the vector depends only on the molecule (a proper node feature, not a
    pair feature). These replace the frozen GIN/ChemBERTa npz the plain graph
    reads.

Stage 2 -- run the signed bipartite graph EXACTLY as GnnSignedExtractor does,
    on those static molecule features + frozen mean-pooled ESM-1b protein
    features, with OUR graph protocol (full-batch, lr 3e-3, 900 epochs,
    grad-clip 1.0, weight_decay 1e-4, last-epoch weights). This is a verbatim
    delegation to `gnn_extractor._run_models`, so `emit="prot"` hands the
    boosting stage the same graph-refined receptor shape as the plain graph --
    the only difference from GnnSignedExtractor being that the aggregated
    molecule vectors were fine-tuned by LORAX first instead of pretrained GIN.

So the two training protocols never mix: LoRA gets LORAX's, the graph gets ours.
The molecule encoder is trained once, frozen, and its outputs are cached; the
graph then trains (optionally bagged `n_models` times) on top.

transformers + peft are imported lazily (via lorax_extractor), so this module
imports fine without them.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from . import dataset as D
from . import mol_selection
from .tasks import check_task
from .gnn_extractor import _SignedSage, _run_models as _graph_run_models
from .lorax_extractor import (_LoraxMM, _build_lora_chemberta,
                              _train_one as _lorax_train_one, CHEMBERTA_CARD)
from .prosmith_extractor import _M2ORWeights   # Hladis/M2OR sample weights (LORAX protocol)


# --------------------------------------------------------------------------- stage 1 helpers

@torch.inference_mode()
def _static_mol_embeddings(mol_model, tokenizer, smiles, device, max_len, chunk):
    """Freeze the fine-tuned LoRA-ChemBERTa and read one static vector per
    molecule: the masked mean of `last_hidden_state`, with NO cross-attention,
    so the result is a pure molecule node feature (protein-independent)."""
    mol_model.eval()
    tok = tokenizer(list(smiles), padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    ii, am = tok["input_ids"].to(device), tok["attention_mask"].to(device)
    outs = []
    for i in range(0, len(smiles), chunk):
        h = mol_model(input_ids=ii[i:i + chunk],
                      attention_mask=am[i:i + chunk]).last_hidden_state
        m = am[i:i + chunk].unsqueeze(-1).to(h.dtype)
        outs.append(((h * m).sum(1) / (m.sum(1) + 1e-8)).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def _run_models(ext, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
    check_task(ext.task)
    ext._ensure_loaded()

    # Stage-1 coverage: LORAX needs the per-residue ESM matrix + a SMILES string.
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext._covered_stage1(pairs, idx)
        if not mask.all():
            raise KeyError(f"{ext.name}: {int((~mask).sum())}/{len(idx)} {split_name} rows "
                           f"missing a per-residue protein embedding or SMILES "
                           f"(lora_protein={ext.lora_protein_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ik2smiles = dict(zip(pairs["inchikey"], pairs["smiles"]))

    # ------------------------------------------------------------------ STAGE 1
    # Fine-tune the LoRA molecule encoder the LORAX way (its model, its protocol).
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]
    W = _M2ORWeights.maybe_build(pairs)
    w_tr = None if W is None else W.per_row[train_idx]
    w_va = None if W is None else W.per_row[val_idx]
    w_te = None if W is None else W.per_row[test_idx]
    hp1 = ext._lorax_hp(seed)
    ckpt1 = (pathlib.Path(checkpoint_dir) / f"gnnlora_stage1_{ext.name}.pt"
             if checkpoint_dir is not None else None)
    print(f"  {ext.name}: STAGE 1 -- LORAX fine-tune (lr={ext.lora_lr}, epochs={ext.lora_epochs}, "
          f"batch={ext.lora_batch}, lora_r={ext.lora_r}, per-residue prot={ext.lora_protein_path})",
          flush=True)
    _, _, lorax_model = _lorax_train_one(
        ext._build_lorax_stage1, ext._load_tokenizer, train_df, val_df, test_df,
        ext._lora_proteins, w_tr, w_va, w_te, hp1, device, checkpoint_path=ckpt1)
    if ckpt1 is not None and not ckpt1.exists():
        torch.save(lorax_model.state_dict(), ckpt1)

    # Freeze -> static per-molecule features over the whole universe.
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    mols = pd.unique(pairs.iloc[all_idx]["inchikey"])
    smiles = [ik2smiles[m] for m in mols]
    embs = _static_mol_embeddings(lorax_model.mol_model, ext._load_tokenizer(),
                                  smiles, device, ext.max_smiles_len, ext.mol_chunk)
    ext._molecules = {m: embs[i] for i, m in enumerate(mols)}
    ext.mol_hidden = int(embs.shape[1])
    print(f"  {ext.name}: STAGE 1 done -- {len(mols)} static molecule vectors "
          f"(dim {ext.mol_hidden}); freezing encoder", flush=True)
    del lorax_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ STAGE 2
    # Signed graph on the frozen LoRA molecule features, OUR protocol -- verbatim
    # delegation to the plain graph's runner (it reads ext._molecules/_proteins,
    # builds the MP graph, trains _SignedSage at lr=3e-3/900ep, emits `emit`).
    print(f"  {ext.name}: STAGE 2 -- signed graph on LoRA features "
          f"(lr={ext.lr}, epochs={ext.epochs}, q={ext.q}, criterion={ext.criterion}, "
          f"emit={ext.emit}, n_models={ext.n_models})", flush=True)
    return _graph_run_models(ext, pairs, train_idx, val_idx, test_idx, seed,
                             checkpoint_dir=checkpoint_dir)


# --------------------------------------------------------------------------- extractor

@dataclass
class GnnLoraExtractor:
    """Two-stage signed-graph source: (1) fine-tune LoRA-ChemBERTa the LORAX way,
    freeze it, cache one static vector per molecule; (2) run our signed bipartite
    graph on those molecule features + frozen mean-pooled ESM-1b proteins, our
    protocol. `emit="prot"` therefore hands the boosting stage the same
    graph-refined receptor shape as GnnSignedExtractor -- the difference is that
    the aggregated molecules were LORAX-fine-tuned rather than pretrained GIN.

    Graph knobs (q / criterion / edge_threshold / k_mode / emit / hidden /
    n_models) and the graph training protocol (lr / weight_decay / clip_grad /
    epochs) match GnnSignedExtractor. LoRA knobs (lora_r / lora_alpha /
    lora_dropout / chemberta_card / lora_protein_path) and the LoRA training
    protocol (lora_lr / lora_epochs / lora_batch) match LoraxExtractor's M2OR
    config. The two protocols are applied to their own stage only.

    `protein_path`      -- MEAN-pooled ESM npz, the graph's protein node features.
    `lora_protein_path` -- PER-RESIDUE ESM npz, LORAX's cross-attention key/value.

    Needs transformers + peft in the run env (imported lazily)."""

    name: str
    # stage-2 graph protein node features (mean-pooled ESM-1b)
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    # stage-1 LORAX cross-attention key/value (per-residue ESM-1b)
    lora_protein_path: str = "data/embeddings/proteins/esm1b_650m_per_residue_full_full.npz"
    chemberta_card: str = CHEMBERTA_CARD
    n_models: int = 1
    emit: str = "prot"
    hidden: int = 256
    dropout: float = 0.3
    q: float = 0.99
    criterion: str = "greedy_pair_cover"
    edge_threshold: float = 0.0
    k_mode: str = "coverage_quantile"
    # EXPERIMENTAL continuous-label edge modes (off by default; passed through to
    # the graph stage -- see GnnSignedExtractor). Binary M2OR is unchanged.
    edge_center: str = "global"
    edge_weight_mode: str = "none"
    # LoRA (LORAX M2OR config: r=8/alpha=8 on query/key/value, dropout 0.1)
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    num_heads: int = 8
    lin_proj: bool = False          # config_m2or: MLP head
    mlp_hidden: int = 512
    mol_hidden: int = 384           # ChemBERTa-77M hidden width (set for real after stage 1)
    # dummy_compression: freeze the graph's input projections to a train-fit PCA.
    # Left in for parity with GnnSignedExtractor; not recommended (it cost ~0.003
    # AUROC on the plain graph). Default off.
    dummy_compression: bool = False
    # --- stage-2 graph training protocol (OURS: full-batch, last-epoch) ---
    lr: float = 3e-3
    weight_decay: float = 1e-4
    clip_grad: float = 1.0
    epochs: int = 900
    # --- stage-1 LoRA training protocol (LORAX M2OR config) ---
    lora_lr: float = 1e-4
    lora_epochs: int = 15
    lora_batch: int = 21
    max_smiles_len: int = 256
    mol_chunk: int = 128            # molecules per forward when caching static features
    seed_offset: int = 5000
    # graph auxiliary (DeepGraphInfomax) -- off; passed through to the graph stage.
    dgi_weight: float = 0.0
    dgi_scope: str = "shared"
    task: str = "classification"
    pooling: str = "signed_sage_lora2stage"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="gnn_lora")

    def __post_init__(self):
        if self.emit not in ("prot", "both"):
            raise ValueError(f"emit must be 'prot' or 'both', got {self.emit!r}")
        if self.criterion not in mol_selection.CRITERIA:
            raise ValueError(f"criterion must be one of {mol_selection.CRITERIA}")
        if self.k_mode not in mol_selection.K_MODES:
            raise ValueError(f"k_mode must be one of {mol_selection.K_MODES}")
        if self.edge_center not in ("global", "per_receptor"):
            raise ValueError(f"edge_center must be 'global' or 'per_receptor', got {self.edge_center!r}")
        if self.edge_weight_mode not in ("none", "magnitude"):
            raise ValueError(f"edge_weight_mode must be 'none' or 'magnitude', got {self.edge_weight_mode!r}")
        per_model = self.hidden if self.emit == "prot" else 2 * self.hidden
        self.dim_out = self.n_models * per_model
        self.path = f"{self.protein_path} + LoRA({self.chemberta_card})"
        # `molecule_path` is referenced only in the graph runner's coverage-error
        # message; give it a descriptive value so that branch never AttributeErrors.
        self.molecule_path = f"LoRA({self.chemberta_card}) static features"
        self._proteins = None          # mean-pooled ESM (graph node features)
        self._lora_proteins = None     # per-residue ESM (LORAX cross-attn kv)
        self._molecules = None         # filled after stage 1 with static LoRA vectors
        self._tokenizer = None
        self._pca_mol = None
        self._pca_prot = None

    # ---- loading -----------------------------------------------------------
    def _ensure_loaded(self):
        if self._proteins is None:
            self._proteins = D.load_npz_dict(self.protein_path)
        if self._lora_proteins is None:
            self._lora_proteins = D.load_npz_dict(self.lora_protein_path)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.chemberta_card)
        return self._tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_proteins"] = None
        state["_lora_proteins"] = None
        state["_molecules"] = None
        state["_tokenizer"] = None
        return state

    # ---- stage 1 (LORAX) ---------------------------------------------------
    def _build_lorax_stage1(self):
        mol_model, hidden = _build_lora_chemberta(self.chemberta_card, self.lora_r,
                                                  self.lora_alpha, self.lora_dropout)
        prot_dim = next(iter(self._lora_proteins.values())).shape[-1]
        self.mol_hidden = hidden
        return _LoraxMM(mol_model, hidden, prot_dim, self.num_heads,
                        self.lora_dropout, self.lin_proj, self.mlp_hidden)

    def _lorax_hp(self, seed):
        return dict(lr=self.lora_lr, epochs=self.lora_epochs, batch_size=self.lora_batch,
                    max_smiles_len=self.max_smiles_len, task=self.task, seed=seed)

    def _covered_stage1(self, pairs, idx):
        self._ensure_loaded()
        prot = pairs["receptor"].to_numpy()[idx]
        smi = pairs["smiles"].to_numpy()[idx]
        return np.fromiter(((p in self._lora_proteins) and isinstance(s, str) and len(s) > 0
                            for p, s in zip(prot, smi)), dtype=bool, count=len(idx))

    # ---- stage 2 (graph) -- consumed by gnn_extractor._run_models ----------
    def _build_model(self):
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _SignedSage(self.mol_hidden, prot_dim, self.hidden, self.dropout,
                           weighted=(self.edge_weight_mode != "none"),
                           pca_mol=self._pca_mol, pca_prot=self._pca_prot)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    epochs=self.epochs, task=self.task, seed=seed)

    def covered(self, pairs, idx):
        # A row is usable iff both ESM views (mean for the graph, per-residue for
        # LORAX) have its receptor AND it has a SMILES -- the static molecule
        # feature is then DERIVED, so it is not a separate coverage condition.
        # Must not reference `_molecules`: the ensemble intersects coverage UP
        # FRONT (ensemble.py, on_missing="drop") to pick the drop mask, before
        # stage 1 has produced any static vectors -- checking `_molecules` there
        # marks every row uncovered and empties the train split.
        self._ensure_loaded()
        prot = pairs["receptor"].to_numpy()[idx]
        smi = pairs["smiles"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (p in self._lora_proteins)
                            and isinstance(s, str) and len(s) > 0
                            for p, s in zip(prot, smi)), dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
