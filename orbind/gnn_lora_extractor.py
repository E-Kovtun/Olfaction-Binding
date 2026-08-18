"""Pair-level "cls" extractor for orbind.ensemble.run_ensemble: GnnLoraExtractor
-- our signed bipartite GNN (orbind.gnn_extractor) with the molecule node
features produced by a **live, LoRA-adapted ChemBERTa** (LORAX's molecule
encoder) instead of a frozen precomputed npz, trained END-TO-END with the graph.

Motivation
----------
The signed graph refines the *receptor* from its ligand profile, but the ligand
vectors it aggregates have so far been frozen (GIN / ChemBERTa mean-pooled). Here
the molecule node features come from the same LoRA-ChemBERTa LORAX fine-tunes
(orbind.lorax_extractor._build_lora_chemberta: base frozen, LoRA adapters on
query/key/value trainable), so the message passing and the molecule encoder learn
together. The protein side stays exactly as in our graph: frozen mean-pooled
ESM-1b node features, refined only by message passing. `emit="prot"` (default)
therefore hands the boosting stage the same shape it always did -- a graph-refined
receptor vector -- the only change is that the molecules feeding that refinement
are now learned.

What is trainable: the LoRA adapters (ChemBERTa base frozen) + the whole
_SignedSage (projections, four signed convs, decoder). One Adam over all of them.

Training protocol = our graph's, verbatim (the user's choice): full-batch,
lr=3e-3, 900 epochs, grad-clip 1.0, weight_decay 1e-4, NO scheduler / early stop,
the last epoch's weights are kept. One optimizer step per epoch (900 steps total).
Each step re-encodes every molecule through ChemBERTa (chunked and
gradient-checkpointed so the whole-universe forward fits), so the LoRA adapters see
the graph's gradient. n_models defaults to 1 (LoRA is heavy; bags run sequentially).

transformers + peft are imported lazily (via lorax_extractor), so this module
imports fine without them.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

from . import dataset as D
from . import mol_selection
from .tasks import check_task, batch_loss_fn
from .gnn_extractor import (MOL, PROT, _SignedSage, _mp_edges, _edge_index_dict,
                            _edge_weights, _LABEL_AGNOSTIC_CRITERIA)
from .lorax_extractor import _build_lora_chemberta, CHEMBERTA_CARD
from .prosmith_extractor import _M2ORWeights   # Hladis/M2OR sample weights (LORAX protocol)


# --------------------------------------------------------------------------- model

class _GnnLoraModel(nn.Module):
    """LoRA-ChemBERTa molecule encoder + signed bipartite GraphSAGE, one module so
    a single optimizer/backward covers both. `mol_model` is a peft LoRA ChemBERTa
    (base frozen, adapters trainable); `sage` is our _SignedSage over
    [molecule || protein] nodes."""

    def __init__(self, mol_model, mol_hidden: int, prot_dim: int, hidden: int,
                 dropout: float):
        super().__init__()
        self.mol_model = mol_model
        self.sage = _SignedSage(mol_hidden, prot_dim, hidden, dropout, weighted=False)

    def encode_mols(self, input_ids, attn, chunk: int) -> torch.Tensor:
        """Mean-pooled LoRA-ChemBERTa embedding for every molecule in the node
        universe -> [num_mols, mol_hidden]. Chunked to bound the attention
        footprint; each chunk is gradient-checkpointed while training so the
        whole-universe forward's activations are recomputed in backward rather
        than all held at once."""
        def run(ii, am):
            h = self.mol_model(input_ids=ii, attention_mask=am).last_hidden_state
            m = am.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / (m.sum(1) + 1e-8)
        outs = []
        use_ckpt = self.training and torch.is_grad_enabled()
        for i in range(0, input_ids.shape[0], chunk):
            ii, am = input_ids[i:i + chunk], attn[i:i + chunk]
            outs.append(_ckpt.checkpoint(run, ii, am, use_reentrant=False)
                        if use_ckpt else run(ii, am))
        return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------- data plumbing

def _universe(pairs: pd.DataFrame, all_idx, proteins: dict, ik2smiles: dict):
    """Local node-id maps + the SMILES list (molecule order) and the frozen
    mean-ESM protein feature tensor, over every entity in train/val/test."""
    sub = pairs.iloc[all_idx]
    mols = pd.unique(sub["inchikey"])
    prots = pd.unique(sub["receptor"])
    mol_to_i = {m: i for i, m in enumerate(mols)}
    prot_to_i = {p: i for i, p in enumerate(prots)}
    smiles = [ik2smiles[m] for m in mols]
    x_prot = torch.tensor(np.stack([proteins[p] for p in prots]), dtype=torch.float32)
    return mol_to_i, prot_to_i, smiles, x_prot


# --------------------------------------------------------------------------- training

def _train_one(ext, tokenizer, smiles, x_prot, pos_eidx, neg_eidx,
               tr, va, hp, device, checkpoint_path=None):
    """LORAX-protocol training of the graph+LoRA hybrid: the whole graph is still
    encoded every step (LoRA-ChemBERTa over all molecules + signed message
    passing), but supervision is MINIBATCHED over train pairs, the loss is the
    M2OR-weighted BCE (batch_loss_fn), and the best-validation-loss epoch's
    weights are kept -- exactly the schedule lorax_extractor/prosmith_extractor
    use, at their lr (1e-4), not the graph's full-batch/last-epoch one. `tr`/`va`
    are dicts of {mol,prot,y,w} index arrays for train/val. Only the trainable
    subset (LoRA adapters + sage) is (de)serialised."""
    torch.manual_seed(hp["seed"])
    model = ext._build_model().to(device)

    tok = tokenizer(list(smiles), padding=True, truncation=True,
                    max_length=hp["max_smiles_len"], return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attn = tok["attention_mask"].to(device)
    x_prot_d = x_prot.to(device)
    pos_d = {k: v.to(device) for k, v in pos_eidx.items()}
    neg_d = {k: v.to(device) for k, v in neg_eidx.items()}
    trainable_keys = set(n for n, p in model.named_parameters() if p.requires_grad)

    def encode():
        x_mol = model.encode_mols(input_ids, attn, hp["mol_chunk"])
        return model.sage.encode(x_mol, x_prot_d, pos_d, neg_d)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        model.eval()
        with torch.no_grad():
            z = encode()
            return z[MOL].cpu().numpy(), z[PROT].cpu().numpy(), model, trainable_keys

    def dev(a, dt=torch.long):
        return torch.as_tensor(a, dtype=dt, device=device)
    mi_tr, pi_tr = dev(tr["mol"]), dev(tr["prot"])
    yt, wt = dev(tr["y"], torch.float32), dev(tr["w"], torch.float32)
    mi_va, pi_va = dev(va["mol"]), dev(va["prot"])
    yv, wv = dev(va["y"], torch.float32), dev(va["w"], torch.float32)

    batch_loss = batch_loss_fn(hp["task"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

    n, bs = len(mi_tr), hp["batch_size"]
    best_val, best_state = float("inf"), None
    for ep in range(hp["epochs"]):
        model.train()
        perm = torch.randperm(n, device=device)
        run, seen = 0.0, 0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            z = encode()
            loss = batch_loss(model.sage.decode(z, mi_tr[b], pi_tr[b]), yt[b], wt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(params, hp["clip_grad"])
            opt.step()
            run += float(loss) * len(b); seen += len(b)
        model.eval()
        with torch.no_grad():
            z = encode()
            vloss = float(batch_loss(model.sage.decode(z, mi_va, pi_va), yv, wv)) if len(mi_va) else float("inf")
        is_best = vloss < best_val
        if is_best:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items() if k in trainable_keys}
        print(f"    gnn_lora ep {ep + 1}/{hp['epochs']}  train={run / max(seen, 1):.4f}  "
              f"val={vloss:.4f}{'  *best' if is_best else ''}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    with torch.no_grad():
        z = encode()
        return z[MOL].cpu().numpy(), z[PROT].cpu().numpy(), model, trainable_keys


def _run_models(ext, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
    check_task(ext.task)
    if ext.task == "regression" and ext.criterion not in _LABEL_AGNOSTIC_CRITERIA \
            and getattr(ext, "edge_threshold", None) is None:
        raise ValueError(f"{ext.name}: criterion {ext.criterion!r} needs a threshold on a "
                         f"continuous target; set edge_threshold or use "
                         f"{sorted(_LABEL_AGNOSTIC_CRITERIA)}")
    ext._ensure_loaded()
    ik2smiles = dict(zip(pairs["inchikey"], pairs["smiles"]))
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            raise KeyError(f"{ext.name}: {int((~mask).sum())}/{len(idx)} {split_name} rows "
                           f"missing a protein embedding or SMILES")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ext._load_tokenizer()
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    mol_to_i, prot_to_i, smiles, x_prot = _universe(pairs, all_idx, ext._proteins, ik2smiles)

    # M2OR sample weights (quality x imbalance, Hladis eq.) -- LORAX protocol; None off M2OR.
    W = _M2ORWeights.maybe_build(pairs)
    def pack(idx):
        sub = pairs.iloc[idx]
        return {"mol": sub["inchikey"].map(mol_to_i).to_numpy(),
                "prot": sub["receptor"].map(prot_to_i).to_numpy(),
                "y": sub["label"].to_numpy(dtype=np.float32),
                "w": (np.ones(len(idx), np.float32) if W is None else W.per_row[idx].astype(np.float32))}
    tr, va = pack(train_idx), pack(val_idx)

    pos, neg = _mp_edges(pairs, train_idx, mol_to_i, prot_to_i, ext.q, ext.criterion,
                         ext.edge_threshold, ext.k_mode, ext.task)
    n_pos, n_neg = len(pos[0]), len(neg[0])
    print(f"  {ext.name}: MP graph {n_pos} positive / {n_neg} negative edges "
          f"(q={ext.q}, criterion={ext.criterion}, mols={len(smiles)}, "
          f"lora_r={ext.lora_r}, lr={ext.lr}, epochs={ext.epochs}, batch={ext.batch_size})", flush=True)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{ext.name}: signed MP needs both edge signs, got "
                         f"{n_pos} positive / {n_neg} negative.")
    pos_eidx, neg_eidx = _edge_index_dict(pos, neg)

    results = []
    for m in range(ext.n_models):
        hp = ext._hp(seed + ext.seed_offset * m)
        ckpt = (pathlib.Path(checkpoint_dir) / f"gnnlora_{ext.name}_model{m}.pt"
                if checkpoint_dir is not None else None)
        z_mol, z_prot, model, trainable_keys = _train_one(
            ext, tokenizer, smiles, x_prot, pos_eidx, neg_eidx, tr, va, hp, device,
            checkpoint_path=ckpt)
        if ckpt is not None and not ckpt.exists():
            torch.save({k: v for k, v in model.state_dict().items() if k in trainable_keys}, ckpt)
        results.append((m, z_mol, z_prot))

    def features_for(idx):
        sub = pairs.iloc[idx]
        mi = sub["inchikey"].map(mol_to_i).to_numpy()
        pi = sub["receptor"].map(prot_to_i).to_numpy()
        if ext.emit == "prot":
            blocks = [z_prot[pi] for _, _, z_prot in results]
        else:
            blocks = [np.concatenate([z_mol[mi], z_prot[pi]], axis=1) for _, z_mol, z_prot in results]
        return np.concatenate(blocks, axis=1).astype(np.float32)

    return features_for(train_idx), features_for(val_idx), features_for(test_idx)


# --------------------------------------------------------------------------- extractor

@dataclass
class GnnLoraExtractor:
    """Signed bipartite GNN whose molecule node features come from a live,
    LoRA-adapted ChemBERTa (LORAX's molecule encoder), trained end-to-end with the
    graph. Protein node features stay frozen mean-pooled ESM-1b, refined only by
    message passing -- so `emit="prot"` hands the boosting stage the same
    graph-refined receptor shape as GnnSignedExtractor, the difference being that
    the aggregated molecules are learned.

    Graph knobs (q / criterion / edge_threshold / k_mode / emit / hidden) match
    GnnSignedExtractor. LoRA knobs (lora_r / lora_alpha / lora_dropout /
    chemberta_card) match LoraxExtractor's M2OR config. Training protocol is the
    graph's: full-batch, lr=3e-3, epochs=900, grad-clip 1.0, last-epoch weights.

    Needs transformers + peft in the run env (imported lazily)."""

    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_mean.npz"
    chemberta_card: str = CHEMBERTA_CARD
    n_models: int = 1
    emit: str = "prot"
    hidden: int = 256
    dropout: float = 0.3
    q: float = 0.99
    criterion: str = "greedy_pair_cover"
    edge_threshold: float = 0.0
    k_mode: str = "coverage_quantile"
    # LoRA (LORAX M2OR config: r=8/alpha=8 on query/key/value, dropout 0.1)
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    mol_hidden: int = 384
    # LORAX training protocol: low lr, minibatched M2OR-weighted BCE, best-val
    # checkpoint. Full graph still encoded every step; only supervision is batched.
    lr: float = 1e-4
    weight_decay: float = 0.0
    clip_grad: float = 1.0
    epochs: int = 15
    batch_size: int = 256
    max_smiles_len: int = 256
    mol_chunk: int = 128          # molecules per checkpointed ChemBERTa forward
    seed_offset: int = 5000
    task: str = "classification"
    pooling: str = "signed_sage_lora"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="gnn_lora")

    def __post_init__(self):
        if self.emit not in ("prot", "both"):
            raise ValueError(f"emit must be 'prot' or 'both', got {self.emit!r}")
        if self.criterion not in mol_selection.CRITERIA:
            raise ValueError(f"criterion must be one of {mol_selection.CRITERIA}")
        if self.k_mode not in mol_selection.K_MODES:
            raise ValueError(f"k_mode must be one of {mol_selection.K_MODES}")
        per_model = self.hidden if self.emit == "prot" else 2 * self.hidden
        self.dim_out = self.n_models * per_model
        self.path = f"{self.protein_path} + LoRA({self.chemberta_card})"
        self._proteins = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._proteins is None:
            self._proteins = D.load_npz_dict(self.protein_path)

    def _load_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.chemberta_card)
        return self._tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_proteins"] = None
        state["_tokenizer"] = None
        return state

    def _build_model(self):
        mol_model, hidden = _build_lora_chemberta(self.chemberta_card, self.lora_r,
                                                  self.lora_alpha, self.lora_dropout)
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        self.mol_hidden = hidden
        return _GnnLoraModel(mol_model, hidden, prot_dim, self.hidden, self.dropout)

    def _hp(self, seed):
        return dict(lr=self.lr, weight_decay=self.weight_decay, clip_grad=self.clip_grad,
                    epochs=self.epochs, batch_size=self.batch_size, task=self.task, seed=seed,
                    max_smiles_len=self.max_smiles_len, mol_chunk=self.mol_chunk)

    def covered(self, pairs, idx):
        self._ensure_loaded()
        prot = pairs["receptor"].to_numpy()[idx]
        smi = pairs["smiles"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and isinstance(s, str) and len(s) > 0
                            for p, s in zip(prot, smi)), dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
