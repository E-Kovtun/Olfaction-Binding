"""Pair-level "cls"-style extractor for orbind.ensemble.run_ensemble:
LoraxExtractor -- a reimplementation of LORAX (McConachie et al., ICLR 2026,
"LoRA-based Odorant-Receptor Affinity prediction with cross-attention"), in the
same spirit as prosmith_extractor.py: model definition inline and directly
editable, only this project's own generic plumbing reused.

What upstream LORAX does (lorax/model/lorax.py)
-----------------------------------------------
Both modalities are run through their *live* foundation models wrapped in LoRA
adapters (peft), cross-attended, mean-pooled, concatenated, and projected to a
logit. The feature its second stage (`scripts/train_GB.py`) hands to XGBoost is
`cat_rep` -- the concatenation of the two mean-pooled, cross-attended
representations.

This project's variant (two deliberate scope choices, see below)
----------------------------------------------------------------
1. **LoRA on the molecule side only.** The ChemBERTa-77M encoder is loaded live
   and adapted with LoRA (trainable); the protein side is NOT fine-tuned.
2. **Protein = frozen precomputed ESM-1b, per-residue.** Because the protein
   encoder is frozen, there is no reason to run the 650M ESM live -- we feed the
   same per-residue ESM-1b matrices ProSmith already uses
   (`esm1b_650m_per_residue_full_full.npz`) straight in as the cross-attention
   key/value. This is what makes the source directly comparable to our ProSmith
   baseline: **both sit on ESM-1b**, so the delta measures the architecture
   (LoRA molecule encoder + cross-attention), not a 1b->2 protein-encoder swap.
   (LORAX's own config uses ESM-2; mixing that with a ProSmith baseline on
   ESM-1b would confound the two changes.)

So only ChemBERTa is live here; ESM is a frozen npz lookup, exactly like
ProSmith's protein input. The rest -- Hladis/M2OR sample weights, the
train/best-val/checkpoint-resume protocol, n_models bagging, and the ensemble
wiring -- is shared with prosmith_extractor.py.

Forward (mirrors lorax/model/lorax.py, protein branch replaced by the frozen
per-residue matrix):

    smi  = LoRA-ChemBERTa(smiles_tokens).last_hidden_state        # [B, Ls, 384]
    prot = frozen_esm1b_per_residue                               # [B, Lp, 1280]
    smi' = LN(smi  + MHA(q=smi,  k=v=prot, mask=prot_pad))        # cross-attn
    prot'= LN(prot + MHA(q=prot, k=v=smi,  mask=smi_pad))
    cat  = [ masked_mean(smi') || masked_mean(prot') ]            # [B, 1664]
    logit = proj(cat)                                             # training head
    -> cat is the feature handed to the boosting stage.

transformers + peft are imported lazily (only when a model is actually built),
so this module imports fine on a box that lacks them.
"""
from __future__ import annotations

import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import dataset as D
from .prosmith_extractor import _M2ORWeights   # identical Hladis/M2OR weighting

_INIT_LOCK = threading.Lock()   # see _train_one (global manual_seed under threads)

CHEMBERTA_CARD = "DeepChem/ChemBERTa-77M-MTR"


# --------------------------------------------------------------------------- data plumbing

class _LoraxDataset(Dataset):
    """(raw SMILES string, per-residue protein matrix, label, weight).

    The molecule is the raw SMILES (tokenized in the collate by ChemBERTa's own
    tokenizer); the protein is the frozen per-residue ESM-1b matrix."""

    def __init__(self, pairs: pd.DataFrame, proteins: dict, weights):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins
        self.weights = weights

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        return (str(row.smiles),
                np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.float32(row.label),
                np.float32(1.0 if self.weights is None else self.weights[idx]))


def _make_collate(tokenizer, max_smiles_len: int):
    """Tokenize the SMILES batch (pad to batch max) and pad the per-residue
    protein matrices to the batch max. `prot_pad` is True where a slot is
    padding (bool key-padding mask, i.e. "ignore")."""
    def collate(batch):
        smiles, prots, labels, weights = zip(*batch)
        tok = tokenizer(list(smiles), padding=True, truncation=True,
                        max_length=max_smiles_len, return_tensors="pt")
        lengths = [len(p) for p in prots]
        width = max(lengths)
        prot_dim = prots[0].shape[-1]
        x = torch.zeros(len(prots), width, prot_dim, dtype=torch.float32)
        pad = torch.ones(len(prots), width, dtype=torch.bool)
        for i, p in enumerate(prots):
            n = len(p)
            x[i, :n] = torch.from_numpy(p)
            pad[i, :n] = False
        return (dict(tok), x, pad,
                torch.tensor(labels, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32))
    return collate


def _make_loader(df, proteins, weights, tokenizer, max_smiles_len, batch_size, train):
    ds = _LoraxDataset(df, proteins, weights)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      collate_fn=_make_collate(tokenizer, max_smiles_len))


def _to_device(tok: dict, device):
    return {k: v.to(device) for k, v in tok.items()}


@torch.inference_mode()
def _embed(model, loader, device):
    """cat_rep after the last cross-attention -- LORAX's own boosting feature."""
    model.eval()
    out = []
    for smi_tok, prot, pad, _, _ in loader:
        _, rep = model(_to_device(smi_tok, device), prot.to(device), pad.to(device), get_repr=True)
        out.append(rep.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- model

class _LoraxMM(nn.Module):
    """LORAX with LoRA on the molecule encoder only and a frozen precomputed
    per-residue protein matrix as the cross-attention key/value.

    `mol_model` is a live ChemBERTa already wrapped in peft LoRA (base frozen,
    adapters trainable). `prot_dim` is the per-residue ESM width (1280 for
    ESM-1b). The cross-attention, layer norms and projection are fresh and
    trainable; the protein side has no parameters (it is the input matrix)."""

    def __init__(self, mol_model, mol_hidden: int, prot_dim: int,
                 num_heads: int = 8, dropout: float = 0.1, lin_proj: bool = True,
                 mlp_hidden: int = 512):
        super().__init__()
        self.mol_model = mol_model
        self.mol_hidden = mol_hidden
        self.prot_dim = prot_dim

        # molecule attends to protein and vice-versa (batch_first)
        self.smi_MHA = nn.MultiheadAttention(mol_hidden, num_heads, dropout=dropout,
                                              kdim=prot_dim, vdim=prot_dim, batch_first=True)
        self.prot_MHA = nn.MultiheadAttention(prot_dim, num_heads, dropout=dropout,
                                              kdim=mol_hidden, vdim=mol_hidden, batch_first=True)
        self.smi_ln = nn.LayerNorm(mol_hidden)
        self.prot_ln = nn.LayerNorm(prot_dim)

        if lin_proj:
            self.proj = nn.Linear(mol_hidden + prot_dim, 1)
        else:
            self.proj = nn.Sequential(
                nn.Linear(mol_hidden + prot_dim, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
                nn.Linear(mlp_hidden, 1))

    @staticmethod
    def _masked_mean(x, mask):
        # x:[B,L,D]  mask:[B,L] float (1=keep)
        m = mask.unsqueeze(-1)
        return (x * m).sum(1) / (m.sum(1) + 1e-8)

    def forward(self, smi_tok, prot, prot_pad, get_repr: bool = False):
        smi_mask = smi_tok["attention_mask"]                       # [B, Ls] (1=real)
        smi = self.mol_model(**smi_tok).last_hidden_state          # [B, Ls, H]

        # cross-attention (key_padding_mask: True = ignore)
        smi_attn, _ = self.smi_MHA(query=smi, key=prot, value=prot, key_padding_mask=prot_pad)
        prot_attn, _ = self.prot_MHA(query=prot, key=smi, value=smi, key_padding_mask=(smi_mask == 0))
        smi = self.smi_ln(smi + smi_attn)
        prot = self.prot_ln(prot + prot_attn)

        smi_rep = self._masked_mean(smi, smi_mask.float())
        prot_rep = self._masked_mean(prot, (~prot_pad).float())
        cat_rep = torch.cat([smi_rep, prot_rep], dim=-1)           # [B, H + prot_dim]
        logit = self.proj(cat_rep).squeeze(-1)
        return (logit, cat_rep) if get_repr else logit


def _build_lora_chemberta(card: str, r: int, alpha: int, dropout: float):
    """Live ChemBERTa + LoRA on query/key/value (base frozen, adapters
    trainable). Lazy imports so this module loads without transformers/peft."""
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model
    base = AutoModel.from_pretrained(card)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["query", "key", "value"])
    return get_peft_model(base, cfg), base.config.hidden_size


# --------------------------------------------------------------------------- training

def _train_one(build_model, load_tokenizer, train_df, val_df, test_df, proteins,
               w_train, w_val, w_test, hp, device, checkpoint_path=None):
    """Same protocol as prosmith_extractor._train_one: Adam over the trainable
    (LoRA + cross-attn + proj) params, fixed epochs, no scheduler/early-stop,
    keep the best-val-loss weights. Weighted BCE on logits (Hladis/M2OR weights
    when present). If `checkpoint_path` exists, skip training and only embed."""
    tokenizer = load_tokenizer()
    val_loader = _make_loader(val_df, proteins, w_val, tokenizer, hp["max_smiles_len"],
                              hp["batch_size"], train=False)
    test_loader = _make_loader(test_df, proteins, w_test, tokenizer, hp["max_smiles_len"],
                               hp["batch_size"], train=False)

    with _INIT_LOCK:
        torch.manual_seed(hp["seed"])
        model = build_model().to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return _embed(model, val_loader, device), _embed(model, test_loader, device), model

    train_loader = _make_loader(train_df, proteins, w_train, tokenizer, hp["max_smiles_len"],
                                hp["batch_size"], train=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"])

    def batch_loss(logits, y, w):
        return nn.functional.binary_cross_entropy_with_logits(logits, y, weight=w, reduction="mean")

    best_val, best_state = np.inf, None
    for _ in range(hp["epochs"]):
        model.train()
        for smi_tok, prot, pad, y, w in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(_to_device(smi_tok, device), prot.to(device), pad.to(device))
            batch_loss(logits, y.to(device), w.to(device)).backward()
            opt.step()

        model.eval()
        total, n = 0.0, 0
        with torch.inference_mode():
            for smi_tok, prot, pad, y, w in val_loader:
                logits = model(_to_device(smi_tok, device), prot.to(device), pad.to(device))
                total += float(batch_loss(logits, y.to(device), w.to(device))) * len(y)
                n += len(y)
        val_loss = total / max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return _embed(model, val_loader, device), _embed(model, test_loader, device), model


def _run_models(ext, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
    ext._ensure_loaded()
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            raise KeyError(f"{ext.name}: {int((~mask).sum())}/{len(idx)} {split_name} rows "
                           f"missing a protein embedding (protein={ext.protein_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]

    w = _M2ORWeights.maybe_build(pairs)
    w_train = None if w is None else w.per_row[train_idx]
    w_val = None if w is None else w.per_row[val_idx]
    w_test = None if w is None else w.per_row[test_idx]

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        checkpoint_path = (pathlib.Path(checkpoint_dir) / f"lorax_{ext.name}_model{m}.pt"
                           if checkpoint_dir is not None else None)
        e_val, e_test, model = _train_one(
            ext._build_model, ext._load_tokenizer, train_df, val_df, test_df, ext._proteins,
            w_train, w_val, w_test, hp, device, checkpoint_path=checkpoint_path)
        train_loader = _make_loader(train_df, ext._proteins, w_train, ext._load_tokenizer(),
                                    hp["max_smiles_len"], hp["batch_size"], train=False)
        return m, _embed(model, train_loader, device), e_val, e_test, model

    # ChemBERTa fine-tuning is heavy; unlike ProSmith we run the n_models jobs
    # sequentially (n_models defaults to 1 anyway -- LORAX's own scheme).
    results = sorted((model_job(m) for m in range(ext.n_models)), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"lorax_{ext.name}_model{m}.pt"
            if not path.exists():
                torch.save(model.state_dict(), path)

    return (np.concatenate([r[1] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[2] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[3] for r in results], axis=1).astype(np.float32))


@dataclass
class LoraxExtractor:
    """LORAX cls-style source: LoRA-ChemBERTa (molecule, live) cross-attended
    with frozen per-residue ESM-1b (protein). Defaults follow LORAX's own config
    (lorax/configs): LoRA r=8/alpha=8 on query/key/value, 8 heads, lr 1e-3,
    50 epochs, linear projection head. The boosting feature is `cat_rep`
    (mol_hidden + prot_dim = 384 + 1280 = 1664 per model).

    `protein_path` must be a *per-residue* npz keyed by sequence
    ({sequence: float32[L, 1280]}); default is ESM-1b, deliberately matching the
    ProSmith baseline. `chemberta_card` is a HuggingFace model id.

    NOTE: needs `transformers` + `peft` (and network/cache for the ChemBERTa
    weights) in the run environment -- imported lazily, so this class imports
    without them but building a model does not."""

    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_per_residue_full_full.npz"
    chemberta_card: str = CHEMBERTA_CARD
    n_models: int = 1
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    num_heads: int = 8
    lin_proj: bool = True
    mlp_hidden: int = 512
    mol_hidden: int = 384          # ChemBERTa-77M hidden width
    prot_dim: int = 1280           # ESM-1b per-residue width
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 12
    max_smiles_len: int = 256
    seed_offset: int = 5000
    pooling: str = "cross_attn_cat"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="lorax_lora_mol")

    def __post_init__(self):
        self.dim_out = self.n_models * (self.mol_hidden + self.prot_dim)
        self.path = f"{self.protein_path} + {self.chemberta_card}"
        self._proteins = None
        self._tokenizer = None

    def _ensure_loaded(self):
        # per-residue npz is ~1.9 GB and --max-parallel pickles the extractor to
        # each worker, so it's loaded lazily (see ProSmith's same note).
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
        if hidden != self.mol_hidden or prot_dim != self.prot_dim:
            # keep dim_out honest if the actual widths differ from the defaults
            self.mol_hidden, self.prot_dim = hidden, prot_dim
            self.dim_out = self.n_models * (hidden + prot_dim)
        return _LoraxMM(mol_model, hidden, prot_dim, self.num_heads,
                        self.lora_dropout, self.lin_proj, self.mlp_hidden)

    def _hp(self, seed):
        return dict(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    max_smiles_len=self.max_smiles_len, seed=seed)

    def covered(self, pairs, idx):
        self._ensure_loaded()
        prot = pairs["receptor"].to_numpy()[idx]
        smi = pairs["smiles"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and isinstance(s, str) and len(s) > 0
                            for p, s in zip(prot, smi)), dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
