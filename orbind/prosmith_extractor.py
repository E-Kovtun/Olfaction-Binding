"""Standalone pair-level "cls" extractor for orbind.ensemble.run_ensemble:
ProSmithExtractor -- a reimplementation of the MPP/ProSmith multimodal
transformer as adapted for M2OR by the olfactory_foundation_models repo
(depasquale-lab/olfactory_foundation_models, `model/MPP/`), which is itself
an adaptation of AlexanderKroll/ProSmith.

Written fresh here in the same spirit as attention_extractor.py and
gnn_extractor.py: no import from the upstream repo, model definition inline
and directly editable, only this project's own generic plumbing reused.

What the original does
----------------------
`MM_TN` (upstream `model/MPP/utils/modules.py`) projects a *pooled* molecule
vector and a *per-residue* protein matrix into a shared 768-d space with one
Linear+ReLU each, lays them out as

    <cls> SMILES <sep> Protein

and runs 6 post-norm `nn.TransformerEncoderLayer`s (6 heads, ff=4*hidden,
gelu) over the result. The feature this extractor hands to the boosting
stage is `hidden_repr = x[0,:,:]` -- the <cls> position after the last layer,
exactly what upstream's `Bert.forward(get_repr=True)` returns and what its
own second stage (`model/MPP/training_GB.py`) feeds to XGBoost.

Note the molecule side is *one token*, not a token sequence: upstream's
`preprocess/generate_embeddings.py` stores one molfeat-pooled vector per
SMILES, and `utils/datautils.py` (`oned=True`) unsqueezes it to (1, d) before
padding. So our existing pooled npzs (ChemBERTa 384-d, GIN 300-d) are already
the right input -- nothing per-token needs generating.

Upstream's own second stage also builds exactly this project's combo
language: `train_cls` (cls alone), `train_X_all` (residue-mean protein ||
pooled molecule) and `train_X_all_cls` (both). Running this source under
`--combos "1 2 3 12 13 23 123"` alongside a mean-ESM and a pooled-molecule
source therefore reproduces upstream's own design rather than reinterpreting
it.

Deviations from upstream (`faithful_bugs` toggles them off)
-----------------------------------------------------------
Two upstream defects are fixed by default. Both change the model, so any
number produced here is a *corrected reimplementation*, not "ProSmith as
published" -- set `faithful_bugs=True` to reproduce the original behaviour
and measure the delta.

1. Double sigmoid on the M2OR path. Upstream's `Bert.forward` ends in
   `self.sigmoid_layer(x)`, and the M2OR loss (lorax's
   `M2ORWeightedCrossEntropyLoss`, which upstream imports whenever
   "M2OR_full" is in the train dir) then applies `nn.BCEWithLogitsLoss` on
   top -- so the loss sees sigmoid(sigmoid(z)), confined to [0.5, 0.731].
   The model cannot express low probability at all and gradients are
   heavily compressed. Their *reported* ranking metrics are unaffected
   (predictions are read before the loss, and AUROC/AUPRC are invariant
   under a monotone map) -- but the model producing them is undertrained.
   Fixed by returning logits, which also matches LORAX, whose head is a
   bare `self.proj(cat_rep)` with no activation.

2. Padding is never masked. Upstream builds `attention_mask` from
   `torch.zeros`/`torch.ones` (float32) and hands it to
   `src_key_padding_mask`, where PyTorch treats a *float* mask as additive
   to the attention scores rather than as a boolean ignore-mask -- so real
   tokens get +1 on their scores and padding is attended to normally.
   Since both poolers end in ReLU, the ~950 padding slots carry a constant
   nonzero learned vector into every attention op. Fixed with a proper
   bool mask (True = ignore).

Fix 2 pays for itself: once padding genuinely contributes nothing, the 255
unused SMILES slots and the ~700 unused protein slots are provably no-ops,
so the sequence shrinks from a fixed 1276 to 3 + max_protein_len_in_batch
(~323 for ORs). That is ~4x fewer positions; the resulting speedup lands
between 4x (if the feed-forward blocks dominate, which scale linearly) and
~16x (if attention dominates, being O(L^2)). Under `faithful_bugs=True` the
full 1 + 256 + 1 + 1018 layout is rebuilt, because there the padding is part
of the computation.

Not reproduced at all: upstream's `SMILESProteinDataset`, which mutates its
own subset state inside `__getitem__` ("assumes linear data reading") and is
only correct under `shuffle=False, num_workers=1`. We index by position
arrays like every other extractor here, so it has no analogue.

Leakage handling matches the other pair-level sources: `n_models`
independently-seeded whole-train models each embed train/val/test through
themselves and their <cls> vectors are concatenated. The default is
`n_models=1` -- that *is* ProSmith's own scheme (one model per split), and
its per-model cost makes bagging expensive besides.
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
from .tasks import batch_loss_fn, check_task

FAITHFUL_MAX_SMILES_LEN = 256    # upstream utils/datautils.py max_smiles_seq_len
FAITHFUL_MAX_PROT_LEN = 1018     # upstream utils/datautils.py max_prot_seq_len

_INIT_LOCK = threading.Lock()    # see _train_one


# --------------------------------------------------------------------------- Hladis / M2OR sample weights

class _M2ORWeights:
    """Per-row loss weights from Hladis et. al. as used by both LORAX and
    ProSmith on M2OR (lorax/loss_functions/loss_functions.py:
    `M2ORWeightedCrossEntropyLoss`): data-quality weight x class-imbalance
    weight x pair-imbalance weight.

    Upstream reads these from a `raw/full_data.csv` that ships in neither
    repo -- but our full_full pool *is* that table (same 46563 rows, with
    `_DataQuality`, and `label` playing the role of `Responsive`), so the
    weights are derived straight from `pairs` with no external file. Global
    counts are taken over the whole pool, exactly as upstream does.

    Returns None when `pairs` has no `_DataQuality` column (e.g. the
    curated_full regime), which is upstream's own condition for falling back
    to unweighted BCE (`if "M2OR_full" in args.train_dir`)."""

    QUALITY = {("ec50", True): 1.0, ("ec50", False): 1.0,
               ("primaryScreening", True): 0.40, ("primaryScreening", False): 0.69,
               ("secondaryScreening", True): 0.72, ("secondaryScreening", False): 0.77}
    K = 100.0

    @classmethod
    def maybe_build(cls, pairs: pd.DataFrame):
        return cls(pairs) if "_DataQuality" in pairs.columns else None

    def __init__(self, pairs: pd.DataFrame):
        y = pairs["label"].to_numpy().astype(bool)
        quality = pairs["_DataQuality"].to_numpy()
        w_quality = np.array([self.QUALITY[(q, bool(r))] for q, r in zip(quality, y)], dtype=np.float64)
        if np.isnan(w_quality).any():
            raise ValueError("unknown _DataQuality label in pairs")

        n_pos, n_neg = int(y.sum()), int((~y).sum())
        w_class = np.where(y, n_neg / max(n_pos, 1), 1.0)

        mols_per_prot = pairs["receptor"].value_counts()
        prots_per_mol = pairs["inchikey"].value_counts()
        inv_prot = pairs["receptor"].map(1.0 / mols_per_prot).to_numpy()
        inv_mol = pairs["inchikey"].map(1.0 / prots_per_mol).to_numpy()
        w_pair = np.log1p(self.K / 2 * (inv_prot + inv_mol))

        # Precomputed once over the whole pool, then indexed by row position --
        # upstream rebuilds an nn.BCEWithLogitsLoss and loops over the batch in
        # Python on every single forward.
        self.per_row = (w_quality * w_class * w_pair).astype(np.float32)


# --------------------------------------------------------------------------- generic data plumbing (no model logic)

class _PairDataset(Dataset):
    """(pooled molecule vector, per-residue protein matrix, label, weight)."""

    def __init__(self, pairs: pd.DataFrame, proteins: dict, molecules: dict, weights):
        self.pairs = pairs.reset_index(drop=True)
        self.proteins = proteins
        self.molecules = molecules
        self.weights = weights

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        return (np.asarray(self.molecules[row.inchikey], dtype=np.float32),
                np.asarray(self.proteins[row.receptor], dtype=np.float32),
                np.float32(row.label),
                np.float32(1.0 if self.weights is None else self.weights[idx]))


def _make_collate(faithful: bool):
    """Pads proteins to the batch maximum (fixed) or to upstream's fixed 1018
    (faithful). `pad` marks positions to ignore -- meaningful only in the
    fixed path, since the faithful path deliberately attends to padding."""
    def collate(batch):
        mols, prots, labels, weights = zip(*batch)
        lengths = [len(p) for p in prots]
        width = FAITHFUL_MAX_PROT_LEN if faithful else max(lengths)
        prot_dim = prots[0].shape[-1]
        x = torch.zeros(len(prots), width, prot_dim, dtype=torch.float32)
        pad = torch.ones(len(prots), width, dtype=torch.bool)
        for i, p in enumerate(prots):
            n = min(len(p), width)
            x[i, :n] = torch.from_numpy(p[:n])
            pad[i, :n] = False
        return (torch.from_numpy(np.stack(mols)), x, pad,
                torch.tensor(labels, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32))
    return collate


def _make_loader(df, proteins, molecules, weights, batch_size, faithful, train):
    ds = _PairDataset(df, proteins, molecules, weights)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, collate_fn=_make_collate(faithful))


@torch.inference_mode()
def _embed(model, loader, device):
    """The <cls> position after the last transformer layer -- upstream's
    `hidden_repr`, and the actual feature handed to the boosting stage."""
    model.eval()
    out = []
    for mol, prot, pad, _, _ in loader:
        _, rep = model(mol.to(device), prot.to(device), pad.to(device), get_repr=True)
        out.append(rep.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- model

class _Pooler(nn.Module):
    """Linear + ReLU into the shared transformer width. Upstream keeps two
    identical copies of this (`BertSmilesPooler`, `BertProteinPooler`); the
    attribute is named `dense` in both, which is what the pretrained
    state_dict keys off."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.dense(x))


class _Bert(nn.Module):
    """Upstream's `Bert`: a stack of post-norm encoder layers over the
    already-assembled sequence, reading the <cls> position out at the end."""

    def __init__(self, hidden: int, n_layers: int, n_heads: int):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=4 * hidden, activation="gelu")
            for _ in range(n_layers)])
        self.hidden_layer = nn.Linear(hidden, 32)
        self.output_layer = nn.Linear(32, 1)
        self.ReLU = nn.ReLU()

    def forward(self, seq, mask):
        x = seq.permute(1, 0, 2)                      # seq-first, as upstream
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=mask)
        rep = x[0, :, :]                              # <cls>
        return self.output_layer(self.ReLU(self.hidden_layer(rep))).squeeze(-1), rep


class _MM_TN(nn.Module):
    """ProSmith's multimodal transformer. Submodule names deliberately match
    upstream (`s_pooler.dense`, `p_pooler.dense`, `main_bert.transformer_layers.*`,
    `main_bert.hidden_layer`, `main_bert.output_layer`) so upstream's own
    BindingDB-pretrained state_dict transfers key-for-key -- see
    `_load_pretrained`."""

    def __init__(self, mol_dim: int, prot_dim: int, hidden: int = 768,
                 n_layers: int = 6, n_heads: int = 6, faithful_bugs: bool = False):
        super().__init__()
        self.hidden = hidden
        self.faithful_bugs = faithful_bugs
        self.s_pooler = _Pooler(mol_dim, hidden)
        self.p_pooler = _Pooler(prot_dim, hidden)
        self.main_bert = _Bert(hidden, n_layers, n_heads)

    def _sequence(self, mol, prot, prot_pad):
        """<cls> SMILES <sep> Protein, plus the key-padding mask.

        Fixed path: the molecule occupies its single real token and the
        protein is already trimmed to the batch maximum, so the sequence is
        3 + L and the mask is a bool "True = ignore".

        Faithful path: rebuilds upstream's fixed 1 + 256 + 1 + 1018 layout.
        Note the zero padding is applied to the *raw* embeddings before
        pooling (upstream pads in `datautils.__getitem__`, pools in
        `MM_TN.forward`), so every padded slot carries ReLU(bias) rather than
        zero -- and the mask is float, which PyTorch applies additively, so
        those slots are attended to normally. Both are the point of this
        branch."""
        b = mol.shape[0]
        cls = torch.ones(b, 1, self.hidden, device=mol.device)
        sep = torch.zeros(b, 1, self.hidden, device=mol.device)

        if not self.faithful_bugs:
            s = self.s_pooler(mol.unsqueeze(1))                       # (B, 1, H)
            p = self.p_pooler(prot)                                   # (B, L, H)
            head = torch.zeros(b, 3, dtype=torch.bool, device=mol.device)
            return torch.cat([cls, s, sep, p], dim=1), torch.cat([head, prot_pad], dim=1)

        raw = torch.zeros(b, FAITHFUL_MAX_SMILES_LEN, mol.shape[-1], device=mol.device)
        raw[:, 0] = mol
        s = self.s_pooler(raw)
        p = self.p_pooler(prot)
        s_attn = torch.zeros(b, FAITHFUL_MAX_SMILES_LEN, device=mol.device)
        s_attn[:, 0] = 1.0
        mask = torch.cat([torch.ones(b, 1, device=mol.device), s_attn,
                          torch.zeros(b, 1, device=mol.device),
                          (~prot_pad).float()], dim=1)
        return torch.cat([cls, s, sep, p], dim=1), mask

    def forward(self, mol, prot, prot_pad, get_repr: bool = False):
        seq, mask = self._sequence(mol, prot, prot_pad)
        out, rep = self.main_bert(seq, mask)
        if self.faithful_bugs:
            # Upstream's extra sigmoid; the M2OR criterion then applies
            # BCEWithLogits on top of it. Reproduced only under this flag.
            out = torch.sigmoid(out)
        return (out, rep) if get_repr else out


def _load_pretrained(model: nn.Module, path, device) -> int:
    """Copy whatever of upstream's BindingDB checkpoint fits, key by key.

    Mirrors upstream's own tolerant loader (`training.py:328-334`): keys the
    checkpoint lacks stay at their fresh init, and keys whose shapes disagree
    are skipped rather than raising -- which is what lets a checkpoint trained
    with one molecule-embedding width be reused with another (only
    `s_pooler.dense` differs; the transformer body transfers intact). The
    checkpoint was saved from a DDP-wrapped model, so "module." is stripped."""
    raw = torch.load(path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in raw.items()}
    copied = 0
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if key in state and state[key].shape == value.shape:
                value.copy_(state[key])
                copied += 1
    return copied


def _train_one(build_model, train_df, val_df, test_df, proteins, molecules,
               w_train, w_val, w_test, hp, device, checkpoint_path=None):
    """Upstream's training protocol: Adam, fixed epoch count, no early
    stopping and no LR scheduler, keeping the weights from the epoch with the
    best validation loss (`training.py:426-429`).

    The criterion is the Hladis-weighted BCE whenever `weights` is not None
    (upstream's M2OR path), otherwise plain BCE. Either way it operates on
    logits -- see this module's docstring on the double-sigmoid fix.

    If `checkpoint_path` already exists, training is skipped entirely and only
    the embed passes run, so a resumed run reuses the trained model."""
    faithful = hp["faithful_bugs"]
    val_loader = _make_loader(val_df, proteins, molecules, w_val, hp["batch_size"], faithful, train=False)
    test_loader = _make_loader(test_df, proteins, molecules, w_test, hp["batch_size"], faithful, train=False)

    # torch.manual_seed is global while the n_models jobs run concurrently in
    # threads, so seeding and consuming that seed have to be one critical
    # section -- otherwise a sibling's reseed lands between them and the
    # bagged models' inits stop being reproducible.
    with _INIT_LOCK:
        torch.manual_seed(hp["seed"])
        model = build_model().to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return _embed(model, val_loader, device), _embed(model, test_loader, device), model

    if hp["pretrained_path"]:
        n = _load_pretrained(model, hp["pretrained_path"], device)
        print(f"    prosmith: loaded {n} tensors from {hp['pretrained_path']}", flush=True)

    train_loader = _make_loader(train_df, proteins, molecules, w_train, hp["batch_size"], faithful, train=True)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    # weight= + reduction='mean' reproduces upstream's weighted BCE exactly
    # (sum(w_i * l_i) / N, not / sum(w)); the regression branch keeps that
    # convention with squared error. See orbind/tasks.py.
    batch_loss = batch_loss_fn(hp["task"])

    best_val, best_state = np.inf, None
    for epoch in range(hp["epochs"]):
        model.train()
        for mol, prot, pad, y, w in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(mol.to(device), prot.to(device), pad.to(device))
            batch_loss(logits, y.to(device), w.to(device)).backward()
            opt.step()

        model.eval()
        total, n = 0.0, 0
        with torch.inference_mode():
            for mol, prot, pad, y, w in val_loader:
                logits = model(mol.to(device), prot.to(device), pad.to(device))
                total += float(batch_loss(logits, y.to(device), w.to(device))) * len(y)
                n += len(y)
        val_loss = total / max(n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return _embed(model, val_loader, device), _embed(model, test_loader, device), model


def _run_models(ext, pairs: pd.DataFrame, train_idx, val_idx, test_idx, seed: int, checkpoint_dir=None):
    check_task(ext.task)
    if ext.task == "regression" and ext.faithful_bugs:
        # Upstream's extra sigmoid squashes the head's output into
        # [0.5, 0.731]. Harmless for classification (AUROC/AUPRC are
        # rank-invariant, and the fixed path is the default anyway), fatal for
        # a continuous target -- no z-scored response can be reached.
        raise ValueError(
            f"{ext.name}: faithful_bugs=True is incompatible with task='regression' -- "
            f"upstream's double sigmoid bounds the prediction to [0.5, 0.731]")
    ext._ensure_loaded()
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mask = ext.covered(pairs, idx)
        if not mask.all():
            missing = int((~mask).sum())
            raise KeyError(
                f"{ext.name}: {missing}/{len(idx)} {split_name} rows missing a protein or "
                f"molecule embedding (protein={ext.protein_path}, molecule={ext.molecule_path})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = pairs.iloc[train_idx], pairs.iloc[val_idx], pairs.iloc[test_idx]

    w = _M2ORWeights.maybe_build(pairs)
    w_train = None if w is None else w.per_row[train_idx]
    w_val = None if w is None else w.per_row[val_idx]
    w_test = None if w is None else w.per_row[test_idx]

    def model_job(m):
        hp = ext._hp(seed + ext.seed_offset * m)
        checkpoint_path = (pathlib.Path(checkpoint_dir) / f"prosmith_{ext.name}_model{m}.pt"
                           if checkpoint_dir is not None else None)
        e_val, e_test, model = _train_one(
            ext._build_model, train_df, val_df, test_df, ext._proteins, ext._molecules,
            w_train, w_val, w_test, hp, device, checkpoint_path=checkpoint_path)
        train_loader = _make_loader(train_df, ext._proteins, ext._molecules, w_train,
                                    hp["batch_size"], hp["faithful_bugs"], train=False)
        return m, _embed(model, train_loader, device), e_val, e_test, model

    with ThreadPoolExecutor(max_workers=ext.n_models) as pool:
        futures = [pool.submit(model_job, m) for m in range(ext.n_models)]
        results = sorted((f.result() for f in futures), key=lambda r: r[0])

    if checkpoint_dir is not None:
        for m, _, _, _, model in results:
            path = pathlib.Path(checkpoint_dir) / f"prosmith_{ext.name}_model{m}.pt"
            if not path.exists():
                torch.save(model.state_dict(), path)

    return (np.concatenate([r[1] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[2] for r in results], axis=1).astype(np.float32),
            np.concatenate([r[3] for r in results], axis=1).astype(np.float32))


@dataclass
class ProSmithExtractor:
    """ProSmith/MPP cls source. Defaults follow upstream's own CLI
    (`training.py`): hidden 768, 6 layers, 6 heads, Adam at lr 1e-5, 50
    epochs. `batch_size` defaults to 72 -- upstream's per-GPU 12 times the
    6 GPUs its published command assumes, i.e. its effective batch.

    Embedding inputs: `protein_path` must be a *per-residue* npz keyed by
    sequence ({sequence: float32[L, 1280]}), `molecule_path` a pooled npz
    keyed by inchikey. The default protein file is upstream's own ESM-**1b**,
    imported from its data release rather than recomputed -- see
    scripts/embedding_generation/proteins/06_import_ofm_esm1b.py, which must
    be run once first. Point `protein_path` at
    `esm2_650m_per_residue_full_full.npz` to use our ESM-2 instead; that is a
    deviation from upstream and should be reported as one.

    `pretrained_path` points at upstream's BindingDB checkpoint
    (`BindingDB.zip` in https://zenodo.org/records/17228740 -- note the data
    link in upstream's own README is broken, pointing back at the repo). It
    is what upstream's published numbers use; leaving it empty trains from
    scratch, which is a materially weaker model and must be labelled so.

    `faithful_bugs=True` restores the double sigmoid and the unmasked
    padding, at ~15x the cost -- see the module docstring."""

    name: str
    protein_path: str = "data/embeddings/proteins/esm1b_650m_per_residue_full_full.npz"
    molecule_path: str = "data/embeddings/molecules/chemberta_77m_m2or.npz"
    pretrained_path: str = ""
    hidden: int = 768
    n_layers: int = 6
    n_heads: int = 6
    lr: float = 1e-5
    epochs: int = 50
    batch_size: int = 72
    n_models: int = 1
    seed_offset: int = 5000
    faithful_bugs: bool = False
    # The task axis (orbind/tasks.py). `run_ensemble` overwrites this to match
    # the run, so this head's criterion and the boosting head downstream agree.
    task: str = "classification"
    pooling: str = "cls_token"
    dim_out: int = field(init=False, default=0)
    model_name: str = field(init=False, default="prosmith_mm_tn")

    def __post_init__(self):
        self.dim_out = self.n_models * self.hidden
        self.path = f"{self.protein_path} + {self.molecule_path}"
        self._proteins = None
        self._molecules = None
        # Checked here rather than at load time: pretrained_path is the 4th
        # positional field of the --source mini-language, so a miscounted
        # colon silently lands some other value in it, and the failure would
        # otherwise surface as a torch.load traceback inside a worker process
        # minutes into the run.
        if self.pretrained_path and not pathlib.Path(self.pretrained_path).exists():
            raise FileNotFoundError(
                f"{self.name}: pretrained_path {self.pretrained_path!r} does not exist. "
                f"Expected upstream's BindingDB checkpoint (BindingDB.zip from "
                f"https://zenodo.org/records/17228740), or an empty field to train from scratch.")

    def _ensure_loaded(self):
        """Loaded lazily, not in __post_init__ like the other extractors: the
        per-residue protein npz is ~1.9 GB, and --max-parallel pickles the
        extractor to each worker process. Eager loading would push that
        through the spawn pipe once per repeat."""
        if self._proteins is None:
            self._proteins = D.load_npz_dict(self.protein_path)
        if self._molecules is None:
            self._molecules = D.load_npz_dict(self.molecule_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_proteins"] = None
        state["_molecules"] = None
        return state

    def _build_model(self):
        mol_dim = next(iter(self._molecules.values())).shape[-1]
        prot_dim = next(iter(self._proteins.values())).shape[-1]
        return _MM_TN(mol_dim, prot_dim, self.hidden, self.n_layers, self.n_heads, self.faithful_bugs)

    def _hp(self, seed):
        return dict(lr=self.lr, epochs=self.epochs, batch_size=self.batch_size,
                    faithful_bugs=self.faithful_bugs, pretrained_path=self.pretrained_path,
                    task=self.task, seed=seed)

    def covered(self, pairs, idx):
        self._ensure_loaded()
        prot = pairs["receptor"].to_numpy()[idx]
        mol = pairs["inchikey"].to_numpy()[idx]
        return np.fromiter(((p in self._proteins) and (m in self._molecules) for p, m in zip(prot, mol)),
                           dtype=bool, count=len(idx))

    def fit_transform(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=None):
        return _run_models(self, pairs, train_idx, val_idx, test_idx, seed, checkpoint_dir=checkpoint_dir)
