"""Configurable multi-source boosting ensemble (see orbind/ensemble.py).

Each --source registers one embedding extractor under a short name; the
1-based position among --source flags is what the --combos digit-string
mini-language refers to. Two families of source type:

  esm, gin              -- label-independent, entity-level lookups
                            ("name=type:path[:model_name[:pooling]]").
  attn_noisy_or, attn_lse -- supervised, pair-level "cls" sources (see
                            orbind/attention_extractor.py). Each trains
                            n_models (default 5) independently-seeded
                            whole-train torch models -- always concurrent --
                            and concatenates their pooled embeddings; every
                            model embeds train/val/test alike (accepting
                            mild train-row leakage, bounded by early
                            stopping, same tradeoff ProSmith's own cls model
                            makes). n_models=1 reduces to ProSmith's exact
                            scheme. Bare "name=type" uses its baked-in
                            embedding paths (LoRaX ESM-1b mean + our per-atom
                            GIN) and n_models=5; pass
                            "name=type:protein_path:molecule_sites_path:n_models"
                            to override any of them (leave a field blank to
                            keep its default, e.g. "cls=attn_lse:::1").

Three regimes (see orbind/dataset.py vs orbind/regimes.py vs
orbind/regimes_ofm.py for why they aren't unified):

  curated_full  -- pairs from a csv (pairs_curated.csv / pairs_m2or_full.csv),
                   split via --split {stratified,group_molecule,group_receptor}
                   + --seeds (fresh random split per seed).
  full_full     -- pairs reconstructed from LoRaX's own data
                   (orbind.regimes.full_full_pairs), split via
                   --full-full-mode
                   {transductive,inductive_molecule,inductive_molecule_v5} +
                   --repeats (LoRaX fold 1-5 for transductive, cold-molecule
                   seed for the two inductive modes). `inductive_molecule` is
                   our own cold-molecule split (30% holdout, stratified);
                   `inductive_molecule_v5` reproduces the v5 graph screen's
                   own split exactly (20% test / 10% val molecules,
                   unstratified) so ensemble numbers can be compared to that
                   screen head-on -- see orbind/regimes.py.
  ofm           -- the Carey (--dataset cc) and Hallem-Carlson (--dataset hc)
                   datasets from the olfactory foundation models release, with
                   upstream's own 5-fold splits: --split-family
                   {rand,cdhit,scaf} = {i.i.d., unseen receptors, unseen
                   odorants}. HC ships only `rand`. Plus `our_inductive`,
                   ours: cold molecule like scaf but stratified, so every
                   fold is scorable. --repeats picks folds (default 1..5).
                   See orbind/regimes_ofm.py.

The task axis
-------------
--task {classification,regression}. M2OR's target is a 0/1 flag; the Carey and
Hallem targets are continuous z-scored responses, and upstream scores them with
R^2. `--regime ofm` therefore defaults to `regression`, which switches the
boosting head to XGBRegressor, the metrics to R2/RMSE/MAE/Pearson/Spearman, the
ensemble stacker to simplex-on-MSE / linreg, and every cls extractor's
criterion to squared error (orbind/tasks.py).

Every run also emits a `naive[train-mean]` row: the constant train-mean
predictor, scored on the same test rows with the same metrics. Under regression
that is upstream's own naive baseline and the only thing that makes an R^2 near
zero interpretable (R^2 is measured against the TEST mean, so a model can beat
the naive row while still scoring below 0). Under classification the same
constant is the class prevalence -- i.e. the AUPRC floor.

Every invocation creates one timestamped run folder under
results/ensemble_logs/<run_id>/:

  config.json           -- every CLI arg, for reproducibility
  log.txt                -- this process's own stdout (setup + final summary)
  logs/repeat_{R}.log     -- one per repeat, captures its combo/ensemble lines
                            even when --max-parallel runs it in another process
  metrics.csv             -- one row per (repeat, combo) + (repeat, ensemble method)
  scores/repeat_{R}.npz   -- per-ROW validation and test scores of every combo, with the
                            labels beside them. metrics.csv holds only aggregates, and an
                            aggregate cannot be re-thresholded: MCC/F1 are scored at a
                            fixed 0.5 cut, so an operating point chosen on validation
                            needs these. Written even under --skip-checkpoints.
  checkpoints/repeat_{R}/
    boost_{combo}.json           -- one XGBoost booster per combo (unless --skip-checkpoints)
    attn_{source}_model{k}.pt     -- one torch state_dict per model, per pair-level source

--max-parallel N runs repeats concurrently as separate OS processes (own CUDA
context each); a pair-level source's n_models are always trained concurrently
regardless (see orbind/attention_extractor.py), independent of
this flag. With >1 GPU visible (or --gpus explicitly given), repeats are
pinned round-robin one GPU per worker process via CUDA_VISIBLE_DEVICES --
each worker keeps its GPU for every repeat it picks up. With <=1 GPU, no
pinning happens and all workers share the default device as before.

Examples
--------
Solo ESM, solo GIN, and their concatenation, on curated/group_molecule::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime curated_full --pairs data/processed/pairs_curated.csv \\
        --source prot=esm:data/embeddings/proteins/esm2_650m_mean.npz \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 12" --split group_molecule --seeds 42 43 44

Same combos on full_full, both split traditions, LoRaX's own ESM-1b::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 12" --on-missing drop

Adding the cls (attention) source -- solo, paired with each of prot/mol, and
all three together -- on full_full/transductive, all 5 folds, 3 at a time::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive --max-parallel 3 \\
        --source cls=attn_noisy_or \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 3 12 13 23 123" --on-missing drop

The ProSmith/MPP baseline itself, reproducing upstream's own second stage
(its `training_GB.py` builds exactly `cls`, `prot+mol` and `prot+mol+cls`).
The cls source takes a *per-residue* protein npz -- upstream's own ESM-1b,
imported by scripts/embedding_generation/proteins/06_import_ofm_esm1b.py --
while prot stays mean-pooled; both defaults, so only the BindingDB
checkpoint (which upstream's published numbers use) has to be named. Its path
is relative to the CWD, i.e. the repo root -- the checkpoint lives wherever
BindingDB.zip was unpacked, which here is under `data/external/ofm/`, NOT the
bare `saved_model/` of upstream's own tree::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive \\
        --source cls=prosmith::::data/external/ofm/saved_model/pretraining_IC50_6gpus_bs144_1.5e-05_layers6.txt.pkl \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/chemberta_77m_m2or.npz:chemberta_77m \\
        --combos "1 2 3 12 13 23 123" --on-missing drop

Same, but cls as a single ProSmith-style model (n_models=1) instead of the
default 5-model bagging ensemble -- note the blank protein/molecule-path
fields to keep their defaults::

    uv run python scripts/modeling/train/train_ensemble_boost.py \\
        --regime full_full --full-full-mode transductive --max-parallel 3 \\
        --source cls=attn_noisy_or:::1 \\
        --source prot=esm:data/embeddings/proteins/esm1b_650m_mean.npz:esm1b_t33_650M_UR50S \\
        --source mol=gin:data/embeddings/molecules/gin_supervised_contextpred_all_m2or.npz \\
        --combos "1 2 3 12 13 23 123" --on-missing drop
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import queue
import shutil
import sys
import time
from datetime import datetime

import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import importlib

from orbind.baselines import check_xgboost_version
from orbind.ensemble import run_ensemble
from orbind.regimes import full_full_pairs, load_split
from orbind.regimes_ofm import DATASETS as OFM_DATASETS, available_families, ofm_indices, ofm_pairs
from orbind.tasks import TASKS

# Source types dispatch LAZILY: parse_source_arg imports an extractor's module
# only when a source of that type is actually requested, so a run pulls just the
# heavy deps it needs -- gnn_* -> torch_geometric, lorax -> transformers+peft,
# prosmith/attn/esm/gin -> torch only. This is what lets the ProSmith/LORAX
# "controls" run in a PyG-free environment separate from the graph pipeline.
_FACTORY_SPEC = {
    # entity-level: "name=type:path[:model_name[:pooling]]" -- a static npz lookup.
    "esm": ("orbind.ensemble", "EsmExtractor"),
    "gin": ("orbind.ensemble", "GinExtractor"),
    # attention MIL (torch only): "name=type[:protein_path:molecule_sites_path[:n_models]]".
    "attn_noisy_or": ("orbind.attention_extractor", "MilNoisyOrExtractor"),
    "attn_lse": ("orbind.attention_extractor", "MilLseExtractor"),
    # graph-based (pulls torch_geometric):
    # "name=type[:protein_path:molecule_path[:n_models[:emit[:q[:criterion[:edge_threshold[:k_mode]]]]]]]".
    # emit: "prot" (default, v5 probe shape) or "both". gnn_signed_dgi adds a
    # DeepGraphInfomax auxiliary loss (shared scope, lambda=0.5). Field 12 is the
    # v8 alpha gate (blank = off): 12 colons is a lot to type by hand, which is why
    # scripts/modeling/train/run_alpha_gate_sweep.py exists. Fields 16-17 are the two
    # GraphSAGE-regime additions (neighbour sampling, per-layer L2), both off by
    # default -- see scripts/article_sweeps/s5_run_architecture.py for what they buy.
    "gnn_signed": ("orbind.gnn_extractor", "GnnSignedExtractor"),
    "gnn_signed_dgi": ("orbind.gnn_extractor", "GnnSignedDgiExtractor"),
    # TWO-STAGE signed GNN: stage 1 fine-tunes LoRA-ChemBERTa the LORAX way and
    # freezes it -> static per-molecule features; stage 2 runs our signed graph on
    # those + frozen mean-ESM proteins, our protocol. Pulls torch_geometric +
    # transformers + peft. Fields (field 1 = MEAN-pooled ESM npz; field 10 =
    # PER-RESIDUE ESM npz for LORAX cross-attn):
    # "name=gnn_lora[:protein_path:chemberta_card[:n_models[:emit[:q[:criterion[:edge_threshold[:k_mode[:lora_r[:lora_protein_path[:dummy_compression]]]]]]]]]]".
    # See orbind/gnn_lora_extractor.py.
    "gnn_lora": ("orbind.gnn_lora_extractor", "GnnLoraExtractor"),
    # ProSmith/MPP transformer over a *per-residue* protein npz + pooled molecule npz
    # (torch only): "name=prosmith[:protein_path:molecule_path[:n_models[:pretrained_path[:faithful_bugs]]]]".
    # pretrained_path = upstream BindingDB checkpoint; empty trains from scratch
    # (weaker than upstream's published numbers). faithful_bugs=1 restores the
    # double sigmoid + unmasked padding (slower). See orbind/prosmith_extractor.py.
    "prosmith": ("orbind.prosmith_extractor", "ProSmithExtractor"),
    # LORAX (pulls transformers+peft): LoRA-ChemBERTa molecule + cross-attention
    # over frozen per-residue ESM-1b protein. LoRA on the MOLECULE side only and
    # protein frozen ESM-1b, so directly comparable to the ProSmith baseline (both
    # on ESM-1b). "name=lorax[:protein_path:chemberta_card[:n_models[:lora_r[:epochs]]]]"
    # (epochs exposed so a smoke can pass e.g. cls=lorax:::::2).
    # See orbind/lorax_extractor.py.
    "lorax": ("orbind.lorax_extractor", "LoraxExtractor"),
    # MolOR (pulls dgl+dgllife+rdkit): GCN molecule encoder (trained live) cross-
    # attended with frozen per-residue ESM-1b protein -- same ESM-1b as ProSmith
    # and LORAX, so all three are directly comparable. Molecule is a 2D graph, not
    # ChemBERTa (that IS MolOR's identity). "name=molor[:protein_path[:n_models[:epochs]]]".
    # See orbind/molor_extractor.py.
    "molor": ("orbind.molor_extractor", "MolorExtractor"),
    # Hladis et al. ICLR 2023 (pulls rdkit): MPNN-attention over the molecule
    # graph, with the receptor vector broadcast onto every atom -- no cross-
    # attention block at all. Protein defaults to mean-pooled ESM-1b (upstream
    # uses ProtBERT CLS; ESM-1b keeps it comparable to the other three).
    # Emitted feature is only node_d_model=72 wide by default -- upstream's own
    # size, far narrower than prosmith/lorax/molor.
    # "name=hladis[:protein_path[:n_models[:max_steps[:warmup_steps[:eval_every]]]]]".
    # See orbind/hladis_extractor.py.
    "hladis": ("orbind.hladis_extractor", "HladisExtractor"),
}
_ENTITY_TYPES    = {"esm", "gin"}
_ATTENTION_TYPES = {"attn_noisy_or", "attn_lse"}
_GNN_TYPES       = {"gnn_signed", "gnn_signed_dgi"}
_GNNLORA_TYPES   = {"gnn_lora"}
_PROSMITH_TYPES  = {"prosmith"}
_LORAX_TYPES     = {"lorax"}
_MOLOR_TYPES     = {"molor"}
# "name=hladis[:protein_path[:n_models[:max_steps]]]". Budget is in optimizer
# steps, not epochs -- see HladisExtractor's docstring on why the paper and the
# released config disagree and why steps is the self-consistent reading.
_HLADIS_TYPES    = {"hladis"}


def _factory(type_):
    """Import the extractor module lazily and return its class."""
    module, cls = _FACTORY_SPEC[type_]
    return getattr(importlib.import_module(module), cls)
DEFAULT_REPEATS = {"transductive": [1, 2, 3, 4, 5],
                    "inductive_molecule": [42, 43, 44, 45, 46],
                    "inductive_molecule_v5": [42, 43, 44, 45, 46]}
# ofm repeats are upstream's own 5 folds of whichever split family is chosen.
OFM_FOLDS = [1, 2, 3, 4, 5]


def parse_source_arg(raw: str):
    """"name=type:path[:model_name[:pooling]]" (entity-level) or
    "name=type[:protein_path:molecule_sites_path[:n_models]]" (attention,
    pair-level) -> (name, extractor). n_models=1 is ProSmith's own scheme
    (one whole-train model embeds train/val/test); n_models>1 (default 5)
    bags that many independently-seeded models -- see attention_extractor's
    module docstring for why this replaced true OOF fold-holdout retraining."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--source {raw!r} must look like name=type[:...]")
    name, rest = raw.split("=", 1)
    parts = rest.split(":")
    type_ = parts[0]

    if type_ in _ATTENTION_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_sites_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _GNN_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["emit"] = parts[4]
        # q / criterion / edge_threshold were dataclass-only until the Carey
        # datasets arrived: there the receptor x odorant matrix is COMPLETE, so
        # per-molecule coverage is near-uniform and the q=0.99 default keeps a
        # couple of molecules instead of a hub core. They have to be settable.
        if len(parts) > 5 and parts[5]:
            kwargs["q"] = float(parts[5])
        if len(parts) > 6 and parts[6]:
            kwargs["criterion"] = parts[6]
        if len(parts) > 7 and parts[7]:
            kwargs["edge_threshold"] = float(parts[7])
        # k_mode: how q becomes K. Needed for the same reason -- on a complete
        # matrix the coverage quantile is a no-op, so Carey/Hallem sweeps have
        # to ask for "fraction". See orbind.mol_selection.resolve_K.
        if len(parts) > 8 and parts[8]:
            kwargs["k_mode"] = parts[8]
        # Fields 9/10: EXPERIMENTAL continuous-label edge modes (defaults off, so
        # binary M2OR is unchanged). edge_center global|per_receptor (#4),
        # edge_weight_mode none|magnitude (#1). See GnnSignedExtractor.
        if len(parts) > 9 and parts[9]:
            kwargs["edge_center"] = parts[9]
        if len(parts) > 10 and parts[10]:
            kwargs["edge_weight_mode"] = parts[10]
        # field 11: dummy_compression (freeze input projections to a train-fit PCA).
        if len(parts) > 11 and parts[11]:
            kwargs["dummy_compression"] = parts[11] not in ("0", "false", "False")
        # field 12: v8 alpha gate. Empty = the historical graph (no gate at all).
        # z_prot = (1-alpha)*frozen PCA(ESM) + alpha*graph; 0 = pure structure,
        # 1 = the graph alone. See GnnSignedExtractor.alpha.
        if len(parts) > 12 and parts[12]:
            kwargs["alpha"] = float(parts[12])
        # field 13: one-hot receptor node features (blank = the embedding file). With
        # the gate on this is what makes alpha an honest fraction of structure.
        if len(parts) > 13 and parts[13]:
            kwargs["onehot_nodes"] = parts[13] not in ("0", "false", "False")
        # field 14: v9 node dial. Blank = the node features are the embedding file.
        # x_prot = mu + rho*centred(ESM) + (1-rho)*centred(random unit vector per
        # receptor); 1 = the legacy graph exactly, 0 = receptor identity and nothing
        # else. A DIFFERENT axis from field 12 -- that gates the output, this moves the
        # input -- and it runs the other way round. See GnnSignedExtractor.prot_mix.
        if len(parts) > 14 and parts[14]:
            kwargs["prot_mix"] = float(parts[14])
        # field 15: which draw of the identity vectors (default 0).
        if len(parts) > 15 and parts[15]:
            kwargs["mix_seed"] = int(parts[15])
        # Fields 16-17: the GraphSAGE training regime. Field 16 is the per-layer
        # fan-out, written with a dash because the spec separator is a colon:
        # "25-10" = 25 on layer 1, 10 on layer 2, redrawn every epoch, inference still
        # full-neighbourhood. Field 17 is the per-layer L2 normalisation. BLANK MEANS
        # THE EXTRACTOR DEFAULT, which since 23.09.2026 is both of them ON -- to get
        # the historical encoder back, say "0" (or "none"/"off") in field 16 and "0"
        # in field 17. A command written before that date therefore means something
        # different now, which is deliberate: the default is the model we report.
        if len(parts) > 16 and parts[16]:
            raw = parts[16].strip().lower()
            kwargs["fanout"] = () if raw in ("0", "off", "none") else tuple(
                int(f) for f in raw.replace(",", "-").split("-"))
        if len(parts) > 17 and parts[17]:
            kwargs["normalize_layers"] = parts[17] not in ("0", "false", "False")
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _GNNLORA_TYPES:
        # gnn_lora: TWO-STAGE signed GNN. Stage 1 fine-tunes LoRA-ChemBERTa the
        # LORAX way, freezes it -> static per-molecule features; stage 2 runs our
        # signed graph on those, our protocol. field 1 = MEAN-pooled ESM npz (the
        # graph's protein node features); field 2 = chemberta_card (HF id); graph
        # fields 3-8 mirror gnn_signed; field 9 = lora_r; field 10 = the
        # PER-RESIDUE ESM npz for LORAX cross-attention (default set); field 11 =
        # dummy_compression. See orbind/gnn_lora_extractor.py.
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["chemberta_card"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["emit"] = parts[4]
        if len(parts) > 5 and parts[5]:
            kwargs["q"] = float(parts[5])
        if len(parts) > 6 and parts[6]:
            kwargs["criterion"] = parts[6]
        if len(parts) > 7 and parts[7]:
            kwargs["edge_threshold"] = float(parts[7])
        if len(parts) > 8 and parts[8]:
            kwargs["k_mode"] = parts[8]
        if len(parts) > 9 and parts[9]:
            kwargs["lora_r"] = int(parts[9])
        if len(parts) > 10 and parts[10]:
            kwargs["lora_protein_path"] = parts[10]
        if len(parts) > 11 and parts[11]:
            kwargs["dummy_compression"] = parts[11] not in ("0", "false", "False")
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _PROSMITH_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["molecule_path"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["pretrained_path"] = parts[4]
        if len(parts) > 5 and parts[5]:
            kwargs["faithful_bugs"] = parts[5] not in ("0", "false", "False")
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _LORAX_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["chemberta_card"] = parts[2]
        if len(parts) > 3 and parts[3]:
            kwargs["n_models"] = int(parts[3])
        if len(parts) > 4 and parts[4]:
            kwargs["lora_r"] = int(parts[4])
        if len(parts) > 5 and parts[5]:
            kwargs["epochs"] = int(parts[5])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ in _MOLOR_TYPES or type_ in _HLADIS_TYPES:
        kwargs = {}
        if len(parts) > 1 and parts[1]:
            kwargs["protein_path"] = parts[1]
        if len(parts) > 2 and parts[2]:
            kwargs["n_models"] = int(parts[2])
        if type_ in _HLADIS_TYPES:
            # Hladis counts optimizer steps, and the two schedule companions are
            # exposed alongside because they only make sense together: the LR is
            # `init * min(step^-0.5, step * warmup^-1.5)`, so a max_steps below
            # warmup_steps never leaves the ramp, and eval_every fixes how many
            # times best-val weight selection gets to look. Upstream's
            # 10000/6000/500 is sized for M2OR's 41k rows; on a dataset an order
            # of magnitude smaller all three have to come down together.
            for pos, key in ((3, "max_steps"), (4, "warmup_steps"), (5, "eval_every")):
                if len(parts) > pos and parts[pos]:
                    kwargs[key] = int(parts[pos])
            # field 7: print the model's own scalar head's test metrics, which is
            # what the paper reports -- needed to tell a port bug apart from a
            # boosting-head effect. Diagnostic only.
            if len(parts) > 6 and parts[6]:
                kwargs["report_own_head"] = parts[6] not in ("0", "false", "False")
        elif len(parts) > 3 and parts[3]:
            kwargs["epochs"] = int(parts[3])
        return name, _factory(type_)(name=name, **kwargs)

    if type_ not in _ENTITY_TYPES:
        raise argparse.ArgumentTypeError(
            f"unknown source type {type_!r}, have {list(_FACTORY_SPEC)}")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"--source {raw!r} must look like name=type:path")
    path = parts[1]
    kwargs = {}
    if len(parts) > 2 and parts[2]:
        kwargs["model_name"] = parts[2]
    if len(parts) > 3 and parts[3]:
        kwargs["pooling"] = parts[3]
    extractor = _factory(type_)(name=name, path=path, **kwargs)
    return name, extractor


def _detect_gpus(explicit):
    """GPU ids to round-robin repeats across, or None (no pinning needed --
    either the user gave none/one explicitly, or there's <=1 visible GPU)."""
    if explicit is not None:
        return explicit if len(explicit) > 1 else None
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return list(range(n)) if n > 1 else None


class _Tee:
    """Writes to several streams at once -- lets a repeat's own log file
    capture its prints while they still show up live in the console."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        # Never a TTY: keeps libraries (e.g. transformers' loading report) from
        # emitting ANSI colour codes into the captured log file, and avoids the
        # AttributeError they raise when they probe sys.stdout.isatty().
        return False

    def __getattr__(self, name):
        # Delegate anything else a library might probe (fileno, encoding, ...) to
        # the real console stream.
        return getattr(self.streams[0], name)


def _run_one_repeat(regime, pairs, extractors, combos, split, repeat, test_size, val_size,
                     weight_method, on_missing, full_full_mode, run_dir, save_checkpoints,
                     tune_boost_hp, n_trials, optuna_storage,
                     task="classification", ofm_dataset=None, ofm_family=None):
    """One repeat's full run_ensemble call -- top-level (not a closure) so it
    can be pickled and sent to a separate process by --max-parallel. Owns its
    own log file and checkpoint subdir regardless of which process runs it."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = None
    if save_checkpoints:
        checkpoint_dir = run_dir / "checkpoints" / f"repeat_{repeat}"
    # Per-row val/test scores, always -- they are a few KB and they are the only way to
    # choose a decision threshold on validation after the run. `--skip-checkpoints`
    # suppresses model WEIGHTS, which is a different question, so it does not apply here.
    scores_dir = run_dir / "scores"

    with open(log_dir / f"repeat_{repeat}.log", "w", encoding="utf-8") as logf:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, logf)
        try:
            gpu_note = os.environ.get("CUDA_VISIBLE_DEVICES")
            tag = f" (pinned to GPU {gpu_note})" if gpu_note is not None else ""
            print(f"=== repeat {repeat} ==={tag}", flush=True)
            if regime == "curated_full":
                result = run_ensemble(pairs, extractors, combos, split_kind=split, seed=repeat,
                                       test_size=test_size, val_size=val_size,
                                       weight_method=weight_method, on_missing=on_missing,
                                       checkpoint_dir=checkpoint_dir, scores_dir=scores_dir,
                                       tune_boost_hp=tune_boost_hp, n_trials=n_trials,
                                       optuna_storage=optuna_storage, run_id=str(repeat),
                                       task=task)
            else:
                if regime == "ofm":
                    train_idx, val_idx, test_idx = ofm_indices(ofm_dataset, ofm_family, repeat)
                else:
                    train_idx, val_idx, test_idx = load_split(full_full_mode, repeat)
                result = run_ensemble(pairs, extractors, combos,
                                       train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                                       weight_method=weight_method, on_missing=on_missing,
                                       checkpoint_dir=checkpoint_dir, scores_dir=scores_dir,
                                       tune_boost_hp=tune_boost_hp, n_trials=n_trials,
                                       optuna_storage=optuna_storage, run_id=str(repeat),
                                       task=task)
        finally:
            sys.stdout = old_stdout
    return repeat, result


def _run_one_repeat_to_queue(q, *call_args):
    """Wraps _run_one_repeat for a plain multiprocessing.Process worker (see
    the GPU-pinning note in _run() for why this replaced ProcessPoolExecutor's
    initializer-based pinning): puts (repeat, result) on `q` instead of
    returning it, since a bare Process has no return-value channel."""
    q.put(_run_one_repeat(*call_args))


def _run_repeat_pool(repeats, n_slots, gpu_ids, launch, collect, poll=5.0, grace=10.0,
                     log=print):
    """Run `repeats` through `n_slots` worker slots, refilling a slot the moment its
    worker finishes. Returns [(repeat, exitcode)] for workers that died unanswered.

    It replaced fixed chunks of `n_slots` that each waited for their slowest member:
    a repeat whose model loads from a checkpoint is done in minutes, one that trains
    from scratch takes hours, and every chunk mixing the two left a GPU idle for the
    difference -- as did the last chunk of an odd count. The server showed one busy
    card out of two for most of a ProSmith run.

    Slot k is bound to `gpu_ids[k % len(gpu_ids)]` for its whole life, so a refill lands
    on the card that was just freed and no card ever holds more workers than it did
    under the chunked scheme. `launch(repeat, gpu) -> (process, queue)` starts one
    worker; `collect(repeat, result)` records it.

    A worker that dies without answering (a CUDA teardown abort takes SIGABRT and
    leaves nothing on the queue) is recorded and its slot reused; the OTHER repeats
    carry on and are written. Raising at the first death, as the chunked loop did,
    threw away hours of training still running beside it. The caller decides what a
    death means -- here, a non-zero exit after every survivor is on disk.

    The queue is always drained BEFORE the process is joined: a worker cannot exit
    while its result is still in the pipe, so join-then-read can deadlock on a large
    result. A dead worker's result is only given up on after one more `grace`-long
    read, because it may exit between `put` and the liveness check."""
    pending = list(repeats)
    running = {}                           # slot -> (repeat, process, queue)
    failed = []
    while pending or running:
        for slot in range(n_slots):
            if slot not in running and pending:
                repeat = pending.pop(0)
                gpu = gpu_ids[slot % len(gpu_ids)] if gpu_ids else None
                p, q = launch(repeat, gpu)
                running[slot] = (repeat, p, q)
        progressed = False
        for slot, (repeat, p, q) in list(running.items()):
            try:
                got = q.get_nowait()
            except queue.Empty:
                if p.is_alive():
                    continue
                try:
                    got = q.get(timeout=grace)
                except queue.Empty:
                    p.join()
                    del running[slot]
                    failed.append((repeat, p.exitcode))
                    log(f"!!! repeat {repeat} died (exit code {p.exitcode}) without returning "
                        f"metrics -- see logs/repeat_{repeat}.log. The other repeats continue; "
                        f"an abort inside a CUDA teardown looks exactly like this.", flush=True)
                    progressed = True
                    continue
            p.join()
            del running[slot]
            r_repeat, result = got
            log(f"--- repeat {r_repeat} done (see logs/repeat_{r_repeat}.log) ---", flush=True)
            collect(r_repeat, result)
            progressed = True
        if not progressed:
            time.sleep(poll)
    return failed


def build_parser() -> argparse.ArgumentParser:
    """The CLI, separated from `main` so other tools can read its defaults.

    `relaunch_incomplete.py` rebuilds a command line out of a finished run's
    `config.json`, and to do that it has to know which recorded value is a default
    worth omitting and which flag spells a given dest.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="curated_full", choices=["curated_full", "full_full", "ofm"])
    ap.add_argument("--task", default=None, choices=list(TASKS),
                     help="classification (default, M2OR's 0/1 Responsive) or regression "
                          "(the continuous z-scored response of the Carey/Hallem datasets). "
                          "Switches the boosting head, the metric family, the ensemble "
                          "weight fitters, and every cls extractor's criterion. "
                          "--regime ofm defaults to regression.")
    ap.add_argument("--source", action="append", required=True, dest="sources",
                     help="name=type:path[:model_name[:pooling]], repeatable; order fixes combo digits")
    ap.add_argument("--combos", required=True, help='e.g. "1 2 12"')
    ap.add_argument("--weight-method", default="both",
                     choices=["both", "simplex", "logreg", "linreg"],
                     help="stacker over the per-combo predictions. 'logreg' is the "
                          "classification stacker, 'linreg' its least-squares regression "
                          "counterpart; 'both' picks the right pair for --task. With a "
                          "single combo the simplex is degenerate and equals that combo.")
    ap.add_argument("--on-missing", default="raise", choices=["raise", "drop"],
                     help="raise: fail loudly on any embedding gap (default). "
                          "drop: warn with coverage %% and drop uncovered rows, "
                          "same reduced set for every combo + the ensemble.")
    ap.add_argument("--tune-boost", action="store_true",
                     help="tune each combo's XGBoost head independently via optuna "
                          "(see orbind/baselines.py::tune_boost) instead of fixed "
                          "hyperparameters. Off by default -- multiplies runtime by "
                          "roughly --n-trials per combo per repeat.")
    ap.add_argument("--n-trials", type=int, default=30,
                     help="optuna trials per combo when --tune-boost is set (default 30)")
    ap.add_argument("--out-dir", default="results/ensemble_logs",
                     help="base dir; each run gets its own timestamped subfolder here")
    ap.add_argument("--run-name", default=None,
                     help="override the run folder name (default: <regime>-<split_or_mode>-<timestamp>)")
    ap.add_argument("--skip-checkpoints", action="store_true",
                     help="don't save boosters/torch weights, only metrics (smaller, faster to iterate)")
    ap.add_argument("--max-parallel", type=int, default=1,
                     help="run this many repeats concurrently, each in its own OS process "
                          "(separate CUDA context). Default 1 (sequential). "
                          "Independent of OOF parallelism, which is always on inside each repeat.")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                     help="GPU ids to round-robin repeats across when --max-parallel > 1, "
                          "e.g. --gpus 0 1. Default: auto-detect all visible GPUs; "
                          "with 0 or 1 visible, no pinning (all workers share the default device).")

    g1 = ap.add_argument_group("curated_full")
    g1.add_argument("--pairs", default="data/processed/pairs_curated.csv")
    g1.add_argument("--split", default="stratified", choices=["stratified", "group_molecule", "group_receptor"])
    g1.add_argument("--seeds", type=int, nargs="+", default=[42])
    g1.add_argument("--test-size", type=float, default=0.2)
    g1.add_argument("--val-size", type=float, default=0.2)

    g2 = ap.add_argument_group("full_full")
    g2.add_argument("--full-full-mode", default="transductive",
                     choices=["transductive", "inductive_molecule", "inductive_molecule_v5"])
    g2.add_argument("--repeats", type=int, nargs="+", default=None,
                     help="fold 1-5 for transductive, cold-molecule seed for the inductive modes, "
                          "fold 1-5 for --regime ofm. Default: 1..5 / 42..46 respectively.")

    g3 = ap.add_argument_group("ofm (Carey / Hallem-Carlson)")
    g3.add_argument("--dataset", default="cc", choices=sorted(OFM_DATASETS),
                     help="cc = Carey (50 receptors x 110 odorants), "
                          "hc = Hallem-Carlson (24 x 110)")
    g3.add_argument("--split-family", default="rand",
                     choices=["rand", "cdhit", "scaf", "our_inductive"],
                     help="upstream's own split families: rand = i.i.d. (transductive), "
                          "cdhit = unseen receptors, scaf = unseen odorants. "
                          "HC ships only rand. our_inductive is ours: unseen odorants "
                          "again, but stratified by response dynamic range so no fold is "
                          "degenerate (scaf fold 1 has test sd 0.215 and naive R2 -4.92); "
                          "build it with scripts/preprocessing/"
                          "03_build_ofm_our_inductive_splits.py.")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # Fail here, not three hours from now inside an XGBoost destructor.
    xgb_version = check_xgboost_version()

    # The ofm datasets exist for their continuous response; defaulting them to
    # classification would silently binarise the very thing they were fetched for.
    if args.task is None:
        args.task = "regression" if args.regime == "ofm" else "classification"

    if args.regime == "ofm":
        families = available_families(args.dataset)
        if args.split_family not in families:
            ap.error(f"--dataset {args.dataset} ships no {args.split_family!r} splits "
                     f"(upstream released only {families} for it)")

    if args.regime == "curated_full":
        tag = args.split
    elif args.regime == "ofm":
        tag = f"{args.dataset}-{args.split_family}"
    else:
        tag = args.full_full_mode
    run_id = args.run_name or f"{args.regime}-{tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _root / args.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir}")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        # `python` and `xgboost` are not CLI args: they record WHICH environment
        # produced these numbers, which is exactly what we could not reconstruct
        # when the two majors diverged.
        json.dump({**vars(args), "python": sys.executable, "xgboost": xgb_version},
                  f, indent=2, default=str)

    log_path = run_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        old_stdout = sys.stdout
        sys.stdout = _Tee(old_stdout, logf)
        try:
            _run(args, run_dir)
        finally:
            sys.stdout = old_stdout


def _run(args, run_dir) -> None:
    extractors = dict(parse_source_arg(raw) for raw in args.sources)
    for name, ex in extractors.items():
        dim_out = getattr(ex, "dim_out", getattr(ex, "dim", "?"))
        print(f"source {name!r}: type={type(ex).__name__} model={ex.model_name} "
              f"pooling={ex.pooling} dim_out={dim_out} path={ex.path}")

    print(f"task: {args.task}")
    if args.regime == "curated_full":
        pairs_path = pathlib.Path(args.pairs)
        if not pairs_path.is_absolute():
            pairs_path = _root / pairs_path
        pairs = pd.read_csv(pairs_path)
        print(f"pairs: {len(pairs)} rows from {pairs_path}")
        repeats = args.seeds
    elif args.regime == "ofm":
        pairs = ofm_pairs(args.dataset)
        y = pairs["label"]
        print(f"pairs: {len(pairs)} rows from orbind.regimes_ofm.ofm_pairs({args.dataset!r}) "
              f"-- {pairs['receptor'].nunique()} receptors x {pairs['inchikey'].nunique()} odorants, "
              f"split family {args.split_family!r}")
        print(f"  label: continuous, mean={y.mean():.3f} sd={y.std():.3f} "
              f"min={y.min():.3f} max={y.max():.3f}")
        repeats = args.repeats or OFM_FOLDS
    else:
        pairs = full_full_pairs()
        print(f"pairs: {len(pairs)} rows from orbind.regimes.full_full_pairs()")
        repeats = args.repeats or DEFAULT_REPEATS[args.full_full_mode]

    save_checkpoints = not args.skip_checkpoints

    optuna_storage = None
    if args.tune_boost:
        db_path = (run_dir / "optuna_studies.db").as_posix()
        optuna_storage = f"sqlite:///{db_path}"
        # Create the schema (tables) right now, before any repeat starts --
        # otherwise it's only lazily created on the first real tune_boost
        # call (which may be minutes away, behind the cls model's own
        # training), and `optuna-dashboard` opened before that point fails
        # with "no such table: version_info" on the still-schemaless file.
        import optuna as _optuna
        _optuna.create_study(storage=optuna_storage, study_name="_init", load_if_exists=True)
        print(f"\noptuna storage: {optuna_storage}")
        print(f"  live dashboard: optuna-dashboard {optuna_storage!r}")

    # metrics.csv is written incrementally, one repeat at a time, so a run
    # killed partway through (e.g. server preemption) doesn't lose already-
    # finished repeats -- rerunning the same command later only has to redo
    # (cheaply, thanks to checkpoint/optuna reuse -- see attention_extractor's
    # and gnn_extractor's checkpoint_path handling and baselines.tune_boost's
    # trial-count resume) whatever this repeat's combos hadn't finished yet.
    # A repeat's rows fully replace any earlier rows for that same repeat
    # (this run's --combos is authoritative for what "done" means now); any
    # existing metrics.csv is preserved first as metrics.csv.bak.
    metrics_path = run_dir / "metrics.csv"
    all_rows = []
    if metrics_path.exists():
        shutil.copy(metrics_path, run_dir / "metrics.csv.bak")
        all_rows = pd.read_csv(metrics_path).to_dict("records")

    def collect(repeat, result):
        rows = []
        for combo, m in result["combos"].items():
            rows.append({"repeat": repeat, "kind": "combo", "name": "+".join(combo), **m})
        for method, m in result["ensemble"].items():
            w = result["weights"][method]
            rows.append({
                "repeat": repeat, "kind": "ensemble", "name": f"ensemble[{method}]", **m,
                "weights": json.dumps({"+".join(c): round(v, 4) for c, v in w.items()}),
            })
        # The no-information floor gets its own row: under regression an R2 is
        # unreadable without it (see orbind/dataset.py::regression_metrics).
        if "naive" in result:
            rows.append({"repeat": repeat, "kind": "naive", "name": "naive[train-mean]",
                          **result["naive"]})
        all_rows[:] = [r for r in all_rows if r["repeat"] != repeat] + rows
        pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
        print(f"  wrote -> {metrics_path} ({len(all_rows)} rows so far)", flush=True)

    if args.max_parallel <= 1:
        for repeat in repeats:
            print(f"\n--- repeat {repeat} (see logs/repeat_{repeat}.log) ---", flush=True)
            _, result = _run_one_repeat(args.regime, pairs, extractors, args.combos, args.split, repeat,
                                         args.test_size, args.val_size, args.weight_method,
                                         args.on_missing, args.full_full_mode, run_dir, save_checkpoints,
                                         args.tune_boost, args.n_trials, optuna_storage,
                                         args.task, args.dataset, args.split_family)
            collect(repeat, result)
    else:
        gpu_ids = _detect_gpus(args.gpus)
        # "spawn", not the platform default -- on Linux that default is "fork", and
        # _detect_gpus above already touched torch.cuda in this (parent) process to
        # count devices; forking a child that inherits an initialized CUDA context
        # is unsupported and fails with "CUDA error: initialization error". Windows
        # already defaults to spawn, which is why this only surfaces on Linux.
        ctx = multiprocessing.get_context("spawn")

        # Deliberately NOT ProcessPoolExecutor(initializer=...): that initializer
        # only runs *after* the child has already imported this module to be able
        # to unpickle the initializer/task callables in the first place, and that
        # import chain (torch_geometric -> torch_scatter/torch_sparse) can touch
        # CUDA and bind the process to the default device before the initializer
        # ever gets a chance to restrict CUDA_VISIBLE_DEVICES -- confirmed on the
        # real server: the pinning log line printed the right GPU per repeat, but
        # nvidia-smi showed every worker on GPU0 regardless. Setting the env var
        # in the *parent*, immediately before each ctx.Process(...).start(), bakes
        # it into that child's environment before its interpreter (and therefore
        # any import) even begins, which does work.
        if gpu_ids:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log; "
                  f"pinned round-robin across GPUs {gpu_ids} via CUDA_VISIBLE_DEVICES set "
                  f"before each process starts)...", flush=True)
        else:
            print(f"\nrunning {len(repeats)} repeats, up to {args.max_parallel} concurrently "
                  f"(separate processes; each writes logs/repeat_{{R}}.log)...", flush=True)

        def launch(repeat, gpu):
            if gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            q = ctx.Queue()
            p = ctx.Process(target=_run_one_repeat_to_queue, args=(q, args.regime, pairs, extractors,
                                                                    args.combos, args.split, repeat,
                                                                    args.test_size, args.val_size,
                                                                    args.weight_method, args.on_missing,
                                                                    args.full_full_mode, run_dir,
                                                                    save_checkpoints, args.tune_boost,
                                                                    args.n_trials, optuna_storage,
                                                                    args.task, args.dataset,
                                                                    args.split_family))
            p.start()
            print(f"  started repeat {repeat}" + (f" on GPU {gpu}" if gpu is not None else ""),
                  flush=True)
            return p, q

        failed = _run_repeat_pool(repeats, args.max_parallel, gpu_ids, launch, collect)
        if failed:
            raise SystemExit(
                f"{len(failed)} of {len(repeats)} repeats died without returning metrics: "
                + ", ".join(f"repeat {r} (exit {c})" for r, c in failed)
                + f". The others are in {metrics_path}; rerun the same command to redo the "
                  f"missing ones -- trained models load from checkpoints/.")

    df = pd.DataFrame(all_rows)
    print(f"\nfinal -> {metrics_path} ({len(df)} rows)")
    summary_cols = (["R2", "RMSE", "Pearson", "Spearman"] if args.task == "regression"
                    else ["AUROC", "AUPRC", "MCC", "F1"])
    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df.groupby(["kind", "name"])[summary_cols].mean().round(3))
    if save_checkpoints:
        print(f"checkpoints -> {run_dir / 'checkpoints'}/repeat_*/")


if __name__ == "__main__":
    main()
