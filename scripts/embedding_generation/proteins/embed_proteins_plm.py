"""Generate mean-pooled protein embeddings from a modern pLM, keyed BY SEQUENCE.

Output matches the existing protein npz convention exactly -- `ids` = the raw
receptor sequence string (the same value `pairs["receptor"]` holds), `emb` =
float32 [n, d] -- so the file drops straight into the pipeline as
`prot=esm:<npz>` and into `orbind.dataset.load_npz_dict`.

Three backends, lazy-imported so each runs in its own env:
  * prott5 -- Rostlab/prot_t5_xl_half_uniref50-enc (HuggingFace T5 encoder,
    1024-d). Env: transformers + sentencepiece + torch.
  * esmc   -- EvolutionaryScale ESM-C 600M (`esm` SDK, 1152-d).
  * esm3   -- ESM3 open, `esm3-sm-open-v1` (1.4B, `esm` SDK). The only ESM3 whose
    weights are public; medium/large are API-only. Sequence track only: no
    structure or function prompt is given, so this is ESM3 read as a sequence
    encoder -- the same footing as every other pLM in the table.
  esmc and esm3 share one env, separate from everything else: the SDK's package
  is ALSO called `esm` and clashes with fair-esm, so it must never share the env
  that builds ESM-1b/ESM-2. On the server:

      uv venv .venv-esm --python 3.11
      uv pip install --python .venv-esm/bin/python torch --index-url https://download.pytorch.org/whl/cu124
      uv pip install --python .venv-esm/bin/python esm pandas scikit-learn

  The weights download from HuggingFace on first use; if that answers 401,
  `export HF_TOKEN=<read token>` and rerun.

Outputs, under data/embeddings/proteins/:
  {tag}_{ds}.npz              mean over residues, `ids`/`emb` as above. This is the
                              file every mean consumer reads: `prot=esm:<npz>`, the
                              sweep's --prot-embeddings, hladis, the protein-floor
                              and geometry tables.
  {tag}_per_residue_{ds}.npz  with --per-residue (esmc/esm3 only): {sequence:
                              float32[L, d]}, the shape ESM-1b's per-residue file
                              has, for the cross-attention baselines (LORAX,
                              ProSmith, MolOR). The mean file is then DERIVED from
                              it in the same pass, so the two cannot disagree --
                              script 05's single-source-of-truth rule.

Sequences come from the repo's own loaders, so coverage spans M2OR + the insect
datasets with no extra bookkeeping:
  m2or -> orbind.regimes.full_full_pairs   (1237 receptors)
  cc   -> orbind.regimes_ofm.ofm_pairs cc  (50)
  hc   -> orbind.regimes_ofm.ofm_pairs hc  (24)
The shrunk panels need no files of their own: they read their parent's.

    # ProtT5 (in the transformers env):
    python scripts/embedding_generation/proteins/embed_proteins_plm.py --model prott5 --dataset all
    # ESM-C / ESM3 (in .venv-esm):
    python scripts/embedding_generation/proteins/embed_proteins_plm.py --model esmc --dataset all
    python scripts/embedding_generation/proteins/embed_proteins_plm.py --model esm3 --dataset all --per-residue
"""
import argparse
import pathlib
import re
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))


def sequences_for(dataset: str):
    """Unique receptor sequences (raw, as they appear in `pairs['receptor']`)."""
    if dataset == "m2or":
        from orbind.regimes import full_full_pairs
        seqs = full_full_pairs(pool_fold=1)["receptor"].unique().tolist()
    elif dataset in ("cc", "hc"):
        from orbind.regimes_ofm import ofm_pairs
        seqs = ofm_pairs(dataset)["receptor"].unique().tolist()
    else:
        raise SystemExit(f"unknown dataset {dataset!r} (m2or/cc/hc)")
    return [s for s in seqs if isinstance(s, str) and s]


# ------------------------------------------------------------------ ProtT5 ----
def embed_prott5(seqs, batch=8):
    import torch
    from transformers import T5EncoderModel, T5Tokenizer
    card = "Rostlab/prot_t5_xl_half_uniref50-enc"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = T5Tokenizer.from_pretrained(card, do_lower_case=False, legacy=True)
    if device == "cuda":
        # This container's /proc/cpuinfo is unparseable, so ANY CPU tensor
        # compute (here T5's forced fp16->fp32 upcast of its layer norms during
        # load) dies with "Failed to initialize cpuinfo!". Bypass the CPU
        # entirely: init the model on `meta` (no allocation, no CPU op), read the
        # safetensors weights STRAIGHT onto the GPU, and assign them. Every op
        # touches only cuda; GPU forward + .cpu() on the OUTPUT are proven fine
        # (the GNN pipeline does exactly that on this box).
        from transformers import T5Config
        from transformers.utils import cached_file
        cfg = T5Config.from_pretrained(card)
        with torch.device("meta"):
            model = T5EncoderModel(cfg)
        # Resolve weights via transformers' OWN cache logic (same one
        # from_pretrained used). This repo ships pytorch_model.bin (no
        # safetensors); torch.load(map_location="cuda") lands every tensor
        # straight on the GPU, keeping fp16 -- so no CPU fp16->fp32 cast, which
        # is the exact op this container's /proc/cpuinfo kills.
        try:
            from safetensors.torch import load_file
            sd = load_file(cached_file(card, "model.safetensors"), device="cuda")
        except Exception:
            sd = torch.load(cached_file(card, "pytorch_model.bin"),
                            map_location="cuda", weights_only=True)
        model.load_state_dict(sd, strict=False, assign=True)
        model.tie_weights()                       # re-link encoder.embed_tokens -> shared
        leftover = [n for n, p in model.named_parameters() if p.is_meta]
        if leftover:
            raise RuntimeError(f"{len(leftover)} params never loaded (still meta): {leftover[:5]}")
        model = model.eval()
    else:
        model = T5EncoderModel.from_pretrained(card).to(device).eval()
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        prepped = [" ".join(re.sub(r"[UZOB]", "X", s)) for s in chunk]
        enc = tok(prepped, add_special_tokens=True, padding="longest", return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        for k in range(len(chunk)):
            L = int(mask[k].sum().item()) - 1     # drop the trailing </s>
            out.append(h[k, :L].float().mean(0).cpu().numpy())
        print(f"  prott5 {min(i + batch, len(seqs))}/{len(seqs)}", end="\r", flush=True)
    print()
    return np.stack(out).astype(np.float32)


# ------------------------------------------------- EvolutionaryScale SDK ------
# ESM-C and ESM3 come out of the same `esm` SDK and answer the same call, encode
# -> logits(return_embeddings=True), so they share one loop. It returns the
# per-residue matrices; the mean is taken by the caller.
def _esm_sdk_per_residue(model, seqs, label):
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig
    cfg = LogitsConfig(sequence=True, return_embeddings=True)
    out = []
    for i, s in enumerate(seqs):
        tok = model.encode(ESMProtein(sequence=s))
        with torch.no_grad():
            emb = model.logits(tok, cfg).embeddings[0]   # [L+2, d] incl BOS/EOS
        # One token per residue plus BOS/EOS is what makes [1:-1] the residues.
        # If a tokenizer ever merged or dropped a character, the slice would
        # still "work" and quietly misalign every residue after it.
        if emb.shape[0] != len(s) + 2:
            raise RuntimeError(f"{label}: {emb.shape[0]} tokens for a {len(s)}-residue "
                               f"sequence (expected {len(s) + 2} with BOS/EOS)")
        out.append(emb[1:-1].float().cpu().numpy())
        print(f"  {label} {i + 1}/{len(seqs)}", end="\r", flush=True)
    print()
    return out


def _device():
    import torch
    # Straight onto the GPU at load: this container dies on CPU tensor compute
    # ("Failed to initialize cpuinfo!", see embed_prott5), and building on the
    # CPU first and moving afterwards is exactly that.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_esmc(seqs, model_name="esmc_600m"):
    from esm.models.esmc import ESMC
    model = ESMC.from_pretrained(model_name, device=_device()).eval()
    return _esm_sdk_per_residue(model, seqs, "esmc")


def embed_esm3(seqs, model_name="esm3-sm-open-v1"):
    from esm.models.esm3 import ESM3
    model = ESM3.from_pretrained(model_name, device=_device()).eval()
    return _esm_sdk_per_residue(model, seqs, "esm3")


BACKENDS = {"prott5": embed_prott5, "esmc": embed_esmc, "esm3": embed_esm3}
# Backends that hand back per-residue matrices rather than finished means.
PER_RESIDUE = {"esmc", "esm3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--dataset", default="all", help="m2or | cc | hc | all")
    ap.add_argument("--out-tag", default=None, help="npz stem (default = model name)")
    ap.add_argument("--per-residue", action="store_true",
                    help="also write {tag}_per_residue_{ds}.npz (esmc/esm3 only)")
    args = ap.parse_args()
    if args.per_residue and args.model not in PER_RESIDUE:
        ap.error(f"--per-residue needs one of {sorted(PER_RESIDUE)}, not {args.model}")

    datasets = ["m2or", "cc", "hc"] if args.dataset == "all" else [args.dataset]
    tag = args.out_tag or args.model
    embed = BACKENDS[args.model]
    outdir = _root / "data" / "embeddings" / "proteins"
    outdir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        seqs = sequences_for(ds)
        print(f"[{args.model}] {ds}: {len(seqs)} unique receptor sequences", flush=True)
        res = embed(seqs)
        if args.model in PER_RESIDUE:
            if args.per_residue:
                out = outdir / f"{tag}_per_residue_{ds}.npz"
                # keyed by sequence, one ragged matrix each -- ESM-1b's per-residue
                # layout, which `load_npz_dict` reads without an `ids` array
                np.savez_compressed(out, **{s: r.astype(np.float32) for s, r in zip(seqs, res)})
                print(f"  saved {len(res)} per-residue x {res[0].shape[1]} float32 -> {out}",
                      flush=True)
            emb = np.stack([r.mean(0) for r in res]).astype(np.float32)
        else:
            emb = res
        out = outdir / f"{tag}_{ds}.npz"
        np.savez_compressed(out, ids=np.array(seqs, dtype=object), emb=emb)
        print(f"  saved {emb.shape[0]} x {emb.shape[1]} float32 -> {out}", flush=True)


if __name__ == "__main__":
    main()
