"""Generate mean-pooled protein embeddings from a modern pLM, keyed BY SEQUENCE.

Output matches the existing protein npz convention exactly -- `ids` = the raw
receptor sequence string (the same value `pairs["receptor"]` holds), `emb` =
float32 [n, d] -- so the file drops straight into the pipeline as
`prot=esm:<npz>` and into `orbind.dataset.load_npz_dict`.

Two backends, lazy-imported so each runs in its own env:
  * prott5 -- Rostlab/prot_t5_xl_half_uniref50-enc (HuggingFace T5 encoder,
    1024-d). Env: transformers + sentencepiece + torch.
  * esmc   -- EvolutionaryScale ESM-C 600M (`esm` SDK, 1152-d). Env: a FRESH
    venv with `pip install esm` (its package name clashes with fair-esm, so it
    must NOT share the env that builds ESM-1b/ESM-2).

Sequences come from the repo's own loaders, so coverage spans M2OR + the insect
datasets with no extra bookkeeping:
  m2or -> orbind.regimes.full_full_pairs   (1237 receptors)
  cc   -> orbind.regimes_ofm.ofm_pairs cc  (50)
  hc   -> orbind.regimes_ofm.ofm_pairs hc  (24)

    # ProtT5 (in the transformers env):
    python scripts/embedding_generation/proteins/embed_proteins_plm.py --model prott5 --dataset all
    # ESM-C (in .venv-esmc):
    python scripts/embedding_generation/proteins/embed_proteins_plm.py --model esmc   --dataset all
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
    model = T5EncoderModel.from_pretrained(card).to(device).eval()
    if device == "cuda":
        model = model.half()
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


# -------------------------------------------------------------------- ESM-C ---
def embed_esmc(seqs, model_name="esmc_600m"):
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMC.from_pretrained(model_name).to(device).eval()
    cfg = LogitsConfig(sequence=True, return_embeddings=True)
    out = []
    for i, s in enumerate(seqs):
        tok = model.encode(ESMProtein(sequence=s))
        with torch.no_grad():
            emb = model.logits(tok, cfg).embeddings[0]   # [L+2, d] incl BOS/EOS
        out.append(emb[1:-1].float().mean(0).cpu().numpy())
        print(f"  esmc {i + 1}/{len(seqs)}", end="\r", flush=True)
    print()
    return np.stack(out).astype(np.float32)


BACKENDS = {"prott5": embed_prott5, "esmc": embed_esmc}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--dataset", default="all", help="m2or | cc | hc | all")
    ap.add_argument("--out-tag", default=None, help="npz stem (default = model name)")
    args = ap.parse_args()

    datasets = ["m2or", "cc", "hc"] if args.dataset == "all" else [args.dataset]
    tag = args.out_tag or args.model
    embed = BACKENDS[args.model]
    outdir = _root / "data" / "embeddings" / "proteins"
    outdir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        seqs = sequences_for(ds)
        print(f"[{args.model}] {ds}: {len(seqs)} unique receptor sequences", flush=True)
        emb = embed(seqs)
        out = outdir / f"{tag}_{ds}.npz"
        np.savez_compressed(out, ids=np.array(seqs, dtype=object), emb=emb)
        print(f"  saved {emb.shape[0]} x {emb.shape[1]} float32 -> {out}", flush=True)


if __name__ == "__main__":
    main()
