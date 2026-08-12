"""Script 06 — import ESM-1b per-residue embeddings from the olfactory
foundation models data release, instead of recomputing them.

The upstream benchmarking repo (depasquale-lab/olfactory_foundation_models)
generates its protein side with ESM-**1b** (`preprocess/generate_embeddings.py`:
`pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")`, layer 33) and
ships the result as a torch pickle:

    M2OR_full/embeddings/featurized_proteins/prots.pt   ~3.0 GB
    {protein_sequence: float32[seq_len, 1280]}

That is exactly the input ProSmith (orbind/prosmith_extractor.py) wants, and
its keys are already whole sequences -- the same thing our `pairs["receptor"]`
column holds -- so running the ProSmith baseline on upstream's own embeddings
needs an import, not an ESM run. Script 05 stays the source of truth for
ESM-**2**; this one exists so the external baseline can be reported on the
protein features its authors actually used.

(Upstream truncates sequences to 1018 residues when building its fasta. On
this pool the longest receptor is 705, so no key is ever a truncated
sequence and the import is lossless -- the script checks this rather than
assuming it.)

Where the file comes from
-------------------------
https://zenodo.org/records/17228740 -> data.zip. Note the data link in
upstream's own README points back at the repository itself and is broken;
the working URL is the one above, given in the sibling LORAX repo's README.

    curl -L -o data.zip https://zenodo.org/records/17228740/files/data.zip
    unzip -q data.zip 'M2OR_full/embeddings/featurized_proteins/prots.pt'

Outputs (under data/embeddings/proteins/), matching script 05's convention
-------------------------------------------------------------------------
esm1b_650m_per_residue_full_full.npz   — {sequence: float32[seq_len, 1280]}
esm1b_650m_mean_full_full.npz          — {sequence: float32[1280]}  (derived)

The derived mean is checked against the existing `esm1b_650m_mean.npz` (which
came from LoRaX) when that file is present -- an independent confirmation
that this really is the same ESM-1b layer and not, say, ESM-2.

Run
---
    uv run python scripts/embedding_generation/proteins/06_import_ofm_esm1b.py \\
        --prots-pt M2OR_full/embeddings/featurized_proteins/prots.pt

The same release ships ESM-1b for the other two benchmark datasets, built by
the same generator; those receptors are not in the M2OR pool, so both the
coverage check and the mean cross-check have to be turned off:

    ... --prots-pt data/external/ofm/CC/embeddings/featurized_proteins/prots.pt \\
        --tag cc --pool none --compare-mean ""
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

EMB_DIR = _root / "data" / "embeddings" / "proteins"
UPSTREAM_TRUNCATION = 1018   # preprocess/generate_embeddings.py::create_fasta_file


def load_prots_pt(path: pathlib.Path) -> dict[str, np.ndarray]:
    import torch
    try:
        # torch>=2.6 defaults weights_only=True, which refuses a plain pickle
        # of numpy arrays like this one.
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    out = {}
    for seq, v in raw.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        out[seq] = np.ascontiguousarray(arr, dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prots-pt", default="M2OR_full/embeddings/featurized_proteins/prots.pt",
                    help="upstream prots.pt extracted from data.zip")
    ap.add_argument("--tag", default="full_full", help="output file suffix")
    ap.add_argument("--pool", default="full_full", choices=("full_full", "none"),
                    help='pool to check receptor coverage against; "none" for the '
                         "Carey/Hallem releases, whose receptors are not in the M2OR pool")
    ap.add_argument("--compare-mean", default="data/embeddings/proteins/esm1b_650m_mean.npz",
                    help='existing mean-pooled ESM-1b npz to validate against ("" to skip)')
    args = ap.parse_args()

    src = pathlib.Path(args.prots_pt)
    if not src.is_absolute():
        src = _root / src
    if not src.exists():
        raise FileNotFoundError(f"{src} not found -- see this script's docstring for the download")

    print(f"loading {src} ({src.stat().st_size / 1e9:.2f} GB) …", flush=True)
    per_res = load_prots_pt(src)
    lengths = np.array([v.shape[0] for v in per_res.values()])
    dims = {v.shape[1] for v in per_res.values()}
    print(f"  {len(per_res)} sequences, dim={dims}, "
          f"residues min={lengths.min()} median={int(np.median(lengths))} max={lengths.max()}")
    if dims != {1280}:
        raise ValueError(f"expected 1280-d ESM-1b embeddings, got {dims}")

    # Upstream keys by the *truncated* sequence; if anything actually hit the
    # limit, those keys would not match our pool's full sequences and the
    # import would silently lose receptors.
    truncated = [s for s in per_res if len(s) >= UPSTREAM_TRUNCATION]
    if truncated:
        raise ValueError(f"{len(truncated)} sequences at/over upstream's {UPSTREAM_TRUNCATION}-residue "
                         f"truncation -- keys are not full sequences, import would be lossy")

    # Every key should be a sequence whose length matches its own matrix.
    bad = [s for s, v in per_res.items() if len(s) != v.shape[0]]
    if bad:
        raise ValueError(f"{len(bad)} sequences whose length disagrees with their matrix "
                         f"(e.g. len={len(bad[0])} vs {per_res[bad[0]].shape[0]})")

    # Coverage against the pool this is meant to serve.
    if args.pool == "none":
        print("  (pool coverage check disabled via --pool none)")
    else:
        try:
            from orbind.regimes import full_full_pairs
            receptors = set(full_full_pairs()["receptor"])
            missing = receptors - set(per_res)
            print(f"  pool coverage: {len(receptors) - len(missing)}/{len(receptors)} receptors"
                  + (f"  MISSING {len(missing)}" if missing else ""))
        except Exception as e:                                # pool not reconstructible here
            print(f"  (skipped pool coverage check: {e})")

    mean_emb = {s: v.mean(axis=0).astype(np.float32) for s, v in per_res.items()}

    if args.compare_mean:
        ref_path = _root / args.compare_mean
        if ref_path.exists():
            ref = np.load(ref_path, allow_pickle=True)
            ref = {k: ref[k] for k in ref.files} if "ids" not in ref.files else \
                  dict(zip(ref["ids"].tolist(), ref["emb"]))
            shared = set(ref) & set(mean_emb)
            if shared:
                d = np.array([np.abs(np.asarray(ref[k]) - mean_emb[k]).max() for k in shared])
                print(f"  vs {ref_path.name}: {len(shared)} shared sequences, "
                      f"max|delta| median={np.median(d):.2e} worst={d.max():.2e}")
                if d.max() > 1e-3:
                    print("    NOTE: large disagreement -- these may not be the same ESM variant/layer")
            else:
                print(f"  vs {ref_path.name}: no shared sequences to compare")
        else:
            print(f"  (no {ref_path} to compare against)")

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    out_pr = EMB_DIR / f"esm1b_650m_per_residue_{args.tag}.npz"
    out_mean = EMB_DIR / f"esm1b_650m_mean_{args.tag}.npz"
    print(f"writing {out_pr} (compressing ~{sum(v.nbytes for v in per_res.values()) / 1e9:.2f} GB) …",
          flush=True)
    np.savez_compressed(str(out_pr), **per_res)
    np.savez_compressed(str(out_mean), **mean_emb)
    print(f"-> {out_pr}  ({out_pr.stat().st_size / 1e9:.2f} GB)")
    print(f"-> {out_mean} ({out_mean.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
