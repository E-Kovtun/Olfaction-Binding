#!/usr/bin/env python
"""Structure x function: a grid of graphs, each one a point in (k, phi).

The mechanism holdout compares three fixed representations -- raw ESM, GNN one-hot, GNN+ESM --
and when two of them tie there is nothing more to say. This turns that A/B/C into a surface by
giving each source its own knob, so "does the mix beat both pure sources" becomes a question
about the SHAPE of a curve rather than about which of two numbers is larger.

    k    how many PCA components of ESM the receptor NODE FEATURES carry.
         k = 0 is a one-hot identity (the pipeline's `feat="onehot"` arm); k = all is the
         full space (the `GNN+ESM` arm). Rank truncation, not a weight: a scalar on a skip
         connection is undone by the network rescaling itself, while discarded components
         cannot be recovered.
    phi  the fraction of MESSAGE-PASSING edges kept, sampled at random and stratified by
         sign. phi = 1 is the full binding graph, phi = 0 removes message passing entirely,
         leaving a learned transform of the node features.

The two are the same kind of knob pointed at opposite sources: each destroys information
rather than reweighting it, which is what makes the axis mean something.

Why this can be done at all without disturbing the training signal: `_mp_edges` says it in
its own docstring -- molecules are dropped "from message passing only; every train row still
gets decoded/supervised regardless". Message-passing edges and supervised pairs are separate
sets, so phi throttles what reaches the representation while the loss still sees every pair.
Otherwise "less function" would be confounded with "less data".

Two corners have known answers and exist to test the readout, not the model. Measured on
HC/aromatic at 5 epochs, as z against each cell's own permutation null:

    (k = all, phi = 0)   no messages -- pinned to ESM (z = +14.7) and at the null on the
                         class target (z = +0.6): pure structure knows nothing about a class
    (k = 0,   phi = 1)   the one-hot arm -- at the null on ESM (z = +0.6), functional (+2.7)

The third corner is NOT a null floor, which is worth stating because it looks like one.
(k = 0, phi = 0) has neither node information nor messages, but the decoder still supervises
on every pair -- that is the property that makes phi a clean knob -- so what it learns is a
plain factorisation of the training labels. It scores z = +4.9 against the functional
reference and +2.6 against the target. Read it as a third baseline, "labels without message
passing", not as noise. The noise floor is the permutation null in nulls.csv, which is
computed per cell precisely because no cell provides one.

READOUTS. Each cell's receptor embedding is scored with the same three geometries the
holdout uses (RSA, CCA, Procrustes) against three clouds:

    esm     the raw ESM vectors            "how structural did this come out"
    func    the retained response profile  "how functional did this come out"
    target  the held-out class's profile   "and is it any good"

The first two are a manipulation check: they say where the knobs actually put the
representation, which is not the same as where they were set to put it. The third is the
result. Plotting the third against (func - esm) turns a grid of settings into a trajectory in
representation space, and an interior maximum there is what complementarity looks like.

Insects only. On M2OR the functional reference is contaminated by assay design (see the
`tested mask` control in mechanism_holdout), so "how functional did this come out" has no
clean meaning there.

    .venv/bin/python scripts/modeling/analysis/structure_function_grid.py --dataset hc
    .venv/bin/python scripts/modeling/analysis/structure_function_grid.py --dataset hc --trajectory full

Resumable and append-only: a cell already in grid.csv is skipped, and every cell is written
as soon as it finishes. A run killed at hour three keeps hour three.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from scripts.modeling.analysis.mechanism_holdout import (  # noqa: E402
    DATA, DATASETS, GEOMETRY, VARIANTS, _pca, class_members, class_targets, func_redund,
    geometry_nulls, load_dataset, retained_profile, struct_leak, trust)

# -1 means "every component". Logarithmic on purpose: PCA concentrates information in the
# leading components and edge removal only bites near zero, so linear steps waste runs.
K_ALL = [0, 2, 4, 8, 16, 32, 64, 128, -1]
PHI_ALL = [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
K_COARSE = [0, 2, 8, 32, 128, -1]
PHI_COARSE = [0.0, 0.05, 0.25, 0.5, 1.0]

VARIANT = VARIANTS["q0cov"]      # the insects' graph; the quantile cut is the other knob
REFS = ("esm", "func", "target")
KEY = ["k", "phi", "cls", "seed"]


def cells(traj):
    """The (k, phi) cells of a trajectory, in the order they should be run."""
    if traj == "edges":
        # one arm per knob, sharing the (k=all, phi=1) corner: each answers "does THIS source
        # help, holding the other at full" before any interior cell is paid for
        return ([(k, 1.0) for k in K_COARSE] +
                [(-1, phi) for phi in PHI_COARSE if phi != 1.0])
    if traj == "diagonal":
        # trade one source for the other, from pure structure to pure function
        ks = [-1, 32, 8, 2, 0]
        phis = [0.0, 0.05, 0.25, 0.5, 1.0]
        return list(zip(ks, phis))
    if traj == "full":
        return [(k, phi) for k in K_ALL for phi in PHI_ALL]
    raise SystemExit(f"unknown trajectory {traj!r}")


def node_features(proteins, k):
    """Receptor node features at rank `k`: one-hot at 0, PCA scores otherwise.

    PCA scores throughout rather than raw vectors at the top end, so the axis is one
    construction instead of two. At full rank that is a rotation of the raw space, which
    every readout here is invariant to (RSA is cosine-based; CCA and Procrustes reduce both
    sides first), though the trained network can differ slightly because initialisation meets
    a different basis.
    """
    recs = sorted(proteins)
    if k == 0:
        eye = np.eye(len(recs), dtype=np.float32)
        return {r: eye[i] for i, r in enumerate(recs)}, len(recs)
    X = np.stack([proteins[r] for r in recs])
    kk = X.shape[1] if k < 0 else int(min(k, len(recs) - 1, X.shape[1]))
    Z = _pca(X, kk).astype(np.float32) if kk < X.shape[1] else X.astype(np.float32)
    return {r: Z[i] for i, r in enumerate(recs)}, kk


def thin_edges(pos, neg, phi, seed):
    """Keep a random `phi` of the message-passing edges, stratified by sign.

    Stratified because an unstratified draw at small phi can leave a graph that is almost
    all one sign, which changes what the model is looking at rather than how much of it.
    """
    if phi >= 1.0:
        return pos, neg
    rng = np.random.default_rng(seed)

    def cut(e):
        n = len(e[0])
        keep = rng.random(n) < phi
        w = e[2][keep] if len(e) > 2 and e[2] is not None else None
        return (e[0][keep], e[1][keep], w)

    return cut(pos), cut(neg)


def train_cell(spec, p, drop_iks, k, phi, seed, epochs, device):
    """One cell: returns ({receptor: vector}, receptor order, edge counts, effective rank)."""
    import torch
    from orbind.gnn_extractor import (GnnSignedExtractor, _build_universe, _mp_edges,
                                      _edge_index_dict, _train_one)
    ext = GnnSignedExtractor(
        name="cls", protein_path=str(DATA / spec["prot"]), molecule_path=str(DATA / spec["mol"]),
        emit="prot", n_models=1, hidden=256, edge_threshold=0.0, task=spec["task"],
        epochs=epochs, **VARIANT)
    ext._proteins, k_eff = node_features(ext._proteins, k)
    idx = np.where((~p["inchikey"].isin(drop_iks)).to_numpy())[0]
    idx = idx[ext.covered(p, idx)]
    m2i, p2i, xm, xp = _build_universe(p, idx, ext._proteins, ext._molecules)
    pos, neg = _mp_edges(p, idx, m2i, p2i, ext.q, ext.criterion,
                         ext.edge_threshold, ext.k_mode, ext.task)
    pos, neg = thin_edges(pos, neg, phi, seed)
    pe, ne = _edge_index_dict(pos, neg)
    sub = p.iloc[idx]
    torch.manual_seed(seed)
    _, zp, _ = _train_one(ext._build_model, xm, xp, pe, ne,
                          sub["inchikey"].map(m2i).to_numpy(),
                          sub["receptor"].map(p2i).to_numpy(),
                          sub["label"].to_numpy(np.float32), ext._hp(seed), device, None)
    return ({r: zp[i] for r, i in p2i.items()}, p2i,
            dict(n_pos=int(len(pos[0])), n_neg=int(len(neg[0])), k_eff=int(k_eff)))


def run(ds, args):
    import torch
    if DATASETS[ds]["kind"] != "complete_continuous":
        raise SystemExit(f"{ds} is not an insect panel: the functional reference is the "
                         f"response profile, which on a sparse matrix carries assay design "
                         f"(see the `tested mask` control). Use cc or hc.")
    spec = DATASETS[ds]
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = pathlib.Path(args.out) / ds
    out.mkdir(parents=True, exist_ok=True)
    gpath, npath, epath = out / "grid.csv", out / "nulls.csv", out / "embeddings.npz"

    done = set()
    rows = []
    if gpath.exists() and not args.force:
        old = pd.read_csv(gpath)
        rows = old.to_dict("records")
        done = {(float(r["k"]), float(r["phi"]), r["cls"], int(r["seed"])) for r in rows}
    nulls = pd.read_csv(npath).to_dict("records") if npath.exists() and not args.force else []
    emb = {}
    if epath.exists() and not args.force:
        with np.load(epath, allow_pickle=True) as z:
            emb = {kk: z[kk] for kk in z.files}

    p, recs, ods, R, smi = load_dataset(ds)
    names = args.classes or spec["classes"]
    classes = class_members(ods, smi, names, spec["min_members"])
    todo = cells(args.trajectory) if not args.k else [(k, phi) for k in args.k for phi in args.phi]
    print(f"\n=== {ds.upper()} === {len(recs)} receptors x {len(ods)} odorants | device {device}"
          f"\n    classes {list(classes)} | seeds {args.seeds}"
          f"\n    trajectory {args.trajectory}: {len(todo)} cells "
          f"-> {len(todo) * len(classes) * len(args.seeds)} trainings, "
          f"{len(done)} already done", flush=True)

    t0 = time.time()
    for cname, iks in classes.items():
        # the three reference clouds are fixed for a class: only the embedding moves
        tg = class_targets(spec["kind"], p, R, recs, ods, [r for r in recs], iks)
        order = tg["order"]
        gin = None
        prot = np.load(DATA / spec["prot"], allow_pickle=True)
        ref = {"esm": np.stack([prot[r] for r in order]).astype(np.float64),
               "func": retained_profile(R, recs, ods, order, iks),
               "target": np.asarray(tg["rsa_target"], np.float64)}
        prot.close()
        for k, phi in todo:
            for seed in args.seeds:
                if (float(k), float(phi), cname, int(seed)) in done:
                    continue
                c0 = time.time()
                vecs, p2i, info = train_cell(spec, p, iks, k, phi, seed, args.epochs, device)
                miss = [r for r in order if r not in vecs]
                if miss:
                    print(f"  {cname:14} k={k} phi={phi} seed={seed}: {len(miss)} receptors "
                          f"absent from the graph -- skipped", flush=True)
                    continue
                Z = np.stack([vecs[r] for r in order]).astype(np.float64)
                row = dict(k=k, phi=phi, cls=cname, seed=seed, **info,
                           n_receptors=len(order), dim_z=int(Z.shape[1]),
                           seconds=round(time.time() - c0, 1))
                for rname, M in ref.items():
                    for g, fn in GEOMETRY.items():
                        row[f"{g}__{rname}"] = float(fn(Z, M))
                rows.append(row)
                done.add((float(k), float(phi), cname, int(seed)))
                if seed == args.seeds[0]:
                    for rname, M in ref.items():
                        st = geometry_nulls(Z, M, args.n_perm)
                        nulls.append(dict(k=k, phi=phi, cls=cname, ref=rname,
                                          **{f"{g}_null": m for g, (m, _) in st.items()},
                                          **{f"{g}_null_sd": sd for g, (_, sd) in st.items()}))
                if args.embeddings:
                    emb[f"z__{cname}__k{k}__p{phi}__s{seed}"] = Z.astype(np.float16)
                    for rname, M in ref.items():
                        emb[f"ref__{cname}__{rname}"] = np.asarray(M, np.float16)
                    emb[f"order__{cname}"] = np.array(order, dtype=np.str_)
                # written per cell: a run killed at hour three keeps hour three
                pd.DataFrame(rows).to_csv(gpath, index=False)
                pd.DataFrame(nulls).to_csv(npath, index=False)
                if args.embeddings:
                    np.savez_compressed(epath, **emb)
                print(f"  {cname:14} k={str(k):>4} phi={phi:<5} seed={seed}  "
                      f"edges {info['n_pos']}+{info['n_neg']}  dim {Z.shape[1]}  "
                      + "  ".join(f"{g[:4]}/{r[:4]} {row[f'{g}__{r}']:+.3f}"
                                  for r in REFS for g in ("rsa",))
                      + f"   {time.time() - c0:.0f}s", flush=True)

    meta = dict(dataset=ds, variant=VARIANT, epochs=args.epochs, seeds=args.seeds,
                classes=list(classes), n_perm=args.n_perm,
                k_axis=K_ALL, phi_axis=PHI_ALL, refs=list(REFS),
                covered=sorted({(float(r["k"]), float(r["phi"])) for r in rows}),
                seconds=round(time.time() - t0, 1))
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"=== {ds.upper()} {len(rows)} rows over "
          f"{len(meta['covered'])} cells -> {out} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=["hc"], help="cc / hc (insects only)")
    ap.add_argument("--trajectory", default="edges", choices=["edges", "diagonal", "full"],
                    help="edges: one arm per knob (cheapest, run first). diagonal: trade one "
                         "for the other. full: the product of both axes")
    ap.add_argument("--k", type=int, nargs="+", default=None,
                    help="explicit k values (-1 = every component); overrides --trajectory")
    ap.add_argument("--phi", type=float, nargs="+", default=[1.0],
                    help="explicit phi values, used with --k")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="which held-out classes (default: all of the dataset's). Pick the "
                         "best-isolated ones first -- and not `alcohol`, which is 45%% of the "
                         "panel and contains every acid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--no-embeddings", dest="embeddings", action="store_false")
    ap.add_argument("--force", action="store_true", help="recompute cells already in grid.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/sf_grid")
    args = ap.parse_args()
    for ds in args.dataset:
        if ds not in DATASETS:
            raise SystemExit(f"unknown dataset {ds!r}")
        run(ds, args)


if __name__ == "__main__":
    main()
