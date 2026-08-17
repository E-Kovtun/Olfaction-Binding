"""Protein-representation FLOOR sweep: does a real pLM beat amino-acid counts?

For one dataset x one regime, boosts `[protein_feature || ChemBERTa]` with the
pipeline's own fit_boost/predict_scores (auto-GPU) and reports the fold-mean
metric for each protein feature side by side:

  * real pLMs      -- esm1b, esm2, prott5  (loaded from npz, keyed by SEQUENCE;
                      each is included only if its npz exists AND covers every
                      receptor, so a box that is missing one just skips it)
  * classical floor -- AAC, kmer2, AAindex, CTD, PseAAC, BLOSUM (computed here
                      from the receptor sequence, no learning)
  * controls       -- onehot (identity), onehot_only (no molecule), mol_only.

Regimes match the rest of the pipeline's axis (COLD MOLECULE = inductive), so
this table sits next to every other table in the paper:
  transductive   m2or = LORAX `rand` folds; cc/hc = upstream `rand` (receptors
                 AND molecules seen).
  inductive      COLD MOLECULE -- m2or = `inductive_molecule_v5` (folds 1-5 ->
                 seeds 42-46); cc/hc = `our_inductive`. This is where the graph
                 wins on M2OR, so the floor here says exactly the useful thing:
                 in that regime, swapping ESM for AAC (or even onehot) barely
                 moves the score -> the graph's gain is data-driven receptor
                 refinement, not a better encoder. NB receptors are all SEEN, so
                 the protein-feature spread compresses -- that IS the point.
  cold_receptor  OPTIONAL appendix -- hold whole receptors out (m2or =
                 group_receptor seeds; cc = upstream `cdhit`; HC has none). The
                 protein-transfer axis; not the project's inductive.

    # one cell:
    .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py --dataset m2or --regime transductive
    # everything (skips cells a dataset can't do):
    for d in m2or cc hc; do for r in transductive inductive; do \
      .venv/bin/python scripts/modeling/analysis/prot_floor_sweep.py --dataset $d --regime $r; done; done
"""
import argparse
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

from orbind import dataset as D                                 # noqa: E402
from orbind.baselines import fit_boost, predict_scores          # noqa: E402

AA = "ARNDCQEGHILKMFPSTWYV"
AAi = {a: i for i, a in enumerate(AA)}

# ---------------- physicochemical scales (order ARNDCQEGHILKMFPSTWYV) ----------
KD   = [1.8,-4.5,-3.5,-3.5,2.5,-3.5,-3.5,-0.4,-3.2,4.5,3.8,-3.9,1.9,2.8,-1.6,-0.8,-0.7,-0.9,-1.3,4.2]
VOL  = [88.6,173.4,114.1,111.1,108.5,143.8,138.4,60.1,153.2,166.7,166.7,168.6,162.9,189.9,112.7,89.0,116.1,227.8,193.6,140.0]
POL  = [8.1,10.5,11.6,13.0,5.5,10.5,12.3,9.0,10.4,5.2,4.9,11.3,5.7,5.2,8.0,9.2,8.6,5.4,6.2,5.9]
PI   = [6.0,10.76,5.41,2.77,5.07,5.65,3.22,5.97,7.59,6.02,5.98,9.74,5.74,5.48,6.30,5.68,5.60,5.89,5.66,5.96]
PA   = [1.42,0.98,0.67,1.01,0.70,1.11,1.51,0.57,1.00,1.08,1.21,1.16,1.45,1.13,0.57,0.77,0.83,1.08,0.69,1.06]
PB   = [0.83,0.93,0.89,0.54,1.19,1.10,0.37,0.75,0.87,1.60,1.30,0.74,1.05,1.38,0.55,0.75,1.19,1.37,1.47,1.70]
MW   = [89,174,132,133,121,146,147,75,155,131,131,146,149,165,115,105,119,204,181,117]
AAINDEX = np.array([KD, VOL, POL, PI, PA, PB, MW], dtype=np.float64)   # 7 x 20

# PseAAC properties (Chou 2001): hydrophobicity, hydrophilicity, side-chain mass
H1 = [0.62,-2.53,-0.78,-0.90,0.29,-0.85,-0.74,0.48,-0.40,1.38,1.06,-1.50,0.64,1.19,0.12,-0.18,-0.05,0.81,0.26,1.08]
H2 = [-0.5,3.0,0.2,3.0,-1.0,0.2,3.0,0.0,-0.5,-1.8,-1.8,3.0,-1.3,-2.5,0.0,0.3,-0.4,-3.4,-2.3,-1.5]
SM = [15,101,58,59,47,72,73,1,82,57,57,73,75,91,42,31,45,130,107,43]
def _z(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std()
PSE = np.stack([_z(H1), _z(H2), _z(SM)])                              # 3 x 20

# BLOSUM62 (order ARNDCQEGHILKMFPSTWYV)
BL = """4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4"""
BLOSUM = np.array([[float(x) for x in r.split()] for r in BL.splitlines()])   # 20x20

# CTD groupings (7 attributes x 3 groups), standard PROFEAT/iFeature
CTD_GROUPS = {
 "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
 "vdw_volume":     ["GASTPDC", "NVEQIL", "MHKFRYW"],
 "polarity":       ["LIFWCMVY", "PATGS", "HQRKNED"],
 "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
 "charge":         ["KR", "ANCQGHILMFPSTWYV", "DE"],
 "secondary_str":  ["EALMQKRH", "VIYCWFT", "GNPSD"],
 "solvent_acc":    ["ALFCGIVW", "RKQEND", "MPSTHY"],
}


# ---------------- descriptor functions: sequence -> vector --------------------
def _clean(seq):
    return [c for c in seq.upper() if c in AAi]

def d_aac(seq):
    s = _clean(seq); v = np.zeros(20)
    for c in s: v[AAi[c]] += 1
    return v / max(len(s), 1)

def d_kmer2(seq):
    s = _clean(seq); v = np.zeros(400)
    for a, b in zip(s, s[1:]): v[AAi[a] * 20 + AAi[b]] += 1
    return v / max(len(s) - 1, 1)

def d_aaindex(seq):
    s = _clean(seq)
    if not s: return np.zeros(14)
    idx = [AAi[c] for c in s]; vals = AAINDEX[:, idx]
    return np.concatenate([vals.mean(1), vals.std(1)])

def d_blosum(seq):
    s = _clean(seq)
    if not s: return np.zeros(20)
    return BLOSUM[[AAi[c] for c in s]].mean(0)

def _grp_string(seq, groups):
    m = {}
    for gi, g in enumerate(groups):
        for a in g: m[a] = gi
    return [m[c] for c in seq if c in m]

def d_ctd(seq):
    s = _clean(seq); out = []
    for groups in CTD_GROUPS.values():
        gs = _grp_string(s, groups); n = len(gs)
        comp = np.zeros(3)
        for g in gs: comp[g] += 1
        comp = comp / max(n, 1)
        trans = np.zeros(3)
        for x, y in zip(gs, gs[1:]):
            if x != y:
                trans[{(0, 1): 0, (0, 2): 1, (1, 2): 2}[tuple(sorted((x, y)))]] += 1
        trans = trans / max(n - 1, 1)
        dist = []
        for gi in range(3):
            pos = [i + 1 for i, g in enumerate(gs) if g == gi]
            if not pos:
                dist += [0, 0, 0, 0, 0]
            else:
                k = len(pos)
                for q in (0.0, 0.25, 0.5, 0.75, 1.0):
                    j = 0 if q == 0 else int(np.ceil(q * k)) - 1
                    dist.append(pos[j] / n)
        out += list(comp) + list(trans) + dist
    return np.array(out)   # 147

def d_pseaac(seq, lam=5, w=0.05):
    s = _clean(seq)
    if len(s) <= lam: s = s + s[:lam]
    idx = [AAi[c] for c in s]
    f = np.zeros(20)
    for i in idx: f[i] += 1
    f = f / len(idx)
    P = PSE[:, idx]
    tau = np.array([((P[:, :-k] - P[:, k:]) ** 2).mean(0).mean() for k in range(1, lam + 1)])
    denom = f.sum() + w * tau.sum()
    return np.concatenate([f / denom, w * tau / denom])   # 25

DESC = {"aac": d_aac, "kmer2": d_kmer2, "aaindex": d_aaindex,
        "ctd": d_ctd, "pseaac": d_pseaac, "blosum": d_blosum}

# ---------------- per-dataset config ------------------------------------------
# The pLM order here is the order they appear in the table. A file that is
# absent (or doesn't cover every receptor) is skipped with a warning, so the
# same command runs on any box -- ProtT5 shows up wherever prott5_<ds>.npz was
# built, esm2 only on M2OR until insect esm2 exists.
EMB = "data/embeddings"
DATASETS = {
    "m2or": {"task": "classification", "primary": "AUROC",
             "mol": f"{EMB}/molecules/chemberta_77m_m2or.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean.npz",
                      "prott5": f"{EMB}/proteins/prott5_m2or.npz"}},
    "cc":   {"task": "regression", "primary": "R2",
             "mol": f"{EMB}/molecules/chemberta_77m_cc.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean_cc.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean_cc.npz",
                      "prott5": f"{EMB}/proteins/prott5_cc.npz"}},
    "hc":   {"task": "regression", "primary": "R2",
             "mol": f"{EMB}/molecules/chemberta_77m_hc.npz",
             "plms": {"esm1b": f"{EMB}/proteins/esm1b_650m_mean_hc.npz",
                      "esm2":  f"{EMB}/proteins/esm2_650m_mean_hc.npz",
                      "prott5": f"{EMB}/proteins/prott5_hc.npz"}},
}


def load_pairs(dataset):
    """(pairs, task) for the dataset, with `receptor`/`inchikey`/`label` cols."""
    if dataset == "m2or":
        from orbind.regimes import full_full_pairs
        return full_full_pairs(pool_fold=1), "classification"
    from orbind.regimes_ofm import ofm_pairs
    return ofm_pairs(dataset), "regression"


def make_splits(dataset, pairs, y_all, regime, folds):
    """{fold: (train_idx, test_idx)} for the (dataset, regime) cell.

    inductive == COLD MOLECULE (the project's axis): m2or =
    `inductive_molecule_v5` seeds 42-46 (folds 1-5 -> seed 41+fold); cc/hc =
    upstream `our_inductive`. transductive = `rand` folds. cold_receptor
    (optional) holds whole receptors out: m2or = group_receptor seeds; cc =
    upstream `cdhit`; HC ships neither, so it has no cold_receptor cell. Insect
    train = upstream train+val (no tuning here, so val is just more rows).
    """
    splits = {}
    if dataset == "m2or":
        from orbind.regimes import load_split
        for f in folds:
            if regime == "transductive":
                tr, va, te = load_split("transductive", f)
                splits[f] = (tr, te)
            elif regime == "inductive":               # cold MOLECULE (project axis)
                tr, va, te = load_split("inductive_molecule_v5", 41 + f)   # folds 1-5 -> seeds 42-46
                splits[f] = (tr, te)
            else:                                     # cold_receptor (optional appendix)
                trm, tem = D.split(pairs, y_all, kind="group_receptor", seed=f)
                splits[f] = (np.where(trm)[0], np.where(tem)[0])
                print(f"  cold_receptor fold{f}: train {trm.sum()} / test {tem.sum()} rows "
                      f"({pairs['receptor'].iloc[np.where(tem)[0]].nunique()} unseen receptors)",
                      flush=True)
        return splits
    # insects
    from orbind.regimes_ofm import ofm_indices, available_families
    family = {"transductive": "rand", "inductive": "our_inductive",
              "cold_receptor": "cdhit"}[regime]
    if family not in available_families(dataset):
        raise SystemExit(f"{dataset} ships no {family!r} split -> no {regime} cell "
                         f"(has {available_families(dataset)})")
    for f in folds:
        tr, va, te = ofm_indices(dataset, family, f)
        splits[f] = (np.concatenate([tr, va]), te)
    return splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="m2or", choices=sorted(DATASETS))
    ap.add_argument("--regime", default="transductive",
                    choices=["transductive", "inductive", "cold_receptor"])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--mol", default=None, help="override molecule npz")
    ap.add_argument("--extra-prot", action="append", default=[], metavar="name=path",
                    help="add a protein npz (keyed by sequence); repeatable")
    ap.add_argument("--out", default=None,
                    help="default: results/tables/prot_floor_<dataset>_<regime>.csv")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    task = cfg["task"]; primary = cfg["primary"]
    metric_fn = D.METRICS[task]

    print(f"[{args.dataset} / {args.regime}] loading pairs + embeddings ...", flush=True)
    pairs, _ = load_pairs(args.dataset)
    recs = pd.unique(pairs["receptor"]); mols = pd.unique(pairs["inchikey"])
    rec_i = {r: i for i, r in enumerate(recs)}; mol_i = {m: i for i, m in enumerate(mols)}
    row_r = pairs["receptor"].map(rec_i).to_numpy()
    row_m = pairs["inchikey"].map(mol_i).to_numpy()
    y_all = pairs["label"].to_numpy().astype(np.float32)

    mol_path = args.mol or cfg["mol"]
    chem = D.load_npz_dict((_root / mol_path) if not pathlib.Path(mol_path).is_absolute() else mol_path)
    missing_mol = [m for m in mols if m not in chem]
    if missing_mol:
        raise SystemExit(f"{len(missing_mol)} molecules missing from {mol_path}, e.g. {missing_mol[:3]}")
    Xmol = np.stack([chem[m] for m in mols]).astype(np.float32)

    # protein matrices: classical descriptors (always) + onehot + available pLMs
    protmats = {}
    for name, fn in DESC.items():
        protmats[name] = np.stack([fn(r) for r in recs]).astype(np.float32)
    protmats["onehot"] = np.eye(len(recs), dtype=np.float32)

    plm_specs = dict(cfg["plms"])
    for kv in args.extra_prot:                          # ad-hoc additions/overrides
        name, path = kv.split("=", 1)
        plm_specs[name] = path
    plm_names = []
    for name, path in plm_specs.items():
        p = (_root / path) if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
        if not p.exists():
            print(f"  [skip pLM {name}] {path} not found", flush=True)
            continue
        emb = D.load_npz_dict(p)
        miss = [r for r in recs if r not in emb]
        if miss:
            print(f"  [skip pLM {name}] covers {len(recs)-len(miss)}/{len(recs)} receptors", flush=True)
            continue
        protmats[name] = np.stack([emb[r] for r in recs]).astype(np.float32)
        plm_names.append(name)
        print(f"  loaded pLM {name}: {protmats[name].shape}", flush=True)

    print(f"receptors={len(recs)} mols={len(mols)} chem_dim={Xmol.shape[1]} "
          f"pLMs={plm_names} task={task}", flush=True)

    # spec = (row_name, protein_matrix_key or None, use_molecule)
    specs  = [(n, n, True) for n in plm_names]           # real pLMs first
    specs += [("onehot", "onehot", True)]
    specs += [(n, n, True) for n in DESC]                # classical floor
    specs += [("onehot_only", "onehot", False), ("mol_only", None, True)]

    splits = make_splits(args.dataset, pairs, y_all, args.regime, args.folds)

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"results/tables/prot_floor_{args.dataset}_{args.regime}.csv")
    if not out.is_absolute():
        out = _root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, pm, use_mol in specs:
        pdim = protmats[pm].shape[1] if pm is not None else 0
        for f in args.folds:
            tr, te = splits[f]

            def build(idx):
                parts = []
                if use_mol: parts.append(Xmol[row_m[idx]])
                if pm is not None: parts.append(protmats[pm][row_r[idx]])
                return np.concatenate(parts, axis=1).astype(np.float32)

            model = fit_boost(build(tr), y_all[tr], seed=f, task=task)
            p = predict_scores(model, build(te), task=task)
            m = metric_fn(y_all[te], p)
            rows.append({"prot": name, "fold": f, "pdim": pdim,
                         "dim": build(tr[:1]).shape[1], **m})
            print(f"{name:12} fold{f} pdim={pdim:5} " +
                  "  ".join(f"{k}={m[k]:.3f}" for k in list(m)[:2]), flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows); g = df.groupby("prot")
    metric_cols = [c for c in df.columns if c not in ("prot", "fold", "pdim", "dim")]
    print("\n" + "=" * 84)
    print(f"{args.dataset.upper()} / {args.regime.upper()} -- protein floor "
          f"(mol=ChemBERTa), {len(args.folds)}-fold mean, sorted by {primary}  "
          f"[pdim = protein-side dim]")
    print("=" * 84)
    for name in g[primary].mean().sort_values(ascending=False).index:
        s = g.get_group(name)
        def mc(c):
            x = s[c].to_numpy(); return f"{x.mean():.3f}±{1.96 * x.std(ddof=1) / len(x) ** 0.5:.3f}"
        cells = "  ".join(f"{c} {mc(c)}" for c in metric_cols[:4])
        print(f"{name:12} pdim={int(s['pdim'].iloc[0]):5}  {cells}")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
