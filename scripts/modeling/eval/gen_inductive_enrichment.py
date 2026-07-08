"""Inductive_molecule_enrichment — can we enrich a COLD molecule's embedding?

In inductive_molecule the test molecule has NO edges, so the probe normally reads
its RAW features. Idea (user): freeze the trained GNN (encoder + decoder), let the
decoder predict which proteins the cold molecule binds, add those as pseudo-edges,
run message passing to get an enriched z_mol, iterate. Proteins evolve normally in
that forward, BUT the probe's protein column uses the frozen TRAIN z_prot (not the
test-forward proteins).

Validity ladder (each bar = a full XGBoost probe; train on ALL warm train edges,
test on cold test edges; protein column = frozen train z_prot throughout):

  raw      : mol = raw features                       (current inductive baseline)
  noedge   : mol = z_mol encoder output, no edges     (encoding vs raw control)
  imputed  : mol = enriched via decoder top-k edges, T iterations   (the proposal)
  random   : mol = enriched via k RANDOM edges        (negative control)
  oracle   : mol = enriched via TRUE test edges, leave-one-out      (upper bound)

Warm train molecules always use their real graph z_mol so train/test share
one embedding space; only COLD test molecules get the variant treatment.

Two model configs are available (see CONFIGS):
  compressed_h64 : the skinny run (mol pca16, prot pca32, hidden 64) — reuses existing ckpts.
  full_h512      : a FAT net (full ChemBERTa 384 + full ESM-1b 1280, hidden 512) — trains
                   its own checkpoints on demand via ensure_checkpoints().
"""
import pathlib, subprocess, sys
import numpy as np, pandas as pd, torch

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import lorax as L, hetero as H
from orbind.baselines import train_boost
from orbind.dataset import metrics
MOL, PROT = H.MOL, H.PROT

METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
SEEDS = [42, 43, 44]
K = 10          # pseudo-edges per cold molecule
T = 3           # imputation iterations
BARS = ["raw", "noedge", "imputed", "random", "oracle"]
TABLES = _root / "results/graph/full_full/enrichment/reports/tables"
CKPT_NAME = "gnn_signed_unentangled_boost_inductive_molecule_fold1.pt"


# ----------------------------------------------------------------------------- configs
def _compressed_ckpt(seed):
    return (_root / "results/graph/full_full/compression/h64/runs/inductive_shared_raw" /
            f"seed_{seed}/checkpoints" / CKPT_NAME)

def _fat_run_dir(seed):
    return f"results/graph/full_full/enrichment/full_h512_shared/runs/seed_{seed}"

# Full ESM/ChemBERTa are highly anisotropic (~99% of the ESM vector norm is the shared
# mean). Unlike PCA (which centres), raw embeddings fed to the GNN keep that mean, which
# swamps the discriminative signal in the Linear projection + SAGE mean-aggregation. So the
# FAT config trains/probes on STANDARDISED copies (per-dim mean-subtract + /sd), matching
# what the MLP baseline does. This is scoped to the enrichment experiment only.
_RAW_PROT_FULL = "data/external/lorax_m2or/esm1b_650m_mean_lorax.npz"
_RAW_MOL_FULL = "data/embeddings/molecules/chemberta_77m_lorax.pkl"
_STD_PROT = "data/embeddings/proteins/esm1b_650m_mean_lorax_std.pkl"
_STD_MOL = "data/embeddings/molecules/chemberta_77m_lorax_std.pkl"


def ensure_standardized_embeddings():
    """Create z-scored copies of the full embeddings (idempotent). Baking mu/sd into the
    files makes training and the enrichment ladder use identical standardisation."""
    import pickle
    for std_rel, raw_rel, name in [(_STD_PROT, _RAW_PROT_FULL, "ESM"),
                                   (_STD_MOL, _RAW_MOL_FULL, "ChemBERTa")]:
        std_p = _root / std_rel
        if std_p.exists():
            continue
        d = L._load_embedding_dict(_root / raw_rel)          # {key: vector}
        keys = list(d)
        X = np.stack([np.asarray(d[k], dtype=np.float64) for k in keys])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xs = ((X - mu) / sd).astype(np.float32)
        std_p.parent.mkdir(parents=True, exist_ok=True)
        with open(std_p, "wb") as f:
            pickle.dump({k: Xs[i] for i, k in enumerate(keys)}, f)
        print(f"[std] wrote {std_rel}  ({name} centred+scaled; "
              f"mean-norm {np.linalg.norm(mu):.1f} removed)", flush=True)


CONFIGS = {
    "compressed_h64": dict(
        prot_emb=_RAW_PROT_FULL,
        mol_emb=_RAW_MOL_FULL,
        hidden=64, epochs=600, trainable=False, pca_from_checkpoint=True,
        ckpt=_compressed_ckpt,
        out=TABLES / "inductive_enrichment_shared_by_seed.csv"),
    "full_h512": dict(
        prot_emb=_STD_PROT,
        mol_emb=_STD_MOL,
        hidden=512, epochs=600, trainable=True, pca_from_checkpoint=False,
        ckpt=lambda seed: _root / _fat_run_dir(seed) / "checkpoints" / CKPT_NAME,
        run_dir=_fat_run_dir,
        out=TABLES / "inductive_enrichment_full_h512_shared_by_seed.csv"),
}

def ensure_checkpoints(cfg, seeds=SEEDS):
    """Train any missing checkpoints for a trainable config (inherits stdout so the
    notebook shows epoch progress). Each seed gets its own run dir (no clobber)."""
    if not cfg.get("trainable"):
        return
    ensure_standardized_embeddings()          # standardise the full embeddings first
    for seed in seeds:
        if cfg["ckpt"](seed).exists():
            print(f"[fat] seed {seed}: checkpoint cached", flush=True); continue
        cmd = [sys.executable, str(_root / "scripts/modeling/train/train_graph_full_full.py"),
               "--arch", "gnn", "--mp_mode", "signed", "--regime", "inductive_molecule",
               "--fold", "1", "--seed", str(seed), "--epochs", str(cfg["epochs"]),
               "--hidden", str(cfg["hidden"]), "--lr", "0.001", "--grad-clip", "1.0",
               "--lr-scheduler", "--scheduler-patience", "100", "--scheduler-factor", "0.5",
               "--diagnostics",
               "--protein-embeddings", cfg["prot_emb"], "--molecule-embeddings", cfg["mol_emb"],
               "--results-dir", cfg["run_dir"](seed)]
        print(f"[fat] seed {seed}: training SHARED (hidden {cfg['hidden']}, full embeddings, "
              f"{cfg['epochs']} epochs)...", flush=True)
        subprocess.run(cmd, cwd=_root, check=True)


# ----------------------------------------------------------------------------- ladder
def _apply_saved_pca(x, meta):
    X = x.numpy().astype(np.float64)
    return torch.tensor(((X - np.asarray(meta["mean"], dtype=np.float64)) @
                         np.asarray(meta["components"], dtype=np.float64).T).astype(np.float32))


def load_run(seed, cfg):
    esm, chem = L.load_embeddings(cfg["prot_emb"], cfg["mol_emb"])
    ck = torch.load(cfg["ckpt"](seed), map_location="cpu", weights_only=False)
    torch.manual_seed(seed)
    Xm, Xp, splits = L.build("inductive_molecule", 1, esm, chem, seed=seed)
    if cfg.get("pca_from_checkpoint"):
        Xm = _apply_saved_pca(Xm, ck["input_pca"]["molecule"])
        Xp = _apply_saved_pca(Xp, ck["input_pca"]["protein"])
    train_pairs = splits["train"]
    train_pos = torch.tensor(train_pairs["pos"].T, dtype=torch.long)
    train_neg = torch.tensor(train_pairs["neg"].T, dtype=torch.long)
    eidx_train = H.edge_index_dict(train_pos, train_neg, mode="signed")
    x_dict = {MOL: Xm, PROT: Xp}
    model = H.HeteroLink(hidden=cfg["hidden"], dropout=0.3, mp_mode="signed")
    with torch.no_grad():
        model.encode(x_dict, eidx_train)
    model.load_state_dict(ck["best_state"]); model.eval()
    with torch.no_grad():
        z_train = model.encode(x_dict, eidx_train)
    return dict(model=model, x_dict=x_dict, Xm=Xm.numpy(),
                z_mol=z_train[MOL], z_prot=z_train[PROT],
                train_pos=train_pos, train_neg=train_neg,
                probe_train=H.sup_edges(train_pairs),
                val=H.sup_edges(splits["val"]), val_pos=splits["val"]["pos"],
                test=H.sup_edges(splits["test"]), test_pos=splits["test"]["pos"])


def _encode_with_extra(R, extra_pos):
    aug_pos = torch.cat([R["train_pos"], extra_pos], dim=1) if extra_pos.numel() else R["train_pos"]
    eidx = H.edge_index_dict(aug_pos, R["train_neg"], mode="signed")
    with torch.no_grad():
        z = R["model"].encode(R["x_dict"], eidx)
    return z[MOL]


def enrich_imputed(R, m, k=K, T=T):
    zmol = R["z_mol"][m:m + 1].clone()
    n_prot = R["z_prot"].shape[0]
    for _ in range(T):
        with torch.no_grad():
            logits = R["model"].dec(
                torch.cat([zmol.expand(n_prot, -1), R["z_prot"]], dim=-1)).squeeze(-1)
        topk = torch.topk(logits, k).indices
        extra = torch.stack([torch.full((k,), m, dtype=torch.long), topk])
        zmol = _encode_with_extra(R, extra)[m:m + 1]
    return zmol.squeeze(0)


def enrich_random(R, m, rng, k=K):
    prots = torch.tensor(rng.choice(R["z_prot"].shape[0], size=k, replace=False), dtype=torch.long)
    extra = torch.stack([torch.full((k,), m, dtype=torch.long), prots])
    return _encode_with_extra(R, extra)[m]


def enrich_oracle_map(R, m, pos_prots):
    def zmol_for(ctx):
        if len(ctx) == 0:
            return R["z_mol"][m]
        extra = torch.stack([torch.full((len(ctx),), m, dtype=torch.long),
                             torch.tensor(ctx, dtype=torch.long)])
        return _encode_with_extra(R, extra)[m]
    full = zmol_for(pos_prots)
    loo = {p: zmol_for([q for q in pos_prots if q != p]) for p in pos_prots}
    return full, loo


def cold_mol_vecs(R, bar, seed):
    cold = sorted(set(int(m) for m in R["test_pos"][:, 0]).union(
                  int(i) for i in R["test"][0][0].numpy()))
    out = {}
    rng = np.random.default_rng(seed + 7)
    for m in cold:
        if bar == "noedge":
            out[m] = R["z_mol"][m]
        elif bar == "imputed":
            out[m] = enrich_imputed(R, m)
        elif bar == "random":
            out[m] = enrich_random(R, m, rng)
    return out


def build_features(R, bar, seed):
    ip, ytr = R["probe_train"]; it, yte = R["test"]
    zp = R["z_prot"].numpy()
    if bar == "raw":
        mtr = R["Xm"][ip[0].numpy()]
        mte = R["Xm"][it[0].numpy()]
    else:
        mtr = R["z_mol"].numpy()[ip[0].numpy()]
        if bar == "oracle":
            pos_by_mol = {}
            for mm, pp in R["test_pos"]:
                pos_by_mol.setdefault(int(mm), []).append(int(pp))
            cache = {m: enrich_oracle_map(R, m, ps) for m, ps in pos_by_mol.items()}
            rows = []
            for mm, pp in zip(it[0].numpy(), it[1].numpy()):
                mm, pp = int(mm), int(pp)
                if mm in cache:
                    full, loo = cache[mm]
                    rows.append((loo[pp] if pp in loo else full).numpy())
                else:
                    rows.append(R["z_mol"].numpy()[mm])
            mte = np.stack(rows)
        else:
            vecs = cold_mol_vecs(R, bar, seed)
            mte = np.stack([vecs[int(m)].numpy() for m in it[0].numpy()])
    Xtr = np.concatenate([mtr, zp[ip[1].numpy()]], axis=1).astype(np.float32)
    Xte = np.concatenate([mte, zp[it[1].numpy()]], axis=1).astype(np.float32)
    return Xtr, ytr.numpy(), Xte, yte.numpy()


def run_ladder(cfg_name, seeds=SEEDS, train_first=True):
    """Ensure checkpoints (train the fat net if needed), run the ladder, cache CSV, return df."""
    cfg = CONFIGS[cfg_name] if isinstance(cfg_name, str) else cfg_name
    if train_first:
        ensure_checkpoints(cfg, seeds)
    rows = []
    for seed in seeds:
        if not cfg["ckpt"](seed).exists():
            print(f"SKIP seed {seed}: no checkpoint", flush=True); continue
        R = load_run(seed, cfg)
        for bar in BARS:
            Xtr, ytr, Xte, yte = build_features(R, bar, seed)
            sc = train_boost(Xtr, ytr, Xte, seed=seed)
            m = metrics(yte, sc)
            rows.append({"seed": seed, "bar": bar, "dim": Xte.shape[1], "k": K, "T": T,
                         **{k: float(m[k]) for k in METRICS}})
            print(f"seed {seed} {bar:8s} dim={Xte.shape[1]:5d} "
                  + " ".join(f"{k}={m[k]:.3f}" for k in METRICS), flush=True)
    df = pd.DataFrame(rows)
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out"], index=False)
    print(f"\nsaved {cfg['out'].relative_to(_root)} ({len(df)} rows)", flush=True)
    return df


def _as_eval_split(R, name):
    """View validation or test as the cold evaluation split used by build_features."""
    out = dict(R)
    out["test"] = R[name]
    out["test_pos"] = R[f"{name}_pos"]
    return out


def _best_threshold(y, scores, objective):
    from sklearn.metrics import f1_score, matthews_corrcoef
    fn = f1_score if objective == "F1" else matthews_corrcoef
    grid = np.linspace(0.01, 0.99, 99)
    values = [fn(y, (scores >= threshold).astype(int)) for threshold in grid]
    return float(grid[int(np.argmax(values))])


def run_val_thresholds(cfg_name="compressed_h64", seeds=SEEDS):
    """Recompute slim probes and select classification thresholds on cold validation.

    AUROC/AUPRC remain threshold-free. F1 uses the validation-F1 optimum; MCC uses
    the validation-MCC optimum. Test labels never participate in threshold selection.
    """
    from sklearn.metrics import f1_score, matthews_corrcoef
    cfg = CONFIGS[cfg_name] if isinstance(cfg_name, str) else cfg_name
    out_path = TABLES / "inductive_enrichment_shared_val_threshold_by_seed.csv"
    rows = []
    for seed in seeds:
        if not cfg["ckpt"](seed).exists():
            print(f"SKIP seed {seed}: no checkpoint", flush=True); continue
        R = load_run(seed, cfg)
        Rval = _as_eval_split(R, "val")
        for bar in BARS:
            Xtr, ytr, Xval, yval = build_features(Rval, bar, seed)
            _, _, Xtest, ytest = build_features(R, bar, seed)
            # train_boost is deterministic for fixed (Xtr, ytr, seed), so these two
            # calls represent one model evaluated on two matrices.
            score_val = train_boost(Xtr, ytr, Xval, seed=seed)
            score_test = train_boost(Xtr, ytr, Xtest, seed=seed)
            t_f1 = _best_threshold(yval, score_val, "F1")
            t_mcc = _best_threshold(yval, score_val, "MCC")
            base = metrics(ytest, score_test)
            pred_f1 = (score_test >= t_f1).astype(int)
            pred_mcc = (score_test >= t_mcc).astype(int)
            rows.append({"seed":seed, "bar":bar,
                         "threshold_val_F1":t_f1, "threshold_val_MCC":t_mcc,
                         "AUROC":float(base["AUROC"]), "AUPRC":float(base["AUPRC"]),
                         "F1_at_0.5":float(base["F1"]), "MCC_at_0.5":float(base["MCC"]),
                         "F1_at_val_F1":float(f1_score(ytest, pred_f1, zero_division=0)),
                         "MCC_at_val_MCC":float(matthews_corrcoef(ytest, pred_mcc))})
            print(f"seed {seed} {bar:8s} tF1={t_f1:.2f} tMCC={t_mcc:.2f} "
                  f"test F1={rows[-1]['F1_at_val_F1']:.3f} "
                  f"MCC={rows[-1]['MCC_at_val_MCC']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved {out_path.relative_to(_root)} ({len(out)} rows)", flush=True)
    return out


# backward-compat shim for notebook cells
OUT = CONFIGS["compressed_h64"]["out"]
def main(config="compressed_h64"):
    return run_ladder(config, train_first=(config == "full_h512"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="compressed_h64")
    args = ap.parse_args()
    main(args.config)
