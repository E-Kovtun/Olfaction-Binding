"""Train the bipartite link predictor (GNN or GAT) on the FULL_FULL dataset.

full_full = the LORAX / Hladis M2OR release (data/splits_indexes/lorax_m2or), the most
complete variant we have. Unlike curated/full we do NOT make our own random
splits: we respect LORAX's pre-defined folds, where the test set is EC50-only
(highest-quality, dose-response, ~22% positive) while train/val are the full
noisy mix (primary + secondary + ec50, ~5.7% positive). See the notebook
graph_evaluation_full_full.ipynb for the protocol write-up.

Node features (the paper shows the molecule encoder is interchangeable):
  * proteins  Р Р†Р вЂљРІР‚Сњ putative ESM-1b 650M mean-pooled (identity not yet verified), keyed by
                the raw amino-acid sequence; distinct from curated/full ESM2.
  * molecules Р Р†Р вЂљРІР‚Сњ ChemBERTa-77M (384-d), keyed by SMILES. GIN only covers 64% of
                LORAX molecules, so we use the LORAX-provided ChemBERTa here.

Message passing uses TRAIN edges only; supervision uses the LORAX train/val/test
splits directly. Only the unentangled XGBoost probe is reported
([raw_mol_emb || graph_prot_emb]); the MLP probe is dropped.
For transductive runs, --transductive-exp instead reports
([graph_mol_emb || graph_prot_emb]). Inductive runs intentionally keep raw molecules.
Optional --protein-pca-dim/--molecule-pca-dim compression is fit on every node in the
original train split BEFORE any disjoint division; validation/test-only nodes never fit PCA.
With --disjoint-probe-train, GNN MP/decoder training and downstream XGBoost fitting
use disjoint class-stratified halves of the original train labels.

  uv run python legacy/scripts/modeling/train/train_graph_full_full.py --arch gnn --mp_mode signed
  uv run python legacy/scripts/modeling/train/train_graph_full_full.py --arch gat --mp_mode all_edges
  uv run python legacy/scripts/modeling/train/train_graph_full_full.py --regime transductive \
      --arch gnn --mp_mode signed --transductive-exp --disjoint-probe-train

Writes checkpoints to <results-dir>/checkpoints/ in the same shape the
graph_evaluation notebooks expect (regime == "ec50").
"""
import argparse, pathlib, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind.legacy import hetero as H
from orbind.legacy.hetero import HeteroDGI
from orbind.legacy import lorax as L
from orbind.legacy.hetero_gat import HeteroGATLink
from orbind.baselines import train_boost
from orbind.dataset import metrics

METRICS = ["AUROC", "AUPRC", "MCC", "F1", "precision", "recall"]


def variant_name(args):
    q = args.mol_quality_q
    hist_tag = ("_history" if args.history_depth >= 2 else "_hist1") if args.history else ""
    exp_tag = "_transductive_exp" if args.transductive_exp else ""
    disjoint_tag = "_disjoint" if args.disjoint_probe_train else ""
    # DGI tag only when the auxiliary loss is active, so weight=0 reproduces the
    # exact v5 variant name (a clean in-run control): e.g. "_dgishared50".
    dgi_tag = (f"_dgi{args.dgi_scope}{int(round(args.dgi_weight * 100))}"
               if args.dgi_weight > 0 else "")
    # SimGCL contrastive add-on tag (eps and weight), only when active.
    cl_tag = (f"_simgcl_e{int(round(args.cl_eps * 100))}_w{int(round(args.cl_weight * 100))}"
              if args.cl_weight > 0 else "")
    # Main-loss tag (BPR replaces the default pointwise BCE); empty for bce.
    main_tag = "_bpr" if args.main_loss == "bpr" else ""
    return (args.mp_mode + (f"_q{int(q * 100)}" if q > 0 else "") + main_tag + hist_tag
            + ("_rawp" if args.concat_raw_prot else "") + exp_tag + disjoint_tag + dgi_tag + cl_tag)


def save_history(history, csv_path, plot_path):
    """Atomically persist exact per-epoch metrics and a compact diagnostic plot."""
    table = pd.DataFrame(history)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(".tmp.csv")
    table.to_csv(tmp_csv, index=False)
    tmp_csv.replace(csv_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_test = "test_AUROC" in table.columns

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = [
        (axes[0, 0], ["AUROC", "AUPRC"], "Ranking metrics"),
        (axes[0, 1], ["MCC", "F1"], "Threshold metrics @ 0.5"),
        (axes[1, 0], ["precision", "recall"], "Precision / recall @ 0.5"),
    ]
    for ax, names, title in panels:
        for metric in names:
            ax.plot(table.epoch, table[f"val_{metric}"], label=f"val {metric}")
            if has_test:
                ax.plot(table.epoch, table[f"test_{metric}"], "--", alpha=.75,
                        label=f"test {metric}")
        ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    axes[1, 1].plot(table.epoch, table.train_loss, color="black")
    axes[1, 1].set_title("Training loss"); axes[1, 1].grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    suptitle = ("Validation vs test history (--observe_test)" if has_test
                else "Validation history (test hidden Р Р†Р вЂљРІР‚Сњ no --observe_test)")
    fig.suptitle(suptitle)
    fig.tight_layout()
    tmp_plot = plot_path.with_suffix(".tmp.png")
    fig.savefig(tmp_plot, dpi=140, bbox_inches="tight")
    plt.close(fig)
    tmp_plot.replace(plot_path)


def make_model(arch, mp_mode, args):
    if arch == "gnn":
        return H.HeteroLink(hidden=args.hidden, dropout=args.dropout, mp_mode=mp_mode)
    return HeteroGATLink(hidden=args.gat_hidden, heads=args.heads,
                         dropout=args.dropout, mp_mode=mp_mode)


def fit_pca_on_full_train(Xm, Xp, train_split, molecule_dim=0, protein_dim=0):
    """Fit domain PCAs on every node present in the original train split.

    This runs before an optional disjoint GNN/probe split, so shared and disjoint
    protocols use the same full-train PCA basis for a given regime/fold/seed.
    Test/validation-only nodes never participate in fitting.
    """
    train_pairs = np.concatenate([train_split["pos"], train_split["neg"]], axis=0)

    def project(x, fit_idx, dim, domain):
        if not dim:
            return x, None
        fit_idx = np.unique(fit_idx).astype(np.int64)
        X = x.numpy().astype(np.float64)
        Xfit = X[fit_idx]
        max_rank = min(Xfit.shape[0] - 1, Xfit.shape[1])
        if dim > max_rank:
            raise ValueError(f"{domain} PCA dim {dim} exceeds train rank {max_rank}")
        mean = Xfit.mean(axis=0, keepdims=True)
        _, explained_s, vt = np.linalg.svd(Xfit - mean, full_matrices=False)
        components = vt[:dim]
        transformed = ((X - mean) @ components.T).astype(np.float32)
        total_var = np.square(explained_s).sum()
        kept = float(np.square(explained_s[:dim]).sum() / total_var) if total_var else 0.0
        meta = {"domain": domain, "dim": int(dim), "fit_indices": fit_idx,
                "fit_n": int(len(fit_idx)), "mean": mean[0].astype(np.float32),
                "components": components.astype(np.float32),
                "explained_variance_ratio_sum": kept,
                "fit_scope": "all nodes in original train before disjoint split"}
        print(f"  {domain} PCA: {X.shape[1]} -> {dim}, fit on all {len(fit_idx)} train nodes, "
              f"variance={kept:.4f}")
        return torch.tensor(transformed, dtype=torch.float), meta

    Xm, mol_meta = project(Xm, train_pairs[:, 0], molecule_dim, "molecule")
    Xp, prot_meta = project(Xp, train_pairs[:, 1], protein_dim, "protein")
    return Xm, Xp, {"molecule": mol_meta, "protein": prot_meta}


def run(args):
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable in this PyTorch build")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"compute device: {device}" +
          (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
    esm, chem = L.load_embeddings(args.protein_embeddings, args.molecule_embeddings)
    print(f"embeddings: ESM proteins={len(esm)} | ChemBERTa mols={len(chem)}")
    print(f"\n[fold {args.fold} | {args.regime} | {args.arch} | {args.mp_mode}]")
    Xm, Xp, splits = L.build(args.regime, args.fold, esm, chem, seed=args.seed)
    Xm, Xp, input_pca = fit_pca_on_full_train(
        Xm, Xp, splits["train"], molecule_dim=args.molecule_pca_dim,
        protein_dim=args.protein_pca_dim)
    x_dict = {H.MOL: Xm, H.PROT: Xp}

    if args.disjoint_probe_train:
        gnn_train, probe_train = H.split_train_for_probe(
            splits["train"], probe_frac=args.probe_train_frac, seed=args.seed + 101)
    else:
        gnn_train = probe_train = splits["train"]

    # MP graph from TRAIN edges only; optional molecule-coverage quality filter
    mp_pos_a, mp_neg_a = gnn_train["pos"], gnn_train["neg"]
    if args.mol_quality_q > 0.0:
        # Compute coverage on the complete original train, preserving q## semantics;
        # apply the resulting mask only to the GNN half's MP edges.
        mask = H.quality_mol_mask(
            splits["train"]["pos"], splits["train"]["neg"],
            Xm.shape[0], args.mol_quality_q)
        keep = set(int(i) for i in np.where(mask)[0])
        _f = lambda a: a[np.array([int(m) in keep for m in a[:, 0]], dtype=bool)] if len(a) else a
        mp_pos_a, mp_neg_a = _f(mp_pos_a), _f(mp_neg_a)
        print(f"  MP edges after quality filter: {len(mp_pos_a)} pos, {len(mp_neg_a)} neg")
    mp_pos = torch.tensor(mp_pos_a.T, dtype=torch.long)
    mp_neg = torch.tensor(mp_neg_a.T, dtype=torch.long)
    eidx = H.edge_index_dict(mp_pos, mp_neg, mode=args.mp_mode)
    sup = {"train": H.sup_edges(gnn_train),
           "probe_train": H.sup_edges(probe_train),
           "val": H.sup_edges(splits["val"]),
           "test": H.sup_edges(splits["test"])}

    # Keep preprocessing/PCA on CPU, then place the complete full-batch graph on
    # the preferred accelerator once. Only compact NumPy arrays return to CPU for metrics/XGBoost.
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    eidx = {k: v.to(device) for k, v in eidx.items()}
    sup = {k: (idx.to(device), y.to(device)) for k, (idx, y) in sup.items()}
    print(f"  GNN train pos/neg={len(gnn_train['pos'])}/{len(gnn_train['neg'])} | "
          f"probe train={len(probe_train['pos'])}/{len(probe_train['neg'])}")

    model = make_model(args.arch, args.mp_mode, args).to(device)
    with torch.no_grad():
        z_init = model.encode(x_dict, eidx)     # init lazy params

    # ---- optional DeepGraphInfomax auxiliary head ----
    # Both node types already share the encoder's `hidden`-dim output, so DGI
    # pools them (scope="shared") or takes proteins alone (scope="prot"). The
    # discriminator is sized from the actual encoder output dim, and its params
    # join the optimizer. weight=0 => head not built, training == v5 exactly.
    dgi_head = None
    if args.dgi_weight > 0:
        dgi_dim = z_init[H.MOL].shape[-1]
        dgi_head = HeteroDGI(dgi_dim).to(device)
        print(f"  DGI enabled: weight={args.dgi_weight} scope={args.dgi_scope} dim={dgi_dim}")

    def dgi_pool(z):
        """Pool encoder output into the DGI node set, L2-normalized (the signed
        encoder's final layer has no activation, so raw z would saturate the
        sigmoid readout)."""
        if args.dgi_scope == "prot":
            return F.normalize(z[H.PROT], dim=-1)
        return torch.cat([F.normalize(z[H.MOL], dim=-1),
                          F.normalize(z[H.PROT], dim=-1)], dim=0)

    params = list(model.parameters()) + (list(dgi_head.parameters()) if dgi_head else [])
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
        if args.lr_scheduler else None)
    tr_idx, tr_y = sup["train"]
    neg_count = (tr_y == 0).sum().float()
    pos_count = (tr_y == 1).sum().float().clamp_min(1.0)
    pw = (neg_count / pos_count).reshape(1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    # ---- BPR triple bookkeeping (only when --main-loss bpr) ----
    # For each positive (m, p+), the negative p- is drawn each epoch from the
    # REAL tested negatives of the SAME molecule m. Positives whose molecule has
    # no tested negative are dropped (we never invent unobserved negatives).
    # flat_negs/offsets/counts give O(1) vectorized per-epoch sampling.
    bpr = None
    if args.main_loss == "bpr":
        from collections import defaultdict
        neg_by_mol = defaultdict(list)
        for m, p in gnn_train["neg"]:
            neg_by_mol[int(m)].append(int(p))
        kept_mol, kept_pos_prot, flat, offsets, counts = [], [], [], [], []
        off = 0
        for m, p in gnn_train["pos"]:
            negs = neg_by_mol.get(int(m))
            if not negs:
                continue
            kept_mol.append(int(m)); kept_pos_prot.append(int(p))
            flat.extend(negs); offsets.append(off); counts.append(len(negs)); off += len(negs)
        if not kept_mol:
            raise RuntimeError("--main-loss bpr: no positive has a tested negative for its "
                               "molecule; cannot form a single BPR triple")
        bpr = {
            "mol": torch.tensor(kept_mol, dtype=torch.long, device=device),
            "pos_prot": torch.tensor(kept_pos_prot, dtype=torch.long, device=device),
            "flat_negs": np.asarray(flat, dtype=np.int64),
            "offsets": np.asarray(offsets, dtype=np.int64),
            "counts": np.asarray(counts, dtype=np.int64),
        }
        print(f"  BPR: {len(kept_mol)}/{len(gnn_train['pos'])} positives kept "
              f"(molecule has >=1 tested negative); rest dropped from the ranking loss")

    if args.observe_test:
        print(
            "\n" + "!" * 70 + "\n"
            "!!  WARNING: --observe_test ENABLED                                 !!\n"
            "!!  Test-set metrics are computed and logged EVERY epoch.           !!\n"
            "!!  This is ONLY valid for gnn_training_diagnostics.ipynb.          !!\n"
            "!!  DO NOT use this flag for any hyperparameter sweep or tuning.    !!\n"
            "!!  DO NOT report results from a run that used this flag.           !!\n"
            + "!" * 70 + "\n"
        )

    variant = variant_name(args)
    history_dir = _root / args.results_dir / "history"
    seed_tag = f"gnn{args.seed}_boost{args.boost_seed}"
    run_id = f"{args.arch}_{variant}_{args.regime}_fold{args.fold}_{seed_tag}"
    history_csv = history_dir / f"{run_id}.csv"
    history_plot = history_dir / f"{run_id}.png"
    history = []
    best_val_auprc = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        z_train = model.encode(x_dict, eidx)
        if args.main_loss == "bpr":
            # sample one real tested negative per kept positive this epoch
            r = (np.random.rand(len(bpr["offsets"])) * bpr["counts"]).astype(np.int64)
            neg_prot = torch.as_tensor(bpr["flat_negs"][bpr["offsets"] + r],
                                       dtype=torch.long, device=device)
            pos_scores = model.decode(z_train, torch.stack([bpr["mol"], bpr["pos_prot"]]))
            neg_scores = model.decode(z_train, torch.stack([bpr["mol"], neg_prot]))
            link_loss = H.bpr_loss(pos_scores, neg_scores)
            train_logits = pos_scores  # for the diagnostics block below
        else:
            train_logits = model.decode(z_train, tr_idx)
            link_loss = loss_fn(train_logits, tr_y)
        # DGI: corrupt = row-shuffle input features per node type, same edges;
        # re-encode and contrast against the real graph's summary.
        if dgi_head is not None:
            x_corrupt = {k: v[torch.randperm(v.shape[0], device=v.device)]
                         for k, v in x_dict.items()}
            z_corrupt = model.encode(x_corrupt, eidx)
            dgi_loss = dgi_head.loss(dgi_pool(z_train), dgi_pool(z_corrupt))
        else:
            dgi_loss = torch.zeros((), device=device)
        # SimGCL: two noise-perturbed views, per-node-type InfoNCE, summed.
        if args.cl_weight > 0:
            z1 = model.encode(x_dict, eidx, noise_eps=args.cl_eps)
            z2 = model.encode(x_dict, eidx, noise_eps=args.cl_eps)
            cl_loss = (H.info_nce(z1[H.MOL], z2[H.MOL], args.cl_temp)
                       + H.info_nce(z1[H.PROT], z2[H.PROT], args.cl_temp))
        else:
            cl_loss = torch.zeros((), device=device)
        loss = link_loss + args.dgi_weight * dgi_loss + args.cl_weight * cl_loss
        loss.backward()
        if args.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        else:
            grad_norm = float(torch.sqrt(sum(
                p.grad.detach().pow(2).sum()
                for p in params if p.grad is not None)))
        opt.step()

        model.eval()
        with torch.no_grad():
            z_eval = model.encode(x_dict, eidx)
            val_logits = model.decode(z_eval, sup["val"][0])
            val_pred = torch.sigmoid(val_logits).detach().cpu().numpy()
            if args.observe_test:
                test_pred = torch.sigmoid(model.decode(z_eval, sup["test"][0])).detach().cpu().numpy()

        val_m = metrics(sup["val"][1].detach().cpu().numpy(), val_pred)
        row = {"epoch": ep, "train_loss": float(loss.detach()),
               "link_loss": float(link_loss.detach()),
               "dgi_loss": float(dgi_loss.detach()),
               "cl_loss": float(cl_loss.detach()),
               "lr": float(opt.param_groups[0]["lr"]),
               "grad_norm_pre_clip": grad_norm}
        row.update({f"val_{k}": float(v) for k, v in val_m.items()})
        if args.diagnostics:
            param_norm = torch.sqrt(sum(
                p.detach().pow(2).sum() for p in model.parameters()))
            row.update({
                "param_norm": float(param_norm),
                "train_logit_mean": float(train_logits.detach().mean()),
                "train_logit_std": float(train_logits.detach().std()),
                "train_logit_absmax": float(train_logits.detach().abs().max()),
                "val_logit_mean": float(val_logits.mean()),
                "val_logit_std": float(val_logits.std()),
                "val_logit_absmax": float(val_logits.abs().max()),
            })
            if args.arch == "gnn":
                with torch.no_grad():
                    stages = model.encode_history(x_dict, eidx, depth=2)
                for node_type, values in stages.items():
                    x0, x1, x2 = values.chunk(3, dim=-1)
                    for stage_name, stage in (("x0", x0), ("x1", x1), ("x2", x2)):
                        prefix = f"{node_type}_{stage_name}"
                        row[f"{prefix}_zero_frac"] = float((stage == 0).float().mean())
                        row[f"{prefix}_norm_mean"] = float(stage.norm(dim=-1).mean())
                        row[f"{prefix}_absmax"] = float(stage.abs().max())
        if args.observe_test:
            test_m = metrics(sup["test"][1].detach().cpu().numpy(), test_pred)
            row.update({f"test_{k}": float(v) for k, v in test_m.items()})
        history.append(row)

        if val_m["AUPRC"] > best_val_auprc:
            best_val_auprc = val_m["AUPRC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            row["is_best"] = True
        else:
            row["is_best"] = False

        if scheduler is not None:
            scheduler.step(val_m["AUPRC"])

        # CSV is updated every epoch so an interrupted 900-epoch run remains useful.
        save_table = pd.DataFrame(history)
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = history_csv.with_suffix(".tmp.csv")
        save_table.to_csv(tmp_csv, index=False)
        # os.replace can transiently hit WinError 5 when an AV/indexer briefly
        # locks the freshly written file; retry, then fall back to a direct write.
        for _attempt in range(10):
            try:
                tmp_csv.replace(history_csv)
                break
            except PermissionError:
                time.sleep(0.3)
        else:
            save_table.to_csv(history_csv, index=False)
        if ep % args.plot_every == 0 or ep == args.epochs:
            save_history(history, history_csv, history_plot)
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            best_marker = " *" if row["is_best"] else ""
            log = (f"    epoch {ep:3d} | loss={row['train_loss']:.4f} | "
                   f"val AUROC={val_m['AUROC']:.3f} AUPRC={val_m['AUPRC']:.3f}{best_marker}")
            if args.observe_test:
                log += f" | test AUROC={test_m['AUROC']:.3f} AUPRC={test_m['AUPRC']:.3f}"
            if args.diagnostics:
                log += f" | lr={row['lr']:.2e} grad={row['grad_norm_pre_clip']:.2e}"
                # Per-stage zero fractions exist only for GNN (encode_history).
                if "molecule_x1_zero_frac" in row:
                    log += (f" mol_x1_zero={row['molecule_x1_zero_frac']:.3f}"
                            f" prot_x1_zero={row['protein_x1_zero_frac']:.3f}")
                log += f" val_logit_std={row['val_logit_std']:.2e}"
            print(log, flush=True)

    save_history(history, history_csv, history_plot)
    hist_df = pd.DataFrame(history)
    best_i = int(hist_df["val_AUPRC"].idxmax())
    best = hist_df.loc[best_i]
    best_line = f"  best val AUPRC epoch={int(best.epoch)}: val={best.val_AUPRC:.3f}"
    if args.observe_test:
        best_line += f", test={best.test_AUPRC:.3f}"
    print(best_line)
    if args.observe_test:
        for metric in ("AUROC", "AUPRC", "MCC", "F1"):
            corr = hist_df[f"val_{metric}"].corr(hist_df[f"test_{metric}"])
            print(f"  epoch-wise val/test correlation {metric}: r={corr:.3f}")
    print(f"  history -> {history_csv.relative_to(_root)}")
    print(f"  plot    -> {history_plot.relative_to(_root)}")

    # Snapshot the LAST-epoch weights before restoring best-val, so the final
    # encoder can be probed too (older runs only ever kept best_state).
    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- unentangled BOOST probe: [raw chemberta mol || graph-enriched esm prot] ----
    # Probe a given encoder state; returns (primary, raw, enriched) test scores. For
    # transductive we always compute BOTH raw and graph-enriched molecule variants.
    def probe_with(state, tag):
        print(f"  [{tag}] fitting XGBoost probe...", flush=True)
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in state.items()})
        model.eval()
        with torch.no_grad():
            z = (model.encode_history(x_dict, eidx, depth=args.history_depth)
                 if args.history else model.encode(x_dict, eidx))
        Xm_probe = (z[H.MOL] if args.transductive_exp else x_dict[H.MOL]).detach().cpu().numpy()
        Zp = z[H.PROT].detach().cpu().numpy()
        # The graph probe normally never sees the RAW ESM Р Р†Р вЂљРІР‚Сњ only the trained 256-d
        # proj+ReLU bottleneck. --concat_raw_prot puts the full raw ESM-1280 back
        # alongside z_prot, to tell "graph compresses raw detail" from "graph adds signal".
        prot_desc = f"{Zp.shape[1]} {args.arch.upper()}"
        if args.concat_raw_prot:
            Xp_raw = x_dict[H.PROT].detach().cpu().numpy()
            Zp = np.concatenate([Xp_raw, Zp], axis=1)
            prot_desc = f"{Xp_raw.shape[1]} raw ESM + {prot_desc}"

        def feats(Xm, idx):
            return np.concatenate([Xm[idx[0].detach().cpu().numpy()], Zp[idx[1].detach().cpu().numpy()]], axis=1)

        ytr = sup["probe_train"][1].detach().cpu().numpy()
        yte = sup["test"][1].detach().cpu().numpy()
        mol_desc = f"{args.arch.upper()}-enriched" if args.transductive_exp else "raw ChemBERTa"
        sc = train_boost(feats(Xm_probe, sup["probe_train"][0]), ytr,
                         feats(Xm_probe, sup["test"][0]), seed=args.boost_seed)
        print(f"  [{tag}] unentangled_boost EC50-TEST "
              f"({Xm_probe.shape[1] + Zp.shape[1]}d: mol {Xm_probe.shape[1]} {mol_desc} "
              f"+ prot {prot_desc}) "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics(yte, sc).items()))
        sc_raw = sc_enr = None
        if args.regime == "transductive":
            Xm_raw, Xm_enr = (x_dict[H.MOL].detach().cpu().numpy(),
                              z[H.MOL].detach().cpu().numpy())
            sc_raw = (sc if not args.transductive_exp
                      else train_boost(feats(Xm_raw, sup["probe_train"][0]), ytr,
                                       feats(Xm_raw, sup["test"][0]), seed=args.boost_seed))
            sc_enr = (sc if args.transductive_exp
                      else train_boost(feats(Xm_enr, sup["probe_train"][0]), ytr,
                                       feats(Xm_enr, sup["test"][0]), seed=args.boost_seed))
        return sc, sc_raw, sc_enr

    scores = scores_raw = scores_enriched = None
    if args.probe_checkpoint in ("best", "both"):
        print(f"  probing best encoder (epoch {int(best.epoch)}, val AUPRC={best_val_auprc:.4f})")
        scores, scores_raw, scores_enriched = probe_with(
            best_state, f"best-val e{int(best.epoch)}")

    scores_final = scores_final_raw = scores_final_enriched = None
    if args.probe_checkpoint in ("last", "both"):
        scores_final, scores_final_raw, scores_final_enriched = probe_with(
            final_state, "last-epoch")

    # ---- save in the shape graph_evaluation_full_full expects ----
    # variant fully identifies the run within (regime, arch):
    # mp_mode [+q##] [+history] [+rawp] [+transductive_exp] [+disjoint]
    q = args.mol_quality_q
    prefix = args.arch                          # "gnn" or "gat"
    ckpt = {
        "label": f"{variant}_unentangled_boost", "variant": variant,
        "head": "unentangled_boost", "regime": args.regime,
        "arch": args.arch, "fold": args.fold, "seed": args.seed,
        "gnn_seed": args.seed, "boost_seed": args.boost_seed,
        "config": {"mp_mode": args.mp_mode, "mol_quality_q": q, "history": args.history,
                   "history_depth": args.history_depth if args.history else 0,
                   "concat_raw_prot": args.concat_raw_prot,
                   "transductive_exp": args.transductive_exp,
                   "molecule_features": "graph_enriched" if args.transductive_exp else "raw",
                   "disjoint_probe_train": args.disjoint_probe_train,
                   "probe_train_frac": args.probe_train_frac,
                   "dgi_weight": args.dgi_weight,
                   "dgi_scope": args.dgi_scope,
                   "cl_eps": args.cl_eps,
                   "cl_weight": args.cl_weight,
                   "cl_temp": args.cl_temp,
                   "main_loss": args.main_loss,
                   "diagnostics": args.diagnostics,
                   "grad_clip": args.grad_clip,
                   "lr_scheduler": args.lr_scheduler,
                   "scheduler_patience": args.scheduler_patience,
                   "scheduler_factor": args.scheduler_factor,
                   "scheduler_min_lr": args.scheduler_min_lr,
                   "deterministic": args.deterministic,
                   "device_requested": args.device, "device_resolved": str(device),
                   "probe_checkpoint": args.probe_checkpoint,
                   "epochs": args.epochs, "lr": args.lr,
                   "protein_embeddings": args.protein_embeddings,
                   "molecule_embeddings": args.molecule_embeddings,
                   "protein_pca_dim": args.protein_pca_dim,
                   "molecule_pca_dim": args.molecule_pca_dim,
                   "pca_fit_scope": "full original train before disjoint split",
                   "hidden": args.hidden if args.arch == "gnn" else args.gat_hidden},
        "input_pca": input_pca,
        "gnn_train_sup": tuple(t.detach().cpu() for t in sup["train"]),
        "probe_train_sup": tuple(t.detach().cpu() for t in sup["probe_train"]),
        "test_sup": tuple(t.detach().cpu() for t in sup["test"]), "test_scores": scores,
        # For transductive runs both probe variants are always computed; inductive = None.
        "test_scores_raw": scores_raw,
        "test_scores_enriched": scores_enriched,
        # Same probe on the final (last-epoch) encoder, for best-vs-last comparison.
        "test_scores_final": scores_final,
        "test_scores_final_raw": scores_final_raw,
        "test_scores_final_enriched": scores_final_enriched,
        # Encoder weights at best-val epoch and at the final epoch.
        "best_state": best_state,
        "final_state": final_state,
        "decoder_history": history,
        "decoder_history_csv": str(history_csv.relative_to(_root)),
    }
    rd = _root / args.results_dir / "checkpoints"; rd.mkdir(parents=True, exist_ok=True)
    fname = (f"{prefix}_{variant}_unentangled_boost_{args.regime}_fold{args.fold}_"
             f"{seed_tag}.pt")
    torch.save(ckpt, rd / fname)
    print(f"  result bundle -> {args.results_dir}/checkpoints/{fname}")

    model_dir = _root / args.results_dir / "models"; model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_meta = {
        "arch": args.arch, "mp_mode": args.mp_mode, "regime": args.regime,
        "fold": args.fold, "gnn_seed": args.seed, "boost_seed": args.boost_seed,
        "epochs": args.epochs, "best_epoch": int(best.epoch),
        "best_val_AUPRC": float(best_val_auprc), "variant": variant,
        "config": ckpt["config"], "input_pca": input_pca,
    }
    torch.save({**snapshot_meta, "checkpoint": "best_val_auprc", "state_dict": best_state},
               model_dir / f"{run_id}__best_val_auprc.pt")
    torch.save({**snapshot_meta, "checkpoint": "last", "epoch": args.epochs,
                "state_dict": final_state}, model_dir / f"{run_id}__last.pt")
    print(f"  model snapshots -> {args.results_dir}/models/{run_id}__{{best_val_auprc,last}}.pt")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",    default="gnn", choices=["gnn", "gat"])
    ap.add_argument("--mp_mode", default="signed", choices=list(H.MP_MODES))
    ap.add_argument("--regime",  default="transductive",
                    choices=["transductive", "inductive_molecule"])
    ap.add_argument("--mol_quality_q", type=float, default=0.0,
                    help="Restrict MP edges to molecules above this coverage quantile (0=off)")
    ap.add_argument("--history", action="store_true",
                    help="Use protein history concat instead of final z_prot in the probe")
    ap.add_argument("--history_depth", type=int, default=2, choices=[1, 2],
                    help="History depth: 2=[x0||x1||x2] (last MP), 1=[x0||x1] (first MP)")
    ap.add_argument("--concat_raw_prot", action="store_true",
                    help="Probe on [raw ESM-1280 || z_prot] Р Р†Р вЂљРІР‚Сњ gives the probe the full "
                         "raw protein embedding the graph bottleneck otherwise discards")
    ap.add_argument("--transductive-exp", "--transductive_exp", "-transductive_exp",
                    dest="transductive_exp", action="store_true",
                    help="Transductive only: probe on [graph molecule || graph protein] "
                         "instead of [raw molecule || graph protein]")
    ap.add_argument("--disjoint-probe-train", "--disjoint_probe_train",
                    dest="disjoint_probe_train", action="store_true",
                    help="Split train labels: one disjoint half for GNN MP/decoder, "
                         "the other for XGBoost fitting")
    ap.add_argument("--probe-train-frac", type=float, default=0.5,
                    help="Fraction of train labels reserved for XGBoost in disjoint mode")
    ap.add_argument("--dgi-weight", "--dgi_weight", dest="dgi_weight", type=float, default=0.0,
                    help="Weight of the DeepGraphInfomax auxiliary loss added to the link "
                         "loss (0=off, reproduces v5 exactly). Try 0.5 as a first non-zero value.")
    ap.add_argument("--dgi-scope", "--dgi_scope", dest="dgi_scope", default="shared",
                    choices=["shared", "prot"],
                    help="DGI node set: 'shared' (A: molecules+proteins pooled into one "
                         "summary) or 'prot' (C: proteins only — targets the cold-molecule "
                         "niche where the graph helps).")
    ap.add_argument("--cl-eps", "--cl_eps", dest="cl_eps", type=float, default=0.1,
                    help="SimGCL noise magnitude epsilon (Yu et al. SIGIR'22): the L2 radius "
                         "of the uniform embedding perturbation. Only used when --cl-weight>0. "
                         "Paper's sweet spot ~0.1.")
    ap.add_argument("--cl-weight", "--cl_weight", dest="cl_weight", type=float, default=0.0,
                    help="Weight lambda of the SimGCL InfoNCE contrastive loss added to the "
                         "link loss (0=off, reproduces v5). Paper uses 0.2–2.0.")
    ap.add_argument("--cl-temp", "--cl_temp", dest="cl_temp", type=float, default=0.2,
                    help="InfoNCE temperature tau for the SimGCL loss (default 0.2, the "
                         "paper's value). Not part of the primary two-hyperparameter sweep.")
    ap.add_argument("--main-loss", "--main_loss", dest="main_loss", default="bce",
                    choices=["bce", "bpr"],
                    help="Main link objective: 'bce' (default, pointwise BCE with pos_weight, "
                         "reproduces v5) or 'bpr' (pairwise Bayesian Personalized Ranking, "
                         "LightGCN/SimGCL-style; negatives are the molecule's real tested "
                         "non-binders). Composes with --dgi-* and --cl-* add-ons.")
    ap.add_argument("--observe_test", action="store_true",
                    help="[DIAGNOSTICS ONLY] Log test-set metrics every epoch. "
                         "ONLY for gnn_training_diagnostics.ipynb. "
                         "NEVER use during HP sweeps Р Р†Р вЂљРІР‚Сњ test leaks into your mental model.")
    ap.add_argument("--fold",    type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--results-dir", default="results/graph/full_full_manual/")
    ap.add_argument("--protein-embeddings", default=None,
                    help="Protein embedding NPZ (absolute or relative to repository root)")
    ap.add_argument("--molecule-embeddings", default=None,
                    help="Molecule embedding NPZ/pickle (absolute or relative to repository root)")
    ap.add_argument("--protein-pca-dim", type=int, default=0,
                    help="PCA protein inputs to this dimension; fit on all original train nodes")
    ap.add_argument("--molecule-pca-dim", type=int, default=0,
                    help="PCA molecule inputs to this dimension; fit on all original train nodes")
    ap.add_argument("--hidden",     type=int,   default=256)   # GNN width
    ap.add_argument("--gat_hidden", type=int,   default=128)   # GAT width
    ap.add_argument("--heads",      type=int,   default=4)
    ap.add_argument("--dropout",    type=float, default=0.3)
    ap.add_argument("--lr",         type=float, default=5e-3)
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="Clip total gradient norm to this value (0=off)")
    ap.add_argument("--lr-scheduler", action="store_true",
                    help="Reduce LR when validation AUPRC stops improving")
    ap.add_argument("--scheduler-patience", type=int, default=100)
    ap.add_argument("--scheduler-factor", type=float, default=0.5)
    ap.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    ap.add_argument("--diagnostics", action="store_true",
                    help="Persist gradient/logit/parameter and per-stage embedding statistics")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use deterministic Torch algorithms (warn-only) and one CPU thread")
    ap.add_argument("--epochs",     type=int,   default=1500)
    ap.add_argument("--log-every",  type=int,   default=10,
                    help="Print compact validation/test status every N epochs")
    ap.add_argument("--plot-every", type=int,   default=25,
                    help="Refresh the history PNG every N epochs (CSV is saved every epoch)")
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="GNN compute device; auto prefers CUDA and falls back to CPU")
    ap.add_argument("--boost-seed", type=int, default=None,
                    help="Independent XGBoost seed (default: same as --seed for legacy runs)")
    ap.add_argument("--probe-checkpoint", choices=["best", "last", "both"], default="both",
                    help="Encoder checkpoint(s) evaluated by downstream XGBoost")
    args = ap.parse_args()
    if args.boost_seed is None:
        args.boost_seed = args.seed
    if args.transductive_exp and args.regime != "transductive":
        ap.error("--transductive-exp is only valid with --regime transductive")
    if not 0.0 < args.probe_train_frac < 1.0:
        ap.error("--probe-train-frac must be between 0 and 1")
    if args.grad_clip < 0:
        ap.error("--grad-clip must be non-negative")
    if args.dgi_weight < 0:
        ap.error("--dgi-weight must be non-negative (0 disables DGI)")
    if args.cl_weight < 0 or args.cl_eps < 0:
        ap.error("--cl-weight and --cl-eps must be non-negative (cl-weight 0 disables SimGCL)")
    if args.protein_pca_dim < 0 or args.molecule_pca_dim < 0:
        ap.error("PCA dimensions must be non-negative (0 disables PCA)")
    if args.lr_scheduler and not 0.0 < args.scheduler_factor < 1.0:
        ap.error("--scheduler-factor must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
