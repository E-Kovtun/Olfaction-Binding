"""Molecule similarity-graph message passing — external compute (server/GPU).

Builds a kNN graph in molecule (ChemBERTa) embedding space and probes three
molecule representations with XGBoost, in both regimes, over 5 repeats each:

  raw          : [raw mol | raw prot]                     (no molecule graph)
  mp_untrained : parameter-free 1-hop propagation z=A.X   (graph, no training)
  mp_trained   : 1-layer SAGEConv + MLP link head, trained then head dropped

Repeat semantics (matches the full_full v5 convention):
  transductive       -> genuine LoRaX folds 1..5
  inductive_molecule -> fold 1 with cold-split seeds 42..46

The inductive cold val/test split is STRATIFIED by positive fraction so val and
test share ~the same prevalence (removes the AUPRC-floor confound). Embeddings are
standardised (raw ESM/ChemBERTa are mean-dominated).

Writes one tidy CSV of per-repeat metrics; the notebook only reads + plots it.
Usage:
  uv run python scripts/modeling/eval/run_molecule_graph_mp.py --device cuda
"""
import argparse, pathlib, sys, copy
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
from orbind import lorax as L, hetero as H
from orbind.baselines import train_boost
from orbind.dataset import metrics

MOL, PROT = H.MOL, H.PROT
METRICS = ["AUROC", "AUPRC", "MCC", "F1"]
BARS = ["raw", "mp_untrained", "mp_trained"]
REPEATS = {"transductive": [1, 2, 3, 4, 5], "inductive_molecule": [42, 43, 44, 45, 46]}


def standardize(d):
    keys = list(d)
    X = np.stack([np.asarray(d[k], np.float64) for k in keys])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return {k: ((np.asarray(d[k], np.float64) - mu) / sd).astype(np.float32) for k in keys}


def build_knn_edges(X, k):
    Xt = torch.tensor(X, dtype=torch.float)
    Xn = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-8)
    sim = Xn @ Xn.t(); sim.fill_diagonal_(float("-inf"))
    nbr = sim.topk(k, dim=1).indices
    dst = torch.arange(X.shape[0]).repeat_interleave(k); src = nbr.reshape(-1)
    ei = torch.stack([src, dst], dim=0); ei = torch.cat([ei, ei.flip(0)], dim=1)
    return torch.unique(ei, dim=1)


def gcn_propagate(X, ei):
    Xt = torch.tensor(X, dtype=torch.float) if not torch.is_tensor(X) else X
    n = Xt.size(0); sl = torch.arange(n, device=Xt.device)
    e = torch.cat([ei, torch.stack([sl, sl])], dim=1); s, d = e
    dg = torch.zeros(n, device=Xt.device).index_add_(0, d, torch.ones(e.size(1), device=Xt.device))
    dinv = dg.pow(-0.5); dinv[torch.isinf(dinv)] = 0
    out = torch.zeros_like(Xt); out.index_add_(0, d, (dinv[s] * dinv[d]).unsqueeze(-1) * Xt[s])
    return out


def stratified_cold_split(base, val_every=3, seed=42):
    """Rebalance held-out (cold) molecules into val/test with matched prevalence."""
    hp = np.concatenate([base["val"]["pos"], base["test"]["pos"]], 0)
    hn = np.concatenate([base["val"]["neg"], base["test"]["neg"]], 0)
    mols = sorted(set(hp[:, 0]).union(hn[:, 0]))
    pos_by = {m: hp[hp[:, 0] == m] for m in mols}; neg_by = {m: hn[hn[:, 0] == m] for m in mols}
    pf = lambda m: len(pos_by[m]) / max(len(pos_by[m]) + len(neg_by[m]), 1)
    rng = np.random.default_rng(seed)
    order = sorted(mols, key=lambda m: (pf(m), rng.random()))
    vmol = [m for i, m in enumerate(order) if i % val_every == 0]
    tmol = [m for i, m in enumerate(order) if i % val_every != 0]
    def cat(ms, by):
        a = [by[m] for m in ms if len(by[m])]
        return np.concatenate(a, 0) if a else np.zeros((0, 2), int)
    return {"train": base["train"],
            "val":  {"pos": cat(vmol, pos_by), "neg": cat(vmol, neg_by)},
            "test": {"pos": cat(tmol, pos_by), "neg": cat(tmol, neg_by)}}


class MolGraphLink(nn.Module):
    def __init__(self, mol_dim, prot_dim, hidden=256, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.proj_m = nn.Linear(mol_dim, hidden)
        self.conv = SAGEConv(hidden, hidden)
        self.proj_p = nn.Linear(prot_dim, hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode_mol(self, Xm, ei):
        return self.conv(F.relu(self.proj_m(Xm)), ei)

    def forward(self, Xm, ei, Xp, pm, pp):
        zm = self.encode_mol(Xm, ei); zp = F.relu(self.proj_p(Xp))
        return self.head(torch.cat([zm[pm], zp[pp]], dim=-1)).squeeze(-1)


def train_mol_graph(splits, Xm_t, Xp_t, edge_index, device, hidden, epochs, lr, seed, log_tag=""):
    torch.manual_seed(seed)
    tr_idx, ytr = H.sup_edges(splits["train"]); va_idx, yva = H.sup_edges(splits["val"])
    ytr = ytr.to(device); yva_np = yva.numpy()
    model = MolGraphLink(Xm_t.shape[1], Xp_t.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    best_state, best = None, -1.0
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xm_t, edge_index, Xp_t, tr_idx[0], tr_idx[1]), ytr)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vs = torch.sigmoid(model(Xm_t, edge_index, Xp_t, va_idx[0], va_idx[1])).cpu().numpy()
        va = metrics(yva_np, vs)["AUPRC"]
        if va > best:
            best = va
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if ep % 50 == 0 or ep == 1:
            print(f"{log_tag} epoch {ep:4d} | loss {loss.item():.4f} | val AUPRC {va:.3f} (best {best:.3f})", flush=True)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        return model.encode_mol(Xm_t, edge_index).cpu().numpy()


def feat(Xm_a, Xp_a, idx):
    return np.concatenate([Xm_a[idx[0].numpy()], Xp_a[idx[1].numpy()]], axis=1).astype(np.float32)


def run_unit(regime, rep, esm, chem, args, device):
    """One (regime, repeat) unit -> rows for raw / mp_untrained / mp_trained."""
    i = REPEATS[regime].index(rep)
    fold = rep if regime == "transductive" else 1
    split_seed = 42 if regime == "transductive" else rep
    gnn_seed = 42 + i
    tag = f"[{regime} rep{rep}]"
    print(f"{tag} building graph (k={args.k})...", flush=True)
    Xm_t, Xp_t, splits = L.build(regime, fold, esm, chem, seed=split_seed)
    if regime == "inductive_molecule":
        splits = stratified_cold_split(splits, seed=split_seed)          # matched prevalence
    Xm, Xp = Xm_t.numpy(), Xp_t.numpy()
    edge_index = build_knn_edges(Xm, args.k).to(device)
    Xm_dev, Xp_dev = torch.tensor(Xm, device=device), torch.tensor(Xp, device=device)
    Xm_prop = gcn_propagate(Xm_dev, edge_index).cpu().numpy()
    tr_idx, ytr = H.sup_edges(splits["train"]); te_idx, yte = H.sup_edges(splits["test"])
    ytr_np, yte_np = ytr.numpy(), yte.numpy(); prev = float(yte_np.mean())

    rows = []
    def probe(bar, Xm_repr):
        print(f"{tag} probe {bar}...", flush=True)
        sc = train_boost(feat(Xm_repr, Xp, tr_idx), ytr_np, feat(Xm_repr, Xp, te_idx), seed=args.boost_seed)
        m = metrics(yte_np, sc)
        rows.append({"regime": regime, "repeat": rep, "fold": fold, "gnn_seed": gnn_seed,
                     "bar": bar, "mol_dim": Xm_repr.shape[1], "test_prevalence": round(prev, 4),
                     **{k: float(m[k]) for k in METRICS}})
        print(f"{tag} {bar:13s} prev={prev:.3f} " + " ".join(f"{k}={m[k]:.3f}" for k in METRICS), flush=True)

    probe("raw", Xm)
    probe("mp_untrained", Xm_prop)
    print(f"{tag} training molecule graph net...", flush=True)
    z_tr = train_mol_graph(splits, Xm_dev, Xp_dev, edge_index, device,
                           args.hidden, args.epochs, args.lr, gnn_seed, log_tag=tag)
    probe("mp_trained", z_tr)
    print(f"{tag} UNIT_DONE", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--boost-seed", type=int, default=42)
    ap.add_argument("--regimes", nargs="+", default=list(REPEATS))
    ap.add_argument("--regime", default=None, help="single-unit mode: one regime")
    ap.add_argument("--repeat", type=int, default=None, help="single-unit mode: one repeat id")
    ap.add_argument("--out", default="results/graph/mol_graph_mp/tables/molecule_graph_mp_runs.csv")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out = _root / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | epochs={args.epochs} | k={args.k} | hidden={args.hidden}", flush=True)

    esm, chem = L.load_embeddings()
    esm, chem = standardize(esm), standardize(chem)

    if args.regime is not None and args.repeat is not None:
        units = [(args.regime, args.repeat)]                 # one unit (launcher schedules these)
    else:
        units = [(r, rep) for r in args.regimes for rep in REPEATS[r]]

    rows = []
    for regime, rep in units:
        rows += run_unit(regime, rep, esm, chem, args, device)
        pd.DataFrame(rows).to_csv(out, index=False)          # incremental: survive interruption
    print(f"\nsaved {out.relative_to(_root)} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
