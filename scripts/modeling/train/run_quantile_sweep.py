"""Autonomous stage-2 driver for the graph_sweep_v3 grid.

1. Wait until all 12 base grid checkpoints exist (the GAT pass finishing).
2. Pick the single best mp_mode for GNN and for GAT by the *inductive_molecule*
   metric aggregate (mean of AUROC, AUPRC, MCC, F1 -- ROC AUC is NOT emphasised).
3. Run the data-quality quantile sweep for those two variants:
   {chosen gnn mp, chosen gat mp} x q in {0.87, 0.95, 0.99} x {transductive,
   inductive_molecule}, on the same stable recipe, into graph_sweep_v3 so the
   notebook's ff-D cell picks them up (variants signed_q87 / signed_q95 / ...).

Fully self-contained: launch detached and it drives the whole thing to completion.
Writes results/graph/full_full/legacy/pre_v5/sweep_v3/full_full/stage2_selection.json and
stage2_status.csv for a morning-after report.
"""
import sys
import json
import time
import subprocess
from pathlib import Path
from collections import deque

import numpy as np
import torch

ROOT = Path(__file__).resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from orbind.dataset import metrics  # noqa: E402

PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINER = ROOT / "scripts" / "modeling" / "train" / "train_graph_full_full.py"
RESULT_DIR = "results/graph/full_full/legacy/pre_v5/sweep_v3/full_full"
RUN_ROOT = ROOT / RESULT_DIR
CKPT_DIR = RUN_ROOT / "checkpoints"
LOG_DIR = RUN_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FOLD = 1
EPOCHS = 1500
THROTTLE = 3
ARCHS = ["gnn", "gat"]
MPS = ["pos_only", "all_edges", "signed"]
REGIMES = ["transductive", "inductive_molecule"]
QUANTILES = [0.87, 0.95, 0.99]
SELECT_METRICS = ["AUROC", "AUPRC", "MCC", "F1"]  # equal weight; ROC AUC not favoured

STABLE = ["--lr", "1e-3", "--grad-clip", "1.0", "--lr-scheduler",
          "--scheduler-patience", "100", "--scheduler-factor", "0.5",
          "--scheduler-min-lr", "1e-5", "--diagnostics", "--deterministic"]


def base_ckpt(arch, mp, regime):
    return CKPT_DIR / f"{arch}_{mp}_unentangled_boost_{regime}_fold{FOLD}.pt"


def quant_ckpt(arch, mp, regime, q):
    qq = int(q * 100)
    return CKPT_DIR / f"{arch}_{mp}_q{qq}_unentangled_boost_{regime}_fold{FOLD}.pt"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def score_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _, y = ckpt["test_sup"]
    m = metrics(y.numpy(), np.asarray(ckpt["test_scores"]))
    return m


def wait_for_base_grid(timeout_h=6):
    """Block until every base grid checkpoint exists."""
    needed = [base_ckpt(a, mp, r) for a in ARCHS for mp in MPS for r in REGIMES]
    deadline = time.time() + timeout_h * 3600
    while True:
        missing = [p for p in needed if not p.exists()]
        if not missing:
            log(f"all {len(needed)} base grid checkpoints present")
            return True
        if time.time() > deadline:
            log(f"TIMEOUT waiting for base grid; still missing: "
                f"{[p.name for p in missing]}")
            return False
        log(f"waiting for base grid: {len(missing)}/{len(needed)} missing "
            f"(e.g. {missing[0].name})")
        time.sleep(60)


def select_best():
    """Best mp per arch by mean inductive metric aggregate."""
    selection, table = {}, []
    for arch in ARCHS:
        best_mp, best_comp = None, -1e9
        for mp in MPS:
            m = score_ckpt(base_ckpt(arch, mp, "inductive_molecule"))
            comp = float(np.mean([m[k] for k in SELECT_METRICS]))
            table.append({"arch": arch, "mp": mp, "composite": round(comp, 4),
                          **{k: round(m[k], 4) for k in SELECT_METRICS}})
            if comp > best_comp:
                best_comp, best_mp = comp, mp
        selection[arch] = best_mp
        log(f"best {arch}: {best_mp} (inductive composite={best_comp:.4f})")
    (RUN_ROOT / "stage2_selection.json").write_text(
        json.dumps({"selection": selection, "ranking": table,
                    "criterion": f"mean({'+'.join(SELECT_METRICS)}) on inductive_molecule"},
                   indent=2))
    return selection


def build_runs(selection):
    runs = []
    for arch, mp in selection.items():
        for q in QUANTILES:
            for regime in REGIMES:
                runs.append({"arch": arch, "mp": mp, "q": q, "regime": regime,
                             "ckpt": quant_ckpt(arch, mp, regime, q)})
    return runs


def run_pool(runs):
    pending = deque(runs)
    running = []          # (popen, run, fh)
    done = []
    while pending or running:
        while len(running) < THROTTLE and pending:
            r = pending.popleft()
            if r["ckpt"].exists():
                log(f"skip (cached): {r['ckpt'].name}")
                done.append({**_row(r), "state": "cached"})
                continue
            qq = int(r["q"] * 100)
            name = f"{r['arch']}_{r['mp']}_q{qq}_{r['regime']}"
            fh = open(LOG_DIR / f"{name}.log", "w")
            args = [str(PYTHON), str(TRAINER),
                    "--arch", r["arch"], "--mp_mode", r["mp"],
                    "--regime", r["regime"], "--mol_quality_q", str(r["q"]),
                    *STABLE, "--fold", str(FOLD), "--epochs", str(EPOCHS),
                    "--results-dir", RESULT_DIR, "--log-every", "10",
                    "--plot-every", "25"]
            p = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT))
            running.append((p, r, fh))
            log(f"launched {name} (pid {p.pid})")
        time.sleep(15)
        for p, r, fh in running[:]:
            if p.poll() is not None:
                fh.close()
                state = "completed" if r["ckpt"].exists() else "failed"
                done.append({**_row(r), "state": state})
                log(f"{state}: {r['ckpt'].name}")
                running.remove((p, r, fh))
                _write_status(done)
    _write_status(done)
    return done


def _row(r):
    return {"arch": r["arch"], "mp": r["mp"], "q": r["q"],
            "regime": r["regime"], "checkpoint": r["ckpt"].name}


def _write_status(done):
    import csv
    with open(RUN_ROOT / "stage2_status.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "mp", "q", "regime",
                                          "checkpoint", "state"])
        w.writeheader()
        w.writerows(done)


def main():
    log("stage-2 quantile driver started")
    if not wait_for_base_grid():
        log("aborting: base grid incomplete")
        return
    selection = select_best()
    runs = build_runs(selection)
    log(f"quantile sweep: {len(runs)} runs -> {selection}")
    done = run_pool(runs)
    ok = sum(1 for d in done if d["state"] in ("completed", "cached"))
    log(f"stage-2 done: {ok}/{len(done)} ok")


if __name__ == "__main__":
    main()
