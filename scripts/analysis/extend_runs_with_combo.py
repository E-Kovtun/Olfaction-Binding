"""Print the commands that add a `cls+prot+mol` row to existing cls-only runs.

    python scripts/analysis/extend_runs_with_combo.py --pool m2or-transductive-chemberta-fixed

Prints, does not execute. Copy the block it emits.

Why a generator instead of six hand-written commands
----------------------------------------------------
The heavy extractors (prosmith / lorax / molor / gnn) persist their trained model
at `checkpoints/repeat_{R}/<type>_<name>_model{m}.pt` and, when that file exists,
**skip training and only embed**. So re-running a finished run with an extra
combo is nearly free -- but only if the `--source` string reproduces the same
extractor. One changed field and you silently retrain a different model, or the
state_dict refuses to load. This reads each run's own `config.json` and reuses
its cls source verbatim.

The `prot` and `mol` sources are copied from the pool's own boost run for the
same reason: the new combo has to sit on exactly the features the pool's
reference row uses, or the comparison is between two different feature sets.

Combo numbering follows the emitted `--source` order: 1=cls, 2=prot, 3=mol, so
`--combos "1 123"` gives the existing `cls` row plus `cls+prot+mol`. Ensemble
weights (simplex/logreg) are still fitted -- they are written with a different
`kind` and `summarize_runs.py` ignores them.

Environments are chosen per method: prosmith/lorax need `.venv-controls`
(transformers+peft, PyG-free), molor needs `.venv-molor` (dgl), everything else
runs in the default env.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent

BASE = _root / "results" / "ensemble_logs"
ENTITY = {"esm", "gin"}
PYTHON = {"prosmith": ".venv-controls/bin/python",
          "lorax": ".venv-controls/bin/python",
          "molor": ".venv-molor/bin/python"}
DEFAULT_PYTHON = "uv run python"


def parse(src: str) -> tuple[str, str]:
    """'name=type:rest' -> (name, type)"""
    name, _, rest = src.partition("=")
    return name, rest.split(":")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="pool folder under results/ensemble_logs")
    ap.add_argument("--combos", default="1 123")
    ap.add_argument("--gpus", default="0 1 2 3")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()

    pool = pathlib.Path(args.base) / args.pool
    if not pool.exists():
        sys.exit(f"{pool} не найдено")

    runs = {}
    for r in sorted(p for p in pool.iterdir() if p.is_dir()):
        cfg = r / "config.json"
        if cfg.exists():
            runs[r.name] = json.loads(cfg.read_text())

    # the pool's boost run defines the prot/mol half every new combo must sit on
    prot = mol = None
    for name, c in runs.items():
        srcs = [parse(s) for s in c.get("sources", [])]
        if srcs and all(t in ENTITY for _, t in srcs):
            for s in c["sources"]:
                n, _ = parse(s)
                if n == "prot":
                    prot = s
                elif n == "mol":
                    mol = s
    if not (prot and mol):
        sys.exit(f"в пуле {args.pool} не нашёлся бустинговый прогон с источниками prot и mol — "
                 f"без него не с чем сращивать")

    print(f"# prot: {prot}\n# mol:  {mol}\n")
    emitted = 0
    for name, c in sorted(runs.items()):
        srcs = [parse(s) for s in c.get("sources", [])]
        learned = [(n, t) for n, t in srcs if t not in ENTITY]
        if len(learned) != 1 or len(srcs) != 1:
            continue                      # boost-only, or already has prot/mol
        cls_src = c["sources"][0]
        _, typ = learned[0]
        py = PYTHON.get(typ, DEFAULT_PYTHON)
        regime = c.get("regime", "full_full")
        mode = c.get("full_full_mode", "transductive")
        # `config.json` records only the LAST invocation, so a run finished in
        # chunks reports a partial repeat list (inductive_lorax says [45, 46]
        # while holding 5 folds). Always emit the full five for the regime.
        repeats = ([42, 43, 44, 45, 46] if mode == "inductive_molecule_v5"
                   else [1, 2, 3, 4, 5])
        ckpt = pool / name / "checkpoints"
        have = sorted(p.name for p in ckpt.glob("repeat_*")) if ckpt.exists() else []
        print(f"# --- {name}  [{typ}]  чекпойнты: {len(have)} repeat_* "
              f"{'(обучение пропустится)' if len(have) >= 5 else '!! ПЕРЕОБУЧИТСЯ'}")
        print(f"{py} scripts/modeling/train/train_ensemble_boost.py \\\n"
              f"  --regime {regime} --full-full-mode {mode} "
              f"--repeats {' '.join(map(str, repeats))} \\\n"
              f"  --max-parallel {args.max_parallel} --gpus {args.gpus} \\\n"
              f"  --source {cls_src} \\\n"
              f"  --source {prot} \\\n"
              f"  --source {mol} \\\n"
              f'  --combos "{args.combos}" --on-missing drop \\\n'
              f"  --out-dir results/ensemble_logs/{args.pool} \\\n"
              f"  --run-name {name}\n")
        emitted += 1
    if not emitted:
        print("# нечего расширять: в пуле нет прогонов ровно с одним обучаемым источником")


if __name__ == "__main__":
    main()
