"""The article-facing alpha-choice table.

The whole point of this table is a protocol: alpha is chosen on validation and read
once on test. So the tests are about the protocol and not about the arithmetic, which
belongs to `scripts/analysis/alpha_choice.py` and is tested there:

* a run without validation rows must REFUSE, not fall back to test;
* WHICH run is being rendered is chosen from outside and required -- no default root,
  and the output is named after the run, so two runs cannot overwrite each other;
* the confirmed alpha must be the one validation picked, not the one test likes;
* `optimism` must be non-negative and must be measured on test;
* the 1-SE tie set must be marked, because an argmax out of a flat set is noise.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_ac_table", _root / "scripts/article_tables/06_alpha_choice.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

ALPHAS = [0.0, 0.5, 1.0]
COLS = ["AUROC", "AUPRC", "MCC", "F1"]


def sweep(tmp_path, val_peak=0.5, test_peak=1.0, with_val=True, panels=("m2or",),
          flat=False):
    """A sweep on disk whose validation and test curves peak at different alphas."""
    fam = {"m2or": "inductive_molecule_v5", "cc": "our_inductive",
           "hc": "our_inductive"}
    var = {"m2or": "q99greedy", "cc": "q0cov", "hc": "q0cov"}
    cols = {"m2or": COLS, "cc": ["R2", "RMSE", "MAE", "Pearson", "Spearman"],
            "hc": ["R2", "RMSE", "MAE", "Pearson", "Spearman"]}

    def rows(ds, split, peak):
        out = []
        for fold in (1, 2, 3, 4, 5):
            for seed in (42, 43):
                for a in ALPHAS:
                    v = (0.80 if flat
                         else 0.80 + 0.05 * (1 - abs(a - peak)) + 0.001 * seed)
                    out.append(dict(arm="gate", alpha=a, fold=fold, seed=seed,
                                    status="ok", combo="cls+mol", split=split,
                                    variant=var[ds],
                                    **{m: v for m in cols[ds]}))
                out.append(dict(arm="boost_full", alpha=np.nan, fold=fold, seed=seed,
                                status="ok", combo="prot+mol", split=split,
                                variant=var[ds],
                                **{m: 0.80 for m in cols[ds]}))
        return pd.DataFrame(out)

    for ds in panels:
        stem = f"{ds}_{fam[ds]}_chemberta_nodedial"
        rows(ds, "test", test_peak).to_csv(tmp_path / f"metrics_{stem}.csv", index=False)
        if with_val:
            rows(ds, "val", val_peak).to_csv(tmp_path / f"val_metrics_{stem}.csv",
                                             index=False)
    return tmp_path


def run(tmp_path, *extra):
    out = tmp_path / "out"
    rc = T.main(["--sweep-root", str(tmp_path), "--out", str(out), *extra])
    return rc, out


# ----------------------------------------------------------------- the protocol

def test_without_validation_it_refuses_rather_than_using_test(tmp_path, capsys):
    """The one failure mode this file exists to prevent: a choice made on test and
    typeset as if it had been made on validation. It prints the command that scores
    THIS root and stops -- it does not score anything itself."""
    root = sweep(tmp_path, with_val=False)
    rc, out = run(root)
    assert rc == 1
    assert not (out / "alpha_choice.tex").exists()
    printed = capsys.readouterr().out
    assert "val_rescore.py" in printed
    assert str(root) in printed


# ------------------------------------------------------- which run, chosen outside

def test_the_sweep_root_is_required(capsys):
    """No default root: the ESM3 grid and an ESM-1b grid are different runs, and a
    default is how the wrong one gets typeset without anybody noticing."""
    with pytest.raises(SystemExit):
        T.parser().parse_args(["--nodes", "nodedial"])
    assert "--sweep-root" in capsys.readouterr().err


def test_the_run_is_the_roots_own_name(tmp_path):
    assert T.run_name("results/graph/v13_esm3") == "v13_esm3"
    assert T.run_name("results/graph/v9_seeded/") == "v9_seeded"
    assert T.run_name(tmp_path / "run_a") != T.run_name(tmp_path / "run_b")


def test_the_default_output_is_named_after_the_run(tmp_path, monkeypatch):
    """Two runs rendered the same day must not land on top of each other, and that is
    decided by the root's name rather than by whoever typed --out."""
    root = tmp_path / "v_other"
    root.mkdir()
    sweep(root)
    seen = {}

    def out_dir(path):
        seen["path"] = str(path)
        d = tmp_path / "rendered"
        d.mkdir(exist_ok=True)
        return d

    monkeypatch.setattr(T.tk, "out_dir", out_dir)
    assert T.main(["--sweep-root", str(root)]) == 0
    assert seen["path"] == "results/article_tables/v_other/alpha_choice"


def test_the_confirmed_alpha_is_the_one_validation_picked(tmp_path):
    """Validation peaks at 0.5 and test at 1.0. A script that quietly chose on test
    would report 1.0 and look better doing it."""
    rc, out = run(sweep(tmp_path, val_peak=0.5, test_peak=1.0))
    assert rc == 0
    cf = pd.read_csv(out / "alpha_choice_confirm.csv")
    assert (cf["alpha"] == 0.5).all()


def test_optimism_is_what_choosing_on_test_would_have_added(tmp_path):
    """It is measured on test, against the alpha validation picked, and cannot be
    negative -- if it were, the 'best' column would not be the best."""
    _, out = run(sweep(tmp_path, val_peak=0.5, test_peak=1.0))
    cf = pd.read_csv(out / "alpha_choice_confirm.csv")
    r = cf.iloc[0]
    assert r["optimism"] > 0
    assert r["optimism"] == pytest.approx(r["best"] - r["value"])


def test_no_optimism_when_the_two_splits_agree(tmp_path):
    _, out = run(sweep(tmp_path, val_peak=1.0, test_peak=1.0))
    cf = pd.read_csv(out / "alpha_choice_confirm.csv")
    assert cf.iloc[0]["optimism"] == pytest.approx(0.0, abs=1e-9)
    assert cf.iloc[0]["rank"] == 1


def test_at_alpha_overrides_the_choice_and_says_so(tmp_path, capsys):
    """For the sensitivity paragraph. It must not be silent: a table whose alpha came
    from the command line and whose caption says 'chosen on validation' is a lie."""
    _, out = run(sweep(tmp_path, val_peak=0.5), "--at-alpha", "1.0")
    assert (pd.read_csv(out / "alpha_choice_confirm.csv")["alpha"] == 1.0).all()
    assert "asked for on the command line" in capsys.readouterr().out


# ----------------------------------------------------------------- the tables

def test_the_tie_set_is_marked(tmp_path):
    """A perfectly flat dial must not crown a winner: every alpha lands in the 1-SE
    set and the LaTeX carries the dagger."""
    _, out = run(sweep(tmp_path, flat=True))
    rk = pd.read_csv(out / "alpha_choice_rank.csv")
    assert rk["tied"].sum() >= 2
    assert r"\dagger" in (out / "alpha_choice.tex").read_text(encoding="utf-8")


def test_both_tables_and_their_labels_are_written(tmp_path):
    _, out = run(sweep(tmp_path))
    tex = (out / "alpha_choice.tex").read_text(encoding="utf-8")
    assert r"\label{tab:alphachoice}" in tex
    assert r"\label{tab:alphaconfirm}" in tex
    # every body row ends a LaTeX row exactly once
    assert not [l for l in tex.splitlines() if l.rstrip().endswith(r"\\\\")]


def test_the_reference_arms_are_in_the_ranking(tmp_path):
    """`boost` is ranked beside the dial on purpose: 'which row of the table wins on
    average' has to be a number, not a reading of three tables by eye."""
    _, out = run(sweep(tmp_path))
    rk = pd.read_csv(out / "alpha_choice_rank.csv")
    assert "boost" in set(rk["competitor"])


def test_leave_one_dataset_out_appears_only_with_more_than_one(tmp_path):
    _, out = run(sweep(tmp_path, panels=("m2or",)))
    assert not (out / "alpha_choice_loo.csv").exists()


def test_leave_one_dataset_out_runs_on_three_panels(tmp_path):
    _, out = run(sweep(tmp_path, panels=("m2or", "cc", "hc")))
    lo = pd.read_csv(out / "alpha_choice_loo.csv")
    assert len(lo) == 3
    assert set(lo.columns) >= {"held_out", "chosen_on_other_two"}


def test_a_missing_number_prints_as_a_dash_not_as_nan(tmp_path):
    """Half a grid is what this looks like mid-run, and `nan` in a typeset table is
    worse than an explicit gap."""
    row = pd.Series(dict(value=np.nan, base=np.nan, d=np.nan, won=np.nan, n=np.nan,
                         rank=np.nan, n_alphas=3, optimism=np.nan))
    assert T._cells(row) == ("--", "--", "--", "--", "--")
