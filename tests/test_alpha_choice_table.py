"""The article-facing alpha-choice table.

The whole point of this table is a protocol: alpha is chosen on validation and read
once on test. So the tests are about the protocol and not about the arithmetic, which
belongs to `scripts/analysis/alpha_choice.py` and is tested there:

* a run without validation rows must REFUSE, not fall back to test;
* WHICH run is being rendered is chosen from outside and required -- no default root,
  and the output is named after the run, so two runs cannot overwrite each other;
* BOTH boosting heads are rendered, each as its own panel, because they can prefer
  different dial positions and the paper reports one of them;
* the dial trend is tested rather than eyeballed, and a flat dial must come back as
  indistinguishable from zero instead of crowning whichever end drifted up;
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
    "_ac_table", _root / "scripts/article_tables/06_alpha_choice_not_used.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

ALPHAS = [0.0, 0.5, 1.0]
COLS = ["AUROC", "AUPRC", "MCC", "F1"]


def sweep(tmp_path, val_peak=0.5, test_peak=1.0, with_val=True, panels=("m2or",),
          flat=False, combos=("cls+mol",), slope=None):
    """A sweep on disk whose validation and test curves peak at different alphas.

    `combos` writes the same curves under more than one boosting head, which is what
    the two-panel rendering reads. `slope` replaces the peak with a straight line in
    alpha, which is what the trend test needs to have something to find."""
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
                    if flat:
                        v = 0.80
                    elif slope is not None:
                        v = 0.80 + slope * a + 0.001 * seed
                    else:
                        v = 0.80 + 0.05 * (1 - abs(a - peak)) + 0.001 * seed
                    for combo in combos:
                        out.append(dict(arm="gate", alpha=a, fold=fold, seed=seed,
                                        status="ok", combo=combo, split=split,
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


def trend_of(root, head="cls+mol"):
    """`dial_trend` on a root, through the same loader the script uses -- the raw CSVs
    carry no `dataset` column, the loader derives it from the file name."""
    kw = dict(root=str(root), nodes="nodedial", combo=head)
    val = T.ag.load(split="val", **kw)
    metric_of = T.ac._metric_of(val, T.argparse.Namespace(metric=None))
    return T.dial_trend(val, metric_of)


def rank_csv(out, head="cls+mol"):
    return pd.read_csv(out / f"alpha_choice_rank_{T.slug(head)}.csv")


def confirm_csv(out, head="cls+mol"):
    return pd.read_csv(out / f"alpha_choice_confirm_{T.slug(head)}.csv")


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
    cf = confirm_csv(out)
    assert (cf["alpha"] == 0.5).all()


def test_optimism_is_what_choosing_on_test_would_have_added(tmp_path):
    """It is measured on test, against the alpha validation picked, and cannot be
    negative -- if it were, the 'best' column would not be the best."""
    _, out = run(sweep(tmp_path, val_peak=0.5, test_peak=1.0))
    cf = confirm_csv(out)
    r = cf.iloc[0]
    assert r["optimism"] > 0
    assert r["optimism"] == pytest.approx(r["best"] - r["value"])


def test_no_optimism_when_the_two_splits_agree(tmp_path):
    _, out = run(sweep(tmp_path, val_peak=1.0, test_peak=1.0))
    cf = confirm_csv(out)
    assert cf.iloc[0]["optimism"] == pytest.approx(0.0, abs=1e-9)
    assert cf.iloc[0]["rank"] == 1


def test_at_alpha_overrides_the_choice_and_says_so(tmp_path, capsys):
    """For the sensitivity paragraph. It must not be silent: a table whose alpha came
    from the command line and whose caption says 'chosen on validation' is a lie."""
    _, out = run(sweep(tmp_path, val_peak=0.5), "--at-alpha", "1.0")
    assert (confirm_csv(out)["alpha"] == 1.0).all()
    assert "asked for on the command line" in capsys.readouterr().out


# ----------------------------------------------------------------- the tables

def test_the_tie_set_is_marked(tmp_path):
    """A perfectly flat dial must not crown a winner: every alpha lands in the 1-SE
    set and the LaTeX carries the dagger."""
    _, out = run(sweep(tmp_path, flat=True))
    rk = rank_csv(out)
    assert rk["tied"].sum() >= 2
    assert r"\dagger" in (out / "alpha_choice.tex").read_text(encoding="utf-8")


def test_both_tables_and_their_labels_are_written(tmp_path):
    _, out = run(sweep(tmp_path))
    tex = (out / "alpha_choice.tex").read_text(encoding="utf-8")
    assert r"\label{tab:alphachoice-clsmol}" in tex
    assert r"\label{tab:alphaconfirm-clsmol}" in tex
    # every body row ends a LaTeX row exactly once
    assert not [l for l in tex.splitlines() if l.rstrip().endswith(r"\\\\")]


def test_the_reference_arms_are_in_the_ranking(tmp_path):
    """`boost` is ranked beside the dial on purpose: 'which row of the table wins on
    average' has to be a number, not a reading of three tables by eye."""
    _, out = run(sweep(tmp_path))
    rk = rank_csv(out)
    assert "boost" in set(rk["competitor"])


def test_leave_one_dataset_out_is_gone(tmp_path):
    """It measured the wandering of an argmax over a flat surface and read as a
    verdict. The trend test answers the same question without the theatre."""
    _, out = run(sweep(tmp_path, panels=("m2or", "cc", "hc")))
    assert not (out / "alpha_choice_loo.csv").exists()
    assert not hasattr(T, "robustness")


# ----------------------------------------------------------------- both heads

def test_each_head_gets_its_own_panel(tmp_path):
    """The two heads sit on ONE trained graph and can prefer different dial
    positions, so neither may stand in for the other."""
    root = sweep(tmp_path, combos=("cls+mol", "cls+prot+mol"))
    rc, out = run(root)
    assert rc == 0
    for head in ("cls+mol", "cls+prot+mol"):
        assert len(rank_csv(out, head)) and len(confirm_csv(out, head))
    tex = (out / "alpha_choice.tex").read_text(encoding="utf-8")
    for tag in ("clsmol", "clsprotmol"):
        assert rf"\label{{tab:alphachoice-{tag}}}" in tex
        assert rf"\label{{tab:alphaconfirm-{tag}}}" in tex


def test_a_head_the_run_never_fitted_is_skipped_not_failed(tmp_path, capsys):
    """Asking for both heads on a run that holds one is a fact about the run."""
    rc, out = run(sweep(tmp_path, combos=("cls+mol",)))
    assert rc == 0
    assert "not in this run" in capsys.readouterr().out
    assert not (out / "alpha_choice_rank_clsprotmol.csv").exists()


def test_a_head_without_validation_fails_even_if_the_other_rendered(tmp_path):
    """One head's table must not cover for the other head's missing validation."""
    root = sweep(tmp_path, combos=("cls+mol",))
    # the second head exists on TEST only: that is the refusal case, not the absent one
    for f in sorted(root.glob("metrics_*.csv")):
        df = pd.read_csv(f)
        extra = df[df["combo"] == "cls+mol"].assign(combo="cls+prot+mol")
        pd.concat([df, extra]).to_csv(f, index=False)
    assert run(root)[0] == 1


# ----------------------------------------------------------------- the dial trend

def test_a_flat_dial_is_not_a_trend(tmp_path):
    """Every alpha scores the same: there is nothing to correlate, and the script
    must say so rather than reporting a spurious rho."""
    _, out = run(sweep(tmp_path, flat=True))
    assert not (out / "alpha_trend_clsmol.csv").exists()


def test_a_rising_dial_is_found_and_tested(tmp_path, capsys):
    """A straight line in alpha on every panel: rho = +1 in each cell, and the mean
    must be reported as significantly non-zero."""
    _, out = run(sweep(tmp_path, panels=("m2or", "cc", "hc"), slope=0.05))
    tr = pd.read_csv(out / "alpha_trend_clsmol.csv")
    assert len(tr) == 3
    assert tr["rho"].tolist() == pytest.approx([1.0] * 3)
    printed = capsys.readouterr().out
    assert "DIAL TREND" in printed
    assert "rises with alpha" in printed


def test_a_falling_dial_comes_back_negative(tmp_path, capsys):
    _, out = run(sweep(tmp_path, panels=("m2or", "cc", "hc"), slope=-0.05))
    tr = pd.read_csv(out / "alpha_trend_clsmol.csv")
    assert tr["rho"].tolist() == pytest.approx([-1.0] * 3)
    assert "falls with alpha" in capsys.readouterr().out


def test_the_trend_is_summarised_over_cells_not_over_alphas(tmp_path):
    """The unit of evidence is the cell. A pooled correlation over every (cell, alpha)
    point would treat scores ranked against each other inside a cell as independent
    and hand back an interval far too narrow."""
    root = sweep(tmp_path, panels=("m2or", "cc", "hc"), slope=0.05)
    per_cell, tr = trend_of(root)
    assert tr["cells"] == len(per_cell) == 3
    assert per_cell["n_alphas"].eq(len(ALPHAS)).all()


def test_the_trend_interval_covers_zero_when_the_cells_disagree(tmp_path):
    """Two panels rising, one falling by the same amount: the mean correlation is
    nothing, and the verdict must say nothing rather than pick a side."""
    root = sweep(tmp_path, panels=("m2or", "cc"), slope=0.05)
    sweep(root, panels=("hc",), slope=-0.05)
    _, tr = trend_of(root)
    assert tr["lo"] <= 0 <= tr["hi"]
    assert tr["verdict"] == "indistinguishable from zero"
    assert (tr["pos"], tr["neg"]) == (2, 1)


def test_a_missing_number_prints_as_a_dash_not_as_nan(tmp_path):
    """Half a grid is what this looks like mid-run, and `nan` in a typeset table is
    worse than an explicit gap."""
    row = pd.Series(dict(value=np.nan, base=np.nan, d=np.nan, won=np.nan, n=np.nan,
                         rank=np.nan, n_alphas=3, optimism=np.nan))
    assert T._cells(row) == ("--", "--", "--", "--", "--")
