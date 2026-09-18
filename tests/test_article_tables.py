"""The article-table layer: reductions, tests, row selection, and the scripts end to end.

Built on two tiny synthetic trees shaped like the real ones -- a v9 sweep directory and an
ensemble_logs pool -- so nothing here needs a GPU or the server.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "article_tables"))
import tablekit as tk  # noqa: E402

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 43]


def _script(name):
    spec = importlib.util.spec_from_file_location(
        f"_article_{name}", ROOT / "scripts" / "article_tables" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sweep(root, ds="cc", fam="rand", mol="chemberta", graph=0.70, boost=0.66):
    rng = np.random.default_rng(0)
    rows = []
    for f in FOLDS:
        hard = 0.02 * f            # folds differ in difficulty; places must survive it
        for s in SEEDS:
            def reg(v):
                return dict(R2=v, RMSE=1 - v, MAE=0.8 - v / 2, Pearson=v + 0.1,
                            Spearman=v + 0.1)
            rows.append(dict(arm="boost_full", alpha=np.nan, fold=f, seed=s, status="ok",
                             combo="prot+mol", **reg(boost - hard + rng.normal(0, 1e-3))))
            rows.append(dict(arm="naive", alpha=np.nan, fold=f, seed=s, status="ok",
                             combo="const", **reg(0.0)))
            for i, c in enumerate(("cls+mol", "cls+prot+mol")):
                for alpha in (0.0, 1.0):
                    v = graph - hard - 0.1 * (1 - alpha) + 0.005 * i + rng.normal(0, 1e-3)
                    rows.append(dict(arm="gate", alpha=alpha, fold=f, seed=s, status="ok",
                                     combo=c, rsa_fun=0.2 + 0.2 * alpha, rsa_fun_z=6.0,
                                     cca_fun=0.5, cca_fun_z=3.0, procrustes_fun=0.2,
                                     procrustes_fun_z=1.0, **reg(v)))
    pd.DataFrame(rows).to_csv(root / f"metrics_{ds}_{fam}_{mol}_nodedial.csv", index=False)


def _run(eroot, name, source, mol_file, level, combos=("cls", "cls+prot+mol")):
    d = eroot / "cc-rand-molcross-fixed" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(dict(
        regime="ofm", dataset="cc", split_family="rand", task="regression",
        tune_boost=False,
        sources=[f"cls={source}", f"mol=gin:data/embeddings/molecules/{mol_file}"])))
    pd.DataFrame([dict(kind="combo", name=c, repeat=f, R2=level - 0.02 * f, RMSE=1 - level,
                       MAE=0.5, Pearson=0.8, Spearman=0.8)
                  for c in combos for f in FOLDS]).to_csv(d / "metrics.csv", index=False)


@pytest.fixture
def trees(tmp_path):
    tk.clear_cache()
    s, e = tmp_path / "sweep", tmp_path / "ens"
    s.mkdir()
    e.mkdir()
    _sweep(s)
    _run(e, "cc_rand_hladis_concatCB", "hladis", "chemberta_77m_cc.npz", 0.64)
    _run(e, "cc_rand_hladis_concatECFP", "hladis", "ecfp_cc.npz", 0.61)
    yield s, e, tmp_path
    tk.clear_cache()


# ----------------------------------------------------------------------- arithmetic

def test_holm_by_hand():
    got = tk.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": np.nan})
    assert got["a"] == pytest.approx(0.03)
    assert got["c"] == pytest.approx(0.06)
    assert got["b"] == pytest.approx(0.06)        # monotone: never below the one before
    assert np.isnan(got["d"])


def test_paired_test_counts_leads_the_right_way_for_an_error_metric():
    x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=FOLDS)
    y = x + np.array([0.05, 0.06, 0.04, 0.05, 0.07])
    r2 = tk.paired_test(y, x, "R2")
    rmse = tk.paired_test(y, x, "RMSE")
    assert r2["ahead"] == 5 and rmse["behind"] == 5
    assert r2["p"] < 0.01 and r2["delta"] == pytest.approx(0.054)


def test_ranks_inside_a_split_survive_fold_difficulty():
    M = pd.DataFrame({"a": [0.9, 0.5, 0.1], "b": [0.8, 0.4, 0.0]}, index=[1, 2, 3])
    assert tk.split_ranks(M, "R2").mean().to_dict() == {"a": 1.0, "b": 2.0}
    assert tk.split_ranks(M, "RMSE").mean().to_dict() == {"a": 2.0, "b": 1.0}


# --------------------------------------------------------------------- row selection

def test_sweep_rows_are_one_value_per_split(trees):
    s, _, _ = trees
    r = tk.ours_row("cc", "transductive", "chemberta", ["R2"], "cls+mol", 1.0, root=s)
    assert r.splits() == FOLDS and r.seeds == 2
    other = tk.ours_row("cc", "transductive", "chemberta", ["R2"], "cls+prot+mol", 1.0,
                        root=s)
    assert (other.values("R2") - r.values("R2")).mean() == pytest.approx(0.005, abs=2e-3)
    assert not tk.ours_row("cc", "inductive", "chemberta", ["R2"], root=s).present


def test_a_baseline_is_matched_on_the_molecule_file(trees):
    _, e, _ = trees
    cb = tk.baseline_row("hladis", "cc", "transductive", "chemberta", ["R2"], root=e)
    ec = tk.baseline_row("hladis", "cc", "transductive", "ecfp", ["R2"], root=e)
    gin = tk.baseline_row("hladis", "cc", "transductive", "gin", ["R2"], root=e)
    assert cb.combo == "cls+prot+mol" and cb.usable
    assert cb.values("R2").iloc[0] == pytest.approx(0.62)
    assert ec.values("R2").iloc[0] == pytest.approx(0.59)
    assert not gin.usable and any("molecules" in f for f in gin.flags)


def test_a_missing_combo_falls_back_and_says_so(trees):
    _, e, _ = trees
    r = tk.baseline_row("hladis", "cc", "transductive", "chemberta", ["R2"],
                        combo="cls+mol", root=e)
    assert r.present and r.combo != "cls+mol" and any("asked for" in f for f in r.flags)


# --------------------------------------------------------------------------- scripts

def test_main_table_end_to_end(trees):
    s, e, tmp = trees
    m = _script("01_main_tables")
    out = tmp / "main"
    longs = m.main(["--dataset", "cc", "--sweep-root", str(s), "--ensemble-root", str(e),
                    "--out", str(out)])
    st = pd.concat(longs)
    r2 = st[st.metric == "R2"].set_index("key")
    assert r2.loc["ours:cls+prot+mol", "rank"] == 1.0
    # only OUR rows are tested, and the opponent is the best row that is not ours --
    # here the boosting base, which leads Hladis in this fixture
    assert (r2.loc["ours:cls+mol", "ref"] == "boost"
            and r2.loc["ours:cls+prot+mol", "ref"] == "boost")
    assert r2.loc["ours:cls+mol", "p_holm"] < 0.05   # 0.04 ahead on every split
    assert np.isnan(r2.loc["boost", "p_vs_ref"]) and np.isnan(r2.loc["hladis", "p_vs_ref"])
    assert np.isnan(r2.loc["lorax", "mean"])         # not run -> a blank row, not a gap
    tex = (out / "cc.tex").read_text()
    assert r"\cbest{" in tex and "LORAX" in tex and "--" in tex
    assert r"\label{tab:main_cc}" in tex and "Friedman" in tex
    # the p column exists once per metric in the HEADER (the caption also says "$p$"),
    # and the Friedman line sits under the table
    header = next(l for l in tex.splitlines() if l.startswith(r"\textbf{Method"))
    assert header.count("$p$") == 4 and r"\multicolumn{10}{@{}l}" in tex
    assert "not available" in (out / "cc_summary.txt").read_text()


def test_molecule_ablation_end_to_end(trees):
    s, e, tmp = trees
    m = _script("03_molecule_ablation")
    out = tmp / "mol"
    m.main(["--dataset", "cc", "--sweep-root", str(s), "--ensemble-root", str(e),
            "--out", str(out)])
    tex = (out / "cc.tex").read_text()
    assert "ChemBERTa &" in tex and "ECFP &" in tex and r"\textit{Mean rank}" in tex
    long = pd.read_csv(out / "molecule_long.csv")
    ecfp = long[(long.mol_source == "ecfp") & (long.key == "hladis")]
    assert ecfp["mean"].iloc[0] == pytest.approx(0.55)     # Hladis alone on ECFP


def test_geometry_table_reads_the_sweep_and_the_frozen_embeddings(trees):
    s, _, tmp = trees
    pg = tmp / "pg"
    pg.mkdir()
    pd.DataFrame([dict(dataset="cc", regime="transductive", fold=f, embedding=e,
                       rsa_fun=v + 0.01 * f, rsa_fun_z=z, cca_fun=0.3, cca_fun_z=2.5,
                       procrustes_fun=0.1, procrustes_fun_z=0.5)
                  for f in FOLDS for e, v, z in (("esm1b", 0.1, 3.0), ("onehot", np.nan, np.nan))]
                 ).to_csv(pg / "cc_transductive.csv", index=False)
    m = _script("02_geometry_table")
    out = tmp / "geo"
    cells = m.main(["--dataset", "cc", "--sweep-root", str(s), "--protein-geometry", str(pg),
                    "--out", str(out)])
    st = cells["cc"][1]
    rsa = st[st.metric == "rsa_fun"].set_index("key")
    assert rsa.loc["ours:a=1", "mean"] == pytest.approx(0.4)
    assert rsa.loc["esm1b", "p_holm"] < 0.05          # 0.27-0.29 behind on every split
    tex = (out / "geometry_transductive.tex").read_text()
    assert "ESM-1b" in tex and r"$^{\circ}$" in tex         # procrustes z below 1.96


def test_no_tests_drops_the_significance_marks_but_not_the_null_marks(trees):
    """Two different statements share a superscript slot. `--no-tests` is about the
    paired comparison AGAINST OUR GRAPH; the permutation null says whether a row is
    distinguishable from chance at all, which stays either way. Dropping both would
    quietly turn an unaligned row into one the reader takes at face value."""
    s, _, tmp = trees
    pg = tmp / "pg2"
    pg.mkdir()
    pd.DataFrame([dict(dataset="cc", regime="transductive", fold=f, embedding=e,
                       rsa_fun=v + 0.01 * f, rsa_fun_z=z, cca_fun=0.3, cca_fun_z=2.5,
                       procrustes_fun=0.1, procrustes_fun_z=0.5)
                  for f in FOLDS for e, v, z in (("esm1b", 0.1, 3.0),)]
                 ).to_csv(pg / "cc_transductive.csv", index=False)
    m = _script("02_geometry_table")
    args = ["--dataset", "cc", "--sweep-root", str(s), "--protein-geometry", str(pg)]

    on = tmp / "geo_on"
    m.main(args + ["--out", str(on)])
    tex_on = (on / "geometry_transductive.tex").read_text()
    assert r"$^{*}$" in tex_on and "Holm-corrected" in tex_on

    off = tmp / "geo_off"
    m.main(args + ["--out", str(off), "--no-tests"])
    tex_off = (off / "geometry_transductive.tex").read_text()
    assert r"$^{*}$" not in tex_off, "the significance mark survived --no-tests"
    assert "Holm-corrected" not in tex_off, "the caption still explains a mark that is gone"
    assert r"$^{\circ}$" in tex_off, "the permutation-null mark is a different statement"
    assert "ESM-1b" in tex_off and "0.1" in tex_off


# ------------------------------------------------------- the decision threshold

def _bin(rng, n=60, prevalence=0.25, lo=0.10, hi=0.40):
    """Scores that are perfectly separable -- but ONLY below 0.5, which is where every
    imbalanced pool with a probability head lands. At the fixed cut both MCC and F1 are
    0; at a cut chosen on validation they are 1. The gap is the whole point of the
    machinery under test."""
    y = (rng.random(n) < prevalence).astype(np.float32)
    p = np.where(y > 0, rng.normal(hi, 0.01, n), rng.normal(lo, 0.01, n))
    return y, np.clip(p, 0.0, 1.0).astype(np.float32)


def _m2or_sweep(root, rng, with_scores=True):
    stem = "m2or_transductive_q99greedy_chemberta_nodedial"
    rows = []
    for f in FOLDS:
        for s in SEEDS:
            def cell(auroc):
                # MCC/F1 are stored as the 0.5 cut sees them: nothing above it
                return dict(AUROC=auroc, AUPRC=auroc - 0.1, MCC=0.0, F1=0.0)
            rows.append(dict(arm="boost_full", alpha=np.nan, fold=f, seed=s, status="ok",
                             combo="prot+mol", **cell(0.82)))
            for c in ("cls+mol", "cls+prot+mol"):
                rows.append(dict(arm="gate", alpha=1.0, fold=f, seed=s, status="ok",
                                 combo=c, **cell(0.84)))
            if not with_scores:
                continue
            d = root / "dumps" / stem
            d.mkdir(parents=True, exist_ok=True)
            yv, pv = _bin(rng)
            yt, pt = _bin(rng)
            np.savez_compressed(d / f"gate_a1_f{f}_s{s}.npz", pred=pt, pred_prot=pt,
                                pred_va=pv, pred_va_prot=pv, y_val=yv, y_true=yt)
            yv, pv = _bin(rng)
            yt, pt = _bin(rng)
            np.savez_compressed(d / f"boost_full_aNone_f{f}_s{s}.npz", pred=pt,
                                pred_va=pv, y_val=yv, y_true=yt)
    pd.DataFrame(rows).to_csv(root / f"metrics_{stem}.csv", index=False)


def _m2or_baseline(eroot, rng, with_scores=True):
    d = eroot / "m2or-transductive-chemberta-fixed" / "transductive_hladis_esm1b"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(dict(
        regime="full_full", full_full_mode="transductive", task="classification",
        tune_boost=False,
        sources=["cls=hladis", "mol=gin:data/embeddings/molecules/chemberta_77m_m2or.npz"])))
    pd.DataFrame([dict(kind="combo", name="cls+prot+mol", repeat=f, AUROC=0.83,
                       AUPRC=0.73, MCC=0.0, F1=0.0) for f in FOLDS]
                 ).to_csv(d / "metrics.csv", index=False)
    if with_scores:
        (d / "scores").mkdir()
        for f in FOLDS:
            yv, pv = _bin(rng)
            yt, pt = _bin(rng)
            np.savez_compressed(d / "scores" / f"repeat_{f}.npz",
                                **{"val__cls+prot+mol": pv, "test__cls+prot+mol": pt,
                                   "y_val": yv, "y_test": yt})
    return d


@pytest.fixture
def m2or(tmp_path):
    tk.clear_cache()
    s, e = tmp_path / "sweep", tmp_path / "ens"
    s.mkdir()
    e.mkdir()
    yield s, e, tmp_path
    tk.clear_cache()


def test_the_val_cut_is_an_extra_column_and_never_replaces_the_0_5_one(m2or):
    """Every published baseline thresholds at 0.5, so that column has to survive intact;
    the val-chosen cut is shown beside it, not instead of it."""
    s, e, tmp = m2or
    rng = np.random.default_rng(0)
    _m2or_sweep(s, rng)
    _m2or_baseline(e, rng)
    m = _script("01_main_tables")
    out = tmp / "main"
    longs = m.main(["--dataset", "m2or", "--baselines", "hladis", "--sweep-root", str(s),
                    "--ensemble-root", str(e), "--out", str(out)])
    st = pd.concat(longs)
    assert set(st.cut) == {"val"}
    # the 0.5 columns are untouched -- at that cut this fixture separates nothing
    assert (st[st.metric == "MCC"]["mean"].dropna() == 0.0).all()
    assert (st[st.metric == "F1"]["mean"].dropna() == 0.0).all()
    # and the added ones say what the operating point is worth
    assert (st[st.metric == "MCC@val"]["mean"].dropna() > 0.9).all()
    assert (st[st.metric == "F1@val"]["mean"].dropna() > 0.9).all()
    # AUROC is threshold-free and must come through untouched
    assert st[(st.metric == "AUROC") & (st.key == "boost")]["mean"].iloc[0] == pytest.approx(0.82)
    # the rank column must not count MCC twice
    ours = st[(st.key == "ours:cls+mol") & (st.regime == "transductive")]
    assert set(ours.metric) == {"AUROC", "AUPRC", "MCC", "MCC@val", "F1", "F1@val"}
    tex = (out / "m2or.tex").read_text()
    assert r"$^{\mathrm{val}}$" in tex and "fixed 0.5 cut" in tex


def test_one_row_without_scores_leaves_every_row_on_the_0_5_cut_alone(m2or):
    s, e, tmp = m2or
    rng = np.random.default_rng(1)
    _m2or_sweep(s, rng)
    _m2or_baseline(e, rng, with_scores=False)      # the borrowed row has none
    m = _script("01_main_tables")
    out = tmp / "main"
    longs = m.main(["--dataset", "m2or", "--baselines", "hladis", "--sweep-root", str(s),
                    "--ensemble-root", str(e), "--out", str(out)])
    st = pd.concat(longs)
    assert set(st.cut) == {"fixed"}
    assert not any(str(m).endswith("@val") for m in st.metric), \
        "our rows must not get a cut the baseline beside them could not have"
    assert (st[st.metric == "MCC"]["mean"].dropna() == 0.0).all()
    tex = (out / "m2or.tex").read_text()
    assert "not shown" in tex and r"$^{\mathrm{val}}$" not in tex


def test_the_ensemble_writes_per_row_scores_next_to_its_metrics(tmp_path):
    """The producer half: `metrics.csv` holds aggregates, and an aggregate cannot be
    re-thresholded -- so the run has to leave the scores themselves behind."""
    from orbind.ensemble import _save_scores
    _save_scores(tmp_path / "scores", "3", {("cls", "mol"): np.array([0.1, 0.9])},
                 {("cls", "mol"): np.array([0.2, 0.8])},
                 np.array([0.0, 1.0]), np.array([1.0, 0.0]),
                 np.array([7, 8]), np.array([9, 10]))
    f = tmp_path / "scores" / "repeat_3.npz"
    with np.load(f) as z:
        assert set(z.files) == {"y_val", "y_test", "val_idx", "test_idx",
                                "val__cls+mol", "test__cls+mol"}
        assert z["val__cls+mol"].tolist() == pytest.approx([0.1, 0.9])
        assert z["test_idx"].tolist() == [9, 10]
    assert not list(f.parent.glob("*.tmp.npz"))


def test_inventory_runs_on_a_partial_tree(trees):
    s, e, tmp = trees
    m = _script("00_inventory")
    df = m.main(["--only", "main", "molecule", "architecture", "--sweep-root", str(s),
                 "--ensemble-root", str(e), "--expect-seeds", "2", "--out", str(tmp / "inv")])
    q = df[(df.table == "main") & (df.dataset == "cc") & (df.regime == "transductive")]
    st = dict(zip(q["item"], q.status))
    assert st["Our graph (cls+mol)"] == "READY"
    assert st["LORAX"] == "MISSING" and st["Hladis"] == "READY"
    assert (df[df.table == "architecture"].status == "MISSING").all()


def test_a_sweep_whose_files_cannot_be_parsed_says_so_on_the_row(tmp_path):
    """The failure that cost a whole debugging round. Every metrics file under the root
    was dropped because its name would not parse, `alpha_grid.load` gave up, tablekit
    turned that into an empty frame, and the table printed `--` for every graph row
    while volunteering no reason anywhere. The reader's own message must reach the row.
    """
    tk.clear_cache()
    root = tmp_path / "v99"
    root.mkdir()
    (root / "metrics_zz_nosuchfamily_chemberta_nodedial.csv").write_text(
        "arm,alpha,fold,seed,R2\n", encoding="utf-8")
    r = tk.ours_row("cc", "transductive", "chemberta", ["R2"], root=str(root))
    assert not r.present
    why = " ".join(r.flags)
    assert "dropped" in why and "nosuchfamily" in why, why
    b = tk.boost_row("cc", "transductive", "chemberta", ["R2"], root=str(root))
    assert "dropped" in " ".join(b.flags)
    tk.clear_cache()
