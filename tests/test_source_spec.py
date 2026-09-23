"""The `--source` mini-language, and the entity extractor it builds.

A source string is a positional colon-separated field list, up to twelve fields
deep for the graph. Blank fields keep defaults, which is what makes it possible to
set field 9 without restating fields 2-8 -- and what makes an off-by-one silent.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_trainer():
    """Import the trainer as a module without running it."""
    path = ROOT / "scripts" / "modeling" / "train" / "train_ensemble_boost.py"
    spec = importlib.util.spec_from_file_location("train_ensemble_boost", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()
parse_source_arg = trainer.parse_source_arg


# ------------------------------------------------------------------ the syntax

def test_a_source_needs_a_name():
    with pytest.raises(argparse.ArgumentTypeError, match="name=type"):
        parse_source_arg("esm:some/path.npz")


def test_unknown_type_is_rejected_by_name():
    with pytest.raises(argparse.ArgumentTypeError, match="unknown source type"):
        parse_source_arg("cls=transformer")


def test_entity_source_needs_a_path():
    with pytest.raises(argparse.ArgumentTypeError, match="name=type:path"):
        parse_source_arg("prot=esm")


# ------------------------------------------------------------ graph field order

def test_graph_defaults_are_the_headline_configuration():
    name, ex = parse_source_arg("cls=gnn_signed")
    assert name == "cls"
    assert (ex.q, ex.criterion, ex.emit, ex.n_models, ex.k_mode) == (
        0.99, "greedy_pair_cover", "prot", 1, "coverage_quantile")


def test_blank_fields_keep_defaults_without_shifting_the_rest():
    """`:::3` sets field 4 (n_models) and leaves fields 2-3 alone."""
    _, ex = parse_source_arg("cls=gnn_signed:::3")
    assert ex.n_models == 3
    assert ex.q == 0.99
    assert ex.criterion == "greedy_pair_cover"


def test_every_graph_field_lands_where_the_docstring_says(workdir):
    _write_npz("p.npz")
    _write_npz("m.npz")
    _, ex = parse_source_arg(
        "cls=gnn_signed:p.npz:m.npz:2:both:0.5:coverage:0.25:fraction")
    assert ex.protein_path == "p.npz"
    assert ex.molecule_path == "m.npz"
    assert ex.n_models == 2
    assert ex.emit == "both"
    assert ex.q == 0.5
    assert ex.criterion == "coverage"
    assert ex.edge_threshold == 0.25
    assert ex.k_mode == "fraction"


def test_alpha_is_field_12_and_off_by_default():
    """The v8 gate must stay invisible unless asked for: every pre-v8 command has
    to build the historical model, so `alpha` defaults to None, not to 1.0."""
    assert parse_source_arg("cls=gnn_signed")[1].alpha is None
    _, ex = parse_source_arg("cls=gnn_signed::::::::::::0.25")
    assert ex.alpha == 0.25
    assert (ex.q, ex.criterion, ex.n_models) == (0.99, "greedy_pair_cover", 1)


def test_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        parse_source_arg("cls=gnn_signed::::::::::::1.5")


def test_the_graphsage_regime_is_fields_16_and_17_and_on_by_default():
    """DEFAULT MOVED 23.09.2026: a bare `gnn_signed` now trains in GraphSAGE's own
    regime, because that is the model the paper reports. Fields 16-17 exist to say it
    explicitly and to take it back."""
    ex = parse_source_arg("cls=gnn_signed")[1]
    assert ex.fanout == (25, 10) and ex.normalize_layers is True
    _, ex = parse_source_arg("cls=gnn_signed::::::::::::::::25-10:1")
    assert ex.fanout == (25, 10)
    assert ex.normalize_layers is True
    assert (ex.q, ex.criterion, ex.n_models) == (0.99, "greedy_pair_cover", 1)


def test_the_historical_encoder_is_still_askable_from_a_spec():
    """Reproducing a pre-23.09.2026 table needs the old encoder, and a spec that could
    only turn the regime ON would make those numbers unreachable."""
    _, ex = parse_source_arg("cls=gnn_signed::::::::::::::::0:0")
    assert ex.fanout == () and ex.normalize_layers is False
    _, ex = parse_source_arg("cls=gnn_signed::::::::::::::::none:0")
    assert ex.fanout == () and ex.normalize_layers is False


def test_the_fanout_accepts_commas_too():
    """A dash is the documented separator because a colon splits the spec; a comma is
    what a hand reaches for anyway, and silently meaning something else would be worse
    than accepting it."""
    assert parse_source_arg("cls=gnn_signed::::::::::::::::25,10")[1].fanout == (25, 10)


def test_a_one_layer_fanout_is_refused_by_the_extractor():
    """The encoder has two layers; one number would silently mean 'both'."""
    with pytest.raises(ValueError, match="two entries"):
        parse_source_arg("cls=gnn_signed::::::::::::::::25")


def test_the_name_is_the_extractors_name():
    _, ex = parse_source_arg("refined=gnn_signed")
    assert ex.name == "refined"


# ---------------------------------------------------------- the entity lookup

def _write_npz(name):
    np.savez(name, MKV=np.arange(4, dtype=np.float32),
             MKW=np.arange(4, dtype=np.float32) + 10)
    return name


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A scratch cwd.

    Paths in a --source string must be RELATIVE: the field separator is `:`, so a
    Windows absolute path (`C:/...`) splits into two fields. Every path in this
    repository is repo-relative anyway, which is why this has never bitten a real
    run -- but it is a real constraint of the syntax.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def npz(workdir):
    return _write_npz("prot.npz")


def test_entity_source_reads_dim_and_metadata(npz):
    _, ex = parse_source_arg(f"prot=esm:{npz}:esm1b_t33_650M_UR50S:mean")
    assert ex.dim == 4
    assert ex.key_col == "receptor"
    assert ex.model_name == "esm1b_t33_650M_UR50S"
    assert ex.pooling == "mean"


def test_molecule_sources_are_keyed_by_inchikey(npz):
    _, ex = parse_source_arg(f"mol=gin:{npz}")
    assert ex.key_col == "inchikey"


def test_covered_flags_exactly_the_rows_with_an_embedding(npz):
    _, ex = parse_source_arg(f"prot=esm:{npz}")
    pairs = pd.DataFrame({"receptor": ["MKV", "MKW", "MISSING", "MKV"],
                          "inchikey": list("abcd"),
                          "label": [1, 0, 1, 0]})
    idx = np.arange(len(pairs))
    assert list(ex.covered(pairs, idx)) == [True, True, False, True]


def test_lookup_raises_on_an_uncovered_row_and_names_it(npz):
    """`--on-missing raise` leans on this being loud and precise."""
    _, ex = parse_source_arg(f"prot=esm:{npz}")
    pairs = pd.DataFrame({"receptor": ["MKV", "MISSING"],
                          "inchikey": list("ab"),
                          "label": [1, 0]})
    idx = np.arange(len(pairs))
    with pytest.raises(KeyError, match="MISSING"):
        ex.fit_transform(pairs, idx, idx, idx, seed=0)


def test_lookup_is_row_aligned_and_float32(npz):
    _, ex = parse_source_arg(f"prot=esm:{npz}")
    pairs = pd.DataFrame({"receptor": ["MKW", "MKV", "MKW"],
                          "inchikey": list("abc"),
                          "label": [1, 0, 1]})
    idx = np.arange(len(pairs))
    Xtr, Xva, Xte = ex.fit_transform(pairs, idx, idx[:2], idx[2:], seed=0)
    assert Xtr.dtype == np.float32
    assert Xtr.shape == (3, 4)
    assert Xva.shape == (2, 4) and Xte.shape == (1, 4)
    assert np.allclose(Xtr[0], np.arange(4) + 10)   # MKW
    assert np.allclose(Xtr[1], np.arange(4))        # MKV
