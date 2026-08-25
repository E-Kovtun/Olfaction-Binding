"""The `--combos` mini-language.

Digits are positions into the `--source` flags of one command line, so the same
spec means different things under different source orders. That has already
produced one wrong comparison; the point of these tests is that the behaviour is
pinned and visible rather than surprising.
"""
from __future__ import annotations

import pytest

from orbind.ensemble import parse_combo_spec

NAMES = ["cls", "prot", "mol"]


def test_digits_are_one_based_positions():
    assert parse_combo_spec("1 23 123", NAMES) == [
        ("cls",), ("prot", "mol"), ("cls", "prot", "mol")]


def test_the_same_token_means_different_things_under_different_source_orders():
    """The trap, stated as a test: `13` is cls+mol or cls+prot depending only on
    the order the --source flags were given in. Compare combo NAMES, never digits."""
    assert parse_combo_spec("13", ["cls", "prot", "mol"]) == [("cls", "mol")]
    assert parse_combo_spec("13", ["cls", "mol", "prot"]) == [("cls", "prot")]


def test_combo_order_within_a_token_follows_the_digits():
    assert parse_combo_spec("31", NAMES) == [("mol", "cls")]


def test_whitespace_between_tokens_is_free_form():
    assert parse_combo_spec("  1\t23   ", NAMES) == [("cls",), ("prot", "mol")]


@pytest.mark.parametrize("spec, match", [
    ("1 x", "digit string"),
    ("14", "out-of-range"),
    ("0", "out-of-range"),
    ("11", "repeats a source index"),
    ("1 1", "duplicate combos"),
    ("", "zero combos"),
    ("   ", "zero combos"),
])
def test_malformed_specs_are_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_combo_spec(spec, NAMES)
