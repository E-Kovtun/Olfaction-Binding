"""Placement and the live block -- the two things a twelve-worker sweep is read by.

No GPU is needed: `plan_placement` takes the free-memory vector as an argument, and the
dashboard writes into a StringIO with `tty` forced either way. What is pinned here is
the arithmetic (do the counts add up, does a busy card get fewer workers) and the two
rendering contracts (a terminal gets a block that rewrites itself; a pipe gets plain
lines and never an escape code).
"""
import io
import re

import pytest

from orbind.gpu_dashboard import Dashboard, fmt_time, gpu_stats, plan_placement

ESC = re.compile(r"\033\[")


# ------------------------------------------------------------------- placement

def test_equally_free_cards_get_an_equal_share():
    got = plan_placement([0, 1, 2, 3], 12, free_mb=[40000] * 4)
    assert len(got) == 12
    assert sorted(got.count(g) for g in (0, 1, 2, 3)) == [3, 3, 3, 3]


def test_a_busy_card_gets_fewer_workers():
    """The whole reason placement is not plain round-robin: a card someone else is
    already using should not be handed the same load as an idle one, and nothing has
    to tell the sweep about it."""
    free = [40000, 40000, 40000, 4000]        # gpu 3 is 90% occupied
    got = plan_placement([0, 1, 2, 3], 13, free_mb=free)
    assert len(got) == 13
    assert got.count(3) < got.count(0)
    assert got.count(3) >= 1                  # but not starved to nothing


def test_the_counts_always_sum_exactly():
    """Largest-remainder, so no worker is lost or invented at any awkward ratio."""
    for n in range(1, 40):
        for free in ([40000] * 3, [1000, 39000], [7000, 11000, 23000, 5000]):
            got = plan_placement(list(range(len(free))), n, free_mb=free)
            assert len(got) == n, (n, free)
            assert set(got) <= set(range(len(free)))


def test_the_first_workers_are_spread_over_every_card():
    """Workers start in order and the queue is full at the beginning, so a blocked
    layout (0,0,0,1,1,1,...) would leave later cards idle during the slowest part of
    the run. Interleaved, every card is busy from the first moment."""
    got = plan_placement([0, 1, 2, 3], 8, free_mb=[40000] * 4)
    assert set(got[:4]) == {0, 1, 2, 3}


def test_no_gpus_is_not_a_crash():
    assert plan_placement([], 4) == []


def test_gpu_stats_is_empty_without_nvidia_smi():
    """A CPU box is a normal place to run this, and it must not raise."""
    assert isinstance(gpu_stats(), dict)


# ------------------------------------------------------------------- rendering

def _dash(total=10, tty=True, **kw):
    return Dashboard(total, [0, 1], stream=io.StringIO(), refresh=0.0, tty=tty, **kw)


def test_a_terminal_gets_a_block_that_rewrites_itself():
    d = _dash()
    d.bind(0, 0)
    d.bind(1, 1)
    d.start_job(0, "a0.50 f1")
    first = d.stream.getvalue()
    assert "a0.50 f1" in first
    # clear-line codes are expected in every frame; what the FIRST frame must not
    # do is move the cursor up, because there is nothing above it yet to overwrite
    assert not re.match(r"\[\d+A", first), "the first frame moved the cursor up"
    d.done = 5
    d.render(force=True)
    after = d.stream.getvalue()[len(first):]
    assert after.startswith("\033["), "a later frame must move the cursor up first"
    assert "5/10" in after and "50.0%" in after


def test_a_pipe_gets_plain_lines_and_no_escape_codes():
    """Under nohup or in CI the same escape codes would fill the log with garbage."""
    d = _dash(tty=False, quiet_every=0.0)
    d.bind(0, 0)
    d.start_job(0, "a0.50 f1")
    d.done = 3
    d.finish_job(0, "a0.50 f1", "R2=+0.671")
    out = d.stream.getvalue()
    assert not ESC.search(out)
    assert "R2=+0.671" in out


def test_finishing_clears_the_worker_from_the_running_list():
    d = _dash()
    d.bind(0, 0)
    d.start_job(0, "a0.50 f1")
    assert 0 in d.running
    d.finish_job(0, "a0.50 f1", "R2=+0.671")
    assert 0 not in d.running
    assert d.done == 1


def test_notes_survive_redraws_and_are_capped():
    """Failures must not scroll away behind the block that is rewriting itself, but
    they also must not grow it without bound."""
    d = _dash()
    for i in range(9):
        d.note(f"cell {i} failed")
    assert len(d.notes) == 4
    assert "cell 8 failed" in d.stream.getvalue()


def test_progress_reports_a_rate_and_an_eta():
    d = _dash(total=100)
    d.t0 -= 60.0                       # pretend a minute has passed
    d.done = 20
    d.render(force=True)
    out = d.stream.getvalue()
    assert "20/100" in out and "20.0%" in out
    assert "cells/min" in out
    assert "ETA" in out


@pytest.mark.parametrize("sec,want", [(0, "00m00s"), (59, "00m59s"),
                                      (60, "01m00s"), (3661, "1h01m01s"),
                                      (None, "--")])
def test_fmt_time(sec, want):
    assert fmt_time(sec) == want
