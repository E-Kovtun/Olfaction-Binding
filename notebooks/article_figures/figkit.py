"""House style and plot primitives for the article's alpha-dial figures.

The notebooks in this folder draw; they do not aggregate. Every number they show
comes from `scripts/analysis/alpha_grid.py`, which averages MODEL SEEDS inside each
fold and only then takes an interval over folds (`alpha_grid.fold_means`). That
distinction is the whole reason this module exists as a module: a figure that
re-derived its own means would quietly make the 25 (fold, seed) cells of a
five-seed grid look like 25 observations, and halve every interval on the page.

What this module owns:

* the visual contract -- rcParams, the palette, panel grids, bands, legends;
* the axis conventions the dial needs (alpha ticks, one y-scale per panel row);
* saving, so a figure lands in one place with one name.

What it deliberately does not own: which series to plot, which metric, which root.
Those are the notebook's knobs, and they belong where a reader can see them.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

REPO = _root

# --------------------------------------------------------------------------- palette
#: Reused verbatim from notebooks/graph/alpha_gate, where it was checked pairwise for
#: colour-vision deficiency against a white ground (worst all-pairs dE 9.3 deutan,
#: 17.6 normal). Kept identical on purpose: two figures of the same paper that use the
#: same hue for different things is a harder error to spot than a bad hue.
C = {
    "esm": "#1f6fb4",            # structure -- the frozen ESM cloud
    "fun": "#c9531f",            # function -- the response profile
    "gate": "#1f6fb4",           # the dial's own curve where only one is drawn
    "graph_legacy": "#2a8a5f",
    "boost_full": "#6b7280",
    "naive": "#b9bec6",
}
#: Reference arms are dashed, so a line is identifiable without colour -- the figures
#: are read in print and in greyscale at least as often as on screen.
DASH = {"boost_full": (0, (5, 2)), "graph_legacy": (0, (4, 1.5, 1, 1.5)),
        "naive": (0, (1, 2))}
INK, MUTED, GRID = "#111827", "#6b7280", "#e6e8eb"

RC = {
    # Explicit white ground: a dark Jupyter theme otherwise paints black behind axes
    # carrying dark text, and these figures are meant to survive a copy into the paper.
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
    "figure.dpi": 110,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.edgecolor": "#9aa1a9", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.color": "#4b5563", "ytick.color": "#4b5563",
    "text.color": INK, "axes.labelcolor": "#374151",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 4.0,
}


def use_house_style():
    """Apply the style. Call once, in the notebook's import cell."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)


# --------------------------------------------------------------------------- layout
def panels(n, ncols=3, w=3.1, h=2.4, **kw):
    """A grid sized to `n`, with the empty tail switched off and axes returned flat."""
    import matplotlib.pyplot as plt
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             squeeze=False, **kw)
    ax = axes.ravel()
    for a in ax[n:]:
        a.axis("off")
    return fig, ax[:n]


def note_empty(a, text="not run yet"):
    """A panel with nothing in it, KEPT in the grid rather than removed.

    Switching the axes off closes the gap and shifts every later panel, so a
    half-finished grid reads as a complete one with different rows -- and these
    sweeps are looked at while they are still running.
    """
    a.text(0.5, 0.5, text, ha="center", va="center", transform=a.transAxes,
           fontsize=8, color=MUTED)
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)


def figlegend(fig, handles, ncol=4, pad_in=0.62):
    """One legend per figure, below it, describing the ENCODING.

    Never a box inside each panel repeating the same entries N times. It reserves a
    fixed strip (0.62 in) rather than sitting at a figure-relative y, which would
    land it on the bottom row's x labels.
    """
    fig.tight_layout()
    fig.subplots_adjust(bottom=fig.subplotpars.bottom
                        + pad_in / fig.get_size_inches()[1])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=ncol, fontsize=8.5, handlelength=2.4, columnspacing=1.8)


# --------------------------------------------------------------------------- marks
def band(ax, x, lo, hi, color, alpha=0.16):
    """The interval as a filled band -- no edge, so it never reads as a second line."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)


def alpha_axis(ax, alphas):
    """The dial's x axis: labelled quarters, a minor tick at every alpha actually run.

    The minor ticks matter. A curve drawn over 11 alphas and one drawn over 5 look
    identical at this size, and the reader has no other way to tell how much of the
    line is interpolation.
    """
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(list(alphas), minor=True)
    ax.tick_params(axis="x", which="minor", length=2.5, color="#c7ccd1")


def curve_with_band(ax, q, color, marker="o", label=None, band_alpha=0.16):
    """One aggregated series: `mean` as a line, `lo`/`hi` as its band.

    `q` is a slice of an `alpha_grid` result -- columns alpha, mean, lo, hi.
    Returns True if anything was drawn, so a caller can count empty panels.
    """
    if q is None or len(q) == 0:
        return False
    q = q.sort_values("alpha")
    band(ax, q["alpha"], q["lo"], q["hi"], color, band_alpha)
    ax.plot(q["alpha"], q["mean"], color=color, marker=marker, label=label)
    return True


def reference_line(ax, value, arm, label=None):
    """A horizontal arm (boost_full, naive, graph_legacy) the dial is read against."""
    if value is None or not np.isfinite(value):
        return False
    ax.axhline(value, color=C.get(arm, MUTED), lw=1.3,
               ls=DASH.get(arm, (0, (5, 2))), label=label)
    return True


def resolution_mark(ax, half_width, x=0.02, color=None):
    """The I-mark: how small a difference this grid can resolve at all.

    Drawn instead of per-fold whiskers on the curve. The between-fold spread is 2-28x
    wider than the model noise left in a fold mean, so whiskers over folds answer
    "how different are the folds" on a figure asking "did moving the dial change
    anything". This mark answers the second question directly: a bump smaller than it
    is noise, whatever the band says.
    """
    if half_width is None or not np.isfinite(half_width):
        return False
    color = color or MUTED
    y0, y1 = ax.get_ylim()
    mid = y0 + 0.5 * (y1 - y0)
    ax.errorbar([x], [mid], yerr=[half_width], color=color, capsize=3, lw=1.2,
                marker="", zorder=5)
    # The mark is a ruler, not data: keep the limits the curves set, or a wide mark
    # would rescale the panel and flatten the very curve it is there to qualify.
    ax.set_ylim(y0, y1)
    return True


# --------------------------------------------------------------------------- output
def savefig(fig, name, fig_dir, enabled=True):
    """Write one figure under `fig_dir`, printing where it went."""
    if not enabled:
        return None
    d = pathlib.Path(fig_dir)
    if not d.is_absolute():
        d = REPO / d
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.png"
    fig.savefig(path)
    print("wrote", path)
    return path
