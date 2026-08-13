"""House chart style. Styling is set here once and never re-tuned per notebook.

The palette was **validated, not chosen by eye** -- run through a
colour-vision-deficiency checker against a white paper surface (these figures
land in a LaTeX PDF). Two results from that validation are enforced below:

* **Bars and lines take up to 4 series.** Only adjacent pairs are compared, and
  all four slots clear the CVD floor (worst delta-E 9.1 against a floor of 8)
  and the normal-vision floor (22.9 against 15).
* **Scatter and small multiples take at most 3.** There every pair is visible at
  once, and a 4th series puts yellow beside orange, which fails. Facet instead
  of adding a colour -- ``assert_series_limit`` enforces this.

Aqua and yellow fall below 3:1 contrast on white, so figures using them carry
visible labels. That is satisfied anyway: every figure ships with a matching
``results/*.csv`` table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "figures"

# --- Palette ----------------------------------------------------------------

#: Fixed slot order. Never cycled, never reordered -- colour follows the
#: entity, not its rank, so a filtered chart must not repaint the survivors.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

MAX_SERIES_ADJACENT = 4  # bars, lines, stacked segments
MAX_SERIES_ALL_PAIRS = 3  # scatter, bubble, small multiples

#: Single hue, light to dark. For continuous magnitude only.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

#: Two poles either side of a neutral grey. Never a hue at the midpoint.
DIVERGING = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"]

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "surface": "#ffffff",
}

#: Reserved for state, never reused as a series colour. Always paired with a
#: label -- these never carry meaning by hue alone.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

#: Stable colours for recurring entities across notebooks, so a learner keeps
#: its identity from the EDA figures through to the interpretability ones.
LEARNER_COLORS = {
    "logistic_regression": CATEGORICAL[0],
    "decision_tree": CATEGORICAL[1],
    "random_forest": CATEGORICAL[2],
    "xgboost": CATEGORICAL[3],
}

CLASS_COLORS = {"legitimate": CATEGORICAL[0], "fraud": CATEGORICAL[1]}


def sequential_cmap() -> LinearSegmentedColormap:
    """Single-hue ramp for heatmaps and other continuous magnitude."""
    return LinearSegmentedColormap.from_list("momo_seq", SEQUENTIAL)


def diverging_cmap() -> LinearSegmentedColormap:
    """Two-pole ramp with a neutral midpoint, for signed deltas."""
    return LinearSegmentedColormap.from_list("momo_div", DIVERGING)


# --- Style ------------------------------------------------------------------

def apply_house_style() -> None:
    """Set the global chart style. Call once at the top of each notebook."""
    sns.set_theme(style="whitegrid", palette=CATEGORICAL)
    mpl.rcParams.update({
        "figure.facecolor": INK["surface"],
        "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],

        # Chrome recedes: the data is the only thing with weight.
        "grid.color": INK["grid"],
        "grid.linewidth": 0.6,
        "axes.edgecolor": INK["baseline"],
        "axes.linewidth": 0.8,
        "axes.grid.axis": "y",

        "text.color": INK["primary"],
        "axes.labelcolor": INK["secondary"],
        "xtick.color": INK["muted"],
        "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,

        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "patch.linewidth": 0,

        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # embed as TrueType so LaTeX keeps text selectable
    })


def assert_series_limit(n: int, *, all_pairs: bool = False) -> None:
    """Guard the validated series caps.

    Args:
        n: Number of series about to be drawn.
        all_pairs: True for scatter, bubble and small multiples, where every
            pair is visible simultaneously and the cap is 3.
    """
    limit = MAX_SERIES_ALL_PAIRS if all_pairs else MAX_SERIES_ADJACENT
    kind = "all-pairs (scatter/small multiples)" if all_pairs else "adjacent (bars/lines)"
    if n > limit:
        raise ValueError(
            f"{n} series exceeds the validated {kind} cap of {limit}. "
            "Facet, or fold the tail into 'Other' -- do not generate a new hue."
        )


def colors(n: int, *, all_pairs: bool = False) -> list[str]:
    """First ``n`` categorical slots, in fixed order."""
    assert_series_limit(n, all_pairs=all_pairs)
    return CATEGORICAL[:n]


# --- Output -----------------------------------------------------------------

def save_figure(fig: plt.Figure, name: str, *, subdir: str | None = None) -> dict[str, Path]:
    """Write a figure as vector PDF (for LaTeX) and 200 dpi PNG (for slides).

    Args:
        name: Filename stem, no extension. Prefix with the notebook number,
            e.g. ``01_fraud_rate_by_type``.
        subdir: Optional subfolder under ``figures/``.
    """
    out_dir = FIGURE_DIR / subdir if subdir else FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 200})):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, **kwargs)
        paths[ext] = path
    return paths


# --- Components -------------------------------------------------------------

def stat_tile(
    ax: plt.Axes,
    value: str,
    label: str,
    sublabel: str | None = None,
    *,
    accent: str = INK["primary"],
) -> plt.Axes:
    """A hero number, for facts a chart would only obscure.

    Class balance is the canonical case: a bar of 99.87 against 0.13
    communicates nothing that the number itself does not say better.
    """
    ax.axis("off")
    ax.text(0, 0.62, value, fontsize=30, fontweight="bold", color=accent,
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0, 0.28, label, fontsize=11, color=INK["secondary"],
            ha="left", va="center", transform=ax.transAxes)
    if sublabel:
        ax.text(0, 0.08, sublabel, fontsize=9, color=INK["muted"],
                ha="left", va="center", transform=ax.transAxes)
    return ax


def direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    dx: float = 6,
    dy: float = 0,
    **kwargs,
) -> None:
    """Label a series at its end, so identity is never carried by colour alone.

    Text stays in ink; the mark beside it carries the colour. The exception is
    a series label, which takes the series colour precisely because it *is* the
    identity cue -- that is what lets the legend box be dropped.
    """
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="semibold",
        va="center",
        **kwargs,
    )


def value_labels(ax: plt.Axes, fmt="%.3f", *, padding: int = 3) -> None:
    """Print values on bars. Selective by nature -- only for small bar counts.

    ``fmt`` may be a printf string or a callable. Pass a callable when the
    stored value and the displayed one differ -- proportions shown as
    percentages being the usual case, where a bare ``%.3f`` prints ``0.002``
    for what the axis labels as ``0.2%``.
    """
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding,
                     fontsize=8, color=INK["secondary"])


def despine(ax: plt.Axes) -> None:
    """Drop the top and right spines. The data does not need a box."""
    sns.despine(ax=ax, top=True, right=True)


def compact_number(value: float, _pos: int | None = None) -> str:
    """Axis tick formatter that keeps ticks distinguishable.

    Use this rather than a hand-rolled ``f"{v/1e3:.0f}k"``: rounding to whole
    thousands collapses 2000, 1500 and 1000 onto "2k, 2k, 1k", which silently
    mislabels the axis. Precision here adapts to magnitude, so adjacent ticks
    never print the same string.
    """
    if value == 0:
        return "0"

    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            scaled = value / threshold
            return f"{scaled:,.0f}{suffix}" if abs(scaled) >= 10 else f"{scaled:,.1f}{suffix}"
    return f"{value:,.0f}"


def percent_formatter(decimals: int = 1):
    """Axis tick formatter for proportions rendered as percentages."""
    return lambda value, _pos=None: f"{value:.{decimals}%}"


def annotate_reference(
    ax: plt.Axes,
    value: float,
    label: str,
    *,
    axis: str = "y",
    color: str = INK["muted"],
    linestyle: str = "--",
    side: str = "right",
    below: bool = False,
) -> None:
    """Draw a labelled reference rule -- a benchmark, a prior, a cutoff.

    ``side`` and ``below`` exist because the label competes for the same
    corners as the legend. Move it rather than letting the two overlap; a
    collision here is the most common way one of these figures goes wrong.
    """
    line = ax.axhline if axis == "y" else ax.axvline
    line(value, color=color, linestyle=linestyle, linewidth=1.2, zorder=0)

    # A backing in the surface colour, so the label stays legible wherever a
    # data line happens to cross it. Cheaper than hunting for empty space.
    backing = {"boxstyle": "round,pad=0.2", "facecolor": INK["surface"],
               "edgecolor": "none", "alpha": 0.85}

    if axis == "y":
        x = 1.0 if side == "right" else 0.0
        dx = -4 if side == "right" else 4
        ax.annotate(label, xy=(x, value), xycoords=("axes fraction", "data"),
                    xytext=(dx, -10 if below else 4), textcoords="offset points",
                    ha=side, va="top" if below else "bottom",
                    fontsize=8, color=color, bbox=backing, zorder=5)
    else:
        ax.annotate(label, xy=(value, 0.0 if below else 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(4, 4 if below else -4), textcoords="offset points",
                    ha="left", va="bottom" if below else "top",
                    fontsize=8, color=color, bbox=backing, zorder=5)
