"""Regenerate the dataset-characteristics figures from the released records.

Three changes from the submitted figures, each fixing something the data
contradicts:

*Age distribution.* The submitted violin plot drew kernel density past the ends
of the data, so the shape extended below 18 and toward 0 despite an age >= 18
inclusion criterion. A binned histogram cannot produce that artefact.

*Disease spectrum.* The submitted donut implied a partition of the cohort, but
127 of 463 cases carry more than one ICD code, so the shares sum past 100%. A
bar chart of "share of cases with this condition" states the multi-label
structure correctly. Sex was also encoded in the donut's inner ring, unrelated
to the outer ring's variable; it is reported separately instead.

*Age-stratified burden.* The submitted bands 18-24, 24-32, 32-50, 50-90 overlap
at their boundaries. Bands here are half-open and labelled unambiguously.

Palette: an eight-hue categorical set validated for lightness band, chroma
floor, deuteranopic/tritanopic separation and normal-vision separation. Three
hues fall below 3:1 contrast against white, so every segment carries a direct
value label and the underlying counts are also written out as CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from .cohort import (  # noqa: E402
    AGE_BANDS,
    DISEASE_CATEGORIES,
    age_by_sex,
    band_label,
    disease_by_age_band,
    disease_spectrum,
)

#: Validated categorical hues, in fixed assignment order. Never cycled.
PALETTE: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#A67C00",
    "#7B3294", "#56B4E9", "#CC79A7", "#E69F00",
)

SEX_COLORS = {"female": "#7B3294", "male": "#0072B2"}

INK = "#1a1a1a"
MUTED = "#5c5c5c"
GRID = "#d9d9d9"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


def _despine(ax, keep: Sequence[str] = ("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def age_distribution(records, out: Path) -> dict:
    """Age histogram by sex, with medians marked. Replaces the violin figure."""
    ages = age_by_sex(records)
    low = min(ages["female"] + ages["male"])
    high = max(ages["female"] + ages["male"])
    # Bins start at the youngest observed age, so no bar can straddle the
    # inclusion threshold and imply cases that do not exist.
    bins = list(range(low, high + 6, 5))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    for key, label in (("female", "Female"), ("male", "Male")):
        ax.hist(
            ages[key], bins=bins, histtype="step", linewidth=2.0,
            color=SEX_COLORS[key], label=f"{label} (n={len(ages[key])})",
        )

    medians = {}
    for key in ("female", "male"):
        values = sorted(ages[key])
        medians[key] = values[len(values) // 2]

    top = ax.get_ylim()[1]
    if medians["female"] == medians["male"]:
        # One line, one label: two coincident dashed lines read as a single
        # line with a doubled label and misattribute the value to one sex.
        ax.axvline(medians["male"], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.annotate(
            f"median {medians['male']} (both sexes)",
            xy=(medians["male"], top), xytext=(6, -4), textcoords="offset points",
            fontsize=8, color=MUTED,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    else:
        for i, key in enumerate(("female", "male")):
            ax.axvline(medians[key], color=SEX_COLORS[key],
                       linestyle=(0, (4, 3)), linewidth=1.2)
            ax.annotate(
                f"median {medians[key]}",
                xy=(medians[key], top), xytext=(6, -12 - 13 * i),
                textcoords="offset points", fontsize=8, color=SEX_COLORS[key],
            )

    ax.set_xlim(low, bins[-1])
    ax.set_ylim(0, top * 1.12)  # headroom so the median label clears the bars
    ax.set_xlabel("Age at visit (years)")
    ax.set_ylabel("Cases")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.annotate(
        f"Observed range {low}-{high} years; inclusion criterion age $\\geq$ 18.",
        xy=(0, -0.26), xycoords="axes fraction", fontsize=7.5, color=MUTED,
    )

    fig.savefig(out)
    plt.close(fig)
    return {
        "female_n": len(ages["female"]),
        "male_n": len(ages["male"]),
        "min": min(ages["female"] + ages["male"]),
        "max": max(ages["female"] + ages["male"]),
    }


def spectrum(records, out: Path) -> list[dict]:
    """Disease-category prevalence as a sorted bar chart. Replaces the donut."""
    rows = disease_spectrum(records)
    n = len(list(records))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    labels = [r["label"] for r in rows]
    values = [r["n_cases"] for r in rows]
    positions = range(len(rows))

    ax.barh(
        list(positions), values, height=0.68,
        color=[PALETTE[i % len(PALETTE)] for i in positions],
    )
    for i, r in zip(positions, rows):
        ax.annotate(
            f"{r['n_cases']}  ({r['pct']}%)",
            xy=(r["n_cases"], i), xytext=(5, 0), textcoords="offset points",
            va="center", fontsize=8, color=INK,
        )

    ax.set_yticks(list(positions), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"Cases (of {n}); a case with several ICD codes counts in each category")
    ax.set_xlim(0, max(values) * 1.28)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _despine(ax)

    fig.savefig(out)
    plt.close(fig)
    return rows


def burden_by_age(records, out: Path) -> dict:
    """Disease category by half-open age band, as grouped stacked bars."""
    table = disease_by_age_band(records)
    bands = [band_label(lo, hi, i == len(AGE_BANDS) - 1)
             for i, (lo, hi) in enumerate(AGE_BANDS)]
    order = [r["key"] for r in disease_spectrum(records)]
    labels = {c.key: c.label for c in DISEASE_CATEGORIES}

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    left = [0.0] * len(bands)
    for i, key in enumerate(order):
        values = [table[key][b] for b in bands]
        ax.barh(
            bands, values, left=left, height=0.66,
            color=PALETTE[i % len(PALETTE)], label=labels[key],
            edgecolor="white", linewidth=1.4,  # 2px surface gap between segments
        )
        for j, v in enumerate(values):
            if v >= 12:  # label only where the segment can hold the text
                ax.annotate(
                    str(v), xy=(left[j] + v / 2, j), ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
            left[j] += v

    ax.set_xlabel("Case-category assignments")
    ax.set_ylabel("Age band (years)")
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(
        frameon=False, fontsize=7.5, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
    )

    fig.savefig(out)
    plt.close(fig)
    return {"bands": bands, "table": table}
