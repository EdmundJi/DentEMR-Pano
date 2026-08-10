#!/usr/bin/env python3
"""Figures used in the release package and manuscript support files.

    PYTHONPATH=code/src .venv/bin/python code/scripts/make_paper_figures.py \
        dataset/dental_clinical_dataset_v2 --outdir paper

Figure 1 -- a release-package workflow summary for the public code and data.

Figure 2 -- a representative case rendered directly from the released archive,
so field names and content stay tied to the published data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#101828"        # primary text
INK2 = "#475467"       # secondary text
EDGE = "#98A2B3"       # box borders
FILL = "#F8FAFC"       # box fill
FILL_ACCENT = "#EEF4FF"   # the v2 bilingual stage
EDGE_ACCENT = "#6E8DC9"
FILL_OUT = "#F0FDF4"      # release box
EDGE_OUT = "#7BAE7F"

CJK = ["Helvetica Neue", "Songti SC", "Arial Unicode MS"]


def box(ax, x, y, w, h, title, lines, fill=FILL, edge=EDGE):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fill, edgecolor=edge, linewidth=1.1))
    ax.text(x + 0.016, y + h - 0.045, title, fontsize=10.5, fontweight="bold",
            color=INK, va="top")
    ax.text(x + 0.016, y + h - 0.105, "\n".join(lines), fontsize=8.4,
            color=INK2, va="top", linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.2, color=INK2,
                                 shrinkA=2, shrinkB=2))


def figure1(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    W, H, GAP = 0.225, 0.34, 0.028
    xs = [0.012, 0.012 + (W + GAP), 0.012 + 2 * (W + GAP), 0.012 + 3 * (W + GAP)]
    ytop, ybot = 0.60, 0.06

    box(ax, xs[0], ytop, W, H, "1 · Case identification", [
        "Dept. of Stomatology HIS,",
        "Jan 2022 – Dec 2024",
        "Inclusion / exclusion criteria",
        "Five physicians (P1–P5)",
        "Output: 463 outpatient cases"])
    box(ax, xs[1], ytop, W, H, "2 · EMR export", [
        "Semi-structured HIS export",
        "UTF-8 conversion",
        "Template noise removal",
        "Structured field extraction",
        "schema-level validation"])
    box(ax, xs[2], ytop, W, H, "3 · De-identification", [
        "Identifiers never transcribed",
        "Categorical placeholders for",
        "indirect identifiers ([EMPLOYER],",
        "[RELATIVE], [LOCATION])",
        "Dates reduced to YYYY-MM",
        "Residual-identifier scanning"])
    box(ax, xs[3], ytop, W, H, "4 · Structured fields", [
        "17-field schema (Table 4)",
        "Verbatim abbreviations, FDI",
        "notation preserved",
        "ICD-10 codes (28 distinct)",
        "Post-hoc format standardisation"])

    box(ax, xs[0], ybot, W + GAP + W, H, "5 · Panoramic radiographs", [
        "PACS retrieval in DICOM format",
        "DICOM metadata stripping and overlay removal",
        "PNG conversion for release",
        "±7-day visit linkage; quality review",
        "Output: 457 paired images"])
    box(ax, xs[3], ybot, W, H, "6 · English release (v2)", [
        "Protect, segment, deduplicate",
        "LLM translation (temperature 0)",
        "Back-translation + rule checks",
        "Two-clinician review with",
        "blinded adjudication; 8 _en fields"], FILL_ACCENT, EDGE_ACCENT)
    box(ax, xs[2], ybot, W, H, "7 · Release", [
        "Chinese and English record copies",
        "457 grayscale PNG radiographs",
        "metadata.csv · CHANGELOG",
        "Figshare DOI · CC BY 4.0",
        "Analysis code on GitHub"], FILL_OUT, EDGE_OUT)

    m = H / 2
    arrow(ax, xs[0] + W, ytop + m, xs[1], ytop + m)
    arrow(ax, xs[1] + W, ytop + m, xs[2], ytop + m)
    arrow(ax, xs[2] + W, ytop + m, xs[3], ytop + m)
    arrow(ax, xs[0] + W / 2, ytop, xs[0] + W / 2, ybot + H)          # 1 -> 5
    arrow(ax, xs[3] + W / 2, ytop, xs[3] + W / 2, ybot + H)          # 4 -> 6
    arrow(ax, xs[3], ybot + m, xs[2] + W, ybot + m)                  # 6 -> 7
    arrow(ax, xs[0] + W + GAP + W, ybot + m, xs[2], ybot + m)        # 5 -> 7

    fig.savefig(outdir / "fig_workflow.pdf", bbox_inches="tight")
    plt.close(fig)


FIELDS = [
    ("chief_complaint", "Chief complaint"),
    ("history_of_present_illness", "History of present illness"),
    ("oral_examination", "Oral examination"),
    ("imaging_examination", "Imaging examination"),
    ("primary_diagnosis", "Primary diagnosis"),
    ("treatment_plan", "Treatment plan"),
    ("procedure", "Procedure"),
    ("physician_advice", "Physician advice"),
]


def figure2(dataset: Path, pid: str, outdir: Path) -> None:
    rec = json.loads((dataset / "clinical_records" / f"{pid}.json")
                     .read_text(encoding="utf-8"))
    img = plt.imread(dataset / "panoramic_radiographs" / rec["image_file"])

    fig = plt.figure(figsize=(12.6, 6.4))
    axi = fig.add_axes([0.015, 0.30, 0.47, 0.62])
    axi.imshow(img, cmap="gray", vmin=0, vmax=1 if img.max() <= 1 else 255)
    axi.axis("off")
    axi.set_title(f"panoramic_radiographs/{rec['image_file']}",
                  fontsize=8.5, color=INK2, family="monospace", pad=5)

    meta = (f"clinical_records/{pid}.json\n"
            f"age {rec['age']} · {'female' if rec['sex'] == '女' else 'male'} · "
            f"physician {rec['attending_physician']} · "
            f"ICD-10 {rec['primary_diagnosis_icd']}")
    axi.text(0, -0.08, meta, transform=axi.transAxes, fontsize=8.5,
             color=INK2, family="monospace", va="top", linespacing=1.6)

    axt = fig.add_axes([0.505, 0.02, 0.48, 0.96])
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.axis("off")

    y = 0.995
    for key, label in FIELDS:
        zh, en = rec[key], rec[f"{key}_en"]
        axt.text(0, y, f"{key}", fontsize=8.0, family="monospace",
                 color=EDGE_ACCENT, va="top")
        y -= 0.030
        axt.text(0.02, y, zh, fontsize=9.2, color=INK, va="top",
                 fontfamily=CJK, wrap=True, linespacing=1.35)
        y -= 0.032 * (1 + len(zh) // 40)
        axt.text(0.02, y, en, fontsize=8.6, color=INK2, va="top",
                 style="italic", wrap=True, linespacing=1.3)
        y -= 0.028 * (1 + len(en) // 90) + 0.020

    fig.savefig(outdir / "fig_case.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("paper"))
    ap.add_argument("--case", default="0000609184")
    args = ap.parse_args()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = CJK
    plt.rcParams["axes.unicode_minus"] = False
    figure1(args.outdir)
    figure2(args.dataset, args.case, args.outdir)
    print("wrote fig_workflow.pdf, fig_case.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
