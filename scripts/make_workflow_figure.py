#!/usr/bin/env python3
"""Draw Figure 1 (dataset construction workflow) in the style of the original
hand-drawn diagram: five pastel panels with dashed borders and colored arrows,
but with content that matches the released v2 archives (8-bit PNG, split zh/en
release, real 17-field schema).

    python code/scripts/make_workflow_figure.py --out submission_package/figure1.png
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Rectangle)

W, H = 1390, 775
DARK = "#1a1a1a"

PINK = dict(head="#f2bbd9", bg="#fdf3f8", edge="#d98bbb")
BLUE = dict(head="#a8c8e8", bg="#f2f7fc", edge="#7fa8d0")
GREEN = dict(head="#b5d9a8", bg="#f3faf0", edge="#8cbf78")
PURPLE = dict(head="#c9b8e8", bg="#f6f2fc", edge="#9f86cf")
YELLOW = dict(head="#f2d370", bg="#fdf9ec", edge="#d9b93f")


def panel(ax, x, y, w, h, colors, title, title_h=64, title_fs=17):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=colors["bg"], ec=colors["edge"], lw=1.6,
                                linestyle=(0, (5, 3)), zorder=1))
    ax.add_patch(FancyBboxPatch((x, y + h - title_h), w, title_h,
                                boxstyle="round,pad=0,rounding_size=8",
                                fc=colors["head"], ec="none", zorder=2))
    ax.add_patch(Rectangle((x, y + h - title_h), w, title_h / 2,
                           fc=colors["head"], ec="none", zorder=2))
    ax.text(x + w / 2, y + h - title_h / 2, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=DARK, zorder=3)


def arrow(ax, p0, p1, color, lw=3.2, ms=22):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=6,
                                 shrinkA=0, shrinkB=0))


def small_arrow(ax, p0, p1, color):
    arrow(ax, p0, p1, color, lw=2.2, ms=14)


def database(ax, cx, cy, s=1.0):
    w, h = 64 * s, 20 * s
    for i, dy in enumerate((0, 26 * s)):
        y = cy + dy
        ax.add_patch(Rectangle((cx - w / 2, y), w, h, fc=DARK, ec="none", zorder=4))
        ax.add_patch(Ellipse((cx, y), w, 12 * s, fc=DARK, ec="none", zorder=4))
        ax.add_patch(Ellipse((cx, y + h), w, 12 * s, fc=DARK, ec="white",
                             lw=1.2 * s, zorder=5))


def xray_machine(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 4 * s, cy), 8 * s, 66 * s, fc=DARK, zorder=4))
    ax.add_patch(FancyBboxPatch((cx - 26 * s, cy + 58 * s), 52 * s, 14 * s,
                                boxstyle="round,pad=0,rounding_size=5",
                                fc=DARK, zorder=4))
    ax.add_patch(Circle((cx, cy + 26 * s), 15 * s, fc="white", ec=DARK,
                        lw=2.4 * s, zorder=5))
    ax.add_patch(Circle((cx, cy + 26 * s), 5 * s, fc=DARK, zorder=6))
    ax.add_patch(FancyBboxPatch((cx - 20 * s, cy - 8 * s), 40 * s, 10 * s,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc=DARK, zorder=4))


def document(ax, cx, cy, w=54, h=68, label="EMR"):
    fold = 14
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2 - fold, cy - h / 2),
           (cx + w / 2, cy - h / 2 + fold), (cx + w / 2, cy + h / 2),
           (cx - w / 2, cy + h / 2)]
    pts = [(px, 2 * cy - py) for px, py in pts]  # flip so fold is top-right
    ax.add_patch(Polygon(pts, closed=True, fc="white", ec=DARK, lw=2.0, zorder=4))
    ax.text(cx, cy + h / 2 - 16, label, ha="center", va="center", fontsize=11,
            fontweight="bold", color=DARK, zorder=5)
    for i in range(4):
        y = cy + 8 - i * 11
        ax.plot([cx - w / 2 + 9, cx + w / 2 - 9], [y, y], color=DARK,
                lw=1.6, zorder=5)


def pano_film(ax, cx, cy, w=96, h=58, masked=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc="#2b2b2b", ec=DARK, lw=1.4, zorder=4))
    t = np.linspace(0.15 * math.pi, 0.85 * math.pi, 40)
    ax.plot(cx + (w * 0.36) * np.cos(t), cy - h * 0.05 - (h * 0.30) * np.sin(t),
            color="#d8d8d8", lw=4, zorder=5, solid_capstyle="round")
    ax.plot(cx + (w * 0.24) * np.cos(t), cy + h * 0.10 - (h * 0.20) * np.sin(t),
            color="#bfbfbf", lw=3, zorder=5, solid_capstyle="round")
    if masked:
        ax.add_patch(Rectangle((cx - w / 2 + 4, cy + h / 2 - 12), 26, 8,
                               fc="black", ec="#8cbf78", lw=1.0, zorder=6))


def gear(ax, cx, cy, r=22, teeth=8, color="#5a5a5a"):
    for i in range(teeth):
        a = i * 2 * math.pi / teeth
        tw, tl = 0.30, r * 0.42
        pts = []
        for da, rr in ((-tw / 2, r), (tw / 2, r), (tw / 4, r + tl), (-tw / 4, r + tl)):
            pts.append((cx + rr * math.cos(a + da), cy + rr * math.sin(a + da)))
        ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=4))
    ax.add_patch(Circle((cx, cy), r, fc=color, ec="none", zorder=5))
    ax.add_patch(Circle((cx, cy), r * 0.42, fc="white", ec="none", zorder=6))


def folder(ax, x, y, w=34, h=24, fc="#f5c86e"):
    ax.add_patch(FancyBboxPatch((x, y), w * 0.45, h * 0.28,
                                boxstyle="round,pad=0,rounding_size=2",
                                fc=fc, ec=DARK, lw=1.2, zorder=4))
    ax.add_patch(FancyBboxPatch((x, y - h + h * 0.2), w, h * 0.85,
                                boxstyle="round,pad=0,rounding_size=3",
                                fc=fc, ec=DARK, lw=1.2, zorder=5))


def badge(ax, cx, cy, text, fc, w=44, h=20):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc=fc, ec=DARK, lw=1.0, zorder=5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=DARK, zorder=6)


def chain(ax, cx, cy):
    for dx, ang in ((-11, 35), (11, -35)):
        e = Ellipse((cx + dx, cy), 34, 20, angle=ang, fc="none", ec=DARK,
                    lw=5, zorder=5)
        ax.add_patch(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figure1.png")
    args = ap.parse_args()

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=0))

    top_y, top_h = 400, 362
    bot_y, bot_h = 14, 348

    # ── Panel 1: collection ──────────────────────────────────────────────
    p1x, p1w = 10, 442
    panel(ax, p1x, top_y, p1w, top_h, PINK, "1. Data Collection &\nEMR Export",
          title_h=78)
    database(ax, p1x + 100, top_y + 205)
    ax.text(p1x + 100, top_y + 178, "HIS - Hospital\nInformation System",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=DARK)
    small_arrow(ax, (p1x + 172, top_y + 228), (p1x + 228, top_y + 228), PINK["edge"])
    ax.text(p1x + 322, top_y + 228, "Semi-structured\nEMR files", ha="center",
            va="center", fontsize=11.5, color=DARK)
    xray_machine(ax, p1x + 100, top_y + 52)
    ax.text(p1x + 100, top_y + 40, "PACS - Panoramic\nRadiographs", ha="center",
            va="top", fontsize=10.5, fontweight="bold", color=DARK)
    small_arrow(ax, (p1x + 172, top_y + 90), (p1x + 228, top_y + 90), PINK["edge"])
    ax.text(p1x + 322, top_y + 90, "DICOM files", ha="center", va="center",
            fontsize=11.5, color=DARK)

    # ── Panel 2: preprocessing ───────────────────────────────────────────
    p2x, p2w = 492, 442
    panel(ax, p2x, top_y, p2w, top_h, BLUE, "2. Data Preprocessing &\nDe-identification",
          title_h=78)
    document(ax, p2x + 70, top_y + 205)
    small_arrow(ax, (p2x + 118, top_y + 205), (p2x + 178, top_y + 205), BLUE["edge"])
    ax.text(p2x + 300, top_y + 205, "Cleaned EMR texts\n(UTF-8, template noise\nremoved)",
            ha="center", va="center", fontsize=11, color=DARK)
    ax.text(p2x + 70, top_y + 158, "Record cleaning", ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=DARK)
    pano_film(ax, p2x + 70, top_y + 75)
    small_arrow(ax, (p2x + 128, top_y + 75), (p2x + 188, top_y + 75), BLUE["edge"])
    pano_film(ax, p2x + 248, top_y + 75, masked=True)
    ax.text(p2x + 366, top_y + 75, "De-identified\n8-bit PNG", ha="center",
            va="center", fontsize=11, color=DARK)
    ax.text(p2x + 160, top_y + 22, "DICOM header stripping & overlay masking",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=DARK)

    # ── Panel 3: filtering ───────────────────────────────────────────────
    p3x, p3w = 976, 404
    panel(ax, p3x, top_y, p3w, top_h, GREEN, "3. Data Filtering", title_h=64)
    crits = ["(1) Incomplete records", "(2) Duplicates", "(3) Abnormal documents",
             "(4) De-identification errors", "(5) Poor image quality"]
    pill_w, pill_h, gap = 300, 40, 9
    py0 = top_y + top_h - 64 - 24
    for i, c in enumerate(crits):
        y = py0 - (i + 1) * (pill_h + gap) + gap
        ax.add_patch(FancyBboxPatch((p3x + 26, y), pill_w, pill_h,
                                    boxstyle="round,pad=0,rounding_size=10",
                                    fc="#dff0d7", ec=GREEN["edge"], lw=1.6, zorder=4))
        ax.text(p3x + 26 + pill_w / 2, y + pill_h / 2, c, ha="center", va="center",
                fontsize=12, fontweight="bold", color=DARK, zorder=5)
        if i < len(crits) - 1:
            small_arrow(ax, (p3x + 26 + pill_w + 18, y + pill_h / 2),
                        (p3x + 26 + pill_w + 18, y - gap + 2), GREEN["edge"])
    ax.text(p3x + 26 + pill_w / 2, top_y + 22,
            "525 collected  →  463 released", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#4a7a38")

    # ── Panel 4: extraction ──────────────────────────────────────────────
    p4x, p4w = 800, 580
    panel(ax, p4x, bot_y, p4w, bot_h, PURPLE, "4. Structured Field Extraction",
          title_h=56)
    gear(ax, p4x + 96, bot_y + 170, r=24)
    gear(ax, p4x + 138, bot_y + 128, r=17)
    ax.text(p4x + 116, bot_y + 84, "Manual extraction\n(2 research assistants,\nprotocol in Methods)",
            ha="center", va="center", fontsize=10.5, color=DARK)
    tx, tw = p4x + 236, 322
    rows = [("chief_complaint", "text"), ("history_of_present_illness", "text"),
            ("oral_examination", "text"), ("imaging_examination", "text"),
            ("primary_diagnosis", "text"), ("primary_diagnosis_icd", "code"),
            ("treatment_plan", "text"), ("procedure", "text"),
            ("physician_advice", "text"),
            ("+ patient_id, age, sex, physician,", ""),
            ("   categories, image linkage", "")]
    th = 21
    ty0 = bot_y + bot_h - 56 - 14
    ax.add_patch(Rectangle((tx, ty0 - th), tw, th, fc="#d9cdf0", ec=DARK,
                           lw=1.2, zorder=4))
    ax.text(tx + 8, ty0 - th / 2, "Field Name", ha="left", va="center",
            fontsize=10, fontweight="bold", color=DARK, zorder=5)
    ax.text(tx + tw - 40, ty0 - th / 2, "Type", ha="left", va="center",
            fontsize=10, fontweight="bold", color=DARK, zorder=5)
    for i, (name, typ) in enumerate(rows):
        y = ty0 - (i + 2) * th
        ax.add_patch(Rectangle((tx, y), tw, th, fc="#f3eefb" if i % 2 else "white",
                               ec=DARK, lw=0.7, zorder=4))
        ax.text(tx + 8, y + th / 2, name, ha="left", va="center", fontsize=8.6,
                family="monospace", color=DARK, zorder=5)
        if typ:
            ax.text(tx + tw - 40, y + th / 2, typ, ha="left", va="center",
                    fontsize=8.6, family="monospace", color=DARK, zorder=5)
    small_arrow(ax, (tx - 18, bot_y + 150), (p4x + 178, bot_y + 150), PURPLE["edge"])

    # ── Panel 5: pairing & organization ─────────────────────────────────
    p5x, p5w = 10, 700
    panel(ax, p5x, bot_y, p5w, bot_h, YELLOW,
          "5. Image-Text Pairing & Dataset Organization", title_h=56)

    def copy_tree(x0, y0, name, extra):
        folder(ax, x0, y0 + 210)
        ax.text(x0 + 44, y0 + 202, name, ha="left", va="center", fontsize=10.5,
                fontweight="bold", family="monospace", color=DARK)
        folder(ax, x0 + 22, y0 + 168, fc="#f8dc94")
        ax.text(x0 + 62, y0 + 160, "clinical_records/", ha="left", va="center",
                fontsize=9, family="monospace", color=DARK)
        badge(ax, x0 + 256, y0 + 160, "JSON", "#f5d76e")
        ax.text(x0 + 286, y0 + 160, "463", ha="left", va="center", fontsize=9.5,
                color=DARK)
        folder(ax, x0 + 22, y0 + 126, fc="#f8dc94")
        ax.text(x0 + 62, y0 + 118, "panoramic_radiographs/", ha="left", va="center",
                fontsize=9, family="monospace", color=DARK)
        badge(ax, x0 + 256, y0 + 118, "PNG", "#b8dba8")
        ax.text(x0 + 286, y0 + 118, "457", ha="left", va="center", fontsize=9.5,
                color=DARK)
        ax.text(x0 + 22, y0 + 84, "metadata.csv · README.md", ha="left",
                va="center", fontsize=9.5, family="monospace", color=DARK)
        ax.text(x0 + 22, y0 + 62, extra, ha="left", va="center", fontsize=9.5,
                family="monospace", color=DARK)

    copy_tree(p5x + 24, bot_y, "..._v2_zh/  (Chinese source)", "CHANGELOG.md")
    copy_tree(p5x + 366, bot_y, "..._v2_en/  (English, *_en)",
              "translation_screening_log.json")
    chain(ax, p5x + 350, bot_y + 34)
    ax.text(p5x + 350, bot_y + 12, "aligned by patient_id", ha="center",
            va="center", fontsize=10, fontweight="bold", color=DARK)

    # ── Inter-panel arrows ───────────────────────────────────────────────
    arrow(ax, (p1x + p1w + 2, top_y + 180), (p2x - 2, top_y + 180), PINK["edge"])
    arrow(ax, (p2x + p2w + 2, top_y + 180), (p3x - 2, top_y + 180), BLUE["edge"])
    arrow(ax, (p3x + p3w - 40, top_y - 2), (p3x + p3w - 40, bot_y + bot_h + 2),
          GREEN["edge"])
    arrow(ax, (p4x - 2, bot_y + 170), (p5x + p5w + 2, bot_y + 170), PURPLE["edge"])

    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
