#!/usr/bin/env python3
"""Draw Figure 2 (representative case) from a real released record.

    python code/scripts/make_hero_figure.py \
        --dataset dataset/dental_clinical_dataset_v2 --out figure.pdf

Uses patient 0000617342: panoramic radiograph with three highlighted regions
matching the record's imaging_examination findings, and the released narrative
fields (Chinese source + English translation, abridged) grouped along the
clinical reasoning chain. Field names and values are taken verbatim from the
released archives, so the figure cannot drift from the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "PingFang SC",
                                   "Arial Unicode MS", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

PID = "0000617342"
W, H = 1400, 830
DARK = "#1a1a1a"

LAV = dict(head="#d8cdee", edge="#a08cc8")
BLU = dict(head="#bcd5ec", edge="#7fa8d0")
PNK = dict(head="#f4c7de", edge="#d98bbb")

GROUPS = [  # (ribbon color, ribbon label, box face, box edge)
    ("#7fa8d0", "1 Presentation", "#eef4fb", "#7fa8d0"),
    ("#e0b73d", "2 Assessment", "#fdf6e3", "#d9b93f"),
    ("#4f9e63", "3 Diagnosis", "#edf7ef", "#4f9e63"),
    ("#9f86cf", "4 Management", "#f5f0fb", "#9f86cf"),
]

# ROI in full-resolution pano pixels: (x0, y0, x1, y1, color, label)
ROIS = [
    (755, 380, 915, 540, "#2e9e46", "11 distal:\nradiolucency"),
    (1000, 495, 1160, 655, "#1f9bbf", "36 occlusal:\nradiopacity\n(temporary filling)"),
    (745, 525, 905, 685, "#e08a1e", "42 distal:\nradiolucency"),
]


def header(ax, x, y, w, h, colors, title, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6",
                                fc=colors["head"], ec="none", zorder=3))
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=DARK, zorder=4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("dataset/dental_clinical_dataset_v2"))
    ap.add_argument("--out", default="figure.pdf")
    args = ap.parse_args()

    zh = json.loads((args.dataset / "dental_clinical_dataset_v2_zh" /
                     "clinical_records" / f"{PID}.json").read_text())
    img = plt.imread(args.dataset / "dental_clinical_dataset_v2_zh" /
                     "panoramic_radiographs" / f"{PID}_panorama.png")

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="white", ec="none", zorder=0))

    # ── Left column: full pano with ROI boxes ────────────────────────────
    lx, lw = 14, 470
    ax.add_patch(FancyBboxPatch((lx, 14), lw, H - 28,
                                boxstyle="round,pad=0,rounding_size=8",
                                fc="white", ec=LAV["edge"], lw=1.5,
                                linestyle=(0, (5, 3)), zorder=1))
    header(ax, lx + 10, H - 62, lw - 20, 40, LAV, "Panoramic Radiograph")
    pw = lw - 40
    ph = pw * img.shape[0] / img.shape[1]
    px, py = lx + 20, H - 80 - ph
    ax.imshow(img, extent=(px, px + pw, py, py + ph), cmap="gray",
              zorder=2, aspect="auto")
    sx, sy = pw / img.shape[1], ph / img.shape[0]
    for x0, y0, x1, y1, col, _ in ROIS:
        ax.add_patch(Rectangle((px + x0 * sx, py + ph - y1 * sy),
                               (x1 - x0) * sx, (y1 - y0) * sy,
                               fill=False, ec=col, lw=2.2, zorder=5))
    ax.text(px + pw / 2, py - 18,
            f"{PID}_panorama.png   1722 x 922, 8-bit grayscale",
            ha="center", va="center", fontsize=9, family="monospace", color=DARK)

    # structured strip
    strip_y = py - 88
    ax.add_patch(FancyBboxPatch((lx + 20, strip_y), lw - 40, 52,
                                boxstyle="round,pad=0,rounding_size=6",
                                fc="#f4f4f4", ec="#999999", lw=1.0, zorder=2))
    ax.text(lx + 30, strip_y + 26,
            f"patient_id {PID} · age {zh['age']} · sex {zh['sex']} (female)\n"
            f"attending_physician {zh['attending_physician']} · has_image yes",
            ha="left", va="center", fontsize=9.5, color=DARK)

    # magnified ROI crops at bottom of left column
    cw, ch, gap = 138, 150, 14
    cx0 = lx + 20
    cy = 175
    for i, (x0, y0, x1, y1, col, lab) in enumerate(ROIS):
        cx = cx0 + i * (cw + gap)
        crop = img[y0:y1, x0:x1]
        ax.imshow(crop, extent=(cx, cx + cw, cy, cy + ch), cmap="gray",
                  zorder=4, aspect="auto")
        ax.add_patch(Rectangle((cx, cy), cw, ch, fill=False, ec=col, lw=2.6,
                               zorder=5))
        ax.text(cx + cw / 2, cy + ch + 14, lab.split(":")[0], ha="center",
                va="center", fontsize=9.5, fontweight="bold", color=col)
    ax.text(lx + lw / 2, cy - 22, "Magnified regions referenced in "
            "imaging_examination", ha="center", va="center", fontsize=9.5,
            color=DARK, style="italic")
    ax.add_patch(FancyBboxPatch((lx + 20, 32), lw - 40, 92,
                                boxstyle="round,pad=0,rounding_size=6",
                                fc="#f4f4f4", ec="#999999", lw=1.0, zorder=2))
    ax.text(lx + 30, 78,
            "Release: two synchronized copies\n"
            "  zh copy - 8 Chinese narrative fields\n"
            "  en copy - *_en English translations\n"
            "aligned by patient_id / image_file",
            ha="left", va="center", fontsize=9, family="monospace",
            color=DARK, linespacing=1.5, zorder=3)

    # ── Right column: released narrative fields ──────────────────────────
    rx, rw = 510, 876
    ax.add_patch(FancyBboxPatch((rx, 14), rw, H - 28,
                                boxstyle="round,pad=0,rounding_size=8",
                                fc="white", ec=PNK["edge"], lw=1.5,
                                linestyle=(0, (5, 3)), zorder=1))
    header(ax, rx + 10, H - 62, rw - 20, 40, PNK,
           "Structured Clinical Record (released JSON fields, abridged)")

    def fieldrow(ytop_row, name, zh_lines, en_lines):
        y = ytop_row
        ax.text(rx + 26, y, name, ha="left", va="top", fontsize=9.5,
                family="monospace", fontweight="bold", color=DARK, zorder=5)
        y -= 17
        ax.text(rx + 26, y, "\n".join(zh_lines), ha="left", va="top",
                fontsize=9.5, color=DARK, zorder=5, linespacing=1.35)
        y -= 17 * len(zh_lines)
        ax.text(rx + 26, y, "\n".join(en_lines), ha="left", va="top",
                fontsize=8.8, color="#444444", style="italic", zorder=5,
                linespacing=1.3)
        y -= 15 * len(en_lines) + 10
        return ytop_row - y

    boxes = [
        (0, [
            ("chief_complaint", ["左下后牙冷热刺激疼痛半年"],
             ["(Pain on thermal stimulation of the left lower posterior tooth for 6 months.)"]),
            ("history_of_present_illness",
             ["半年前出现左下后牙冷热刺激疼痛，1周前于外院行根管治疗，",
              "现因担心病情变化来我院就诊"],
             ["(Thermal-stimulation pain for 6 months; root canal treatment at another",
              "hospital 1 week ago; attends today concerned about changes in the condition.)"]),
        ]),
        (1, [
            ("oral_examination",
             ["11远中邻面龋，探(-)，叩(-)；36合面可见暂封材料，探(-)，叩(-)；",
              "42远中可见龋坏，叩(+)"],
             ["(11 distal caries, probing (-), percussion (-); 36 occlusal temporary filling;",
              "42 distal caries, percussion (+).)"]),
            ("imaging_examination",
             ["11远中邻面可见低密度影，36合面可见高密度影，42远中可见低密度影"],
             ["(11 distal radiolucency; 36 occlusal radiopacity; 42 distal radiolucency.)"]),
        ]),
        (2, [
            ("primary_diagnosis", ["1牙髓炎 2龋病"],
             ["(1. Pulpitis  2. Dental caries)"]),
            ("primary_diagnosis_icd", ["K04.x;K02.x"],
             ["(ICD-10 atoms are masked to chapter level in the release.)"]),
        ]),
        (3, [
            ("treatment_plan",
             ["根管治疗；后期A直接树脂修复 / B树脂修复后冠修复 / C打桩+树脂 /",
              "D桩冠修复……患者知情同意，选择方案1"],
             ["(Root canal treatment; later resin or crown restoration (options A-D) ...",
              "informed consent obtained; option 1 chosen.)"]),
            ("physician_advice",
             ["1嘱患者禁用患牙咬物；2口腔卫生宣教，建议行牙周洁治；",
              "3定期复诊，不适随诊"],
             ["(Avoid biting with the affected tooth; oral hygiene instruction and",
              "periodontal scaling recommended; regular follow-up if any discomfort.)"]),
        ]),
    ]

    def box_height(rows):
        h = 16
        for _, zh_l, en_l in rows:
            h += 17 + 17 * len(zh_l) + 15 * len(en_l) + 10
        return h

    ytop = H - 80
    for gi, rows in boxes:
        col_r, lab, face, edge = GROUPS[gi]
        hgt = box_height(rows)
        y0 = ytop - hgt
        ax.add_patch(FancyBboxPatch((rx + 14, y0), rw - 92, hgt,
                                    boxstyle="round,pad=0,rounding_size=8",
                                    fc=face, ec=edge, lw=1.6, zorder=2))
        ax.add_patch(FancyBboxPatch((rx + rw - 68, y0 + 6), 44, hgt - 12,
                                    boxstyle="round,pad=0,rounding_size=8",
                                    fc=col_r, ec="none", zorder=2))
        ax.text(rx + rw - 46, y0 + hgt / 2, lab, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", rotation=270,
                zorder=3)
        yy = ytop - 12
        for name, zh_l, en_l in rows:
            used = fieldrow(yy, name, zh_l, en_l)
            yy -= used
        if gi < 3:
            ax.add_patch(FancyArrowPatch((rx + (rw - 92) / 2, y0),
                                         (rx + (rw - 92) / 2, y0 - 13),
                                         arrowstyle="-|>", mutation_scale=16,
                                         lw=2.2, color="#555555", zorder=4))
        ytop = y0 - 15

    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
