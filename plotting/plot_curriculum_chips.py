#!/usr/bin/env python3
"""Curriculum-order figure: rows = pairs.

For each (source -> target) pair we show two chip rows:
  Left  panel: OSDA classes ADDED to the target-private set (target side).
  Right panel: PDA classes ADDED to the source-private set (source side).
              (Both setups keep the 4 shared knowns; OSDA grows target,
               PDA grows source.)
Chips are ordered left -> right from n=1 (hardest) to n=last (easiest),
colour-graded by FNO mean cosine distance (viridis).
UniDA = both columns applied simultaneously.

Reads:
  feature_distance_4known_fno_mean/rankings_4known_fno_mean.json     (OSDA)
  feature_distance_4known_fno_mean_pda/rankings_4known_fno_mean_pda.json (PDA)

Writes:
  figures/curriculum_chips.png
"""
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSDA_JSON = os.path.join(ROOT, "feature_distance_4known_fno_mean",
                         "rankings_4known_fno_mean.json")
PDA_JSON  = os.path.join(ROOT, "feature_distance_4known_fno_mean_pda",
                         "rankings_4known_fno_mean_pda.json")
OUT_PATH  = os.path.join(ROOT, "figures", "curriculum_chips.png")

PAIRS = [
    "RealWorld -> Pamap2", "RealWorld -> MHEALTH",
    "Pamap2 -> RealWorld",  "Pamap2 -> MHEALTH",
    "MHEALTH -> RealWorld", "MHEALTH -> Pamap2",
]

SHORT = {
    "Other/Transient": "Other",
    "Ascending Stairs": "AscStairs",
    "Descending Stairs": "DescStairs",
    "Vacuum Cleaning": "Vacuum",
    "Rope Jumping": "RopeJump",
    "Nordic Walking": "NordicWalk",
    "Climbing Stairs": "ClimbStairs",
    "Climbing Up": "ClimbUp",
    "Climbing Down": "ClimbDown",
    "Knees Bending": "KneeBend",
    "Arm Elevation": "ArmElev",
    "Waist Bends": "WaistBend",
    "Jump F&B": "Jumping",
}


def ranked(rankings):
    # Hardest first = lowest cosine distance first (low distance = closer to
    # the source-known prototypes = harder to separate from them).
    return [(r["unknown"], r["mean_cosine_dist"])
            for r in sorted(rankings, key=lambda x: x["mean_cosine_dist"])]


def draw_chips(ax, ranking, x0, y, vmin, vmax, max_w):
    if not ranking:
        ax.text(x0, y, "(none)", fontsize=9, va="center",
                ha="left", color="#888888", style="italic")
        return

    cmap = plt.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
    gap = 0.005
    chip_h = 0.80

    n = len(ranking)
    chip_w = (max_w - (n - 1) * gap) / n
    label_fs = 8.5 if chip_w > 0.045 else (7.5 if chip_w > 0.038 else 6.5)
    tag_fs   = 6.5 if chip_w < 0.045 else 7

    cursor = x0
    for i, (cls, score) in enumerate(ranking, 1):
        label = SHORT.get(cls, cls)
        color = cmap(norm(score))
        t = norm(score)
        text_color = "white" if t < 0.55 else "black"

        box = mpatches.FancyBboxPatch(
            (cursor, y - chip_h / 2), chip_w, chip_h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0.7, edgecolor="black", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(cursor + 0.004, y + chip_h * 0.30, f"{i}",
                fontsize=tag_fs, color=text_color, ha="left", va="center",
                fontweight="bold", zorder=3)
        ax.text(cursor + chip_w / 2, y, label, fontsize=label_fs,
                color=text_color, ha="center", va="center", zorder=3)
        ax.text(cursor + chip_w / 2, y - chip_h * 0.34, f"{score:.2f}",
                fontsize=tag_fs, color=text_color, ha="center", va="center",
                zorder=3)
        cursor += chip_w + gap


def main():
    osda = json.load(open(OSDA_JSON))
    pda  = json.load(open(PDA_JSON))

    all_scores = []
    for d in (osda, pda):
        for v in d.values():
            all_scores.extend(r["mean_cosine_dist"] for r in v["rankings"])
    vmin, vmax = min(all_scores), max(all_scores)

    n_rows = len(PAIRS)
    fig, ax = plt.subplots(figsize=(18, 1.35 * n_rows + 1.6))

    label_x       = 0.008
    left_chips_x  = 0.200
    right_chips_x = 0.625
    max_chip_w    = 0.370

    row_h = 1.0
    for i, pair in enumerate(PAIRS):
        y = (n_rows - 1 - i) * row_h + 0.5
        if i % 2 == 0:
            ax.add_patch(mpatches.Rectangle(
                (0, y - row_h / 2), 1.0, row_h,
                facecolor="#f4f4f4", edgecolor="none", zorder=0))

        ax.text(label_x, y, pair, fontsize=10.5, fontweight="bold",
                va="center", ha="left", zorder=4)

        draw_chips(ax, ranked(osda[pair]["rankings"]),
                   left_chips_x, y, vmin, vmax, max_chip_w)
        draw_chips(ax, ranked(pda[pair]["rankings"]),
                   right_chips_x, y, vmin, vmax, max_chip_w)

    top_y = (n_rows - 1) * row_h + 0.5 + row_h * 0.70
    ax.text(left_chips_x, top_y,
            "OSDA — classes added to the TARGET side   (n=1 hardest →)",
            fontsize=11.5, fontweight="bold", va="bottom", ha="left")
    ax.text(right_chips_x, top_y,
            "PDA — classes added to the SOURCE side   (n=1 hardest →)",
            fontsize=11.5, fontweight="bold", va="bottom", ha="left")

    ax.axvline(0.190, color="#cccccc", linewidth=0.6,
               ymin=0.04, ymax=0.92)
    ax.axvline(0.597, color="#999999", linewidth=0.8, linestyle="--",
               ymin=0.04, ymax=0.92)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, n_rows * row_h + 0.9)
    ax.set_axis_off()

    cax = fig.add_axes([0.30, 0.04, 0.40, 0.018])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                norm=Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("mean cosine distance to nearest source-known prototype "
                 "(higher = more separable / easier)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
