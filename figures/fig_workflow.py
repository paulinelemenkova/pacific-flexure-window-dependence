#!/usr/bin/env python3
"""fig_workflow -- the end-to-end research workflow.

A diagram, not a plot. Five stacked stages, read downwards: open-access grids,
preprocessing, the two parallel inversions, validation, and the released
output. The fork is the point of the figure -- the same profiles are put
through the windowed baseline and through the physics-informed formulation, so
that the two can be compared on identical data.

Fonts: mathtext ignores font.sans-serif and defaults to the DejaVu set, so
every mathtext face is pinned to Nimbus Sans below. \\mathcal and \\hat are
avoided: Nimbus Sans has no calligraphic face and lacks the combining
circumflex, and either would silently pull in a fallback font.

    python3 fig_workflow.py --outdir figures
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ColorBrewer Pastel1, 7 classes, as published at cpt-city
# (cb/qual/Pastel1_07; Harrower & Brewer 2003). RGB triplets taken verbatim
# from the .cpt, converted to hex. Deliberately different from the palette of
# fig_architecture so the two schematics do not read as one figure.
#
#   251/180/174  #fbb4ae   1  pink
#   179/205/227  #b3cde3   2  blue
#   204/235/197  #ccebc5   3  green
#   222/203/228  #decbe4   4  purple
#   254/217/166  #fed9a6   5  orange
#   255/255/204  #ffffcc   6  yellow   -- unused: too pale against white
#   229/216/189  #e5d8bd   7  tan
C_DATA = "#b3cde3"     # blue    -- inputs
C_PREP = "#fed9a6"     # orange  -- preprocessing
C_BASE = "#e5d8bd"     # tan     -- baseline branch
C_PINN = "#ccebc5"     # green   -- proposed branch
C_VALID = "#decbe4"    # purple  -- validation
C_OUT = "#fbb4ae"      # pink    -- released output
# Pastel1 is already a pale scheme, so only a light veil is applied.
ALPHA = 0.90
EDGE = "0.35"
NEW_EDGE = "#d2453f"


def register_nimbus():
    from matplotlib import font_manager
    for d in ("/usr/share/fonts/opentype/urw-base35",
              "/usr/local/share/fonts/urw-base35",
              "/opt/homebrew/share/fonts", "/Library/Fonts",
              os.path.expanduser("~/Library/Fonts")):
        for f in glob.glob(os.path.join(d, "NimbusSans-*.otf")):
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass


def set_rc():
    register_nimbus()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],
        "font.size": 9,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def box(ax, x, y, w, h, title, body, fc, edge=EDGE, lw=0.9, fs=8.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=lw, edgecolor=edge, facecolor=fc,
                                alpha=ALPHA, zorder=3))
    if title and body:
        ax.text(x, y + h * 0.20, title, ha="center", va="center",
                fontsize=fs + 0.5, weight="bold", zorder=4)
        ax.text(x, y - h * 0.20, body, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3)
    else:
        ax.text(x, y, title or body, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="0.30", lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", linewidth=lw,
                                 linestyle=ls, color=color, zorder=2,
                                 mutation_scale=9, shrinkA=1.0, shrinkB=1.0))


def bot(b):
    return (b[0], b[1] - b[3] / 2)


def top(b):
    return (b[0], b[1] + b[3] / 2)


def build():
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 4.4), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W = 0.94
    y1, y2, y3, y4, y5 = 0.905, 0.720, 0.510, 0.300, 0.095
    h = 0.115

    b1 = box(ax, 0.5, y1, W, h, "Open-access grids",
             "bathymetry  |  seafloor age  |  sediment thickness  |  "
             "satellite gravity  |  plate boundaries", C_DATA)

    b2 = box(ax, 0.5, y2, W, h, "Preprocessing",
             "automatic axis tracing  $\\rightarrow$  trench-normal profiles, "
             "10 km spacing  $\\rightarrow$  screening  $\\rightarrow$  "
             "sediment and thermal correction", C_PREP)

    # the fork: identical profiles, two inversions
    bL = box(ax, 0.262, y3, 0.440, 0.145, "Windowed least squares",
             "baseline of the companion studies\n"
             "window swept, 498 admissible windows", C_BASE)
    bR = box(ax, 0.738, y3, 0.440, 0.145, "Physics-informed inversion",
             "governing equation in the loss\nno analysis window", C_PINN,
             edge=NEW_EDGE, lw=1.6)

    b4 = box(ax, 0.5, y4, W, h, "Validation and comparison",
             "synthetic recovery of known parameters  |  the swept "
             "windows  |  published determinations", C_VALID)

    b5L = box(ax, 0.262, y5, 0.440, h, "Per-segment database",
              "elastic thickness, moment,\nshear force, curvature", C_OUT)
    b5R = box(ax, 0.738, y5, 0.440, h, "Maps and figures",
              "rendered with GMT and Matplotlib", C_OUT)

    arrow(ax, bot(b1), top(b2))
    for b in (bL, bR):
        arrow(ax, (b[0], y2 - h / 2), top(b))
        arrow(ax, bot(b), (b[0], y4 + h / 2))
    for b in (b5L, b5R):
        arrow(ax, (b[0], y4 - h / 2), top(b))

    ax.text(0.5, 0.617, "identical profiles enter both branches",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="0.35")
    ax.text(0.5, 0.408, "the two are compared on the same segments",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="0.35")

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}. Schematic; no data. "
              "Source: authors.")
    fig.text(0.005, -0.015, credit, fontsize=8, ha="left", va="top")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_workflow")
    a = ap.parse_args()
    fig = build()
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
