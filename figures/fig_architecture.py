#!/usr/bin/env python3
"""fig_architecture -- scheme of the physics-informed flexural inversion.

A diagram, not a plot: it carries no data and has no quantitative axes, so the
house tick-and-grid rules do not apply. Everything else does -- Nimbus Sans,
8-10 pt text, colour-blind-safe fills, nothing overlapping anything.

The figure answers one question: where does the physics enter, and what is
trainable? Reading left to right along the top, collocation points go into the
network, the network returns the deflection, and automatic differentiation
turns that single output into the derivatives the flexure equation needs.
Reading left to right along the bottom, the three loss terms are assembled and
fed back to the two sets of trainable quantities -- the network weights and the
per-segment flexural parameter. No analysis window appears anywhere, which is
the point of the method.

    python3 fig_architecture.py --outdir figures
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Palette: muted, perceptually distinct, and separable in greyscale by
# luminance as well as hue (input pale, physics mid, loss dark).
# One hue per column, the loss box a lighter tint of the same hue, so a
# column reads as a unit without needing an arrow to say so.
C_A, C_A2 = "#a8d5f5", "#dcefFB"      # network
C_B, C_B2 = "#ffd28a", "#ffeed2"      # differentiation / physics
C_C, C_C2 = "#a9e3a0", "#dff5db"      # boundary
C_LOSS = "#d9c9f2"                    # composite loss
C_TRAIN = "#ffd9d6"                   # update
EDGE = "0.35"
TRAIN_EDGE = "#d2453f"


def register_nimbus():
    """House rule: Nimbus Sans, never DejaVu Sans."""
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
        # Math is rendered by mathtext, which ignores font.sans-serif and
        # defaults to the DejaVu set. Every face is therefore pinned to Nimbus
        # Sans explicitly, or the figure ships DejaVu glyphs despite the text
        # resolving correctly.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans",
        "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def box(ax, x, y, w, h, text, fc, fs=9, weight="normal"):
    """Rounded box centred on (x, y), with centred text."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.010,rounding_size=0.014",
                       linewidth=0.8, edgecolor=EDGE, facecolor=fc,
                       zorder=3)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            weight=weight, zorder=4, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, style="-|>", ls="-", lw=1.0, color="0.30", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, linestyle=ls,
                                 linewidth=lw, color=color, zorder=2,
                                 mutation_scale=9,
                                 shrinkA=1.0, shrinkB=1.0,
                                 connectionstyle=f"arc3,rad={rad}"))


def right(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0] - b[2] / 2, b[1])


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


def top(b):
    return (b[0], b[1] + b[3] / 2)


def trunk(ax, x0, x1, yc, layers=(1, 5, 5, 1)):
    """Fully connected trunk drawn as nodes and edges."""
    xs = np.linspace(x0, x1, len(layers))
    coords = []
    for xi, n in zip(xs, layers):
        ys = np.linspace(yc - 0.024, yc + 0.024, n) if n > 1 else [yc]
        coords.append([(xi, y) for y in ys])
    for a, b in zip(coords[:-1], coords[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], lw=0.35, color="0.72",
                        zorder=2)
    for col in coords:
        for (xi, yi) in col:
            ax.add_patch(Circle((xi, yi), 0.0060, facecolor="white",
                                edgecolor=EDGE, lw=0.7, zorder=4))
    return coords


def build():
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 3.3), constrained_layout=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Three columns, each a stage of the method; the loss term produced by a
    # stage sits directly beneath it, so every forward arrow is short and
    # vertical. The gradient returns in a single straight segment.
    xa, xb, xc = 0.175, 0.500, 0.825
    wcol = 0.300
    y1, h1 = 0.855, 0.190
    y2, h2 = 0.585, 0.165
    y3, h3 = 0.330, 0.140

    # ---------------- row 1: the three stages --------------------------
    ax.add_patch(FancyBboxPatch((xa - wcol / 2, y1 - h1 / 2), wcol, h1,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=1.6, edgecolor=TRAIN_EDGE,
                                facecolor=C_A, zorder=3))
    ax.text(xa, y1 + 0.062, "Network", ha="center", va="center",
            fontsize=9, weight="bold", zorder=4)
    trunk(ax, xa - 0.072, xa + 0.072, y1 - 0.004, layers=(1, 4, 4, 1))
    # "w-hat" is written with the spacing circumflex rather than \\hat{},
    # whose combining accent (U+0302) is absent from Nimbus Sans and would
    # silently pull in a STIX fallback glyph.
    ax.text(xa, y1 - 0.068,
            "$x \\rightarrow$  \u02c6$w(x;\\theta)$",
            ha="center", va="center", fontsize=8.5, zorder=4)

    b_ad = box(ax, xb, y1, wcol, h1,
               "$\\bf{Automatic\\ differentiation}$\n"
               "$w\'\'$,  $w\'\'\'$,  $w\'\'\'\'$\n"
               "$\\rightarrow \\kappa,\\ M,\\ V$", C_B, fs=8.5)
    b_bnd = box(ax, xc, y1, wcol, h1,
                "$\\bf{Boundary\\ evaluation}$\n"
                "axis and far field", C_C, fs=8.5)
    b_net = (xa, y1, wcol, h1)
    arrow(ax, right(b_net), left(b_ad))
    arrow(ax, right(b_ad), left(b_bnd))

    # ---------------- row 2: the loss term of each stage ---------------
    b_data = box(ax, xa, y2, wcol, h2,
                 "Data misfit\n$L_\\mathrm{data}$", C_A2, fs=8.5)
    ax.add_patch(FancyBboxPatch((xb - wcol / 2, y2 - h2 / 2), wcol, h2,
                                boxstyle="round,pad=0.008,rounding_size=0.014",
                                linewidth=1.6, edgecolor=TRAIN_EDGE,
                                facecolor=C_B2, zorder=3))
    ax.text(xb, y2, "Physics residual  $L_\\mathrm{phys}$\n"
            "$R=D(T_e)\\,w\'\'\'\'+\\Delta\\rho\\,g\\,w-q$",
            ha="center", va="center", fontsize=8.5, zorder=4, linespacing=1.35)
    b_phys = (xb, y2, wcol, h2)
    b_bc = box(ax, xc, y2, wcol, h2,
               "Boundary\n$L_\\mathrm{bc}$", C_C2, fs=8.5)
    for a, b in ((b_net, b_data), (b_ad, b_phys), (b_bnd, b_bc)):
        arrow(ax, bottom(a), top(b))

    # ---------------- row 3: assembly and update ------------------------
    b_upd = box(ax, 0.145, y3, 0.240, h3, "Update\n$\\theta$,  $T_e$",
                C_TRAIN, fs=8.5)
    b_loss = box(ax, 0.610, y3, 0.580, h3,
                 "$L=\\lambda_dL_\\mathrm{data}"
                 "+\\lambda_pL_\\mathrm{phys}"
                 "+\\lambda_bL_\\mathrm{bc}$", C_LOSS, fs=9)
    for a, xt in ((b_data, 0.395), (b_phys, 0.610), (b_bc, 0.810)):
        arrow(ax, bottom(a), (xt, y3 + h3 / 2))
    arrow(ax, left(b_loss), right(b_upd), ls=(0, (4, 2)), color=TRAIN_EDGE)

    ax.text(0.5, 0.150, "Red outline marks where the trainable quantities enter; "
            "no analysis window enters at any stage",
            ha="center", va="center", fontsize=8.5, color=TRAIN_EDGE)

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
    ap.add_argument("--name", default="fig_architecture")
    a = ap.parse_args()
    fig = build()
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
