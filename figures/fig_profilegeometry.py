#!/usr/bin/env python3
"""fig_profilegeometry -- annotated definition sketch of the profile geometry.

Defines, on one idealised trench-normal section, every quantity the
parameterisation measures and every search window it uses. This figure is
DELIBERATELY SCHEMATIC: the section is the analytic broken-plate solution of
morphometry.deflection, not a sampled profile, because a definition sketch must
show the geometry in its clean form. No number on it is a measurement, and the
caption says so.

What makes it authoritative rather than decorative is that the search windows
are drawn from morphometry.py's own constants -- AXIS_WINDOW, CREST_MIN_X,
REGIONAL_PAD -- so if those are retuned the sketch follows automatically and
cannot drift out of step with the code.

    python3 fig_profilegeometry.py --outdir figures
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from morphometry import (deflection, AXIS_WINDOW, CREST_MIN_X, REGIONAL_PAD)

# Idealised section. Values chosen to look like a circum-Pacific margin; they
# are illustrative and are never reported as results.
ALPHA_KM = 75.0         # flexural parameter (Te ~ 30 km, typical)
W0_KM = 2.55            # deflection at the axis, km
D_R_KM = 5.45           # regional reference depth, km
LAND_GRAD = 0.021       # landward slope, km per km
X_LAND = -110.0
X_SEA = 430.0



def _register_urw():
    """Make Nimbus Sans findable even when it is not in the system font cache.

    Nimbus Sans is the URW clone of Helvetica and is metrically compatible with
    it, so it gives the Helvetica look without depending on a proprietary font
    being installed. It ships with ghostscript / fonts-urw-base35 on Linux; on
    macOS install it (e.g. `brew install --cask font-urw-base35`) or the list
    below falls through to Helvetica, which macOS does have.
    """
    from matplotlib import font_manager
    import glob
    for path in ("/usr/share/fonts/opentype/urw-base35",
                 "/usr/local/share/fonts/urw-base35",
                 "/opt/homebrew/share/fonts"):
        for f in glob.glob(path + "/NimbusSans-*.otf"):
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass

def set_rc():
    _register_urw()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.labelsize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.default": "regular", "savefig.dpi": 600,
    })


def style_axes(ax, xlabel, ylabel):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5, pad=2)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.85")
    ax.grid(which="minor", lw=0.3, color="0.92")
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def covers_data(fig, ax, artist, data_xy, text_only=True):
    """Count data points inside a label's text box (rule 1/7)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = (Text.get_window_extent(artist, renderer=r) if text_only
          else artist.get_window_extent(renderer=r))
    pts = ax.transData.transform(data_xy)
    return int((((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                 (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))).sum())


def push_clear(fig, ax, labels, data_xy, max_steps=25, step_px=3.0):
    """Move each label away from the section until its text box is clear.

    Direction is taken from the label's own vertical alignment: one set below
    the section, one above. Working in pixels means the result does not depend
    on the axis limits or the figure size, both of which change whenever the
    layout is adjusted.
    """
    from matplotlib.text import Text
    for a in labels:
        va = a.get_va()
        if va not in ("top", "bottom"):
            continue
        sign = +1.0 if va == "top" else -1.0      # 'top' sits below the data
        for _ in range(max_steps):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            bb = Text.get_window_extent(a, renderer=r)
            pts = ax.transData.transform(data_xy)
            hit = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                   (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum()
            if not hit:
                break
            xt, yt = a.get_position()
            px, py = ax.transData.transform((xt, yt))
            _, y_new = ax.transData.inverted().transform((px, py - sign * step_px))
            a.set_position((xt, y_new))
    fig.canvas.draw()


def nudge_inside(fig, ax, labels, margin_px=3.0):
    """Shift any label whose text box runs past the frame back inside it."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer=r)
    for a in labels:
        bb = Text.get_window_extent(a, renderer=r)
        dx = 0.0
        if bb.x0 < ab.x0 + margin_px:
            dx = (ab.x0 + margin_px) - bb.x0
        elif bb.x1 > ab.x1 - margin_px:
            dx = (ab.x1 - margin_px) - bb.x1
        if dx:
            xt, yt = a.get_position()
            px, py = ax.transData.transform((xt, yt))
            x_new, _ = ax.transData.inverted().transform((px + dx, py))
            a.set_position((x_new, yt))
    fig.canvas.draw()


def section():
    """The idealised section: landward slope, axis, seaward flexure."""
    x = np.arange(X_LAND, X_SEA + 0.5, 0.5)
    d = np.where(x >= 0.0,
                 deflection(np.clip(x, 0.0, None), W0_KM, ALPHA_KM) + D_R_KM,
                 (W0_KM + D_R_KM) + LAND_GRAD * x)
    return x, d


def build():
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 4.4), constrained_layout=True)
    x, d = section()

    # key positions, located exactly as morphometry.descriptors locates them
    i0 = int(np.argmax(np.where(np.abs(x) <= AXIS_WINDOW, d, -np.inf)))
    x0, d0 = float(x[i0]), float(d[i0])
    win = (x > x0 + CREST_MIN_X) & (x <= x0 + 240.0)
    ib = int(np.argmin(d[win]))
    x_b, d_b = float(x[win][ib]), float(d[win][ib])
    x_ref = x_b + REGIONAL_PAD

    lo_d, hi_d = float(d.min()), float(d.max())
    pad = 0.09 * (hi_d - lo_d)
    ax.set_xlim(X_LAND, X_SEA)
    ax.set_ylim(hi_d + 4.2 * pad, lo_d - 2.6 * pad)
    y_bot, y_top = ax.get_ylim()

    # ---- search windows of the parameterisation, as light bands. They are
    # named in a legend outside the axes rather than in the panel, so the
    # plotting area is left to the parameter definitions alone.
    from matplotlib.patches import Patch
    # Colour alone cannot separate these: composited on white their
    # luminances differ by under 0.002, so in greyscale or to a colour-blind
    # reader they are one flat tone. The hatch carries the distinction and the
    # colour merely reinforces it.
    bands = [(-AXIS_WINDOW, AXIS_WINDOW, "#c4a35a", 0.22, "///",
              f"Axis sought within {AXIS_WINDOW:.0f} km of the node"),
             (x0 + CREST_MIN_X, x_b, "#4c8fbd", 0.15, "\\\\",
              f"Crest sought beyond {CREST_MIN_X:.0f} km"),
             (x_ref, X_SEA, "#5aa572", 0.17, "...",
              f"Reference beyond crest $+$ {REGIONAL_PAD:.0f} km")]
    plt.rcParams["hatch.linewidth"] = 0.4
    handles = []
    for xa, xb_, col, al, hatch, name in bands:
        ax.axvspan(xa, xb_, facecolor=col, alpha=al, lw=0, zorder=0)
        ax.axvspan(xa, xb_, facecolor="none", edgecolor="0.62", lw=0,
                   hatch=hatch, alpha=0.55, zorder=0)
        handles.append(Patch(facecolor=col, alpha=al, edgecolor="0.55",
                             lw=0.5, hatch=hatch, label=name))

    ax.fill_between(x, d, y_bot, color="#b08968", alpha=0.07, lw=0, zorder=0)

    ax.plot(x, d, color="0.15", lw=1.4, zorder=6)
    ax.axhline(D_R_KM, color="0.35", lw=0.8, ls=(0, (5, 3)), zorder=6)
    ax.axvline(x0, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=6)
    ax.plot([x0, x_b], [d0, d_b], ls="none", marker="o", ms=5.0,
            mfc="white", mec="0.15", mew=1.1, zorder=8)

    # the landward reach is illustrative; nothing there is fitted
    ax.axvspan(X_LAND, x0, color="white", alpha=0.55, lw=0, zorder=6.5)

    style_axes(ax, "Distance from trench axis, seaward positive (km)",
               "Depth (km), positive downwards")
    ax.xaxis.set_major_locator(MultipleLocator(100))
    # A figure-level legend placed outside the axes; an axes-anchored one this
    # wide fights constrained_layout and collapses the panel.
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.005), ncol=3, frameon=False,
                     fontsize=8.5, handlelength=1.5, handleheight=1.0,
                     handletextpad=0.9, columnspacing=2.2, borderpad=0.2)
    leg.set_zorder(9)

    txt = dict(fontsize=9, color="0.15", zorder=9)
    arrow = dict(arrowstyle="->", lw=0.7, color="0.35", shrinkA=1.5,
                 shrinkB=1.5)
    lab = []
    y_lo = hi_d + 1.3 * pad          # blank band beneath the section

    # ---- the two located points
    lab.append(ax.annotate(r"Axis $(x_0,\ d_0)$", xy=(x0, d0),
                           xytext=(x0, y_lo), ha="center", va="top",
                           arrowprops=arrow, **txt))
    lab.append(ax.annotate(r"Outer-rise crest $(x_b,\ d_b)$", xy=(x_b, d_b),
                           xytext=(x_b + 8.0, d_b - 1.5 * pad), ha="left",
                           va="bottom", arrowprops=arrow, **txt))

    # ---- the regional reference, labelled over deep water at the left
    lab.append(ax.text(X_LAND + 8.0, D_R_KM - 0.30 * pad,
                       r"Regional reference depth $d_r$", ha="left",
                       va="bottom", **txt))

    # ---- outer-rise amplitude: a small vertical extent, labelled clear of it
    x_wb = x_b + 34.0
    ax.annotate("", xy=(x_wb, D_R_KM), xytext=(x_wb, d_b),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.20"),
                zorder=9)
    lab.append(ax.annotate(r"$w_b$", xy=(x_wb, 0.5 * (D_R_KM + d_b)),
                           xytext=(x_wb + 40.0, D_R_KM + 1.5 * pad),
                           ha="left", va="center", arrowprops=arrow, **txt))

    # ---- crest distance, measured above the section where the panel is empty
    y_xb = lo_d - 1.5 * pad
    ax.annotate("", xy=(x0, y_xb), xytext=(x_b, y_xb),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.20"),
                zorder=9)
    lab.append(ax.text(0.5 * (x0 + x_b), y_xb - 0.25 * pad, r"$x_b$",
                       ha="center", va="bottom", **txt))

    # ---- the two mean gradients, mirrored about the axis
    dxs = 68.0
    # the label must clear the section across its own width, not merely at
    # the point the arrow touches
    half = 46.0
    y_grad = max(float(np.nanmax(d[np.abs(x - (x0 + sgn * dxs)) <= half]))
                 for sgn in (-1.0, +1.0)) + 0.95 * pad
    for xt, name in ((x0 - dxs, "Landward\ngradient $S_l$"),
                     (x0 + dxs, "Seaward\ngradient $S_s$")):
        lab.append(ax.annotate(name, xy=(xt, float(np.interp(xt, x, d))),
                               xytext=(xt, y_grad), ha="center", va="top",
                               arrowprops=arrow, **txt))

    # ---- sign convention, in the empty band beneath the section
    ax.annotate("", xy=(x0 + 150.0, y_lo + 0.9 * pad),
                xytext=(x0 + 60.0, y_lo + 0.9 * pad),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="0.30"),
                zorder=9)
    lab.append(ax.text(x0 + 158.0, y_lo + 0.9 * pad, "seaward, $x>0$",
                       ha="left", va="center", fontsize=8.5, color="0.30",
                       zorder=9))

    # ---- lay the labels clear of the section, then audit
    data_xy = np.column_stack([x, d])
    nudge_inside(fig, ax, lab)
    push_clear(fig, ax, lab, data_xy)
    hits = 0
    for a in lab:
        h = covers_data(fig, ax, a, data_xy)
        if h:
            print(f"  label '{a.get_text()[:28]}' covers {h} points")
        hits += h
    print(f"overlap check -- {len(lab)} labels cover {hits} data points")

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, NumPy. "
              "Schematic; no data. Source: authors.")
    fig.text(0.005, -0.085, credit, fontsize=8, ha="left", va="top")

    print(f"sketch: x0 = {x0:.0f} km, crest at {x_b:.0f} km, "
          f"w_b = {D_R_KM - d_b:.2f} km, reference beyond {x_ref:.0f} km")
    return fig, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_profilegeometry")
    a = ap.parse_args()
    fig, hits = build()
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)
    if hits:
        print("WARNING: a label covers the section -- fix before publishing")


if __name__ == "__main__":
    main()
