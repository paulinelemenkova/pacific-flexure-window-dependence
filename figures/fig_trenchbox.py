#!/usr/bin/env python3
"""fig_trenchbox -- how window-dependent each margin is, as a box per trench.

The window sweep casts every profile of a margin against every admissible
window, so the population of recovered thicknesses at one margin is not a
single number but a distribution whose WIDTH is the window sensitivity of that
margin. This figure draws that distribution as one box per trench:

    box      interquartile range of the windowed thickness
    line     median
    whiskers 5th-95th percentile
    band     the interval-free estimate of Section 4, which has no window

A tall box is a margin whose recovered thickness depends heavily on the fitting
interval; a short box, one that is robust to it. The margins are ordered by
median so the eye reads a gradient, and coloured by a spectral scale so each of
the twenty is separable in the legend, matching the comparison figure.

WHAT THIS USES, AND WHY NOT segments.csv
----------------------------------------
The recovered run has too few segments per margin for a box -- four margins
have exactly one segment, and a box from one point is a line. The sweep grid
has hundreds to thousands of windowed estimates per margin, which is the
distribution the paper's argument is actually about, so it is the right input.

    --sweep     sweep_grid.csv        trench, profile, xl_km, xs_km, te_km
    --segments  runs/r02/segments.csv the interval-free reference line
    --min-xs    only windows with a seaward limit at least this far out are
                pooled, since the near-axis windows are the unusable regime;
                200 km matches the abstract

    python3 fig_trenchbox.py --sweep sweep_grid.csv \\
        --segments runs/r02/segments.csv --outdir figures
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

MIN_N = 50          # a margin with fewer windowed estimates is not boxed

DISPLAY = {
    "izu-bonin": "Izu-Bonin", "kuril-kamchatka": "Kuril-Kamchatka",
    "peru-chile": "Peru-Chile", "middle_america": "Middle America",
    "new_britain": "New Britain", "new_hebrides": "Vanuatu",
    "san_cristobal": "San Cristobal",
}


def pretty(key):
    k = (key or "").strip()
    low = k.lower().replace(" ", "_").replace("-", "_")
    two = key.lower()
    if two in DISPLAY:
        return DISPLAY[two]
    if low.replace("_", "-") in DISPLAY:
        return DISPLAY[low.replace("_", "-")]
    return " ".join(w.capitalize() for w in low.split("_"))


def register_nimbus():
    """Register Nimbus Sans from any of the usual install roots, .otf or .ttf,
    searched recursively so a Homebrew cask install one level deeper is found."""
    from matplotlib import font_manager
    n = 0
    for d in ("/usr/share/fonts/opentype/urw-base35",
              "/usr/local/share/fonts/urw-base35",
              "/opt/homebrew/share/fonts", "/Library/Fonts",
              "/System/Library/Fonts", "/usr/share/fonts",
              os.path.expanduser("~/Library/Fonts")):
        if not os.path.isdir(d):
            continue
        for ext in ("otf", "ttf"):
            for f in glob.glob(os.path.join(d, "**", f"NimbusSans*.{ext}"),
                               recursive=True):
                try:
                    font_manager.fontManager.addfont(f); n += 1
                except Exception:
                    pass
    return n


def set_rc():
    n = register_nimbus()
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Helvetica and Arial ship on macOS without the URW fonts and are
        # metrically close to Nimbus, so the figure matches the rest of the
        # set. DejaVu is the last resort and is what the warning below flags.
        "font.sans-serif": ["Nimbus Sans", "Nimbus Sans L", "Helvetica",
                            "Helvetica Neue", "Arial", "Liberation Sans"],
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "mathtext.default": "regular", "savefig.dpi": 600,
    })
    from matplotlib import font_manager
    resolved = font_manager.FontProperties(
        family=plt.rcParams["font.sans-serif"]).get_name()
    print(f"font: {n} Nimbus face(s) registered, matplotlib resolved "
          f"'{resolved}'")
    if "Nimbus" not in resolved and "Helvetica" not in resolved \
            and "Arial" not in resolved:
        print("  WARNING: fell back to " + resolved + ". Install the URW fonts "
              "(brew install --cask font-urw-base35) so this figure matches "
              "the others.", file=sys.stderr)


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="sweep_grid.csv")
    ap.add_argument("--segments", default="runs/r02/segments.csv")
    ap.add_argument("--min-xs", type=float, default=200.0)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_trenchbox")
    a = ap.parse_args()

    if not os.path.exists(a.sweep):
        sys.exit(f"not found: {a.sweep}")

    pooled = defaultdict(list)
    with open(a.sweep) as fh:
        for r in csv.DictReader(fh):
            if f(r["xs_km"]) >= a.min_xs:
                te = f(r["te_km"])
                if np.isfinite(te) and te > 0:
                    pooled[r["trench"]].append(te)

    data = {t: np.array(v) for t, v in pooled.items() if len(v) >= MIN_N}
    if not data:
        sys.exit("no trench had enough windowed estimates to box")
    # order by median so the figure reads as a gradient
    order = sorted(data, key=lambda t: np.median(data[t]))
    labels = [pretty(t) for t in order]
    print(f"{len(order)} trenches boxed, {a.min_xs:.0f} km seaward floor")

    ref = None
    if os.path.exists(a.segments):
        te = [f(r["te_km"]) for r in csv.DictReader(open(a.segments))]
        te = [x for x in te if np.isfinite(x)]
        ref = float(np.median(te))
        print(f"interval-free reference median {ref:.2f} km")

    set_rc()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    cmap = plt.get_cmap("Spectral")
    colours = [cmap(i / (len(order) - 1)) for i in range(len(order))]

    if ref is not None:
        ax.axhspan(ref * 0.98, ref * 1.02, color="0.55", alpha=0.18, lw=0,
                   zorder=1)
        ax.axhline(ref, color="0.35", lw=1.0, ls=(0, (5, 3)), zorder=2)
        # Label in the RIGHT MARGIN at the line's height, outside the data
        # area, so it can never sit on a box whatever order the margins take.
        # The transform is data-y (the ref height) with axis-x just past 1.0.
        from matplotlib.transforms import blended_transform_factory
        tf = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.012, ref, f"interval-free\n{ref:.1f} km", transform=tf,
                va="center", ha="left", fontsize=7.5, color="0.30", zorder=6)

    box = ax.boxplot([data[t] for t in order], positions=range(1, len(order) + 1),
                     widths=0.62, patch_artist=True, whis=(5, 95), showfliers=False,
                     medianprops=dict(color="0.15", lw=1.3),
                     whiskerprops=dict(color="0.35", lw=0.9),
                     capprops=dict(color="0.35", lw=0.9),
                     boxprops=dict(lw=0.7, edgecolor="0.25"), zorder=4)
    for patch, c in zip(box["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.92)

    ax.set_ylabel("Windowed effective elastic thickness $T_e$ (km)")
    ax.set_xlim(0.4, len(order) + 0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, min(120, max(np.percentile(data[t], 95)
                                for t in order) * 1.08))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", right=True, direction="in")
    ax.tick_params(which="major", length=4.5)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(axis="y", which="major", lw=0.5, color="0.88")
    ax.set_axisbelow(True)

    # The reading of the figure, placed in the empty upper-left. Box height is
    # window sensitivity, so the message is the RANGE of heights: the most
    # window-sensitive margin swings several times as much as the least. The
    # numbers are the interquartile spread as a percentage of each margin's
    # own median, which is what Equation (windowmetric) formalises.
    rel = {t: 100 * (np.percentile(data[t], 75) - np.percentile(data[t], 25))
           / np.median(data[t]) for t in order}
    hi_t = max(rel, key=rel.get); lo_t = min(rel, key=rel.get)
    med_rel = float(np.median(list(rel.values())))
    note = (r"$\bf{Window\ sensitivity\ of\ the\ classical\ estimator}$"
            "\n"
            r"box height = interquartile spread of $T_e$ over the window grid"
            "\n"
            f"median margin: {med_rel:.0f} % of its own $T_e$   "
            f"(range {rel[lo_t]:.0f} % at {pretty(lo_t)} "
            f"to {rel[hi_t]:.0f} % at {pretty(hi_t)})")
    ax.text(0.015, 0.975, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="0.20", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", lw=0.5,
                      alpha=0.9))
    print(f"widest box {pretty(hi_t)} ({rel[hi_t]:.0f}%), "
          f"narrowest {pretty(lo_t)} ({rel[lo_t]:.0f}%)")

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, NumPy. Source: authors.")
    fig.text(0.005, -0.02, credit, fontsize=8, ha="left", va="top")

    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
