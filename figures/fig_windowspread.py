#!/usr/bin/env python3
"""fig_windowspread -- the window dependence, swept.

The central claim of the paper made visual: move the interval fitted on a fixed
profile, holding the profile and the physical model fixed, and the recovered
elastic thickness moves with it.

    (a) all profiles, seaward limit swept   (b) all profiles, landward limit
    (c) the least window-sensitive margin   (d) the most window-sensitive

Two quantities share each frame, on twin axes, in the manner of the ablation
figure this replaces:

    purple, left    median recovered thickness over the profiles contributing
                    at that window setting, with the interquartile range as a
                    band -- the estimate an analyst would report
    orange, right   the interquartile spread as a percentage of the median at
                    that setting -- the dependence itself, and the quantity the
                    abstract reports

A broken vertical line marks a conventional choice of the swept limit; a grey
horizontal line marks the interval-free estimate of Section 4, which has no
window and therefore no position on the abscissa.

Panels (c) and (d) are chosen by rule, not by eye: the margins with the
smallest and largest interquartile spread over windows with a seaward limit of
at least 200 km. Naming them in the titles keeps the selection auditable.

INPUT
-----
    --sweep     sweep_grid.csv   trench, profile, xl_km, xs_km, te_km
    --segments  runs/r02/segments.csv   for the interval-free reference

    python3 fig_windowspread.py --sweep sweep_grid.csv \\
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

C_TE, C_SPREAD, C_KEEP = "#7a4fa3", "#d95f02", "#444444"
MIN_XS = 200.0          # windows below this are the unusable regime
MIN_N = 12              # a setting with fewer profiles is not plotted


def register_nimbus():
    """Find Nimbus Sans wherever it is installed and report what resolved.

    The flat glob of the other scripts misses a Homebrew cask install, which
    puts the faces one directory deeper, so this searches recursively and
    accepts both .otf and .ttf. It then prints the family matplotlib actually
    selected: a silent fallback to DejaVu is why this figure did not match the
    others, and a printed name makes that visible instead of guessable.
    """
    from matplotlib import font_manager
    roots = ("/usr/share/fonts/opentype/urw-base35",
             "/usr/local/share/fonts/urw-base35",
             "/opt/homebrew/share/fonts", "/usr/share/fonts",
             "/Library/Fonts", "/System/Library/Fonts",
             os.path.expanduser("~/Library/Fonts"))
    found = 0
    for d in roots:
        if not os.path.isdir(d):
            continue
        for ext in ("otf", "ttf"):
            for f in glob.glob(os.path.join(d, "**", f"NimbusSans*.{ext}"),
                               recursive=True):
                try:
                    font_manager.fontManager.addfont(f)
                    found += 1
                except Exception:
                    pass
    return found


def set_rc():
    n = register_nimbus()
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Helvetica and Arial are present on macOS even without the URW
        # fonts, and both are metrically close to Nimbus Sans, so the figure
        # still matches the others. DejaVu is the last resort and is what the
        # warning below is for.
        "font.sans-serif": ["Nimbus Sans", "Nimbus Sans L", "Helvetica",
                            "Helvetica Neue", "Arial", "Liberation Sans"],
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular", "savefig.dpi": 600,
    })
    from matplotlib import font_manager
    resolved = font_manager.FontProperties(
        family=plt.rcParams["font.sans-serif"]).get_name()
    print(f"font: {n} Nimbus face(s) registered, matplotlib resolved "
          f"'{resolved}'")
    if "Nimbus" not in resolved and "Helvetica" not in resolved:
        print("  WARNING: neither Nimbus Sans nor Helvetica resolved, so this "
              "figure will not match the others. Install the URW base35 fonts "
              "(brew install --cask font-urw-base35).", file=sys.stderr)


def style(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, direction="in")
    ax.tick_params(which="major", length=4.0, pad=2)
    ax.tick_params(which="minor", length=2.2)
    ax.grid(which="major", lw=0.45, color="0.88")
    ax.grid(which="minor", lw=0.28, color="0.94")
    ax.set_axisbelow(True)


def curve(points):
    """points: {x: [te, ...]} -> x, median, q1, q3, spread per cent."""
    xs = sorted(k for k, v in points.items() if len(v) >= MIN_N)
    med, q1, q3, sp = [], [], [], []
    for x in xs:
        te = np.array([t for t in points[x] if np.isfinite(t) and t > 0])
        m = float(np.median(te))
        a, b = np.percentile(te, [25, 75])
        med.append(m); q1.append(a); q3.append(b)
        sp.append(100.0 * (b - a) / m if m > 0 else np.nan)
    return np.array(xs), np.array(med), np.array(q1), np.array(q3), np.array(sp)


def panel(ax, axr, pts, title, xlab, mark, ref, tag):
    if not pts:
        ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center",
                va="center", color="0.45")
        ax.set_title(title, fontsize=9.5, weight="bold", pad=3)
        return
    x, med, q1, q3, sp = curve(pts)
    ax.fill_between(x, q1, q3, color=C_TE, alpha=0.16, lw=0, zorder=2)
    ax.plot(x, med, color=C_TE, lw=1.3, marker="o", ms=3.0, mfc="white",
            mew=0.9, zorder=4)
    if ref is not None:
        ax.axhline(ref, color="0.45", lw=0.9, ls=(0, (1, 2)), zorder=3)
    ax.set_ylabel("$T_e$ (km)", color=C_TE)
    ax.tick_params(axis="y", colors=C_TE)
    ax.set_xlabel(xlab)
    ax.set_title(title, fontsize=9.5, weight="bold", pad=3)
    style(ax)

    axr.plot(x, sp, color=C_SPREAD, lw=1.1, ls="--", marker="s", ms=2.8,
             mfc="white", mew=0.8, zorder=3)
    axr.set_ylabel("spread (% of median)", color=C_SPREAD)
    axr.tick_params(axis="y", colors=C_SPREAD, which="both", direction="in",
                    labelsize=8)
    axr.set_ylim(bottom=0.0)

    if mark is not None:
        ax.axvline(mark, color=C_KEEP, lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.text(0.030, 0.055, f"({tag})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9.5, weight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75",
                      lw=0.4, alpha=0.92))
    print(f"  ({tag}) {title}: {len(x)} settings, Te median "
          f"{med.min():.1f}-{med.max():.1f} km, spread "
          f"{np.nanmin(sp):.0f}-{np.nanmax(sp):.0f} %")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="sweep_grid.csv")
    ap.add_argument("--segments", default="runs/r02/segments.csv")
    ap.add_argument("--mark-seaward", type=float, default=250.0)
    ap.add_argument("--mark-landward", type=float, default=0.0)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_windowspread")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.sweep)))
    print(f"{len(rows)} sweep points")

    ref = None
    if os.path.exists(a.segments):
        te = [float(r["te_km"]) for r in csv.DictReader(open(a.segments))]
        ref = float(np.median(te))
        print(f"interval-free reference: median {ref:.2f} km over {len(te)} segments")

    # pooled sweeps
    by_xs, by_xl = defaultdict(list), defaultdict(list)
    per_trench = defaultdict(lambda: defaultdict(list))
    pool = defaultdict(list)
    for r in rows:
        te = float(r["te_km"])
        if not np.isfinite(te) or te <= 0:
            continue
        xs, xl = float(r["xs_km"]), float(r["xl_km"])
        by_xs[xs].append(te)
        if xs >= MIN_XS:
            by_xl[xl].append(te)
            per_trench[r["trench"]][xs].append(te)
            pool[r["trench"]].append(te)

    # panels (c) and (d) chosen by rule
    rank = []
    for t, v in pool.items():
        te = np.array(v); m = float(np.median(te))
        q1, q3 = np.percentile(te, [25, 75])
        rank.append((100.0 * (q3 - q1) / m, t))
    rank.sort()
    lo_t, hi_t = rank[0][1], rank[-1][1]
    print(f"least sensitive {lo_t} ({rank[0][0]:.0f} %), "
          f"most sensitive {hi_t} ({rank[-1][0]:.0f} %)")

    def pretty(k):
        return " ".join(w.capitalize() for w in
                        k.replace("-", "_").split("_"))

    set_rc()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), constrained_layout=True)
    ax = axes.ravel()
    panel(ax[0], ax[0].twinx(), by_xs,
          "All margins, seaward limit swept",
          "seaward window limit (km)", a.mark_seaward, ref, "a")
    panel(ax[1], ax[1].twinx(), by_xl,
          "All margins, landward limit swept",
          "landward window limit (km)", a.mark_landward, ref, "b")
    panel(ax[2], ax[2].twinx(), per_trench[lo_t],
          f"Least window-sensitive: {pretty(lo_t)}",
          "seaward window limit (km)", a.mark_seaward, ref, "c")
    panel(ax[3], ax[3].twinx(), per_trench[hi_t],
          f"Most window-sensitive: {pretty(hi_t)}",
          "seaward window limit (km)", a.mark_seaward, ref, "d")

    # Short enough to sit inside the figure width. The full reading of each
    # element belongs in the caption, where a journal expects it, rather than
    # in a banner that overruns the panels.
    fig.text(0.5, 1.002,
             "Purple: $T_e$, median and interquartile range.   "
             "Orange: spread.   Dashed: conventional window.   "
             "Dotted: interval-free $T_e$.",
             ha="center", va="bottom", fontsize=8.0, color="0.30")
    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, NumPy. Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=8, ha="left", va="top")

    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
