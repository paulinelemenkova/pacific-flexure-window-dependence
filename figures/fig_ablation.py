#!/usr/bin/env python3
"""fig_ablation -- sensitivity of the recovered elastic thickness and of the
physics residual to four choices in the inversion.

    (a) weight on the physics term      (b) hidden units per layer
    (c) collocation-point density       (d) profile spacing

Each panel sweeps one setting with the others held at the retained
configuration, and plots two quantities on twin axes: the recovered elastic
thickness (left, purple) and the physics residual at convergence (right,
orange). The axis labels and tick colours match the curves, so the twin scale
cannot be misread -- which is the usual objection to twin axes and the reason
they are avoided in fig_convergence, where four curves share one frame.

A vertical line marks the configuration retained for the results. The figure
answers one question: does the recovered thickness depend on choices that are
not physical? A flat purple curve either side of that line is the answer the
paper needs; a sloping one is a finding that belongs in the Discussion.

INPUT
-----
CSV with a header row, one row per swept point:

    parameter,value,te_km,residual[,te_lo,te_hi,retained]

`parameter` is one of physics_weight, hidden_units, collocation, spacing.
`retained` is 1 on the single row per parameter that was kept, blank or 0
otherwise. `te_lo`/`te_hi` are optional; supply them and the spread is drawn
as a band, omit them and only the central curve appears.

    python3 fig_ablation.py --table ablation.csv --outdir figures
    python3 fig_ablation.py --selftest         # LAYOUT CHECK ONLY
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, LogLocator

SEED = 0
C_TE, C_RES, C_KEEP = "#7a4fa3", "#d95f02", "#444444"

# key -> (panel title, x label, logarithmic abscissa)
PANELS = [
    ("physics_weight", "Weight on the physics term",
     "$\\lambda_p$ / $\\lambda_d$", True),
    ("hidden_units", "Network width", "hidden units per layer", True),
    ("collocation", "Collocation density",
     "collocation points per profile", False),
    ("spacing", "Profile spacing", "along-strike spacing (km)", False),
]


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
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def style(ax, logx=False, logy=False):
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(LogLocator(base=10.0,
                                              subs=tuple(np.arange(2, 10)),
                                              numticks=100))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if logy:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, direction="in")
    ax.tick_params(which="major", length=4.0, pad=2)
    ax.tick_params(which="minor", length=2.2)


def f(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


def read_table(path):
    by = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r.get("parameter") or "").strip()
            if not key:
                continue
            by.setdefault(key, []).append(
                dict(x=f(r.get("value")), te=f(r.get("te_km")),
                     res=f(r.get("residual")), lo=f(r.get("te_lo")),
                     hi=f(r.get("te_hi")),
                     keep=(r.get("retained") or "").strip() in ("1", "true",
                                                                "True", "yes")))
    for k in by:
        by[k].sort(key=lambda d: d["x"])
    return by


def synthetic_table():
    """Plausible ablation. LAYOUT TESTING ONLY -- never published."""
    rng = np.random.default_rng(SEED)
    out = {}
    grids = {"physics_weight": np.logspace(-2, 2, 9),
             "hidden_units": np.array([8., 16., 32., 64., 128., 256.]),
             "collocation": np.arange(100., 901., 100.),
             "spacing": np.array([5., 10., 15., 20., 30., 40.])}
    keep = {"physics_weight": 1.0, "hidden_units": 64.0,
            "collocation": 500.0, "spacing": 10.0}
    for k, g in grids.items():
        rows = []
        for x in g:
            if k == "physics_weight":
                te = 31.0 + 2.4 * np.tanh(np.log10(x))
                res = 4.0e-3 / (1.0 + 6.0 * x) + 4e-5
            elif k == "hidden_units":
                te = 31.0 - 3.0 * np.exp(-x / 40.0)
                res = 6.0e-4 * np.exp(-x / 55.0) + 6e-5
            elif k == "collocation":
                te = 31.0 - 1.6 * np.exp(-x / 220.0)
                res = 5.0e-4 * np.exp(-x / 260.0) + 5e-5
            else:
                te = 31.0 - 0.02 * (x - 10.0)
                res = 8.0e-5 + 3.0e-6 * x
            te += 0.10 * rng.standard_normal()
            rows.append(dict(x=float(x), te=float(te),
                             res=float(res * (1 + 0.06 * rng.standard_normal())),
                             lo=float(te - 1.1), hi=float(te + 1.1),
                             keep=bool(np.isclose(x, keep[k]))))
        out[k] = rows
    return out


def build(by, synthetic=False):
    set_rc()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), constrained_layout=True)

    te_all = [d["te"] for rows in by.values() for d in rows
              if np.isfinite(d["te"])]
    lo = min(te_all) - 1.5
    hi = max(te_all) + 1.5

    for ax, (key, title, xlab, logx), tag in zip(axes.ravel(), PANELS, "abcd"):
        rows = by.get(key, [])
        if not rows:
            ax.text(0.5, 0.5, f"no rows for\n{key}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.45")
            ax.set_title(title, fontsize=9.5, weight="bold")
            style(ax, logx)
            continue
        x = np.array([d["x"] for d in rows])
        te = np.array([d["te"] for d in rows])
        res = np.array([d["res"] for d in rows])

        if all(np.isfinite(d["lo"]) and np.isfinite(d["hi"]) for d in rows):
            ax.fill_between(x, [d["lo"] for d in rows], [d["hi"] for d in rows],
                            color=C_TE, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, te, color=C_TE, lw=1.3, marker="o", ms=3.4, mfc="white",
                mew=0.9, zorder=4)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("$T_e$ (km)", color=C_TE)
        ax.tick_params(axis="y", colors=C_TE)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=9.5, weight="bold", pad=3)
        style(ax, logx)
        ax.grid(which="major", lw=0.45, color="0.88")
        ax.grid(which="minor", lw=0.28, color="0.94")
        ax.set_axisbelow(True)

        axr = ax.twinx()
        axr.plot(x, res, color=C_RES, lw=1.1, ls="--", marker="s", ms=3.0,
                 mfc="white", mew=0.8, zorder=3)
        axr.set_yscale("log")
        axr.set_ylabel("physics residual", color=C_RES)
        axr.tick_params(axis="y", colors=C_RES, which="both", direction="in",
                        labelsize=8)

        kept = [d for d in rows if d["keep"]]
        if kept:
            xk = kept[0]["x"]
            ax.axvline(xk, color=C_KEEP, lw=0.9, ls=(0, (5, 3)), zorder=1)
            ax.plot([xk], [kept[0]["te"]], marker="o", ms=6.5, mfc="none",
                    mec=C_KEEP, mew=1.3, zorder=6)

        ax.text(0.030, 0.055, f"({tag})", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9.5, weight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.75",
                          lw=0.4, alpha=0.92))

        span = np.nanmax(te) - np.nanmin(te)
        print(f"{key:16s}: Te varies by {span:.2f} km over the sweep"
              + (f";  retained {xk:g}" if kept else ";  no retained row"))

    fig.text(0.5, 1.005, "Purple, left axis: recovered elastic thickness.   "
             "Orange, right axis: physics residual.   "
             "Broken line: retained configuration.",
             ha="center", va="bottom", fontsize=8.5, color="0.30")

    if synthetic:
        axes[0, 0].text(0.5, 0.5, "SYNTHETIC", transform=axes[0, 0].transAxes,
                        ha="center", va="center", fontsize=15, color="0.80",
                        rotation=22, alpha=0.6, zorder=8)

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, NumPy. Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=8, ha="left", va="top")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_ablation")
    a = ap.parse_args()

    if a.selftest:
        fig = build(synthetic_table(), synthetic=True)
        name = a.name + "_SELFTEST"
    else:
        if not a.table:
            ap.error("--table is required (or use --selftest)")
        fig = build(read_table(a.table))
        name = a.name

    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
