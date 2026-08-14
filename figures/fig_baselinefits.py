#!/usr/bin/env python3
"""fig_profilefits -- observed and recovered bending profiles, one
representative segment for each of the twenty trenches.

WHICH CURVE IS DRAWN
--------------------
Without --pinn this figure shows the CLASSICAL windowed broken-plate fit
(morphometry.flexural_fit), i.e. the baseline the paper argues against, and the
Te printed in each cell is the classical value. The manuscript caption claims
the solid line is the physics-informed recovery; that claim is true only when
--pinn is supplied. With --pinn the recovered deflection becomes the solid
curve, the residual strip is measured against it, and the classical fit is
retained as a thin dashed comparison.

    --pinn expects a CSV with columns: trench,segment,x_km,depth_km
    one row per sample point, depth positive downward, x measured from the
    trench axis and positive seaward -- the same convention as the panels.

Twenty cells in a 5 x 4 grid, each split into a tall panel carrying the
observed section and the fitted broken-plate solution, and a short strip
beneath carrying the residual. The outer-rise crest is marked. The point of
the figure is comparability: identical axes on every cell, so a margin where
the elastic model fits badly is visible at a glance rather than having to be
looked up in a table.

The fit is morphometry.flexural_fit and the orientation rule is the one in
build_dataframe.py, so the panels show the project's own baseline rather than
a lookalike written for the figure.

    python3 fig_profilefits.py --dir profiles --outdir figures
    python3 fig_profilefits.py --dir profiles --window 320 --outdir figures

INPUT
-----
A directory of grdtrack tables, one per trench, named "<trench>.txt" exactly
as extract_profiles.sh writes them. The trench name is taken from the filename
and used to look up the landward azimuth, so the files must be named for the
keys in build_dataframe.LANDWARD_AZIMUTH. One segment per trench is shown: the
middle one by default, or the segment named in --index-file.
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
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from morphometry import flexural_fit, deflection, _axis, _smooth, CREST_MIN_X
from fig_windowproblem import read_grdtrack, orient_seaward

NCOL, NROW = 5, 4
C_OBS, C_FIT, C_RES = "0.35", "#c0392b", "#31708e"
C_PINN = "#1b3f8b"          # recovered deflection, when --pinn is supplied

# The manuscript calls this margin Vanuatu (Section 2.2); the profile files are
# named new_hebrides. Panels carry the manuscript's name so figure and text
# cannot disagree.
DISPLAY_NAME = {"new_hebrides": "Vanuatu"}

# The three groups of Section 2.2, in the order the text introduces them, so the
# panel sequence follows the narrative rather than the alphabet.
GROUPS = [
    ("northwestern and western Pacific",
     ["aleutian", "kuril-kamchatka", "japan", "izu-bonin", "mariana",
      "yap", "palau", "ryukyu", "manila", "philippine"]),
    ("southwestern Pacific",
     ["new_britain", "san_cristobal", "vityaz", "new_hebrides", "tonga",
      "kermadec", "hikurangi", "puysegur"]),
    ("eastern Pacific",
     ["middle_america", "peru-chile"]),
]
GROUP_ORDER = {t: i for i, (_, ts) in enumerate(GROUPS) for t in ts}


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
        "font.size": 8, "axes.labelsize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.labelpad": 2, "axes.linewidth": 0.7,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def style(ax, minor=True):
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=3.2, pad=1.5)
    ax.tick_params(which="minor", length=1.8)
    ax.grid(which="major", lw=0.4, color="0.88")
    ax.grid(which="minor", lw=0.25, color="0.94")
    ax.set_axisbelow(True)


def covers_data(ax, artist, fig):
    """Rule 1: count plotted points inside a text or legend box. A label that
    sits on the data is a defect, and on this figure the descending limb runs
    straight through the lower-left corner where the labels used to be fixed."""
    fig.canvas.draw()
    try:
        box = artist.get_window_extent(fig.canvas.get_renderer())
    except TypeError:
        box = artist.get_window_extent()
    n = 0
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        xy = ln.get_xydata()
        if len(xy) == 0:
            continue
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        p = ax.transData.transform(xy)
        n += int(((p[:, 0] >= box.x0) & (p[:, 0] <= box.x1) &
                  (p[:, 1] >= box.y0) & (p[:, 1] <= box.y1)).sum())
    return n


# Corner candidates for the two-line cell label, tried in order. Lower right is
# first because the deep half of the seaward axis is empty in every panel: the
# profile climbs from the trench floor at the left to the abyssal plain at the
# top, leaving that corner clear.
LABEL_CORNERS = [
    (0.955, 0.055, 0.145, "right", "bottom"),
    (0.045, 0.055, 0.145, "left", "bottom"),
    (0.045, 0.955, 0.865, "left", "top"),
    (0.955, 0.955, 0.865, "right", "top"),
]


def place_label(ax, fig, name, te_txt):
    """Trench name over the classical Te, placed in whichever corner of this
    cell is empty. Returns the number of covered points at the chosen corner."""
    best = None
    for x, y_te, y_name, ha, va in LABEL_CORNERS:
        t_name = ax.text(x, y_name, name, transform=ax.transAxes, ha=ha, va=va,
                         fontsize=7.5, weight="bold", zorder=7,
                         bbox=dict(boxstyle="round,pad=0.16", fc="white",
                                   ec="0.75", lw=0.4, alpha=0.9))
        t_te = ax.text(x, y_te, te_txt, transform=ax.transAxes, ha=ha, va=va,
                       fontsize=7, color=C_FIT, zorder=7,
                       bbox=dict(boxstyle="round,pad=0.10", fc="white",
                                 ec="none", alpha=0.75))
        hits = covers_data(ax, t_name, fig) + covers_data(ax, t_te, fig)
        if hits == 0:
            return 0
        if best is None or hits < best[0]:
            if best is not None:
                best[1].remove(); best[2].remove()
            best = (hits, t_name, t_te)
        else:
            t_name.remove(); t_te.remove()
    return best[0]


def one_trench(path, trench, index, window, side_opt="auto"):
    """Return everything one cell needs, or None if the segment is unusable."""
    segs = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                segs += 1
    idx = index if index is not None else max(0, segs // 2)
    dd, zz, lo, la = read_grdtrack(path, idx)
    x, w_m, side, _ = orient_seaward(dd, zz, lo, la, trench, side_opt)
    d = -w_m / 1000.0                       # elevation m -> depth km
    dx = float(np.median(np.diff(x)))
    ds = _smooth(np.where(np.isfinite(d), d, np.nanmedian(d)), dx)
    _, x0 = _axis(x, ds)
    f = flexural_fit(x, d, x0=x0, x_fit_max=window)
    if not f.get("valid"):
        return None

    m = (x >= x0) & (x <= x0 + window)
    xf = x[m]
    model = deflection(xf - x0, f["w0"], f["alpha"]) + f["d_r"]
    res = d[m] - model

    hi = min(x[-1] - 40.0, x0 + 240.0)
    win = (x > x0 + CREST_MIN_X) & (x <= hi)
    crest = None
    if win.sum() > 10:
        ib = int(np.nanargmin(ds[win]))
        crest = (float(x[win][ib]), float(ds[win][ib]))

    return dict(trench=trench, idx=idx, x=x, d=d, xf=xf, model=model,
                res=res, xres=xf, pinn=None,
                x0=x0, te=f["Te"], rms=f["rms"], crest=crest)


def attach_pinn(cells, path):
    """Overlay the recovered deflection and re-reference the residual to it.

    CSV columns: trench,segment,x_km,depth_km. Rows whose (trench, segment) do
    not match a plotted cell are ignored; cells with no matching rows keep the
    classical curve and are reported, because a figure that silently mixes the
    two would be worse than one that shows neither.
    """
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = (r["trench"].strip(), int(r["segment"]))
            rows.setdefault(k, []).append((float(r["x_km"]),
                                           float(r["depth_km"])))
    missing = []
    for c in cells:
        got = rows.get((c["trench"], c["idx"]))
        if not got:
            missing.append(c["trench"])
            continue
        got.sort()
        xp = np.array([g[0] for g in got])
        dp = np.array([g[1] for g in got])
        c["pinn"] = (xp, dp)
        m = (c["x"] >= xp.min()) & (c["x"] <= xp.max())
        c["xres"] = c["x"][m]
        c["res"] = c["d"][m] - np.interp(c["x"][m], xp, dp)
        c["rms"] = float(np.sqrt(np.nanmean(c["res"] ** 2)))
    if missing:
        print(f"note: no recovered deflection for {', '.join(missing)}; "
              f"those cells show the classical fit", file=sys.stderr)
    return len(cells) - len(missing)


def build(cells, window):
    set_rc()
    fig = plt.figure(figsize=(6.9, 8.7))
    outer = fig.add_gridspec(NROW, NCOL, hspace=0.26, wspace=0.09,
                             left=0.085, right=0.995, top=0.995,
                             bottom=0.120)          # room for label + key + credit
    has_pinn = any(c.get("pinn") is not None for c in cells)

    xlo, xhi = -80.0, 550.0
    dlo = min(np.nanmin(c["d"]) for c in cells)
    dhi = max(np.nanmax(c["d"]) for c in cells)
    rmax = max(float(np.nanmax(np.abs(c["res"]))) for c in cells)
    rmax = max(0.10, np.ceil(rmax * 20) / 20)

    hits = 0
    for k, c in enumerate(cells):
        r, col = divmod(k, NCOL)
        inner = outer[r, col].subgridspec(2, 1, height_ratios=[3.0, 1.0],
                                          hspace=0.04)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)

        ax.plot(c["x"], c["d"], color=C_OBS, lw=0.6, zorder=3)
        if c.get("pinn") is not None:
            # the recovered deflection is the emphasised curve; the classical
            # windowed fit stays as the thin dashed baseline it is
            ax.plot(c["xf"], c["model"], color=C_FIT, lw=0.7, ls=(0, (4, 2)),
                    zorder=4)
            ax.plot(c["pinn"][0], c["pinn"][1], color=C_PINN, lw=1.1, zorder=5)
        else:
            ax.plot(c["xf"], c["model"], color=C_FIT, lw=1.1, zorder=5)
        ax.axvline(c["x0"], color="0.65", lw=0.5, ls=(0, (3, 2)), zorder=2)
        if c["crest"]:
            # open circle, as the caption describes it
            ax.plot([c["crest"][0]], [c["crest"][1]], marker="o", ms=3.4,
                    mfc="none", mec="0.15", mew=0.8, ls="none", zorder=6)

        axr.axhline(0.0, color="0.55", lw=0.5, zorder=2)
        axr.plot(c["xres"], c["res"],
                 color=C_RES, lw=0.6, zorder=3)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(dhi, dlo)                       # depth positive downwards
        axr.set_ylim(-rmax, rmax)
        for a in (ax, axr):
            a.xaxis.set_major_locator(MultipleLocator(250))
            style(a)
        axr.yaxis.set_major_locator(MultipleLocator(rmax))
        plt.setp(ax.get_xticklabels(), visible=False)

        name = DISPLAY_NAME.get(
            c["trench"],
            c["trench"].replace("_", " ").replace("-", "\u2013").title())
        hits += place_label(ax, fig, name, f"$T_e$ {c['te']:.0f} km")

        if col == 0:
            ax.set_ylabel("Depth (km)")
            axr.set_ylabel("Res. (km)", fontsize=7.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(axr.get_yticklabels(), visible=False)
        if r != NROW - 1:
            plt.setp(axr.get_xticklabels(), visible=False)

    # One centred x label, in the reserved foot below the bottom tick labels.
    fig.text(0.54, 0.082, "Distance (km)", ha="center", va="top", fontsize=9)

    # A key, because five visual elements were previously explained only in the
    # caption: observed section, fit, crest, trench axis, residual.
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=C_OBS, lw=0.9, label="observed section")]
    if has_pinn:
        keys += [Line2D([], [], color=C_PINN, lw=1.3,
                        label="recovered deflection"),
                 Line2D([], [], color=C_FIT, lw=0.9, ls=(0, (4, 2)),
                        label=f"classical fit, {window:.0f} km window")]
    else:
        keys += [Line2D([], [], color=C_FIT, lw=1.3,
                        label=f"classical fit, {window:.0f} km window")]
    keys += [Line2D([], [], color="0.15", marker="o", ls="none", ms=3.4,
                    mfc="none", mew=0.8, label="outer-rise crest"),
             Line2D([], [], color="0.65", lw=0.7, ls=(0, (3, 2)),
                    label="trench axis"),
             Line2D([], [], color=C_RES, lw=0.9, label="residual")]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, 0.030),
               ncol=len(keys), frameon=False, fontsize=7.5, handlelength=1.9,
               columnspacing=1.1, handletextpad=0.5)

    v = sys.version_info
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "unavailable"
    credit = ("Software used for plotting figure: Python "
              f"{v.major}.{v.minor}.{v.micro}, "
              f"Matplotlib {matplotlib.__version__}, "
              f"SciPy {scipy_v}, NumPy {np.__version__}. "
              "Data: GEBCO bathymetry. Source: authors.")
    fig.text(0.005, 0.012, credit, fontsize=8, ha="left", va="bottom")

    if hits:
        print(f"WARNING: cell labels cover {hits} data points -- "
              f"fix before publishing")
    else:
        print("overlap check -- cell labels cover 0 data points")
    print(f"window {window:.0f} km;  Te from "
          f"{min(c['te'] for c in cells):.0f} to "
          f"{max(c['te'] for c in cells):.0f} km;  worst RMS "
          f"{max(c['rms'] for c in cells):.3f} km "
          f"({max(cells, key=lambda c: c['rms'])['trench']})")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of <trench>.txt")
    ap.add_argument("--window", type=float, default=250.0,
                    help="seaward fit limit, km (flexural_fit default 250)")
    ap.add_argument("--index-file", help="CSV trench,segment to pin the choice")
    ap.add_argument("--pinn", help="CSV trench,segment,x_km,depth_km of the "
                    "recovered deflection; without it the panels show the "
                    "classical windowed fit only")
    ap.add_argument("--order", choices=("group", "alpha"), default="group",
                    help="panel order: the three groups of Section 2.2 "
                         "(default) or alphabetical")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_profilefits")
    a = ap.parse_args()

    pinned = {}
    if a.index_file:
        with open(a.index_file) as fh:
            for row in csv.DictReader(fh):
                pinned[row["trench"].strip()] = int(row["segment"])

    files = sorted(glob.glob(os.path.join(a.dir, "*.txt")))
    if not files:
        raise SystemExit(f"no .txt profiles in {a.dir}")
    cells = []
    for f in files:
        t = os.path.splitext(os.path.basename(f))[0]
        c = one_trench(f, t, pinned.get(t), a.window)
        if c is None:
            print(f"note: {t} produced no valid fit; cell omitted",
                  file=sys.stderr)
            continue
        cells.append(c)

    if a.order == "group":
        unknown = [c["trench"] for c in cells if c["trench"] not in GROUP_ORDER]
        if unknown:
            print(f"note: not in any group, placed last: {', '.join(unknown)}",
                  file=sys.stderr)
        cells.sort(key=lambda c: (GROUP_ORDER.get(c["trench"], 99),
                                  c["trench"]))

    if a.pinn:
        n_ok = attach_pinn(cells, a.pinn)
        print(f"recovered deflection attached to {n_ok} of {len(cells)} cells")
    else:
        print("no --pinn given: panels show the CLASSICAL windowed fit, "
              "not the physics-informed recovery")

    for c in cells:
        print(f"  {c['trench']:16s} segment {c['idx']:3d}  "
              f"Te {c['te']:5.1f} km  rms {c['rms']:.3f} km")

    n = NROW * NCOL
    if len(cells) != n:
        print(f"note: {len(cells)} trenches available, grid holds {n}",
              file=sys.stderr)
    cells = cells[:n]

    fig = build(cells, a.window)
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
