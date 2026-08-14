#!/usr/bin/env python3
"""fig_profilefits -- observed and physics-informed bending profiles, one
representative segment for each of the twenty trenches.

Twenty cells in a 5 x 4 grid. Each cell carries a tall panel with the observed
deflection and the deflection recovered by the physics-informed inversion, and
a short strip beneath it carrying the residual against the scatter of the
individual profiles that were stacked into the segment. Identical axes on every
cell, so a margin the network fits badly is visible at a glance rather than
having to be looked up in a table.

INPUT
-----
--deflection  required; runs/r02/deflection.csv, exported from checkpoint.pt.
              Columns: segment, trench, x_km, w_obs_m, w_pinn_m. Deflection is
              in metres, positive downward, zero in the far field, and x runs
              from the trench axis seaward -- there is no landward side. The
              file is the only source of plotted data: no grdtrack table is
              read, and no classical fit is computed, so the panels show the
              inversion the caption claims and nothing else.

--segments    optional; runs/r02/segments.csv. Only one column is read,
              mad_profiles_m: the median absolute deviation of the individual
              profiles that were stacked into each segment. It is drawn as a
              band about zero in the residual strip, which is the reference the
              residual has to be judged against -- a misfit inside the band is
              within the scatter of the observations themselves. Without this
              file the strips are drawn with no band and the script says so.

SEGMENT CHOICE
--------------
One segment per trench. RMS of (w_obs - w_pinn) is computed for every segment;
for each trench the segment whose RMS lies nearest the median RMS of that
trench is drawn. Where a trench has an even number of segments the lower of the
two central values is used, so the rule always names a segment that exists and
the choice cannot turn on floating-point noise. The rule is deterministic and
belongs in the caption; --index-file overrides it segment by segment.

    python3 fig_profilefits.py --deflection runs/r02/deflection.csv \
        --segments runs/r02/segments.csv --outdir figures

SCALES
------
All twenty cells share one deflection scale and one residual scale, because the
figure exists to be read across margins. Both scales are set from robust
quantiles of the plotted data rather than from its extremes: a handful of
points otherwise stretch the axes until every shallow margin is a flat line,
which is the failure mode this version was written to avoid. Anything outside
the frame is clipped and flagged by a caret on the axis edge, and the count is
printed. --wlim / --rlim set the limits by hand, --yscale free abandons the
shared scale altogether (state it in the caption if you use it).

ENVIRONMENT
-----------
Render in `flexure` (Python 3.12.13, Matplotlib 3.11.1, NumPy 2.4.6), so the
credit line this script writes matches the versions stated in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import CARETDOWNBASE, CARETUPBASE
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

NCOL, NROW = 5, 4

# The palette of fig_convergence.py, so the two figures read as one set.
# Observed sections are black: they are the measurement, and black is the one
# choice that needs no defence. royalblue2 carries the recovery, red3 the
# residual; the two never share a panel, and both stay separable from black in
# greyscale and under deuteranopia.
C_OBS = "#000000"
C_PINN = "#436EEE"          # royalblue2
C_RES = "#CD0000"           # red3
C_MAD = "0.86"              # profile-scatter band: context ink, not data
LW_MAIN, LW_SERIES = 1.1, 1.1

# Panel order is geographic, anticlockwise round the Pacific from the north,
# and matches the order of Table 2. A trench absent from the CSV leaves its
# cell blank rather than shifting every panel after it.
TRENCH_ORDER = [
    "aleutian", "kuril-kamchatka", "japan", "izu-bonin", "mariana",
    "yap", "palau", "ryukyu", "manila", "philippine",
    "new_britain", "san_cristobal", "vityaz", "new_hebrides", "tonga",
    "kermadec", "hikurangi", "puysegur", "middle_america", "peru-chile",
]

# The manuscript calls this margin Vanuatu (Section 2.2); the run names it
# new_hebrides. Panels carry the manuscript's name so figure and text cannot
# disagree.
DISPLAY_NAME = {
    "aleutian": "Aleutian",
    "kuril-kamchatka": "Kuril\u2013Kamchatka",
    "kuril_kamchatka": "Kuril\u2013Kamchatka",
    "japan": "Japan",
    "izu-bonin": "Izu\u2013Bonin",
    "izu_bonin": "Izu\u2013Bonin",
    "mariana": "Mariana",
    "yap": "Yap",
    "palau": "Palau",
    "ryukyu": "Ryukyu",
    "manila": "Manila",
    "philippine": "Philippine",
    "new_britain": "New Britain",
    "san_cristobal": "San Cristobal",
    "vityaz": "Vityaz",
    "new_hebrides": "Vanuatu",
    "tonga": "Tonga",
    "kermadec": "Kermadec",
    "hikurangi": "Hikurangi",
    "puysegur": "Puysegur",
    "middle_america": "Middle America",
    "peru-chile": "Peru\u2013Chile",
    "peru_chile": "Peru\u2013Chile",
}


# ----------------------------------------------------------------- house style
def set_rc(strict_font=True):
    """House rcParams. Nothing here is a matplotlib default."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica"],   # never DejaVu
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.labelpad": 2.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
        "lines.solid_capstyle": "round",
        "savefig.transparent": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    if strict_font:
        from matplotlib.font_manager import FontProperties, findfont
        fp = FontProperties(family=plt.rcParams["font.sans-serif"])
        if "dejavu" in findfont(fp).lower():
            sys.exit("Nimbus Sans not found and DejaVu is forbidden; install "
                     "fonts-urw-base35, or rerun with --font-relax.")


def style(ax, yminor=None, grid_axis="both"):
    """Ticks, grid and spines, applied identically to every panel."""
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    if yminor is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(yminor))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", axis=grid_axis,
            color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


# ------------------------------------------------------------------- data side
def read_deflection(path):
    """Return {segment: dict} with x, w_obs, w_pinn, res, rms, trench."""
    need = ("segment", "trench", "x_km", "w_obs_m", "w_pinn_m")
    rows = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in need if c not in (rd.fieldnames or [])]
        if missing:
            sys.exit(f"{path}: missing column(s) {', '.join(missing)}")
        for r in rd:
            seg = r["segment"].strip()
            d = rows.setdefault(seg, {"segment": seg,
                                      "trench": r["trench"].strip(),
                                      "x": [], "w_obs": [], "w_pinn": []})
            d["x"].append(float(r["x_km"]))
            d["w_obs"].append(float(r["w_obs_m"]))
            d["w_pinn"].append(float(r["w_pinn_m"]))

    out = {}
    for seg, d in rows.items():
        x = np.asarray(d["x"], float)
        o = np.argsort(x)                       # never trust file order
        d["x"] = x[o]
        d["w_obs"] = np.asarray(d["w_obs"], float)[o]
        d["w_pinn"] = np.asarray(d["w_pinn"], float)[o]
        d["res"] = d["w_obs"] - d["w_pinn"]
        good = np.isfinite(d["res"])
        if good.sum() < 10:
            print(f"note: {seg} has fewer than ten finite samples; skipped",
                  file=sys.stderr)
            continue
        d["rms"] = float(np.sqrt(np.mean(d["res"][good] ** 2)))
        out[seg] = d
    if not out:
        sys.exit(f"{path}: no usable segments")
    return out


def read_mad(path):
    """Return {segment: mad_profiles_m}; empty dict if unavailable."""
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"note: {path} not found; residual strips drawn without the "
              f"profile-scatter band", file=sys.stderr)
        return {}
    mad = {}
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if "mad_profiles_m" not in (rd.fieldnames or []):
            print(f"note: {path} has no mad_profiles_m column; bands omitted",
                  file=sys.stderr)
            return {}
        for r in rd:
            try:
                v = float(r["mad_profiles_m"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v > 0:
                mad[r["segment"].strip()] = v
    return mad


def choose(segments, pinned):
    """One segment per trench: RMS nearest that trench's median RMS.

    Even counts take the lower of the two central values, so the rule always
    names a segment that exists and cannot turn on floating-point noise.
    """
    by_trench = {}
    for seg, d in segments.items():
        by_trench.setdefault(d["trench"], []).append(seg)

    picked = OrderedDict()
    for trench, segs in by_trench.items():
        if trench in pinned:
            want = pinned[trench]
            if want in segments:
                picked[trench] = want
                continue
            print(f"note: pinned segment {want} not in the CSV; falling back "
                  f"to the median rule for {trench}", file=sys.stderr)
        ordered = sorted(segs, key=lambda s: (segments[s]["rms"], s))
        picked[trench] = ordered[(len(ordered) - 1) // 2]
    return picked


def nice_step(span, target=4):
    """A round tick interval giving roughly *target* major ticks over *span*."""
    if span <= 0:
        return 1.0
    raw = span / max(target, 1)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return m * mag
    return 10.0 * mag


def shared_limits(cells, mad, q, qr):
    """Robust common scales, and the residual half-range.

    Extremes are not used: the quantile pair *q* trims the few points that
    would otherwise flatten every shallow margin. The residual half-range is
    also held above the largest profile-scatter band, so the band it is judged
    against is always inside the frame.
    """
    w = np.concatenate([np.concatenate([c["w_obs"], c["w_pinn"]])
                        for c in cells])
    w = w[np.isfinite(w)]
    lo, hi = np.percentile(w, [100.0 - q, q])
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    r = np.concatenate([c["res"] for c in cells])
    r = np.abs(r[np.isfinite(r)])
    rm = float(np.percentile(r, qr))
    if mad:
        rm = max(rm, 1.25 * max(mad.get(c["segment"], 0.0) for c in cells))
    rm = nice_step(2 * rm, target=2)
    return float(lo), float(hi), float(rm)


# ------------------------------------------------------------- drawing helpers
def mark_clipped(ax, x, y, lo, hi, color):
    """Caret on the frame wherever a series leaves it. Returns the count."""
    n = 0
    for sel, marker, edge in ((y < lo, CARETUPBASE, lo),
                              (y > hi, CARETDOWNBASE, hi)):
        sel = sel & np.isfinite(y)
        if not sel.any():
            continue
        n += int(sel.sum())
        xs = x[sel]
        keep = np.concatenate(([True], np.diff(xs) > 60.0))   # thin them out
        ax.plot(xs[keep], np.full(keep.sum(), edge), linestyle="none",
                marker=marker, ms=2.8, color=color, clip_on=False, zorder=6)
    return n


def covers_data(bb, ax):
    """Data points under a display-space bbox. Must be zero (house rule 10)."""
    hits = 0

    def count(xy):
        p = ax.transData.transform(np.asarray(xy))
        return int(((p[:, 0] >= bb.x0) & (p[:, 0] <= bb.x1) &
                    (p[:, 1] >= bb.y0) & (p[:, 1] <= bb.y1)).sum())

    for ln in ax.get_lines():
        d = np.column_stack(ln.get_data())
        if d.shape[0] > 2:
            hits += count(d)
    return hits


def place_label(ax, fig, name, sub):
    """Put the bold trench name over its regular RMS where they cover no data.

    Two stacked Text objects -- the name in bold, the RMS below it in regular
    weight, matching the baseline figure -- are placed as a block at four
    candidate corners (upper pair, then lower). The block's combined bounding
    box is tested against the data; the first clear corner is kept and any
    provisional pair from an earlier corner removed, so a cell never carries
    two labels. Returns the data-hit count of the chosen corner (0 if clear).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def draw_block(x, y, ha, va):
        # When name is empty the block is the sub line alone -- the name now
        # lives in the panel title, so no blank bold row is drawn.
        if not name:
            ts = ax.text(x, y, sub, transform=ax.transAxes, ha=ha, va=va,
                         fontsize=8, zorder=7)
            return (ts,)
        # name and sub as two lines; va controls whether the block grows down
        # (top anchor) or up (bottom anchor) from (x, y).
        if va == "top":
            tn = ax.text(x, y, name, transform=ax.transAxes, ha=ha, va="top",
                         fontsize=8, fontweight="bold", zorder=7)
            fig.canvas.draw()
            h = tn.get_window_extent(renderer=r).height / \
                ax.get_window_extent(renderer=r).height
            ts = ax.text(x, y - h - 0.015, sub, transform=ax.transAxes,
                         ha=ha, va="top", fontsize=8, zorder=7)
        else:
            ts = ax.text(x, y, sub, transform=ax.transAxes, ha=ha, va="bottom",
                         fontsize=8, zorder=7)
            fig.canvas.draw()
            h = ts.get_window_extent(renderer=r).height / \
                ax.get_window_extent(renderer=r).height
            tn = ax.text(x, y + h + 0.015, name, transform=ax.transAxes,
                         ha=ha, va="bottom", fontsize=8, fontweight="bold",
                         zorder=7)
        return tn, ts

    def block_hits(pair):
        fig.canvas.draw()
        return sum(covers_data(t.get_window_extent(renderer=r), ax)
                   for t in pair)

    best, best_hits = None, None
    for x, y, ha, va in ((0.03, 0.95, "left", "top"),
                         (0.97, 0.95, "right", "top"),
                         (0.03, 0.06, "left", "bottom"),
                         (0.97, 0.06, "right", "bottom")):
        pair = draw_block(x, y, ha, va)
        hits = block_hits(pair)
        if hits == 0:
            if best is not None:
                for t in best:
                    t.remove()
            return 0
        if best_hits is None or hits < best_hits:
            if best is not None:
                for t in best:
                    t.remove()
            best, best_hits = pair, hits
        else:
            for t in pair:
                t.remove()
    return best_hits


# ------------------------------------------------------------------- the figure
def build(cells, mad, wlim, rlim, shared, q, qr):
    if shared:
        lo, hi, rm = shared_limits(cells, mad, q, qr)
    else:
        lo = hi = rm = None
    if wlim:
        lo, hi = wlim
    if rlim:
        rm = rlim

    fig = plt.figure(figsize=(7.48, 8.6))
    # Explicit margins: constrained_layout floats each residual strip away from
    # its own panel, which makes the pairing ambiguous. The bottom margin is
    # deep enough to hold the x label and the key beneath it, so the key cannot
    # collide with the tick labels the way it did before.
    outer = fig.add_gridspec(NROW, NCOL, left=0.075, right=0.985,
                             top=0.972, bottom=0.105,
                             wspace=0.10, hspace=0.26)

    by_trench = {c["trench"]: c for c in cells}
    order = [t for t in TRENCH_ORDER if t in by_trench]
    order += sorted(t for t in by_trench if t not in TRENCH_ORDER)

    clipped, hits, bottom_axes = 0, 0, []
    for k, trench in enumerate(order[:NROW * NCOL]):
        c = by_trench[trench]
        row, col = divmod(k, NCOL)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3.0, 1.0],
                                            hspace=0.06)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)

        if not shared:
            w = np.concatenate([c["w_obs"], c["w_pinn"]])
            lo, hi = float(np.nanmin(w)), float(np.nanmax(w))
            pad = 0.08 * (hi - lo) or 1.0
            lo, hi = lo - pad, hi + pad
            rm = nice_step(2.4 * float(np.nanmax(np.abs(c["res"]))), target=2)

        ax.plot(c["x"], c["w_obs"], color=C_OBS, lw=LW_SERIES, zorder=3)
        ax.plot(c["x"], c["w_pinn"], color=C_PINN, lw=LW_MAIN, zorder=4)

        m = mad.get(c["segment"])
        if m is not None:
            axr.axhspan(-m, m, color=C_MAD, lw=0, zorder=1)
        axr.axhline(0.0, color="0.55", lw=0.7, zorder=2)
        axr.plot(c["x"], c["res"], color=C_RES, lw=LW_SERIES, zorder=3)

        ax.set_xlim(0.0, 550.0)
        ax.set_ylim(hi, lo)                      # deflection positive downward
        axr.set_ylim(-rm, rm)
        if shared:
            clipped += mark_clipped(ax, c["x"], c["w_obs"], lo, hi, C_OBS)
            clipped += mark_clipped(ax, c["x"], c["w_pinn"], lo, hi, C_PINN)
            clipped += mark_clipped(axr, c["x"], c["res"], -rm, rm, C_RES)

        wstep = nice_step(hi - lo, target=5)
        for a in (ax, axr):
            a.xaxis.set_major_locator(MultipleLocator(250))
        style(ax, yminor=wstep / 2.0)
        ax.yaxis.set_major_locator(MultipleLocator(wstep))
        style(axr, yminor=rm / 2.0)
        axr.yaxis.set_major_locator(MultipleLocator(rm))
        plt.setp(ax.get_xticklabels(), visible=False)

        name = DISPLAY_NAME.get(
            c["trench"],
            c["trench"].replace("_", " ").replace("-", "\u2013").title())
        # Trench name above the panel, top-left, like the panel tags of the
        # window-sensitivity figure. Placed with ax.text at y=1.02 in axes
        # coordinates rather than set_title, which does not render on this
        # Matplotlib build. The RMS stays inside, where it covers no data.
        ax.text(0.0, 1.02, name, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8, fontweight="bold", zorder=7)
        hits += place_label(ax, fig, "", f"RMS {c['rms']:.0f} m")

        if col == 0:
            ax.set_ylabel("Deflection $w$ (m)")
            axr.set_ylabel("Resid. (m)", fontsize=7.5)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(axr.get_yticklabels(), visible=False)
        if row == NROW - 1:
            bottom_axes.append(axr)
        else:
            plt.setp(axr.get_xticklabels(), visible=False)

    # One x label for the whole figure. Twenty of them, one per bottom cell,
    # ran into each other; the quantity is the same on every panel, so one
    # centred label carries it.
    if bottom_axes:
        box = bottom_axes[0].get_position()
        fig.text(0.5, max(0.030, box.y0 - 0.038),
                 "Distance from trench axis (km)", ha="center", va="top",
                 fontsize=9)

    keys = [Line2D([], [], color=C_OBS, lw=LW_SERIES,
                   label="observed deflection"),
            Line2D([], [], color=C_PINN, lw=LW_MAIN,
                   label="physics-informed recovery"),
            Line2D([], [], color=C_RES, lw=LW_SERIES,
                   label="residual, observed $-$ recovered")]
    if mad:
        keys.append(plt.Rectangle((0, 0), 1, 1, color=C_MAD,
                                  label="scatter of stacked profiles (MAD)"))

    # The key sits below the bottom row, and its top edge is measured against
    # the lowest x-axis label rather than guessed, so the two cannot collide.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ylow = min((a.get_window_extent(renderer=r).y0 -
                a.xaxis.get_tightbbox(r).height) if a.xaxis.get_tightbbox(r)
               else a.get_window_extent(renderer=r).y0
               for a in bottom_axes) if bottom_axes else 0.0
    y = max(0.008, (ylow / fig.get_window_extent().height) - 0.072)
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(keys), frameon=False, handlelength=2.2,
               columnspacing=1.6, handletextpad=0.6, borderpad=0.0)

    credit = ("Software used for plotting figure: Python %s, Matplotlib %s, "
              "NumPy %s. Data: run r02 deflection export. Source: authors."
              % (".".join(map(str, sys.version_info[:3])),
                 matplotlib.__version__, np.__version__))
    fig.text(0.5, 0.004, credit, ha="center", va="bottom", fontsize=8,
             color="0.30")
    return fig, clipped, hits, (lo, hi, rm)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deflection", required=True,
                    help="runs/r02/deflection.csv")
    ap.add_argument("--segments", default=None,
                    help="runs/r02/segments.csv, for mad_profiles_m")
    ap.add_argument("--index-file", default=None,
                    help="CSV with columns trench,segment to pin the choice")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--name", default="fig_profilefits")
    ap.add_argument("--yscale", choices=("shared", "free"), default="shared")
    ap.add_argument("--quantile", type=float, default=99.0,
                    help="upper quantile setting the shared deflection scale")
    ap.add_argument("--rquantile", type=float, default=95.0,
                    help="upper quantile setting the shared residual scale; "
                         "kept below --quantile so that one badly fitted "
                         "margin cannot flatten every other strip")
    ap.add_argument("--wlim", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"), help="deflection limits, metres")
    ap.add_argument("--rlim", type=float, default=None,
                    help="residual half-range, metres")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--font-relax", action="store_true",
                    help="allow a font fallback instead of failing")
    a = ap.parse_args()

    set_rc(strict_font=not a.font_relax)

    segments = read_deflection(a.deflection)
    mad = read_mad(a.segments)

    pinned = {}
    if a.index_file:
        with open(a.index_file, newline="") as fh:
            for row in csv.DictReader(fh):
                pinned[row["trench"].strip()] = row["segment"].strip()

    picked = choose(segments, pinned)
    cells = [segments[s] for s in picked.values()]
    cells.sort(key=lambda c: (TRENCH_ORDER.index(c["trench"])
                              if c["trench"] in TRENCH_ORDER else 99,
                              c["trench"]))

    n = NROW * NCOL
    if len(cells) != n:
        print(f"note: {len(cells)} trenches available, grid holds {n}",
              file=sys.stderr)

    fig, clipped, hits, (lo, hi, rm) = build(
        cells, mad, a.wlim, a.rlim, a.yscale == "shared", a.quantile,
        a.rquantile)

    worst = max(cells, key=lambda c: c["rms"])
    best = min(cells, key=lambda c: c["rms"])
    print(f"segments drawn: {len(cells)} of {len(segments)}")
    print(f"RMS range: {best['rms']:.0f} m ({best['segment']}) to "
          f"{worst['rms']:.0f} m ({worst['segment']})")
    print(f"deflection scale: {lo:.0f} to {hi:.0f} m; "
          f"residual half-range: {rm:.0f} m")
    print(f"clipped samples: {clipped}")
    print(f"label-over-data hits: {hits}"
          + ("" if hits == 0 else "  <-- inspect before submitting"))
    if mad:
        inside = sum(1 for c in cells
                     if c["segment"] in mad
                     and c["rms"] <= mad[c["segment"]])
        print(f"segments whose RMS is inside the profile scatter: "
              f"{inside} of {len(cells)}")

    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(a.outdir, f"{a.name}.{ext}")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02,
                    dpi=a.dpi if ext == "png" else None)
        print("wrote", out)

    print("\nsegments drawn, for the caption and the reproducibility note:")
    for trench, seg in picked.items():
        print(f"  {DISPLAY_NAME.get(trench, trench):<16} {seg:<20} "
              f"RMS {segments[seg]['rms']:.0f} m")


if __name__ == "__main__":
    main()
