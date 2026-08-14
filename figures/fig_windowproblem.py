#!/usr/bin/env python3
"""fig_windowproblem -- the motivating figure of the window-dependence paper.

Shows, for ONE observed bending profile, that the effective elastic thickness
recovered by classical windowed least-squares inversion depends on the analysis
window the analyst chooses. Main panel: the observed deflection with four
alternative windows and the fit obtained inside each. Inset: recovered elastic
thickness as a continuous function of the seaward window limit, with the four
choices marked.

    Panel (a)  observed profile + 4 windowed fits
    Inset      Te(x_seaward) sweep, 4 choices marked

INPUT (required, no default data is shipped)
--------------------------------------------
Either of two forms:

1. A GMT multisegment table as written by `extract_profiles.sh`, i.e. the
   output of `grdtrack -C1100k/1k+v`, five whitespace-delimited columns

       lon  lat  dist  azimuth  depth

   with segments separated by `>` header lines, 701 samples per profile at
   1 km spacing, cast symmetric at +/-550 km. Select one profile with
   --index. Because `+v` orients the profiles consistently east-west or
   south-north rather than seaward, the seaward limb is identified here the
   same way `build_dataframe.py` does it -- by which limb reaches the abyssal
   reference rather than climbing the arc. Override with --side if the
   heuristic picks wrongly for an unusual margin.

2. A two-column CSV with a header `dist_km,depth_m`, distance from the axis
   positive seaward, for a profile already trimmed and oriented upstream.

Distances are km, depths metres and negative downwards, as GEBCO supplies them.

Physics
-------
Broken elastic plate, flexure of the incoming lithosphere seaward of the axis:

    w(x) = exp(-x/alpha) * (A cos(x/alpha) + B sin(x/alpha)) + w0

with flexural parameter and rigidity

    alpha = (4 D / (drho g))**0.25
    D     = E Te**3 / (12 (1 - nu**2))

so that Te follows from the fitted alpha. Fitting alpha, A, B and the reference
level w0 inside a window [x1, x2] is the classical procedure; sweeping x1
is the experiment.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# --------------------------------------------------------------- physics ---
# The baseline inversion MUST be the project's own procedure, not a lookalike,
# or the window sweep measures a different method from the one in the companion
# papers. morphometry.py is imported directly: it has no module-level side
# effects, so importing it is safe. Elastic constants, the pure-cosine
# broken-plate solution of Eq. 2, the parameter bounds and te_from_alpha all
# come from there.
try:
    from morphometry import (deflection, flexural_fit, te_from_alpha,
                             _axis, _smooth)
except ImportError:                                    # pragma: no cover
    raise SystemExit(
        "morphometry.py must be importable (put it beside this script or on "
        "PYTHONPATH). This figure reproduces its flexural_fit exactly; a "
        "reimplementation here would defeat the purpose of the experiment.")

SEED = 0

# Seaward reaches to compare. The first two are the values ALREADY hard-coded
# in the project: flexural_fit defaults to 250 km, parameterise calls it with
# 320 km, and sample_profile extracts to 250 km. The remaining two bracket the
# range adopted in the published literature. Replace with the exact windows of
# Hunter, Bry & White and Contreras-Reyes when those are to hand.
WINDOWS_KM = [250.0, 320.0, 200.0, 400.0]
WINDOW_NOTES = {250.0: "flexural_fit default",
                320.0: "parameterise default"}


def fit_window(x_km, d_km, x0, x_fit_max):
    """One windowed inversion, delegated to morphometry.flexural_fit."""
    f = flexural_fit(x_km, d_km, x0=x0, x_fit_max=x_fit_max)
    if not f.get("valid"):
        return np.nan, None
    return f["Te"], f


def sweep_seaward(x_km, d_km, x0, lo=50.0, hi=550.0, n=101):
    """Te as a continuous function of the seaward window limit.

    The sweep begins close to the trench axis, where the window holds only the
    inner limb of the depression, and runs out to the full seaward reach cast
    by extract_profiles.sh, so that the whole trajectory of the estimate is
    visible: unstable while the window is shorter than the flexural
    wavelength, then settling once the outer rise and the abyssal reference
    are both enclosed.
    """
    limits = np.linspace(lo, hi, n)
    te = np.array([fit_window(x_km, d_km, x0, xf)[0] for xf in limits])
    return limits, te


def read_grdtrack(path, index):
    """One profile from a grdtrack -C multisegment table.

    Columns are lon lat dist azimuth depth; `>` starts a new segment. Returns
    (dist_km, depth_m) with NaN samples dropped, sorted on distance.
    """
    segs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    segs.append(cur)
                cur = []
                continue
            f = line.split()
            if len(f) < 5:
                continue
            try:
                lo, la, d, z = float(f[0]), float(f[1]), float(f[2]), float(f[4])
            except ValueError:
                continue          # NaN written as text by GMT
            if np.isfinite(d) and np.isfinite(z):
                cur.append((d, z, lo, la))
    if cur:
        segs.append(cur)
    if not segs:
        raise SystemExit(f"no profiles parsed from {path}")
    if not 0 <= index < len(segs):
        raise SystemExit(f"--index {index} out of range: file holds "
                         f"{len(segs)} profiles (0..{len(segs)-1})")
    arr = np.asarray(segs[index], dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


# Direction of the OVERRIDING plate from the trench axis, degrees clockwise
# from north. Copied verbatim from build_dataframe.py, which is the project's
# single source of truth for margin orientation. Do not diverge from it: if a
# margin is added or corrected there, mirror the change here.
LANDWARD_AZIMUTH = {
    "aleutian": 350.0, "kuril-kamchatka": 315.0, "japan": 270.0,
    "izu-bonin": 270.0, "mariana": 270.0, "yap": 270.0, "palau": 270.0,
    "ryukyu": 315.0, "manila": 90.0, "philippine": 270.0,
    "new_britain": 0.0, "san_cristobal": 45.0, "vityaz": 180.0,
    "new_hebrides": 90.0, "tonga": 270.0, "kermadec": 270.0,
    "hikurangi": 270.0, "puysegur": 90.0, "middle_america": 45.0,
    "peru-chile": 90.0,
}
DEFAULT_LANDWARD = -1


def side_from_azimuth(dist, lon, lat, az_landward):
    """Sign of `dist` on the landward side, from the profile's own geometry.

    Identical to build_dataframe.side_from_azimuth: the bearing from the axis
    node to the positive-distance end is compared with the direction of the
    overriding plate, and the landward side is positive when they agree to
    within 90 degrees. No depths are used, so the intra-oceanic margins -- where
    neither far field rises to a continent and a depth rule is close to a coin
    toss -- are resolved as reliably as the continental ones.
    """
    i0 = int(np.argmin(np.abs(dist)))
    i1 = int(np.argmax(dist))
    dlon = np.radians(lon[i1] - lon[i0])
    p0, p1 = np.radians(lat[i0]), np.radians(lat[i1])
    y = np.sin(dlon) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x)) % 360.0
    diff = abs((bearing - az_landward + 180.0) % 360.0 - 180.0)
    return +1 if diff < 90.0 else -1


def orient_seaward(dist_km, depth_m, lon, lat, trench, side_opt="auto",
                   seaward_km=550.0, landward_km=100.0):
    """Return the seaward limb, distance positive away from the axis.

    build_dataframe.py trims with `dist * side <= LAND_KM` and
    `dist * side >= -SEAWARD_KM`, where `side` is the sign of `dist` on the
    LANDWARD flank. The seaward flank is therefore where `dist * side` is
    negative, and seaward distance is `-dist * side`.
    """
    if side_opt == "landward-positive":
        side = +1
    elif side_opt == "landward-negative":
        side = -1
    else:
        az = LANDWARD_AZIMUTH.get(trench.strip().lower())
        if az is None:
            side = DEFAULT_LANDWARD
            print(f"note: '{trench}' not in LANDWARD_AZIMUTH; assuming "
                  f"landward side {side:+d}. Check against build_dataframe.py.",
                  file=sys.stderr)
        else:
            side = side_from_azimuth(dist_km, lon, lat, az)

    d_seaward = -dist_km * side
    # A short landward reach is retained so that the axis is seen as the floor
    # of a depression with limbs on both sides. build_dataframe.py trims at
    # LAND_KM = 150 km; 100 km is used here purely so the panel is not
    # dominated by the arc slope, and nothing is fitted landward of the axis.
    m = (d_seaward >= -landward_km) & (d_seaward <= seaward_km)
    d, z = d_seaward[m], depth_m[m]
    o = np.argsort(d)
    label = "positive" if side < 0 else "negative"
    return d[o], z[o], side, label


# ------------------------------------------------------------------- style

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
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.default": "regular", "savefig.dpi": 600,
    })


def style_axes(ax, xlabel=None, ylabel=None, minor=True):
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5, pad=2)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.85")
    ax.grid(which="minor", lw=0.3, color="0.92")
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def covers_data(fig, ax, artist, data_xy, text_only=False):
    """Rule 1/7 check: count data points falling inside an artist's bbox.

    For an Annotation, `get_window_extent` spans the text AND its leader arrow.
    The arrow is meant to touch the data -- that is its job -- so with
    text_only=True the bounding box of the text alone is measured, which is
    what the no-overlap rule actually governs.
    """
    fig.canvas.draw()
    if text_only:
        from matplotlib.text import Text
        bb = Text.get_window_extent(artist, renderer=fig.canvas.get_renderer())
    else:
        bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    pts = ax.transData.transform(data_xy)
    inside = ((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
              (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1))
    return int(inside.sum())


# -------------------------------------------------------------------- plot
def nudge_inside(fig, ax, artists, margin_px=3.0):
    """Shift any label whose text box runs past the frame back inside it.

    Measured in pixels and applied in data units, so it works whatever the
    label text turns out to be on a given profile.
    """
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ab = ax.get_window_extent(renderer=r)
    for a in artists:
        bb = Text.get_window_extent(a, renderer=r)
        dx_px = 0.0
        if bb.x0 < ab.x0 + margin_px:
            dx_px = (ab.x0 + margin_px) - bb.x0
        elif bb.x1 > ab.x1 - margin_px:
            dx_px = (ab.x1 - margin_px) - bb.x1
        if dx_px:
            xt, yt = a.get_position()
            x_px, y_px = ax.transData.transform((xt, yt))
            xt_new, _ = ax.transData.inverted().transform((x_px + dx_px, y_px))
            a.set_position((xt_new, yt))
    fig.canvas.draw()


def annotate_features(ax, x_km, ds, x0, d0, lo_d, hi_d, pad):
    """Label the morphological features, text in blank space with leader arrows.

    Positions are derived from the data, not hard-coded, so the labels follow
    the section when the profile changes. Two horizontal bands are reliably
    empty on an inverted depth axis: above the shallowest data and below the
    deepest. All text goes in one of those two; only the arrows touch the data.
    The crest is located by the rule of morphometry.descriptors -- the
    shallowest smoothed sample beyond CREST_MIN_X of the axis.
    """
    y_top = lo_d - 3.3 * pad           # blank band above the shallowest data
    kw = dict(fontsize=9, color="0.20", zorder=7,
              arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35",
                              shrinkA=1.5, shrinkB=1.5))
    ann = []

    # The axis is already marked by the broken vertical line, so the label sits
    # at the head of that line and needs no leader at all.
    ann.append(ax.text(x0, y_top, "Trench axis", fontsize=9, color="0.20",
                       ha="center", va="bottom", zorder=7))

    # Landward and seaward slopes: same depth, mirrored about the axis, both
    # inside the depression where the panel is empty. Short vertical leaders.
    dxs = 0.085 * (x_km[-1] - x_km[0])
    xs = [x0 - dxs, x0 + dxs]
    # One depth for both labels, set from the shallower limb so that neither
    # sits on the section; the two arrows then differ in length only because
    # the limbs themselves are asymmetric, which is the point of the figure.
    span = [(x_km >= min(x, x0) - 0.5 * dxs) & (x_km <= max(x, x0) + 0.5 * dxs)
            for x in xs]
    y_slope = max(float(np.nanmax(ds[m])) for m in span if m.any()) + 0.9 * pad
    slope_ann = []
    for xt, name in zip(xs, ("Landward slope", "Seaward slope")):
        if not (x_km[0] < xt < x_km[-1]):
            continue
        a = ax.annotate(name, xy=(xt, float(np.interp(xt, x_km, ds))),
                        xytext=(xt, y_slope), ha="center", va="top", **kw)
        slope_ann.append(a)
        ann.append(a)

    # outer-rise crest, located by the rule of morphometry.descriptors
    hi_x = min(x_km[-1] - 40.0, x0 + 240.0)
    win = (x_km > x0 + 25.0) & (x_km <= hi_x)
    if win.sum() > 10:
        ib = int(np.nanargmin(ds[win]))
        xc, dc = float(x_km[win][ib]), float(ds[win][ib])
        ann.append(ax.annotate("Outer-rise crest", xy=(xc, dc),
                               xytext=(xc, dc - 0.85 * pad), ha="center",
                               va="bottom", **kw))

    # abyssal reference, read well beyond the crest
    far = x_km >= x0 + 380.0
    if far.sum() > 20:
        xf = float(np.median(x_km[far]))
        df = float(np.median(ds[far]))
        ann.append(ax.annotate("Abyssal reference", xy=(xf, df),
                               xytext=(xf, df - 0.85 * pad), ha="center",
                               va="bottom", **kw))
    return ann, slope_ann


def build(x_km, depth_km, trench, synthetic=False):
    """x positive seaward (km); depth in km POSITIVE DOWNWARDS, per morphometry.py."""
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 3.9), constrained_layout=True)

    dx = float(np.median(np.diff(x_km)))
    ds = _smooth(np.where(np.isfinite(depth_km), depth_km,
                          np.nanmedian(depth_km)), dx)
    _, x0 = _axis(x_km, ds)                       # deepest point within 30 km

    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    styles = ["-", "--", "-.", ":"]

    ax.plot(x_km, depth_km, color="0.35", lw=0.8, zorder=2,
            label="Observed profile")

    results = []
    for xf, c, ls in zip(WINDOWS_KM, cols, styles):
        te, f = fit_window(x_km, depth_km, x0, xf)
        results.append((xf, te))
        if f is None:
            continue
        xg = np.linspace(x0, x0 + xf, 400)
        ax.plot(xg, deflection(xg - x0, f["w0"], f["alpha"]) + f["d_r"],
                ls=ls, lw=1.3, color=c, zorder=6,
                solid_capstyle="round",
                label=(f"{xf:.0f} km window"
                       + (" (code default)" if xf in WINDOW_NOTES else "")
                       + f":  $T_e$ = {te:.1f} km"))

    lo_d, hi_d = np.nanmin(depth_km), np.nanmax(depth_km)
    pad = 0.08 * (hi_d - lo_d)
    ax.set_ylim(hi_d + 2.4 * pad, lo_d - 4.8 * pad)   # inverted; bands for labels
    ax.set_xlim(x_km.min(), x_km.max())
    y_bottom, y_topaxis = ax.get_ylim()          # inverted: y_bottom > y_topaxis
    ax.fill_between(x_km, depth_km, y_bottom, color="#b08968", alpha=0.16,
                    lw=0, zorder=0)                  # lithosphere
    ax.fill_between(x_km, y_topaxis, depth_km, color="#4c8fbd", alpha=0.11,
                    lw=0, zorder=0)                  # water column
    ax.axvspan(float(x_km[0]), x0, color="white", alpha=0.55, lw=0,
               zorder=6.5)                          # landward: shown, never fitted
    ax.axvline(x0, color="0.55", lw=0.7, ls=(0, (4, 3)), zorder=6.8)
    d0 = float(depth_km[int(np.argmin(np.abs(x_km - x0)))])
    style_axes(ax, xlabel="Distance from trench axis, seaward positive (km)",
               ylabel="Depth (km)")
    ax.xaxis.set_major_locator(MultipleLocator(100))   # guarantees a 0 tick

    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.16),
                    mode="expand", ncol=2, frameon=False, borderpad=0.3,
                    handlelength=3.0, handletextpad=0.9, labelspacing=0.45,
                    columnspacing=2.4, fontsize=8.5)
    leg.set_zorder(6)

    axi = ax.inset_axes([0.585, 0.135, 0.400, 0.360])
    lim, te_sweep = sweep_seaward(x_km, depth_km, x0)
    axi.plot(lim, te_sweep, color="0.20", lw=0.9, zorder=3)
    for (xf, te), c in zip(results, cols):
        if np.isfinite(te):
            axi.plot([xf], [te], "o", ms=4.0, color=c, mec="white", mew=0.6,
                     zorder=5)
    style_axes(axi, xlabel="Seaward window limit (km)",
               ylabel="Recovered $T_e$ (km)")
    axi.tick_params(labelsize=8.5, pad=1.5)
    axi.xaxis.labelpad = 1.0
    axi.yaxis.labelpad = 1.0
    axi.xaxis.label.set_size(8.5)
    axi.yaxis.label.set_size(8.5)
    for sp in axi.spines.values():
        sp.set_linewidth(0.7)
    axi.patch.set_facecolor("#f7f9fb")
    axi.patch.set_alpha(0.96)

    ann, slope_ann = annotate_features(ax, x_km, ds, x0, d0, lo_d, hi_d, pad)
    nudge_inside(fig, ax, ann)
    # A nudged label would leave its leader slanting; re-anchor the two slope
    # arrows to the label's own abscissa so both stay vertical and parallel.
    for a in slope_ann:
        xt, _ = a.get_position()
        a.xy = (xt, float(np.interp(xt, x_km, ds)))
    fig.canvas.draw()

    data_xy = np.column_stack([x_km, depth_km])
    hits = (covers_data(fig, ax, leg, data_xy)
            + covers_data(fig, ax, axi, data_xy))
    for a in ann:
        h = covers_data(fig, ax, a, data_xy, text_only=True)
        if h:
            print(f"  annotation '{a.get_text()}' covers {h} data points")
        hits += h
    print(f"overlap check -- legend, inset and {len(ann)} annotations cover "
          f"{hits} data points")

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, SciPy, NumPy. "
              "Data: GEBCO. Source: authors.")
    fig.text(0.005, -0.015, credit, fontsize=8, ha="left", va="top")

    if synthetic:
        ax.text(0.5, 0.5, "SYNTHETIC -- LAYOUT CHECK ONLY",
                transform=ax.transAxes, ha="center", va="center", fontsize=20,
                color="0.80", rotation=22, zorder=8, alpha=0.55)

    te_vals = np.array([r[1] for r in results], dtype=float)
    print(f"{trench}: axis at x0 = {x0:.0f} km;  Te = "
          + ", ".join(f"{t:.1f}" for t in te_vals)
          + f" km  (spread {np.nanmax(te_vals) - np.nanmin(te_vals):.1f} km)")
    return fig, hits


def synthetic_profile():
    """Analytic broken-plate section, km positive down. LAYOUT TESTING ONLY."""
    rng = np.random.default_rng(SEED)
    x = np.arange(-100.0, 551.0, 1.0)
    d = np.where(x >= 0.0, deflection(np.clip(x, 0, None), 2.60, 120.0) + 5.40,
                 8.00 + 0.021 * x)                 # landward: arc slope, km
    d += 0.055 * rng.standard_normal(x.size)
    return x, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", help="grdtrack -C table, or CSV dist_km,depth_m")
    ap.add_argument("--index", type=int, default=0,
                    help="which profile in a multisegment grdtrack table")
    ap.add_argument("--landward", type=float, default=100.0,
                    help="landward reach shown, km (display only; never fitted)")
    ap.add_argument("--side", default="auto",
                    choices=("auto", "landward-positive", "landward-negative"),
                    help="override the overriding-plate bearing test")
    ap.add_argument("--trench", default="", help="trench name, for the log line")
    ap.add_argument("--outdir", default=".", help="where to write the figure")
    ap.add_argument("--name", default="fig_windowproblem",
                    help="output basename, content-named (no float number)")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic layout check; output must not be published")
    a = ap.parse_args()

    if a.selftest:
        x, w = synthetic_profile()
        fig, hits = build(x, w, "SYNTHETIC", synthetic=True)
        name = a.name + "_SELFTEST"
    else:
        if not a.profile:
            ap.error("--profile is required (or use --selftest)")
        head = open(a.profile).readline()
        if head.lower().startswith("dist_km"):
            import csv
            d, z = [], []
            with open(a.profile) as fh:
                for row in csv.DictReader(fh):
                    d.append(float(row["dist_km"]))
                    z.append(float(row["depth_m"]))
            x = np.asarray(d)
            w = -np.asarray(z) / 1000.0   # elevation m -> depth km
            o = np.argsort(x)
            x, w = x[o], w[o]
        else:
            dd, zz, lo, la = read_grdtrack(a.profile, a.index)
            x, w_m, side, limb = orient_seaward(dd, zz, lo, la, a.trench,
                                               a.side, landward_km=a.landward)
            w = -w_m / 1000.0            # elevation m -> depth km, per morphometry
            print(f"profile {a.index}: landward side {side:+d}, seaward limb "
                  f"= {limb} distances, {x.size} samples, "
                  f"{x.min():.0f} to {x.max():.0f} km")
        fig, hits = build(x, w, a.trench or a.profile)
        name = a.name

    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{name}.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
        print("wrote", out)
    if hits:
        print("WARNING: an annotation covers data -- fix before publishing")


if __name__ == "__main__":
    main()
