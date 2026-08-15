#!/usr/bin/env python3
"""fig_windowsensitivity -- recovered elastic thickness as a function of BOTH
window limits, across the twenty circum-Pacific margins.

fig_windowproblem sweeps only the seaward limit, on one profile. This figure
sweeps the landward limit as well, and does it on every margin in the study, so
that the reader sees whether the drift is a property of one profile, of one
trench, or of the method. Each panel is a filled surface of the elastic
thickness returned by the classical windowed inversion over the plane of
(landward limit, seaward limit); the window choices adopted by previous studies
are marked, and a contour at the window-free estimate shows which windows
happen to reproduce it.

Every fit is morphometry.flexural_fit, so the surface is the behaviour of the
project's own baseline, not of a lookalike written for this figure.

TWO MODES
  survey (default) -- one panel per trench, 5 rows x 4 columns, each panel the
      per-window median over a sample of that trench's profiles. This is the
      figure that matches the study object: twenty margins, not four, and the
      per-trench Delta_w it prints is the same quantity Table 4 tabulates.

        python3 fig_windowsensitivity.py --profiles profiles_csv \
            --outdir figures

  panel -- four named profiles, 2 x 2, for a close reading of individual
      surfaces with labelled contours.

        python3 fig_windowsensitivity.py --profiles profiles_csv \
            --panel mariana:35 --panel peru-chile:201 \
            --panel tonga:2 --panel aleutian:145 --outdir figures

  A single profile is one anecdote. Aggregating each panel over many profiles
  of the same margin removes the arbitrariness of which profile was chosen, and
  the per-window median is the right statistic: it is what Delta_w is
  normalised by, and it resists the handful of windows on any margin where the
  inversion returns something extreme.

INPUT
  --profiles points at the directory of per-profile CSVs, one file per profile,
  named <trench>_<pid>.csv with columns r, s, p, depth, faa, vgg, hs. The same
  store 06_flexure.py reads. `p` is signed distance in km from the axis and is
  ALREADY POSITIVE SEAWARD, running -150 to +550 km at 1 km, so no side
  inference is needed and none is done: the earlier version's dependence on
  fig_windowproblem.orient_seaward is gone, and with it the risk that the two
  window figures disagree about which flank is which.

  Sediment is unloaded by Airy backstripping exactly as 06_flexure.py does
  (hs * SED_FACTOR), which matters because the margins span the sediment range
  from starved Mariana to fed Peru-Chile. --no-sediment turns it off.

GEOMETRY
  The stored profiles reach 550 km seaward, so that is the default cap. Note
  that extract_profiles.sh in the repository casts only 250 km; it is NOT the
  script that produced this store, and a 250 km cap would discard the reach in
  which window dependence is worst.

REJECTION, AND WHY IT IS NOT OPTIONAL
  flexural_fit reports valid=True whenever least_squares converges, including
  when the flexural parameter has railed against its bounds of 15 and 200 km.
  Left in, every margin returns the same Te range of 3.6-112.7 km -- those two
  numbers are te_from_alpha at the bounds, not measurements -- and Delta_w
  inflates to 13-31. 06_flexure.py rejects railed fits; so does this script.
  flexural_fit also raises ValueError when its own initial guess falls outside
  its own bounds, which happens on real profiles, so every call is guarded.

OPTIONS
  --windowfree  one value per panel, km, in panel order. The estimate returned
                by the physics-informed inversion; drawn as a heavy contour.
                Omit and the contour is skipped -- nothing is invented when the
                value is not yet available.
  --published   published window choices as "landward,seaward" pairs in km;
                nothing is marked unless this is given,
                applied to every panel. PLACEHOLDERS until replaced.
  --scale       median (default) plots Te divided by each panel's own median,
                on a scale centred on one, so that margins of very different
                absolute Te stay comparable AS SENSITIVITY SURFACES and the
                colour axis carries Delta_w's own quantity. shared plots
                absolute Te on one scale: directly comparable between margins,
                but a margin with small spread flattens to a single colour.
  --nprofiles   profiles sampled per trench in survey mode, evenly spaced along
                the margin. Raise for a more stable median at linear cost.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import distance_transform_edt
from matplotlib.ticker import (AutoMinorLocator, FixedLocator,
                               FuncFormatter, MaxNLocator, NullFormatter)

from morphometry import flexural_fit, _axis, _smooth

# Seaward reach of the stored profiles, km. Matches the p column of the CSVs,
# NOT HALFLEN in extract_profiles.sh, which describes a different extraction.
HALFLEN_KM = 550.0

# Shortest window that can constrain a two-parameter flexural fit, km.
MIN_SPAN_KM = 80.0

# Airy backstripping of the sediment column, as in 06_flexure.py.
RHO_M, RHO_W, RHO_S = 3300.0, 1030.0, 2000.0
SED_FACTOR = (RHO_M - RHO_S) / (RHO_M - RHO_W)

# Bounds on the flexural parameter inside morphometry.flexural_fit, km. A fit
# that lands on one of these has been stopped by the optimiser, not by the
# data, and is discarded. Mirror any change made there.
ALPHA_BOUNDS = (15.0, 200.0)
ALPHA_MARGIN = 0.01
RMS_MAX_KM = 1.2

# The twenty trenches in the order of Table 2 and of axes_full/, so that the
# panel grid reads in the same sequence as the rest of the manuscript rather
# than alphabetically.
# Display names where the directory key and the manuscript label differ. The
# KEY is the folder under profiles_csv/ and must not be renamed; the VALUE is
# what appears on the panel and in the printed table, and must match the name
# used in Table 2 and in the text. "New Hebrides" and "Vanuatu" are the same
# margin; the manuscript uses Vanuatu throughout, so the figure must too.
DISPLAY_NAME = {"new_hebrides": "Vanuatu"}

TRENCH_ORDER = [
    "aleutian", "kuril-kamchatka", "japan", "izu-bonin",
    "mariana", "yap", "palau", "ryukyu",
    "manila", "philippine", "new_britain", "san_cristobal",
    "vityaz", "new_hebrides", "tonga", "kermadec",
    "hikurangi", "puysegur", "middle_america", "peru-chile",
]

# Window choices of previous studies are NOT built in. There is deliberately no
# default: a marker drawn from a stand-in value attributes an analysis window to
# a study that did not choose it, and the resulting figure asserts something
# about the literature that is not true. Supply real pairs on the command line,
#     --published 0,150 20,220
# as (landward limit, seaward limit) in km from the axis, each satisfying
# landward < seaward <= HALFLEN_KM or it is dropped with a note rather than
# drawn outside the panel. With no --published argument no markers are drawn,
# and the caption must then carry no clause about published windows.


# ----------------------------------------------------------------- profiles --
def profile_files(profdir, trench):
    return sorted(glob.glob(os.path.join(profdir, f"{trench}_[0-9]*.csv")))


def read_profile_file(path, sediment=True):
    """One profile. Returns (x_km positive seaward, d_km positive down).

    `p` is already oriented in the stored file, so nothing is flipped here.
    """
    d = pd.read_csv(path).dropna(subset=["p", "depth"])
    p = d["p"].to_numpy(float)
    z = d["depth"].to_numpy(float)
    if sediment and "hs" in d:
        z = z + np.nan_to_num(d["hs"].to_numpy(float)) * SED_FACTOR
    return p, -z / 1000.0


def read_profile(profdir, trench, pid, sediment=True):
    cand = [os.path.join(profdir, f"{trench}_{pid:04d}.csv"),
            os.path.join(profdir, f"{trench}_{pid}.csv")]
    path = next((c for c in cand if os.path.exists(c)), None)
    if path is None:
        near = profile_files(profdir, trench)
        raise SystemExit(
            f"no profile {trench}:{pid} in {profdir}"
            + (f" ({len(near)} files for '{trench}', "
               f"{os.path.basename(near[0])} .. {os.path.basename(near[-1])})"
               if near else f" (no files for '{trench}' at all)"))
    x, d = read_profile_file(path, sediment)
    return x, d, os.path.basename(path)


# ------------------------------------------------------------------ surface --
def one_fit(x_km, d_km, x0, x_fit_max):
    """One windowed inversion, with both of flexural_fit's failure modes caught.

    Returns NaN for: a raised ValueError (its initial guess outside its own
    bounds), a non-converged fit, a flexural parameter railed against a bound,
    or a misfit above RMS_MAX_KM.
    """
    try:
        f = flexural_fit(x_km, d_km, x0=x0, x_fit_max=x_fit_max)
    except (ValueError, RuntimeError):
        return np.nan, "raised"
    if not f.get("valid"):
        return np.nan, "not converged"
    a = f["alpha"]
    if not (ALPHA_BOUNDS[0] * (1 + ALPHA_MARGIN) < a
            < ALPHA_BOUNDS[1] * (1 - ALPHA_MARGIN)):
        return np.nan, "alpha at bound"
    if f["rms"] > RMS_MAX_KM:
        return np.nan, "misfit too large"
    return f["Te"], "ok"


def surface(x_km, d_km, near, far):
    """Te over the plane of window limits, using the project's own inversion."""
    dx = float(np.median(np.diff(x_km)))
    ds = _smooth(np.where(np.isfinite(d_km), d_km, np.nanmedian(d_km)), dx)
    _, x0 = _axis(x_km, ds)
    te = np.full((near.size, far.size), np.nan)
    why = {}
    for i, xn in enumerate(near):
        for j, xf in enumerate(far):
            if xf - xn < MIN_SPAN_KM:      # too short to constrain the fit
                continue
            t, w = one_fit(x_km, d_km, x0 + xn, xf - xn)
            te[i, j] = t
            why[w] = why.get(w, 0) + 1
    return te, x0, why


def trench_surface(profdir, trench, near, far, nprofiles, sediment=True):
    """Per-window median Te over an evenly spaced sample of one trench.

    Sampling is by position along the margin, not at random, so the sample
    represents the whole arc rather than clustering wherever the profile
    numbering happens to be dense.
    """
    files = profile_files(profdir, trench)
    if not files:
        return None, 0, 0
    if len(files) <= nprofiles:
        take = files
    else:
        # np.linspace(...).astype(int) truncates, so two neighbouring samples
        # can collapse onto the same index and the panel then reports fewer
        # profiles than were requested -- Kermadec returned 23 of a requested
        # 24. Rounding and de-duplicating keeps the count as asked while
        # preserving the even spacing along strike.
        idx = np.unique(np.rint(
            np.linspace(0, len(files) - 1, nprofiles)).astype(int))
        take = [files[i] for i in idx]
    stack, used = [], 0
    for path in take:
        x, d = read_profile_file(path, sediment)
        te, _, _ = surface(x, d, near, far)
        if np.isfinite(te).any():
            stack.append(te)
            used += 1
    if not stack:
        return None, 0, len(files)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.dstack(stack), axis=2)
    return med, used, len(files)


def ink_for(cmap_name, norm, vmin, vmax, value):
    """Black or white, whichever reads against the colour map at `value`.

    Relative luminance of the mapped colour, sRGB weights. The 0.55 threshold
    rather than 0.5 leans towards white text, which survives better against
    the mid-tones of a sequential ramp than black does.
    """
    cm = plt.get_cmap(cmap_name)
    if norm is not None:
        t = float(norm(value))
    else:
        t = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    r, g, b, _ = cm(np.clip(t, 0.0, 1.0))
    return "black" if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else "white"


def spline_resample(field, near, far, factor):
    """Bicubic spline resampling of one surface, FOR DISPLAY ONLY.

    The window grid is coarse -- 5 km in the landward limit and 20 or 25 in the
    seaward -- because every node costs one non-linear inversion per profile.
    Drawn raw, contourf steps between nodes and the isolines come out as
    staircases that no reader should mistake for structure. The spline is
    applied to the plotted field only; every number reported, Delta_w included,
    is computed on the unsmoothed grid.

    NaNs cannot go into a spline, so the holes are filled by nearest-neighbour
    before fitting and masked out again afterwards -- the fill never reaches
    the drawn surface, it only stops the spline ringing at the edge of a gap.
    """
    if factor is None or factor <= 1:
        return field, near, far
    bad = ~np.isfinite(field)
    if bad.all():
        return field, near, far
    filled = field.copy()
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False,
                                     return_indices=True)
        filled = field[tuple(idx)]
    ky = min(3, near.size - 1)
    kx = min(3, far.size - 1)
    if kx < 1 or ky < 1:
        return field, near, far
    spl = RectBivariateSpline(near, far, filled, kx=kx, ky=ky, s=0)
    near_f = np.linspace(near[0], near[-1], (near.size - 1) * factor + 1)
    far_f = np.linspace(far[0], far[-1], (far.size - 1) * factor + 1)
    out = spl(near_f, far_f)
    if bad.any():
        # Nearest-neighbour, not spline, for the mask: a smoothed mask would
        # bleed the hole outward or seal it shut depending on the threshold.
        mi = np.array([int(np.abs(near - v).argmin()) for v in near_f])
        mj = np.array([int(np.abs(far - v).argmin()) for v in far_f])
        out = np.where(bad[np.ix_(mi, mj)], np.nan, out)
    return out, near_f, far_f


def window_metric(te):
    """Delta_w of Eq. (15): (max - min) / median over the swept window set."""
    finite = te[np.isfinite(te)]
    if finite.size < 2:
        return np.nan
    med = np.median(finite)
    if not np.isfinite(med) or med == 0.0:
        return np.nan
    return float((finite.max() - finite.min()) / med)


# -------------------------------------------------------------------- style --
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


def pick_cmap(scale, name="rainbow"):
    """Colour map for the surface. Default rainbow, by request.

    NOTE, and it belongs in the caption rather than in a comment: rainbow (like
    jet) is NOT perceptually uniform, NOT colour-blind safe and NOT
    greyscale-safe. Its lightness is non-monotonic, so it inserts visual edges
    at the cyan and yellow transitions that the data does not contain, and a
    deuteranopic or a printed reader cannot order its colours. The caption of
    Figure 4 currently claims "the colour scale is perceptually uniform and
    colour-blind safe"; with rainbow that sentence is false and must be removed
    or the map changed back. --cmap accepts anything matplotlib knows, so
    `--cmap cmc.vik` (diverging, for --scale median) or `--cmap cmc.lipari`
    (sequential, for --scale shared) restores a safe ramp in one word.
    cmcrameri is imported first because testing `"cmc.vik" in plt.colormaps()`
    without the import is always False, which silently downgraded every figure
    in an earlier version.
    """
    if name.startswith("cmc."):
        try:
            import cmcrameri.cm  # noqa: F401  (registers cmc.* on import)
        except Exception:
            fallback = "RdBu_r" if scale == "median" else "viridis"
            print(f"note: cmcrameri not installed, falling back to {fallback}")
            return fallback
    return name


def set_rc(base=9.0):
    register_nimbus()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "DejaVu Sans"],
        "font.size": base, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def style_axes(ax, minor=True):
    if minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in",
                   color="0.85")
    ax.tick_params(which="major", length=3.4, pad=2)
    ax.tick_params(which="minor", length=2.0)


# --------------------------------------------------------------------- plot --
def build(panels, near, far, windowfree, published, scale, cmap_name,
          nrows, ncols, figsize, tagsize, base_font, vlim=None,
          log_norm=True, smooth=1, contour_every=3, label_ink="white"):
    """panels: list of (te, title, note). te may be None for a missing trench."""
    set_rc(base_font)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True,
                             sharey=True, constrained_layout=True)
    # Room for the two-line header annotation above every panel.
    fig.get_layout_engine().set(hspace=0.09)
    axes = np.atleast_1d(axes).ravel()
    label_contours = len(panels) <= 6

    fields = []
    for te, *_ in panels:
        if te is None:
            fields.append(None)
            continue
        med = np.nanmedian(te)
        fields.append(te / med if scale == "median" and np.isfinite(med)
                      else te)
    finite = [f for f in fields if f is not None and np.isfinite(f).any()]
    if not finite:
        raise SystemExit("every panel is empty: no window on the grid returned "
                         "a usable fit. Check --profiles.")

    if scale == "median":
        # The ratio is multiplicative and tightly peaked: over the twenty
        # margins 57 per cent of windows fall between 0.8 and 1.25 times the
        # panel median, 83 per cent between 0.67 and 1.5, and 94 per cent
        # between 0.5 and 2, while the tail reaches ten. Spreading a linear
        # scale over that whole tail spends nearly the entire colour map on
        # 6 per cent of the data and leaves the rest in one band -- which is
        # why the surfaces looked uniformly green whatever colour map was
        # chosen. The cure is the normalisation, not the palette: the range is
        # clipped to one octave either side of the median and the levels are
        # geometric, so 0.5 and 2.0 sit equidistant from 1.0, exactly as a
        # ratio requires, and 1.0 lands on the middle level boundary without
        # any two-slope trickery. Windows outside the octave go to the arrow
        # ends. --vlim widens or narrows it.
        vmin, vmax = (vlim if vlim else (0.5, 2.0))
        if not 0 < vmin < 1 < vmax:
            raise SystemExit("--vlim for --scale median wants 0 < LO < 1 < HI")
        if log_norm:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            levels = np.geomspace(vmin, vmax, 21)
        else:
            norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
            levels = np.unique(np.concatenate(
                [np.linspace(vmin, 1.0, 11), np.linspace(1.0, vmax, 11)]))
        cb_label = r"Recovered $T_e$ / panel median"
        nice = np.array([0.2, 0.25, 0.33, 0.4, 0.5, 0.67, 0.8, 0.9, 1.0,
                         1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        cb_ticks = nice[(nice >= vmin) & (nice <= vmax)]
    else:
        # --vlim is absolute km here, not a ratio. Worth setting: a shared
        # scale taken from the 2nd and 98th percentiles across panels is
        # dragged upward by the narrow spikes at short seaward limits, and the
        # plateau that occupies most of every panel then sits in the bottom
        # fifth of the colour map.
        if vlim:
            vmin, vmax = float(vlim[0]), float(vlim[1])
        else:
            vmin = min(np.nanpercentile(f, 2) for f in finite)
            vmax = max(np.nanpercentile(f, 98) for f in finite)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(vmin) - 1.0, float(vmin) + 1.0
        # Elastic thickness is positive and right-skewed: the plateau that
        # fills most of every panel sits near 10-16 km while the spikes at
        # short seaward limits reach 90. Linear levels therefore park the
        # plateau in the bottom fifth of the map and the whole figure reads as
        # one dark band. Geometric levels give equal colour to equal RATIOS of
        # Te, which is the natural spacing for a thickness, and spread the
        # plateau across the middle of the ramp. --linear-norm reverts it.
        if log_norm and vmin > 0:
            norm = LogNorm(vmin=vmin, vmax=vmax)
            levels = np.geomspace(vmin, vmax, 21)
            nice = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40,
                             50, 60, 80, 100], dtype=float)
            cb_ticks = nice[(nice >= vmin) & (nice <= vmax)]
        else:
            norm = None
            levels = np.linspace(vmin, vmax, 21)
            cb_ticks = None
        cb_label = r"Recovered $T_e$ (km)"
    cmap = pick_cmap(scale, cmap_name)

    tags = [f"({chr(97 + k)})" for k in range(len(panels))]
    im = None
    for k, ax in enumerate(axes):
        if k >= len(panels):
            ax.set_visible(False)
            continue
        te, title, note = panels[k]
        fld = fields[k]
        style_axes(ax, minor=len(panels) <= 6)

        if fld is None or not np.isfinite(fld).any():
            ax.text(0.5, 0.5, "no usable fit", transform=ax.transAxes,
                    ha="center", va="center", fontsize=tagsize,
                    style="italic", color="0.35")
        else:
            drawn, near_d, far_d = spline_resample(fld, near, far, smooth)
            # Re-impose the minimum-span rule analytically on the fine grid.
            # Carried over by nearest neighbour it came out as a staircase;
            # the rule is a straight line, so it should be drawn as one.
            drawn = np.where(
                (far_d[None, :] - near_d[:, None]) < MIN_SPAN_KM,
                np.nan, drawn)
            im = ax.contourf(far_d, near_d, drawn, levels=levels, cmap=cmap,
                             norm=norm, extend="both")
            # Every contour_every-th level is drawn and labelled. On the
            # ratio scale only the levels above one are drawn, because the
            # surface is read as "how far above the median can a window take
            # you" and a mirror set below one doubles the ink for nothing.
            lab = (levels[levels > 1.0][::contour_every] if scale == "median"
                   else levels[::contour_every])
            cs = ax.contour(far_d, near_d, drawn, levels=lab, colors="white",
                            linewidths=0.5)
            if label_contours:
                fmt = "%.1f" if scale == "median" else "%.0f"
                # No halo. label_ink="white" matches the isolines themselves;
                # "auto" picks black or white per level from the relative
                # luminance of the colour map there, which is the safer choice
                # if the ramp runs pale at one end -- a white label on the
                # yellow of viridis is close to invisible.
                for lev in lab:
                    ink = (ink_for(cmap, norm, vmin, vmax, lev)
                           if label_ink == "auto" else label_ink)
                    for t in ax.clabel(cs, [lev], fmt=fmt, fontsize=7,
                                       inline=True):
                        t.set_color(ink)

            for xn, xf in published:
                # clip_on=False so that a window with a landward limit of zero
                # -- which several published choices have -- draws as a whole
                # circle on the axis line instead of a half one.
                ax.plot([xf], [xn], marker="o",
                        ms=5 if label_contours else 3.2, mfc="white",
                        mec="0.15", mew=0.9, zorder=5, clip_on=False)

            if windowfree is not None and k < len(windowfree):
                wf = float(windowfree[k])
                med = np.nanmedian(te)
                lvl = (wf / med if scale == "median" and np.isfinite(med)
                       else wf)
                if np.isfinite(lvl) and vmin < lvl < vmax:
                    ax.contour(far_d, near_d, drawn, levels=[lvl],
                               colors="#d2453f", linewidths=1.1, zorder=4)
                else:
                    print(f"note: window-free value {wf:g} km for {title} lies "
                          "outside the surface; contour not drawn")

        dw = window_metric(te) if te is not None else np.nan
        # The header carries the tag and the trench name and nothing else. In
        # a four-centimetre panel a long name -- Kuril-Kamchatka, New Hebrides,
        # Middle America, Peru-Chile -- plus a metric plus a profile count ran
        # off the end and collided with the neighbour, so both numbers now sit
        # inside the panel.
        head = f"{tags[k]} {title}"
        # Two lines, not one: on a four-centimetre panel a long name plus the
        # metric plus the profile count did not fit on the header line, and
        # (b) Kuril-Kamchatka and (s) Middle America ran into the annotation.
        # Stacking the two numbers halves the width the annotation needs.
        if np.isfinite(dw):
            note = (f"$\\Delta_w$ = {dw:.2f}"
                    + (f"\n{note}" if note else ""))
        note = note or ""
        # Outside the axes, above the frame. Inside the panel the tag sat over
        # the hot ridge at small seaward limits, which is the feature the
        # figure exists to show; no box is needed out here.
        ax.text(0.0, 1.015, head, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=tagsize, weight="bold", zorder=6)
        if note:
            # On the header line, right aligned, so that nothing is written
            # over the surface at all. Smaller and in dimgray, so it reads as
            # an annotation of the panel rather than part of its title.
            # Condensed face and a smaller size, so the annotation occupies
            # as little of the header line as possible; stretch="condensed"
            # resolves to Nimbus Sans Narrow, Arial Narrow or Helvetica
            # Condensed, whichever the system carries, and falls back to the
            # regular face without error if none is present.
            ax.text(1.0, 1.155, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=tagsize - 1.8, color="dimgray",
                    stretch="condensed", linespacing=1.25, zorder=6)

    grid = axes.reshape(nrows, ncols)
    for ax in grid[-1, :]:
        if ax.get_visible():
            ax.set_xlabel("Seaward window limit (km)")
    for ax in grid[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Landward window limit (km)")

    cb = fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                      fraction=0.035, pad=0.015, aspect=45, extend="both")
    cb.set_label(cb_label)
    cb.ax.tick_params(labelsize=base_font - 1)
    if cb_ticks is not None:
        # A log-normalised bar defaults to 2 x 10^0 and similar; the quantity
        # is a plain ratio, so the ticks are written as ratios and the minor
        # decade labels suppressed.
        cb.ax.yaxis.set_major_locator(FixedLocator(cb_ticks))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v:g}"))
        cb.ax.yaxis.set_minor_formatter(NullFormatter())
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    else:
        cb.locator = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10])
        cb.update_ticks()

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}."
              f"{sys.version_info.micro}, "
              f"Matplotlib {matplotlib.__version__}, "
              f"NumPy {np.__version__}. Data: GEBCO, GlobSed. "
              "Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=base_font - 1, ha="left",
             va="top")

    print(f"\n{'panel':>20s} {'Te min':>8s} {'Te max':>8s} {'median':>8s} "
          f"{'range':>8s} {'Delta_w':>8s} {'windows':>8s}")
    for te, title, note in panels:
        if te is None or not np.isfinite(te).any():
            print(f"{title:>20s} {'--':>8s}   (no usable window)")
            continue
        f = te[np.isfinite(te)]
        print(f"{title:>20s} {f.min():8.1f} {f.max():8.1f} "
              f"{np.median(f):8.1f} {f.max() - f.min():8.1f} "
              f"{window_metric(te):8.3f} {f.size:8d}")
    print("Te in km; Delta_w is Eq. (15), dimensionless.")
    return fig


# --------------------------------------------------------------------- main --
def parse_panel(s):
    if ":" not in s:
        raise SystemExit(f"--panel wants '<trench>:<profile>', got '{s}'")
    trench, pid = s.rsplit(":", 1)
    try:
        return trench.strip().lower(), int(pid)
    except ValueError:
        raise SystemExit(f"--panel profile id must be an integer, got '{pid}'")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True,
                    help="directory of <trench>_<pid>.csv profiles")
    ap.add_argument("--panel", action="append", default=None,
                    metavar="TRENCH:PID",
                    help="four times for the 2x2 single-profile mode; omit for "
                         "the twenty-trench survey")
    ap.add_argument("--trenches", nargs="*", default=None,
                    help="survey subset, in order; default all twenty")
    ap.add_argument("--nprofiles", type=int, default=24,
                    help="profiles sampled per trench in survey mode")
    ap.add_argument("--windowfree", type=float, nargs="*", default=None,
                    help="physics-informed estimate per panel, km, in order")
    ap.add_argument("--published", nargs="*", default=None,
                    help='window choices as "landward,seaward" in km')
    ap.add_argument("--no-sediment", action="store_true",
                    help="skip the Airy sediment unloading")
    ap.add_argument("--cmap", default="turbo",
                    help="matplotlib colour map; turbo for a rainbow with a "
                         "true yellow and better uniformity than jet, cmc.vik "
                         "for a perceptually uniform, colour-blind safe one")
    ap.add_argument("--contour-every", type=int, default=3,
                    help="draw and label every Nth level; lower is denser")
    ap.add_argument("--label-ink", default="white",
                    choices=("white", "black", "auto"),
                    help="contour-label colour; auto picks per level from the "
                         "luminance of the colour map")
    ap.add_argument("--smooth", type=int, default=4,
                    help="bicubic spline upsampling factor for the drawn "
                         "surface only; 1 disables it. Reported numbers are "
                         "always computed on the unsmoothed grid")
    ap.add_argument("--vlim", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="colour limits for --scale median, as ratios either "
                         "side of one; default 0.5 2.0")
    ap.add_argument("--linear-norm", action="store_true",
                    help="linear instead of geometric spacing of the ratio "
                         "levels; the ratio is multiplicative, so this is "
                         "usually wrong")
    ap.add_argument("--scale", choices=("median", "shared"), default="median",
                    help="colour axis: per-panel median-normalised, or shared "
                         "absolute Te")
    ap.add_argument("--halflen", type=float, default=HALFLEN_KM,
                    help="profile seaward reach, km; caps the seaward axis")
    ap.add_argument("--near", type=float, nargs=3,
                    metavar=("MIN", "MAX", "STEP"), default=None,
                    help="landward-limit grid, km")
    ap.add_argument("--far", type=float, nargs=3,
                    metavar=("MIN", "MAX", "STEP"), default=None,
                    help="seaward-limit grid, km")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_windowsensitivity")
    a = ap.parse_args()

    survey = not a.panel
    if not survey and len(a.panel) != 4:
        raise SystemExit(f"four --panel arguments needed, {len(a.panel)} given")

    # The survey grid is coarser on purpose: twenty small panels cannot resolve
    # a 5 km step, and the sweep is twenty times longer.
    near_d = (0.0, 50.0, 5.0) if survey else (0.0, 50.0, 5.0)
    # The seaward axis starts at MIN_SPAN_KM, not at 50: every window below it
    # is shorter than the fit can constrain, so the old first three columns
    # were blank on every panel by construction. A wedge still remains above
    # landward 0 -- at a landward limit of 50 km the shortest legal window
    # ends at 130 -- so `--far 130 550 10` removes the blank entirely, at the
    # cost of the short-window region that the figure is partly about.
    far_d = (MIN_SPAN_KM, 550.0, 20.0) if survey else (MIN_SPAN_KM, 550.0, 10.0)
    nd, fd = tuple(a.near or near_d), tuple(a.far or far_d)
    near = np.arange(nd[0], nd[1] + 1e-9, nd[2])
    far = np.arange(fd[0], fd[1] + 1e-9, fd[2])
    if far.max() > a.halflen:
        print(f"note: seaward grid clipped from {far.max():.0f} to "
              f"{a.halflen:.0f} km, the profile seaward reach")
        far = far[far <= a.halflen + 1e-9]
    if far.size == 0 or near.size == 0:
        raise SystemExit("empty window grid; check --near and --far")
    if far.max() - near.min() < MIN_SPAN_KM:
        raise SystemExit(f"no window on the grid spans {MIN_SPAN_KM:.0f} km")

    pub = []
    for p in (a.published or []):
        xn, xf = (float(v) for v in p.split(","))
        if xn < xf <= a.halflen:
            pub.append((xn, xf))
        else:
            print(f"note: published window ({xn:.0f}, {xf:.0f}) km lies "
                  f"outside the profile and is not drawn")
    if pub:
        print(f"marking {len(pub)} published window(s) supplied on the "
              f"command line")

    panels = []
    if survey:
        names = a.trenches or TRENCH_ORDER
        n_win = int(((far[None, :] - near[:, None]) >= MIN_SPAN_KM).sum())
        print(f"survey: {len(names)} trenches, up to {a.nprofiles} profiles "
              f"each, {n_win} windows per profile")
        for trench in names:
            te, used, avail = trench_surface(
                a.profiles, trench, near, far, a.nprofiles,
                sediment=not a.no_sediment)
            title = DISPLAY_NAME.get(
                trench,
                trench.replace("_", " ").replace("-", "\u2013").title())
            note = f"n = {used}/{avail}" if avail else "no profiles"
            print(f"  {trench:<18s} {used:3d} of {avail:3d} profiles used")
            panels.append((te, title, note))
        nrows, ncols = 5, 4
        figsize, tagsize, base_font = (7.6, 9.8), 7.0, 7.5
    else:
        for spec in a.panel:
            trench, pid = parse_panel(spec)
            x, d_km, fname = read_profile(a.profiles, trench, pid,
                                          sediment=not a.no_sediment)
            te, x0, why = surface(x, d_km, near, far)
            n_ok = int(np.isfinite(te).sum())
            n_try = int(((far[None, :] - near[:, None]) >= MIN_SPAN_KM).sum())
            print(f"{fname}: axis at {x0:+.0f} km, {n_ok} of {n_try} windows "
                  "fitted")
            for w, n in sorted(why.items(), key=lambda kv: -kv[1]):
                if w != "ok":
                    print(f"  rejected, {w}: {n}")
            title = f"{trench.replace('_', ' ')} {pid}"
            panels.append((te, title, ""))
        nrows, ncols = 2, 2
        figsize, tagsize, base_font = (7.0, 5.6), 8.5, 9.0

    fig = build(panels, near, far, a.windowfree, pub, a.scale, a.cmap,
                nrows, ncols, figsize, tagsize, base_font,
                vlim=tuple(a.vlim) if a.vlim else None,
                log_norm=not a.linear_norm, smooth=a.smooth,
                contour_every=a.contour_every, label_ink=a.label_ink)
    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
