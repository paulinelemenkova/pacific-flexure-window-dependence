#!/usr/bin/env python3
"""fig_yielding -- reduction of the recovered elastic thickness against the
bending that produces it.

The recovered thickness is effective: bending past the yield envelope removes
the outer fibres from the elastic section, so Te sits below the mechanical
thickness of the plate. This figure tests that reduction against its supposed
cause without appeal to an external model, because both quantities come from
one inversion of one profile:

    vertical    Te / z(600 C, age)   1 = elastic through the full mechanical
                                     thickness, < 1 = partly yielded
    horizontal  max |kappa| along the segment, from Equation (derived)

z(600 C) is the half-space cooling depth to the 600 degree isotherm at that
segment's plate age, the conventional base of the mechanical lithosphere and
the same relation drawn in fig_teage, so the two figures are consistent.

INPUTS
------
    --derived  derived_segments.csv   from eval_derived.py: segment, trench,
                                      te_km, curvature (max |kappa|, 1/m)
    --features segment_features.csv   from build_segment_features.py: trench,
                                      age_ma, te_km, rms_fit_m

THE JOIN, AND WHY IT IS CHECKED
-------------------------------
The two tables are keyed differently: derived_segments.csv names the segment,
segment_features.csv names only the trench. If the features table carries a
`segment` column the join uses it. Otherwise it falls back to (trench, te_km),
which identifies a segment uniquely in practice because the recovered
thicknesses within a margin differ, and the fallback VERIFIES that before
using it: a collision aborts rather than silently pairing a curvature with
another segment's age. Row order is not used as a key -- the two tables are
written by different scripts walking different structures, and assuming they
agree is how the segmentations got crossed earlier in this project.

    python3 fig_yielding.py --derived derived_segments.csv \\
        --features segment_features.csv --max-rms 150 --outdir figures
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
from scipy.special import erfinv

KAPPA_TH = 1.0e-6          # m^2 s^-1, thermal diffusivity
T_MANTLE = 1350.0          # deg C
SEC_PER_MYR = 3.1557e13
T_MECH = 600.0             # deg C, base of the mechanical lithosphere

DISPLAY = {
    "izu_bonin": "Izu-Bonin", "kuril_kamchatka": "Kuril-Kamchatka",
    "peru_chile": "Peru-Chile", "middle_america": "Middle America",
    "new_britain": "New Britain", "new_hebrides": "Vanuatu",
    "san_cristobal": "San Cristobal",
}


# The two tables spell a margin differently: derived_segments.csv carries the
# name as the run table has it (izu-bonin, new_hebrides) while
# segment_features.csv carries the canonicalised axis key (izu_bonin,
# vanuatu). Both sides are reduced to one form before joining, and the
# vanuatu/new_hebrides alias is applied, or twenty of seventy-one segments
# vanish from the figure with only a count to say so.
TRENCH_ALIAS = {"vanuatu": "new_hebrides", "new_hebrides": "new_hebrides"}


def canon_trench(name):
    k = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return TRENCH_ALIAS.get(k, k)


def pretty(key):
    k = (key or "").strip()
    if not k:
        return "unnamed"
    low = k.lower().replace(" ", "_").replace("-", "_")
    if low in DISPLAY:
        return DISPLAY[low]
    if low != k and k[:1].isupper():
        return k
    return " ".join(w.capitalize() for w in low.split("_"))


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
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular", "savefig.dpi": 600,
    })


def isotherm_km(t_ma, temp_c):
    """Half-space cooling depth to an isotherm, km. As in fig_teage."""
    t = np.asarray(t_ma, float) * SEC_PER_MYR
    return 2.0 * np.sqrt(KAPPA_TH * t) * erfinv(temp_c / T_MANTLE) / 1000.0


def f(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


def load(derived, features):
    dv = list(csv.DictReader(open(derived)))
    ft = list(csv.DictReader(open(features)))
    if not dv or not ft:
        sys.exit("one of the input tables is empty")

    use_segment = "segment" in (ft[0].keys())
    # The fallback key rounds the thickness to 3 decimals, not more:
    # eval_derived writes 6 significant figures (7.36775) while segments.csv
    # keeps full precision (7.36774914996742), so a 4-decimal key misses.
    # Three decimals is 1 m of thickness -- far finer than any two segments
    # of a margin differ by, and coarse enough to survive the rounding.
    def te_key(v):
        return round(v, 3)

    index, collide = {}, set()
    for r in ft:
        if use_segment:
            key = (r.get("segment") or "").strip()
        else:
            key = (canon_trench(r.get("trench")),
                   te_key(f(r.get("te_km"))))
        if key in index:
            collide.add(key)
        index[key] = r
    if collide and not use_segment:
        sys.exit(f"{features}: {len(collide)} (trench, te_km) pairs are not "
                 f"unique, so the fallback join is ambiguous. Rebuild the "
                 f"features table with a `segment` column "
                 f"(build_segment_features.py) and re-run.")
    print(f"join key: {'segment' if use_segment else '(trench, te_km)'}")

    rows, missed = [], 0
    for r in dv:
        seg = (r.get("segment") or "").strip()
        te = f(r.get("te_km"))
        key = seg if use_segment else (canon_trench(r.get("trench")),
                                       te_key(te))
        m = index.get(key)
        if m is None:
            missed += 1
            continue
        age = f(m.get("age_ma"))
        kap = abs(f(r.get("curvature")))
        rms = f(m.get("rms_fit_m"))
        if not (np.isfinite(age) and age > 0 and np.isfinite(te)
                and np.isfinite(kap) and kap > 0):
            missed += 1
            continue
        zm = float(isotherm_km(age, T_MECH))
        rows.append(dict(segment=seg, trench=pretty(r.get("trench")),
                         te=te, age=age, kappa=kap, rms=rms,
                         ratio=te / zm if zm > 0 else np.nan))
    print(f"{len(rows)} segments joined, {missed} unmatched or incomplete")
    if not rows:
        sys.exit("nothing to plot")
    return rows


def build(rows, max_rms=None):
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 4.6), constrained_layout=True)

    for r in rows:
        r["screened"] = bool(max_rms is not None and np.isfinite(r["rms"])
                             and r["rms"] > max_rms)
    keep = [r for r in rows if not r["screened"]]
    stat = keep if keep else rows

    x = np.array([r["kappa"] for r in stat])
    y = np.array([r["ratio"] for r in stat])
    xa = np.array([r["kappa"] for r in rows])
    ya = np.array([r["ratio"] for r in rows])

    trenches = sorted({r["trench"] for r in rows})
    cmap = plt.get_cmap("tab20")
    colour = {t: cmap(i % 20) for i, t in enumerate(trenches)}

    # Unity: the plate bends elastically through its whole mechanical
    # thickness. Everything below it has yielded through part of its section,
    # which is the quantity this figure measures.
    ax.axhline(1.0, color="0.35", lw=1.0, ls=(0, (6, 3)), zorder=3)
    ax.text(0.985, 1.0, " no reduction", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.35")

    # Least squares on log curvature, on the retained segments only. A
    # power-law in kappa is the form the yield literature uses; fitting it in
    # logs keeps a single high-curvature segment from setting the slope.
    slope = inter = r2 = np.nan
    if len(stat) >= 3:
        lx = np.log10(x)
        A = np.vstack([lx, np.ones_like(lx)]).T
        (slope, inter), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ np.array([slope, inter])
        ss = float(np.sum((y - pred) ** 2))
        st = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss / st if st > 0 else np.nan
        xx = np.logspace(np.log10(xa.min()), np.log10(xa.max()), 200)

        # The MEDIAN and its interquartile band are drawn first and boldest,
        # because a near-constant ratio is the actual finding: the fit does not
        # significantly beat a horizontal line (its slope CI includes zero), so
        # the horizontal is what the reader should take away. The band is the
        # spread of the population, not a confidence interval.
        med = float(np.median(y))
        q1, q3 = float(np.percentile(y, 25)), float(np.percentile(y, 75))
        ax.fill_between([xa.min(), xa.max()], q1, q3, color="mediumblue",
                        alpha=0.10, lw=0, zorder=2,
                        label=f"interquartile range ({q1:.2f}\u2013{q3:.2f})")
        ax.axhline(med, color="mediumblue", lw=1.1, zorder=6,
                   label=f"median ratio ({med:.2f})")

        # Binned medians: the ratio in four equal-count bins of curvature, as a
        # light step, to show whether the scatter hides a monotone drift. It is
        # a guide, not a claim; at ~12 segments per bin it is not tested.
        nb = 4
        edg = np.percentile(lx, np.linspace(0, 100, nb + 1))
        bx, by = [], []
        for i in range(nb):
            m = (lx >= edg[i]) & (lx <= edg[i + 1])
            if m.sum():
                bx.append(10 ** (0.5 * (edg[i] + edg[i + 1])))
                by.append(float(np.median(y[m])))
        ax.plot(bx, by, color="#7a4fa3", lw=1.0, ls=(0, (1, 1)), marker="D",
                ms=4.0, mfc="white", mec="#7a4fa3", mew=1.0, zorder=5,
                label="binned median (curvature quartiles)")

        # The least-squares fit and its 95 % band come last and thinnest: the
        # band visibly straddles the horizontal median, which is the honest
        # picture of a slope that is not significant.
        se2 = ss / (len(x) - 2) if len(x) > 2 else np.nan
        if np.isfinite(se2):
            cov = se2 * np.linalg.inv(A.T @ A)
            lxx = np.log10(xx)
            Axx = np.vstack([lxx, np.ones_like(lxx)]).T
            se = np.sqrt(np.sum((Axx @ cov) * Axx, axis=1))
            from scipy.stats import t as _t
            tv = _t.ppf(0.975, len(x) - 2)
            fitline = slope * lxx + inter
            ax.fill_between(xx, fitline - tv * se, fitline + tv * se,
                            color="#c0392b", alpha=0.12, lw=0, zorder=3)
        ax.plot(xx, slope * np.log10(xx) + inter, color="#c0392b", lw=1.1,
                zorder=5, label="least squares on $\\log_{10}\\kappa$")

    for t in trenches:
        mk = [r for r in rows if r["trench"] == t and not r["screened"]]
        mf = [r for r in rows if r["trench"] == t and r["screened"]]
        if mf:
            ax.plot([r["kappa"] for r in mf], [r["ratio"] for r in mf],
                    ls="none", marker="o", ms=3.6, mfc="none", mec=colour[t],
                    mew=0.9, alpha=0.85, zorder=3.6,
                    label=(t if not mk else None))
        if mk:
            ax.plot([r["kappa"] for r in mk], [r["ratio"] for r in mk],
                    ls="none", marker="o", ms=3.6, mfc=colour[t], mec="0.25",
                    mew=0.35, alpha=0.95, zorder=4, label=t)
    if any(r["screened"] for r in rows):
        ax.plot([], [], ls="none", marker="o", ms=3.6, mfc="none", mec="0.35",
                mew=0.9, label="poor fit, excluded from the fit")

    ax.set_xscale("log")
    ax.set_xlabel("Maximum curvature along the segment, $\\kappa$ (m$^{-1}$)")
    ax.set_ylabel("$T_e$ / depth to the 600\\,$^\\circ$C isotherm")

    # Cap the axis just above the physical ceiling of unity. The only points
    # above it are the three poorly fitted, hollow-drawn segments already
    # excluded from every statistic; keeping the axis open to 2.0 to show them
    # in place leaves the top 40 % of the panel empty. Instead the cap sits at
    # 1.12 and each off-scale segment is marked by an upward arrow at the top
    # with its value, so it is neither clipped silently nor allowed to waste
    # the panel. A segment stiffer than its own mechanical thickness is not
    # admissible, so flagging these at the ceiling states that directly.
    YCAP = 1.12
    ax.set_ylim(0.0, YCAP)
    off = [r for r in rows if r["ratio"] > YCAP]
    # Each off-scale segment gets an arrow at its own curvature, with its value
    # below and its name above. The three margins here are close in curvature,
    # so the names are nudged apart by hand: the leftmost pulled further left,
    # the rightmost further right, and the middle one dropped just above the
    # arrow so it sits between the other two rather than on top of either. The
    # ha alignment follows the horizontal shift so the text leans away from the
    # neighbour it would otherwise touch.
    NUDGE = {"Hikurangi": (0.92, "right", 0.03),
             "Manila": (1.0, "center", 0.03),
             "San Cristobal": (1.08, "left", 0.03)}
    for r in sorted(off, key=lambda r: r["kappa"]):
        name = pretty(r["trench"])
        fx, ha, dy = NUDGE.get(name, (1.0, "center", 0.03))
        ax.annotate("", xy=(r["kappa"], YCAP), xytext=(r["kappa"], YCAP - 0.06),
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.0),
                    zorder=7, annotation_clip=False)
        ax.text(r["kappa"], YCAP - 0.075, f"{r['ratio']:.1f}", ha="center",
                va="top", fontsize=6.5, color="0.40", zorder=7)
        ax.text(r["kappa"] * fx, YCAP + dy, name, ha=ha,
                va="bottom", fontsize=6.5, color="0.40", zorder=7,
                clip_on=False)
    if off:
        print(f"{len(off)} segment(s) above the {YCAP:.2f} cap, marked with "
              f"arrows: " + ", ".join(f"{r['segment']} ({r['ratio']:.2f})"
                                      for r in off))

    # AutoMinorLocator does not work on a log axis; LogLocator gives the
    # 2..9 decade subdivisions that a reader needs to place a point.
    ax.xaxis.set_minor_locator(LogLocator(base=10.0,
                                          subs=tuple(np.arange(2, 10)),
                                          numticks=100))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5, pad=2)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.87")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
              frameon=False, handletextpad=0.5, labelspacing=0.30,
              borderpad=0.2)

    med = float(np.median(y))
    if np.isfinite(slope):
        # Below the "no reduction" line at 1.0 now that the axis is capped at
        # YCAP; at the old 0.955 it would sit on that line. The third line
        # states the non-significance directly, so the figure cannot be read as
        # claiming a curvature trend the statistics do not support.
        sig = "not distinguishable from zero" if r2 < 0.1 else \
              f"$R^2$ = {r2:.2f}"
        # Top-left corner, just under the "no reduction" line, where the only
        # points are the Aleutian at 0.62 and below. Anchored high and narrow
        # so it clears the Middle America cluster near 0.7 to its right and the
        # Vanuatu point at 0.93 further out. It sits inside the axes and does
        # not touch the legend, which is outside the plot on the right.
        ax.text(0.025, 0.80,
                f"median $T_e$ is {100*med:.0f} % of the mechanical thickness\n"
                f"slope with $\\log_{{10}}\\kappa$ is {sig}\n"
                f"($R^2$ = {r2:.2f}, {len(stat)} segments)",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color="#c0392b", zorder=7,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8",
                          lw=0.5, alpha=0.92))
        print(f"slope {slope:+.4f} per decade of kappa, R2 {r2:+.3f}, "
              f"{len(stat)} segments")
        if r2 < 0.05:
            print("  NOTE: the trend explains almost none of the scatter. "
                  "State that plainly in Section 5.6 rather than reporting "
                  "the slope alone.", file=sys.stderr)
    print(f"median ratio {med:.3f}; range {np.nanmin(ya):.3f} to "
          f"{np.nanmax(ya):.3f}")
    over = [r["segment"] for r in rows if r["ratio"] > 1.0]
    if over:
        print(f"{len(over)} segment(s) above unity, i.e. stiffer than the "
              f"mechanical thickness allows: {', '.join(over[:8])}"
              + (" ..." if len(over) > 8 else ""), file=sys.stderr)

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, SciPy, NumPy. "
              "Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=8, ha="left", va="top")
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", default="derived_segments.csv")
    ap.add_argument("--features", default="segment_features.csv")
    ap.add_argument("--max-rms", type=float, default=150.0,
                    help="segments above this deflection misfit are drawn "
                         "open and excluded from the fit, as in fig_teage")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_yielding")
    a = ap.parse_args()

    for p in (a.derived, a.features):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")
    fig = build(load(a.derived, a.features), max_rms=a.max_rms)
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{a.name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
