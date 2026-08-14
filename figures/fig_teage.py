#!/usr/bin/env python3
"""fig_teage -- window-free elastic thickness against seafloor age.

Segments are coloured by trench, the classical isotherm curves for a
half-space cooling model are overlaid, and the fitted age relation is drawn
with a bootstrap confidence band. The script also solves the fitted
coefficient back into the isotherm it implies, which is the number the
Discussion needs: a thickness that tracks the 300 degree isotherm rather than
the 600 degree one is the signature of inelastic reduction, and that
comparison is the reason this figure exists.

Half-space cooling gives the depth to an isotherm as

    z(T, t) = 2 sqrt(kappa t) erfinv(T / T_m)

so every isotherm is a square root of age through the origin, and the fitted
relation Te = a sqrt(t) can be inverted for T directly.

INPUT
-----
CSV with a header row, one row per segment:

    trench,age_ma,te_km[,te_err]

Rows missing either the age or the thickness are dropped and counted. Nothing
is imputed.

    python3 fig_teage.py --table segment_features.csv --outdir figures
    python3 fig_teage.py --selftest          # LAYOUT CHECK ONLY
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
from matplotlib.ticker import AutoMinorLocator
from scipy.special import erfinv

SEED = 0
KAPPA = 1.0e-6          # m^2 s^-1, thermal diffusivity
T_MANTLE = 1350.0       # deg C
SEC_PER_MYR = 3.1557e13
ISOTHERMS = (300.0, 450.0, 600.0)


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
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def style(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5, pad=2)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.87")
    ax.grid(which="minor", lw=0.3, color="0.94")
    ax.set_axisbelow(True)


def clear_label_x(tt, z, age, te, lo=0.55, hi=0.96, n_try=24):
    """Index into tt at which an isotherm label sits in the emptiest space.

    Scans candidate positions along the curve and returns the one with the
    fewest data points in the box the text will occupy. A fixed fraction of
    the axis cannot do this: where the cloud sits depends on the data, and on
    a real table the 300 degree curve runs through the lower edge of it."""
    xspan = float(np.max(age)) - float(np.min(age))
    yspan = float(np.max(te)) - float(np.min(te))
    best, best_n = int(hi * (tt.size - 1)), None
    for frac in np.linspace(lo, hi, n_try):
        j = int(frac * (tt.size - 1))
        dx, dy = 0.055 * xspan, 0.055 * yspan
        n = int(np.sum((age > tt[j] - dx) & (age < tt[j] + dx) &
                       (te > z[j] - dy) & (te < z[j] + 2.2 * dy)))
        if best_n is None or n < best_n:
            best, best_n = j, n
        if n == 0 and frac > 0.80:
            break
    return best


def isotherm_km(t_ma, temp_c):
    """Depth to the given isotherm, km, for half-space cooling."""
    t = np.asarray(t_ma, dtype=float) * SEC_PER_MYR
    return 2.0 * np.sqrt(KAPPA * t) * erfinv(temp_c / T_MANTLE) / 1000.0


def implied_isotherm(a_km_per_sqrt_ma):
    """Invert the fitted coefficient of Te = a sqrt(age) for the isotherm."""
    z = a_km_per_sqrt_ma * 1000.0                      # m per sqrt(Ma)
    arg = z / (2.0 * np.sqrt(KAPPA * SEC_PER_MYR))
    if not -1.0 < arg < 1.0:
        return np.nan
    from scipy.special import erf
    return float(T_MANTLE * erf(arg))


def f(v):
    v = (v or "").strip()
    try:
        return float(v)
    except ValueError:
        return np.nan


# Display names for the legend. The tables carry canonicalised keys --
# lowercased, hyphens and spaces folded to underscores by canon() in
# build_te_segments.py -- so "Kuril-Kamchatka" arrives as "kuril_kamchatka".
# The hyphenated and two-word margins cannot be recovered by a rule, hence an
# explicit table; everything not listed falls through to title case, and any
# key that does so is reported, so a new margin is never silently mislabelled.
#
# NOTE ON VANUATU. The value tables file this margin under `new_hebrides` and
# build_te_segments.py maps the axis name `vanuatu` onto it. The manuscript,
# including Table~\ref{tab:trenches} and fig_studyarea, calls it Vanuatu, so
# that is what the legend says. Change the one line below if the paper is
# retitled to the historical name instead.
DISPLAY = {
    "izu_bonin":       "Izu-Bonin",
    "kuril_kamchatka": "Kuril-Kamchatka",
    "peru_chile":      "Peru-Chile",
    "middle_america":  "Middle America",
    "new_britain":     "New Britain",
    "new_hebrides":    "Vanuatu",
    "san_cristobal":   "San Cristobal",
}

_unmapped = set()


def pretty(key):
    """Canonical table key -> the name a reader expects in the legend."""
    k = (key or "").strip()
    if not k:
        return "unnamed"
    low = k.lower().replace(" ", "_").replace("-", "_")
    if low in DISPLAY:
        return DISPLAY[low]
    if low != k and k[:1].isupper():
        return k                      # already a display name, leave it alone
    if "_" in low:
        _unmapped.add(low)
    return " ".join(w.capitalize() for w in low.split("_"))


def read_table(path, max_rms=None):
    """Read the feature table, optionally screening on the deflection misfit.

    The residual screen is the one already applied to the thickness
    distribution in the Results: segments whose root-mean-square deflection
    error exceeds the threshold are the ones the elastic model fits poorly,
    and it is those that contribute the upper extreme of the recovered range.
    Screening here on the same stated criterion removes them by rule rather
    than by eye. It is OFF unless --max-rms is given, and the count and the
    identity of everything it removes are reported.
    """
    rows, dropped, screened = [], 0, []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            age, te = f(r.get("age_ma")), f(r.get("te_km"))
            if not (np.isfinite(age) and np.isfinite(te)) or age <= 0:
                dropped += 1
                continue
            rms = f(r.get("rms_fit_m"))
            # FLAGGED, NOT DELETED. Removing the poorly fitted segments
            # outright also removed six whole margins from the figure -- every
            # New Britain, Vanuatu, Manila and Hikurangi segment exceeds the
            # threshold -- so the sheet stopped being a picture of the belt.
            # They are drawn hollow and excluded from every statistic instead,
            # which keeps the margin coverage honest and the numbers clean.
            bad = bool(max_rms is not None and np.isfinite(rms) and rms > max_rms)
            if bad:
                screened.append((pretty(r.get("trench")), te, rms))
            rows.append(dict(trench=pretty(r.get("trench")),
                             age=age, te=te, err=f(r.get("te_err")), rms=rms,
                             screened=bad))
    if not rows:
        raise SystemExit(f"{path}: no usable rows")
    print(f"{len(rows)} segments used, {dropped} dropped for a missing "
          "age or thickness")
    if max_rms is not None:
        print(f"{len(screened)} segments exceed the {max_rms:.0f} m residual "
              f"screen: drawn hollow, excluded from the statistics")
        for t, te, rms in sorted(screened, key=lambda s: -s[2]):
            print(f"    {t:16s} Te={te:6.2f} km  rms={rms:7.1f} m")
        if not any(np.isfinite(r["rms"]) for r in rows) and not screened:
            print("  note: no rms_fit_m column in this table -- rebuild it "
                  "with build_segment_features.py --extra-cols rms_fit_m",
                  file=sys.stderr)
    if _unmapped:
        print("note: no display name for " + ", ".join(sorted(_unmapped))
              + " -- title-cased; add them to DISPLAY if a hyphen or a "
              "second word is wanted", file=sys.stderr)
    return rows


def synthetic_table():
    """Plausible age-thickness cloud. LAYOUT TESTING ONLY."""
    rng = np.random.default_rng(SEED)
    names = ["Aleutian", "Kuril-Kamchatka", "Japan", "Izu-Bonin", "Mariana",
             "Yap", "Palau", "Ryukyu", "Manila", "Philippine", "New Britain",
             "San Cristobal", "Vityaz", "New Hebrides", "Tonga", "Kermadec",
             "Hikurangi", "Puysegur", "Middle America", "Peru-Chile"]
    rows = []
    for i, n in enumerate(names):
        age0 = 8.0 + 150.0 * rng.random()
        for _ in range(rng.integers(6, 16)):
            age = max(2.0, age0 + rng.normal(0.0, 12.0))
            te = 2.9 * np.sqrt(age) + rng.normal(0.0, 3.2)
            rows.append(dict(trench=n, age=float(age), te=float(max(4.0, te)),
                             err=float(1.0 + 1.4 * rng.random())))
    return rows


def fit_sqrt(age, te, trench=None, n_boot=2000, rng=None, cluster=True):
    """Least-squares a in Te = a sqrt(age), with a bootstrap band.

    The resampling unit is the TRENCH, not the segment, whenever `trench` is
    given. Segments are nested within trenches and share that margin's age,
    sediment load and inversion setup, so they are not independent draws;
    resampling them individually treats ~130 correlated segments as ~130
    independent ones and returns an interval narrower than the data support.
    Since the fitted coefficient is inverted into an isotherm temperature that
    the Discussion leans on, that interval must not be overstated. Pass
    cluster=False for the old per-segment behaviour."""
    rng = rng or np.random.default_rng(SEED)
    x = np.sqrt(age)
    a = float(np.sum(x * te) / np.sum(x * x))
    boots = np.empty(n_boot)
    n = age.size
    if cluster and trench is not None:
        names = sorted(set(trench))
        idx = {nm: np.where(np.asarray(trench) == nm)[0] for nm in names}
        for i in range(n_boot):
            pick = rng.integers(0, len(names), len(names))
            k = np.concatenate([idx[names[j]] for j in pick])
            xb = x[k]
            boots[i] = np.sum(xb * te[k]) / np.sum(xb * xb)
    else:
        for i in range(n_boot):
            k = rng.integers(0, n, n)
            xb = x[k]
            boots[i] = np.sum(xb * te[k]) / np.sum(xb * xb)
    return a, np.percentile(boots, [2.5, 97.5])


def sqrt_skill(age, te, a):
    """Variance explained by Te = a*sqrt(t) relative to a constant Te.

    The through-origin square-root law is the physical model, but nothing
    forces it to describe the data better than "Te is the same everywhere".
    Comparing the two is the whole test of whether an age dependence has been
    resolved at all, and it must be made before the fitted coefficient is
    inverted into an isotherm: on a cloud with no age trend the least-squares
    coefficient still takes a finite value, and inverting it yields a
    temperature that looks like a result and is not one.

    Returns a value below zero when the fitted law is worse than the mean.
    """
    ss_fit = float(np.sum((te - a * np.sqrt(age)) ** 2))
    ss_mean = float(np.sum((te - te.mean()) ** 2))
    return 1.0 - ss_fit / ss_mean if ss_mean > 0 else float("nan")


def build(rows, synthetic=False, cluster=True):
    set_rc()
    fig, ax = plt.subplots(figsize=(6.9, 4.6), constrained_layout=True)

    # Every segment is plotted, so the legend covers the whole belt; only the
    # retained ones enter the fit, the median and the interquartile range.
    keep = [r for r in rows if not r.get("screened")]
    flag = [r for r in rows if r.get("screened")]
    stat = keep if keep else rows
    age = np.array([r["age"] for r in stat])
    te = np.array([r["te"] for r in stat])
    age_all = np.array([r["age"] for r in rows])
    te_all = np.array([r["te"] for r in rows])
    trenches = sorted({r["trench"] for r in rows})
    cmap = plt.get_cmap("tab20")
    if len(trenches) > 20:
        print(f"WARNING: {len(trenches)} trenches but tab20 has 20 colours; "
              f"{len(trenches) - 20} share a colour with another margin",
              file=sys.stderr)
    colour = {t: cmap(i % 20) for i, t in enumerate(trenches)}

    tt = np.linspace(1.0, 1.05 * age_all.max(), 300)
    for temp, ls in zip(ISOTHERMS, ("--", "-.", ":")):
        z = isotherm_km(tt, temp)
        ax.plot(tt, z, color="0.35", lw=0.9, ls=ls, zorder=3)
        # Label placed where the curve's own neighbourhood is emptiest, and
        # offset ABOVE the curve rather than along it. The previous fixed
        # position (86 per cent of the axis) with a 3 pt offset put the text
        # in the path of its own line: over the width of "600 C" the curve
        # climbs about 12 pt and the 450 curve about 9 pt, both far more than
        # the 3 pt clearance, so two of the three labels were struck through.
        # The white halo is insurance, not the mechanism.
        jx = clear_label_x(tt, z, age, te)
        ax.annotate(f"{temp:.0f} °C", xy=(tt[jx], z[jx]),
                    xytext=(0, 7), textcoords="offset points", fontsize=7.5,
                    color="0.30", zorder=6, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    a, ci = fit_sqrt(age, te, trench=[r["trench"] for r in stat],
                     cluster=cluster)
    skill = sqrt_skill(age, te, a)
    med = float(np.median(te))
    q1, q3 = (float(np.percentile(te, 25)), float(np.percentile(te, 75)))

    # WHICH CURVE IS DRAWN BOLD is decided by which model the data support,
    # because a rising red line through a cloud with no trend asserts with ink
    # what the caption would then have to deny in words. Where the square-root
    # law wins it is the emphasised series; where a constant wins, the constant
    # is emphasised and the square-root law is kept as a thin dashed reference
    # so the reader can still see where it falls relative to the isotherms.
    # The band and the fitted line carried no label at all, so the legend --
    # built only from the per-trench calls below -- explained neither. The
    # isotherms are direct-labelled on the curves and need no entry.
    if np.isfinite(skill) and skill > 0:
        ax.fill_between(tt, ci[0] * np.sqrt(tt), ci[1] * np.sqrt(tt),
                        color="#c0392b", alpha=0.16, lw=0, zorder=2,
                        label="95 % CI on the fit")
        ax.plot(tt, a * np.sqrt(tt), color="#c0392b", lw=2.0, zorder=5,
                label=r"fit $T_e = a\,t^{1/2}$")
    else:
        # Flat strip: the spread of the POPULATION, deliberately constant --
        # it is what the constant model asserts, and the running curves below
        # are what test that assertion against age.
        ax.fill_between(tt, q1, q3, color="#c0392b", alpha=0.10, lw=0,
                        zorder=2, label="interquartile range, all segments")
        ax.axhline(med, color="mediumblue", lw=1.2, zorder=6,
                   label=r"median $T_e$, all segments")
        ax.plot(tt, a * np.sqrt(tt), color="#c0392b", lw=1.0, ls=(0, (5, 3)),
                alpha=0.70, zorder=4,
                label=r"$T_e = a\,t^{1/2}$ for reference")


    n_err = 0
    for t in trenches:
        # Hollow markers for segments over the residual threshold, filled for
        # the rest. The label goes on whichever class this margin actually has
        # so that every margin appears in the legend exactly once, including
        # the six that consist only of poorly fitted segments.
        mk = [r for r in rows if r["trench"] == t and not r.get("screened")]
        mf = [r for r in rows if r["trench"] == t and r.get("screened")]
        if mf:
            ax.plot([r["age"] for r in mf], [r["te"] for r in mf], ls="none",
                    marker="o", ms=3.6, mfc="none", mec=colour[t], mew=0.9,
                    alpha=0.85, zorder=3.6,
                    label=(t if not mk else None))
        m = mk
        xs = [r["age"] for r in m]
        ys = [r["te"] for r in m]
        # te_err was parsed by read_table and then silently discarded, so the
        # input contract advertised an uncertainty column the figure never
        # showed. Drawn now, behind the markers, in the trench's own colour.
        es = [r["err"] for r in m]
        if any(np.isfinite(e) and e > 0 for e in es):
            ax.errorbar(xs, ys, yerr=[e if np.isfinite(e) else 0.0 for e in es],
                        ls="none", ecolor=colour[t], elinewidth=0.7,
                        capsize=1.4, capthick=0.7, alpha=0.55, zorder=3.5)
            n_err += sum(1 for e in es if np.isfinite(e) and e > 0)
        if xs:
            ax.plot(xs, ys, ls="none",
                    marker="o", ms=3.6, mfc=colour[t], mec="0.25", mew=0.35,
                    alpha=0.95, zorder=4, label=t)

    if flag:
        ax.plot([], [], ls="none", marker="o", ms=3.6, mfc="none",
                mec="0.35", mew=0.9,
                label="poor fit, excluded from the statistics")
    ax.set_xlim(0.0, 1.05 * age_all.max())
    ax.set_ylim(0.0, 1.10 * max(te_all.max(),
                                isotherm_km(age_all.max(), max(ISOTHERMS))))
    style(ax)
    ax.set_xlabel("Seafloor age (Ma)")
    ax.set_ylabel("Window-free effective elastic thickness (km)")

    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
              frameon=False, handletextpad=0.5, labelspacing=0.30,
              borderpad=0.2)

    temp = implied_isotherm(a)
    t_lo, t_hi = implied_isotherm(ci[0]), implied_isotherm(ci[1])
    # An isotherm is quoted ONLY where the square-root law actually beats a
    # constant. Where it does not, the coefficient is an artefact of forcing a
    # rising curve through a cloud with no trend, and the temperature it
    # inverts to is meaningless however precisely it is printed.
    # The annotation states the finding, whichever it is. A thickness that is
    # uniform along the whole circum-Pacific outer rise is a result about the
    # lithosphere, not a failure of the fit, and it is written as one: the
    # leading number is the quantity the data support, and the square-root law
    # follows as the comparison that was tested.
    if np.isfinite(skill) and skill > 0:
        headline = (f"$T_e = {a:.2f}\\,t^{{1/2}}$  (95 % {ci[0]:.2f}–{ci[1]:.2f})\n"
                    f"tracks the {temp:.0f} °C isotherm "
                    f"({t_lo:.0f}–{t_hi:.0f} °C)\n"
                    f"and explains {100*skill:.0f} % of the variance in $T_e$")
    else:
        headline = (f"$T_e$ = {med:.1f} km, uniform with plate age\n"
                    f"(interquartile range {q1:.1f}–{q3:.1f} km, "
                    f"{len(stat)} segments)\n"
                    f"a constant describes the population better than\n"
                    f"$T_e = a\\,t^{{1/2}}$, so no single isotherm is tracked")
    ax.text(0.025, 0.955,
            headline,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            color="#c0392b", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", lw=0.5,
                      alpha=0.92))

    if synthetic:
        ax.text(0.55, 0.35, "SYNTHETIC", transform=ax.transAxes, ha="center",
                va="center", fontsize=18, color="0.80", rotation=22,
                alpha=0.6, zorder=8)

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}, "
              f"Matplotlib {matplotlib.__version__}, SciPy, NumPy. "
              "Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=8, ha="left", va="top")

    print(f"fit Te = {a:.3f} sqrt(age),  95 % CI {ci[0]:.3f}-{ci[1]:.3f} "
          f"({'by trench' if cluster else 'per segment'})")
    print(f"skill of sqrt(t) against a constant Te: {skill:+.3f}")
    if np.isfinite(skill) and skill > 0:
        print(f"implied isotherm {temp:.0f} deg C ({t_lo:.0f}-{t_hi:.0f})")
    else:
        print(f"Te uniform with age: median {med:.2f} km, IQR {q1:.2f}-{q3:.2f}. "
              f"A constant\n  outperforms the square-root law, so the fitted "
              f"coefficient is not inverted\n  into an isotherm. This is a "
              f"statement about the recovered population; if a\n  thermal "
              f"control is expected, the ablation on the physics weighting is "
              f"the\n  experiment that would show whether Te is data-driven "
              f"per segment.", file=sys.stderr)
    if n_err:
        print(f"{n_err} segments carry a te_err and are drawn with error bars")
    print(f"{len(trenches)} trenches;  age {age.min():.0f}-{age.max():.0f} Ma;"
          f"  Te {te.min():.1f}-{te.max():.1f} km")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_teage")
    ap.add_argument("--max-rms", type=float, default=None,
                    help="drop segments whose rms_fit_m exceeds this, in "
                         "metres; 150 reproduces the residual screen already "
                         "applied to the thickness distribution in the Results")
    ap.add_argument("--bootstrap", choices=("trench", "segment"),
                    default="trench",
                    help="resampling unit for the confidence band; segments "
                         "are nested in trenches, so 'trench' is the honest "
                         "default and 'segment' reproduces the old behaviour")
    a = ap.parse_args()

    if a.selftest:
        fig = build(synthetic_table(), synthetic=True,
                    cluster=(a.bootstrap == "trench"))
        name = a.name + "_SELFTEST"
    else:
        if not a.table:
            ap.error("--table is required (or use --selftest)")
        fig = build(read_table(a.table, max_rms=a.max_rms),
                    cluster=(a.bootstrap == "trench"))
        name = a.name

    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)


if __name__ == "__main__":
    main()
