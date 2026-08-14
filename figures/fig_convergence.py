#!/usr/bin/env python3
"""fig_convergence -- convergence of the composite loss and of the recovered
elastic thickness through training.

Two stacked panels sharing the iteration axis:

  (a) the composite loss and its three weighted components, on a logarithmic
      scale, with the Adam to L-BFGS transition marked
  (b) the recovered elastic thickness for one segment, with the spread across
      the ensemble

The panels are stacked rather than drawn on a twin axis. A twin axis would put
a logarithmic loss and a linear thickness on the same frame, and a reader
cannot tell at a glance which curve belongs to which scale; stacking costs one
extra centimetre and removes the ambiguity.

WHAT THE AXIS COUNTS
--------------------
Iterations, not epochs. Training is not epoch-based: collocation points are
resampled every iteration and the segment batch is redrawn, so there is no pass
over a fixed dataset to count. A history file whose first column is named
`epoch` is still accepted, and relabelled with a note.

TWO STAGES
----------
Optimisation runs Adam first and L-BFGS afterwards (Section 4.5). The two
stages descend at different rates, and a single curve drawn through both
invites the reader to read the break as instability instead of as a change of
optimiser. Pass --stage-switch with the iteration at which L-BFGS takes over
and the transition is drawn and labelled on both panels.

WEIGHTED OR UNWEIGHTED COMPONENTS
--------------------------------
train_pinn.py records the three terms UNWEIGHTED, and loss_total as their plain
sum -- not the weighted objective the optimiser descends. Pass --components to
declare which convention the file uses; the default is `unweighted`, matching
the training loop as written.

The sum test alone cannot tell the two apart: unweighted components sum to an
unweighted total just as weighted ones sum to a weighted total, so a file
written either way passes it. The test is therefore used only to catch the
genuine error -- components that do not sum to the recorded total at all -- and
the declared convention sets the legend labels, so a reader is never told that
a curve carries a weight it does not.

INPUT
-----
A CSV written by the training loop, one row per recorded iteration:

    iteration,loss_total,loss_data,loss_phys,loss_bc,te_km[,te_lo,te_hi]

`te_lo` and `te_hi` are optional; supply them and the ensemble spread is drawn
as a band, omit them and only the central trajectory appears. Nothing is
invented when a column is absent.

ENSEMBLE SPREAD
---------------
The band in panel (b) is the dispersion of the recovered thickness across
independently seeded members (Section 4.7), and it can only be drawn when those
members exist. Give --ensemble a glob over the per-seed history files and the
band is computed from them:

    python3 fig_convergence.py --history runs/r02/history.csv \\
        --ensemble 'runs/r*/history.csv' --stage-switch 4186 \\
        --segment aleutian_01 --outdir figures

With a single member the glob matches one file, no dispersion exists, and the
script says so and draws the central trajectory alone. It never synthesises a
band: a shaded interval on a published figure is a claim about reproducibility,
and one run cannot support it.

    python3 fig_convergence.py --history runs/r02/history.csv \\
        --stage-switch 4186 --segment aleutian_01 --outdir figures
    python3 fig_convergence.py --selftest        # LAYOUT CHECK ONLY
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
# R/X11 numbered names (royalblue2, red3, ...) are unknown to matplotlib;
# these are their hex values. This is NOT Okabe-Ito -- the palette was chosen
# by eye, and red3 against green4 is the deuteranopia pair. Line style carries
# every distinction independently of colour (the composite alone is solid), so
# the panel survives greyscale and CVD on style rather than on hue.
C_TOTAL, C_DATA, C_PHYS, C_BC = "#436EEE", "#CD0000", "#FFC125", "#008B00"
C_TE = "#BF3EFF"
#   royalblue2  red3        goldenrod1  green4       darkorchid1

# Two line weights and no more: the emphasised series and everything else.
LW_EMPH, LW_REST = 2.0, 1.2

# Stopping criterion of Section 4.5, reproduced here so that the annotation and
# the manuscript cannot drift apart.
STOP_REL_TOL = 1.0e-4
STOP_PATIENCE = 2000


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
        "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.labelpad": 2, "axes.linewidth": 0.8,
        # mathtext ignores font.sans-serif and would otherwise ship DejaVu
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans", "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold", "mathtext.sf": "Nimbus Sans",
        "mathtext.tt": "Nimbus Sans", "mathtext.cal": "Nimbus Sans",
        "mathtext.default": "regular",
        "savefig.dpi": 600,
    })


def style_axes(ax, log=False):
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(LogLocator(base=10.0,
                                              subs=tuple(np.arange(2, 10) / 10 * 10),
                                              numticks=100))
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.tick_params(which="major", length=4.5, pad=2)
    ax.tick_params(which="minor", length=2.5)
    ax.grid(which="major", lw=0.5, color="0.85")
    ax.grid(which="minor", lw=0.3, color="0.93")
    ax.set_axisbelow(True)


def covers_data(fig, ax, artist, data_xy):
    """Rule 1/7: count data points inside a legend or label box."""
    fig.canvas.draw()
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    pts = ax.transData.transform(data_xy)
    return int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())


def read_history(path):
    """Read the training history, tolerating `epoch` as a name for `iteration`.

    A row whose values do not parse is dropped whole. Dropping the offending
    value alone would leave the columns at different lengths, which surfaces
    much later as an unreadable broadcasting error inside Matplotlib.
    """
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        fields = [f.strip() for f in (reader.fieldnames or [])]
        if "iteration" not in fields and "epoch" in fields:
            print("note: column 'epoch' read as 'iteration'")
        for row in reader:
            clean, ok = {}, True
            for k, v in row.items():
                if k is None:
                    continue
                k = k.strip()
                if k == "epoch":
                    k = "iteration"
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok and clean:
                rows.append(clean)

    if not rows:
        raise SystemExit(f"{path}: no numeric rows")
    need = ("iteration", "loss_total", "loss_data", "loss_phys", "loss_bc",
            "te_km")
    missing = [k for k in need if k not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {', '.join(missing)}")

    n = len(rows)
    h = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if k in r]
        if len(vals) != n:
            raise SystemExit(f"{path}: column '{k}' is ragged")
        h[k] = np.asarray(vals, dtype=float)
    return h


def check_weighting(h, convention):
    """Check the components sum to the composite, and report the convention.

    This detects a genuinely broken file -- components that do not sum to the
    recorded total -- and nothing more. It cannot distinguish weighted from
    unweighted components, because both sum to their own total; the caller
    declares which, and the declaration drives the legend.
    """
    parts = h["loss_data"] + h["loss_phys"] + h["loss_bc"]
    total = h["loss_total"]
    scale = float(np.median(np.abs(total)))
    if scale <= 0:
        return
    err = float(np.median(np.abs(parts - total))) / scale
    if err > 1.0e-3:
        print(f"WARNING: components do not sum to the composite (median "
              f"relative discrepancy {err:.2e}). One of the four columns is "
              f"not what the header says it is; the panel will mislead.")
    else:
        print(f"components sum to the composite (residual {err:.1e}); "
              f"labelled as {convention}")


def read_ensemble(pattern, h):
    """Per-iteration spread of the recovered thickness across seeded members.

    Returns (lo, hi, n_members) or (None, None, n) when fewer than two members
    match. The band is the full range across members at each iteration, which
    for a handful of seeds reports what was actually observed rather than a
    dispersion estimated from too few samples. Members are truncated to the
    shortest common history, since a run stopped by the patience rule need not
    stop at the same iteration as its neighbours.
    """
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        print(f"ensemble: {len(paths)} member(s) matched {pattern!r} -- no "
              f"dispersion exists, so no band is drawn. Run further seeds "
              f"(Section 4.7) and the band appears without further change.")
        return None, None, len(paths)

    series = []
    for p in paths:
        try:
            m = read_history(p)
        except SystemExit as exc:
            print(f"ensemble: skipping {p} ({exc})")
            continue
        series.append(m["te_km"])
    n = min(len(v) for v in series)
    if n < len(h["te_km"]):
        print(f"ensemble: truncated to {n} iterations, the shortest member")
    stack = np.vstack([v[:n] for v in series])
    print(f"ensemble: {len(series)} members, band is the per-iteration range")
    return stack.min(axis=0), stack.max(axis=0), len(series)


def rolling_median(y, win):
    """Centred rolling median, edges held at the nearest full window."""
    win = max(int(win), 1)
    if win <= 1 or y.size < win:
        return y.copy()
    pad = win // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    out = np.empty_like(y)
    for i in range(y.size):
        out[i] = np.median(padded[i:i + win])
    return out


def settle_point(h):
    """First iteration at which the stopping criterion of Section 4.5 is met.

    The criterion is applied to a rolling median of the composite loss, not to
    the raw trace. A stochastic loss keeps setting new record minima purely
    through sampling noise, so a running best-so-far on the raw values never
    stalls and the criterion would report that training never converged.
    """
    it, lt = h["iteration"], h["loss_total"]
    step = float(np.median(np.diff(it))) if it.size > 1 else 1.0
    span = max(int(round(STOP_PATIENCE / step)), 1)
    if it.size <= span:
        return None
    smooth = rolling_median(lt, max(span // 10, 3))
    best = np.minimum.accumulate(smooth)
    for k in range(span, it.size):
        ref = best[k - span]
        if ref <= 0:
            continue
        if (ref - best[k]) / ref < STOP_REL_TOL:
            return float(it[k])
    return None


def synthetic_history(n=30000, switch=20000):
    """Plausible two-stage trace. LAYOUT TESTING ONLY -- never published."""
    rng = np.random.default_rng(SEED)
    it = np.arange(1, n + 1, dtype=float)
    j = lambda a, tau, floor: a * np.exp(-it / tau) + floor
    ld = np.abs(j(3.0e-1, 2000.0, 4.0e-4) * (1 + 0.10 * rng.standard_normal(n)))
    lp = np.abs(j(9.0e-1, 1400.0, 1.1e-4) * (1 + 0.14 * rng.standard_normal(n)))
    lb = np.abs(j(4.0e-2, 1000.0, 2.0e-5) * (1 + 0.12 * rng.standard_normal(n)))
    # L-BFGS drops the residual by a further order of magnitude, then plateaus,
    # so that the stopping criterion is reached inside the trace and the
    # settle_point code path is exercised by the self-test
    tail = it > switch
    drop = np.where(tail, 0.1 + 0.9 * np.exp(-(it - switch) / 700.0), 1.0)
    ld, lp, lb = ld * drop, lp * drop, lb * drop
    lt = ld + lp + lb                      # weighted components, so they sum
    te = 47.0 - 19.0 * np.exp(-it / 3000.0) + \
        0.35 * np.exp(-it / 6000.0) * rng.standard_normal(n)
    half = 2.6 * np.exp(-it / 5000.0) + 0.55
    return dict(iteration=it, loss_total=lt, loss_data=ld, loss_phys=lp,
                loss_bc=lb, te_km=te, te_lo=te - half, te_hi=te + half)


def build(h, synthetic=False, switch=None, segment="", convention="unweighted",
          ensemble=None):
    set_rc()
    check_weighting(h, convention)
    wd = "\\lambda_d" if convention == "weighted" else ""
    wp = "\\lambda_p" if convention == "weighted" else ""
    wb = "\\lambda_b" if convention == "weighted" else ""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.9, 4.6), sharex=True,
                                   height_ratios=[1.55, 1.0],
                                   constrained_layout=True)
    it = h["iteration"]

    # a logarithmic axis cannot show a non-positive loss; mask explicitly
    # instead of letting Matplotlib drop the points without saying so
    def pos(y):
        bad = ~(y > 0)
        if bad.any():
            print(f"note: {int(bad.sum())} non-positive loss values masked")
        return np.where(bad, np.nan, y)

    # The composite is drawn UNDER the data term and at the same weight as it.
    # Composite and data run within a factor of two of each other for the whole
    # run, and a 2.0 pt composite oscillating at batch frequency smeared over
    # the data term and hid it. Panel (a) therefore carries no emphasised
    # series: the composite is identified by being the only solid line.
    ax1.plot(it, pos(h["loss_total"]), color=C_TOTAL, lw=LW_REST, zorder=4,
             label="composite  $\\mathcal{L}$")
    ax1.plot(it, pos(h["loss_data"]), color=C_DATA, lw=LW_REST, ls="--", zorder=6,
             label="data  $" + wd + "\\mathcal{L}_\\mathrm{data}$")
    ax1.plot(it, pos(h["loss_phys"]), color=C_PHYS, lw=LW_REST, ls="-.", zorder=4,
             label="physics  $" + wp + "\\mathcal{L}_\\mathrm{phys}$")
    ax1.plot(it, pos(h["loss_bc"]), color=C_BC, lw=LW_REST, ls=":", zorder=4,
             label="boundary  $" + wb + "\\mathcal{L}_\\mathrm{bc}$")
    style_axes(ax1, log=True)
    ax1.set_ylabel("Loss (dimensionless)")
    ax1.set_xlim(it.min(), it.max())
    # Headroom for the legend. The placer below can only choose among corners
    # that exist; on a run that plateaus, every corner holds data and the best
    # it can do is cover one point. A decade and a half of empty axis above the
    # curves gives it somewhere to go, and costs nothing on a log scale.
    _top = np.nanmax(pos(h["loss_total"]))
    if np.isfinite(_top):
        ax1.set_ylim(top=10.0 ** (np.log10(_top) + 1.4))

    if ensemble is not None and ensemble[0] is not None:
        lo, hi, nmem = ensemble
        k = len(lo)
        ax2.fill_between(it[:k], lo, hi, color=C_TE, alpha=0.18, lw=0, zorder=2,
                         label=f"ensemble spread ({nmem} members)")
    elif "te_lo" in h and "te_hi" in h:
        ax2.fill_between(it, h["te_lo"], h["te_hi"], color=C_TE, alpha=0.18,
                         lw=0, zorder=2, label="ensemble spread")
    ax2.plot(it, h["te_km"], color=C_TE, lw=LW_REST, zorder=4,
             label="recovered $T_e$")
    tail = max(10, len(it) // 50)
    te_final = float(np.median(h["te_km"][-tail:]))
    ax2.axhline(te_final, color="0.45", lw=0.8, ls=(0, (5, 3)), zorder=3)
    style_axes(ax2)
    ax2.set_ylabel("Recovered $T_e$ (km)")
    ax2.set_xlabel("Iteration")

    # the optimiser transition, on both panels, so the break in (a) and the
    # settling in (b) are read against the same event
    if switch is not None and it.min() < switch < it.max():
        for ax in (ax1, ax2):
            ax.axvline(switch, color="0.35", lw=0.9, ls=(0, (4, 2.5)),
                       zorder=3)
        # placed ABOVE the frame, not inside it: the top of panel (a) carries
        # the legend and the bottom carries the descending boundary term, so
        # the only reliably empty strip is outside the axes
        for text, dx, ha in (("Adam", -4, "right"), ("L-BFGS", 4, "left")):
            ax1.annotate(text, xy=(switch, 1.0),
                         xycoords=("data", "axes fraction"),
                         xytext=(dx, 3), textcoords="offset points",
                         ha=ha, va="bottom", fontsize=8, color="0.30",
                         zorder=6, annotation_clip=False)

    stop = settle_point(h)
    if stop is not None:
        print(f"stopping criterion first met at iteration {stop:.0f} "
              f"({STOP_REL_TOL:g} over {STOP_PATIENCE} iterations)")
    else:
        print("stopping criterion not met within the recorded history")

    # The corner a legend should occupy depends on the run, not on the
    # designer's expectation of it. A descending loss leaves the upper right
    # free; a loss that plateaus early does not, and the fixed placement then
    # buried the curves it was labelling. Each legend is therefore built at
    # every candidate corner, the covered data points counted, and the corner
    # with the fewest retained.
    def place(ax, ys, **kw):
        xy = np.vstack([np.column_stack([it, h[k]]) for k in ys])
        best, best_n, best_loc = None, None, None
        for loc in ("upper right", "upper left", "lower right", "lower left",
                    "center right", "center left"):
            leg = ax.legend(loc=loc, **kw)
            n = covers_data(fig, ax, leg, xy)
            if best_n is None or n < best_n:
                best_n, best_loc = n, loc
            leg.remove()
            if n == 0:
                break
        leg = ax.legend(loc=best_loc, **kw)
        if best_n:
            print(f"  legend placed {best_loc}, still covers {best_n} points")
        else:
            print(f"  legend placed {best_loc}, clear of the data")
        return leg, best_n, best_loc

    leg1, n1, loc1 = place(ax1, ("loss_total", "loss_data", "loss_phys", "loss_bc"),
                     ncol=2, framealpha=0.92, borderpad=0.4, handlelength=2.6,
                     handletextpad=0.7, labelspacing=0.35, columnspacing=1.6)
    leg2, n2, loc2 = place(ax2, ("te_km",), framealpha=0.92, borderpad=0.4,
                     handlelength=2.6, handletextpad=0.7)
    # the band, where drawn, is part of the data the tag must not cover
    if ensemble is not None and ensemble[0] is not None:
        h["te_lo"], h["te_hi"] = ensemble[0], ensemble[1]
    hits = n1 + n2
    print(f"overlap check -- legends cover {hits} data points")

    # Panel tags go top-left by default and move only when the top-left is
    # occupied; which corner that is depends on the run, so each candidate is
    # measured against the data exactly as the legends were, and the corner
    # already taken by the legend is excluded before measuring.
    tagbox = dict(boxstyle="round,pad=0.20", fc="white", ec="0.75", lw=0.5,
                  alpha=0.92)

    def tag(ax, text, keys, taken):
        xy = np.vstack([np.column_stack([it[:len(h[k])], h[k]]) for k in keys])
        corners = (("upper left", 0.012, 0.955, "left", "top"),
                   ("upper right", 0.988, 0.955, "right", "top"),
                   ("lower left", 0.012, 0.045, "left", "bottom"),
                   ("lower right", 0.988, 0.045, "right", "bottom"))
        best = None
        for name, x, y, ha, va in corners:
            if name == taken:
                continue
            t = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                        fontsize=10, weight="bold", zorder=6, bbox=tagbox)
            c = covers_data(fig, ax, t, xy)
            if best is None or c < best[0]:
                if best is not None:
                    best[1].remove()
                best = (c, t)
            else:
                t.remove()
            if c == 0:
                break
        print(f"  tag {text[:3]} placed, covers {best[0]} points")
        return best[0]

    lab_b = "(b)" if not segment else f"(b) segment {segment}"
    hits += tag(ax1, "(a)", ("loss_total", "loss_data", "loss_phys", "loss_bc"),
                loc1)
    hits += tag(ax2, lab_b, [k for k in ("te_km", "te_lo", "te_hi") if k in h],
                loc2)

    if synthetic:
        ax1.text(0.5, 0.45, "SYNTHETIC -- LAYOUT CHECK ONLY",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=17, color="0.80", rotation=12, alpha=0.6, zorder=7)

    credit = ("Software used for plotting figure: Python "
              f"{sys.version_info.major}.{sys.version_info.minor}."
              f"{sys.version_info.micro}, "
              f"Matplotlib {matplotlib.__version__}, "
              f"NumPy {np.__version__}. Source: authors.")
    fig.text(0.005, -0.012, credit, fontsize=8, ha="left", va="top")

    print(f"final composite loss {h['loss_total'][-1]:.3e};  "
          f"Te settles at {te_final:.1f} km over {int(it.max())} iterations")
    if "te_lo" in h and "te_hi" in h:
        half = float(np.median((h["te_hi"] - h["te_lo"])[-tail:]) / 2.0)
        print(f"ensemble half-width at convergence {half:.2f} km")
    else:
        print("no ensemble band drawn; panel (b) is one member")
    return fig, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", help="CSV written by the training loop")
    ap.add_argument("--stage-switch", type=float, default=None,
                    help="iteration at which L-BFGS takes over from Adam")
    ap.add_argument("--segment", default="",
                    help="segment identifier annotated on panel (b)")
    ap.add_argument("--components", choices=("weighted", "unweighted"),
                    default="unweighted",
                    help="whether the three loss columns already carry their "
                         "lambda; train_pinn.py writes them unweighted")
    ap.add_argument("--ensemble", default=None,
                    help="glob over per-seed history files, e.g. "
                         "'runs/r*/history.csv'; the band in panel (b) is "
                         "their per-iteration range. Fewer than two members "
                         "matched means no band is drawn.")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic layout check; output must not be published")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--name", default="fig_convergence")
    a = ap.parse_args()

    if a.selftest:
        switch = a.stage_switch if a.stage_switch is not None else 20000
        syn = synthetic_history(switch=int(switch))
        fig, hits = build(syn, synthetic=True, switch=switch,
                          segment=a.segment or "example",
                          convention="unweighted")
        name = a.name + "_SELFTEST"
    else:
        if not a.history:
            ap.error("--history is required (or use --selftest)")
        if a.stage_switch is None:
            print("note: --stage-switch not given, the Adam to L-BFGS "
                  "transition will not be marked")
        hist = read_history(a.history)
        ens = read_ensemble(a.ensemble, hist) if a.ensemble else None
        fig, hits = build(hist, switch=a.stage_switch, segment=a.segment,
                          convention=a.components, ensemble=ens)
        name = a.name

    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = f"{a.outdir.rstrip('/')}/{name}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=600 if ext == "png" else None)
        print("wrote", out)
    if hits:
        print("WARNING: a legend covers data -- fix before publishing")


if __name__ == "__main__":
    main()
