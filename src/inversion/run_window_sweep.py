#!/usr/bin/env python3
"""run_window_sweep.py -- tabulate the window dependence behind
Table~\\ref{tab:windowspread} and the \\Delta_w statements of Section 4.2.

This does not reimplement anything. It imports the sweep and the metric from
fig_windowsensitivity.py, so the numbers it writes are by construction the same
quantity the figure draws: every fit is morphometry.flexural_fit through
`surface`, and the spread is `window_metric`, Eq. (15). Keeping one source for
the computation is the whole point; a parallel implementation here could drift
from the figure and no one would notice until a referee added the panels up.

TWO STATISTICS, DELIBERATELY
----------------------------
The figure annotates \\Delta_w on the PER-WINDOW MEDIAN surface: profiles are
medianed first (`trench_surface`), then the spread is taken. Section 4.2 says
in as many words that the TABLE must instead carry the PER-PROFILE population,
which is larger, because medianing across profiles before taking the range
suppresses the very variation the metric is meant to capture. So both are
computed and written:

  dw_median_surface : matches the number printed on each panel of the figure.
  dw_profile_*      : the distribution of \\Delta_w over the individual
                      profiles of a margin -- median, IQR, min, max -- which is
                      the population the manuscript tabulates.

The Te spread columns the table asks for (Min, Max, IQR in km) are taken over
the same per-profile pooled set of admissible windowed Te values, so a reader
can see the kilometres behind the dimensionless metric.

WINDOW-FREE COLUMN
------------------
The final column is the physics-informed estimate per trench, read from
runs/r02/segments.csv (median te_km over the segments of the margin). It is not
recomputed here; it is the output of the training run.

USAGE
-----
    python3 run_window_sweep.py --profiles profiles_csv \\
        --segments runs/r02/segments.csv --nprofiles 24 \\
        --out sweep_grid.csv --tex windowspread_rows.tex

--nprofiles defaults to 24, not the figure's survey default of 12, because the
file's own %NOTE records that 8 per margin does not give a stable per-trench
\\Delta_w. Raise it further and confirm the last two significant figures of
each row stop moving before you submit.

OUTPUT
------
  sweep_grid.csv        one row per (trench, profile, xl, xs, te_km): the raw
                        pooled surface, so the table can be recomputed or
                        audited without re-running the sweep.
  <out>_by_trench.csv   the per-trench summary that the table rows come from.
  windowspread_rows.tex the LaTeX body rows, ready to \\input or paste.

Run in `flexure` (Python 3.12.13, SciPy 1.18.0), the same environment as the
figure, so flexural_fit behaves identically.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Reuse the figure's own machinery. Importing the module runs only its
# top-level definitions (constants and functions); its main() is guarded by
# __name__ == "__main__", so nothing is drawn.
try:
    from fig_windowsensitivity import (
        TRENCH_ORDER, MIN_SPAN_KM, HALFLEN_KM,
        profile_files, read_profile_file, surface, window_metric,
    )
except ImportError as e:
    sys.exit(
        "cannot import from fig_windowsensitivity.py -- run this script from "
        "the same directory (essential_data_scripts), in the `flexure` "
        "environment. Original error: %s" % e)


def pooled_profile_surfaces(profdir, trench, near, far, nprofiles, sediment):
    """Every admissible windowed Te for an evenly spaced sample of one trench.

    Returns two things: a list of per-profile Te surfaces (each a near x far
    array, NaN where the window is inadmissible or the fit was rejected), and
    the count of profiles that yielded at least one finite Te. Sampling matches
    trench_surface: even spacing along the arc, not random, not the densest
    numbering.
    """
    files = profile_files(profdir, trench)
    if not files:
        return [], 0, 0
    take = files if len(files) <= nprofiles else [
        files[i] for i in
        np.linspace(0, len(files) - 1, nprofiles).astype(int)]
    surfaces, used = [], 0
    for path in take:
        x, d = read_profile_file(path, sediment)
        te, _, _ = surface(x, d, near, far)
        if np.isfinite(te).any():
            surfaces.append(te)
            used += 1
    return surfaces, used, len(files)


def iqr(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def summarise(trench, surfaces, near, far, windowfree):
    """One table row from a margin's per-profile Te surfaces.

    dw_median_surface reproduces the figure: median across profiles first, then
    window_metric. The dw_profile_* fields are the per-profile population the
    manuscript tabulates: window_metric applied to each profile's own surface,
    then summarised across profiles. The Te Min/Max/IQR are over the pooled
    admissible Te of all sampled profiles, in kilometres.
    """
    stack = np.dstack(surfaces)                       # near x far x nprofile
    import warnings
    with np.errstate(invalid="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
        med_surface = np.nanmedian(stack, axis=2)
    dw_med = window_metric(med_surface)

    per_profile_dw = np.array([window_metric(s) for s in surfaces], float)
    per_profile_dw = per_profile_dw[np.isfinite(per_profile_dw)]

    pooled = stack[np.isfinite(stack)]
    n_windows = int(np.isfinite(stack).sum())

    return {
        "trench": trench,
        "n_profiles": len(surfaces),
        "n_windows": n_windows,
        "te_min_km": float(pooled.min()) if pooled.size else np.nan,
        "te_max_km": float(pooled.max()) if pooled.size else np.nan,
        "te_iqr_km": iqr(pooled),
        "dw_median_surface": dw_med,
        "dw_profile_median": float(np.median(per_profile_dw))
        if per_profile_dw.size else np.nan,
        "dw_profile_iqr": iqr(per_profile_dw),
        "dw_profile_min": float(per_profile_dw.min())
        if per_profile_dw.size else np.nan,
        "dw_profile_max": float(per_profile_dw.max())
        if per_profile_dw.size else np.nan,
        "windowfree_km": windowfree,
    }


def raw_rows(trench, surfaces, take_ids, near, far):
    """Long-format rows (trench, profile, xl, xs, te_km) for the audit CSV."""
    out = []
    for pid, s in zip(take_ids, surfaces):
        fi = np.isfinite(s)
        for i, j in zip(*np.where(fi)):
            out.append((trench, pid, float(near[i]), float(far[j]),
                        float(s[i, j])))
    return out


DISPLAY = {
    "aleutian": "Aleutian", "kuril-kamchatka": "Kuril--Kamchatka",
    "japan": "Japan", "izu-bonin": "Izu--Bonin", "mariana": "Mariana",
    "yap": "Yap", "palau": "Palau", "ryukyu": "Ryukyu", "manila": "Manila",
    "philippine": "Philippine", "new_britain": "New Britain",
    "san_cristobal": "San Cristobal", "vityaz": "Vityaz",
    "new_hebrides": "Vanuatu", "tonga": "Tonga", "kermadec": "Kermadec",
    "hikurangi": "Hikurangi", "puysegur": "Puysegur",
    "middle_america": "Middle America", "peru-chile": "Peru--Chile",
}


def windowfree_by_trench(segments_csv):
    """Median physics-informed Te per trench from the training output."""
    if not segments_csv or not os.path.exists(segments_csv):
        print("note: --segments not given or not found; window-free column "
              "left blank", file=sys.stderr)
        return {}
    s = pd.read_csv(segments_csv)
    if "trench" not in s or "te_km" not in s:
        print("note: segments.csv lacks trench/te_km; window-free blank",
              file=sys.stderr)
        return {}
    return s.groupby("trench").te_km.median().to_dict()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True,
                    help="directory of <trench>_<pid>.csv, as the figure reads")
    ap.add_argument("--segments", default=None,
                    help="runs/r02/segments.csv for the window-free column")
    ap.add_argument("--trenches", nargs="*", default=None,
                    help="subset, in order; default all twenty of TRENCH_ORDER")
    ap.add_argument("--nprofiles", type=int, default=24,
                    help="profiles sampled per trench (default 24; 8 is too "
                         "few for a stable Delta_w, per the figure's own note)")
    ap.add_argument("--no-sediment", action="store_true",
                    help="skip Airy sediment unloading, to match --no-sediment "
                         "on the figure")
    ap.add_argument("--near", type=float, nargs=3, default=(0.0, 50.0, 5.0),
                    metavar=("MIN", "MAX", "STEP"),
                    help="landward-limit grid; default matches the survey")
    ap.add_argument("--far", type=float, nargs=3,
                    default=(MIN_SPAN_KM, 550.0, 20.0),
                    metavar=("MIN", "MAX", "STEP"),
                    help="seaward-limit grid; default matches the survey")
    ap.add_argument("--out", default="sweep_grid.csv",
                    help="long-format raw surface CSV")
    ap.add_argument("--tex", default="windowspread_rows.tex",
                    help="LaTeX table body rows")
    a = ap.parse_args()

    near = np.arange(a.near[0], a.near[1] + 1e-9, a.near[2])
    far = np.arange(a.far[0], a.far[1] + 1e-9, a.far[2])
    far = far[far <= HALFLEN_KM + 1e-9]
    if near.size == 0 or far.size == 0 or far.max() - near.min() < MIN_SPAN_KM:
        sys.exit("empty or too-short window grid; check --near/--far")

    trenches = a.trenches or TRENCH_ORDER
    wf = windowfree_by_trench(a.segments)
    sediment = not a.no_sediment

    print(f"sweeping {len(trenches)} trenches, {a.nprofiles} profiles each, "
          f"grid {near.size}x{far.size} = up to {near.size*far.size} windows "
          f"per profile", file=sys.stderr)

    summary, raw = [], []
    for t in trenches:
        take = profile_files(a.profiles, t)
        if not take:
            print(f"note: no profiles for {t}; row omitted", file=sys.stderr)
            continue
        ids = take if len(take) <= a.nprofiles else [
            take[i] for i in
            np.linspace(0, len(take) - 1, a.nprofiles).astype(int)]
        pid_labels = [os.path.splitext(os.path.basename(p))[0] for p in ids]
        surfaces, used, avail = pooled_profile_surfaces(
            a.profiles, t, near, far, a.nprofiles, sediment)
        if not surfaces:
            print(f"note: {t} yielded no admissible fit on the grid; row "
                  f"omitted", file=sys.stderr)
            continue
        row = summarise(t, surfaces, near, far, wf.get(t, np.nan))
        row["n_available"] = avail
        summary.append(row)
        raw.extend(raw_rows(t, surfaces, pid_labels[:len(surfaces)], near, far))
        print(f"  {DISPLAY.get(t, t):16} n={used}/{avail:<4} "
              f"dw_surface={row['dw_median_surface']:.2f} "
              f"dw_profile_med={row['dw_profile_median']:.2f}",
              file=sys.stderr)

    if not summary:
        sys.exit("no trench produced a usable row; nothing written")

    dfs = pd.DataFrame(summary)
    order = {t: i for i, t in enumerate(TRENCH_ORDER)}
    dfs = dfs.sort_values("trench", key=lambda s: s.map(order))

    by_trench = a.out.replace(".csv", "_by_trench.csv")
    dfs.to_csv(by_trench, index=False)
    pd.DataFrame(raw, columns=["trench", "profile", "xl_km", "xs_km",
                               "te_km"]).to_csv(a.out, index=False)

    # LaTeX rows: Trench & Te-min & Te-max & Te-IQR & Delta_w & window-free.
    # Delta_w here is the PER-PROFILE MEDIAN, the population the manuscript
    # tabulates; the median-surface value (matching the figure panels) stays in
    # the CSV for cross-checking but is not the tabulated one.
    with open(a.tex, "w") as fh:
        for _, r in dfs.iterrows():
            wfree = ("%.0f" % r.windowfree_km
                     if np.isfinite(r.windowfree_km) else "---")
            fh.write(
                "    %-16s & %4.0f & %4.0f & %4.0f & %5.2f & %s \\\\\n" % (
                    DISPLAY.get(r.trench, r.trench),
                    r.te_min_km, r.te_max_km, r.te_iqr_km,
                    r.dw_profile_median, wfree))

    # Aggregate line for the Section 4.2 sentence.
    dwp = dfs.dw_profile_median.to_numpy()
    lo_t = dfs.iloc[int(np.nanargmin(dwp))]
    hi_t = dfs.iloc[int(np.nanargmax(dwp))]
    print("\nFor the Section 4.2 aggregate sentence (per-profile median "
          "Delta_w):", file=sys.stderr)
    print("  Delta_w ranges from %.2f (%s) to %.2f (%s), median %.2f across "
          "the %d margins." % (
              np.nanmin(dwp), DISPLAY.get(lo_t.trench, lo_t.trench),
              np.nanmax(dwp), DISPLAY.get(hi_t.trench, hi_t.trench),
              np.nanmedian(dwp), len(dfs)), file=sys.stderr)
    print("  (Figure panels use the median-surface Delta_w, %.2f to %.2f, "
          "median %.2f -- systematically smaller, as the text states.)" % (
              np.nanmin(dfs.dw_median_surface), np.nanmax(dfs.dw_median_surface),
              np.nanmedian(dfs.dw_median_surface)), file=sys.stderr)

    print("\nwrote %s  (%d raw rows)" % (a.out, len(raw)))
    print("wrote %s  (%d per-trench rows)" % (by_trench, len(dfs)))
    print("wrote %s  (table body, per-profile Delta_w)" % a.tex)


if __name__ == "__main__":
    main()
