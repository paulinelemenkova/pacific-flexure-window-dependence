#!/usr/bin/env python3
"""build_segment_features.py -- assemble the table fig_teage.py needs.

fig_teage.py wants, one row per segment:

    trench,age_ma,te_km[,te_err]

Nothing on disk has that shape. runs/r02/segments.csv carries te_km but no
age; segment_flexure.csv carries Te but neither an age nor a trench column.
The missing quantity is SEAFLOOR AGE, which is not a product of the inversion
and has to be sampled from a published age grid at each segment's position.

WHAT THIS DOES
--------------
1. Reuses build_te_segments.py's own join, so a segment's geometry is
   identified exactly as it is for the maps -- the trench axis split into as
   many contiguous runs as that trench has segments, run k belonging to
   segment k. Re-deriving that here by hand would risk a different mapping
   between the table and the figures.
2. Takes each segment's mid-point along its run as the sampling position.
3. Samples the age grid there with `gmt grdtrack`.
4. Writes trench, age_ma, te_km (and te_err if a column is named).

THE AGE GRID IS YOURS TO SUPPLY
-------------------------------
No age grid ships with this repository and none is invented here: a fabricated
age axis would make the fitted isotherm meaningless. Use a published grid, and
cite it in the manuscript beside the figure. The usual choice for this kind of
work is the EarthByte present-day age grid of Seton et al. (2020),
age.2020.1.GTS2012.2m.nc, age of oceanic crust in Myr on a -180/180 mesh:

    https://www.earthbyte.org/webdav/ftp/earthbyte/agegrid/2020/Grids/

GMT also serves the same dataset remotely as @earth_age_02m, which needs
no download. Cite: Seton, M., Muller, R. D., Zahirovic, S., Williams, S.,
Wright, N., Cannon, J., Whittaker, J., Matthews, K., McGirr, R. (2020),
A global dataset of present-day oceanic crustal age and seafloor spreading
parameters, G-cubed, doi:10.1029/2020GC009214.

Any grid readable by GMT works, in Ma, on either -180/180 or 0/360; grdtrack
handles the wrap. Verify the units before trusting the output:

    gmt grdinfo <grid>          # z range should look like 0 to ~280 Ma

USAGE
-----
    python3 build_segment_features.py \\
        --axes trench_axes_full.gmt \\
        --values runs/r02/segments.csv --value te_km \\
        --ages age.2020.1.GTS2012.2m.nc \\
        --out segment_features.csv

    # or, with no download at all -- GMT fetches and caches the same
    # EarthByte grid itself:
    python3 build_segment_features.py \\
        --axes trench_axes_full.gmt \\
        --values runs/r02/segments.csv --value te_km \\
        --ages @earth_age_02m \\
        --out segment_features.csv

    # then
    python3 fig_teage.py --table segment_features.csv --outdir figures

NOTE ON te_err
--------------
--err-col is accepted but there is nothing to point it at yet: the ensemble
standard deviation of Section 4 requires more than one member and run.json
records a single seed-0 run. Left unset, no te_err column is written and
fig_teage.py simply draws no error bars.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys

try:
    from build_te_segments import read_axes, read_values, split_runs, ALIASES
except ImportError:
    sys.exit("build_te_segments.py must sit beside this script (its join is "
             "reused so the table and the maps agree segment for segment)")


def _offset(lon, lat, tlon, tlat, d_km, sign):
    """Point d_km to one side of the local tangent (tlon, tlat) at (lon, lat)."""
    kx = 111.320 * math.cos(math.radians(lat))
    ky = 110.574
    tx, ty = tlon * kx, tlat * ky
    n = math.hypot(tx, ty)
    if n == 0.0:
        return None
    tx, ty = tx / n, ty / n
    px, py = -ty * sign, tx * sign
    return lon + px * d_km / kx, lat + py * d_km / ky


def segment_positions(axes, vals, n_nodes=5, offset_km=100.0):
    """Sampling geometry for each segment.

    Returns [(axis_key, values_key, segment_index, value, [(lon, lat), ...])]
    where the
    coordinate list holds 2*n_nodes candidate points: n_nodes stations spread
    along the segment's own run of axis nodes, each offset offset_km to BOTH
    sides of the local strike.

    WHY NOT THE AXIS ITSELF. The first version of this script sampled the age
    grid at the segment mid-point, on the trench axis, and lost roughly two
    thirds of the segments to undefined nodes -- the age model is blank in the
    trench and across the accretionary wedge. Sampling on the axis is also the
    wrong quantity: the elastic thickness describes the INCOMING plate, so the
    age wanted is that of the subducting plate outboard of the trench, not the
    age at the hinge.

    WHICH SIDE IS SEAWARD. The two sides are sampled and the older defined one
    is taken. Where the overriding side is continental the age model is blank
    there and the choice is forced; where both sides are oceanic, as at the
    Izu-Bonin and Mariana margins, the incoming Pacific lithosphere is the
    older of the two and the back-arc basin the younger, so the same rule
    selects it. The rule is stated rather than hidden because it is an
    assumption: --offset-km 0 restores on-axis sampling if you want to check
    what it costs.
    """
    out, skipped = [], []
    for tkey, pts in axes.items():
        # The key under which the VALUES are filed is not always the axis key:
        # the axis says "vanuatu", the tables say "new_hebrides", and ALIASES
        # bridges them. That resolved key is returned alongside the axis key,
        # because any later lookup into the same tables -- the uncertainty
        # column, the extra columns carried through for screening -- must use
        # it too. Using the axis key there silently returned nothing, wrote a
        # blank residual for every Vanuatu segment, and let four of them
        # through a screen that should have removed them.
        vkey = None
        for cand in (tkey, ALIASES.get(tkey), tkey.replace("_", "-")):
            if cand is not None and cand in vals:
                vkey = cand
                break
        segvals = vals.get(vkey) if vkey else None
        if not segvals or not pts:
            skipped.append(tkey)
            continue
        runs = split_runs(len(pts), len(segvals))
        for (idx, value), (lo, hi) in zip(segvals, runs):
            if hi <= lo:
                continue
            span = hi - lo
            stations = sorted({lo + int((k + 0.5) * span / n_nodes)
                               for k in range(n_nodes)})
            coords = []
            for j in stations:
                j = min(max(j, lo), hi - 1)
                a = pts[max(j - 1, lo)]
                b = pts[min(j + 1, hi - 1)]
                tlon, tlat = b[0] - a[0], b[1] - a[1]
                if tlon == 0.0 and tlat == 0.0:
                    continue
                for sign in (+1, -1):
                    p = _offset(pts[j][0], pts[j][1], tlon, tlat,
                                offset_km, sign)
                    if p is not None:
                        coords.append(p)
            if not coords:
                coords = [pts[(lo + hi) // 2]]
            out.append((tkey, vkey, idx, value, coords))
    return out, skipped


def sample_ages(rows, grid):
    """One age per segment, reduced from its candidate points.

    All candidates for all segments go to grdtrack in a single call, then each
    segment takes the OLDER of its two sides (see segment_positions), using the
    median over the along-strike stations on that side so that one blank node
    cannot decide the value.

    Three flags are deliberate, and each cost a debugging round:

    --GMT_HISTORY=false   a gmt.history left by an earlier mapping command
                          silently reapplies its region; grdtrack then drops
                          every point outside it and exits 0 with no output,
                          which looks exactly like a parsing failure.
    longitudes to -180/180  build_te_segments writes 0/360, the EarthByte grid
                          is stored on -180/180.
    -fg                   declare the input geographic rather than relying on
                          inference from the grid.
    """
    flat, index = [], []
    for i, (_, _, _, _, coords) in enumerate(rows):
        for lon, lat in coords:
            x = lon - 360.0 if lon > 180.0 else lon
            flat.append(f"{x:.6f}\t{lat:.6f}")
            index.append(i)

    cmd = ["gmt", "grdtrack", "-G" + grid, "-fg", "--GMT_HISTORY=false"]
    try:
        p = subprocess.run(cmd, input="\n".join(flat) + "\n",
                           capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("gmt not found on PATH -- grdtrack is needed to sample the grid")
    if p.returncode != 0:
        sys.exit(f"grdtrack failed:\n{p.stderr.strip()}")

    got = []
    for line in p.stdout.splitlines():
        if not line.strip() or line.startswith(("#", ">")):
            continue
        f = line.split()
        try:
            got.append(float(f[2]) if len(f) >= 3 else math.nan)
        except ValueError:
            got.append(math.nan)
    if len(got) != len(flat):
        sys.exit(f"grdtrack returned {len(got)} rows for {len(flat)} points.\n"
                 f"command: {' '.join(cmd)}\n"
                 f"first input row: {flat[0]}\n"
                 f"grdtrack stderr: {p.stderr.strip() or '(silent)'}")

    # regroup: candidates alternate side A, side B, side A, ... per station
    per_seg = [[] for _ in rows]
    for k, v in zip(index, got):
        per_seg[k].append(v)

    ages, n_both, n_one = [], 0, 0
    for vals in per_seg:
        a = [v for v in vals[0::2] if math.isfinite(v) and v > 0]
        b = [v for v in vals[1::2] if math.isfinite(v) and v > 0]
        ma = median(a) if a else math.nan
        mb = median(b) if b else math.nan
        if a and b:
            n_both += 1
            ages.append(max(ma, mb))          # the incoming plate is the older
        elif a or b:
            n_one += 1
            ages.append(ma if a else mb)
        else:
            ages.append(math.nan)
    print(f"  age defined on both sides for {n_both} segments (older taken), "
          f"one side only for {n_one}", file=sys.stderr)
    return ages


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="trench_axes_full.gmt")
    ap.add_argument("--values", default="runs/r02/segments.csv",
                    help="per-segment table carrying the thickness")
    ap.add_argument("--value", default="te_km",
                    help="thickness column: te_km in segments.csv, "
                         "Te in segment_flexure.csv")
    ap.add_argument("--err-col", default=None,
                    help="optional uncertainty column -> te_err")
    ap.add_argument("--extra-cols", default="rms_fit_m",
                    help="comma-separated extra columns copied through to the "
                         "output, so a figure can screen on them without "
                         "reopening the run table; default carries the "
                         "deflection misfit for the residual screen")
    ap.add_argument("--ages", required=True,
                    help="seafloor-age grid in Ma, readable by GMT")
    ap.add_argument("--offset-km", type=float, default=100.0,
                    help="perpendicular distance seaward of the axis at which "
                         "the age is sampled; 0 restores on-axis sampling, "
                         "which loses most segments to undefined nodes")
    ap.add_argument("--nodes", type=int, default=5,
                    help="stations along each segment, each sampled on both "
                         "sides; the median over a side is used")
    ap.add_argument("--out", default="segment_features.csv")
    a = ap.parse_args()

    # GMT remote grids are names, not paths: @earth_age_02m and friends are
    # resolved and cached by GMT itself under ~/.gmt/server. Only a real path
    # is checked here, or the remote form would be rejected before grdtrack
    # ever sees it.
    if not a.ages.startswith("@") and not os.path.exists(a.ages):
        sys.exit(f"age grid not found: {a.ages}\n"
                 f"Give a real path, or a GMT remote name such as "
                 f"@earth_age_02m.\n"
                 f"See the header: the grid must be supplied, not invented.")

    axes = read_axes(a.axes)
    vals = read_values(a.values, a.value)
    errs = read_values(a.values, a.err_col) if a.err_col else {}
    extra_names = [c.strip() for c in (a.extra_cols or "").split(",") if c.strip()]
    extras = {c: read_values(a.values, c) for c in extra_names}

    rows, skipped = segment_positions(axes, vals, n_nodes=a.nodes,
                                      offset_km=a.offset_km)

    # The reverse check. segment_positions walks the AXES, so a trench that has
    # recovered values but no axis record is never visited and disappears with
    # no message at all -- which is how Vityaz went missing from the age figure
    # while still being counted in the run. Report it explicitly.
    axis_keys = set(axes) | {ALIASES[k] for k in ALIASES if k in axes}
    orphan = sorted(k for k in vals
                    if k not in axis_keys
                    and k.replace("_", "-") not in axis_keys
                    and k not in set(ALIASES.values()))
    if orphan:
        n_lost = sum(len(vals[k]) for k in orphan)
        print(f"  WARNING: {len(orphan)} trench(es) carry values but have NO "
              f"axis record, so {n_lost} segment(s) cannot be positioned and "
              f"are absent from the output: {', '.join(orphan)}",
              file=sys.stderr)
        print(f"  -> trace an axis for them into {a.axes}, or drop them from "
              f"{a.values} so the inventory and the figures agree",
              file=sys.stderr)
    if not rows:
        sys.exit("no segments matched between the axes and the value table")
    print(f"{len(rows)} segments positioned across "
          f"{len(axes) - len(skipped)} trenches, "
          f"{a.nodes} stations each at {a.offset_km:.0f} km either side",
          file=sys.stderr)
    if skipped:
        print(f"  {len(skipped)} trench axes had no matching values: "
              f"{', '.join(skipped)}", file=sys.stderr)
        print(f"  -> add an ALIASES entry in build_te_segments.py to include them",
              file=sys.stderr)

    ages = sample_ages(rows, a.ages)

    n_written = n_nan = 0
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        header = (["trench", "age_ma", "te_km"]
                  + (["te_err"] if a.err_col else []) + extra_names)
        w.writerow(header)
        for (tkey, vkey, idx, value, _coords), age in zip(rows, ages):
            # fig_teage.py drops rows with a missing age or thickness and counts
            # them. They are written through rather than silently removed here,
            # so the two counts can be compared and a systematic gap noticed --
            # a whole trench on continental crust, say, where the age grid is
            # NaN.
            if not (math.isfinite(age) and age > 0) or not math.isfinite(value):
                n_nan += 1
            row = [tkey, f"{age:.3f}" if math.isfinite(age) else "",
                   f"{value:.6g}" if math.isfinite(value) else ""]
            if a.err_col:
                e = dict(errs.get(vkey, [])).get(idx, math.nan)
                row.append(f"{e:.6g}" if math.isfinite(e) else "")
            for c in extra_names:
                v = dict(extras[c].get(vkey, [])).get(idx, math.nan)
                row.append(f"{v:.6g}" if math.isfinite(v) else "")
            w.writerow(row)
            n_written += 1

    print(f"wrote {a.out}: {n_written} rows, {n_nan} with no usable age or "
          f"thickness", file=sys.stderr)
    if n_nan:
        print("  check whether these cluster on one margin before proceeding",
              file=sys.stderr)


if __name__ == "__main__":
    main()
