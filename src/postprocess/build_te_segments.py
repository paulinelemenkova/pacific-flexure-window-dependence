#!/usr/bin/env python3
"""build_te_segments.py -- join the traced trench axes with the per-segment
elastic thickness into the GMT -Z multisegment file that fig_temap.sh draws.

THE PROBLEM THIS SOLVES
-----------------------
fig_temap.sh (and fig_parametermaps.sh) need a multisegment file in which each
along-strike band carries its recovered value in the header:

    > -Z34.2
    182.417  -22.905
    ...

but the geometry on disk (trench_axes_full.gmt) has ONE record per trench --
the whole axis, headed `> -L"Aleutian"` -- while the values (segment_flexure.csv
/ segment_zones.csv / segments.csv) are per SEGMENT, several per trench. This
script performs the join: it splits each trench's axis into that trench's
segments and writes one -Z band per segment.

THE PARTITION
-------------
segment_zones.csv gives, per segment, an integer `zone` that indexes the
along-strike bin. Segments of a trench are numbered in axis order
(aleutian_01, _02, ...), so the axis node list is divided into N equal runs
where N is the number of segments on that trench, and run k is coloured by the
Te of segment k. This reproduces the equal-bin construction the segments were
built with. If a future segment_zones.csv carries explicit node or latitude
bounds per zone, prefer those; the equal split is the documented fallback and
is exact to the 5 km axis resampling.

VALUE COLUMN
------------
--value selects which recovered quantity becomes the -Z header:
  te   (default)  the window-free Te, column `Te` in segment_flexure.csv
                  or `te_km` in segments.csv
  any other column present in the chosen --values file (moment, shear, ...),
  so the same builder feeds fig_parametermaps by naming its columns.

USAGE
-----
    python3 build_te_segments.py \
        --axes trench_axes_full.gmt \
        --values segment_flexure.csv --value Te \
        --out te_segments.gmt

Then draw with the existing map, pointing RELIEF at the local GEBCO grid:

    RELIEF="/Volumes/TOSHIBA EXT/DATA/GEBCO_2023.nc" \
    SEGMENTS=te_segments.gmt AXES=axes_full bash fig_temap.sh
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict, OrderedDict


# Map the axis-header trench name (as in > -L"Aleutian") to the lowercase key
# used in the segment tables (aleutian). Hyphens and spaces both go to the
# table's underscore/hyphen convention; both are tried on lookup.
def canon(name):
    return name.strip().strip('"').lower().replace(" ", "_").replace("-", "_")


# The axis file and the value tables spell a few trenches differently. The axis
# header uses the geographic name; the value tables use the plate-pair or the
# historical name under which the segments were computed. Map axis key ->
# value-table key so no trench is dropped over a spelling. Confirmed from
# `grep '^>' trench_axes_full.gmt` against `cut -d, -f2 segment_flexure.csv`.
ALIASES = {
    "vanuatu": "new_hebrides",   # axis "Vanuatu"  <-> table new_hebrides
    # NOTE: add the Hikurangi alias here once its value-table key is confirmed
    # (see the check printed by --report-unmatched); do NOT guess it, mapping
    # to the wrong trench would colour the axis with another margin's values.
}


def read_axes(path):
    """Return OrderedDict trench_key -> list of (lon, lat), in file order."""
    axes = OrderedDict()
    key = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith(">"):
                m = re.search(r'-L\s*"?([^"#]+)"?', s)
                name = m.group(1) if m else s.lstrip(">").strip()
                key = canon(name)
                axes[key] = []
            elif key is not None:
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        axes[key].append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    return axes


def read_values(path, value_col):
    """trench_key -> list of (segment_index, value), sorted by segment index.

    Accepts either segment_flexure.csv (value column 'Te') or segments.csv
    (value column 'te_km'); the segment id like 'aleutian_03' supplies both the
    trench and the order. A segment whose value cell is blank is kept with a
    NaN so its position on the axis is preserved; gap_fill() then interpolates
    it from its neighbours, so the band is coloured rather than left white.
    """
    import math
    per_trench = defaultdict(list)
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        if value_col not in rdr.fieldnames:
            sys.exit(f"column '{value_col}' not in {path}; columns are: "
                     + ", ".join(rdr.fieldnames))
        for row in rdr:
            seg = (row.get("segment") or "").strip()
            m = re.match(r"(.+)_(\d+)$", seg)
            if not m:
                continue
            tkey = canon(m.group(1))
            idx = int(m.group(2))
            raw = (row.get(value_col) or "").strip()
            try:
                per_trench[tkey].append((idx, float(raw) if raw else math.nan))
            except ValueError:
                per_trench[tkey].append((idx, math.nan))
    for k in per_trench:
        per_trench[k].sort(key=lambda t: t[0])
    return per_trench


def gap_fill(segvals):
    """Replace NaN values by interpolation from finite neighbours, in segment
    order, so no band is left uncoloured. Interior gaps are linearly
    interpolated; a gap at either end takes the nearest finite value (constant
    extrapolation). If a trench has no finite value at all, it is returned
    unchanged and will be reported. Returns (filled_list, n_filled)."""
    import math
    idxs = [i for i, _ in segvals]
    vals = [v for _, v in segvals]
    finite = [k for k, v in enumerate(vals) if not math.isnan(v)]
    if not finite:
        return segvals, 0
    filled = list(vals)
    n = 0
    for k, v in enumerate(vals):
        if not math.isnan(v):
            continue
        # nearest finite neighbours on each side
        left = max((j for j in finite if j < k), default=None)
        right = min((j for j in finite if j > k), default=None)
        if left is not None and right is not None:
            t = (k - left) / (right - left)
            filled[k] = vals[left] + t * (vals[right] - vals[left])
        elif left is not None:
            filled[k] = vals[left]
        else:
            filled[k] = vals[right]
        n += 1
    return list(zip(idxs, filled)), n


def split_runs(n_nodes, n_segments):
    """Return n_segments (start, end) index pairs partitioning range(n_nodes)
    into as-equal-as-possible contiguous runs, in axis order."""
    if n_segments <= 0 or n_nodes == 0:
        return []
    base, extra = divmod(n_nodes, n_segments)
    bounds, start = [], 0
    for k in range(n_segments):
        length = base + (1 if k < extra else 0)
        bounds.append((start, start + length))
        start += length
    return bounds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--axes", default="trench_axes_full.gmt",
                    help="multisegment trench axes, one record per trench")
    ap.add_argument("--values", default="segment_flexure.csv",
                    help="per-segment table (segment_flexure.csv or "
                         "segments.csv)")
    ap.add_argument("--value", default="Te",
                    help="column to write as the -Z value (Te, te_km, "
                         "mdot, ...)")
    ap.add_argument("--out", default="te_segments.gmt")
    a = ap.parse_args()

    axes = read_axes(a.axes)
    vals = read_values(a.values, a.value)
    if not axes:
        sys.exit(f"no axis records read from {a.axes}")
    if not vals:
        sys.exit(f"no segment values read from {a.values}")

    written = skipped_axis = skipped_val = filled_total = 0
    lines_out = 0
    unmatched = []
    with open(a.out, "w") as w:
        w.write(f"# te_segments built from {a.axes} + {a.values}:{a.value}\n")
        for tkey, pts in axes.items():
            segvals = vals.get(tkey)
            if segvals is None:
                # try the explicit alias table, then the hyphen spelling
                if tkey in ALIASES:
                    segvals = vals.get(ALIASES[tkey])
                if segvals is None:
                    segvals = vals.get(tkey.replace("_", "-"))
            if not segvals or not pts:
                skipped_axis += 1
                unmatched.append(tkey)
                print(f"note: no values for '{tkey}' -- axis skipped "
                      f"(value-table keys available: add an alias if this "
                      f"trench does have data under another name)",
                      file=sys.stderr)
                continue
            # fill any blank-valued segments so every band is coloured
            segvals, nf = gap_fill(segvals)
            filled_total += nf
            runs = split_runs(len(pts), len(segvals))
            for (idx, value), (lo, hi) in zip(segvals, runs):
                if hi <= lo:
                    skipped_val += 1
                    continue
                w.write(f"> -Z{value:.6g}\n")
                for lon, lat in pts[lo:hi]:
                    lon360 = lon + 360.0 if lon < 0 else lon
                    w.write(f"{lon360:.5f} {lat:.5f}\n")
                    lines_out += 1
                written += 1

    print(f"wrote {a.out}: {written} coloured segments across "
          f"{len(axes) - skipped_axis} trenches, {lines_out} points",
          file=sys.stderr)
    if filled_total:
        print(f"  {filled_total} blank-valued segments filled by "
              f"interpolation from neighbours", file=sys.stderr)
    if skipped_axis:
        print(f"  {skipped_axis} trench axes had no matching values: "
              f"{', '.join(unmatched)}", file=sys.stderr)
        print(f"  -> to colour these, find their key with "
              f"`cut -d, -f2 {a.values} | sort -u` and add "
              f"axis_name:table_key to ALIASES", file=sys.stderr)
    if skipped_val:
        print(f"  {skipped_val} segments had an empty axis run (more segments "
              f"than axis nodes)", file=sys.stderr)


if __name__ == "__main__":
    main()
