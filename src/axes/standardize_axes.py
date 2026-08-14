#!/usr/bin/env python3
"""standardize_axes.py -- bring trench_axes_full.gmt into line with the study
inventory: add a traced axis, drop a margin, and leave every record in the one
convention the rest of the pipeline assumes.

WHY THIS EXISTS
---------------
trench_axes_full.gmt holds one multisegment record per trench, headed by the
trench name and resampled to a common along-strike step. build_te_segments.py
and build_segment_features.py both walk it, so a margin missing from it is
invisible to every figure however complete its entry in the run tables. Vityaz
was in exactly that state: runs/r02/segments.csv carries vityaz_01, and
runs/r02/deflection.csv carries its 545 stacked samples, so profiles were
extracted from an axis that never reached this file.

WHAT IT DOES
------------
  --add axes/vityaz.txt --name Vityaz     append a traced axis as a new record
  --drop Cascadia                         remove a record by name
  --step-km 5                             resample to the common step

and, on every record it writes:

  * longitudes brought onto the convention already used in the file, checked
    rather than assumed, because the maps span the date line and a record on
    the wrong convention draws as a chord across the Pacific;
  * great-circle resampling to the common step, so along-strike binning in
    build_te_segments.py divides a uniformly sampled node list;
  * the retracing check of the tracing method -- path length against
    end-to-end distance, rejected above 1.6 -- so an axis that doubles back is
    refused here rather than silently corrupting the binning downstream.

The original file is copied to <name>.bak before anything is written.

USAGE
-----
    # look for the axis the profiles were extracted from
    ls -l axes/vityaz.txt

    python3 standardize_axes.py --into trench_axes_full.gmt \\
        --add axes/vityaz.txt --name Vityaz --drop Cascadia

    # then rebuild everything downstream
    python3 build_te_segments.py --axes trench_axes_full.gmt \\
        --values runs/r02/segments.csv --value te_km --out te_segments.gmt

IF axes/vityaz.txt IS GONE
--------------------------
Retrace it with the tools from the trench-classification study. Vityaz is a
relict trench with no active plate boundary, so it has no PB2002 subduction
segment to seed from and the seed is built from the grid instead:

    bash seed_from_grid.sh gebco_pacific.nc vityaz 170.0 175.0 -16.0 -11.5 lon
    # inspect seeds/vityaz.txt before trusting it:
    gmt begin seed_vityaz png
      gmt grdimage gebco_pacific.nc -R169/177/-18/-10 -JM14c -Cgeo -I+d
      gmt plot seeds/vityaz.txt -W1.5p,red -Sc0.1c -Gred
      gmt coast -W0.2p -Ba
    gmt end show

    # then add to trenches.conf:  vityaz  SEED  170.0 175.0 -16.0 -11.5
    bash trace_axes.sh gebco_pacific.nc      # writes axes/vityaz.txt

The trench trends NW-SE, which is the case the banding handles worst: neither
`lon` nor `lat` cuts cleanly across a diagonal feature, so try both and keep
the seed that follows the lineament in the relief. Three stacked profiles and
a 415 m deflection misfit say the depression there is short and poorly
expressed, which is what a relict trench looks like.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def parse_name(header):
    """Trench name out of a GMT segment header.

    A quoted name is taken whole. The first version split on whitespace and
    kept the first word, which turned "Middle America" into "Middle" and
    "San Cristobal" into "San" -- harmless in the file, since headers are
    written back verbatim, but it made --drop unable to match any two-word
    margin and printed a misleading name list.
    """
    if "-L" not in header:
        return None
    tail = header.split("-L", 1)[1].strip()
    if not tail:
        return None
    if tail.startswith('"'):
        end = tail.find('"', 1)
        return tail[1:end] if end > 0 else tail[1:].strip()
    return tail.split()[0]


def read_records(path):
    """[(header, name, [(lon, lat), ...])] in file order."""
    recs, hdr, name, pts = [], None, None, []
    with open(path) as fh:
        for line in fh:
            s = line.rstrip("\n")
            if not s.strip():
                continue
            if s.startswith(">"):
                if hdr is not None:
                    recs.append((hdr, name, pts))
                hdr, pts = s, []
                name = None
                name = parse_name(s)
                continue
            if s.startswith("#"):
                continue
            f = s.split()
            if len(f) >= 2:
                try:
                    pts.append((float(f[0]), float(f[1])))
                except ValueError:
                    pass
    if hdr is not None:
        recs.append((hdr, name, pts))
    return recs


def read_axis(path):
    pts = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith((">", "#")):
                continue
            f = s.split()
            if len(f) >= 2:
                try:
                    pts.append((float(f[0]), float(f[1])))
                except ValueError:
                    pass
    return pts


def resample(pts, step_km):
    """Great-circle resampling to a uniform along-track step."""
    if len(pts) < 2:
        return pts
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + haversine_km(*pts[i - 1], *pts[i]))
    total = cum[-1]
    if total < step_km:
        return pts
    out, j = [], 0
    n = int(total // step_km)
    for k in range(n + 1):
        d = k * step_km
        while j < len(cum) - 2 and cum[j + 1] < d:
            j += 1
        span = cum[j + 1] - cum[j]
        t = 0.0 if span <= 0 else (d - cum[j]) / span
        lon = pts[j][0] + t * (pts[j + 1][0] - pts[j][0])
        lat = pts[j][1] + t * (pts[j + 1][1] - pts[j][1])
        out.append((lon, lat))
    return out


def retrace_ratio(pts):
    """Path length over end-to-end distance. Above 1.6 the axis doubles back."""
    if len(pts) < 2:
        return float("nan")
    path = sum(haversine_km(*pts[i - 1], *pts[i]) for i in range(1, len(pts)))
    chord = haversine_km(*pts[0], *pts[-1])
    return path / chord if chord > 0 else float("inf")


def detect_convention(recs):
    """Is the existing file on 0/360 or -180/180? Decided, not assumed."""
    lons = [p[0] for _, _, pts in recs for p in pts]
    if not lons:
        return "0360"
    return "0360" if min(lons) >= 0.0 and max(lons) > 180.0 else "pm180"


def to_convention(pts, conv):
    out = []
    for lon, lat in pts:
        if conv == "0360":
            while lon < 0.0:
                lon += 360.0
            while lon >= 360.0:
                lon -= 360.0
        else:
            while lon > 180.0:
                lon -= 360.0
            while lon <= -180.0:
                lon += 360.0
        out.append((lon, lat))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", default="trench_axes_full.gmt")
    ap.add_argument("--add", default=None, help="two-column lon lat axis file")
    ap.add_argument("--name", default=None, help="record name for --add")
    ap.add_argument("--drop", action="append", default=[],
                    help="record name to remove; repeatable")
    ap.add_argument("--step-km", type=float, default=5.0)
    ap.add_argument("--max-ratio", type=float, default=1.6,
                    help="reject an added axis whose path/chord exceeds this")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.into):
        sys.exit(f"not found: {a.into}")
    if a.add and not a.name:
        sys.exit("--add requires --name")

    recs = read_records(a.into)
    conv = detect_convention(recs)
    print(f"{a.into}: {len(recs)} records, longitudes on "
          f"{'0/360' if conv == '0360' else '-180/180'}")

    drop_keys = {d.strip().lower().replace(" ", "_").replace("-", "_")
                 for d in a.drop}
    kept, removed = [], []
    for hdr, name, pts in recs:
        key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
        (removed if key in drop_keys else kept).append((hdr, name, pts))
    for _, name, pts in removed:
        print(f"  dropped: {name} ({len(pts)} nodes)")
    for d in a.drop:
        if not any((n or "").lower().replace(" ", "_").replace("-", "_")
                   == d.strip().lower().replace(" ", "_").replace("-", "_")
                   for _, n, _ in removed):
            print(f"  WARNING: --drop {d} matched no record", file=sys.stderr)

    added = None
    if a.add:
        if not os.path.exists(a.add):
            sys.exit(f"axis file not found: {a.add}\n"
                     f"See the header: retrace it with seed_from_grid.sh and "
                     f"trace_axes.sh, or point --add at wherever it lives.")
        raw = read_axis(a.add)
        if len(raw) < 2:
            sys.exit(f"{a.add}: fewer than two nodes, nothing to add")
        pts = to_convention(resample(raw, a.step_km), conv)
        ratio = retrace_ratio(pts)
        length = sum(haversine_km(*pts[i - 1], *pts[i])
                     for i in range(1, len(pts)))
        print(f"  adding {a.name}: {len(raw)} raw nodes -> {len(pts)} at "
              f"{a.step_km:.0f} km, {length:.0f} km long, "
              f"path/chord {ratio:.2f}")
        if ratio > a.max_ratio:
            sys.exit(f"REFUSED: path/chord {ratio:.2f} exceeds {a.max_ratio}. "
                     f"The axis doubles back and would extract profiles twice "
                     f"along the same margin from opposite limbs. Tighten the "
                     f"seed box and retrace.")
        if any((n or "").lower() == a.name.lower() for _, n, _ in kept):
            sys.exit(f"REFUSED: {a.name} is already a record in {a.into}")
        added = (f'> -L"{a.name}"', a.name, pts)
        kept.append(added)

    if a.dry_run:
        print("dry run: nothing written")
        return

    shutil.copy2(a.into, a.into + ".bak")
    with open(a.into, "w") as w:
        for hdr, name, pts in kept:
            w.write(hdr + "\n")
            for lon, lat in pts:
                w.write(f"{lon:.5f} {lat:.5f}\n")
    print(f"wrote {a.into} with {len(kept)} records "
          f"(backup at {a.into}.bak)")
    print("  names: " + ", ".join(sorted((n or "?") for _, n, _ in kept)))


if __name__ == "__main__":
    main()
