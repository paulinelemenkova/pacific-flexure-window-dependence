#!/usr/bin/env python3
"""collect_ablation.py -- assemble ablation.csv from a set of training runs.

fig_ablation.py reads one row per swept point:

    parameter,value,te_km,residual,te_lo,te_hi,retained

Each row here is one completed run of train_pinn.py. The run directories hold
per-segment results, so the panel value is a reduction over the 71 segments:

    te_km      median recovered thickness over the segments of that run
    te_lo/hi   first and third quartiles, drawn by the figure as a band
    residual   median loss_phys over the segments

The median rather than the mean, and quartiles rather than a standard
deviation, because the recovered thickness is strongly right-skewed -- a
single Hikurangi-like segment at 67 km would otherwise move the curve and make
a flat sweep look sloped.

WHICH PARAMETER A RUN BELONGS TO is read from run.json, not from the directory
name, so a mislabelled folder cannot silently enter the wrong panel:

    physics_weight  args.lambda_ratio       (requires the train_pinn patch)
    hidden_units    args.width
    collocation     args.collocation
    spacing         SEGMENT_KM / args.max_profiles_per_segment, in km

The last is the effective along-strike spacing: profiles are cast every 5 km
and a segment spans SEGMENT_KM, so capping the stack at n profiles samples the
segment every SEGMENT_KM/n kilometres. That is what the panel means by profile
spacing, and it is computed here rather than assumed.

USAGE
-----
    python3 collect_ablation.py --runs runs/abl_* runs/r02 \\
        --retained runs/r02 --out ablation.csv

    python3 fig_ablation.py --table ablation.csv --outdir figures

A run appears in every panel whose parameter it varies. The retained run sits
on all four sweeps at its own value, which is why --retained names it: it is
the single point the vertical line marks in each panel.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

SEGMENT_KM = 148.0          # train_pinn.SEGMENT_KM

# panel key -> (json args field, how the swept value is formed)
AXES = {
    "physics_weight": ("lambda_ratio", lambda v: float(v)),
    "hidden_units":   ("width",        lambda v: float(v)),
    "collocation":    ("collocation",  lambda v: float(v)),
    "spacing":        ("max_profiles_per_segment",
                       lambda v: SEGMENT_KM / float(v) if float(v) > 0 else np.nan),
}


def summarise(run_dir):
    """(args, median Te, q1, q3, median physics residual) for one run."""
    rj = os.path.join(run_dir, "run.json")
    sc = os.path.join(run_dir, "segments.csv")
    if not (os.path.exists(rj) and os.path.exists(sc)):
        return None
    meta = json.load(open(rj))
    rows = list(csv.DictReader(open(sc)))
    if not rows:
        return None
    te = np.array([float(r["te_km"]) for r in rows])
    lp = np.array([float(r["loss_phys"]) for r in rows])
    return (meta.get("args", {}),
            float(np.median(te)),
            float(np.percentile(te, 25)), float(np.percentile(te, 75)),
            float(np.median(lp)), len(rows), meta.get("elapsed_s"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--retained", default=None,
                    help="the run directory whose configuration was kept; it "
                         "is marked on every panel")
    ap.add_argument("--out", default="ablation.csv")
    a = ap.parse_args()

    keep = os.path.normpath(a.retained) if a.retained else None
    out, seen, skipped = [], set(), []
    for d in a.runs:
        d = os.path.normpath(d)
        s = summarise(d)
        if s is None:
            skipped.append(d)
            continue
        args, med, q1, q3, res, n, secs = s
        is_keep = (d == keep)
        for panel, (field, conv) in AXES.items():
            if field not in args or args[field] is None:
                continue
            try:
                val = conv(args[field])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            key = (panel, round(val, 6))
            if key in seen and not is_keep:
                continue
            seen.add(key)
            out.append(dict(parameter=panel, value=f"{val:.6g}",
                            te_km=f"{med:.4f}", residual=f"{res:.6g}",
                            te_lo=f"{q1:.4f}", te_hi=f"{q3:.4f}",
                            retained="1" if is_keep else "",
                            run=d, n_seg=n))
    if skipped:
        print(f"skipped {len(skipped)} directory(ies) without run.json and "
              f"segments.csv: {', '.join(skipped)}", file=sys.stderr)
    if not out:
        sys.exit("no usable runs found")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["parameter", "value", "te_km",
                                           "residual", "te_lo", "te_hi",
                                           "retained", "run", "n_seg"])
        w.writeheader()
        for r in sorted(out, key=lambda r: (r["parameter"], float(r["value"]))):
            w.writerow(r)

    print(f"wrote {a.out}: {len(out)} rows")
    for panel in AXES:
        rows = [r for r in out if r["parameter"] == panel]
        if not rows:
            print(f"  {panel:16s} NO POINTS -- that panel will be empty")
            continue
        vals = sorted(float(r["value"]) for r in rows)
        te = [float(r["te_km"]) for r in rows]
        mark = "yes" if any(r["retained"] for r in rows) else "NO RETAINED ROW"
        print(f"  {panel:16s} {len(rows)} points, {vals[0]:g} to {vals[-1]:g}; "
              f"Te varies {max(te) - min(te):.2f} km; retained marked: {mark}")
    if len({r["n_seg"] for r in out}) > 1:
        print("  WARNING: the runs do not all cover the same number of "
              "segments; a sweep that changes the sample is not a sweep of "
              "one setting", file=sys.stderr)


if __name__ == "__main__":
    main()
