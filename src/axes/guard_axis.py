#!/usr/bin/env python3
r"""guard_axis.py -- reject traced-axis nodes that left the trench.

The tracer takes the deepest sample on each perpendicular search profile. Where
a second depression runs parallel to the trench, that rule can lock onto the
wrong one: the Okinawa Trough shadows the Ryukyu Trench at 1000-2200 m against
the trench's 6700 m, and no bounding box separates them because the two overlap
in both longitude and latitude.

A GLOBAL depth threshold does not work. Trenches shoal towards their
terminations, and by a large amount: Izu-Bonin runs from 9848 m to 5265 m and
Kermadec from 10222 m to 5364 m, so any cut-off deep enough to exclude a
back-arc trough also amputates the ends of those trenches.

The discriminator is not depth but the SHAPE of the along-strike depth series.
A trench shoals gradually, so a node sits close to the median of its immediate
neighbours; a node that has jumped into a parallel depression is an isolated
excursion of several kilometres against that local median. This script
therefore compares each node with a rolling median over --window nodes and
drops those shallower than it by more than --tol.

Usage
  python3 guard_axis.py --in _axis_depth.txt --out _axis_raw.txt \
      --window 21 --tol 2000
Input columns: lon lat depth_m  (depth negative downwards)
"""
import argparse

import numpy as np


def rolling_median(z, window):
    n = len(z)
    w = max(3, min(window | 1, n if n % 2 else n - 1))
    half = w // 2
    padded = np.pad(z, half, mode="edge")
    return np.array([np.median(padded[i:i + w]) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--tol", type=float, default=2000.0,
                    help="drop a node this much shallower than its local median, m")
    ap.add_argument("--name", default="axis")
    args = ap.parse_args()

    a = np.loadtxt(args.src)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] < 3:
        raise SystemExit(f"  {args.name}: fewer than 3 nodes, nothing to guard")

    lon, lat, z = a[:, 0], a[:, 1], a[:, 2]
    local = rolling_median(z, args.window)
    keep = z <= local + args.tol
    dropped = int((~keep).sum())

    np.savetxt(args.dst, np.column_stack([lon[keep], lat[keep]]), fmt="%.6f")
    msg = (f"  {args.name}: {len(z)} nodes, median depth {-np.median(z):.0f} m, "
           f"{dropped} dropped by the depth guard")
    if dropped:
        worst = z[~keep] - local[~keep]
        msg += f" (up to {worst.max():.0f} m above local median)"
    print(msg)
    if dropped > 0.15 * len(z):
        print(f"  {args.name}: WARNING -- more than 15 per cent of nodes "
              f"rejected. The seed may be following the wrong feature; "
              f"inspect axes_check.png before extracting profiles.")


if __name__ == "__main__":
    main()
