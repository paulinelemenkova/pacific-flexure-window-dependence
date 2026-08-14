"""
make_profile_geometry.py

Builds the geometry files for Figure 3: the digitised trench axis, the
trench-normal profiles cast from it, and one highlighted profile.

The example trench is the Mariana Trench, whose axis is taken here as the
PB2002 subduction segment MA/PA of Bird (2003). The sampling criteria are those
stated in Section 4.1 of the manuscript:

    profile spacing along the axis   SPACING_KM = 10 km
    profile half-length either side  HALFLEN_KM = 250 km
    sampling interval along profile  SAMPLE_KM  = 1 km

Distances and azimuths are geodesic (WGS84) throughout, so profiles are truly
perpendicular to the local strike of the axis rather than perpendicular in a
projected plane.

Outputs (GMT multisegment, lon 0-360):
    data/mariana_axis.txt        the resampled axis
    data/mariana_profiles.txt    every profile
    data/mariana_profiles_5.txt  every 5th profile, for legible plotting
    data/mariana_highlight.txt   one profile, highlighted in the figure
    data/mariana_samples.txt     sample points along the highlighted profile
"""
import os
import numpy as np
from pyproj import Geod

TRENCH = os.environ.get("TRENCH", "tonga")   # any trench name in axes/
SPACING_KM = 10.0
HALFLEN_KM = 250.0
SAMPLE_KM = 1.0
DENSIFY_KM = 1.0           # step used to densify the raw polyline

geod = Geod(ellps="WGS84")


def read_axis(path):
    """The traced axis, lon lat, as written by trace_axes.sh."""
    pts = [tuple(map(float, l.split()[:2])) for l in open(path)
           if l.strip() and not l.startswith(">")]
    return np.array(pts)


def densify(pts, step_km):
    """Insert points along each leg so the polyline is evenly sampled."""
    out = [tuple(pts[0])]
    for (lo1, la1), (lo2, la2) in zip(pts[:-1], pts[1:]):
        _, _, d = geod.inv(lo1, la1, lo2, la2)
        n = max(int(d / 1000.0 / step_km), 1)
        if n > 1:
            inter = geod.npts(lo1, la1, lo2, la2, n - 1)
            out.extend(inter)
        out.append((lo2, la2))
    return np.array(out)


def along_track(pts):
    d = [0.0]
    for (lo1, la1), (lo2, la2) in zip(pts[:-1], pts[1:]):
        _, _, s = geod.inv(lo1, la1, lo2, la2)
        d.append(d[-1] + s / 1000.0)
    return np.array(d)


axis_raw = read_axis(f"axes/{TRENCH}.txt")
axis = densify(axis_raw, DENSIFY_KM)
s = along_track(axis)
total = s[-1]

# nodes every SPACING_KM along the axis
targets = np.arange(0.0, total + 1e-9, SPACING_KM)
idx = np.searchsorted(s, targets).clip(0, len(axis) - 1)
nodes = axis[idx]

# local strike from the neighbouring nodes, then the perpendicular
profiles = []
for i in range(len(nodes)):
    j0, j1 = max(i - 1, 0), min(i + 1, len(nodes) - 1)
    az, _, _ = geod.inv(nodes[j0][0], nodes[j0][1], nodes[j1][0], nodes[j1][1])
    a, b = az + 90.0, az - 90.0
    lo_a, la_a, _ = geod.fwd(nodes[i][0], nodes[i][1], a, HALFLEN_KM * 1000.0)
    lo_b, la_b, _ = geod.fwd(nodes[i][0], nodes[i][1], b, HALFLEN_KM * 1000.0)
    profiles.append(((lo_b, la_b), tuple(nodes[i]), (lo_a, la_a), b, a))


def w(path, segs):
    with open(path, "w") as f:
        for k, pl in enumerate(segs):
            f.write(f"> profile {k}\n")
            for lo, la in pl:
                f.write(f"{lo % 360:.5f} {la:.5f}\n")


w(f"data/{TRENCH}_axis.txt", [[tuple(p) for p in axis]])
w(f"data/{TRENCH}_profiles.txt", [[p[0], p[1], p[2]] for p in profiles])
w(f"data/{TRENCH}_profiles_5.txt", [[p[0], p[1], p[2]] for p in profiles[::5]])

# a profile near the middle, chosen so that it is one of those drawn every
# fifth in the figure, and therefore coincides with a plotted line
hi = (len(profiles) // 2 // 5) * 5
w(f"data/{TRENCH}_highlight.txt", [[profiles[hi][0], profiles[hi][1], profiles[hi][2]]])

# sample points along the highlighted profile, at the true 1 km interval
lo0, la0 = profiles[hi][1]
az_b, az_a = profiles[hi][3], profiles[hi][4]
with open(f"data/{TRENCH}_samples.txt", "w") as f:
    for d in np.arange(SAMPLE_KM, HALFLEN_KM + 1e-9, SAMPLE_KM):
        for az in (az_b, az_a):
            lo, la, _ = geod.fwd(lo0, la0, az, d * 1000.0)
            f.write(f"{lo % 360:.5f} {la:.5f}\n")
    f.write(f"{lo0 % 360:.5f} {la0:.5f}\n")

print(f"trench                  : {TRENCH}")
print(f"axis length             : {total:.0f} km")
print(f"profiles at {SPACING_KM:.0f} km       : {len(profiles)}")
print(f"profile half-length     : {HALFLEN_KM:.0f} km")
print(f"samples per profile     : {int(2 * HALFLEN_KM / SAMPLE_KM) + 1}")
print(f"highlighted profile     : no. {hi} at "
      f"{profiles[hi][1][0]:.2f}E, {profiles[hi][1][1]:.2f}N")
print("axis lon range          : %.2f to %.2f" % (axis[:, 0].min() % 360, axis[:, 0].max() % 360))
print("axis lat range          : %.2f to %.2f" % (axis[:, 1].min(), axis[:, 1].max()))
