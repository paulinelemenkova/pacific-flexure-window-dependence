#!/usr/bin/env python3
"""
morphometry.py -- profile extraction and per-profile parameterisation.

This is the Section 4.1 and 4.2 workflow of the manuscript, written so that it
can be run the moment a bathymetric grid is available:

  sample_profile()   extract a trench-normal profile from a grid
  descriptors()      axial depth, outer-rise amplitude and distance, seaward
                     and landward mean gradients, and the indices A and R
  flexural_fit()     invert the seaward flank for the flexural parameter and
                     the effective elastic thickness
  segment_summary()  median and interquartile range of every descriptor over
                     the profiles of one segment

The grid is reached through a *sampler* callable rather than directly, so the
routines are independent of the grid format and can be tested against an
analytic section whose answers are known exactly (see test_morphometry.py).

Conventions follow the manuscript and Figure 4: x is distance from the trench
axis in km, positive seaward; depth is in km, positive downwards.
"""

import numpy as np
from scipy.optimize import least_squares

R_EARTH = 6371.0088

# elastic constants of Eq. 3, standard oceanic-lithosphere values
E_YOUNG = 70.0e9        # Pa
NU_POISSON = 0.25
RHO_M = 3300.0          # kg m-3
RHO_W = 1030.0          # kg m-3
G_GRAV = 9.81           # m s-2

# search windows, all in km
AXIS_WINDOW = 30.0      # the axial depth is the deepest point within this of x=0
CREST_MIN_X = 25.0      # the outer-rise crest is sought beyond this distance
REGIONAL_PAD = 60.0     # the reference depth is taken beyond crest + this
SMOOTH_KM = 6.0         # running-mean window used before turning points
BREAK_FRAC = 0.35       # landward break: gradient falls below this fraction
BREAK_RUN_KM = 10.0     # ... and stays below it over this distance


# --------------------------------------------------------------- extraction -
def destination(lon, lat, az_deg, dist_km):
    d = dist_km / R_EARTH
    p1, l1, a = np.radians(lat), np.radians(lon), np.radians(az_deg)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(a))
    l2 = l1 + np.arctan2(np.sin(a) * np.sin(d) * np.cos(p1),
                         np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(l2), np.degrees(p2)


def grid_sampler(da, lon_name="lon", lat_name="lat"):
    """Bilinear sampler over an xarray DataArray of elevation in metres."""
    def sample(lons, lats):
        pts = {lon_name: ("p", np.asarray(lons)),
               lat_name: ("p", np.asarray(lats))}
        z = da.interp(**pts, method="linear").values
        return -np.asarray(z, dtype=float) / 1000.0   # elevation m -> depth km
    return sample


def sample_profile(sampler, lon0, lat0, az_seaward,
                   x_land=150.0, x_sea=250.0, spacing=1.0):
    """Trench-normal profile through (lon0, lat0); x positive seaward, km."""
    x = np.arange(-x_land, x_sea + spacing / 2, spacing)
    az = np.where(x >= 0, az_seaward, (az_seaward + 180.0) % 360.0)
    lon, lat = destination(lon0, lat0, az, np.abs(x))
    return x, sampler(lon, lat)


# ------------------------------------------------------------- descriptors --
def _smooth(y, dx, window_km=SMOOTH_KM):
    n = max(1, int(round(window_km / dx)))
    if n < 2:
        return y.copy()
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - n // 2 - 1), mode="edge"), k,
                       mode="valid")


def _lsq_gradient(x, d):
    """Magnitude of the least-squares gradient of depth on distance."""
    if len(x) < 3:
        return np.nan
    return abs(np.polyfit(x, d, 1)[0])


def _axis(x, ds):
    """Index, position and depth of the deepest point near x = 0."""
    near = np.abs(x) <= AXIS_WINDOW
    i0 = int(np.nanargmax(np.where(near, ds, -np.inf)))
    return i0, float(x[i0])


def _landward_break(x, ds, x0):
    """Distance and depth of the trench slope break, landward of the axis.

    Worked entirely in kilometres rather than in samples, so that the answer
    does not depend on the sampling interval of the grid.
    """
    land = x <= x0
    xl = x[land][::-1]
    dl = ds[land][::-1]
    if len(xl) < 8:
        return np.nan, np.nan
    s = x0 - xl                                   # distance landward, km
    g = np.abs(np.gradient(dl, xl))
    core = (s >= 3.0) & (s <= 30.0)
    if not core.any():
        return np.nan, np.nan
    gmax = float(np.nanmax(g[core]))
    thr = BREAK_FRAC * gmax
    run = 0.0
    for i in range(1, len(s)):
        if s[i] < 8.0:
            continue
        if g[i] < thr:
            run += s[i] - s[i - 1]
            if run >= BREAK_RUN_KM:
                j = i
                while j > 0 and g[j - 1] < thr and s[j] - s[j - 1] > 0:
                    j -= 1
                return float(xl[j]), float(dl[j])
        else:
            run = 0.0
    return float(xl[-1]), float(dl[-1])


def descriptors(x, d, d_r=None, measured_fraction=None):
    """Morphometric descriptors of one profile.  Returns a dict.

    d_r, the regional reference depth of the unbent plate, may be supplied
    from the flexural fit, whose far-field asymptote estimates it better than
    a median over an interval that the bending signal has not fully decayed
    across.  If it is not supplied it is taken as the median beyond the crest.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    good = np.isfinite(d)
    if good.sum() < 0.5 * len(d):
        return dict(valid=False)
    dx = float(np.median(np.diff(x)))
    ds = _smooth(np.where(good, d, np.nanmedian(d)), dx)

    i0, x0 = _axis(x, ds)
    d0 = float(d[i0])

    # outer-rise crest: the shallowest point seaward of the axis, taken from
    # the smoothed profile so that a single noisy sample cannot define it
    x_hi = min(x[-1] - 40.0, x0 + 240.0)
    win = (x > x0 + CREST_MIN_X) & (x <= x_hi)
    if win.sum() < 10:
        return dict(valid=False)
    ib = int(np.nanargmin(ds[win]))
    x_b_abs = float(x[win][ib])
    d_b = float(ds[win][ib])

    if d_r is None:
        far = x >= x_b_abs + REGIONAL_PAD
        d_r = float(np.nanmedian(d[far])) if far.sum() >= 10 else np.nan
    far_reach = float(x[-1] - x_b_abs)

    w_b = d_r - d_b
    x_b = x_b_abs - x0
    rise_ok = np.isfinite(w_b) and w_b > 0.02      # a rise of at least 20 m
    if not rise_ok:
        w_b, x_b = np.nan, np.nan

    x_break_abs, d_break = _landward_break(x, ds, x0)

    m_land = (x >= x_break_abs) & (x <= x0)
    m_sea = (x >= x0) & (x <= x_b_abs)
    S_l = _lsq_gradient(x[m_land], d[m_land])
    S_s = _lsq_gradient(x[m_sea], d[m_sea])

    tot = S_l + S_s
    A = (S_l - S_s) / tot if np.isfinite(tot) and tot > 0 else np.nan
    R = (d0 - d_r) / w_b if rise_ok and np.isfinite(d_r) else np.nan

    return dict(valid=True, rise_resolved=bool(rise_ok), x0=x0, d0=d0,
                d_r=d_r, x_b=x_b, w_b=w_b, d_b=d_b,
                x_break=x_break_abs - x0, d_break=d_break,
                S_s=S_s, S_l=S_l, A=A, R=R, far_reach=far_reach,
                measured_fraction=measured_fraction)


# ---------------------------------------------------------------- flexure ---
def deflection(x, w0, alpha):
    """Broken-plate deflection, the decaying solution of Eq. 2."""
    return w0 * np.exp(-x / alpha) * np.cos(x / alpha)


def te_from_alpha(alpha_km):
    """Effective elastic thickness (km) from the flexural parameter (km)."""
    alpha = alpha_km * 1000.0
    D = alpha ** 4 * (RHO_M - RHO_W) * G_GRAV / 4.0
    te = (12.0 * (1.0 - NU_POISSON ** 2) * D / E_YOUNG) ** (1.0 / 3.0)
    return te / 1000.0


def alpha_from_te(te_km):
    te = te_km * 1000.0
    D = E_YOUNG * te ** 3 / (12.0 * (1.0 - NU_POISSON ** 2))
    return (4.0 * D / ((RHO_M - RHO_W) * G_GRAV)) ** 0.25 / 1000.0


def flexural_fit(x, d, x0=0.0, d_r=None, x_fit_max=250.0):
    """Invert the seaward flank for the flexural parameter and Te.

    Fits d(x) = d_r + w0 exp(-(x-x0)/alpha) cos((x-x0)/alpha) over the interval
    from the axis to x_fit_max, solving for w0, alpha and d_r.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    m = (x >= x0) & (x <= x0 + x_fit_max) & np.isfinite(d)
    if m.sum() < 20:
        return dict(valid=False)
    xx, dd = x[m] - x0, d[m]
    d_r0 = float(np.median(dd[-max(5, len(dd) // 5):])) if d_r is None else d_r
    p0 = [max(dd[0] - d_r0, 0.5), 60.0, d_r0]

    def resid(p):
        return deflection(xx, p[0], p[1]) + p[2] - dd

    sol = least_squares(resid, p0, bounds=([0.05, 15.0, d_r0 - 2.0],
                                           [8.0, 200.0, d_r0 + 2.0]))
    w0, alpha, dref = sol.x
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    return dict(valid=bool(sol.success), w0=float(w0), alpha=float(alpha),
                d_r=float(dref), Te=te_from_alpha(float(alpha)), rms=rms)


# ---------------------------------------------------------------- segments --
def segment_summary(records):
    """Median and interquartile range of every descriptor over one segment."""
    keys = [k for k in ("d0", "d_r", "x_b", "w_b", "S_s", "S_l", "A", "R",
                        "alpha", "Te", "rms", "measured_fraction")
            if any(k in r for r in records)]
    out = {"n_profiles": len(records)}
    for k in keys:
        v = np.array([r[k] for r in records if r.get(k) is not None
                      and np.isfinite(r.get(k, np.nan))], dtype=float)
        if v.size:
            out[k] = float(np.median(v))
            out[k + "_iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
        else:
            out[k] = np.nan
            out[k + "_iqr"] = np.nan
    return out


def parameterise(x, d, x_fit_max=320.0, measured_fraction=None):
    """Full per-profile parameterisation: flexural fit, then descriptors.

    The flexural fit is run first because its far-field asymptote supplies the
    regional reference depth that the outer-rise amplitude is measured from.
    """
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)
    dx = float(np.median(np.diff(x)))
    ds = _smooth(np.where(np.isfinite(d), d, np.nanmedian(d)), dx)
    _, x0 = _axis(x, ds)
    fit = flexural_fit(x, d, x0=x0, x_fit_max=x_fit_max)
    des = descriptors(x, d, d_r=fit.get("d_r") if fit.get("valid") else None,
                      measured_fraction=measured_fraction)
    if fit.get("valid"):
        des.update(alpha=fit["alpha"], Te=fit["Te"], w0=fit["w0"],
                   rms=fit["rms"])
    return des
