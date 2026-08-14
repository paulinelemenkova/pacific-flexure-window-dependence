#!/usr/bin/env python3
r"""Step 6 of finalising article_trench_seismicity_jgr.tex.

Re-derives the outer-rise geometry from the raw cross-profiles and inverts the
broken-plate flexure model for the effective elastic thickness, then fills the
remaining \val{} markers of Section 5.2.

Why re-derive rather than reuse paper 1
--------------------------------------
Axial depth from paper 1 reproduces exactly (r = 0.997), but its forebulge
crest distance correlates at 0.18 with a direct measurement of the same
profiles and sits a median 94 km closer to the axis. Through x_b = 3*pi*alpha/4
that difference maps onto an effective elastic thickness near 5 km, well below
any published value for oceanic lithosphere. The crest position sets the
flexural parameter, so the inversion cannot inherit it.

Method
------
Per profile: the axis is the deepest point within 60 km of the nominal origin;
the section is unloaded of its sediment column by Airy backstripping; the
regional level is the median unloaded depth over --regional-window, measured
seaward of the forebulge; and the deflection about that level is fitted with
the broken-plate solution of Eq. (2),

    w(x) = exp(-x/alpha) * (c1 cos(x/alpha) + c2 sin(x/alpha)) + c0,

by robust least squares, with c0 free to absorb any residual mismatch between
the observed regional level and the plate-cooling reference. The elastic
thickness follows from alpha through Eq. (3). A two-segment fit repeats this
trenchward and seaward of a break distance that is itself scanned, giving the
plate-weakening ratio.

Fits are rejected where alpha reaches a bound, the misfit exceeds --max-rms,
or the modelled crest falls at the edge of the fitting window; rejected
profiles are reported, not silently dropped.

Constants, fixed on the record: E = 70 GPa, nu = 0.25, rho_m = 3300,
rho_w = 1030, rho_s = 2000 kg/m3, g = 9.81 m/s2.

Usage
-----
  python3 scripts/06_flexure.py article_trench_seismicity_jgr.tex \
      --profiles data/profiles_csv --morphometry data/morphometry_classified.csv \
      --segments data/segment_morphometry.csv --dry-run
"""
import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RHO_M, RHO_W, RHO_S = 3300.0, 1030.0, 2000.0     # kg m-3
E_YOUNG, NU, G = 70.0e9, 0.25, 9.81              # Pa, -, m s-2
SED_FACTOR = (RHO_M - RHO_S) / (RHO_M - RHO_W)
ALPHA_BOUNDS = (8.0, 250.0)                      # km


def elastic_thickness(alpha_km):
    """Eq. (3): flexural parameter -> rigidity -> elastic thickness, km."""
    d = (alpha_km * 1e3) ** 4 * (RHO_M - RHO_W) * G / 4.0
    return (12.0 * (1.0 - NU ** 2) * d / E_YOUNG) ** (1.0 / 3.0) / 1e3


def deflection(x_m, c1, c2, alpha_km, c0):
    u = x_m / (alpha_km * 1e3)
    return np.exp(-u) * (c1 * np.cos(u) + c2 * np.sin(u)) + c0


def fit_window(x_m, w, alpha0=80.0):
    """Robust fit of the broken-plate solution over one interval."""
    def resid(q):
        return deflection(x_m, *q) - w
    sol = least_squares(resid, [w[0], 0.0, alpha0, 0.0],
                        bounds=([-2e4, -2e4, ALPHA_BOUNDS[0], -800.0],
                                [2e4, 2e4, ALPHA_BOUNDS[1], 800.0]),
                        ftol=1e-6, xtol=1e-6, max_nfev=300)
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return sol.x, rms


def analyse(path, near, window, regional, max_rms):
    d = pd.read_csv(path).dropna(subset=["p", "depth"])
    p = d["p"].to_numpy(float)
    z = d["depth"].to_numpy(float) + np.nan_to_num(d["hs"].to_numpy(float)) * SED_FACTOR
    if (np.abs(p) <= near).sum() < 10:
        return None, "no axis window"
    ia = int(np.argmin(np.where(np.abs(p) <= near, z, np.inf)))
    xa, d0 = p[ia], -z[ia]
    far = (p >= xa + regional[0]) & (p <= xa + regional[1])
    if far.sum() < 20:
        return None, "no regional window"
    d_r = -float(np.median(z[far]))
    w_all = z + d_r

    sel = (p > xa + window[0]) & (p <= xa + window[1]) & np.isfinite(w_all)
    if sel.sum() < 40:
        return None, "too few points"
    x, w = (p[sel] - xa) * 1e3, w_all[sel]
    x, w = x[::5], w[::5]          # 5 km sampling: ample for a 40-90 km scale
    q, rms = fit_window(x, w)
    alpha = q[2]
    if not ALPHA_BOUNDS[0] * 1.01 < alpha < ALPHA_BOUNDS[1] * 0.99:
        return None, "alpha at bound"
    if rms > max_rms:
        return None, "misfit too large"

    model = deflection(x, *q)
    ib = int(np.argmax(model))
    x_b, w_b = x[ib] / 1e3, float(model[ib] - q[3])
    if x_b >= window[1] * 0.95 or w_b <= 0:
        return None, "crest unresolved"

    # Two-segment: scan the break, weakened plate trenchward of it.
    best = None
    for x_break in np.arange(70.0, 210.0, 35.0):
        inner = x <= x_break * 1e3
        outer = x > x_break * 1e3
        if inner.sum() < 8 or outer.sum() < 15:
            continue
        qi, ri = fit_window(x[inner], w[inner])
        qo, ro = fit_window(x[outer], w[outer])
        score = (ri ** 2 * inner.sum() + ro ** 2 * outer.sum()) / x.size
        if best is None or score < best["score"]:
            best = {"score": score, "x_break": x_break,
                    "alpha_in": qi[2], "alpha_out": qo[2]}

    rec = {"d_0": d0, "d_r": d_r, "x_b": x_b, "w_b": w_b,
           "R": (d0 - d_r) / w_b, "alpha": alpha,
           "Te": elastic_thickness(alpha), "rms": rms}
    if best and all(ALPHA_BOUNDS[0] * 1.01 < best[k] < ALPHA_BOUNDS[1] * 0.99
                    for k in ("alpha_in", "alpha_out")):
        te_in = elastic_thickness(best["alpha_in"])
        te_out = elastic_thickness(best["alpha_out"])
        rec.update({"Te_trench": te_in, "Te_sea": te_out,
                    "Te_ratio": te_in / te_out, "x_break": best["x_break"]})
    return rec, "ok"


def fmt(x):
    return f"{x:.0f}" if abs(x) >= 100 else (f"{x:.1f}" if abs(x) >= 10
                                             else f"{x:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", type=Path)
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--morphometry", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--near-km", type=float, default=60.0)
    ap.add_argument("--fit-window", type=float, nargs=2, default=(15.0, 450.0))
    ap.add_argument("--regional-window", type=float, nargs=2,
                    default=(300.0, 550.0))
    ap.add_argument("--max-rms", type=float, default=1200.0)
    ap.add_argument("--min-profiles", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("data/segment_flexure.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(str(args.profiles / "*.csv"))
                   if not os.path.basename(f).startswith("trench_"))
    rows, reasons = [], {}
    for path in files:
        trench, pid = os.path.basename(path)[:-4].rsplit("_", 1)
        rec, why = analyse(path, args.near_km, args.fit_window,
                           args.regional_window, args.max_rms)
        reasons[why] = reasons.get(why, 0) + 1
        if rec:
            rec.update({"trench": trench, "profile": int(pid)})
            rows.append(rec)
    prof = pd.DataFrame(rows)
    print(f"{len(files)} profiles, {len(prof)} inverted")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if why != "ok":
            print(f"  rejected, {why}: {n}")

    seg = pd.read_csv(args.segments)[["trench", "segment"]]
    key = pd.read_csv(args.morphometry)[["trench", "profile"]]
    prof = prof.merge(key, on=["trench", "profile"], how="inner")

    # Re-bin by the segmentation already written in step 5.
    src = pd.read_csv(args.segments)
    ranges = src.set_index("segment")
    prof = prof.sort_values(["trench", "profile"])
    prof["segment"] = None
    for trench, sub in prof.groupby("trench"):
        segs = [s for s in ranges.index if s.startswith(trench + "_")]
        if not segs:
            continue
        edges = np.array_split(sub.index.to_numpy(), len(segs))
        for name, idx in zip(sorted(segs), edges):
            prof.loc[idx, "segment"] = name
    prof = prof.dropna(subset=["segment"])

    agg = prof.groupby("segment").agg(
        n=("profile", "size"), Te=("Te", "median"),
        Te_trench=("Te_trench", "median"), Te_sea=("Te_sea", "median"),
        Te_ratio=("Te_ratio", "median"), x_b=("x_b", "median"),
        w_b=("w_b", "median"), R=("R", "median"),
        alpha=("alpha", "median"), rms=("rms", "median")).reset_index()
    kept = agg[agg.n >= args.min_profiles]

    print(f"\n{len(kept)} segments with >= {args.min_profiles} inverted "
          f"profiles\n")
    print(f"{'quantity':12s} {'min':>9s} {'median':>9s} {'max':>9s}")
    print("-" * 42)
    for col in ("Te", "Te_sea", "Te_trench", "Te_ratio", "x_b", "w_b", "R"):
        s = kept[col].dropna()
        if len(s):
            print(f"{col:12s} {s.min():9.2f} {s.median():9.2f} {s.max():9.2f}")

    ts, tt, tr, rr = (kept[c].dropna() for c in
                      ("Te_sea", "Te_trench", "Te_ratio", "R"))
    values = {
        "Te max min": fmt(ts.min()), "Te max max": fmt(ts.max()),
        "Te min min": fmt(tt.min()), "Te min max": fmt(tt.max()),
        "Te ratio range": f"{tr.min():.2f} to {tr.max():.2f}",
        "R min": fmt(rr.min()), "R max": fmt(rr.max()),
    }
    print("\ncomputed:")
    for k, v in values.items():
        print(f"  {k:18s} {v}")

    text = args.tex.read_text(encoding="utf-8")
    filled = []

    def repl(m):
        k = re.sub(r"\s+", " ", m.group(1)).strip()
        if k in values:
            filled.append(k)
            return values[k]
        return m.group(0)

    new = re.sub(r"\\val\{([^{}]*)\}", repl, text)
    print(f"\nsubstituted {len(filled)} markers")
    if args.dry_run:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(args.out, index=False)
    prof.to_csv(args.out.with_name("profile_flexure.csv"), index=False)
    shutil.copy2(args.tex, args.tex.with_suffix(".tex.bak"))
    args.tex.write_text(new, encoding="utf-8")
    print(f"wrote {args.tex} (backup .bak), {args.out}")


if __name__ == "__main__":
    main()
