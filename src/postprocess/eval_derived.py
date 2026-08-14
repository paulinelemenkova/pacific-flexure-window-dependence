#!/usr/bin/env python3
"""eval_derived.py -- the post-training evaluation pass.

Recovers the three derived flexural quantities of Equation (derived) from the
trained solution in checkpoint.pt and writes them per segment in the units
fig_parametermaps.sh expects:

    M(x) = -D w''(x)      bending moment per unit length      N
    V(x) = -D w'''(x)     shear force per unit length         N/m
    kappa(x) = w''(x)     curvature                           1/m

train_pinn.py writes the thickness and the loss terms per segment but not
these, which is why the parameter maps had no input. Nothing here is refitted:
the network, the embeddings and the 71 recovered thicknesses are read from the
checkpoint and only differentiated.

THE SCALING IS THE WHOLE PROBLEM
--------------------------------
The trunk is trained on scaled variables -- Trunk.forward takes x/L and
returns w/w0 -- so a derivative of the network output is NOT a derivative of
the deflection. From train_pinn.py:

    xs = x / L          L = 550000 m, the common seaward reach (run.json L_m)
    ws = w / w0         w0 = max|w| of that segment, in metres (segments.csv)

so w(x) = w0 * w_hat(x/L) and each derivative carries a factor 1/L:

    kappa = (w0 / L^2)  w_hat''
    M     = -D (w0 / L^2) w_hat''
    V     = -D (w0 / L^3) w_hat'''

Getting any of those factors wrong yields a smooth, plausibly shaped field
that is wrong by orders of magnitude, which is exactly the failure mode this
figure has already had once. They are asserted here in one place and the
script prints a magnitude check against the panel ranges so the error is
visible rather than silent.

D is train_pinn.rigidity, imported rather than reimplemented, so E and nu
cannot drift from the values the inversion actually used.

WHERE EACH QUANTITY IS TAKEN
----------------------------
The figure carries one number per segment, so a field must be reduced:

  moment     M at the trench axis, x = 0   -- the caption reads "bending
             moment per unit length along strike", and the axis is where the
             plate is most strongly bent and where the boundary condition of
             the broken-plate solution applies.
  shear      V at the trench axis, x = 0   -- the script's own header calls
             panel (b) "shear force at the axis".
  curvature  max |kappa| over the profile  -- the caption reads "maximum
             curvature along the profile".

Both the axial and the extremal value of every quantity are written, so the
choice can be changed in the figure without recomputing: --reduce axis|max
selects which triple is named moment/shear/curvature.

USAGE
-----
    python3 eval_derived.py \\
        --checkpoint runs/r02/checkpoint.pt \\
        --segments   runs/r02/segments.csv \\
        --deflection runs/r02/deflection.csv \\
        --run        runs/r02/run.json \\
        --out derived_segments.csv

    RELIEF="/Volumes/TOSHIBA EXT/DATA/GEBCO_2023.nc" \\
    SEGMENTS=te_segments.gmt VALUES=derived_segments.csv \\
    REG=116/290/-60/62 OUT=fig_parametermaps bash fig_parametermaps.sh

NOT WRITTEN: te_err
-------------------
Panel (d) is the ensemble standard deviation of Section 4. run.json records
seed 0 and a single member, and a standard deviation over one sample does not
exist. No te_err column is emitted; fig_parametermaps.sh leaves a segment with
a blank field undrawn, so panel (d) comes out empty until the ensemble is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from train_pinn import (Trunk, rigidity, E_YOUNG, NU_POISSON,
                        TE_MIN_KM, TE_MAX_KM)
except ImportError as e:
    sys.exit(f"train_pinn.py must sit beside this script: {e}")


def derivatives(model, xs, sid, order=3):
    """w_hat and its derivatives with respect to the SCALED coordinate.

    Mirrors train_pinn.d4, but stops at the third derivative because M needs
    the second and V the third; the fourth is only wanted by the physics
    residual.
    """
    xs = xs.clone().requires_grad_(True)
    w = model(xs, sid)
    out = [w]
    g = w
    for _ in range(order):
        g = torch.autograd.grad(g, xs, torch.ones_like(g), create_graph=True)[0]
        out.append(g)
    return [t.detach().squeeze(-1).cpu().numpy() for t in out]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="runs/r02/checkpoint.pt")
    ap.add_argument("--segments", default="runs/r02/segments.csv")
    ap.add_argument("--deflection", default="runs/r02/deflection.csv")
    ap.add_argument("--run", default="runs/r02/run.json")
    ap.add_argument("--out", default="derived_segments.csv")
    ap.add_argument("--reduce", choices=("axis", "max"), default="axis",
                    help="which reduction is named moment/shear/curvature; "
                         "curvature is always the profile maximum, per the "
                         "figure caption")
    a = ap.parse_args()

    for p in (a.checkpoint, a.segments, a.deflection, a.run):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    meta = json.load(open(a.run))
    L = float(meta["L_m"])
    args = meta.get("args", {})
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)

    rows = list(csv.DictReader(open(a.segments)))
    n_seg = int(ck.get("n_seg", len(rows)))
    if len(rows) != n_seg:
        sys.exit(f"{a.segments} has {len(rows)} rows but the checkpoint holds "
                 f"{n_seg} segments; they must be the same run")

    # Row i of segments.csv is embedding index i: train_pinn.build_segments
    # walks sorted(by_trench) and appends in that order, and main() writes the
    # table in the same order. Verified against the file.
    model = Trunk(n_seg=n_seg,
                  width=int(args.get("width", 64)),
                  depth=int(args.get("depth", 5)),
                  emb=int(args.get("embedding", 4)),
                  fourier=int(args.get("fourier", 16)),
                  sigma=float(args.get("fourier_sigma", 2.0)))
    model.load_state_dict(ck["model"])
    model.eval()
    # train_pinn.te_of(): clamp(10**log_te, TE_MIN_KM, TE_MAX_KM). NOT exp --
    # using exp gave a rigidity 45 times too small, and since D enters M and V
    # linearly the moments would have been wrong by that factor while still
    # looking like a smooth, plausible field. The agreement check below is
    # what caught it and is left in as a permanent guard.
    te_km = torch.clamp(10.0 ** ck["log_te"], TE_MIN_KM, TE_MAX_KM)
    te_km = te_km.detach().cpu().numpy()
    tab_te = np.array([float(r["te_km"]) for r in rows])
    dev = float(np.max(np.abs(te_km - tab_te)))
    if dev > 1e-6 * max(1.0, float(np.max(tab_te))):
        sys.exit(f"thickness from the checkpoint disagrees with {a.segments} "
                 f"by up to {dev:.3g} km. The two must be the same run, and "
                 f"te must be recovered exactly as train_pinn.te_of() does it.")
    print(f"Te from checkpoint matches {a.segments} to {dev:.2e} km "
          f"across {len(rows)} segments")

    # x grid per segment, from the stacked deflections the run itself wrote
    grid = {}
    with open(a.deflection) as fh:
        for r in csv.DictReader(fh):
            grid.setdefault(r["segment"], []).append(float(r["x_km"]))

    out, checks = [], []
    for k, r in enumerate(rows):
        name = r["segment"]
        w0 = float(r["w0_m"])
        xs_km = np.array(sorted(grid.get(name, [])), dtype=float)
        if xs_km.size < 3:
            print(f"  {name}: no deflection grid, skipped", file=sys.stderr)
            continue
        x_m = xs_km * 1.0e3
        xs = torch.tensor(x_m / L, dtype=torch.float64).unsqueeze(1)
        sid = torch.full((xs.shape[0],), k, dtype=torch.long)
        model_d = model.double()
        _, d1, d2, d3 = derivatives(model_d, xs, sid, order=3)

        D = float(rigidity(torch.tensor(float(te_km[k]))))
        kappa = (w0 / L ** 2) * d2                 # 1/m
        moment = -D * kappa                        # N
        shear = -D * (w0 / L ** 3) * d3            # N/m

        j = int(np.argmin(np.abs(x_m)))            # the trench axis
        jk = int(np.argmax(np.abs(kappa)))         # extremal curvature
        rec = dict(segment=name, trench=r["trench"],
                   te_km=f"{te_km[k]:.6g}",
                   moment_axis=f"{moment[j]:.6g}",
                   shear_axis=f"{shear[j]:.6g}",
                   kappa_axis=f"{kappa[j]:.6g}",
                   moment_max=f"{moment[jk]:.6g}",
                   shear_max=f"{shear[int(np.argmax(np.abs(shear)))]:.6g}",
                   kappa_max=f"{kappa[jk]:.6g}")
        out.append(rec)
        checks.append((abs(moment[j]), abs(shear[j]), abs(kappa[jk])))

    if not out:
        sys.exit("no segments evaluated")

    # The columns fig_parametermaps.sh reads, plus the full set for provenance.
    pick = "axis" if a.reduce == "axis" else "max"
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["segment", "moment", "shear", "curvature", "trench", "te_km",
                "moment_axis", "shear_axis", "kappa_axis",
                "moment_max", "shear_max", "kappa_max"]
        w.writerow(head)
        for rec in out:
            w.writerow([rec["segment"],
                        rec[f"moment_{pick}"],
                        rec[f"shear_{pick}"],
                        rec["kappa_max"],          # caption: profile maximum
                        rec["trench"], rec["te_km"],
                        rec["moment_axis"], rec["shear_axis"], rec["kappa_axis"],
                        rec["moment_max"], rec["shear_max"], rec["kappa_max"]])

    m = np.array([c[0] for c in checks])
    v = np.array([c[1] for c in checks])
    kp = np.array([c[2] for c in checks])
    print(f"wrote {a.out}: {len(out)} segments, reduction '{a.reduce}'")
    print(f"  |M|   median {np.median(m):.3e} N     range {m.min():.3e} to {m.max():.3e}")
    print(f"  |V|   median {np.median(v):.3e} N/m   range {v.min():.3e} to {v.max():.3e}")
    print(f"  |kap| median {np.median(kp):.3e} 1/m  range {kp.min():.3e} to {kp.max():.3e}")
    print(f"  D from E={E_YOUNG:.3g} Pa, nu={NU_POISSON}, L={L:.0f} m")
    # A magnitude check against the panel defaults, so a scaling error is
    # visible here rather than as a uniformly blank or saturated panel.
    print("\n  panel ranges in fig_parametermaps.sh: M +/-8e16 N, "
          "V 0-6e12 N/m, kappa 0-8e-7 1/m")
    for label, arr, hi in (("M", m, 8e16), ("V", v, 6e12), ("kappa", kp, 8e-7)):
        frac = np.median(arr) / hi
        note = ("saturates" if frac > 1 else
                "invisible, rescale the panel" if frac < 0.01 else "fits")
        print(f"    {label:5s} median is {frac:8.4f} of the panel maximum -- {note}")


if __name__ == "__main__":
    main()
