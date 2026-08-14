#!/usr/bin/env python3
"""export_deflection.py -- write the recovered deflection of every segment.

Rebuilds the stacked segments exactly as the training run built them, restores
the trunk from checkpoint.pt, and evaluates it on each segment's own abscissa.
The result is one CSV carrying BOTH the observed stack and the recovery, so the
figure script needs no torch and cannot drift from the run it depicts.

Run this in the TRAINING environment (pinn: torch, numpy, pandas). The figure
is then drawn in the analysis environment from the CSV alone.

    python3 export_deflection.py --run runs/r02 --profiles profiles_csv \
        --out runs/r02/deflection.csv

Output columns
--------------
    segment, trench, x_km, w_obs_m, w_pinn_m
x_km is measured from the trench axis, positive seaward; both deflections are
in metres, positive downward, referred to the regional level, so each decays to
zero in the far field. This is the deflection of Equation (12), not depth.

Every hyper-parameter is read from run.json rather than re-specified here, so
the reconstruction cannot silently disagree with the run: a mismatch in the
segment count or the model geometry raises instead of producing a plausible
wrong curve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from train_pinn import Trunk, build_segments, TE_MIN_KM, TE_MAX_KM


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory holding "
                    "checkpoint.pt, run.json and segments.csv")
    ap.add_argument("--profiles", default=None,
                    help="profile directory; defaults to the one in run.json")
    ap.add_argument("--out", default=None,
                    help="output CSV; defaults to <run>/deflection.csv")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.run, "run.json")))
    args = meta["args"]
    profdir = a.profiles or args["profiles"]
    out = a.out or os.path.join(a.run, "deflection.csv")

    # --- rebuild the segments the run was trained on -------------------------
    segs = build_segments(profdir,
                          args.get("trenches"),
                          args["max_profiles_per_segment"],
                          sediment=not args["no_sediment"],
                          stack=args["stack"])
    if len(segs) != meta["segments"]:
        sys.exit(f"rebuilt {len(segs)} segments but the run recorded "
                 f"{meta['segments']}; the profile directory is not the one "
                 f"that was trained on")

    # L is the common scale of Section 4.4 and is recorded, so it is taken from
    # the run rather than recomputed -- recomputing it would silently rescale
    # every input if a single profile were added or removed.
    L = float(meta["L_m"])
    L_rebuilt = float(max(s["x"].max() for s in segs))
    if not np.isclose(L, L_rebuilt):
        sys.exit(f"run.json records L = {L:.1f} m but the rebuilt segments "
                 f"give {L_rebuilt:.1f} m; inputs differ from training")

    # --- restore the trunk ---------------------------------------------------
    ck = torch.load(os.path.join(a.run, "checkpoint.pt"), map_location="cpu",
                    weights_only=False)
    if ck["n_seg"] != len(segs):
        sys.exit(f"checkpoint holds {ck['n_seg']} segments, rebuilt "
                 f"{len(segs)}")
    model = Trunk(len(segs), width=args["width"], depth=args["depth"],
                  emb=args["embedding"], fourier=args["fourier"],
                  sigma=args["fourier_sigma"])
    model.load_state_dict(ck["model"])   # strict: B, the fixed Fourier matrix,
    model.eval()                         # is a buffer and is restored with it

    # Te = clamp(10**log_te, ...) -- a base-10 log parameterisation, NOT the
    # softplus of Listing 1. The two agree only for small values, so using the
    # wrong one understates the largest thicknesses by tens of kilometres.
    te = torch.clamp(10.0 ** ck["log_te"], TE_MIN_KM, TE_MAX_KM).detach().numpy()
    print(f"restored iteration {ck['iteration']} (stage {ck['stage']}), "
          f"{len(segs)} segments, L = {L/1e3:.0f} km")

    # --- evaluate ------------------------------------------------------------
    rows = []
    with torch.no_grad():
        for k, s in enumerate(segs):
            w0 = float(np.max(np.abs(s["w"])))
            xs = torch.tensor(s["x"] / L, dtype=torch.float32).reshape(-1, 1)
            sid = torch.full((xs.shape[0],), k, dtype=torch.long)
            w_hat = model(xs, sid).squeeze(1).numpy()
            rows.append(pd.DataFrame({
                "segment": s["name"], "trench": s["trench"],
                "x_km": s["x"] / 1.0e3,
                "w_obs_m": s["w"],
                "w_pinn_m": w_hat * w0,          # undo the per-segment scaling
            }))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows, {df.segment.nunique()} segments)")

    # --- agreement check against segments.csv --------------------------------
    # rms_fit_m was written by the training run; recomputing it here from the
    # exported curves is an end-to-end test that the scaling, the segment
    # ordering and the embedding index were all reconstructed correctly. A
    # mismatch means the export is wrong, however plausible the curves look.
    ref = pd.read_csv(os.path.join(a.run, "segments.csv"))
    got = (df.assign(e=(df.w_obs_m - df.w_pinn_m) ** 2)
             .groupby("segment").e.mean().pow(0.5).rename("rms_here"))
    cmp = ref.set_index("segment")[["te_km", "rms_fit_m"]].join(got)
    d = (cmp.rms_here - cmp.rms_fit_m).abs()
    print(f"rms agreement with segments.csv: median {d.median():.1f} m, "
          f"worst {d.max():.1f} m ({d.idxmax()})")
    if d.max() > 1.0:
        print("WARNING: exported curves disagree with the recorded fit -- "
              "do not plot these until the cause is found", file=sys.stderr)

    te_ref = ref.set_index("segment").te_km.reindex([s["name"] for s in segs])
    dte = np.abs(te - te_ref.to_numpy())
    print(f"Te agreement: worst {np.nanmax(dte):.4f} km")


if __name__ == "__main__":
    main()
