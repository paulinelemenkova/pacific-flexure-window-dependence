#!/usr/bin/env python3
"""train_pinn -- window-free flexural inversion by physics-informed learning.

Implements Section 4.4 and Algorithm 2 of the manuscript: the deflection is
represented by a network whose fourth derivative is available exactly through
automatic differentiation, the elastic plate equation is imposed as a residual
at collocation points drawn from the whole seaward reach, and the elastic
thickness is a trainable coefficient of the same graph. No window enters the
formulation, so nothing corresponding to x_l or x_s is ever chosen.

    python3 train_pinn.py --profiles profiles_csv --outdir runs/r01
    python3 train_pinn.py --profiles profiles_csv --outdir runs/r01 \\
        --trenches mariana tonga --adam 20000 --lbfgs 300

WHAT IT WRITES
  history.csv    one row per iteration: iteration, loss_total, loss_data,
                 loss_phys, loss_bc, te_km -- exactly the schema
                 fig_convergence.py reads. The three components are recorded
                 AFTER their weights are applied, so that they sum to the
                 composite; fig_convergence.check_weighting verifies this and
                 the gradient-norm rebalancing makes the unweighted terms fail
                 it.
  segments.csv   one row per segment: the recovered Te and the per-segment
                 losses at convergence. This is the window-free column of
                 Table 4 and the --windowfree input to fig_windowsensitivity.
  run.json       every hyperparameter, the seed, the library versions and the
                 stopping state, so the run can be reproduced exactly.

SEGMENTS
  Profiles are grouped into along-strike segments of SEGMENT_KM by the order of
  their profile id, which is the order in which trace_axes.sh walks the margin.
  One elastic thickness is trained per segment; the trunk is shared by all of
  them. Sharing is the regulariser: a single profile constrains a fourth-order
  equation weakly, whereas a trunk common to several hundred profiles must
  represent a family of shapes the governing equation admits. Segment identity
  enters through a learned embedding concatenated with the scaled coordinate.

SCALES
  Both scales of Equation (10) are constants and not trained: L is the seaward
  reach in metres, and w0 is the axial deflection of the segment. Without them
  the fourth derivative in Equation (11) is ill-conditioned -- x^4 spans
  twenty-two orders of magnitude in SI units on a 550 km profile, and the
  residual is then dominated by floating-point noise rather than by the physics.

WHAT IS NOT DONE HERE
  The ensemble of Section 4.7 is one run per seed: pass --seed and combine the
  segments.csv files afterwards. The derived quantities M, V and kappa of
  Equation (14) are evaluated from the trained solution by a separate script,
  since they are diagnostics of the result rather than part of the training.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Physical constants, identical to morphometry.py. Mirror any change there.
E_YOUNG = 70.0e9          # Pa
NU_POISSON = 0.25
RHO_M = 3300.0            # kg m-3
RHO_W = 1030.0            # kg m-3
RHO_S = 2000.0            # kg m-3
G_GRAV = 9.81             # m s-2
D_RHO = RHO_M - RHO_W

# Airy backstripping of the sediment column, as in 06_flexure.py.
SED_FACTOR = (RHO_M - RHO_S) / (RHO_M - RHO_W)

# Along-strike segment length, km. Matches the 148 km partition of Table 2.
SEGMENT_KM = 148.0

# Bounds on the adaptive loss weights, and the gradient norm below which a
# rebalancing step is skipped rather than attempted.
LAM_CAP = 50.0
GRAD_FLOOR = 1.0e-14

# Half-width of the axis search, km, as in morphometry.AXIS_WINDOW.
AXIS_WINDOW_KM = 60.0

# Reference depth is the median over this seaward interval, km, matching the
# --regional-window default of 06_flexure.py so that the deflection this script
# fits is the same quantity the baseline fits.
REGIONAL_KM = (300.0, 550.0)

# Elastic thickness is trained as log10(Te/km) so that it cannot go negative
# and so that a step in the optimiser is a fixed relative change. The bounds
# are those of morphometry.flexural_fit expressed as thicknesses.
TE_INIT_KM = 25.0
TE_MIN_KM, TE_MAX_KM = 2.0, 120.0


# ------------------------------------------------------------------- data --
def load_profile(path, sediment=True):
    """One profile as (x_m seaward-positive, w_m deflection, ok).

    `p` in the stored file is already positive seaward. The deflection is the
    depth referred to the regional level, positive downward, so that it decays
    to zero in the far field -- which is what the boundary term of
    Equation (12) asserts.
    """
    d = pd.read_csv(path).dropna(subset=["p", "depth"])
    p = d["p"].to_numpy(float)
    z = d["depth"].to_numpy(float)
    if sediment and "hs" in d:
        z = z + np.nan_to_num(d["hs"].to_numpy(float)) * SED_FACTOR
    depth = -z                                   # positive down, metres

    near = np.abs(p) <= AXIS_WINDOW_KM
    if near.sum() < 10:
        return None
    x0 = float(p[near][int(np.argmax(depth[near]))])

    far = (p >= x0 + REGIONAL_KM[0]) & (p <= x0 + REGIONAL_KM[1])
    if far.sum() < 20:
        return None
    d_r = float(np.median(depth[far]))

    keep = p >= x0
    x = (p[keep] - x0) * 1.0e3                   # metres from the axis
    w = depth[keep] - d_r                        # metres, positive down
    if x.size < 100 or not np.isfinite(w).all():
        return None
    return x, w


def stack_median(recs, step_km=1.0):
    """One representative curve per segment: the pointwise median of its
    profiles on a common grid.

    Stacking the raw profiles instead puts a floor under the data loss that no
    network can pass, because a single curve w(x) cannot reproduce eight
    profiles that differ from one another. Measured on Tonga that floor is
    5.8e-4, of the same order as the loss a converged fit should reach, so the
    optimiser spends its effort on scatter it cannot remove. The median is used
    rather than the mean because a single profile crossing a seamount would
    otherwise drag the whole segment.
    """
    hi = min(float(r[0].max()) for r in recs)
    g = np.arange(0.0, hi + 1e-9, step_km * 1.0e3)
    M = np.vstack([np.interp(g, r[0], r[1], left=np.nan, right=np.nan)
                   for r in recs])
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(M, axis=0)
        mad = np.nanmedian(np.abs(M - med), axis=0)
    ok = np.isfinite(med)
    return g[ok], med[ok], float(np.nanmedian(mad[ok]))


def build_segments(profdir, trenches, nmax, sediment=True, stack="median"):
    """Group profiles into along-strike segments and stack them.

    Profiles are numbered in the order trace_axes.sh walks the margin, so
    consecutive ids are adjacent on the arc and a fixed number of them spans a
    fixed arc length. The grouping is therefore by id, not by re-measuring
    distance along the axis.
    """
    files = sorted(glob.glob(os.path.join(profdir, "*_[0-9]*.csv")))
    by_trench = {}
    for f in files:
        t = os.path.basename(f).rsplit("_", 1)[0]
        by_trench.setdefault(t, []).append(f)
    if trenches:
        want = set(trenches)
        unknown = sorted(want - set(by_trench))
        if unknown:
            raise SystemExit(f"no profiles for {', '.join(unknown)} in "
                             f"{profdir}; available: "
                             f"{', '.join(sorted(by_trench))}")
        by_trench = {t: v for t, v in by_trench.items() if t in want}
    if not by_trench:
        raise SystemExit(f"no profiles found in {profdir}")
    n_files = sum(len(v) for v in by_trench.values())

    segs, dropped = [], 0
    for t in sorted(by_trench):
        fl = sorted(by_trench[t])
        # Profiles are cast every 5 km along the axis by extract_profiles.sh,
        # so SEGMENT_KM of arc is SEGMENT_KM/5 profiles.
        per = max(1, int(round(SEGMENT_KM / 5.0)))
        for k in range(0, len(fl), per):
            chunk = fl[k:k + per]
            recs = []
            for f in chunk:
                r = load_profile(f, sediment)
                if r is None:
                    dropped += 1
                    continue
                recs.append(r)
            if not recs:
                continue
            if nmax and len(recs) > nmax:
                idx = np.linspace(0, len(recs) - 1, nmax).astype(int)
                recs = [recs[i] for i in idx]
            if stack == "median":
                x, w, mad = stack_median(recs)
            else:
                x = np.concatenate([r[0] for r in recs])
                w = np.concatenate([r[1] for r in recs])
                mad = float("nan")
            if x.size < 50:
                continue
            segs.append({"name": f"{t}_{k // per + 1:02d}", "trench": t,
                         "x": x, "w": w, "n_prof": len(recs), "mad_m": mad})
    print(f"{len(segs)} segments from {n_files} profiles across "
          f"{len(by_trench)} margins ({dropped} profiles rejected)")
    return segs


# ------------------------------------------------------------------ model --
class Trunk(nn.Module):
    """Shared trunk with Fourier feature encoding and a per-segment embedding.

    Input is the scaled coordinate x/L, output the scaled deflection. tanh
    activations are used because the loss differentiates the output four times:
    a piecewise linear activation has a second derivative that vanishes almost
    everywhere and gives an identically zero physics residual, which trains to
    a low loss without ever enforcing the governing equation.

    The coordinate is lifted into random Fourier features before the first
    layer. A plain coordinate input makes a tanh network spectrally biased
    towards the low frequencies, and the bending profile is not low frequency
    near the axis: the trench wall and the outer rise occupy the first fifth of
    a five-hundred-kilometre domain. Measured on a Tonga segment under a
    data-only fit of fifteen hundred iterations, the encoding reduces the
    residual from 7.4e-5 to 8.1e-6, that is from 40 m to 13 m. The frequency
    matrix is drawn once and held fixed; sigma sets the bandwidth and is kept
    modest because the fourth derivative of the residual amplifies each
    frequency by its fourth power.

    The coordinate is NOT recentred to [-1, 1]. That transformation, which is
    usually beneficial, is measurably harmful here (1.9e-3 against 7.4e-5 in
    the same test) because it places the sharply curved part of the profile at
    the edge of the input range rather than at its origin.
    """

    def __init__(self, n_seg, width=64, depth=5, emb=4, fourier=16,
                 sigma=2.0):
        super().__init__()
        self.emb = nn.Embedding(n_seg, emb)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.fourier = int(fourier)
        if self.fourier > 0:
            B = torch.randn(1, self.fourier) * float(sigma)
            self.register_buffer("B", B)
            d_in = 2 * self.fourier + emb
        else:
            d_in = 1 + emb
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        layers += [nn.Linear(d_in, 1)]
        self.net = nn.Sequential(*layers)

    def features(self, xs):
        if self.fourier <= 0:
            return xs
        ph = 2.0 * np.pi * (xs @ self.B)
        return torch.cat([torch.sin(ph), torch.cos(ph)], dim=1)

    def forward(self, xs, sid):
        return self.net(torch.cat([self.features(xs), self.emb(sid)], dim=1))


def d4(model, xs, sid):
    """w-hat and its fourth derivative with respect to the SCALED coordinate.

    Four nested grad calls, each with create_graph=True so the next one can
    differentiate through it, and so that the final loss remains differentiable
    with respect to theta and Te.
    """
    xs = xs.requires_grad_(True)
    w = model(xs, sid)
    g = w
    for _ in range(4):
        g = torch.autograd.grad(g, xs, torch.ones_like(g),
                                create_graph=True)[0]
    return w, g


def rigidity(te_km):
    te = te_km * 1.0e3
    return E_YOUNG * te ** 3 / (12.0 * (1.0 - NU_POISSON ** 2))


# ------------------------------------------------------------------- loss --
def losses(model, te_km, batch, L, n_coll, gen, fixed=None):
    """The three terms of Equation (12), each already dimensionless.

    `fixed` supplies a deterministic set of collocation points. Resampling them
    every call is right for Adam, which is stochastic anyway, and wrong for
    L-BFGS: a quasi-Newton method builds a curvature estimate from successive
    gradients, and if the objective changes between evaluations the strong
    Wolfe line search fails on its first trial step and the optimiser exits
    immediately. In the first run of this script L-BFGS stopped after fourteen
    of two hundred allowed iterations for exactly that reason.
    """
    l_d = l_p = l_b = 0.0
    for s in batch:
        sid = s["sid"]
        # data
        w_hat = model(s["xs"], sid.expand(s["xs"].shape[0]))
        l_d = l_d + torch.mean(((w_hat.squeeze(1) - s["ws"])) ** 2)
        # physics, on freshly drawn collocation points
        if fixed is not None:
            xi = fixed
        else:
            xi = torch.rand(n_coll, 1, generator=gen, device=s["xs"].device,
                            dtype=s["xs"].dtype)
        w_c, w4 = d4(model, xi, sid.expand(n_coll))
        D = rigidity(te_km[s["k"]])
        # d4w/dx4 = (w0/L^4) d4(w-hat)/d(x/L)^4, and the leading factor of
        # Equation (11) divides by D_RHO*g*w0, so w0 cancels exactly.
        res = (D / (D_RHO * G_GRAV * L ** 4)) * w4 + w_c
        l_p = l_p + torch.mean(res ** 2)
        # far field: deflection and slope vanish at x = L
        xe = torch.ones(1, 1, device=s["xs"].device, dtype=s["xs"].dtype)
        xe = xe.requires_grad_(True)
        w_e = model(xe, sid.expand(1))
        dw_e = torch.autograd.grad(w_e, xe, torch.ones_like(w_e),
                                   create_graph=True)[0]
        l_b = l_b + (w_e ** 2).squeeze() + (dw_e ** 2).squeeze()
    n = float(len(batch))
    return l_d / n, l_p / n, l_b / n


def grad_norm(loss, params):
    """Mean absolute gradient of one loss term over the trunk parameters."""
    g = torch.autograd.grad(loss, params, retain_graph=True,
                            allow_unused=True)
    tot, cnt = 0.0, 0
    for t in g:
        if t is not None:
            tot += float(t.abs().sum())
            cnt += t.numel()
    return tot / max(cnt, 1)


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--trenches", nargs="*", default=None)
    ap.add_argument("--max-profiles-per-segment", type=int, default=8)
    ap.add_argument("--no-sediment", action="store_true")
    ap.add_argument("--stack", choices=("median", "raw"), default="median",
                    help="one median curve per segment, or every profile "
                         "stacked; raw puts an irreducible floor under the "
                         "data loss")
    ap.add_argument("--fourier", type=int, default=16,
                    help="random Fourier features on the coordinate; 0 for a "
                         "plain coordinate input")
    ap.add_argument("--fourier-sigma", type=float, default=2.0,
                    help="bandwidth of the Fourier features")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--embedding", type=int, default=4)
    ap.add_argument("--collocation", type=int, default=256)
    ap.add_argument("--batch-segments", type=int, default=16)
    ap.add_argument("--adam", type=int, default=20000)
    ap.add_argument("--lbfgs", type=int, default=300,
                    help="second-stage iterations. Each one evaluates every "
                         "segment on a fixed collocation set and the line "
                         "search calls that closure several times, so this "
                         "stage costs far more per iteration than Adam; a few "
                         "hundred iterations refine, thousands merely wait")
    ap.add_argument("--lr", type=float, default=3.0e-3)
    ap.add_argument("--lr-te", type=float, default=1.0e-3,
                    help="separate rate for the thickness; a common rate lets "
                         "Te overshoot far above its final value early on")
    ap.add_argument("--lr-decay", type=float, default=0.5,
                    help="geometric decay applied every 5000 Adam iterations")
    ap.add_argument("--rebalance", type=int, default=100,
                    help="gradient-norm weight update interval; 0 disables")
    ap.add_argument("--patience", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1.0e-4)
    ap.add_argument("--track", default=None,
                    help="segment name for the te_km column of history.csv; "
                         "defaults to the first segment")
    ap.add_argument("--checkpoint", type=int, default=500,
                    help="write a resumable checkpoint every N iterations; "
                         "0 disables")
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoint.pt in --outdir, appending "
                         "to the existing history")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    dt = torch.float64                 # the fourth derivative needs the range
    os.makedirs(a.outdir, exist_ok=True)

    segs = build_segments(a.profiles, a.trenches,
                          a.max_profiles_per_segment,
                          sediment=not a.no_sediment, stack=a.stack)
    if not segs:
        raise SystemExit("no segment survived loading")

    # L is common to every segment, as Section 4.4 requires: it is fixed by the
    # extraction geometry, not chosen per margin.
    L = float(max(s["x"].max() for s in segs))
    for k, s in enumerate(segs):
        w0 = float(np.max(np.abs(s["w"])))
        if not np.isfinite(w0) or w0 <= 0:
            w0 = 1.0
        s["w0"] = w0
        s["k"] = k
        s["sid"] = torch.tensor([k], device=dev)
        s["xs"] = torch.tensor(s["x"] / L, device=dev,
                               dtype=dt).reshape(-1, 1)
        s["ws"] = torch.tensor(s["w"] / w0, device=dev, dtype=dt)

    track = 0
    if a.track:
        names = [s["name"] for s in segs]
        if a.track not in names:
            raise SystemExit(f"--track {a.track} not among {len(names)} "
                             f"segments; first few are {names[:5]}")
        track = names.index(a.track)

    model = Trunk(len(segs), a.width, a.depth, a.embedding,
                  a.fourier, a.fourier_sigma).to(dev).to(dt)
    log_te = torch.full((len(segs),), float(np.log10(TE_INIT_KM)),
                        device=dev, dtype=dt, requires_grad=True)
    trunk_params = [p for p in model.parameters()]
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)

    lam = {"d": 1.0, "p": 1.0, "b": 1.0}
    ckpt_path = os.path.join(a.outdir, "checkpoint.pt")
    hist_path = os.path.join(a.outdir, "history.csv")

    # A run of this length must survive a closed lid, a flat battery and an
    # interrupted terminal. Without a checkpoint the only artefact before the
    # final write is the history file, which records the losses but not the
    # weights, so an interrupted run cannot be continued and must be repeated
    # from the beginning.
    start_it = 0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if ck["n_seg"] != len(segs):
            raise SystemExit(f"checkpoint holds {ck['n_seg']} segments, the "
                             f"present run builds {len(segs)}; the profile "
                             "selection must match to resume")
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            log_te.copy_(ck["log_te"].to(dev))
        lam = ck["lam"]
        start_it = int(ck["iteration"])
        print(f"resumed from {ckpt_path} at iteration {start_it}")
    elif a.resume:
        print(f"note: --resume given but {ckpt_path} does not exist; "
              "starting from scratch")

    mode = "a" if start_it else "w"
    hist = open(hist_path, mode, newline="")
    log = csv.writer(hist)
    if not start_it:
        log.writerow(["iteration", "loss_total", "loss_data", "loss_phys",
                      "loss_bc", "te_km"])

    def save_ckpt(iteration, stage):
        if not a.checkpoint:
            return
        torch.save({"model": model.state_dict(), "log_te": log_te.detach(),
                    "lam": lam, "iteration": iteration, "stage": stage,
                    "n_seg": len(segs), "seed": a.seed},
                   ckpt_path)
        hist.flush()
        os.fsync(hist.fileno())

    lam_log_fh = open(os.path.join(a.outdir, "lambda_log.csv"), "a" if start_it
                      else "w", newline="")
    lam_log = csv.writer(lam_log_fh)
    if not start_it:
        lam_log.writerow(["iteration", "lam_d", "lam_p", "lam_b",
                          "grad_d", "grad_p", "grad_b"])

    def te_of():
        return torch.clamp(10.0 ** log_te, TE_MIN_KM, TE_MAX_KM)

    def step_losses():
        idx = np.random.choice(len(segs),
                               size=min(a.batch_segments, len(segs)),
                               replace=False)
        batch = [segs[i] for i in idx]
        return losses(model, te_of(), batch, L, a.collocation, gen)

    best, best_it, it = np.inf, 0, 0
    t0 = time.time()
    opt = torch.optim.Adam([{"params": trunk_params, "lr": a.lr},
                            {"params": [log_te], "lr": a.lr_te}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000,
                                            gamma=a.lr_decay)
    stop_reason = "adam budget exhausted"

    for it in range(start_it + 1, a.adam + 1):
        opt.zero_grad(set_to_none=True)
        l_d, l_p, l_b = step_losses()
        if a.rebalance and it % a.rebalance == 0:
            # Wang et al. (2021): scale each term so that its gradient norm on
            # the trunk matches that of the composite. Without it the data term
            # dominates early and the physics residual is never enforced.
            gd = grad_norm(l_d, trunk_params)
            gp = grad_norm(l_p, trunk_params)
            gb = grad_norm(l_b, trunk_params)
            gmax = max(gd, gp, gb)
            # The ratio gmax/g is unbounded, and the boundary term is the one
            # that makes it diverge: it is evaluated at a single coordinate,
            # so its gradient on the trunk is smaller than the other two by
            # orders of magnitude and falls further as the far-field condition
            # is satisfied. Left unclamped, lambda_b reached 1.5e3 in a run of
            # this study, the weighted boundary term then dominated the
            # composite in bursts, and the optimiser spent four thousand
            # iterations without improving on its own best loss. Two guards:
            # a rebalancing step is skipped when any gradient norm is too
            # small to form a meaningful ratio, and every weight is confined
            # to LAM_CAP either side of unity.
            if min(gd, gp, gb) > GRAD_FLOOR:
                beta = 0.9
                for k, g in (("d", gd), ("p", gp), ("b", gb)):
                    lam[k] = beta * lam[k] + (1 - beta) * (gmax / g)
                    lam[k] = float(np.clip(lam[k], 1.0 / LAM_CAP, LAM_CAP))
                if lam_log is not None:
                    lam_log.writerow([it, lam["d"], lam["p"], lam["b"],
                                      gd, gp, gb])
        loss = lam["d"] * l_d + lam["p"] * l_p + lam["b"] * l_b
        loss.backward()
        opt.step()
        sched.step()

        tot = float(loss.detach())
        log.writerow([it, tot, float((lam["d"] * l_d).detach()),
                      float((lam["p"] * l_p).detach()),
                      float((lam["b"] * l_b).detach()),
                      float(te_of()[track].detach())])
        if tot < best * (1.0 - a.tol):
            best, best_it = tot, it
        if it - best_it > a.patience:
            stop_reason = f"no improvement of {a.tol:g} over {a.patience} iters"
            break
        if a.checkpoint and it % a.checkpoint == 0:
            save_ckpt(it, "adam")
        if it % 500 == 0:
            print(f"  adam {it:6d}  L={tot:.3e}  d={float(l_d.detach()):.2e} "
                  f"p={float(l_p.detach()):.2e} b={float(l_b.detach()):.2e}  "
                  f"Te[{segs[track]['name']}]={float(te_of()[track].detach()):.1f} km "
                  f"({time.time() - t0:.0f}s)")

    switch = it
    save_ckpt(switch, "switch")
    print(f"stage switch at iteration {switch}: {stop_reason}")

    if a.lbfgs > 0:
        # Deterministic from here on: a fixed collocation set on a regular
        # grid, every segment in every evaluation, and the weights frozen at
        # the values Adam left. L-BFGS then sees a fixed objective, which is
        # what its line search requires.
        coll = torch.linspace(0.0, 1.0, a.collocation, device=dev,
                              dtype=dt).reshape(-1, 1)
        lb = torch.optim.LBFGS(trunk_params + [log_te], max_iter=a.lbfgs,
                               history_size=50, tolerance_grad=1e-12,
                               tolerance_change=1e-14,
                               line_search_fn="strong_wolfe")
        counter = {"i": switch}

        def closure():
            lb.zero_grad(set_to_none=True)
            l_d, l_p, l_b = losses(model, te_of(), segs, L, a.collocation,
                                   gen, fixed=coll)
            loss = lam["d"] * l_d + lam["p"] * l_p + lam["b"] * l_b
            loss.backward()
            counter["i"] += 1
            log.writerow([counter["i"], float(loss.detach()),
                          float((lam["d"] * l_d).detach()),
                          float((lam["p"] * l_p).detach()),
                          float((lam["b"] * l_b).detach()),
                          float(te_of()[track].detach())])
            return loss

        lb.step(closure)
        save_ckpt(counter["i"], "lbfgs")
        print(f"L-BFGS finished at iteration {counter['i']} "
              f"({time.time() - t0:.0f}s)")
    hist.close()
    lam_log_fh.close()

    # ---- per-segment output -------------------------------------------
    te = te_of().detach().cpu().numpy()
    rows = []
    for s in segs:
        l_d, l_p, l_b = losses(model, te_of(), [s], L, a.collocation, gen)
        rows.append({"segment": s["name"], "trench": s["trench"],
                     "n_profiles": s["n_prof"], "n_samples": int(s["x"].size),
                     "w0_m": s["w0"], "te_km": float(te[s["k"]]),
                     "loss_data": float(l_d), "loss_phys": float(l_p),
                     "loss_bc": float(l_b),
                     # the same two quantities in metres, so that the fit can
                     # be judged against the scatter it is fitting through
                     "rms_fit_m": float(l_d) ** 0.5 * s["w0"],
                     "mad_profiles_m": s.get("mad_m", float("nan"))})
    pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "segments.csv"),
                              index=False)

    meta = {"seed": a.seed, "device": str(dev), "L_m": L,
            "segments": len(segs), "stage_switch": switch,
            "stop_reason": stop_reason, "lambda": lam,
            "args": vars(a), "elapsed_s": round(time.time() - t0, 1),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}
    with open(os.path.join(a.outdir, "run.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nwrote {a.outdir}/history.csv, segments.csv, run.json")
    print(f"Te over {len(segs)} segments: median {np.median(te):.1f} km, "
          f"range {te.min():.1f}-{te.max():.1f} km")
    print(f"\nnext:\n  python3 fig_convergence.py --history {a.outdir}"
          f"/history.csv --stage-switch {switch} "
          f"--segment {segs[track]['name']} --outdir figures")


if __name__ == "__main__":
    main()
