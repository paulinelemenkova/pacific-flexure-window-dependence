#!/usr/bin/env bash
# run_ablation.sh -- the sweep behind fig_ablation.
#
# Repeats the training of train_pinn.py once per configuration, varying one
# setting at a time with the others at the retained values of the run r02
# configuration, and leaves one run directory per point for
# collect_ablation.py to reduce into ablation.csv.
#
#   bash run_ablation.sh profiles_csv
#   DRY=1 bash run_ablation.sh profiles_csv     # print the commands only
#
# READ THIS BEFORE STARTING IT
# ----------------------------
# Run r02 took 13809 s -- three hours fifty minutes -- on one CPU for 4502
# iterations over 71 segments. The default grids below are 5 + 5 + 5 = 15 runs
# for the three axes that need no code change, so about 58 hours end to end,
# and 20 runs if the physics-weight axis is added. That is the real cost of
# this figure and it is stated here rather than discovered on the second day.
#
# Three ways to cut it, in order of how little they cost you scientifically:
#   * --device cuda if a GPU is available; r02 ran on CPU.
#   * Run the four axes concurrently in four shells. They are independent.
#   * Reduce ADAM below. A shorter budget changes the answer, so if you do
#     this, use the SAME budget for every point INCLUDING the retained one,
#     and re-run the retained configuration under that budget rather than
#     reusing r02 -- otherwise the vertical line marks a point measured on a
#     different scale from the curve it sits on.
#
# WHAT EACH AXIS IS
# -----------------
#   hidden_units   --width                      panel (b), direct
#   collocation    --collocation                panel (c), direct
#   spacing        --max-profiles-per-segment   panel (d). Profiles are cast
#                  every 5 km and a segment spans SEGMENT_KM = 148 km, so
#                  capping the stack at n profiles samples the segment every
#                  148/n km. n = 2, 4, 8, 15, 30 gives 74, 37, 18.5, 9.9 and
#                  4.9 km. Note this changes how many profiles are stacked,
#                  which is what the panel is about.
#   physics_weight panel (a). NOT RUNNABLE AS train_pinn.py STANDS: the three
#                  loss weights are adaptive, rebalanced every --rebalance
#                  iterations from the gradient norms, so there is no fixed
#                  lambda_p/lambda_d to sweep. It needs a small patch, given
#                  at the end of this file. Until then the panel stays empty
#                  and fig_ablation prints "no rows for physics_weight".
set -eu

PROFILES=${1:?usage: bash run_ablation.sh <profiles directory>}
OUTROOT=${OUTROOT:-runs}
ADAM=${ADAM:-20000}
DEVICE=${DEVICE:-cpu}
SEED=${SEED:-0}
DRY=${DRY:-}

WIDTHS=${WIDTHS:-"16 32 64 128 256"}
COLLOC=${COLLOC:-"32 64 128 256 512"}
NPROF=${NPROF:-"2 4 8 15 30"}
RATIOS=${RATIOS:-"0.01 0.1 1 10 100"}

run_one() {  # run_one <tag> <extra args...>
    local tag=$1; shift
    local out="${OUTROOT}/abl_${tag}"
    if [ -f "${out}/segments.csv" ]; then
        printf 'skip %s (already complete)\n' "${tag}"
        return
    fi
    printf '\n=== %s ===\n' "${tag}"
    set -- python3 train_pinn.py --profiles "${PROFILES}" --outdir "${out}" \
        --adam "${ADAM}" --device "${DEVICE}" --seed "${SEED}" "$@"
    if [ -n "${DRY}" ]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}

printf 'ablation sweep: %s\n' "${PROFILES}"
printf 'one run took 13809 s in r02; count the runs below before starting\n'

for w in ${WIDTHS};  do run_one "width_${w}"     --width "${w}";       done
for c in ${COLLOC};  do run_one "colloc_${c}"    --collocation "${c}"; done
for n in ${NPROF};   do run_one "nprof_${n}"     --max-profiles-per-segment "${n}"; done

# Only attempted if the patch below has been applied.
if python3 train_pinn.py --help 2>/dev/null | grep -q -- '--lambda-ratio'; then
    for r in ${RATIOS}; do
        run_one "lam_${r}" --lambda-ratio "${r}" --rebalance 0
    done
else
    printf '\nnote: train_pinn.py has no --lambda-ratio, so the physics-weight\n'
    printf 'panel is skipped. See the patch at the end of run_ablation.sh.\n' >&2
fi

printf '\ncollect with:\n'
printf '  python3 collect_ablation.py --runs %s/abl_* %s/r02 \\\n' \
       "${OUTROOT}" "${OUTROOT}"
printf '      --retained %s/r02 --out ablation.csv\n' "${OUTROOT}"
printf '  python3 fig_ablation.py --table ablation.csv --outdir figures\n'

# ---------------------------------------------------------------------------
# PATCH FOR THE PHYSICS-WEIGHT PANEL
# ---------------------------------------------------------------------------
# In train_pinn.py, beside the other arguments:
#
#     ap.add_argument("--lambda-ratio", type=float, default=None,
#                     help="fix lambda_p/lambda_d instead of adapting the "
#                          "weights; use with --rebalance 0 to sweep the "
#                          "physics weight")
#
# and where the weights are initialised, currently
#
#     lam = {"d": 1.0, "p": 1.0, "b": 1.0}
#
# write
#
#     lam = {"d": 1.0, "p": 1.0, "b": 1.0}
#     if a.lambda_ratio is not None:
#         lam["p"] = float(a.lambda_ratio)
#         if a.rebalance:
#             raise SystemExit("--lambda-ratio requires --rebalance 0; "
#                              "an adaptive weight is not a swept one")
#
# Nothing else changes: the rebalancing block is already guarded by
# `if a.rebalance and ...`, so --rebalance 0 leaves the weights at their
# initial values for the whole run. The ratio then reaches run.json through
# the existing args dump, which is where collect_ablation.py reads it.
#
# One caveat worth putting in the caption. Fixing the weights removes the
# adaptive scheme that the retained configuration uses, so the physics-weight
# panel is not a sweep through the retained point but a sweep of a DIFFERENT
# scheme that passes near it. Say so, or the vertical line in panel (a) claims
# more than it should.
