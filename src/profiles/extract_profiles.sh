#!/usr/bin/env bash
# Trench-normal profiles from the bathymetric grid, one trench at a time.
# Criteria of Section 4.1: 10 km spacing, 250 km half-length, 1 km sampling.
set -eu

GRID=${1:?bathymetric grid}          # e.g. gebco_pacific.nc
AXIS=${2:?digitised trench axis}     # lon lat, one trench
OUT=${3:?output table}

SPACING=10k                          # between successive profiles
HALFLEN=250k                         # either side of the axis
STEP=1k                              # sampling along each profile
LENGTH=500k                          # 2 x HALFLEN, as grdtrack expects

# 1. resample the axis at a constant along-track interval. The axis arrives
#    already traced and smoothed from trace_axes.sh, so no filtering is needed
#    here; smoothing it twice would shorten the trace at both ends.
gmt sample1d "${AXIS}" -Af -T${SPACING} -fg > axis_nodes.txt

# 3. cast one cross-profile per node, perpendicular to the local strike, and
#    sample the grid along it. Columns: lon lat dist azimuth depth
gmt grdtrack axis_nodes.txt -G"${GRID}" -C${LENGTH}/${STEP}+v -fg > "${OUT}"

# grep -c exits 1 when the count is zero, which under set -e would abort the
# script with no message; count without letting the status escape.
n_nodes=$(grep -cv '^>' axis_nodes.txt || true)
n_prof=$(grep -c '^>' "${OUT}" || true)
printf 'nodes: %s  profiles: %s\n' "${n_nodes}" "${n_prof}"

if [ "${n_prof:-0}" -eq 0 ]; then
    echo "ERROR: no profiles written." >&2
    echo "  Check that the axis longitudes match the grid. The axis spans" >&2
    awk 'NR==1{lo=hi=$1} {if($1<lo)lo=$1; if($1>hi)hi=$1}
         END{printf "    lon %.2f to %.2f\n", lo, hi}' "${AXIS}" >&2
    gmt grdinfo "${GRID}" -C | awk '{printf "  and the grid spans lon %.2f to %.2f\n", $2, $3}' >&2
    echo "  If the axis is on -180..180 and the grid on 0..360, re-run" >&2
    echo "  trace_axes.sh with the current version, which wraps longitudes." >&2
    exit 1
fi
