#!/usr/bin/env bash
#
# trace_axes.sh  (patched)
#
# Trace the axis of every trench as the locus of locally deepest grid nodes,
# writing axes/<trench>.txt for extract_profiles.sh (Listing 1).
#
#   bash trace_axes.sh gebco_pacific.nc
#
# TWO CHANGES FROM THE PREVIOUS VERSION
# -------------------------------------
# 1. SEED CONSTRUCTION. Step 1 used to strip the '>' separators from PB2002
#    and concatenate every matching piece into one point list. Where a pair
#    code owns several disjoint pieces inside the box, that produced a
#    self-crossing path: the Ryukyu axis ran 2389 km with its endpoints 38 km
#    apart, so profiles were extracted twice along the same margin from
#    opposite limbs. prepare_seed.py now clips, drops short pieces, chains by
#    nearest endpoint, refuses to bridge gaps wider than MAX_GAP_KM, and
#    orients the result consistently.
#
# 2. A RETRACING CHECK. Every traced axis is measured for path length against
#    end-to-end distance. A ratio above 1.6 means the axis doubles back, and
#    the run stops rather than writing an axis that silently corrupts the
#    along-strike binning downstream. Arcuate margins legitimately reach 1.3
#    (Mariana), so the threshold sits above them.
#
# Method, matching Section 4.1 of the manuscript:
#   1. the PB2002 subduction segment of the margin, clipped to the trench's
#      bounding box, is used as a SEED locating the trench;
#   2. the seed is resampled at SEED_KM along track;
#   3. at every seed node a short search profile of half-length SEARCH_KM is
#      cast perpendicular to the local strike and the grid sampled along it;
#   4. the deepest sample on that search profile becomes the axis node, so the
#      axis follows the bathymetry rather than the published plate boundary;
#   5. the resulting polyline is smoothed so that the profile azimuth derived
#      from it in Listing 1 is stable.
#
# Outputs
#   axes/<trench>.txt   the traced axis, lon lat
#   axes_check.png      every axis over shaded relief, for visual verification
#   axes_report.txt     nodes, displacement from seed and retracing ratio
set -eu

GRID=${1:?bathymetric grid, e.g. gebco_pacific.nc}
CONF=${CONF:-trenches_full.conf}
SUB=${SUB:-data/pb2002_subduction.txt}
PREP=${PREP:-scripts/prepare_seed.py}

SEED_KM=10                        # spacing of seed nodes along track
SEARCH_KM=${SEARCH_KM:-60}        # half-length of the perpendicular search profile
SEARCH_STEP=0.5                   # sampling interval along the search profile, km
SMOOTH_KM=25                      # median filter width applied to the traced axis
MIN_PIECE_KM=${MIN_PIECE_KM:-40}  # discard boundary pieces shorter than this
MAX_GAP_KM=${MAX_GAP_KM:-250}     # never bridge a wider gap between pieces
MAX_RATIO=${MAX_RATIO:-1.6}       # path / end-to-end above which an axis retraces
GUARD=${GUARD:-scripts/guard_axis.py}
DEPTH_TOL=${DEPTH_TOL:-2000}      # drop an axis node this much shallower than
                                  # the rolling median of its neighbours (m)
GUARD_WIN=${GUARD_WIN:-21}        # rolling-median window, nodes

mkdir -p axes
: > axes_report.txt

while read -r name pair x0 x1 y0 y1; do
    case "${name}" in ''|'#'*) continue ;; esac

    # ---- 1. seed: chained, not concatenated
    if [ "${pair}" = "SEED" ]; then
        cp "seeds/${name}.txt" _seed.txt
        echo "  ${name}: hand-digitised seed"
    else
        if ! python3 "${PREP}" --pairs "${SUB}" --code "${pair}" \
                --box "${x0}" "${x1}" "${y0}" "${y1}" --out _seed.txt \
                --min-piece "${MIN_PIECE_KM}" --max-gap "${MAX_GAP_KM}"; then
            printf '%-18s SKIPPED: no usable seed for pair %s\n' \
                   "${name}" "${pair}" | tee -a axes_report.txt
            continue
        fi
    fi

    n_seed=$(wc -l < _seed.txt | tr -d ' ')
    if [ "${n_seed}" -lt 3 ]; then
        printf '%-18s SKIPPED: seed has %s points (check pair code %s)\n' \
               "${name}" "${n_seed}" "${pair}" | tee -a axes_report.txt
        continue
    fi

    # ---- 2. resample the seed at a constant along-track interval
    gmt sample1d _seed.txt -Af -T${SEED_KM}k -fg > _seed_rs.txt

    # ---- 3. search profiles perpendicular to the seed, sampling the grid
    gmt grdtrack _seed_rs.txt -G"${GRID}" \
        -C$((2 * SEARCH_KM))k/${SEARCH_STEP}k+v -fg > _search.txt

    # ---- 4. the deepest sample on each search profile is the axis node.
    #         The node depth is carried through so step 4b can reject nodes
    #         that fell into a parallel depression rather than the trench.
    awk '
        /^>/ { if (have) print blon, blat, best; have = 0; best = 1e30; next }
        NF >= 5 && $5 != "NaN" {
            if ($5 < best) { best = $5; blon = $1; blat = $2; have = 1 }
        }
        END { if (have) print blon, blat, best }
    ' _search.txt > _axis_depth.txt

    # ---- 4b. depth guard. A trench shoals gradually towards its ends --
    #          Izu-Bonin runs 9848 m to 5265 m -- so a global depth cut-off
    #          deep enough to exclude a back-arc trough would amputate those
    #          terminations. The discriminator is the SHAPE of the along-strike
    #          depth series instead: a node that jumped into a parallel
    #          depression is an isolated excursion against the rolling median
    #          of its neighbours. Tested: zero false rejections on five of six
    #          trenches, and all six nodes of a simulated Okinawa-Trough
    #          capture removed.
    if ! python3 "${GUARD}" --in _axis_depth.txt --out _axis_raw.txt \
            --window "${GUARD_WIN}" --tol "${DEPTH_TOL}" --name "${name}"; then
        printf '%-18s SKIPPED: depth guard failed\n' "${name}" | tee -a axes_report.txt
        continue
    fi
    if [ "$(wc -l < _axis_raw.txt)" -lt 3 ]; then
        printf '%-18s SKIPPED: fewer than 3 nodes survive the depth guard\n' \
               "${name}" | tee -a axes_report.txt
        continue
    fi

    # ---- 5. smooth, so the local strike is stable. The wrap to 0..360 must
    #         happen BEFORE the median filter, or filtering across +/-180
    #         averages 179.9 and -179.9 to about 0.
    gmt mapproject _axis_raw.txt -G+uk -fg |
        awk '{lon = $1; if (lon < 0) lon += 360; print lon, $2, $3}' |
        gmt filter1d -Fm${SMOOTH_KM} -N2 |
        awk '{print $1, $2}' > "axes/${name}.txt"

    n_axis=$(wc -l < "axes/${name}.txt" | tr -d ' ')

    # ---- 6. retracing check: an axis that doubles back must not go downstream
    read -r path span ratio <<EOF
$(awk '
    function rad(d) { return d * 3.14159265358979 / 180 }
    function gc(o1,a1,o2,a2,   dlo,dla,h) {
        dlo = rad(o2-o1); dla = rad(a2-a1)
        h = sin(dla/2)^2 + cos(rad(a1))*cos(rad(a2))*sin(dlo/2)^2
        return 6371.0088 * 2 * atan2(sqrt(h), sqrt(1-h))
    }
    NR == 1 { flo = $1; fla = $2; plo = $1; pla = $2; next }
    { s += gc(plo,pla,$1,$2); plo = $1; pla = $2 }
    END { e = gc(flo,fla,plo,pla)
          printf "%.0f %.0f %.2f", s, e, (e > 0 ? s/e : 999) }
' "axes/${name}.txt")
EOF
    flag=""
    if awk "BEGIN{exit !(${ratio} > ${MAX_RATIO})}"; then
        flag="  *** RETRACES -- axis rejected"
        mv "axes/${name}.txt" "axes/${name}.txt.rejected"
    fi

    shift_km=$(paste _seed_rs.txt _axis_raw.txt 2>/dev/null | awk '
        function rad(d) { return d * 3.14159265358979 / 180 }
        NF >= 4 {
            dlon = rad($3 - $1); dlat = rad($4 - $2)
            a = sin(dlat/2)^2 + cos(rad($2)) * cos(rad($4)) * sin(dlon/2)^2
            s += 6371 * 2 * atan2(sqrt(a), sqrt(1-a)); n++
        }
        END { if (n) printf "%.1f", s/n; else printf "n/a" }')

    printf '%-18s %5s nodes  %6s km path  %6s km span  ratio %s  shift %s km%s\n' \
           "${name}" "${n_axis}" "${path}" "${span}" "${ratio}" "${shift_km}" \
           "${flag}" | tee -a axes_report.txt
done < "${CONF}"

rm -f _seed.txt _seed_rs.txt _search.txt _axis_raw.txt _axis_depth.txt

if ls axes/*.rejected >/dev/null 2>&1; then
    echo
    echo "REJECTED AXES (they retrace; fix the seed before extracting profiles):"
    ls axes/*.rejected
fi

if ! ( gmt begin axes_check png
        gmt grdimage "${GRID}" -R100/300/-58/66 -JQ20c -Cgeo -I+d
        gmt coast -Ggray80 -W0.2p,gray40
        gmt plot "${SUB}" -W0.6p,gray30
        for f in axes/*.txt; do gmt plot "$f" -W1.6p,red; done
        gmt basemap -Bxa20f10 -Bya20f10 -BWSne+t"Traced trench axes"
      gmt end ) 2>/dev/null; then
    echo "note: axes_check.png could not be rendered (Ghostscript)."
fi

echo
echo "wrote $(ls axes/*.txt 2>/dev/null | wc -l | tr -d ' ') axes"
echo "CHECK axes_report.txt and axes_check.png before running extract_profiles.sh"
