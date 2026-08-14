#!/usr/bin/env bash
# fig_studyarea -- the twenty circum-Pacific trenches of this study.
#
# Shaded bathymetric relief, the automatically traced trench axes, the plate
# boundaries, seafloor-age contours and the along-strike segmentation, with a
# whole-world locator inset. This is the reader's first orientation, so it
# carries the geography and nothing else: no recovered quantity is shown, and
# no colour on the map encodes a result.
#
# WHAT CHANGED, AND WHY
# ---------------------
# 1. AXES COME FROM ONE MULTISEGMENT FILE, NOT A DIRECTORY OF LOOSE FILES.
#    The previous version looped over AXES/*.txt and plotted whatever it found,
#    with no check on how many files there were or where they lay. It drew
#    three lines instead of twenty, and one of the three ran down the middle of
#    the Pacific where no trench exists. trench_axes_full.gmt carries all
#    twenty axes as named records, so the count and the identity of every line
#    are fixed by the input rather than by the contents of a directory.
# 2. EVERY AXIS IS CHECKED AGAINST THE FRAME AND AGAINST THE EXPECTED LIST.
#    A trench missing from the file, or an axis leaving the map, is named in
#    the run report instead of passing silently.
# 3. THE WESTERN LIMIT IS 118E, NOT 150E. Manila reaches 119.16E, so the 120E
#    that would seem the round number clips its western tip; 118E leaves a
#    margin. The eastern limit is 292E for the same reason at Peru-Chile, which
#    reaches 288.75E.
# 4. THE LOCATOR BOX IS PLOTTED IN 0-360, NOT SHIFTED INTO -180/180. The old
#    version subtracted 360 from both limits and plotted on -Rd, which put the
#    western limit at -242, outside that domain, and GMT closed the polygon the
#    long way round the globe; the box came out as a lens spanning half the
#    planet. The inset now uses -Rg, needs no shift, and the box edges are
#    densified so they follow parallels and meridians instead of great circles.
#
# USAGE
#   ./fig_studyarea.sh
#   AXES_GMT=trench_axes_full.gmt PB=PB2002_boundaries.gmt ./fig_studyarea.sh
#   LABELS=trench_labels.txt ./fig_studyarea.sh      # after editing the auto file
#
# INPUTS (environment variables; each optional layer is drawn only if its input
# is present, so the script degrades to a plain basemap rather than failing)
#   AXES_GMT  multisegment file of traced axes, one record per trench, headers
#             carrying -L"Name". Default trench_axes_full.gmt.
#   PB        plate boundaries as a GMT multisegment file (e.g. PB2002).
#   CONTEXT   multisegment file, same format as AXES_GMT, of trenches shown for
#             geographic context but NOT analysed -- at present the relict
#             Vityaz Trench, whose 164 km of traced axis carries no defined
#             seafloor age, so the reference depth of Equation (2) cannot be
#             formed. Drawn grey and dashed with an open marker and a grey
#             italic label, so it cannot be read as one of the twenty, and
#             excluded from the count and from the completeness check.
#   LABELS    "lon lat justification name" per trench. If unset, the script
#             writes trench_labels_auto.txt from the axis midpoints and uses
#             it; edit that file and pass it back as LABELS to nudge crowded
#             labels apart.
#   SEG_KM    along-strike segment length for the ticks. Default 148, matching
#             the segmentation of the manuscript.
#   OCEAN     ocean name written across the middle of the map. Default
#             "Pacific Ocean"; set empty to suppress it.
#   OCEAN_FONT  its font. Default 12p,Helvetica-Bold,ivory. Ivory on the deep
#             ocean needs no backing box, and a box would break the relief.
#   OCEAN_POS   "lon lat" for the name. Default is the centre of REG, derived
#             rather than fixed so the name follows the frame if REG changes.
#   GEBCO     the GEBCO_2026 grid the paper actually cites, as a local netCDF.
#             PREFERRED. Table 1 of the manuscript declares GEBCO_2026, but
#             GMT's @earth_relief is derived from an EARLIER GEBCO release, so
#             falling back to it means the figure is drawn from different data
#             than the paper claims. Set this and the discrepancy disappears.
#   RES       target cell size for the background. Default 2m, which is about
#             one screen pixel at this map width and 600 dpi. GEBCO_2026 is 15
#             arc-seconds, roughly eight times finer than the page can show and
#             1.2 billion nodes over this region, so it is reduced before
#             plotting rather than handed to grdimage whole.
#   FASTPREVIEW  set to 1 to reduce with grdsample (fast, aliases) instead of
#             grdfilter (slow, anti-aliased). Previews only.
#   RELIEF    fallback bathymetric grid when GEBCO is unset. Default
#             @earth_relief_06m.
#   BACKDROP  last-resort georeferenced RGB relief IMAGE, used only when no
#             elevation grid can be opened at all -- offline, say, or with
#             GMT's remote server unreachable. An image carries no z values, so
#             there is no hinged palette, no illumination and no elevation
#             colour bar; the run says so and the output is a preview, not the
#             publication figure. The image must already run 0-360 if the
#             region crosses the dateline, because GMT wraps grids but not
#             images.
#   AGE       seafloor-age grid for the contours. Default @earth_age_06m, which
#             is GMT's copy of the Seton et al. (2020) EarthByte grid that
#             Table 1 cites. Point it at the local netCDF instead if you want
#             the figure to read the same file as the rest of the analysis.
#   AGE_INT   contour interval, Myr. Default 20.
#   AGE_ANN   annotation interval, Myr. Default 40.
#   REG       region. Default 118/292/-56/62.
#   CPT       master CPT for the relief. Default cmocean/topo.
#   OUT       output basename. Default fig_studyarea.
set -eu

AXES_GMT=${AXES_GMT:-trench_axes_full.gmt}
PB=${PB:-}
CONTEXT=${CONTEXT:-vityaz_context.gmt}
LABELS=${LABELS:-}
SEG_KM=${SEG_KM:-148}
OCEAN=${OCEAN-Pacific Ocean}
OCEAN_FONT=${OCEAN_FONT:-12p,Helvetica-Bold,ivory}
OCEAN_POS=${OCEAN_POS:-}
# Locator, bottom-left. At the original 4.4 cm the panel reached 4.70 cm across
# the map and lay over the Puysegur axis, which starts at 4.42 cm; because the
# panel is semi-transparent the axis showed through faintly and the collision
# was easy to miss. 3.9 cm leaves 0.22 cm of clearance. The overlap check at
# the foot of this script tests the result rather than trusting the arithmetic.
INSET_JUST=${INSET_JUST:-BL}
INSET_W=${INSET_W:-3.9c}
INSET_H=${INSET_H:-2.13c}
INSET_OFF=${INSET_OFF:-0.30c/0.30c}
# Globe width derived from the panel, so that shrinking the panel cannot leave
# the globe overflowing it.
INSET_GLOBE=${INSET_GLOBE:-$(awk -v w="${INSET_W%c}" 'BEGIN{printf "%.2fc", w*0.95}')}
FRAME=${FRAME:-WESN}
GEBCO=${GEBCO:-}
RES=${RES:-2m}
FASTPREVIEW=${FASTPREVIEW:-0}
RELIEF=${RELIEF:-@earth_relief_06m}
BACKDROP=${BACKDROP:-}
AGE=${AGE:-@earth_age_06m}
AGE_INT=${AGE_INT:-20}
AGE_ANN=${AGE_ANN:-40}
REG=${REG:-118/292/-56/62}
CPT=${CPT:-cmocean/topo}
OUT=${OUT:-fig_studyarea}

W=17c
PROJ=Q${W}
TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

IFS=/ read -r RW RE RS RN <<EOF
${REG}
EOF

# The twenty trenches the paper reports. Names are matched case-insensitively,
# with spaces, hyphens and underscores treated alike, so that "New Britain",
# "new_britain" and "new-britain" all count as present.
EXPECTED="Aleutian Kuril-Kamchatka Japan Izu-Bonin Mariana Yap Palau Ryukyu \
Manila Philippine New_Britain San_Cristobal Vanuatu Tonga Kermadec Hikurangi \
Puysegur Cascadia Middle_America Peru-Chile"

norm() { printf '%s' "$1" | tr 'A-Z' 'a-z' | tr ' -' '__'; }

gmt set FONT_ANNOT_PRIMARY 8p,Helvetica FONT_LABEL 9p,Helvetica \
        FONT_TITLE 10p,Helvetica-Bold MAP_FRAME_PEN 0.7p \
        MAP_FRAME_TYPE plain MAP_TICK_LENGTH_PRIMARY 2.5p \
        MAP_GRID_PEN_PRIMARY 0.25p,white@25 MAP_ANNOT_OFFSET 2.5p \
        PS_MEDIA a2 GMT_GRAPHICS_DPU 600i

# The relief and age grids are the only layers that need a grid server. If
# either is unreachable -- no network, or GMT's remote data deactivated -- the
# script continues on a plain land and sea base instead of aborting, and says
# so. That render verifies the geometry (axes, segmentation, frame, locator)
# but is NOT the publication figure, which needs the relief.
HAVE_RELIEF=0
HAVE_AGE=1

# Background, in order of preference:
#   1. the GEBCO_2026 grid the paper cites, subset and reduced here;
#   2. GMT's remote earth_relief, WITH A WARNING that it is a different GEBCO
#      release from the one Table 1 declares;
#   3. no grid at all, in which case a plain land and sea base is drawn.
if [ -n "${GEBCO}" ]; then
    if [ ! -f "${GEBCO}" ]; then
        printf 'ERROR: GEBCO=%s does not exist.\n' "${GEBCO}" >&2
    elif gmt grdcut "${GEBCO}" -R"${REG}" -Gcut.nc 2>/dev/null; then
        # Reduce to mapping resolution. grdfilter is a true low-pass, so the
        # 15 arc-second seabed fabric is averaged out instead of being sampled
        # at every Nth node, which is what makes a decimated GEBCO look noisy
        # and moire-patterned at page scale. -x uses every core; it is the slow
        # step, hence FASTPREVIEW.
        if [ "${FASTPREVIEW}" -eq 1 ]; then
            gmt grdsample cut.nc -I"${RES}" -Grelief.nc
            printf 'NOTE: FASTPREVIEW reduction (aliased). Do not publish.\n' >&2
        else
            FW=$(gmt math -Q "${RES%[a-z]}" 2 MUL =)
            gmt grdfilter cut.nc -Fg"${FW}"m -D4 -I"${RES}" -Grelief.nc -x
        fi
        HAVE_RELIEF=1
        printf 'background: %s, subset to %s and reduced to %s\n' \
               "${GEBCO}" "${REG}" "${RES}"
    else
        printf 'ERROR: could not cut %s to %s.\n' "${GEBCO}" "${REG}" >&2
        printf '  The region crosses the dateline and GEBCO ships on -180/180.\n' >&2
        printf '  Shift the grid to 0-360 once, in place, then re-run:\n' >&2
        printf '      gmt grdedit %s -L+p\n' "${GEBCO}" >&2
    fi
fi

if [ "${HAVE_RELIEF}" -eq 0 ] && [ -z "${GEBCO}" ]; then
    if gmt grdcut "${RELIEF}" -R"${REG}" -Grelief.nc 2>/dev/null; then
        HAVE_RELIEF=1
        printf 'WARNING: drawing %s, which is derived from an EARLIER GEBCO\n' "${RELIEF}" >&2
        printf '  release than the GEBCO_2026 that Table 1 of the manuscript\n' >&2
        printf '  declares. Set GEBCO=<GEBCO_2026 netCDF> so that the figure and\n' >&2
        printf '  the data table describe the same grid.\n' >&2
    fi
fi

HAVE_BACKDROP=0
if [ "${HAVE_RELIEF}" -eq 1 ]; then
    gmt grdgradient relief.nc -A315 -Ne0.5 -Gshade.nc
elif [ -n "${BACKDROP}" ] && [ -f "${BACKDROP}" ]; then
    HAVE_BACKDROP=1
    printf 'WARNING: no elevation grid; using the image %s as a backdrop.\n' "${BACKDROP}" >&2
    printf '  It carries no z values, so illumination and the elevation colour\n' >&2
    printf '  bar are omitted. Preview only.\n' >&2
else
    printf 'WARNING: no bathymetric grid available; drawing a plain base.\n' >&2
fi
gmt grdcut "${AGE}" -R"${REG}" -Gage.nc 2>/dev/null || HAVE_AGE=0
if [ "${HAVE_AGE}" -eq 0 ]; then
    printf 'WARNING: age grid %s unavailable, so the age contours are omitted.\n' "${AGE}" >&2
    printf '  The caption states that the map carries contours of seafloor age,\n' >&2
    printf '  so a run without them contradicts its own caption and must not be\n' >&2
    printf '  published. Supply AGE as a local netCDF, or restore access to the\n' >&2
    printf '  GMT data server, and re-run.\n' >&2
fi

# Relief palette. cmocean/topo is perceptually uniform -- lightness rises
# monotonically within the ocean arm and again within the land arm -- so depth
# reads correctly off the colour bar and the map survives greyscale printing.
# It carries a HARD HINGE at sea level, so a -T range spanning zero stretches
# the marine and terrestrial halves independently and the coastline always
# falls exactly on the colour break. No z-increment is given: resampling a
# hinged master at fixed steps aliases its colour changes.
#   Alternatives, same hinge behaviour:  SCM/oleron  SCM/bukavu  gmt/geo
[ "${HAVE_RELIEF}" -eq 1 ] && gmt makecpt -C"${CPT}" -T-8000/2000 -Z > relief.cpt

# -------------------------------------------------------------------------
# Split the multisegment axis file into one file per trench, wrapping
# longitude to 0-360 on the way. The wrap must happen here, before sampling
# and before plotting: Tonga is stored at -175 to -172 and would otherwise be
# drawn on the far side of the map from Kermadec, which adjoins it.
# -------------------------------------------------------------------------
found=""
if [ -s "${AXES_GMT}" ]; then
    awk -v dir="${TMP}" '
        /^#/ { next }
        /^>/ {
            name = "unnamed"
            if (match($0, /-L"[^"]+"/))
                name = substr($0, RSTART + 3, RLENGTH - 4)
            gsub(/[ \/]/, "_", name)
            file = dir "/" name ".xy"
            print name >> (dir "/_names.txt")
            next
        }
        NF >= 2 && file != "" {
            lon = $1 + 0; if (lon < 0) lon += 360
            print lon, $2 > file
        }
    ' "${AXES_GMT}"
    [ -s "${TMP}/_names.txt" ] && found=$(cat "${TMP}/_names.txt")
else
    printf 'WARNING: %s not found; no trench axes will be drawn\n' "${AXES_GMT}" >&2
fi

: > "${TMP}/_report.txt"
: > "${TMP}/_labels.txt"
n_drawn=0

gmt begin "${OUT}" pdf,png
  if [ "${HAVE_RELIEF}" -eq 1 ]; then
      gmt grdimage relief.nc -Crelief.cpt -Ishade.nc -R"${REG}" -J${PROJ} -t18
  elif [ "${HAVE_BACKDROP}" -eq 1 ]; then
      gmt grdimage "${BACKDROP}" -R"${REG}" -J${PROJ} -t18
  else
      gmt basemap -R"${REG}" -J${PROJ} -B0
      gmt coast -Slightsteelblue1
  fi

  # seafloor age, as thin contours: the covariate the Discussion leans on
  if [ "${HAVE_AGE}" -eq 1 ]; then
      gmt grdcontour age.nc -C"${AGE_INT}" \
          -A"${AGE_ANN}"+f6p,Helvetica,gray25 -Wcthinnest,gray55 \
          -Wathin,gray35 -S8 -T+d0.15c/0.05c
      printf 'age contours drawn every %s Myr, annotated every %s\n' \
             "${AGE_INT}" "${AGE_ANN}"
  fi

  if [ "${HAVE_BACKDROP}" -eq 1 ]; then
      # the backdrop already renders land; fill it and the image disappears
      gmt coast -Wthinnest,gray30 -Dl -A5000 -N1/0.25p,gray45
  else
      gmt coast -Ggray78 -Wthinnest,gray45 -Dl -A5000 -N1/0.25p,gray55
  fi

  # Ocean name, in the middle of the frame. Placed before the axes so that a
  # trench crossing the middle would overprint the name and not the reverse.
  if [ -n "${OCEAN}" ]; then
      if [ -z "${OCEAN_POS}" ]; then
          OCEAN_POS=$(awk -v w="${RW}" -v e="${RE}" -v s="${RS}" -v n="${RN}" \
                          'BEGIN{printf "%.4f %.4f", (w+e)/2, (s+n)/2}')
      fi
      printf '%s %s\n' "${OCEAN_POS}" "${OCEAN}" |
          gmt text -F+f"${OCEAN_FONT}"+jCM
      printf 'ocean name "%s" at %s\n' "${OCEAN}" "${OCEAN_POS}"
  fi

  if [ -n "${PB}" ] && [ -s "${PB}" ]; then
      gmt plot "${PB}" -W0.7p,black,-
      printf 'plate boundaries drawn\n'
  fi

  # ---- trenches shown for context only: grey, dashed, no segment ticks, and
  #      an open marker so a 164 km stub is visible at this map width. Drawn
  #      before the analysed axes so that red is never overprinted by grey.
  n_context=0
  if [ -n "${CONTEXT}" ] && [ -s "${CONTEXT}" ]; then
      awk '!/^#/' "${CONTEXT}" | gmt plot -W1.1p,gray35,4_2:0 -fg
      awk '
          /^#/ { next }
          /^>/ {
              if (n > 0) { printf "%.5f %.5f ML %s\n", mx, my, nm > "/dev/stderr"; cnt++ }
              nm = "context"
              if (match($0, /-L"[^"]+"/))
                  nm = substr($0, RSTART + 3, RLENGTH - 4)
              n = 0; next
          }
          NF >= 2 { n++; sx += $1; sy += $2; mx = sx / n; my = sy / n }
          END { if (n > 0) printf "%.5f %.5f ML %s\n", mx, my, nm > "/dev/stderr" }
      ' "${CONTEXT}" 2> "${TMP}/_context_lab.txt"
      if [ -s "${TMP}/_context_lab.txt" ]; then
          awk '{print $1, $2}' "${TMP}/_context_lab.txt" |
              gmt plot -Sc0.16c -W0.8p,gray35 -Gwhite
          n_context=$(wc -l < "${TMP}/_context_lab.txt" | tr -d ' ')
      fi
      # the context LABEL is deferred to after the analysed labels: drawn here
      # it was overprinted by the San Cristobal box, which passes through the
      # same latitude band
      printf 'context trenches drawn (not analysed): %s\n' "${n_context}"
  fi

  # ---- the twenty axes, drawn one at a time so that each can be checked
  for name in ${found}; do
      f="${TMP}/${name}.xy"
      [ -s "$f" ] || continue

      # containment: an axis leaving the frame means either the region is too
      # tight or the axis is wrong, and both used to pass unnoticed
      read -r xmin xmax ymin ymax npts <<EOF
$(awk 'NR==1{a=b=$1; c=d=$2}
       {if($1<a)a=$1; if($1>b)b=$1; if($2<c)c=$2; if($2>d)d=$2}
       END{printf "%.2f %.2f %.2f %.2f %d", a, b, c, d, NR}' "$f")
EOF
      flag=""
      if awk -v a="$xmin" -v b="$xmax" -v c="$ymin" -v d="$ymax" \
             -v rw="$RW" -v re="$RE" -v rs="$RS" -v rn="$RN" \
             'BEGIN{exit !(a<rw || b>re || c<rs || d>rn)}'; then
          flag="  *** OUTSIDE FRAME"
      fi

      gmt plot "$f" -W2.0p,firebrick

      # Segmentation: one tick at the start of each along-strike bin. -AR, not
      # -Af: -Af only ADDS intermediate points and, because the axes are
      # already sampled at 5 km, it returned every node unchanged, so the
      # "ticks" were the whole axis redrawn as dots. -AR resamples at
      # equidistant locations and adjusts the spacing to fit the track length
      # exactly, which is how the 148 km bins were defined.
      nseg=0
      if gmt sample1d "$f" -AR -T${SEG_KM}k -fg > "${TMP}/${name}.seg" 2>/dev/null &&
         [ -s "${TMP}/${name}.seg" ]; then
          gmt plot "${TMP}/${name}.seg" -Sc0.055c -Gwhite -W0.3p,black
          nseg=$(wc -l < "${TMP}/${name}.seg" | tr -d ' ')
      fi

      # Label anchor at the axis midpoint. Justification is chosen by which
      # half of the frame the anchor falls in: a label to the RIGHT of an
      # eastern trench runs off the map -- Peru-Chile was clipped to "Pe" --
      # and also lands on the continent, so eastern anchors take MR and the
      # label goes to the left, over the ocean.
      awk -v nm="${name}" -v n="${npts}" -v mid="$(awk -v a="$RW" -v b="$RE" \
              'BEGIN{printf "%.4f", (a+b)/2}')" \
          'NR==int(n/2)+1{gsub(/_/, " ", nm);
                          print $1, $2, ($1 > mid ? "MR" : "ML"), nm}' "$f" \
          >> "${TMP}/_labels.txt"

      printf '%-16s %5s nodes  %3s segments  lon %7s to %7s  lat %6s to %6s%s\n' \
          "${name}" "${npts}" "${nseg}" "${xmin}" "${xmax}" "${ymin}" "${ymax}" \
          "${flag}" >> "${TMP}/_report.txt"
      n_drawn=$((n_drawn + 1))
  done

  # ---- labels: the supplied file if given, otherwise the generated one
  # Labels are drawn in two passes because the offset must follow the
  # justification: ML labels are nudged right, MR labels left. One pass with a
  # single -D would push the MR labels back over their own axis.
  LABFILE=""
  if [ -n "${LABELS}" ] && [ -s "${LABELS}" ]; then
      LABFILE="${LABELS}"
  elif [ -s "${TMP}/_labels.txt" ]; then
      cp "${TMP}/_labels.txt" trench_labels_auto.txt
      LABFILE="trench_labels_auto.txt"
  fi
  if [ -n "${LABFILE}" ]; then
      TXT="-F+f7.5p,Helvetica-Bold,black+j -Gwhite@30 -W0.2p,gray60 -C0.04c/0.04c"
      awk '$3 !~ /R$/' "${LABFILE}" | gmt text ${TXT} -D0.15c/0
      awk '$3 ~ /R$/'  "${LABFILE}" | gmt text ${TXT} -D-0.15c/0
  fi

  # Context labels last of all, and offset upward as well as sideways: the
  # analysed labels are horizontal boxes that reach across several degrees of
  # longitude, so a context label level with one of them is hidden by it.
  if [ -s "${TMP}/_context_lab.txt" ]; then
      # Set exactly as the analysed trenches: same font, same size, black on
      # the same white halo box. NOTE that the label therefore no longer marks
      # this trench as excluded -- the only cues left are the broken grey axis,
      # the open marker and the caption.
      awk '{$3 = "LB"; print}' "${TMP}/_context_lab.txt" |
          gmt text -F+f7.5p,Helvetica-Bold,black+j -Gwhite@30 -W0.2p,gray60 \
                   -C0.04c/0.04c -D0.20c/0.30c
  fi

  # Frame last, so the graticule sits over the relief rather than under it.
  # Thin, semi-transparent white lines read as a reference lattice on the dark
  # ocean without competing with the trench axes; the grid interval matches the
  # annotation interval so no line is unlabelled.
  gmt basemap -Bxa30f10g30 -Bya20f10g20 -B"${FRAME}"

  # Locator. An orthographic globe is the usual choice, but this study area
  # spans 174 degrees of longitude, so on a globe the outline hugs the limb and
  # conveys nothing. A whole-world Mollweide shows where the region sits at a
  # glance. Plotted on -Rg, so the 0-360 limits of REG need no shifting, with
  # the box edges densified at one degree so that they follow parallels and
  # meridians rather than being resampled along great circles.
  gmt inset begin -Dj${INSET_JUST}+w${INSET_W}/${INSET_H}+o${INSET_OFF} \
                  -F+gwhite@20+p0.5p,gray50+c0.06c
    gmt coast -Rg -JW180/${INSET_GLOBE} -Ggray72 -Sgray93 -A20000 -Bg60 -Wfaint,gray55
    awk -v w="${RW}" -v e="${RE}" -v s="${RS}" -v n="${RN}" 'BEGIN{
        for (x = w; x <= e; x += 1) print x, s
        for (y = s; y <= n; y += 1) print e, y
        for (x = e; x >= w; x -= 1) print x, n
        for (y = n; y >= s; y -= 1) print w, y
    }' | gmt plot -Rg -JW180/${INSET_GLOBE} -W1.0p,firebrick -L
  gmt inset end

  if [ "${HAVE_RELIEF}" -eq 1 ]; then
      gmt colorbar -Crelief.cpt -DJBC+w9c/0.28c+h+o0/1.0c \
          -Bxa2000f500+l"Elevation (m)"
  fi
gmt end

if [ "${HAVE_RELIEF}" -eq 0 ] || [ "${HAVE_AGE}" -eq 0 ]; then
    printf '\nGEOMETRY CHECK ONLY: a grid layer was missing, so this render is\n' >&2
    printf 'not the publication figure. Re-run with the grids available.\n' >&2
fi

# -------------------------------------------------------------------------
# Run report. The figure is correct only if every expected trench appears once
# and none leaves the frame, so both are stated rather than assumed.
# -------------------------------------------------------------------------
printf '\n--- axes drawn -------------------------------------------------\n'
[ -s "${TMP}/_report.txt" ] && cat "${TMP}/_report.txt"

missing=""
for want in ${EXPECTED}; do
    hit=0
    for got in ${found}; do
        if [ "$(norm "$want")" = "$(norm "$got")" ]; then hit=1; break; fi
    done
    [ "${hit}" -eq 0 ] && missing="${missing} ${want}"
done

# Does the locator panel sit on top of any axis? This is what went wrong with
# the bottom-left placement: the panel is semi-transparent, so the axis was
# still faintly visible and the defect was easy to miss by eye.
if [ "${n_drawn}" -gt 0 ]; then
    read -r MAPW MAPH <<EOF
$(gmt mapproject -R"${REG}" -J${PROJ} -W 2>/dev/null)
EOF
    if [ -n "${MAPW:-}" ]; then
        covered=$(cat "${TMP}"/*.xy 2>/dev/null |
            gmt mapproject -R"${REG}" -J${PROJ} 2>/dev/null |
            awk -v W="${MAPW}" -v H="${MAPH}" -v just="${INSET_JUST}" \
                -v iw="${INSET_W%c}" -v ih="${INSET_H%c}" -v off="${INSET_OFF}" '
                BEGIN{
                    split(off, o, "/"); ox = o[1] + 0; oy = o[2] + 0
                    x0 = (just ~ /L$/) ? ox : W - ox - iw
                    y0 = (just ~ /^B/) ? oy : H - oy - ih
                    x1 = x0 + iw; y1 = y0 + ih
                }
                $1 >= x0 && $1 <= x1 && $2 >= y0 && $2 <= y1 { n++ }
                END{ print n + 0 }')
        if [ "${covered:-0}" -gt 0 ]; then
            printf 'WARNING: the locator panel covers %s axis nodes.\n' "${covered}" >&2
            printf '  Move it with INSET_JUST (TR, TL, BR, BL) or INSET_OFF.\n' >&2
        fi
    fi
fi

printf '\ntrench axes drawn: %s of 20 (analysed)\n' "${n_drawn}"
[ "${n_context:-0}" -gt 0 ] && \
    printf 'plus %s trench shown for context only, excluded from the study\n' "${n_context}"
if [ -n "${missing}" ]; then
    printf 'MISSING from %s:%s\n' "${AXES_GMT}" "${missing}" >&2
    printf 'The figure is incomplete; do not publish it until every trench is present.\n' >&2
fi
if grep -q 'OUTSIDE FRAME' "${TMP}/_report.txt" 2>/dev/null; then
    printf 'An axis leaves the frame: widen REG, or check that axis.\n' >&2
fi
[ -f trench_labels_auto.txt ] && \
    printf 'wrote trench_labels_auto.txt -- edit it and pass back as LABELS to nudge crowded labels\n'

printf 'wrote %s.pdf and %s.png\n' "${OUT}" "${OUT}"
printf 'background %s, age %s, region %s, cpt %s, segments %s km\n' \
       "${GEBCO:-${RELIEF}}" "${AGE}" "${REG}" "${CPT}" "${SEG_KM}"
printf 'locator box %s/%s/%s/%s plotted on -Rg (no 360 shift)\n' \
       "${RW}" "${RE}" "${RS}" "${RN}"
