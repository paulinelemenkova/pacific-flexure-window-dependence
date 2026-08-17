#!/usr/bin/env bash
# fig_temap -- circum-Pacific map of the recovered effective elastic thickness.
#
# Segment-wise values are drawn as coloured along-strike bands on shaded
# bathymetric relief. The bathymetry is coloured with oleron (Crameri, hinged
# at sea level) and held subordinate to the Te bands by a -t25 transparency;
# pass RELIEFCPT=gray to restore the desaturated grey base.
#
# USAGE
#   SEGMENTS=te_segments.gmt ./fig_temap.sh
#   SEGMENTS=te_segments.gmt CPT=paired ./fig_temap.sh    # old qualitative scale
#   SEGMENTS=te_segments.gmt RELIEFCPT=gray ./fig_temap.sh
#   ./fig_temap.sh                     # basemap only, for checking the layout
#
# INPUTS (environment variables)
#   SEGMENTS  GMT multisegment file, one segment per along-strike bin, each
#             header carrying the recovered thickness as a -Z value:
#
#                 > -Z34.2
#                 182.417  -22.905
#                 182.398  -22.815
#                 ...
#                 > -Z29.7
#                 ...
#
#             psxy then colours each polyline by its own Z through the CPT,
#             which is why the value travels in the header rather than in a
#             separate table: the geometry and the value cannot fall out of
#             step. Build it from the traced axes and the per-segment table
#             with a few lines of awk, or from build_dataframe's output.
#
#   AXES      accepted for backward compatibility but NO LONGER DRAWN. The
#             traced-axis file and the coloured-band file are independent
#             tracings that do not register exactly, so the axis is omitted;
#             segments with no recovered value are drawn as white bands,
#             explained in the map key.
#   RELIEF    bathymetric grid. Default: the local GEBCO 2026 grid on the
#             TOSHIBA volume (/Volumes/TOSHIBA/DATA/GEBCO_2026.nc). Quoted
#             everywhere it is used, so a future path with a space still works;
#             keep the quotes if you edit it. Pass RELIEF=@earth_relief_06m to
#             fall back to the GMT server. CHECK the release named here against
#             the Open Research section before submitting the sheet.
#   RELIEFCPT palette for the bathymetry. Default oleron (Crameri, hard hinge
#             at sea level). RELIEFCPT=gray restores the desaturated base that
#             keeps the elastic thickness the only saturated colour on the sheet.
#   GRDRES    optional resampling step applied to the cut grid, e.g. 01m.
#             Unset by default (the grid is used at its native step). GEBCO
#             is 15 arc-seconds; over this region that is roughly
#             36000 x 29000 nodes, far more than a 17 cm sheet can show, and
#             grdimage will be slow and memory-hungry. At 600 dpi one pixel
#             spans about 2 arc-minutes here, so GRDRES=01m loses nothing
#             visible and cuts the node count by a factor of about eighty.
#   REG       region. Default 116/290/-60/62. NOTE this crosses the date line;
#             GEBCO is stored on -180/180, and grdcut wraps a global grid
#             automatically. If it refuses, run once
#               gmt grdedit "$RELIEF" -L+p -G/tmp/gebco_0360.nc
#             and point RELIEF at the result.
#
#             THE WEST LIMIT WAS 140E AND IS NOW 116E. At 140E the sheet cut
#             off six of the twenty margins that fig_studyarea labels --
#             Manila (~119-121E), Ryukyu (~123-130E), Philippine (~126-128E),
#             Palau (~134E), Yap (~137-138E) and the western end of Izu-Bonin
#             -- so segments that had been inverted were simply absent from
#             the figure, with nothing on the sheet to say so. 116E clears
#             Manila, the westernmost, by about three degrees; fig_studyarea
#             starts at 120E and clips Manila against its own frame, which is
#             why this figure does not just copy that limit.
#
#             The east limit stays at 290 (70W): Peru-Chile reaches about 75W
#             and is already comfortably inside.
#
#             Widening from 150 to 174 degrees of longitude at a fixed 17 cm
#             width shrinks the sheet from about 13.8 cm to about 11.9 cm
#             high, since -JQ keeps degrees square. Nothing else needs moving:
#             the key and the ocean label are placed in map
#             coordinates, not page coordinates.
#   CPT       palette for the thickness classes. Default grad6: a GRADUAL
#             sequential ramp violet -> blue -> green -> yellow -> brown -> red,
#             one evenly spaced hue per class, requested by the coauthor so a
#             reader can infer "thicker" from "warmer". Twelve fixed triplets
#             are pasted onto the twelve geometric boundaries (see TLEVELS),
#             exactly as the old Paired set was, so NO interpolation happens and
#             the RGB values can be quoted in the caption. One hue PER CLASS,
#             not per km: makecpt -C<ramp> -T<boundaries> would spread the ramp
#             linearly over 4-68 km and collapse the five sub-12 km classes --
#             where most segments sit -- into one violet-blue smear, the very
#             jet/rainbow failure this avoids.
#
#             CPT=paired restores the old twelve-colour ColorBrewer qualitative
#             set (non-monotonic: a reader cannot read magnitude from it).
#             Any other value (batlow, roma, ...) falls back to the makecpt
#             path and is resampled over TLEVELS.
#
#             CAVEAT worth a line in the caption: a violet-to-red ramp is a
#             rainbow-family scale -- NOT certified colour-blind safe and NOT
#             greyscale-safe. If Section 4.7 keeps its claim that every scale on
#             the sheet is perceptually uniform, colour-blind readable and
#             greyscale-safe (Crameri 2020), that sentence and this palette
#             conflict: soften the sentence, or use CPT=batlow / CPT=roma.
#   TLEVELS   comma-separated class boundaries for the discrete colour scale,
#             km. Default 4,5,6,8,10,12,16,20,25,32,40,50,68 -- thirteen
#             boundaries, so twelve classes, spaced geometrically rather than
#             at a fixed interval. The recovered thickness is strongly
#             right-skewed (median ~9.4 km, IQR ~8.1-10.7, full range 4-67),
#             so a uniform step wide enough to reach 67 km would put half the
#             segments in one or two classes. Geometric steps put the fine
#             resolution where the segments are and widen towards the tail;
#             the end boundaries sit at the extremes of the distribution, so
#             nothing lands in an overflow triangle and none is drawn.
#   TMIN/TMAX/TINC   legacy LINEAR override, km. Ignored unless all three are
#             set, in which case TLEVELS is not used. Kept so old invocations
#             still run; new work should pass TLEVELS. A GMT master must be
#             named in CPT for this path -- the grad6 sentinel is not one.
#   OUT       output basename. Default fig_temap.
set -eu

SEGMENTS=${SEGMENTS:-}
AXES=${AXES:-}
RELIEF=${RELIEF:-/Volumes/TOSHIBA/DATA/GEBCO_2026.nc}
RELIEFCPT=${RELIEFCPT:-oleron}
GRDRES=${GRDRES:-}
REG=${REG:-116/290/-60/62}
CPT=${CPT:-grad6}
TLEVELS=${TLEVELS:-4,5,6,8,10,12,16,20,25,32,40,50,68}
TMIN=${TMIN:-}; TMAX=${TMAX:-}; TINC=${TINC:-}
OUT=${OUT:-fig_temap}

W=17c
PROJ=Q${W}

# The key, the ocean label and the colour-bar caption are anchored in MAP
# coordinates, so widening REG rescales the sheet under them. -JQ keeps
# degrees square, so the page height follows the aspect ratio and one
# centimetre of page is (lon range / 17) degrees whichever region is used.
# Deriving the caption latitude from that keeps it a fixed distance below the
# frame instead of drifting into the colour bar: at 140/290 the old hard-wired
# -78 sat 2.04 cm below the frame, and at 116/290 the same -78 would sit 1.76
# cm below it, on top of the bar's annotations.
REG_W=${REG%%/*};        REG_REST=${REG#*/}
REG_E=${REG_REST%%/*};   REG_REST=${REG_REST#*/}
REG_S=${REG_REST%%/*};   REG_N=${REG_REST#*/}
LON_MID=$(awk -v a="${REG_W}" -v b="${REG_E}" 'BEGIN{ printf "%.1f", (a+b)/2 }')
CAP_LAT=$(awk -v s="${REG_S}" -v a="${REG_W}" -v b="${REG_E}" \
    'BEGIN{ printf "%.1f", s - 2.05 * (b - a) / 17 }')

gmt set FONT_ANNOT_PRIMARY 8p,Helvetica FONT_LABEL 9p,Helvetica \
        FONT_TITLE 10p,Helvetica-Bold MAP_FRAME_PEN 0.7p \
        MAP_FRAME_TYPE plain MAP_TICK_LENGTH_PRIMARY 2.5p \
        MAP_GRID_PEN_PRIMARY 0.25p,white MAP_ANNOT_OFFSET 2.5p \
        PS_MEDIA a2 GMT_GRAPHICS_DPU 600i

# Fail early and legibly on a bad path rather than inside grdcut. The remote
# @earth_relief_* names are not files, so they skip the check.
case "${RELIEF}" in
  @*) : ;;
  *) if [ ! -f "${RELIEF}" ]; then
         printf 'error: relief grid not found: %s\n' "${RELIEF}" >&2
         printf 'is the TOSHIBA volume mounted?\n' >&2
         exit 1
     fi ;;
esac

# Shaded relief. The hillshade intensity (grdgradient) is unchanged; only the
# colour under it changes from grey to oleron below.
gmt grdcut "${RELIEF}" -R"${REG}" -Grelief.nc
if [ -n "${GRDRES}" ]; then
    printf 'resampling relief to %s\n' "${GRDRES}"
    gmt grdsample relief.nc -I"${GRDRES}" -Grelief_r.nc
    mv relief_r.nc relief.nc
fi
gmt grdinfo relief.nc -C | awk '{printf "relief grid: %d x %d nodes\n", $10, $11}'
gmt grdgradient relief.nc -A315 -Ne0.5 -Gshade.nc

# BATHYMETRY: oleron (Crameri), a Scientific Colour Map with a HARD HINGE at
# sea level, matching fig_residualmap so the two sheets share a base. Because
# the hinge is hard, a -T range that spans zero stretches the ocean half onto
# -8000..0 and the land half onto 0..6000 INDEPENDENTLY, so the colour break
# lands exactly on the coastline. No z-increment and no -Z: resampling a hinged
# master at fixed steps (the old -Cgray -T-8000/2000/250 -Z) aliases the
# transitions at the hinge. The grey coast fill covers the land half, so only
# oleron's ocean half shows here; +6000 is kept so the ramp still reads if that
# fill is ever removed.
#
# TRADE-OFF: the grey base this replaces kept the elastic thickness the only
# saturated colour on the sheet. With a coloured base and a coloured Te scale
# the two compete; the -t25 transparency on grdimage (further down) is what
# keeps the base subordinate, so do NOT remove it. Pass RELIEFCPT=gray to
# restore the desaturated base.
printf 'relief palette: %s (hinged at sea level)\n' "${RELIEFCPT}"
gmt makecpt -C"${RELIEFCPT}" -T-8000/6000 > relief.cpt

# Discrete scale: the recovered thickness is not resolved finely enough to
# justify a continuous ramp, and discrete bands are read off a map faster.
# The classes are geometric, not uniform -- see TLEVELS in the header for why.
#
# The twelve gradual-ramp colours, evenly spaced samples along the six-anchor
# sequence violet@0 -> blue@0.2 -> green@0.4 -> yellow@0.6 -> brown@0.8 ->
# red@1.0. Held here rather than fetched through -C<ramp> so the file cannot
# change under the figure when GMT's bundled CPTs are updated, and so the RGB
# triplets can be quoted in the caption. The old ColorBrewer Paired set is kept
# below for CPT=paired.
GRAD12='138/43/226 94/60/214 49/77/202 40/109/153 40/145/94 96/175/62
        189/200/48 228/192/40 184/135/40 155/84/39 180/57/35 205/30/30'
PAIRED12='166/206/227 31/120/180 178/223/138 51/160/44 251/154/153 227/26/28
          253/191/111 255/127/0 202/178/214 106/61/154 255/255/153 177/89/40'

if [ -n "${TMIN}" ] && [ -n "${TMAX}" ] && [ -n "${TINC}" ]; then
    LCPT=${CPT}
    if [ "${LCPT}" = "grad6" ]; then
        LCPT=roma
        printf 'note: legacy linear scale cannot use the grad6 sentinel; using roma.\n' >&2
        printf 'pass CPT=<a GMT master> to choose another.\n' >&2
    fi
    printf 'colour scale: LINEAR %s-%s km in %s km steps (legacy override)\n' \
           "${TMIN}" "${TMAX}" "${TINC}"
    gmt makecpt -C"${LCPT}" -T${TMIN}/${TMAX}/${TINC} > te.cpt

elif [ "${CPT}" = "grad6" ]; then
    # GRADUAL SEQUENTIAL RAMP, one evenly spaced hue per class, pasted onto the
    # geometric boundaries EXACTLY as Paired is below. Each line is
    #   z_lo  colour  z_hi  colour
    # with the same colour at both ends, so the slice is flat and GMT never
    # interpolates the file. Twelve distinct hues keep the crowded sub-12 km
    # classes separable while the bar still reads violet->red left to right.
    NB=$(printf '%s' "${TLEVELS}" | awk -F, '{print NF}')
    NC=$((NB - 1))
    NG=$(printf '%s\n' ${GRAD12} | wc -l | tr -d ' ')
    if [ "${NC}" -ne "${NG}" ]; then
        printf 'error: %s classes from TLEVELS but %s ramp colours.\n' \
               "${NC}" "${NG}" >&2
        printf 'grad6 ships %s colours; give %s boundaries in TLEVELS,\n' \
               "${NG}" "$((NG + 1))" >&2
        printf 'or pass CPT=batlow for a resamplable ramp.\n' >&2
        exit 1
    fi
    printf 'colour scale: gradual violet->red, %s discrete classes, boundaries %s\n' \
           "${NC}" "${TLEVELS}"
    {
        printf '# Effective elastic thickness, km. Gradual violet->red ramp,\n'
        printf '# %s classes, one evenly spaced hue each, no interpolation.\n' "${NC}"
        printf '# COLOR_MODEL = RGB\n'
        printf '%s\n' "${TLEVELS}" | tr ',' '\n' > /tmp/te_levels.$$
        printf '%s\n' ${GRAD12} > /tmp/te_colours.$$
        awk 'NR==FNR {lev[FNR]=$1; nlev=FNR; next}
             {printf "%s\t%s\t%s\t%s\n", lev[FNR], $1, lev[FNR+1], $1}' \
            /tmp/te_levels.$$ /tmp/te_colours.$$
        # Background/foreground take the end classes; N stays white and paints
        # the "No Te recovered" bands -- the only thing now distinguishing an
        # unrecovered segment from a recovered one, so it must be drawn.
        printf 'B\t%s\n' "$(printf '%s\n' ${GRAD12} | head -1)"
        printf 'F\t%s\n' "$(printf '%s\n' ${GRAD12} | tail -1)"
        printf 'N\twhite\n'
    } > te.cpt
    rm -f /tmp/te_levels.$$ /tmp/te_colours.$$

elif [ "${CPT}" = "paired" ]; then
    # One class per Paired colour, written slice by slice. Each line is
    #   z_lo  colour  z_hi  colour
    # with the same colour at both ends, which is what makes the slice flat
    # instead of a gradient. GMT never interpolates a file in this form, so the
    # twelve colours reach the page exactly as listed above.
    NB=$(printf '%s' "${TLEVELS}" | awk -F, '{print NF}')
    NC=$((NB - 1))
    NP=$(printf '%s\n' ${PAIRED12} | wc -l | tr -d ' ')
    if [ "${NC}" -ne "${NP}" ]; then
        printf 'error: %s classes from TLEVELS but %s Paired colours.\n' \
               "${NC}" "${NP}" >&2
        printf 'Paired is a fixed set of twelve; give %s boundaries in\n' \
               "$((NP + 1))" >&2
        printf 'TLEVELS, or pass CPT=batlow for a resamplable ramp.\n' >&2
        exit 1
    fi
    printf 'colour scale: Paired, %s discrete classes, boundaries %s\n' \
           "${NC}" "${TLEVELS}"
    {
        printf '# Effective elastic thickness, km. ColorBrewer Paired,\n'
        printf '# %s classes, one colour each, no interpolation.\n' "${NC}"
        printf '# COLOR_MODEL = RGB\n'
        printf '%s\n' "${TLEVELS}" | tr ',' '\n' > /tmp/te_levels.$$
        printf '%s\n' ${PAIRED12} > /tmp/te_colours.$$
        awk 'NR==FNR {lev[FNR]=$1; nlev=FNR; next}
             {printf "%s\t%s\t%s\t%s\n", lev[FNR], $1, lev[FNR+1], $1}' \
            /tmp/te_levels.$$ /tmp/te_colours.$$
        printf 'B\t%s\n' "$(printf '%s\n' ${PAIRED12} | head -1)"
        printf 'F\t%s\n' "$(printf '%s\n' ${PAIRED12} | tail -1)"
        printf 'N\twhite\n'
    } > te.cpt
    rm -f /tmp/te_levels.$$ /tmp/te_colours.$$

else
    printf 'colour scale: geometric, boundaries %s, master %s\n' \
           "${TLEVELS}" "${CPT}"
    gmt makecpt -C"${CPT}" -T"${TLEVELS}" > te.cpt
    # Unrecovered segments must draw white on every palette, not just the
    # hand-built ones above. Replace any N line makecpt wrote with white.
    grep -v '^N' te.cpt > te.cpt.$$ && printf 'N\twhite\n' >> te.cpt.$$ \
        && mv te.cpt.$$ te.cpt
fi
NCLASS=$(grep -vc '^#\|^[BFN]' te.cpt)
printf 'classes: %s\n' "${NCLASS}"

gmt begin "${OUT}" pdf,png
  gmt grdimage relief.nc -Crelief.cpt -Ishade.nc -R"${REG}" -J${PROJ} -t25
  gmt coast -Ggray75 -Wthinnest,gray45 -Dl -A5000

  # Graticule, matching fig_studyarea: white hairlines at 30 degrees of
  # longitude and 20 of latitude, the same intervals the frame annotates.
  # MAP_GRID_PEN_PRIMARY above already sets 0.25p,white.
  #
  # Drawn HERE, between the relief and the segments, rather than folded into
  # the annotated frame at the end. The frame is the last thing plotted, so a
  # g code there would lay white hairlines over every coloured band and nick
  # each one at 30-degree intervals -- on a figure whose white bands carry the
  # meaning "no Te recovered", that is not a cosmetic problem. +n suppresses
  # the frame and annotations so this call contributes the grid and nothing
  # else; the annotated frame comes below.
  gmt basemap -Bxg30 -Byg20 -B+n

  # The traced axis is deliberately NOT drawn as a separate line: the axis file
  # and the coloured-band file are independent tracings that do not register
  # exactly, so overplotting them shows their disagreement rather than adding
  # information. A segment that carries no recovered value is drawn white
  # rather than left out, so the axis stays continuous; this is stated in the key.

  if [ -n "${SEGMENTS}" ] && [ -s "${SEGMENTS}" ]; then
      # +cl colours the line by the segment's -Z value through te.cpt.
      # Round line caps extend each band by half its width at both ends, so
      # consecutive segments meet and the one-node gaps between them (an
      # artefact of the 5 km resampling) no longer show as white notches on
      # the relief. The cap takes the band's own colour, not a fixed one.
      gmt plot "${SEGMENTS}" -Cte.cpt -W3.2p+cl --PS_LINE_CAP=round
      n=$(grep -c '^>' "${SEGMENTS}" || true)
      printf 'segments drawn: %s\n' "${n}"
  else
      printf 'note: SEGMENTS not set -- basemap only\n' >&2
      echo "205 -20 BASEMAP ONLY: supply SEGMENTS" |
          gmt text -F+f13p,Helvetica-Bold,red+jCM -N
  fi

  gmt basemap -Bxa30f10 -Bya20f10 -BWSne

  # Ocean label, as in fig_studyarea. Placed at 205E/5N: the middle of the
  # open central Pacific, clear of every margin, of the key to its south and
  # of the widened western edge. White on the relief, with no halo, so it reads
  # as a base-map label rather than as data. -N because the anchor is a map
  # coordinate and no clipping is wanted.
  #
  # 14p italic, set in capitals: the conventional cartographic treatment for
  # a hydrographic name, and lighter than the 16p bold it replaces, so the
  # coloured segments stay the most prominent thing on the sheet. Note the
  # font is Helvetica-Oblique, NOT Helvetica-Italic -- the latter is not one
  # of the 35 standard PostScript faces and GMT will fall back silently.
  echo "205 5 PACIFIC OCEAN" |
      gmt text -F+f14p,Helvetica-Oblique,white+jCM -N

  # NOTE: the on-map key that named the white bands has been removed at the
  # author's request; only the colour bar below the map remains. White bands
  # are still drawn (N white in te.cpt) for segments carrying no recovered
  # value, so the along-strike line stays continuous -- but nothing on the
  # sheet now says what white means. State it in the caption, or a reader will
  # take a white band for a thin-plate result rather than a missing one.

  # -L draws equal-sized rectangles and annotates every class boundary. With
  # widths proportional to z the 4-5 km class would be a fortieth of the bar
  # and its two labels would collide, which is exactly the region carrying
  # most of the segments. No +e: the end boundaries are the extremes of the
  # recovered distribution, so there is nothing to put in an overflow
  # triangle. -B cannot be combined with -L, so the axis label is drawn
  # separately, below the annotations, in map coordinates with -N.
  #
  # -L still fits the gradual ramp: the twelve hues progress smoothly, but the
  # scale is a discrete twelve-class key looked up boundary by boundary, so
  # every boundary must be annotated and every class given equal width to be
  # read in -- which is exactly what -L guarantees.
  gmt colorbar -Cte.cpt -DJBC+w9c/0.28c+h+o0/1.0c -L
  echo "${LON_MID} ${CAP_LAT} Recovered effective elastic thickness, @~T@~@-e@- (km)" |
      gmt text -F+f9p,Helvetica+jCM -N
gmt end

printf 'wrote %s.pdf and %s.png\n' "${OUT}" "${OUT}"
printf 'relief: %s   relief palette: %s   Te palette: %s   classes: %s\n' \
       "${RELIEF}" "${RELIEFCPT}" "${CPT}" "${NCLASS}"
