#!/usr/bin/env bash
# fig_datacoverage -- coverage of the gridded inputs over the circum-Pacific
# domain, with the traced trench axes superimposed, as a 2x2 panel.
#
#   (a) seafloor age          (b) sediment thickness
#   (c) free-air anomaly      (d) fourth layer, selected by PANEL_D
#
# The figure answers one question: do the gridded inputs actually carry a value
# at every margin the study uses? The colour scales therefore exist to show that
# values are present and sensible, not to be read off precisely.
#
# THE FOURTH PANEL
#   PANEL_D=geoid  (default) EGM2008 geoid height, metres. The trench geoid low
#                  and the outer-rise high are the long-wavelength expression of
#                  the same bending the inversion fits, so the panel is a second,
#                  independent view of the flexure and not merely decoration.
#   PANEL_D=vgg    Vertical gravity gradient, Eotvos. Sharper on the outer rise
#                  than the free-air anomaly, but derived from the same altimetry
#                  as panel (c) and therefore not independent of it.
#   PANEL_D=tid    GEBCO type identifier, reduced to measured vs predicted. This
#                  is the only panel that shows a genuine coverage gap rather
#                  than a field, and it is the one Section 2.2 already promises.
#                  Requires a local TID grid; there is no remote equivalent.
#
# USAGE
#   SED=GlobSed-v3.nc ./fig_datacoverage.sh
#   SED=GlobSed-v3.nc AGE=age.3.6.nc FAA=grav_33.1.nc \
#       GEOID=egm2008_pac.nc AXES_GMT=trench_axes_full.gmt ./fig_datacoverage.sh
#   SED=GlobSed-v3.nc PANEL_D=tid TID=gebco_2026_tid.nc ./fig_datacoverage.sh
#
# PB2002 as a plain lon/lat multisegment table, if you do not already have one:
#   curl -sL -o pb.json https://raw.githubusercontent.com/fraxen/tectonicplates/\
# master/GeoJSON/PB2002_boundaries.json
#   python3 -c "import json,sys; d=json.load(open('pb.json'))
#   [print('>')or[print('%.4f %.4f'%(x,y)) for x,y in c]
#    for f in d['features']
#    for c in ([f['geometry']['coordinates']] if f['geometry']['type']=='LineString'
#              else f['geometry']['coordinates'])]" > PB2002_plates.gmt
#
# INPUTS (environment variables; only SED is mandatory)
#   SED       total sediment thickness, m. GlobSed. NO remote equivalent exists,
#             so the script refuses to run without it.
#   AGE       oceanic crustal age, Myr.   Default @earth_age_06m (EarthByte).
#   FAA       free-air anomaly, mGal.     Default @earth_faa_01m (IGPP v33).
#   GEOID     geoid height, m.            Default @earth_geoid_01m (EGM2008).
#   VGG       vertical gravity gradient.  Default @earth_vgg_01m (IGPP v32).
#   TID       GEBCO type identifier grid. Local file only; no default.
#   PANEL_D   geoid | vgg | tid.          Default geoid.
#   AXES_GMT  multisegment file of traced axes. Default trench_axes_full.gmt.
#   PLATES    PB2002 plate boundaries as a GMT multisegment table, lon lat, on
#             either longitude convention. Default PB2002_plates.gmt. Omitted
#             silently if absent.
#   REG       region. Default 118/292/-56/62, matching fig_studyarea.
#   OUT       output basename. Default fig_datacoverage.
#   W         panel width. Default 7.6c; two of these fill \textwidth.
#
# Longitudes are handled on 0-360 to keep the Pacific unbroken; axes stored on
# -180..180 are wrapped on the fly, so either convention works.
set -eu

SED=${SED:?sediment-thickness grid required (GlobSed); no remote equivalent exists}
AGE=${AGE:-@earth_age_06m}
FAA=${FAA:-@earth_faa_01m}
GEOID=${GEOID:-@earth_geoid_01m}
VGG=${VGG:-@earth_vgg_01m}
TID=${TID:-}
PANEL_D=${PANEL_D:-geoid}
AXES_GMT=${AXES_GMT:-trench_axes_full.gmt}
PLATES=${PLATES:-PB2002_plates.gmt}
REG=${REG:-118/292/-56/62}
OUT=${OUT:-fig_datacoverage}
W=${W:-8.8c}
PROJ=Q${W}

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

# Type is deliberately small: four panels on one text width leave little room,
# and the panel tags and titles are labels, not headlines.
gmt set FONT_ANNOT_PRIMARY 5.5p,Helvetica FONT_LABEL 6.5p,Helvetica \
        FONT_TITLE 8p,Helvetica-Bold FONT_TAG 8p,Helvetica-Bold \
        MAP_FRAME_PEN 0.5p MAP_FRAME_TYPE plain MAP_TICK_LENGTH_PRIMARY 1.5p \
        MAP_GRID_PEN_PRIMARY 0.2p,white MAP_ANNOT_OFFSET 1.5p \
        MAP_TITLE_OFFSET 2p PS_MEDIA a2 GMT_GRAPHICS_DPU 600i

# The frame is declared ONCE, on subplot begin, and each panel then adds only
# its title. Passing the -B axis specification to every panel's own basemap
# call instead makes subplot draw its automatic annotations as well, and the
# two sets land on different degree steps -- that is what produced the
# doubled, colliding longitude labels (150E on top of 160E, and so on).
FRM="-Bxa40f10g20 -Bya30f10g30 -BWSne"

# Contour labels: small, black, and spaced along the line rather than at every
# crossing, so they annotate without becoming a second texture.
LBL="+f4p,Helvetica,black"
LBLGAP="-Gd4c"

# ---- colour scales ---------------------------------------------------------
# All four are perceptually uniform and colour-blind safe (Crameri). vik and
# roma are diverging and are anchored on zero, which is a real reference for
# both the free-air anomaly and the geoid height.
# Age runs light-young to dark-old, the sense used in the standard plate-age
# maps, so -I reverses batlow's default direction.
# lajolla reversed: cream at zero age darkening to deep maroon at 180 Ma. Warm
# and sequential, and deliberately not batlow, which is already carrying the
# published map in the companion paper.
gmt makecpt -Clajolla -I -T0/180/10 -Z > "${TMP}/age.cpt"
# GlobSed over this region: median 399 m, 90th percentile 3463 m. A 0-2000 m
# ramp saturates the top decile -- which is exactly the well-fed margins whose
# axial fill correction matters most -- so the range runs to 4000 m.
gmt makecpt -Cbamako -T0/4000/200 -Z -I > "${TMP}/sed.cpt"
# haxby, not a diverging pastel: nearly all open-ocean free-air anomaly sits
# within a few tens of mGal, so a +/-150 perceptual ramp washed the whole basin
# out. The narrower window costs saturation at the trench axes, which are the
# features that should read darkest anyway.
gmt makecpt -Chaxby -T-120/120/10 -Z > "${TMP}/faa.cpt"
# EGM2008 over this region runs -64.4 to +85.8 m. A symmetric range would
# waste half the ramp and an asymmetric diverging ramp puts its light middle
# off zero, so the palette is hinged on zero explicitly (+h0).
# turbo: not zero-hinged, but the geoid low over the western Pacific and the
# high off Central America are the features to separate, and turbo pulls them
# further apart than a diverging ramp anchored on an arbitrary datum.
gmt makecpt -Cturbo -T-70/90/5 -Z > "${TMP}/geoid.cpt"
gmt makecpt -Cbroc+h0 -T-40/40/4 -Z > "${TMP}/vgg.cpt"
# Two flat classes for the type identifier: measured soundings against depths
# predicted from gravity. Not a ramp -- the quantity is categorical.
cat > "${TMP}/tid.cpt" <<'CPT'
0	steelblue	1	steelblue	;Measured
1	orange3	2	orange3	;Predicted
CPT

# ---- trench axes, from the file fig_studyarea uses -------------------------
AXES_ALL="${TMP}/axes_all.gmt"
: > "${AXES_ALL}"
if [ -s "${AXES_GMT}" ]; then
    awk '/^#/{next}
         /^>/{ nm="axis"; if (match($0,/-L"[^"]+"/)) nm=substr($0,RSTART+3,RLENGTH-4)
               print "> " nm; next }
         NF>=2 { lon=$1+0; if (lon<0) lon+=360; print lon, $2 }' \
        "${AXES_GMT}" > "${AXES_ALL}"
    n_axes=$(grep -c '^>' "${AXES_ALL}" || true)
    printf 'trench axes drawn: %s\n' "${n_axes}"
    [ "${n_axes}" -ne 20 ] && \
        printf 'WARNING: %s axes, expected 20.\n' "${n_axes}" >&2
else
    printf 'WARNING: %s not found; panels drawn without axes.\n' "${AXES_GMT}" >&2
fi
draw_axes() { [ -s "${AXES_ALL}" ] && gmt plot "${AXES_ALL}" -W0.6p,black || true; }

# ---- plate boundaries -----------------------------------------------------
# Drawn twice: a white casing first, then the dark line over it, so the
# boundaries stay legible on both the dark end of the age scale and the pale
# interior of the gravity panel. Wrapped to 0-360 like the axes.
PLATES_ALL="${TMP}/plates.gmt"
: > "${PLATES_ALL}"
if [ -s "${PLATES}" ]; then
    awk '/^#/{next} /^>/{print ">"; next}
         NF>=2 { lon=$1+0; if (lon<0) lon+=360; print lon, $2 }' \
        "${PLATES}" > "${PLATES_ALL}"
    printf 'plate-boundary file: %s\n' "${PLATES}"
else
    printf 'NOTE: %s not found; panels drawn without plate boundaries.\n' "${PLATES}" >&2
fi
draw_plates() {
    [ -s "${PLATES_ALL}" ] || return 0
    gmt plot "${PLATES_ALL}" -W0.9p,white
    gmt plot "${PLATES_ALL}" -W0.3p,gray15
}

# ---- which layers are actually readable ------------------------------------
have() { gmt grdinfo "$1" -C > /dev/null 2>&1 && echo 1 || echo 0; }
HAVE_SED=$(have "${SED}")
HAVE_AGE=$(have "${AGE}")
HAVE_FAA=$(have "${FAA}")
[ "${HAVE_SED}" -eq 0 ] && printf 'WARNING: sediment grid %s unreadable.\n' "${SED}" >&2
[ "${HAVE_AGE}" -eq 0 ] && printf 'WARNING: age grid %s unavailable.\n' "${AGE}" >&2
[ "${HAVE_FAA}" -eq 0 ] && printf 'WARNING: gravity grid %s unavailable.\n' "${FAA}" >&2

# The fourth panel resolves to one grid, one cpt, one title, one bar annotation.
case "${PANEL_D}" in
  geoid) D_GRID="${GEOID}"; D_CPT="${TMP}/geoid.cpt"
         D_TITLE="Geoid height"; D_INT=10; D_ANN=20; D_BAR="-Bxa30f10+l\"Geoid (m)\"" ;;
  vgg)   D_GRID="${VGG}";   D_CPT="${TMP}/vgg.cpt"
         D_TITLE="Vertical gravity gradient"; D_INT=10; D_ANN=20; D_BAR="-Bxa20f10+l\"VGG (Eotvos)\"" ;;
  tid)   D_GRID="${TID:?PANEL_D=tid requires TID=<GEBCO type identifier grid>}"
         D_CPT="${TMP}/tid.cpt"
         D_TITLE="Bathymetric source"; D_INT=1; D_ANN=1; D_BAR="-L0.15c" ;;
  *)     printf 'PANEL_D must be geoid, vgg or tid (got %s)\n' "${PANEL_D}" >&2; exit 1 ;;
esac
HAVE_D=$(have "${D_GRID}")
[ "${HAVE_D}" -eq 0 ] && printf 'WARNING: fourth-panel grid %s unavailable.\n' "${D_GRID}" >&2

# The type identifier is categorical: codes below 40 are measured soundings,
# 40 and above are depths predicted from gravity. Collapse to 0/1 so the two
# classes cannot be read as a continuous ramp.
if [ "${PANEL_D}" = tid ] && [ "${HAVE_D}" -eq 1 ]; then
    gmt grdmath "${D_GRID}" 40 GE = "${TMP}/tid.nc"
    D_GRID="${TMP}/tid.nc"
fi

# A panel whose grid is missing still gets a frame and a note, so the layout
# does not silently reflow and the gap is visible rather than absent.
blank_panel() {
    gmt basemap -B0
    gmt coast -Ggray70 -Wthinnest,gray40 -Dl -A5000
    echo "grid unavailable" | gmt text -F+cCM+f8p,Helvetica-Oblique,red3 -Gwhite@30
}

# ---- the figure ------------------------------------------------------------
gmt begin "${OUT}" pdf,png
  gmt subplot begin 2x2 -Fs${W} -M0.25c/1.25c -A"a)"+jTL+o0.12c \
                        -R"${REG}" -J${PROJ} ${FRM}

    # (a) seafloor age -- sets the thermal reference depth of every profile
    gmt subplot set 0
      if [ "${HAVE_AGE}" -eq 1 ]; then
          gmt grdimage "${AGE}" -C"${TMP}/age.cpt" -Q
          gmt grdcontour "${AGE}" -C20 -A40${LBL} ${LBLGAP} \
                         -Wc0.15p,gray35@50 -Wa0.3p,gray25
          gmt coast -Ggray70 -Wthinnest,gray40 -Dl -A5000
          draw_plates
          draw_axes
          gmt basemap -B+t"Seafloor age"
          gmt colorbar -C"${TMP}/age.cpt" -DJBC+w${W}/0.18c+h+o0/0.35c \
                       -Bxa60f20+l"Age (Ma)"
      else
          blank_panel; gmt basemap -B+t"Seafloor age"
      fi

    # (b) sediment thickness -- GlobSed, supplied locally
    gmt subplot set 1
      if [ "${HAVE_SED}" -eq 1 ]; then
          gmt grdimage "${SED}" -C"${TMP}/sed.cpt" -Q
          gmt grdcontour "${SED}" -C500 -A1000${LBL} ${LBLGAP} \
                         -Wc0.15p,gray35@50 -Wa0.3p,gray25
          gmt coast -Ggray70 -Wthinnest,gray40 -Dl -A5000
          draw_plates
          draw_axes
          gmt basemap -B+t"Sediment thickness"
          gmt colorbar -C"${TMP}/sed.cpt" -DJBC+w${W}/0.18c+h+o0/0.35c \
                       -Bxa1000f500+l"Thickness (m)"
      else
          blank_panel; gmt basemap -B+t"Sediment thickness"
      fi

    # (c) free-air anomaly
    gmt subplot set 2
      if [ "${HAVE_FAA}" -eq 1 ]; then
          gmt grdimage "${FAA}" -C"${TMP}/faa.cpt" -Q
          gmt coast -Ggray70 -Wthinnest,gray40 -Dl -A5000
          draw_plates
          draw_axes
          gmt basemap -B+t"Free-air anomaly"
          gmt colorbar -C"${TMP}/faa.cpt" -DJBC+w${W}/0.18c+h+o0/0.35c \
                       -Bxa60f20+l"FAA (mGal)"
      else
          blank_panel; gmt basemap -B+t"Free-air anomaly"
      fi

    # (d) fourth layer
    gmt subplot set 3
      if [ "${HAVE_D}" -eq 1 ]; then
          gmt grdimage "${D_GRID}" -C"${D_CPT}" -Q
          # The type identifier is two flat classes; contours of it are noise.
          if [ "${PANEL_D}" != tid ]; then
              gmt grdcontour "${D_GRID}" -C"${D_INT}" -A"${D_ANN}"${LBL} ${LBLGAP} \
                             -Wc0.15p,gray20@50 -Wa0.3p,gray10
          fi
          gmt coast -Ggray70 -Wthinnest,gray40 -Dl -A5000
          draw_plates
          draw_axes
          gmt basemap -B+t"${D_TITLE}"
          eval gmt colorbar -C\"${D_CPT}\" -DJBC+w${W}/0.18c+h+o0/0.35c "${D_BAR}"
      else
          blank_panel; gmt basemap -B+t"${D_TITLE}"
      fi

  gmt subplot end
gmt end

# ---- coverage report: the number the figure exists to give ------------------
if [ -s "${AXES_ALL}" ]; then
    printf '\n--- axis nodes carrying a value ---------------------------------\n'
    for pair in "age:${AGE}:${HAVE_AGE}" "sediment:${SED}:${HAVE_SED}" \
                "gravity:${FAA}:${HAVE_FAA}" "${PANEL_D}:${D_GRID}:${HAVE_D}"; do
        lab=${pair%%:*}; rest=${pair#*:}; grid=${rest%:*}; ok=${rest##*:}
        if [ "${ok}" -eq 1 ]; then
            grep -v '^>' "${AXES_ALL}" |
            gmt grdtrack -G"${grid}" -fg 2>/dev/null |
            awk -v l="${lab}" '{n++; if ($3 == $3 && $3 != "NaN") k++}
                 END{printf "  %-10s %6d of %6d axis nodes (%.1f%%)\n", l, k, n, 100*k/n}'
        else
            printf '  %-10s grid unavailable\n' "${lab}"
        fi
    done
fi

printf '\nwrote %s.pdf and %s.png\n' "${OUT}" "${OUT}"
printf 'panel (d) = %s; grids: AGE=%s SED=%s FAA=%s D=%s\n' \
       "${PANEL_D}" "${AGE}" "${SED}" "${FAA}" "${D_GRID}"
