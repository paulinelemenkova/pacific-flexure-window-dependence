#!/usr/bin/env bash
# fig_residualmap -- where the elastic-plate model fails.
#
# The physics residual at convergence, mapped segment by segment over the same
# circum-Pacific base as fig_temap, with an inset giving its distribution. A
# recovered parameter is only as good as the model that produced it, so this
# figure is the honest companion to fig_temap: it shows which margins the
# elastic description cannot reproduce, and therefore where the mapped
# thickness should not be read at face value.
#
# USAGE
#   SEGMENTS=residual_segments.gmt ./fig_residualmap.sh
#   SEGMENTS=residual_segments.gmt RMAX=5e-3 ./fig_residualmap.sh
#   SEGMENTS=residual_segments.gmt RELIEF=@earth_relief_06m ./fig_residualmap.sh
#   SEGMENTS=residual_segments.gmt INSETLON=260 INSETLAT=-55 ./fig_residualmap.sh
#
# WHAT CHANGED IN THIS REVISION (read before comparing against an older run)
#
#   A. BATHYMETRY IS NOW oleron, NOT geo. oleron is one of Crameri's Scientific
#      Colour Maps: perceptually uniform, colour-vision-deficiency safe, and
#      carrying a HARD HINGE at sea level, so the ocean half and the land half
#      of the -T range are stretched independently and the colour break falls
#      exactly at 0 m rather than wherever a linear stretch happens to put it.
#      This is what Section 4.7 of the manuscript claims of every scale on the
#      sheet; geo is hinged but the old call also passed a fixed z-increment
#      (-T-8000/2000/250) together with -Z, which resamples a hinged master at
#      fixed steps and aliases the colour transitions near the hinge. The
#      increment is now dropped and the master is used as designed. The upper
#      limit is raised to +6000 m so that, if the grey land fill is ever
#      removed, the continental relief still reads on its own half of the ramp
#      instead of saturating at the foreground colour above 2000 m.
#
#   B. THE DISTRIBUTION INSET IS SMALLER AND SITS AT MAP CENTRE. It was a
#      4.55 x 2.75 cm panel anchored bottom-centre. It is now a 4.05 x 2.35 cm
#      panel anchored at MC, over the open central Pacific -- the largest area
#      of the sheet carrying neither coastline nor segment band -- so the inset
#      no longer sits on the graticule labels or crowds the colour bar. The
#      panel is again sized to the plot plus exactly the furniture the axes
#      need, so no annotation is clipped: 0.92 cm at the left for the y
#      annotations and label, 0.72 cm below for the x pair, 0.18 cm of air on
#      the other two sides. The panel is anchored GEOGRAPHICALLY, at 55 S /
#      100 W in the empty far South Pacific, and the script checks the
#      footprint against the segment file before drawing -- see D below.
#
#   C. THE HISTOGRAM PALETTE IS CHECKED BEFORE IT IS USED. HISTCPT already
#      defaulted to the Statistik Stadt Zuerich qualitative set, but a missing
#      cpt-city tree failed silently inside gmt begin: makecpt wrote an empty
#      hist.cpt, every bar took the foreground colour, and the inset came out
#      solid black -- the exact failure the comments below describe. The path
#      is now validated up front, beside the RESCPT check, with a message that
#      names the file it wanted.
#
# EARLIER FIXES, RETAINED (unchanged in this revision)
#
#   1. LONGITUDE WRAPPING -- the bug that emptied the map. REG spans 116-290,
#      so every longitude must be on 0-360. The old script wrapped the AXES
#      files but fed SEGMENTS to psxy untouched, so any segment stored on
#      -180/180 fell outside the region and was silently dropped:
#      Tonga-Kermadec (-175), Middle America (-95), Chile (-72) and the rest
#      of the eastern and southwestern margins never drew at all. Worse, a
#      segment straddling the date line with mixed signs drew as a straight
#      chord across the open Pacific rather than as a band on a trench, which
#      is the diagonal line through the middle of the old figure. Both are
#      fixed by normalising SEGMENTS the way AXES was always normalised, and
#      any segment still spanning more than 180 degrees after wrapping is
#      dropped with a warning rather than drawn as a chord.
#
#   2. INSET CLIPPING -- the plot origin is shifted into the panel so the axis
#      furniture is drawn inside the clip rectangle, not outside it.
#
#   3. HISTOGRAM Y RANGE -- computed from the binned data and rounded up, so a
#      tall bin cannot be silently cut off at the top of the frame.
#
#   4. COLOUR-SCALE STEP. Under -Qo the makecpt increment is a CODE, not a
#      decade width, and GMT accepts only three values:
#          1  one class per decade
#          2  classes at 1, 2 and 5 within each decade
#          3  classes at 1, 2, 3 ... 9 within each decade
#      Anything else aborts with
#          makecpt [ERROR]: For -Qo logarithmic spacing, inc must be 1, 2, or 3
#      RINC is validated against {1,2,3} here so a bad value fails with an
#      explanation rather than with GMT's terser message.
#
#   5. HARMONISED WITH fig_temap -- same relief default and GRDRES option,
#      round line caps so consecutive bands meet instead of showing white
#      notches, white bands plus a key entry for segments carrying no value,
#      and no separately drawn axis. AXES is still accepted and now ignored, as
#      in fig_temap, because the traced-axis file and the band file are
#      independent tracings that do not register.
#
# INPUTS (environment variables)
#   SEGMENTS  GMT multisegment file, one segment per along-strike bin, each
#             header carrying the residual as a -Z value. Same convention as
#             fig_temap, so the two figures are driven by files built the same
#             way and their bands are guaranteed to coincide:
#
#                 > -Z0.00042
#                 182.417  -22.905
#                 ...
#
#             Longitudes may be given on either -180/180 or 0/360; they are
#             normalised here.
#   AXES      accepted for backward compatibility but NO LONGER DRAWN.
#   SCRIPTDIR where this script lives. Resolved from the invocation path, so
#             the run line works from the project root, from the script
#             directory, or by absolute path. Falls back to the archive
#             location under 250_JGR_ML_C/scripts/essential_data_scripts.
#   GEBCODIR  the external volume holding the bathymetry. Default
#             "/Volumes/TOSHIBA EXT/DATA". The path contains a space, so every
#             use is quoted; keep the quotes if you edit it.
#   RELIEF    bathymetric grid. Default ${GEBCODIR}/GEBCO_2023.nc, matching
#             fig_temap. If that file is absent but the volume is mounted, the
#             newest GEBCO*.nc there is used and a note is printed -- check it
#             against the release cited in the Open Research section before
#             accepting the sheet. Pass RELIEF=@earth_relief_06m for the GMT
#             server copy.
#   GRDRES    optional resampling step applied to the cut grid, e.g. 01m.
#             Unset by default.
#   GRDFILT   1 (DEFAULT) boxcar-filters the grid before decimating it, which
#             removes the grdsample aliasing warning and the hillshade speckle
#             it warns about. This is what makes the shaded relief crisp: plain
#             decimation of a 15 arcsec grid to 01m keeps one cell in sixteen
#             and throws the rest away, and grdgradient then differentiates the
#             aliased surface. Costs one extra pass over the cut grid.
#             GRDFILT=0 restores plain grdsample. Sharper still: omit GRDRES
#             entirely and let GEBCO through at its native 15 arcsec.
#   GRDFILTD  grdfilter distance mode. Default 3 (width in km, longitude
#             spacing scaled by cos(latitude)). 4 uses full spherical
#             distances. 0 is NOT usable: a geographic grid rejects it.
#   REG       region. Default 116/290/-60/62, matching the widened fig_temap so
#             the two sheets share a base.
#   RMIN/RMAX colour limits. Default 1e-4/1e-1, logarithmic, because the
#             residual spans decades and a linear scale would compress every
#             well-fitted margin into one colour.
#   RINC      log-spacing CODE for -Qo: 1 (per decade), 2 (1-2-5) or
#             3 (1-2-...-9). NOT a decade width. Default 3.
#   CPTDIR    root of the unzipped cpt-city collection. Default
#             ${SCRIPTDIR}/cpt-city, i.e. beside the script rather than in the
#             current directory, so the figure builds the same from anywhere.
#   RESCPT    palette for the residual bands. Default gist/ncar.cpt, a bright
#             multi-hue ramp. RESCPT=lajolla restores the sequential Crameri
#             scale -- see the note beside the makecpt call, which matters for
#             the claim made in Section 4.7 of the manuscript.
#   HISTCPT   palette filling the histogram bars. Default ssz/qual-mixed-12,
#             the Statistik Stadt Zuerich qualitative set; the bars are
#             coloured by bin so the distribution reads at inset size.
#   RELIEFCPT palette for the bathymetry. Default oleron (Crameri, hard hinge
#             at sea level). RELIEFCPT=gray restores the desaturated base that
#             keeps the residual the only saturated colour on the sheet.
#   GRIDX     meridian spacing for the white graticule. Default 30, matching
#   GRIDY     the annotation interval; parallels default to 20, likewise.
#   OCEANLABEL text set in white italic caps at the centre of the frame.
#             Default "PACIFIC OCEAN". Set OCEANLABEL="" to omit it.
#   INSETLON  longitude of the inset anchor, on 0-360. Default 260 (100 W).
#   INSETLAT  latitude of the inset anchor. Default -55.
#   INSETJUST which corner of the inset sits on that anchor. Default BC, so
#             the anchor is the BOTTOM edge of the panel -- see the footprint
#             arithmetic beside the inset call for why MC will not fit.
#   INSETOFF  inset offset from that anchor, GMT +o syntax. Default 0/0c.
#   OUT       output basename. Default fig_residualmap.
set -eu

# ---------------------------------------------------------------------------
# MACHINE PATHS -- the only block that is specific to this workstation.
# Everything below refers to these three names, so moving the project or the
# external volume is a two-line edit here and nothing else.
#
#   SCRIPTDIR  where this script and its companions live. Resolved from the
#              path the script was invoked with, so it is correct whether you
#              run it from its own directory, from the project root, or by
#              absolute path. cd's into a subshell so the caller's working
#              directory is untouched. The literal default below is the
#              archive location, used only if the resolution somehow fails.
#   GEBCODIR   the external volume holding the bathymetric grid. NOTE the
#              space in "TOSHIBA EXT": every use is quoted, and it must stay
#              quoted if you edit these lines.
#   CPTDIR     the unzipped cpt-city collection, expected beside the script
#              rather than in the current directory, so the figure builds the
#              same way from anywhere.
# ---------------------------------------------------------------------------
SCRIPTDIR=${SCRIPTDIR:-$(cd "$(dirname "$0")" 2>/dev/null && pwd)}
SCRIPTDIR=${SCRIPTDIR:-/Users/polinalemenkova/Documents/My_Papers/1-PAPERS-ACCEPTED/250_JGR_ML_C/scripts/essential_data_scripts}
GEBCODIR=${GEBCODIR:-/Volumes/TOSHIBA EXT/DATA}

SEGMENTS=${SEGMENTS:?multisegment file with -Z residual required}
AXES=${AXES:-}
# Default to the GEBCO release named in the Open Research section. If that
# exact file is absent but the volume is mounted, take the newest GEBCO*.nc in
# GEBCODIR and SAY SO -- a figure built from a different release than the paper
# claims is a reproducibility failure, so this never happens silently.
RELIEF=${RELIEF:-${GEBCODIR}/GEBCO_2023.nc}
if [ "${RELIEF#@}" = "${RELIEF}" ] && [ ! -f "${RELIEF}" ] && [ -d "${GEBCODIR}" ]; then
    ALT=$(ls -1t "${GEBCODIR}"/GEBCO*.nc 2>/dev/null | head -1 || true)
    if [ -n "${ALT}" ]; then
        printf 'note: %s absent; using %s instead.\n' "${RELIEF}" "${ALT}" >&2
        printf 'CHECK that this release is the one cited in the Open Research section.\n' >&2
        RELIEF=${ALT}
    fi
fi
GRDRES=${GRDRES:-}
GRDFILT=${GRDFILT:-1}
GRDFILTD=${GRDFILTD:-3}
# A bare filename is looked for in the current directory first, then beside the
# script, so the run line works from either place.
if [ ! -f "${SEGMENTS}" ] && [ -f "${SCRIPTDIR}/${SEGMENTS}" ]; then
    SEGMENTS=${SCRIPTDIR}/${SEGMENTS}
fi
if [ ! -f "${SEGMENTS}" ]; then
    printf 'error: segment file not found: %s\n' "${SEGMENTS}" >&2
    printf 'looked in the current directory and in %s\n' "${SCRIPTDIR}" >&2
    printf 'build it with: python3 build_te_segments.py --value loss_phys ...\n' >&2
    exit 1
fi
REG=${REG:-116/290/-60/62}
# The r02 physics residual runs from about 9e-5 to 1.4e-1. The previous
# 1e-5/1e-2 ceiling put a third of the segments -- every one above 1e-2 --
# into the background colour, which read as the same near-white as a segment
# with no value at all. 1e-4/1e-1 covers the data with one decade of headroom
# each side and keeps the hues spread across the populated range.
RMIN=${RMIN:-1e-4}
RMAX=${RMAX:-1e-1}
RINC=${RINC:-3}
CPTDIR=${CPTDIR:-${SCRIPTDIR}/cpt-city}
RESCPT=${RESCPT:-${CPTDIR}/gist/ncar.cpt}
HISTCPT=${HISTCPT:-${CPTDIR}/ssz/qual-mixed-12.cpt}
RELIEFCPT=${RELIEFCPT:-oleron}
GRIDX=${GRIDX:-30}
GRIDY=${GRIDY:-20}
OCEANLABEL=${OCEANLABEL:-PACIFIC OCEAN}
INSETLON=${INSETLON:-260}
INSETLAT=${INSETLAT:--55}
INSETJUST=${INSETJUST:-BC}
INSETOFF=${INSETOFF:-0/0c}
OUT=${OUT:-fig_residualmap}

W=17c
PROJ=Q${W}

# ---- inset geometry, in one place -----------------------------------------
# Panel = plot + exactly the furniture the axes need. Change IW/IH and the
# panel follows; nothing else in the script hard-codes these numbers.
IW=2.95c        # histogram plot width
IH=1.45c        # histogram plot height
IML=0.92c       # left margin: y annotations + y label
IMB=0.72c       # bottom margin: x annotations + x label
IPW=4.05c       # panel width  = IML + IW + 0.18c
IPH=2.35c       # panel height = IMB + IH + 0.18c

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
         printf 'is the TOSHIBA EXT volume mounted? or pass RELIEF=@earth_relief_06m\n' >&2
         exit 1
     fi ;;
esac

# Palette availability is checked BEFORE any plotting starts. A cpt-city path
# that does not resolve used to fail inside gmt begin, where makecpt writes an
# empty file and the run continues to a black histogram or an uncoloured band
# set -- a wrong figure rather than an error. Names without a .cpt suffix are
# GMT master palettes (oleron, lajolla, gray, ...) and are left to GMT.
check_cpt() {   # $1 = palette value, $2 = variable name, for the message
    case "$1" in
      *.cpt) if [ ! -f "$1" ]; then
                 printf 'error: %s palette not found: %s\n' "$2" "$1" >&2
                 printf 'expected the cpt-city tree at: %s\n' "${CPTDIR}" >&2
                 printf 'unzip cpt-city.zip there, or set CPTDIR/%s.\n' "$2" >&2
                 exit 1
             fi ;;
      *) : ;;
    esac
}
check_cpt "${RESCPT}"  RESCPT
check_cpt "${HISTCPT}" HISTCPT

# -------------------------------------------------------------------------
# Normalise the segment longitudes onto 0-360, then screen out any segment
# that still spans the seam. Headers pass through untouched so the -Z value
# stays attached to its own geometry; that pairing is the whole reason the
# value travels in the header rather than in a side table.
# -------------------------------------------------------------------------
SEGS=segments_0360.gmt

awk '
  /^[[:space:]]*$/ { next }
  /^[>#]/          { print; next }
  {
      lon = $1 + 0
      while (lon <   0) lon += 360
      while (lon >= 360) lon -= 360
      printf "%.6f\t%s\n", lon, $2
  }' "${SEGMENTS}" > "${SEGS}.tmp"

# A segment whose wrapped longitudes span more than 180 degrees crosses the
# seam and would draw as a chord across the open ocean. Report and drop it;
# the real fix belongs upstream, in whatever builds the segment file.
awk -v out="${SEGS}" '
  function flush_seg(   i, span) {
      if (!have) return
      span = (n > 1) ? mx - mn : 0
      if (span > 180) { dropped++ } else {
          print hdr > out
          for (i = 1; i <= n; i++) print buf[i] > out
          kept++
      }
      have = 0; n = 0
  }
  # ONLY ">" opens a segment. build_te_segments.py writes a leading comment
  # ("# te_segments built from ...") and treating "#" as a header made that
  # comment a zero-point segment, which is why this reported one more segment
  # than psxy drew. Comments are passed straight through instead.
  /^#/    { print > out; next }
  /^>/    { flush_seg(); hdr = $0; have = 1; next }
  {
      n++; buf[n] = $0
      if (n == 1) { mn = mx = $1 + 0 }
      else { if ($1 + 0 < mn) mn = $1 + 0; if ($1 + 0 > mx) mx = $1 + 0 }
  }
  END {
      flush_seg()
      printf "segments kept: %d\n", kept
      if (dropped > 0)
          printf "WARNING: %d segment(s) span the seam after wrapping and were dropped\n",
                 dropped
  }' "${SEGS}.tmp"
rm -f "${SEGS}.tmp"

NSEG=$(grep -c '^>' "${SEGS}" || true)
NVAL=$(grep -c '^>.*-Z' "${SEGS}" || true)
printf 'segments drawn: %s (with a -Z value: %s, white: %s)\n' \
       "${NSEG}" "${NVAL}" "$((NSEG - NVAL))"
# runs/r02/segments.csv holds 71 segments over 20 margins. build_te_segments.py
# reports how many it actually joined and which trench axes it had to skip for
# want of a matching key -- read ITS stderr, not just this count. A shortfall
# here means the segment file is short, not that this script dropped anything.
if [ "${NSEG}" -lt 65 ]; then
    printf 'WARNING: %s segments drawn; runs/r02 has 71. Check for trenches\n' \
           "${NSEG}" >&2
    printf 'skipped by build_te_segments.py for a missing ALIASES entry.\n' >&2
fi

# -------------------------------------------------------------------------
# Relief, identical to fig_temap so the two sheets genuinely share a base.
# -------------------------------------------------------------------------
# Remove any scratch grids left by a previous run FIRST. GMT's NetCDF-4 writer
# will not overwrite an existing relief.nc, and an interrupted or
# permission-denied run leaves a half-written one behind; the next run then
# fails with "Can't add HDF5 file metadata [relief.nc]", which reads like a
# corrupt-input error but is really a refusal to clobber a stale output. These
# are derived files, regenerated every run, so deleting them is always safe.
rm -f relief.nc relief_r.nc shade.nc
gmt grdcut "${RELIEF}" -R"${REG}" -Grelief.nc
if [ -n "${GRDRES}" ]; then
    # COARSENING, so filter before decimating. grdsample alone answers with
    #     grdsample [WARNING]: Output sampling interval in x exceeds input
    #     interval and may lead to aliasing.
    # and it is right: GEBCO_2023 is on a 15-arcsecond grid, so going to 01m
    # throws away three cells in four by picking one and ignoring the rest.
    # On a shaded-relief base that shows as speckle in the hillshade, because
    # grdgradient then differentiates the aliased surface. grdfilter with a
    # boxcar the width of the output cell averages each cell instead, which is
    # the textbook decimation and costs one extra pass over the grid.
    # Set GRDFILT=0 to go back to plain grdsample.
    #
    # THE FILTER WIDTH MUST BE IN KILOMETRES, NOT DEGREES. GEBCO_2023 is
    # flagged geographic, and for such a grid grdfilter rejects -D0 with
    #     grdfilter [ERROR]: Option -D: Input grid is geographic but your
    #     distance mode is Cartesian
    # because -D0 is the only mode that takes the width in grid units. Every
    # mode valid for a geographic grid -- 1 through 5 -- expects kilometres.
    # So the output cell size is converted to degrees and then to km at
    # 111.195 km per degree (the mean spherical degree).
    #
    # -D3 scales the longitude spacing by cos(latitude) row by row, so the
    # filter footprint stays a true circle on the ground rather than being
    # stretched east-west toward the poles. Strictly, matching the output CELL
    # would want a fixed width in degrees in both directions -- which is what
    # GMT itself does for its remote grids, calling it Cartesian Gaussian
    # filtering -- but that is exactly the -D0 mode a geographic grid forbids.
    # The cost is mild over-smoothing in longitude at high latitude: at 60 N
    # the footprint spans about twice the intended number of columns. At 174
    # degrees across a 17 cm sheet that is far below one output pixel, and the
    # result is isotropic in real distance, which is what a hillshade wants
    # anyway. GRDFILTD=4 switches to full spherical distances if preferred.
    if [ "${GRDFILT}" = "1" ]; then
        FILTKM=$(awk -v r="${GRDRES}" 'BEGIN {
            n = r; s = ""
            if (r ~ /[sSmMdD]$/) { s = substr(r, length(r), 1)
                                   n = substr(r, 1, length(r) - 1) }
            n += 0
            if (s == "s" || s == "S")      d = n / 3600
            else if (s == "m" || s == "M") d = n / 60
            else                           d = n
            printf "%.6g", d * 111.195 }')
        printf 'resampling relief to %s (boxcar %s km, -D%s, no aliasing)\n' \
               "${GRDRES}" "${FILTKM}" "${GRDFILTD}"
        gmt grdfilter relief.nc -Fb"${FILTKM}" -D"${GRDFILTD}" \
                      -I"${GRDRES}" -Grelief_r.nc
    else
        printf 'resampling relief to %s (plain decimation)\n' "${GRDRES}"
        gmt grdsample relief.nc -I"${GRDRES}" -Grelief_r.nc
    fi
    if [ ! -f relief_r.nc ]; then
        printf 'error: resampling to %s produced no grid.\n' "${GRDRES}" >&2
        printf 'rerun with GRDFILT=0, or unset GRDRES to use the grid as cut.\n' >&2
        exit 1
    fi
    mv relief_r.nc relief.nc
fi
gmt grdgradient relief.nc -A315 -Ne0.5 -Gshade.nc

# BATHYMETRY: oleron, a Crameri Scientific Colour Map with a HARD HINGE at
# sea level. Because the hinge is hard, giving -T a range that spans zero is
# enough: GMT stretches the sea half onto -8000..0 and the land half onto
# 0..6000 INDEPENDENTLY, so the colour break lands exactly on the coastline
# instead of wherever a linear stretch across both signs would put it. That is
# also why no z-increment is appended here -- resampling a hinged master at
# fixed steps (the old -T-8000/2000/250) aliases the transitions around the
# hinge. Give the range only and let the master be used as designed.
#
# The land half is covered by the grey coast fill below, so on this sheet only
# oleron's ocean half is visible; the +6000 top is kept so the ramp still reads
# correctly if that fill is ever dropped.
#
# NOTE the trade-off a coloured base introduces. The grey base it replaces was
# chosen so that the residual was the only saturated colour on the page; with a
# coloured base and a bright residual scale the two compete. The -t25
# transparency on grdimage is what keeps the base subordinate, so do not remove
# it. Pass RELIEFCPT=gray to restore the diagnostic version.
# The -Z is deliberately absent. GMT answers a -T without an increment with
#     makecpt [WARNING]: Without inc in -T option, -Z has no effect (ignored)
# because a master used at its own resolution is already continuous; -Z only
# means anything when you are also resampling at fixed steps, which is exactly
# what must NOT be done to a hinged palette.
printf 'relief palette: %s (hinged at sea level)\n' "${RELIEFCPT}"
gmt makecpt -C"${RELIEFCPT}" -T-8000/6000 > relief.cpt

# Logarithmic and sequential, dark where the model fails, so a poorly fitted
# margin is the visually heaviest thing on the page -- the correct emphasis for
# a diagnostic figure. With -Qo the -T limits are in the original units and the
# increment is a spacing CODE, not a width: 1, 2 or 3 only (see RINC above).
case "${RINC}" in
  1|2|3) : ;;
  *) printf 'error: RINC=%s is not a valid -Qo spacing code.\n' "${RINC}" >&2
     printf 'Use 1 (one class per decade), 2 (1-2-5) or 3 (1-2-...-9).\n' >&2
     printf 'It is a code, not a decade width; GMT rejects anything else.\n' >&2
     exit 1 ;;
esac
# gist/ncar is a bright multi-hue ramp: violet and blue at the well-fitted end,
# green and yellow through the middle, red at the worst. It is NOT jet -- the
# hue order differs and it does not have jet's cyan-yellow compression -- but it
# shares jet's weaknesses: not perceptually uniform, not colour-blind safe, not
# greyscale-safe. Section 4.7 of the manuscript claims all three of every scale
# on the sheet, so either soften that sentence or run with RESCPT=lajolla, which
# is the Crameri sequential scale and satisfies the claim as written. No -I
# here: ncar runs dark-to-bright with increasing z, which already puts the worst
# fits at the bright end. lajolla runs the other way, so add -I if you switch.
printf 'residual palette: %s\n' "${RESCPT}"
gmt makecpt -C"${RESCPT}" -T"${RMIN}"/"${RMAX}"/"${RINC}" -Qo > res.cpt
# A segment carrying no -Z value takes the NaN colour: white, as in fig_temap,
# and named in the key, so the along-strike line stays continuous and a gap is
# never misread as a low residual.
printf 'N\twhite\n' >> res.cpt

# The Z values on their own, for the distribution inset. They are stored as
# log10 because the residual spans decades: histogram bins of constant width
# in the raw value would put every well-fitted margin in one bin.
awk '/^>/ { for (i = 1; i <= NF; i++)
              if (substr($i, 1, 2) == "-Z" && substr($i, 3) + 0 > 0)
                  print log(substr($i, 3) + 0) / log(10) }' \
    "${SEGS}" > res_log10.txt
awk '/^>/ { for (i = 1; i <= NF; i++)
              if (substr($i, 1, 2) == "-Z" && substr($i, 3) + 0 > 0)
                  print substr($i, 3) }' \
    "${SEGS}" > res_values.txt
printf 'segments with a positive residual: %s\n' "$(wc -l < res_values.txt)"
sort -g res_values.txt | awk '
    { v[NR] = $1; s += $1 }
    END { printf "median residual %.3e, mean %.3e, worst %.3e\n",
                 (NR % 2) ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2,
                 s / NR, v[NR] }'

# Inset y range from the data rather than a hard-wired 40 per cent, so a tall
# bin cannot be silently cut off at the top of the frame.
# 0.25 in log units over the -4..-1 range is 12 bins, which matches the twelve
# colours of qual-mixed-12 one-to-one, so no two adjacent bars share a hue.
BIN=0.25
lo=$(awk 'BEGIN{ printf "%.0f", log('"${RMIN}"') / log(10) }')
hi=$(awk 'BEGIN{ printf "%.0f", log('"${RMAX}"') / log(10) }')
YMAX=$(awk -v b="${BIN}" '
    { k = int($1 / b); if ($1 < 0) k--; c[k]++; n++ }
    END { for (j in c) if (c[j] > m) m = c[j]
          p = 100 * m / n
          printf "%d", (int(p / 10) + 1) * 10 }' res_log10.txt)
YANN=$(awk -v y="${YMAX}" 'BEGIN{ printf "%d", (y > 40) ? 20 : 10 }')
printf 'inset: log10 range %s to %s, tallest bin rounds up to %s per cent\n' \
       "${lo}" "${hi}" "${YMAX}"
printf 'inset: %s x %s panel, %s corner at %s E / %s, offset %s\n' \
       "${IPW}" "${IPH}" "${INSETJUST}" "${INSETLON}" "${INSETLAT}" "${INSETOFF}"

# -------------------------------------------------------------------------
# INSET FOOTPRINT CHECK. The panel is now anchored in MAP COORDINATES rather
# than to a frame corner, so nothing guarantees it stays inside the frame or
# clear of the bands -- that has to be computed. -JQ is equidistant, so one
# degree of longitude and one of latitude are the same length on the page and
# the conversion is a single scale factor:
#
#     cm per degree = W / (lon range) = 17 / 174 = 0.09770
#
# which makes the 4.05 x 2.35 cm panel 41.5 deg wide and 24.1 deg tall. That
# is why INSETJUST defaults to BC and not MC: centred on 55 S the panel would
# reach 67 S, seven degrees below the 60 S frame edge, and be clipped. Anchored
# BC it occupies 55 S to 31 S and sits inside. Its right edge falls near 79 W,
# leaving a few degrees of clearance from the Chile trench at about 75 W.
#
# Both facts are checked here rather than trusted, and both are reported, so
# moving the anchor cannot silently put the histogram on top of the data.
# -------------------------------------------------------------------------
eval "$(awk -v reg="${REG}" -v w="${W}" -v ipw="${IPW}" -v iph="${IPH}" \
            -v lon="${INSETLON}" -v lat="${INSETLAT}" -v just="${INSETJUST}" '
BEGIN {
    split(reg, r, "/")
    sub(/c$/, "", w); sub(/c$/, "", ipw); sub(/c$/, "", iph)
    dpc = (r[2] - r[1]) / w                 # degrees per cm
    wd  = ipw * dpc; hd = iph * dpc         # panel size in degrees
    v = substr(just, 1, 1); h = substr(just, 2, 1)
    if (v == "T")      { y1 = lat; y0 = lat - hd }
    else if (v == "B") { y0 = lat; y1 = lat + hd }
    else               { y0 = lat - hd / 2; y1 = lat + hd / 2 }
    if (h == "L")      { x0 = lon; x1 = lon + wd }
    else if (h == "R") { x1 = lon; x0 = lon - wd }
    else               { x0 = lon - wd / 2; x1 = lon + wd / 2 }
    printf "IX0=%.2f; IX1=%.2f; IY0=%.2f; IY1=%.2f; IWD=%.1f; IHD=%.1f\n",
           x0, x1, y0, y1, wd, hd
    printf "IOUT=%d\n", (x0 < r[1] || x1 > r[2] || y0 < r[3] || y1 > r[4])
}')"
printf 'inset footprint: %s to %s E, %s to %s (%s x %s degrees)\n' \
       "${IX0}" "${IX1}" "${IY0}" "${IY1}" "${IWD}" "${IHD}"
if [ "${IOUT}" = "1" ]; then
    printf 'WARNING: the inset panel extends outside %s and will be clipped.\n' \
           "${REG}" >&2
    printf 'Move it with INSETLON/INSETLAT, or change INSETJUST (BC anchors\n' >&2
    printf 'the bottom edge, TC the top, MC the centre).\n' >&2
fi
HIT=$(awk -v x0="${IX0}" -v x1="${IX1}" -v y0="${IY0}" -v y1="${IY1}" '
    /^[>#]/ { next }
    NF >= 2 && $1 + 0 >= x0 && $1 + 0 <= x1 && $2 + 0 >= y0 && $2 + 0 <= y1 { n++ }
    END { printf "%d", n + 0 }' "${SEGS}")
if [ "${HIT}" -gt 0 ]; then
    printf 'WARNING: %s segment node(s) fall under the inset panel.\n' "${HIT}" >&2
    printf 'The histogram would cover coloured trench. Move it with\n' >&2
    printf 'INSETLON/INSETLAT before accepting this sheet.\n' >&2
else
    printf 'inset clear of all %s segments\n' "${NSEG}"
fi

# -------------------------------------------------------------------------
# HISTOGRAM PALETTE -- built here, by hand, NOT by makecpt.
#
# This is the fix for
#     histogram [ERROR]: Color palette table hist.cpt is empty
#     [Session gmt (0)]: Error returned from GMT API: GMT_CPT_READ_ERROR (8)
# which left the inset unfilled even though the SSZ file was found (the path
# check above passed, so the file exists and is readable).
#
# The cause is makecpt, not the file. qual-mixed-12 is a QUALITATIVE master:
# twelve unrelated hues on integer z slices, with no ordering to interpolate
# along. Asking makecpt to restretch that onto -4/-1 is asking it to
# interpolate between categories, and rather than do something meaningless it
# writes nothing -- an empty table, silently, because makecpt's own exit status
# is still zero. The old comment in this script blamed the increment; that was
# the wrong diagnosis. Neither -T lo/hi nor -T lo/hi/inc produces a usable
# table from a qualitative master.
#
# So the twelve colours are read straight out of the master and re-laid on the
# twelve histogram bins. Category k keeps its own hue and simply acquires new z
# bounds, which is exactly what a qualitative palette is for. Both cpt-city
# spellings are handled: "z0 R/G/B z1 R/G/B" and "z0 R G B z1 R G B", with any
# trailing ";label" and any B/F/N line ignored. If the bin count and the colour
# count ever diverge the colours cycle, so the inset still draws.
# -------------------------------------------------------------------------
NBIN=$(awk -v l="${lo}" -v h="${hi}" -v b="${BIN}" \
           'BEGIN { printf "%d", (h - l) / b + 0.5 }')
awk -v lo="${lo}" -v b="${BIN}" -v n="${NBIN}" '
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  {
      line = $0
      sub(/;.*/, "", line)              # drop any ";label"
      nf = split(line, f, /[[:space:]]+/)
      if (f[1] == "") { for (i = 1; i < nf; i++) f[i] = f[i + 1]; nf-- }
      if (nf < 4) next
      if (f[1] == "B" || f[1] == "F" || f[1] == "N") next
      if (f[1] !~ /^-?[0-9.]+$/) next
      if (nf >= 8 && f[3] ~ /^[0-9]+$/ && f[4] ~ /^[0-9]+$/)
          col[++m] = f[2] "/" f[3] "/" f[4]   # z0 R G B z1 R G B
      else
          col[++m] = f[2]                     # z0 R/G/B z1 R/G/B, or #hex
  }
  END {
      if (m == 0) exit 1
      for (i = 0; i < n; i++) {
          c = col[(i % m) + 1]
          printf "%g\t%s\t%g\t%s\n", lo + i * b, c, lo + (i + 1) * b, c
      }
      printf "B\twhite\nF\tgray30\nN\tgray80\n"
      printf "%d", m > "hist_ncol.txt"
  }' "${HISTCPT}" > hist.cpt || true
if [ ! -s hist.cpt ]; then
    printf 'error: no colour slices could be read from %s\n' "${HISTCPT}" >&2
    printf 'the file exists but is not in a cpt layout this script recognises.\n' >&2
    printf 'run: head -20 "%s"  and check it is a GMT .cpt, not a .svg or .pdf.\n' \
           "${HISTCPT}" >&2
    exit 1
fi
NCOL=$(cat hist_ncol.txt 2>/dev/null || echo '?')
rm -f hist_ncol.txt
printf 'histogram palette: %s -- %s colours re-laid on %s bins of %s in log10\n' \
       "${HISTCPT}" "${NCOL}" "${NBIN}" "${BIN}"
if [ "${NCOL}" != "${NBIN}" ]; then
    printf 'note: %s colours for %s bins, so hues repeat. For one hue per bin\n' \
           "${NCOL}" "${NBIN}" >&2
    printf 'set BIN so that (%s - %s) / BIN equals %s.\n' "${hi}" "${lo}" "${NCOL}" >&2
fi

gmt begin "${OUT}" pdf,png
  gmt grdimage relief.nc -Crelief.cpt -Ishade.nc -R"${REG}" -J${PROJ} -t25
  gmt coast -Ggray75 -Wthinnest,gray45 -Dl -A5000

  # Graticule, drawn HERE and not on the annotated frame below, so the white
  # lines pass under the coloured bands instead of scoring white notches
  # across them -- on this figure white already means "no residual recovered".
  # -B+n suppresses the frame and annotations so only the gridlines are laid
  # down; the pen is the thin white MAP_GRID_PEN_PRIMARY set at the top.
  gmt basemap -Bxg${GRIDX} -Byg${GRIDY} -B+n

  # The traced axis is deliberately NOT drawn: it and the band file are
  # independent tracings that do not register, so overplotting them shows
  # their disagreement rather than adding information.
  if [ -n "${AXES}" ]; then
      printf 'note: AXES accepted but no longer drawn (see header)\n' >&2
  fi

  # Round caps extend each band by half its width at both ends so consecutive
  # segments meet. Without this the one-node gaps left by the 5 km resampling
  # show as white notches -- and on THIS figure white means "no residual
  # recovered", so the artefact would read as a statement about the data.
  gmt plot "${SEGS}" -Cres.cpt -W3.2p+cl --PS_LINE_CAP=round

  gmt basemap -Bxa30f10 -Bya20f10 -BWSne

  # Ocean name at the geometric centre of the frame, computed from REG so it
  # follows if the region is ever changed. Italic is the cartographic
  # convention for a water body; white reads over the pale relief without the
  # weight a black label would carry, and this is a background label, not data.
  if [ -n "${OCEANLABEL}" ]; then
      awk -v reg="${REG}" -v t="${OCEANLABEL}" 'BEGIN {
          split(reg, r, "/")
          printf "%.4f\t%.4f\t%s\n", (r[1] + r[2]) / 2, (r[3] + r[4]) / 2, t }' \
          | gmt text -F+f14p,Helvetica-Oblique,white+jCM
  fi

  # Distribution of the residual, so the map is read against its own spread.
  #
  # SIZE AND PLACEMENT. The panel is sized to the plot plus exactly the
  # furniture the axes need (see IW/IH/IML/IMB at the top) and anchored at map
  # centre, over the open central Pacific -- the largest area of the sheet
  # carrying no coastline and no segment band. Smaller fonts inside the inset
  # keep it subordinate to the map.
  #
  # hist.cpt is built above, by hand, NOT by makecpt -- see the block headed
  # "HISTOGRAM PALETTE" for why.
  gmt inset begin -Dg${INSETLON}/${INSETLAT}+w${IPW}/${IPH}+j${INSETJUST}+o${INSETOFF} \
                  -F+gwhite@10+p0.5p,gray40+c0.06c
    # Faint grid UNDER the bars: an empty gridded frame first, then the
    # histogram over it with no frame of its own.
    gmt basemap -R${lo}/${hi}/0/${YMAX} -JX${IW}/${IH} -X${IML} -Y${IMB} \
                -Bxg1 -Byg5 --MAP_GRID_PEN_PRIMARY=0.2p,gray85
    gmt histogram res_log10.txt -R${lo}/${hi}/0/${YMAX} -JX${IW}/${IH} \
                  -T${BIN} -Chist.cpt -W0.2p,gray30 -Z1 \
                  -Bxa1f0.5+l"log@-10@- physics residual" \
                  -Bya${YANN}f5+l"% of segments" -BWSne \
                  --FONT_ANNOT_PRIMARY=6.5p,Helvetica \
                  --FONT_LABEL=7p,Helvetica \
                  --MAP_TICK_LENGTH_PRIMARY=1.5p \
                  --MAP_ANNOT_OFFSET=1.2p \
                  --MAP_FRAME_PEN=0.6p,gray30
  gmt inset end

  # Key for the white bands -- drawn ONLY if there are any. On the r02 file
  # every segment carries a value, so the key would have explained a symbol
  # that does not appear on the sheet, which is worse than no key at all.
  if [ "$((NSEG - NVAL))" -gt 0 ]; then
      gmt legend -Dg150/-48+jMC+w6.4c -F+gwhite@15+p0.4p,gray40+r2p <<'LEG'
S 0.30c s 0.34c white 0.4p,gray45 0.85c No residual recovered
LEG
  fi

  gmt colorbar -Cres.cpt -DJBC+w9c/0.28c+h+o0/1.0c+e -Q \
               -Bxa1f3+l"Physics residual at convergence"
gmt end

printf 'wrote %s.pdf and %s.png\n' "${OUT}" "${OUT}"
printf 'relief: %s\n' "${RELIEF}"
printf 'scale: %s to %s, logarithmic, %s decade classes\n' \
       "${RMIN}" "${RMAX}" "${RINC}"
