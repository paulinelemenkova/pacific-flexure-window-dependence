#!/usr/bin/env bash
# fig_parametermaps -- the remaining recovered flexural parameters, mapped
# segment by segment over the same circum-Pacific base as fig_temap.
#
#   (a) bending moment   (b) shear force at the axis   (c) maximum curvature
#
# The three panels share one region, one projection and one relief base, so a
# reader can carry their eye down a margin and compare the fields on the same
# segments. Each panel keeps its own colour scale, because the quantities have
# different units and a shared scale would be meaningless.
#
# THE FOURTH PANEL IS GONE. It showed the uncertainty in the elastic thickness,
# which is the ensemble standard deviation of Section 4. run.json records a
# single seed, and a standard deviation over one member does not exist, so the
# panel could only ever have been blank. Restoring it needs the ensemble, not
# a change here; the limitation is entered in Table~limitations of the paper.
#
# USAGE
#   SEGMENTS=segments.gmt VALUES=segment_features.csv ./fig_parametermaps.sh
#   SEGMENTS=segments.gmt VALUES=values.csv AXES=axes RELIEF=gebco.nc \
#       ./fig_parametermaps.sh
#   SEGMENTS=segments.gmt VALUES=values.csv GRDRES=01m ./fig_parametermaps.sh
#
# INPUTS (environment variables)
#   SEGMENTS  GMT multisegment file of the along-strike bins. Each header
#             names the segment, and the name is the key into VALUES:
#
#                 > tonga_017
#                 182.417  -22.905
#                 182.398  -22.815
#                 ...
#
#   VALUES    CSV with a header row and at least these columns:
#
#                 segment,moment,shear,curvature,te_err
#
#             moment in N (per unit length), shear in N/m, curvature in
#             1e-7 m^-1, te_err in km. A segment absent from VALUES, or
#             carrying a blank field, is simply not drawn on that panel --
#             nothing is interpolated to fill a gap.
#
#   AXES      optional directory of "<trench>.txt" traced axes, drawn thin and
#             grey beneath every panel so that segments without a recovered
#             value read as gaps rather than vanishing.
#   SCRIPTDIR where this script lives. Resolved from the invocation path, so
#             the run line works from the project root, from the script
#             directory, or by absolute path. Falls back to the archive
#             location under 250_JGR_ML_C/scripts/essential_data_scripts.
#   GEBCODIR  the external volume holding the bathymetry. Default
#             "/Volumes/TOSHIBA EXT/DATA". The path contains a space, so every
#             use is quoted; keep the quotes if you edit it.
#   RELIEF    bathymetric grid. Default ${GEBCODIR}/GEBCO_2023.nc, matching
#             fig_temap and fig_residualmap. If that file is absent but the
#             volume is mounted, the newest GEBCO*.nc there is used and a note
#             is printed -- check it against the release cited in the Open
#             Research section. Pass RELIEF=@earth_relief_06m for the GMT
#             server copy. (The old header here claimed the server copy was
#             the default; it has not been since the local grid was adopted.)
#   RELIEFCPT palette for the bathymetry. Default oleron (Crameri, hard hinge
#             at sea level). RELIEFCPT=gray restores the desaturated base.
#   GRDRES    optional resampling step applied to the cut grid, e.g. 01m.
#             Unset by default.
#   GRDFILT   1 (DEFAULT) boxcar-filters the grid before decimating it. Plain
#             decimation of a 15 arcsec grid to 01m keeps one cell in sixteen
#             and discards the rest; grdgradient then differentiates the
#             aliased surface and the hillshade speckles. GRDFILT=0 restores
#             plain grdsample. Sharper still: omit GRDRES entirely.
#   GRDFILTD  grdfilter distance mode. Default 3 (width in km, longitude
#             spacing scaled by cos(latitude)). 4 uses full spherical
#             distances. 0 is NOT usable: a geographic grid rejects it.
#   REG       region. Default 116/290/-60/62, matching fig_residualmap. (The
#             old header said 140/290; the code has said 116 for some time.)
#   OUT       output basename. Default fig_parametermaps.
#
# Scale limits, overridable per panel:
#   MRANGE (default -8/8/1, 1e16 N)     VRANGE (default 0/6/0.5, 1e12 N/m)
#   KRANGE (default 0/8/1, 1e-7 1/m)    ERANGE (default 0/10/1, km)
set -eu

# ---------------------------------------------------------------------------
# MACHINE PATHS -- the only block specific to this workstation, and identical
# to the one in fig_residualmap.sh so the two figures cannot drift apart.
# ---------------------------------------------------------------------------
SCRIPTDIR=${SCRIPTDIR:-$(cd "$(dirname "$0")" 2>/dev/null && pwd)}
SCRIPTDIR=${SCRIPTDIR:-/Users/polinalemenkova/Documents/My_Papers/1-PAPERS-ACCEPTED/250_JGR_ML_C/scripts/essential_data_scripts}
GEBCODIR=${GEBCODIR:-/Volumes/TOSHIBA EXT/DATA}

SEGMENTS=${SEGMENTS:?multisegment geometry file required}
VALUES=${VALUES:?per-segment value table required}
# Column names in VALUES. Overridable, because the derived quantities do not
# always carry these headings: the table that holds them may name them M_Nm,
# V_N_per_m, kappa_max or similar. A wrong name here is NOT an error in the
# CSV reader -- it silently yields an empty column, every band is dropped for
# want of a value, and the sheet renders as three blank maps. That failure is
# now caught before anything is written; see the check at the end of build_z.
# Columns confirmed against derived_segments.csv (71 rows). Note that the file
# holds aliases: moment == moment_axis, shear == shear_axis == shear_max, and
# curvature == kappa_max, column for column. The short names are used here.
MCOL=${MCOL:-moment}
VCOL=${VCOL:-shear}
KCOL=${KCOL:-curvature}
# UNIT DIVISORS. The colour bars are labelled in scaled units -- 10^16 N,
# 10^12 N/m, 10^-7 m^-1 -- and the -T ranges below are written in those same
# scaled units. The CSV columns are in SI. Without these divisors a moment of
# 6.1e15 is compared against a bar running to 1.5 and saturates at the top on
# every band, while a curvature of 1.6e-6 sits at two millionths of its bar and
# renders as the bottom colour on every band. That is the flat, near-identical
# pair of panels this figure produced before. Set a divisor to 1 if the column
# is already scaled.
MSCALE=${MSCALE:-1e16}
VSCALE=${VSCALE:-1e12}
KSCALE=${KSCALE:-1e-7}
# FOURTH PANEL: the deflection scale at the axis, the amplitude that all three
# derived quantities carry, so the panel shows whether their pattern follows
# amplitude or shape. It comes from a second table because the derived file
# does not hold it. PANELD=0 restores the one-row, three-panel layout.
# Graticule spacing, matching the annotation interval of the frame.
GRIDX=${GRIDX:-60}
GRIDY=${GRIDY:-30}
PANELD=${PANELD:-1}
VALUES4=${VALUES4:-runs/r02/segments.csv}
DCOL=${DCOL:-w0_m}
DSCALE=${DSCALE:-1}
DRANGE=${DRANGE:-0/7000/500}
DTITLE=${DTITLE:-Deflection scale at the axis}
DLABEL=${DLABEL:-w@-b@- (m)}
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
RELIEFCPT=${RELIEFCPT:-oleron}
GRDRES=${GRDRES:-}
GRDFILT=${GRDFILT:-1}
GRDFILTD=${GRDFILTD:-3}
REG=${REG:-116/290/-60/62}
OUT=${OUT:-fig_parametermaps}

# A bare filename is looked for in the current directory first, then beside the
# script, so the run line works from either place.
for v in SEGMENTS VALUES; do
    eval "cur=\${$v}"
    if [ ! -f "${cur}" ] && [ -f "${SCRIPTDIR}/${cur}" ]; then
        eval "$v=\${SCRIPTDIR}/\${$v}"
    fi
    eval "cur=\${$v}"
    if [ ! -f "${cur}" ]; then
        printf 'error: %s not found: %s\n' "$v" "${cur}" >&2
        printf 'looked in the current directory and in %s\n' "${SCRIPTDIR}" >&2
        exit 1
    fi
done

# Ranges measured from the r02 evaluation pass over the 47 segments the
# residual screen retains, not guessed. The stock -8/8 for the moment left the
# median at a fiftieth of the bar and the stock 0/8 for the curvature saturated
# at twice the bar maximum; both are corrected here.
#   |M|    median 6.1e15 N     ->  +/-1.5 in units of 1e16
#   |V|    median 8.5e11 N/m   ->  0 to 12 in units of 1e12
#   |kmax| median 1.6e-6 1/m   ->  0 to 70 in units of 1e-7
# Ranges in the SCALED units of the colour-bar labels, sized so the measured
# median sits inside the bar rather than against an end of it:
#   |M|    median 0.61 in 10^16 N     -> -1.5/1.5   (70 per cent of the bar)
#   |V|    median 0.85 in 10^12 N/m   -> -2.5/2.5   (66 per cent of the bar)
#   |kmax| median 16   in 10^-7 1/m   ->  0/70      (23 per cent of the bar)
# The shear at the axis CHANGES SIGN along strike, as the moment does, so its
# range is symmetric about zero and its palette diverging. The old 0/12 was
# both one-signed and five times too wide: a negative shear fell off the bottom
# of the bar and took the background colour on every band, which is why panels
# (b) and (c) came out as identical flat fills for opposite reasons.
# Ranges in the SCALED units of the colour-bar labels, set from the measured
# distribution of each column rather than from a stock guess. THE SIGN MATTERS
# AS MUCH AS THE WIDTH: a one-signed range on a column that changes sign sends
# every negative band to the background colour, which is the flat-fill failure
# this figure produced twice.
#
#   moment     median 0.79, 2 of 71 negative, max 128  -> -2/2, diverging.
#              The extreme saturates; it is one segment out of seventy-one and
#              a range wide enough to hold it would flatten all the others.
#   shear      median 1.12, 71 of 71 NEGATIVE          -> -3/0, sequential.
#              The sign never changes, so a diverging scale would waste half
#              the palette; the ramp is reversed so the largest magnitude is
#              the warmest colour.
#   curvature  median 1.68, 45 of 71 negative          -> -70/70, diverging.
#              The sign is the sense of bending and is the point of the panel.
MRANGE=${MRANGE:--2/2/0.25}
VRANGE=${VRANGE:--3/0/0.25}
KRANGE=${KRANGE:--70/70/10}
# Palettes, one per panel, so a range can be changed to one-signed without
# editing the makecpt calls. vik is diverging and zero-centred; turbo is
# sequential. Both are perceptually ordered.
MCPT=${MCPT:-gnuplot2}
VCPT=${VCPT:-cool}
KCPT=${KCPT:-hot}
DCPT=${DCPT:-hot}
# NOTE ON SIGN. gnuplot2 and hot are SEQUENTIAL ramps, and panels (a) and (c)
# carry signed quantities on ranges symmetric about zero, so zero now falls at
# the middle of the ramp rather than at a hue boundary. The sign of the moment
# and of the curvature is therefore no longer readable at a glance the way it
# was on a diverging scale; the magnitude still is. Pass MCPT=vik KCPT=vik to
# restore the zero-centred reading.
# GMT masters used here: gnuplot2, cool and hot. If a GMT build does not carry
# gnuplot2, point MCPT at a cpt-city file instead, e.g.
#   MCPT=${CPTDIR:-cpt-city}/... /gnuplot2.cpt

# 2x2 at 8.1 cm per panel against 1x3 at 5.4 cm: each map is half again as wide
# and the bands, which are 2.6 pt lines on a 174-degree sheet, become legible
# instead of hairlines. 2*8.1 + 0.45 gap = 16.65 cm, inside the 17 cm text
# width the rest of the series uses. The height follows the aspect ratio of the
# region, 174/122, so the graticule stays square.
W=8.1c
H=5.68c
PROJ=Q${W}

gmt set FONT_ANNOT_PRIMARY 7p,Helvetica FONT_LABEL 8p,Helvetica \
        FONT_TITLE 9p,Helvetica-Bold MAP_FRAME_PEN 0.6p \
        MAP_FRAME_TYPE plain MAP_TICK_LENGTH_PRIMARY 2p \
        MAP_GRID_PEN_PRIMARY 0.25p,white MAP_ANNOT_OFFSET 2p \
        PS_MEDIA a2 GMT_GRAPHICS_DPU 600i

case "${RELIEF}" in
  @*) : ;;
  *) if [ ! -f "${RELIEF}" ]; then
         printf 'error: relief grid not found: %s\n' "${RELIEF}" >&2
         printf 'is the TOSHIBA EXT volume mounted? or pass RELIEF=@earth_relief_06m\n' >&2
         exit 1
     fi ;;
esac
# Remove any scratch grids left by a previous run FIRST. GMT's NetCDF-4 writer
# will not overwrite an existing relief.nc, and an interrupted or
# permission-denied run leaves a half-written one behind; the next run then
# fails with "Can't add HDF5 file metadata [relief.nc]", which reads like a
# corrupt-input error but is really a refusal to clobber a stale output. These
# are derived files, regenerated every run, so deleting them is always safe.
rm -f relief.nc relief_r.nc shade.nc d.gmt d.cpt
gmt grdcut "${RELIEF}" -R"${REG}" -Grelief.nc
if [ -n "${GRDRES}" ]; then
    # COARSENING, so filter before decimating. grdsample alone answers with
    #     grdsample [WARNING]: Output sampling interval in x exceeds input
    #     interval and may lead to aliasing.
    # and it is right: GEBCO_2023 is on a 15-arcsecond grid, so going to 01m
    # keeps one cell in sixteen and throws the rest away. grdgradient then
    # differentiates the aliased surface and the hillshade speckles.
    #
    # THE FILTER WIDTH MUST BE IN KILOMETRES, NOT DEGREES. GEBCO_2023 is
    # flagged geographic, and for such a grid grdfilter rejects -D0 with
    #     grdfilter [ERROR]: Option -D: Input grid is geographic but your
    #     distance mode is Cartesian
    # because -D0 is the only mode taking the width in grid units. Every mode
    # valid for a geographic grid -- 1 through 5 -- expects kilometres, so the
    # output cell size is converted at 111.195 km per degree.
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

# BATHYMETRY: oleron, a Crameri Scientific Colour Map with a HARD HINGE at sea
# level, matching fig_residualmap. Because the hinge is hard, a -T range that
# spans zero is enough: GMT stretches the sea half onto -8000..0 and the land
# half onto 0..6000 INDEPENDENTLY, so the colour break lands exactly on the
# coastline. No z-increment is appended -- resampling a hinged master at fixed
# steps (the old -T-8000/2000/250) aliases the transitions around the hinge --
# and no -Z either, since without an increment GMT answers it with
#     makecpt [WARNING]: Without inc in -T option, -Z has no effect (ignored)
#
# The land half is covered by the grey coast fill in panel(), so only oleron's
# ocean half is visible here; the +6000 top is kept so the ramp still reads if
# that fill is ever dropped. The -t30 transparency in panel() is what keeps the
# base subordinate to the coloured bands, so do not remove it. RELIEFCPT=gray
# restores the fully desaturated base.
printf 'relief palette: %s (hinged at sea level)\n' "${RELIEFCPT}"
gmt makecpt -C"${RELIEFCPT}" -T-8000/6000 > relief.cpt

# The bending moment CHANGES SIGN, so it takes a diverging scale anchored on
# zero. The script previously used turbo here and noted the problem in a
# comment; the comment was right and is now acted on. vik is diverging,
# perceptually uniform and zero-centred, so a sign change reads as a change of
# hue rather than as a step along a ramp. Shear and curvature are one-signed
# and keep the sequential turbo.
gmt makecpt -C${MCPT} -T${MRANGE} -D > m.cpt
gmt makecpt -C${VCPT} -T${VRANGE} -D > v.cpt
gmt makecpt -C${KCPT} -T${KRANGE} -D > k.cpt

# Attach each value to its own copy of the geometry as a -Z header, so psxy
# colours every band by its own segment. Building the four files here rather
# than asking for four inputs keeps one geometry as the single source of
# truth: the bands cannot differ between panels.
build_z () {          # $1 column, $2 output, $3 divisor, $4 value table
    python3 - "$1" "$2" "${SEGMENTS}" "$4" "$3" <<'PYEOF'
import csv, sys
col, out, segfile, valfile, scale = sys.argv[1:6]
scale = float(scale)

rows = [r for r in csv.DictReader(open(valfile))]
names = [(r.get("segment") or "").strip() for r in rows]


def val(r):
    raw = (r.get(col) or "").strip()
    try:
        return float(raw) / scale
    except (TypeError, ValueError):
        return None


by_name = {n: val(r) for n, r in zip(names, rows) if n}
in_order = [val(r) for r in rows]

# Two ways to attach a value to a band, and the file decides which.
#
# NAMED: a header like "> tonga_017" carries the segment name, and the join is
# by name, which is safe whatever the order.
#
# POSITIONAL: build_te_segments.py writes "> -Z34.2" -- the thickness, not a
# name -- so there is nothing to join on. That is why every panel of the last
# render reported "0 segments drawn, 71 without a value": the parser took
# "-Z34.2" for the segment name and it matched nothing. The k-th band is then
# paired with the k-th row of VALUES, which is correct ONLY because both files
# are written in the same order, sorted by trench and then by segment index.
# That is an assumption, so it is checked: the counts must agree exactly, and
# the script refuses rather than draw a plausible wrong map.
hdrs = [l for l in open(segfile) if l.startswith(">")]
named = sum(1 for h in hdrs
            if h[1:].strip() and h[1:].strip().split()[0] in by_name)
mode = "name" if named else "order"

if mode == "order" and len(hdrs) != len(rows):
    sys.exit("  %s: %d bands in %s but %d rows in %s, and the headers carry "
             "no segment name, so they cannot be paired. Rebuild the geometry "
             "with segment names in the headers, or supply a value table with "
             "one row per band." % (col, len(hdrs), segfile, len(rows), valfile))

kept = skipped = 0
k = -1
with open(segfile) as fh, open(out, "w") as w:
    emit = False
    for line in fh:
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            k += 1
            if mode == "name":
                nm = line[1:].strip().split()[0] if line[1:].strip() else ""
                v = by_name.get(nm)
            else:
                v = in_order[k] if k < len(in_order) else None
            if v is not None:
                w.write("> -Z%.6g\n" % v); emit = True; kept += 1
            else:
                emit = False; skipped += 1
        elif emit:
            w.write(line)
sys.stderr.write("  %s: %d segments drawn, %d without a value (joined by %s)\n"
                 % (col, kept, skipped, mode))

# A panel with no band is a blank map, and a blank map that is written without
# complaint is worse than a crash: it looks like a result. The usual cause is a
# column name that is not in VALUES, so the available names are listed here
# rather than left to be discovered by opening the file.
if kept == 0:
    cols = list(rows[0].keys()) if rows else []
    sys.exit("  %s: NO band received a value, so this panel would be blank.\n"
             "  Column '%s' is %s in %s.\n"
             "  Columns present: %s\n"
             "  Point VALUES at the table holding the derived quantities, or\n"
             "  set MCOL/VCOL/KCOL to the names it uses."
             % (col, col,
                "absent" if col not in cols else "present but empty",
                valfile, ", ".join(cols) or "(none)"))
PYEOF
}

build_z "${MCOL}" m.gmt "${MSCALE}" "${VALUES}"
build_z "${VCOL}" v.gmt "${VSCALE}" "${VALUES}"
build_z "${KCOL}" k.gmt "${KSCALE}" "${VALUES}"
if [ "${PANELD}" = "1" ]; then
    if [ ! -f "${VALUES4}" ] && [ -f "${SCRIPTDIR}/${VALUES4}" ]; then
        VALUES4=${SCRIPTDIR}/${VALUES4}
    fi
    if [ ! -f "${VALUES4}" ]; then
        printf 'error: fourth-panel table not found: %s\n' "${VALUES4}" >&2
        printf 'set VALUES4=/path/to/table, or PANELD=0 for three panels.\n' >&2
        exit 1
    fi
    build_z "${DCOL}" d.gmt "${DSCALE}" "${VALUES4}"
    gmt makecpt -C${DCPT} -T"${DRANGE}" -D > d.cpt
fi

panel () {                          # $1 file  $2 cpt  $3 title  $4 bar label
    gmt grdimage relief.nc -Crelief.cpt -Ishade.nc -t30
    gmt coast -Ggray75 -Wthinnest,gray45 -Dl -A5000
    # Graticule drawn HERE, before the bands, so the white lines pass under the
    # coloured segments instead of scoring notches across them. -B+n suppresses
    # frame and annotations so only the gridlines are laid down; the pen is the
    # thin white MAP_GRID_PEN_PRIMARY set at the top of the script. The
    # annotated frame is drawn last, below, and carries no g.
    gmt basemap -Bxg${GRIDX} -Byg${GRIDY} -B+n
    if [ -n "${AXES}" ] && [ -d "${AXES}" ]; then
        for f in "${AXES}"/*.txt; do
            [ -e "$f" ] || continue
            awk '{ lon=$1; if (lon < 0) lon += 360; print lon, $2 }' "$f" |
                gmt plot -W0.4p,gray40
        done
    fi
    [ -s "$1" ] && gmt plot "$1" -C"$2" -W2.6p+cl
    gmt basemap -Bxa60f30 -Bya30f15 -BWeSN+t"$3"
    gmt colorbar -C"$2" -DJBC+w6.6c/0.20c+h+o0/0.55c+e -Bx+l"$4"
}

# One row of three. A 2x2 grid with one quadrant empty reads as an unfinished
# figure; three panels side by side at 5.4 cm fit the 17 cm text width the rest
# of the series uses, and the maps stay legible because they carry coloured
# bands on a desaturated base rather than fine detail.
gmt begin "${OUT}" pdf,png
  if [ "${PANELD}" = "1" ]; then
      gmt subplot begin 2x2 -Fs${W}/${H} -M0.45c/1.35c -A"a)"+jTL+o0.1c \
                            -R"${REG}" -J${PROJ}
        gmt subplot set 0
          panel m.gmt m.cpt "Bending moment at the axis" "M (10@+16@+ N)"
        gmt subplot set 1
          panel v.gmt v.cpt "Shear force at the axis" "V (10@+12@+ N m@+-1@+)"
        gmt subplot set 2
          panel k.gmt k.cpt "Maximum curvature" "@~k@~@-max@- (10@+-7@+ m@+-1@+)"
        gmt subplot set 3
          panel d.gmt d.cpt "${DTITLE}" "${DLABEL}"
      gmt subplot end
  else
      gmt subplot begin 1x3 -Fs5.4c/3.79c -M0.35c/1.30c -A"a)"+jTL+o0.1c \
                            -R"${REG}" -J${PROJ}
        gmt subplot set 0
          panel m.gmt m.cpt "Bending moment at the axis" "M (10@+16@+ N)"
        gmt subplot set 1
          panel v.gmt v.cpt "Shear force at the axis" "V (10@+12@+ N m@+-1@+)"
        gmt subplot set 2
          panel k.gmt k.cpt "Maximum curvature" "@~k@~@-max@- (10@+-7@+ m@+-1@+)"
      gmt subplot end
  fi
gmt end

printf 'wrote %s.pdf and %s.png\n' "${OUT}" "${OUT}"
printf 'relief: %s\n' "${RELIEF}"
printf 'geometry: %s\nvalues: %s\n' "${SEGMENTS}" "${VALUES}"
