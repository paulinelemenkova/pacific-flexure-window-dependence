#!/usr/bin/env python3
"""name_segments.py -- put segment names into the headers of a multisegment file.

WHY THIS EXISTS
    build_te_segments.py writes headers as "> -Z34.2": the value, not a name.
    fig_parametermaps.sh can therefore only pair band k with row k of VALUES,
    which is correct ONLY when the two files have the same number of records in
    the same order. te_segments.gmt has one band per segment in the run, while
    derived_segments.csv holds only the segments that survive the residual
    screen, so the counts differ and the positional join is refused -- rightly,
    because pairing them by position would draw the wrong value on the wrong
    margin without any error.

    This script rewrites the headers as "> tonga_017 -Z34.2", taking the names
    from the run table in the order the bands were built. The -Z value is kept,
    so the file still works everywhere it worked before; the name is simply
    added in front of it. fig_parametermaps.sh then joins by name and quietly
    skips the segments that carry no derived value, which is what the figure
    caption already says it does.

USAGE
    python3 name_segments.py te_segments.gmt runs/r02/segments.csv \
            te_segments_named.gmt
"""
import csv
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
segfile, runfile, outfile = sys.argv[1:4]

rows = list(csv.DictReader(open(runfile)))
names = [(r.get("segment") or "").strip() for r in rows]
if not names or not any(names):
    sys.exit("error: %s has no 'segment' column" % runfile)

hdrs = [l for l in open(segfile) if l.startswith(">")]
if len(hdrs) != len(names):
    sys.exit("error: %d bands in %s but %d rows in %s.\n"
             "These two files must be built from the same run, in the same\n"
             "order, or the names cannot be attached."
             % (len(hdrs), segfile, len(names), runfile))

k = -1
written = 0
with open(segfile) as fh, open(outfile, "w") as w:
    for line in fh:
        if line.startswith("#"):
            w.write(line)
        elif line.startswith(">"):
            k += 1
            rest = line[1:].strip()
            w.write("> %s %s\n" % (names[k], rest) if rest
                    else "> %s\n" % names[k])
            written += 1
        else:
            w.write(line)

print("wrote %s: %d bands, headers now '> <segment> -Z<value>'"
      % (outfile, written))
