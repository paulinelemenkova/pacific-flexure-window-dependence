#!/usr/bin/env python3
"""Assemble the full-length axis of every trench on the roster from PB2002.

Each trench is one or more PB2002 boundary records, concatenated end to end in
along-strike order and resampled at uniform geodesic arc length.  The mapping
below was reconstructed from, and reproduces, the L column of Table 1: every
trench that already carried a length matches it to within rounding.
"""
import math, re, csv, collections

SRC = "/mnt/skills/user/cartographic-processing/assets/geology/PB2002_boundaries.gmt"
R = 6371.0
STEP = 5.0          # resampling interval, km

# trench -> list of (PB2002 pair, record index).  Records are given in the
# order in which they are to be concatenated along strike.
MAP = collections.OrderedDict([
    ("Aleutian",        [("NA/PA", 0)]),
    ("Kuril-Kamchatka", [("OK/PA", 0)]),
    ("Japan",           [("PA\\OK", 0)]),
    ("Izu-Bonin",       [("PS/PA", 1)]),
    ("Mariana",         [("MA/PA", 0)]),
    ("Yap",             [("PS/PA", 0)]),
    ("Palau",           [("CL-PS", 0)]),
    ("Ryukyu",          [("ON/PS", 0), ("ON/PS", 1)]),
    ("Manila",          [("PS/SU", 0)]),
    ("Philippine",      [("PS\\SU", 0)]),
    ("New Britain",     [("SB/SS", 0)]),
    ("San Cristobal",   [("PA/AU", 0)]),
    ("Vanuatu",         [("NH/AU", 0)]),
    ("Tonga",           [("TO/PA", 0)]),
    ("Kermadec",        [("KE/PA", 0)]),          # split with Hikurangi below
    ("Hikurangi",       [("KE/PA", 0)]),          # split with Kermadec below
    ("Puysegur",        [("PA/AU", 1), ("AU\\PA", 0)]),
    ("Cascadia",        [("JF\\NA", 0)]),         # MISSING from the analysis run
    ("Middle America",  [("CO\\NA", 0), ("CA/CO", 0)]),
    ("Peru-Chile",      [("NZ\\SA", 0), ("NZ\\SA", 1), ("NZ\\ND", 0), ("NZ\\AP", 0)]),
])
# KE/PA is one record spanning Kermadec and Hikurangi; Table 1 splits it
# 1590 / 716 km.  The split is applied at that arc length from the north end.
KEPA_SPLIT_KM = 1590.0

TABLE_L = {"Aleutian":3772,"Kuril-Kamchatka":2223,"Japan":793,"Izu-Bonin":1128,
           "Mariana":1867,"Yap":691,"Palau":848,"Ryukyu":1125,"Manila":610,
           "Philippine":1634,"New Britain":655,"San Cristobal":1386,
           "Vanuatu":1351,"Tonga":1126,"Kermadec":1590,"Hikurangi":716,
           "Puysegur":834,"Cascadia":None,"Middle America":2256,"Peru-Chile":6344}


def read_pb2002(path):
    num = r'[+-]?\d+\.\d+E[+-]\d+'
    pat = re.compile(rf'^\s*({num})\s*,\s*({num})\s*$')
    hdr = re.compile(r'^([A-Z]{2}[\\/-][A-Z]{2})\b')
    pairs, cur = collections.OrderedDict(), None
    for raw in open(path, encoding="latin-1"):
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        m = pat.match(line)
        if m and cur is not None:
            cur.append((float(m.group(1)), float(m.group(2))))
            continue
        h = hdr.match(line.strip())
        if h:
            pairs.setdefault(h.group(1), []).append([])
            cur = pairs[h.group(1)][-1]
        elif "end of line segment" in line:
            cur = None
    return pairs


def gc(p, q):
    (x1, y1), (x2, y2) = p, q
    a, b = math.radians(y1), math.radians(y2)
    return R * math.acos(max(-1.0, min(1.0,
        math.sin(a)*math.sin(b) + math.cos(a)*math.cos(b)*math.cos(math.radians(x2-x1)))))


def length(pts):
    return sum(gc(p, q) for p, q in zip(pts, pts[1:]))


def join(records):
    """Chain records end to end, greedily taking the nearest free endpoint.

    PB2002 stores the records of one margin in no particular order and with no
    consistent sense, so a naive concatenation inserts a spurious leg between
    the tail of one record and the head of the next.  On Peru-Chile that
    inflated the axis by 3273 km.  Each step therefore tests all four
    endpoint pairings and appends or prepends whichever is closest.
    """
    out, rest = list(records[0]), [list(r) for r in records[1:]]
    while rest:
        best = None
        for j, rec in enumerate(rest):
            for at_tail, pt, rev in ((True, rec[0], False), (True, rec[-1], True),
                                     (False, rec[-1], False), (False, rec[0], True)):
                d = gc(out[-1] if at_tail else out[0], pt)
                if best is None or d < best[0]:
                    best = (d, j, at_tail, rev)
        _, j, at_tail, rev = best
        rec = rest.pop(j)
        if rev:
            rec.reverse()
        out = out + rec if at_tail else rec + out
    return out


def cut(pts, s0, s1):
    """Return the part of the polyline between arc lengths s0 and s1.

    The two cut points are interpolated inside the straddling leg, so that
    Kermadec and Hikurangi partition KE/PA exactly rather than both keeping
    the whole leg that contains the split.
    """
    def at(s):
        t = 0.0
        for p, q in zip(pts, pts[1:]):
            d = gc(p, q)
            if t + d >= s:
                f = 0.0 if d == 0 else (s - t) / d
                return (p[0] + f*(q[0]-p[0]), p[1] + f*(q[1]-p[1]))
            t += d
        return pts[-1]
    total = length(pts)
    s0, s1 = max(0.0, s0), min(total, s1)
    out, t = [at(s0)], 0.0
    for p, q in zip(pts, pts[1:]):
        d = gc(p, q)
        if s0 < t + d and t < s1:
            if t > s0:
                out.append(p)
        t += d
    out.append(at(s1))
    return out


def unwrap(pts):
    """Make the longitude sequence continuous across the antimeridian.

    resample() interpolates linearly in longitude, so a leg running from
    +179.9 to -179.9 would be expanded into a sweep right across the globe.
    Unwrapping first keeps the leg 0.2 degrees wide.  Aleutian, Kermadec and
    Hikurangi all cross.
    """
    out = [pts[0]]
    for x, y in pts[1:]:
        px = out[-1][0]
        while x - px > 180.0:
            x -= 360.0
        while x - px < -180.0:
            x += 360.0
        out.append((x, y))
    return out


def wrap360(pts):
    """Emit 0-360 for a margin that crosses the antimeridian, else -180/180."""
    lo = [p[0] for p in pts]
    if min(lo) < -180.0 or max(lo) > 180.0 or (min(lo) < 0.0 < max(lo)
                                               and max(lo) - min(lo) > 180.0):
        return [(x % 360.0, y) for x, y in pts]
    return pts


def resample(pts, step):
    out, carry = [pts[0]], 0.0
    for p, q in zip(pts, pts[1:]):
        d = gc(p, q)
        if d == 0:
            continue
        t = step - carry
        while t <= d:
            f = t / d
            out.append((p[0] + f*(q[0]-p[0]), p[1] + f*(q[1]-p[1])))
            t += step
        carry = (carry + d) % step
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def main():
    pairs = read_pb2002(SRC)
    gmt = open("trench_axes_full.gmt", "w")
    gmt.write("# Full-length trench axes assembled from the PB2002 model of Bird (2003).\n"
              "# One multisegment record per trench, resampled at %.0f km.\n"
              "# Columns: longitude latitude\n" % STEP)
    rows = []
    for name, recs in MAP.items():
        pts = unwrap(join([pairs[k][i] for k, i in recs]))
        # KE/PA is digitised south to north; trenches.conf puts Kermadec on
        # the NORTHERN part (-37.0 to -23.6) and Hikurangi on the southern
        # (-41.5 to -37.0). Orient the polyline north-first before cutting,
        # or the two names come out swapped.
        if name in ("Kermadec", "Hikurangi") and pts[0][1] < pts[-1][1]:
            pts.reverse()
        if name == "Kermadec":
            pts = cut(pts, 0.0, KEPA_SPLIT_KM)
        elif name == "Hikurangi":
            pts = cut(pts, KEPA_SPLIT_KM, 1e9)
        pts = unwrap(pts)
        L = length(pts)
        pts = wrap360(resample(pts, STEP))
        src = "+".join(f"{k}[{i}]" for k, i in recs)
        gmt.write(f"> -L\"{name}\" # {src}\n")
        for x, y in pts:
            gmt.write(f"{x:.5f} {y:.5f}\n")
        ref = TABLE_L[name]
        rows.append((name, src, L, ref, None if ref is None else L - ref,
                     max(1, round(L / 148.0))))
    gmt.close()

    with open("trench_axes_full.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trench", "pb2002_records", "L_full_km", "L_table1_km",
                    "difference_km", "N_bins_at_148km"])
        for name, src, L, ref, dif, n in rows:
            w.writerow([name, src, f"{L:.0f}", "" if ref is None else ref,
                        "" if dif is None else f"{dif:+.0f}", n])

    print(f"{'trench':17s} {'full L':>8s} {'Table 1':>8s} {'diff':>7s} {'bins':>5s}  records")
    tot = totn = 0
    for name, src, L, ref, dif, n in rows:
        tot += L; totn += n
        print(f"{name:17s} {L:8.0f} {('-' if ref is None else ref):>8} "
              f"{('new' if dif is None else f'{dif:+.0f}'):>7} {n:5d}  {src}")
    print(f"{'TOTAL':17s} {tot:8.0f} {'':>8} {'':>7} {totn:5d}")


if __name__ == "__main__":
    main()
