#!/usr/bin/env python3
"""
Join auxiliary.txt onto morphometry_classified.csv.

build_dataframe.py builds the morphometric table from the profiles only; the
sampled grid values in auxiliary.txt are never merged in. This script does that
join and writes the result back, so that statistics.py has something to test.

Usage:
    python3 merge_auxiliary.py                      # uses the default names
    python3 merge_auxiliary.py auxiliary.txt morphometry_classified.csv

It prints what it found in auxiliary.txt before merging anything. Read that
output: if the column guess is wrong, fix the NAMES_BY_WIDTH table below rather
than letting a silent mis-assignment through.
"""
import sys, os
import numpy as np
import pandas as pd

AUX  = sys.argv[1] if len(sys.argv) > 1 else "auxiliary.txt"
MORF = sys.argv[2] if len(sys.argv) > 2 else "morphometry_classified.csv"
OUT  = MORF  # written back in place; a .bak copy is kept

# Expected layout of auxiliary.txt, keyed by number of columns.
# sample_auxiliary.sh emits one row per profile location.
NAMES_BY_WIDTH = {
    4: ["trench", "profile", "age_Ma", "sed_m"],
    5: ["trench", "profile", "lon", "lat", "age_Ma"],
    6: ["trench", "profile", "lon", "lat", "age_Ma", "sed_m"],
    7: ["trench", "profile", "lon", "lat", "age_Ma", "sed_m", "conv"],
}

def norm(x):
    return "".join(c for c in str(x).lower() if c.isalnum())

# ------------------------------------------------------------------ read aux
if not os.path.exists(AUX):
    sys.exit(f"{AUX} not found")

raw = pd.read_csv(AUX, sep=r"\s+", header=None, engine="python",
                  comment="#", na_values=["NaN", "nan", "NA", "-", ""])

# drop a text header row if sample_auxiliary.sh wrote one
if raw.shape[0] and not str(raw.iloc[0, 1]).replace(".", "").replace("-", "").isdigit():
    try:
        float(raw.iloc[0, 1])
    except ValueError:
        raw = raw.iloc[1:].reset_index(drop=True)

w = raw.shape[1]
print(f"{AUX}: {len(raw)} rows x {w} columns")
print(raw.head(3).to_string(index=False, header=False))

if w not in NAMES_BY_WIDTH:
    sys.exit(f"\nunexpected width {w}. Add an entry for it to NAMES_BY_WIDTH "
             f"and re-run, or send me the three rows printed above.")
raw.columns = NAMES_BY_WIDTH[w]
print(f"\nread as: {list(raw.columns)}")

for c in raw.columns:
    if c != "trench":
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

aux_vars = [c for c in ("age_Ma", "sed_m", "conv") if c in raw.columns]
print("auxiliary variables present:", aux_vars or "NONE")
for c in aux_vars:
    v = raw[c].dropna()
    print(f"  {c}: {len(v)}/{len(raw)} non-null, range {v.min():.4g} to {v.max():.4g}")

# ----------------------------------------------------------------- read morf
m = pd.read_csv(MORF)
print(f"\n{MORF}: {len(m)} rows, columns {list(m.columns)}")

# ------------------------------------------------------------------- the join
raw["_k"] = raw["trench"].map(norm)
m["_k"]   = m["trench"].map(norm)

unmatched = set(m["_k"]) - set(raw["_k"])
if unmatched:
    print("\n!! trench keys in the morphometry table with no match in auxiliary.txt:")
    print("  ", sorted(unmatched))
    print("   auxiliary.txt has:", sorted(set(raw['_k'])))

merged = m.merge(raw[["_k", "profile"] + aux_vars], on=["_k", "profile"], how="left")

hit = merged[aux_vars].notna().all(axis=1).sum() if aux_vars else 0
print(f"\nmatched on (trench, profile): {hit}/{len(m)}")

# fallback: nearest sampled location, if the profile numbering disagrees
if hit < 0.5 * len(m) and {"lon", "lat"}.issubset(raw.columns) and {"lon", "lat"}.issubset(m.columns):
    print("poor match on profile index -- falling back to nearest sampled location")
    merged = m.drop(columns=[c for c in aux_vars if c in m.columns]).copy()
    for c in aux_vars:
        merged[c] = np.nan
    for k, gm in merged.groupby("_k"):
        ga = raw[raw["_k"] == k]
        if ga.empty:
            continue
        d = ((gm["lon"].values[:, None] - ga["lon"].values[None, :]) ** 2 +
             (gm["lat"].values[:, None] - ga["lat"].values[None, :]) ** 2)
        j = d.argmin(axis=1)
        near = np.sqrt(d.min(axis=1)) < 0.15          # ~15 km
        for c in aux_vars:
            vals = ga[c].values[j]
            vals[~near] = np.nan
            merged.loc[gm.index, c] = vals
    hit = merged[aux_vars].notna().all(axis=1).sum()
    print(f"matched on coordinates: {hit}/{len(m)}")

merged = merged.drop(columns=["_k"])

print("\nfinal non-null counts:")
for c in aux_vars:
    print(f"  {c}: {merged[c].notna().sum()}/{len(merged)}")

if hit == 0:
    sys.exit("\nnothing matched -- do not overwrite. Send me the printed rows above.")

if os.path.exists(OUT):
    pd.read_csv(OUT).to_csv(OUT + ".bak", index=False)
merged.to_csv(OUT, index=False)
print(f"\nwrote {OUT} ({len(merged)} rows, {len(merged.columns)} columns); "
      f"previous version kept as {OUT}.bak")
print("next: python3 statistics.py " + OUT)
