"""Flyrock end to end with a synthetic hole table: model comparison -> governing hole -> clearance -> Roth.

Usage: python flyrock.py   (no files are written). The hole table is invented; k is a site constant you must supply.
"""

import pandas as pd

from blastingpy import flyrock as fr

# 1. Six synthetic holes, 165 mm. Hole 5 has a charge column shorter than 1 m (the actual mass is used).
holes = pd.DataFrame({
    "hole": ["H1", "H2", "H3", "H4", "H5", "H6"],
    "diameter_mm": 165.0,
    "burden_m": [4.5, 4.0, 3.5, 3.0, 4.0, 3.2],
    "stemming_m": [3.5, 3.0, 2.8, 2.2, 1.6, 2.5],
    "linear_charge_kg_m": fr.linear_charge(165.0, 850.0),  # full-hole geometry, 850 kg/m3
    "hole_angle_from_vertical_deg": [10.0, 10.0, 10.0, 10.0, 10.0, 0.0],
    "charge_length_m": [8.0, 8.0, 8.0, 8.0, 0.6, 8.0],
}).set_index("hole")

# 2. Per-hole ranges of each model. They are different models, not a consensus: nothing is averaged.
#    k = 27 is the published value for hard competent rock (13.5 for soft; the data range is 15-37).
table = fr.compare(holes, k=27.0)
print("Hole table (ranges in m):")
print(table[["burden_m", "stemming_m", "rm_charge", "rm_face_burst_m", "rm_cratering_m", "rm_stemming_ejection_m",
             "rm_max_m", "lundborg_max_m"]].round(1).to_string())
print()
for message in table.attrs["warnings"]:
    print("  warning:", message)

# 3. The hole that governs: the largest Richards & Moore range.
worst = table["rm_max_m"].idxmax()
range_m = float(table.loc[worst, "rm_max_m"])
print(f"Flyrock: hole {worst} governs, Richards & Moore maximum range {range_m:.0f} m "
      f"(Lundborg {table.loc[worst, 'lundborg_max_m']:.0f} m)")

# 4. Clearance. The models have no safety margin; clearance adds it (people x4, equipment x2) and, for people,
#    optionally the Peruvian minimum of 500 m. The result says which one governs.
for target in ("people", "equipment"):
    for regulation in (None, "PE-DS024-2016-EM"):
        c = fr.clearance(range_m, target=target, regulation=regulation)
        print(f"Clearance {target}, regulation={regulation}: {c.distance_m:.0f} m (governs: {c.governs}, "
              f"factor {c.factor:g})")
        for note in c.notes[1:]:
            print("  note:", note)
mild = float(table["rm_max_m"].min())  # a mild hole: 4 x its range is below 500 m, so the regulation governs
c = fr.clearance(mild, target="people", regulation="PE-DS024-2016-EM")
print(f"Clearance people for the mildest hole ({mild:.0f} m), regulation=PE-DS024-2016-EM: {c.distance_m:.0f} m "
      f"(governs: {c.governs})")
print()

# 5. Minimum stemming of the governing hole so the cratering range does not exceed 120 m.
m = float(table.loc[worst, "rm_charge"])
print(f"Minimum stemming for a 120 m cratering range, hole {worst}: {fr.rm_min_confinement(27.0, m, 120.0):.1f} m")

# 6. Roth for one hole: c/m, Gurney velocity with ANFO (detonation velocity 4000 m/s, invented), range at 45 degrees.
cm = fr.roth_charge_mass_ratio(float(holes.loc[worst, "linear_charge_kg_m"]), 2700.0, float(holes.loc[worst, "burden_m"]))
gurney = fr.roth_gurney_velocity(4000.0, fr.Explosive.Anfo)
v = fr.roth_launch_velocity(gurney, cm)
print(f"Roth, hole {worst}: c/m = {cm:.4f}, Gurney velocity {gurney:.0f} m/s, launch velocity {v:.0f} m/s, "
      f"maximum range {fr.max_range_level(v):.0f} m")
print()

# 7. The published critiques travel with the result.
print("Critiques:")
for text in table.attrs["critiques"]:
    print(" -", text)
