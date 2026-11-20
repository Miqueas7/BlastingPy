"""Bench pattern end to end with an invented outline: burden methods -> layout -> V sequence -> charge -> DXF -> PPV.

Usage: python bench_pattern.py [output_directory]   (default: a temporary folder; the DXF is written there)
Needs the extra dxf (pip install "blastingpy[dxf]"). Outline, crest, rock, explosive and delays are all invented.
"""

import pathlib
import sys
import tempfile

import numpy as np

from blastingpy import attenuation, datasets, pattern
from blastingpy import nearfield as nf

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True)

# 1. The four burden methods side by side. They are different models, not a consensus: you choose.
#    165 mm holes, 10 m bench, 2700 kg/m3 rock, 850 kg/m3 ANFO, delayed initiation. Lopez Jimeno needs the UCS and Ash
#    the explosive class; Langefors-Kihlstrom is listed but not implemented (the secondary sources disagree).
methods = pattern.burden_methods(diameter_mm=165.0, bench_height_m=10.0, rock_density_kg_m3=2700.0,
                                 explosive_density_kg_m3=850.0, initiation="delayed", ucs_mpa=90.0,
                                 explosive_class="low_density")
print(methods[["burden_m", "spacing_m", "stemming_m", "subdrill_m", "verification"]].round(2).to_string())
konya = methods.loc["konya"]  # the only [V1] method here; the example takes it
burden, spacing = round(float(konya.burden_m), 1), round(float(konya.spacing_m), 1)
print(f"\nChosen: Konya, burden {burden} m, spacing {spacing} m (drilled-pattern frame)\n")

# 2. Invented outline (local coordinates, m) and the free-face crest along its front. The outline is the limit
#    of the rock to blast, so the last row ends up about one burden from its back.
outline = np.array([[0.0, 0.0], [48.0, 0.0], [48.0, 26.0], [0.0, 26.0]])
crest = np.array([[0.0, 0.0], [48.0, 0.0]])
holes = pattern.layout(outline, crest, burden_m=burden, spacing_m=spacing, arrangement="rectangular")
print(f"Layout: {len(holes)} holes in {holes.row.nunique()} rows, {int(holes.edge.sum())} edge holes")

# 3. V sequence: apex at the central hole of the front row. The delays are yours to give (here 17 ms and 42 ms).
holes = pattern.sequence(holes, scheme="v", hole_delay_ms=17.0, row_delay_ms=42.0)
print(f"Sequence: V, last hole fires at {holes.delay_ms.max():.0f} ms")

# 4. Charge per hole: column H + J - T, fully charged. Subdrill and stemming come from the chosen method.
holes = pattern.charge(holes, bench_height_m=10.0, subdrill_m=float(konya.subdrill_m),
                       stemming_m=float(konya.stemming_m), diameter_mm=165.0, explosive_density_kg_m3=850.0)
print(f"Charge: {holes.charge_kg.iloc[0]:.0f} kg per hole, {holes.charge_kg.sum() / 1000:.1f} t in all")

# 5. DXF: outline, crest, holes by row, labels and the tie-in. Needs the extra dxf.
dxf_path = out / "malla.dxf"
pattern.to_dxf(holes, dxf_path, outline=outline, crest=crest, diameter_mm=165.0)
print(f"DXF written: {dxf_path}")

# 6. PPV at an invented monitor with the same machinery as the near-field example: a synthetic seed and site law.
holes = holes.set_index("hole_id")[["x_m", "y_m", "charge_kg", "firing_time_s"]]
seed = (datasets.synthetic_record(seed=5, sample_rate_hz=2048.0, duration_s=0.5, pretrigger_s=0.0, frequency_hz=30.0,
                                  mic_pa=None), "vertical")
law = attenuation.fit(datasets.synthetic_events(seed=5, n=40, distance_range=(20.0, 400.0),
                                                charge_range=(20.0, 300.0)), law="sqrt")
blast = nf.simulate_blast(holes, (24.0, -150.0), seed, law=law, wave_velocity_m_s=3000.0)
print(f"Simulated PPV at the monitor: {blast.ppv_mm_s:.1f} mm/s; largest charge within 8 ms: "
      f"{blast.max_charge_8ms_kg:g} kg")
for message in blast.warnings:
    print("  warning:", message)
