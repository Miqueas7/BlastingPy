"""Near field end to end with synthetic data: Holmberg-Persson -> Heelan column -> whole blast and its Monte Carlo.

Usage: python nearfield.py [output_directory]   (default: the current directory)

Every number below is invented (rock, explosive, hole table, seed waveform, site law); nothing comes from a real site.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # write the chart to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from blastingpy import attenuation, datasets
from blastingpy import nearfield as nf

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

# 1. Holmberg-Persson around a 3 m charge of 1 kg/m on the z axis. K, alpha and beta are required: here the constants
#    that NIOSH reports for surface blasts, passed explicitly. They are data, not a default; fit your own.
hp = nf.HOLMBERG_PERSSON_SURFACE
charge = dict(k=hp["k"], alpha=hp["alpha"], beta=hp["beta"], linear_charge_kg_m=1.0, z_start=-1.5, z_end=1.5)
print(f"Holmberg-Persson constants: K = {hp['k']:g}, alpha = {hp['alpha']:g}, beta = {hp['beta']:g} "
      f"[{hp['verification']}], {hp['citation']}")
for note in hp["notes"]:
    print("  note:", note)
for r in (0.5, 1.0, 2.0):  # at mid-charge (z = 0)
    print(f"Holmberg-Persson at r = {r:g} m: original {nf.hp_original(**charge, r=r, z=0.0):.0f} mm/s, "
          f"NIOSH modified {nf.hp_niosh(**charge, r=r, z=0.0):.0f} mm/s")

# The critical PPV of an invented rock: T = 10 MPa, Vp = 5000 m/s, static E = 50 GPa.
critical = nf.ppv_critical(10e6, 5000.0, 50e9)
extent = {name: nf.hp_damage_extent(model, **charge, z=0.0, ppv_limit_mm_s=critical)
          for name, model in (("original", nf.HpModel.Original), ("NIOSH modified", nf.HpModel.Niosh))}
print(f"Holmberg-Persson damage extent at mid-charge for a critical PPV of {critical:.0f} mm/s (T Vp / E): "
      + ", ".join(f"{name} {metres:.2f} m" for name, metres in extent.items()))

# The two contours side by side, with the critical PPV drawn on each. Both are static: no detonation velocity.
r, z = np.linspace(0.1, 4.0, 40), np.linspace(-3.5, 3.5, 71)
figure, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
for axis, model in zip(axes, ("original", "niosh")):
    ppv = nf.hp_grid(model, **charge, r=r, z=z)
    filled = axis.contourf(r, z, np.log10(ppv), levels=np.linspace(2.0, 4.0, 11), extend="both")
    axis.contour(r, z, ppv, levels=[critical], colors="k")
    axis.plot([0.0, 0.0], [-1.5, 1.5], "r", linewidth=4)  # the charge
    axis.set(title=f"Holmberg-Persson, {model}", xlabel="r (m)")
axes[0].set_ylabel("z (m)")
figure.colorbar(filled, ax=axes, label="log10 PPV (mm/s); black line: critical PPV")
figure.savefig(out / "holmberg_persson.png", dpi=120, bbox_inches="tight")
print()

# 2. Heelan around a 6 m column primed at the bottom (element 1 of 41), linear (A = 1) and non-linear (A = 0.7).
#    Invented source and rock: von Neumann pressure 5 GPa, n = 6, decay 8000 1/s; Vp 5000, Vs 2900 m/s, 2700 kg/m3.
source = nf.PressureSource(vn_pressure_pa=5e9, n=6, gamma_per_s=8000.0)
medium = nf.Medium(p_velocity_m_s=5000.0, s_velocity_m_s=2900.0, density_kg_m3=2700.0)
for exponent in (1.0, 0.7):
    column = nf.Column(hole_radius_m=0.05, z_bottom=0.0, z_top=6.0, elements=41, primer_element=1, vod_m_s=4500.0,
                       charge_exponent=exponent)
    peaks = {z_m: nf.heelan_ppv(source, medium, column, 5.0, z_m) for z_m in (-2.0, 3.0, 8.0)}  # the automatic grid
    print(f"Heelan, A = {exponent:g}, vector PPV at r = 5 m: "
          + ", ".join(f"z = {z_m:g} m: {p.ppv_mm_s:.0f} mm/s" for z_m, p in peaks.items())
          + f" (Omega_A = {peaks[3.0].scaled_frequency:.3g})")
    for message in peaks[3.0].warnings:
        print("  warning:", message)
print("Heelan: with bottom priming the peak is larger above the column than below it; the detonation velocity "
      "matters, which Holmberg-Persson cannot show.")
print()

# 3. A whole blast: 12 holes of 80 kg in two rows, 5 m apart, 17 ms between holes and 42 ms between rows.
holes = pd.DataFrame({"x_m": [5.0 * i for i in range(6)] * 2, "y_m": [0.0] * 6 + [4.0] * 6, "charge_kg": 80.0,
                      "firing_time_s": [0.017 * i for i in range(6)] + [0.042 + 0.017 * i for i in range(6)]},
                     index=[f"H{i}" for i in range(1, 13)])
monitor = (12.5, 120.0)
# The seed stands in for the record of a single-hole shot (one component, no pretrigger); the site law is fitted to
# synthetic events whose distances and charges cover the blast.
seed = (datasets.synthetic_record(seed=5, sample_rate_hz=2048.0, duration_s=0.5, pretrigger_s=0.0, frequency_hz=30.0,
                                  mic_pa=None), "vertical")
law = attenuation.fit(datasets.synthetic_events(seed=5, n=40, distance_range=(20.0, 400.0),
                                                charge_range=(20.0, 300.0)), law="sqrt")
setup = dict(law=law, wave_velocity_m_s=3000.0)
blast = nf.simulate_blast(holes, monitor, seed, **setup)
print(f"Whole blast (site law K = {law.k:.0f}, beta = {law.beta:.2f}, median amplitudes; vertical component):")
print(blast.arrivals.round(4).to_string())
print(f"Simulated PPV, linear superposition: {blast.ppv_mm_s:.1f} mm/s; largest charge within 8 ms: "
      f"{blast.max_charge_8ms_kg:g} kg")
for weighting in nf.WEIGHTINGS[1:]:  # Blair's (2008) non-linear weights need the mean spacing
    weighted = nf.simulate_blast(holes, monitor, seed, weighting=weighting, spacing_m=5.0, **setup)
    print(f"Simulated PPV, weighting={weighting!r}: {weighted.ppv_mm_s:.1f} mm/s")
    for message in weighted.warnings:
        print("  warning:", message)
print()

# 4. Monte Carlo: a delay scatter of 2 ms (yours to give: the source publishes none) and 25 % on the amplitude.
level = 40.0  # mm/s, an invented level
monte_carlo = nf.monte_carlo_blast(holes, monitor, seed, delay_sigma_s=0.002, amplitude_cv=0.25, simulations=2000,
                                   rng_seed=1, **setup)
print(f"Monte Carlo, {monte_carlo.peaks_mm_s.size} simulations: mean {monte_carlo.mean:.1f} mm/s, standard deviation "
      f"{monte_carlo.std:.1f} mm/s, 95th percentile {monte_carlo.percentile(95):.1f} mm/s "
      f"(without scatter: {monte_carlo.deterministic_ppv_mm_s:.1f} mm/s)")
print(f"Monte Carlo probability of exceeding {level:g} mm/s: {monte_carlo.probability_exceeding(level):.1%}")
for message in monte_carlo.warnings:
    print("  warning:", message)
print()

# 5. Delay sweep of Hinzen (1988): the peak of two seeds against the delay between them. It belongs to this seed.
sweep = nf.delay_sweep(seed, delays_s=np.arange(0.0, 0.051, 0.005))
print("Delay sweep (peak of two holes over the peak of both fired together):")
print(sweep.round(3).to_string(index=False))
