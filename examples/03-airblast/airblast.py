"""Airblast end to end with synthetic data only: records -> site law -> limit -> published equation.

Usage: python airblast.py   (no files are written)
"""

import numpy as np

from blastingpy import airblast, datasets, waveform

# 1. Eight blasts at different distances and charges. The microphone peak follows an invented cube-root law
#    P = 20000 (D / W^(1/3))^-1.2 Pa with mild scatter (seeded); the PPV is any plausible value.
distances_m = [150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0]
charges_kg = [40.0, 80.0, 60.0, 120.0, 100.0, 200.0, 150.0, 250.0]
rng = np.random.default_rng(3)
records = []
for i, (d, w) in enumerate(zip(distances_m, charges_kg), start=1):
    peak_pa = 20000.0 * (d / w ** (1 / 3)) ** -1.2 * 10 ** rng.normal(0.0, 0.03)
    records.append(datasets.synthetic_record(seed=i, ppv_mm_s=5.0, mic_pa=peak_pa))

# 2. One events row per record; the microphone fills airblast_pa and airblast_freq_hz.
events = waveform.events_from_records(records, distance_m=distances_m, charge_kg=charges_kg)
print(f"Events table ({len(events)} events):")
print(events[["distance_m", "charge_kg", "airblast_pa", "airblast_freq_hz"]].round(2).to_string(index=False))
print()

# 3. Site law of the overpressure: the same regression as the PPV law, with the cube-root form.
law = airblast.fit(events)
print(f"Airblast site law: K={law.k:.0f}, beta={law.beta:.2f}, sigma={law.sigma:.3f}, n={law.n}")
for message in law.warnings:
    print("  warning:", message)

# 4. Prediction and maximum charge at a receiver 400 m away, against the 133 dB limit (2 Hz system).
limit = airblast.limit_db("OSMRE-30CFR816.67", system="2Hz")
p = airblast.predict(law, 400.0, 100.0)
w = airblast.max_charge(law, 400.0, limit.pressure_pa)
print(f"Prediction at 400 m, 100 kg: {p.value:.1f} Pa ({airblast.to_db(p.value):.1f} dB), upper bound")
print(f"Max charge per delay at 400 m for {limit.level_db:.0f} dB: {w.value:.0f} kg")

# 5. Verdict per event. The measuring system is required: a linear microphone with a 2 Hz cutoff, typical of a
#    blasting seismograph, is "2Hz" (airblast.system_from_cutoff(2.0) says so).
verdict = airblast.evaluate(events, "OSMRE-30CFR816.67", system="2Hz")
print(f"Limit {limit.level_db:.0f} dB (OSMRE-30CFR816.67, 2Hz): {verdict['status'].value_counts().to_dict()}")
print(verdict[["distance_m", "charge_kg", "airblast_db", "margin_db", "status"]].round(1).to_string(index=False))
print()

# 6. A published equation (median, fitted with a 0.1 Hz system) against the same 2 Hz limit: the library warns.
eq = "USBM-RI8485/quarry/0.1Hz"
out = airblast.evaluate_prediction(eq, 400.0, 100.0, "OSMRE-30CFR816.67", system="2Hz")
print(f"{eq}: {out['level_db']:.1f} dB, margin {out['margin_db']:.1f} dB, {out['status']}")
for message in out["warnings"]:
    print("  warning:", message)
