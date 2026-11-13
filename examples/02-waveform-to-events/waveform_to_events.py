"""From seismograph records to the site law and a compliance verdict, with synthetic data only.

Usage: python waveform_to_events.py [output_directory]   (default: the current directory)
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # write the chart to a file, never open a window
import matplotlib.pyplot as plt

from blastingpy import attenuation, criteria, datasets, io, plotting, waveform

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out.mkdir(parents=True, exist_ok=True)

# 1. Six blasts at different distances and charges. Each PPV follows an invented site law
#    PPV = 1000 (D / sqrt(W))^-1.6 with a little scatter; the synthetic record stands in for the seismograph.
distances_m = [100.0, 150.0, 200.0, 300.0, 400.0, 500.0]
charges_kg = [20.0, 50.0, 30.0, 80.0, 60.0, 100.0]
scatter = [1.10, 0.92, 1.05, 0.90, 1.08, 0.95]
records = []
for i, (d, w, s) in enumerate(zip(distances_m, charges_kg, scatter), start=1):
    ppv = 1000.0 * (d / w ** 0.5) ** -1.6 * s
    record = datasets.synthetic_record(seed=i, ppv_mm_s=ppv, frequency_hz=20.0, quantize=True)

    # 2. Save it as the minimal Instantel ASCII export: a row of channel names and one row per sample.
    path = out / f"record_{i}.txt"
    cols = [record.channels[c] for c in ("transverse", "vertical", "longitudinal", "mic")]
    with open(path, "w") as f:
        f.write("Tran\tVert\tLong\tMicL\n")
        for row in zip(*cols):
            f.write("\t".join(f"{v:.5f}" for v in row) + "\n")

    # 3. Read it back. The file has no sample rate or pretrigger, so you give them; units come from the resolution.
    records.append(io.read_instantel_ascii(path, sample_rate_hz=1024, pretrigger_s=0.25))

# 4. Peak, vector sum and frequency of the first record.
summary = waveform.analyze(records[0])
peak = max(summary.channels[c].peak for c in ("transverse", "vertical", "longitudinal"))
print(f"PPV {peak:.2f} mm/s, vector sum {summary.pvs_mm_s:.2f} mm/s, airblast {summary.mic_db_l:.1f} dB(L)")

# 5. One row of the events table per record: the largest component and its zero-crossing frequency.
events = waveform.events_from_records(records, distance_m=distances_m, charge_kg=charges_kg)
print(f"Events table ({len(events)} events):")
print(events[["distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz"]].round(2).to_string(index=False))
print()

# 6. Site law and verdict, as in example 1. Six events fit but warn (fewer than 30).
law = attenuation.fit(events, law="sqrt")
print(f"Site law: K={law.k:.0f}, beta={law.beta:.2f}, sigma={law.sigma:.3f}, n={law.n}")
for message in law.warnings:
    print("  warning:", message)
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
print(f"USBM-RI8507-1980 / appendix-b-drywall: {verdict['status'].value_counts().to_dict()}")

# 7. The first record as a chart.
plotting.plot_record(records[0])
plt.savefig(out / "record.png", dpi=120, bbox_inches="tight")
