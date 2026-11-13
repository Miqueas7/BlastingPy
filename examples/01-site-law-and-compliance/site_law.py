"""From a table of blast events to the site law and a compliance verdict, with synthetic data only."""

import matplotlib

matplotlib.use("Agg")  # write the charts to files, never open a window
import matplotlib.pyplot as plt

import blastingpy as bp
from blastingpy import attenuation, criteria, datasets, plotting

# 1. The blank template you would fill with your campaign (two invented rows).
bp.write_events_template("events_template.csv")

# 2. Synthetic events stand in for a campaign; save them as a CSV like the one you would have.
datasets.synthetic_events("sqrt", k=1000, beta=1.6, sigma=0.15, n=40, seed=1).to_csv("events.csv", index=False)

# 3. Read the table: map your column names to the canonical ones (SI units inside).
events = bp.read_events(
    "events.csv",
    columns={"distance_m": "distance_m", "charge_kg": "charge_kg", "ppv_mm_s": "ppv_mm_s",
             "ppv_kind": "ppv_kind", "frequency_hz": "frequency_hz"},
)

# 4. Fit the site law PPV = K (D / sqrt(W))^-beta; the upper bound is K * 10^(2 sigma).
law = attenuation.fit(events, law="sqrt")
print(f"Site law: K={law.k:.0f}, beta={law.beta:.2f}, sigma={law.sigma:.3f}, n={law.n}, upper K={law.k_upper:.0f}")

# 5. Maximum charge per delay at 300 m for a 12.7 mm/s (0.5 in/s) limit, on the upper bound.
result = attenuation.max_charge(law, 300.0, 12.7)
print(f"Max charge per delay at 300 m for 12.7 mm/s: {result.value:.1f} kg")
for message in result.warnings:
    print("  warning:", message)

# 6. Verdict of every event against two profiles; a profile never replaces the standard itself.
for profile, criterion in (("USBM-RI8507-1980", "appendix-b-drywall"), ("PE-MEM-1995", "distance-table")):
    verdict = criteria.evaluate(events, profile, criterion)
    print(f"{profile} / {criterion}: {verdict['status'].value_counts().to_dict()}")

# 7. The two standard charts.
plotting.plot_site_law(law, events)
plt.savefig("site_law.png", dpi=120, bbox_inches="tight")
plotting.plot_compliance(events, "USBM-RI8507-1980", "appendix-b-drywall")
plt.savefig("compliance.png", dpi=120, bbox_inches="tight")
