# Example 1: site law and compliance

From a table of blast events to the site attenuation law, the maximum charge per delay and a verdict per event.
It uses synthetic data only (`datasets.synthetic_events`, seeded); nothing here comes from a real site.

```bash
python -m pip install ".[io,viz]"      # from the repository root
cd examples/01-site-law-and-compliance
python site_law.py
```

The script writes `events_template.csv`, `events.csv`, `site_law.png` and `compliance.png` in the current directory
and prints a short summary. It never opens a window.

| Step | Code | What it does |
|---|---|---|
| 1 | `bp.write_events_template` | Writes the blank table (headers and two invented rows) you fill with your campaign. |
| 2 | `datasets.synthetic_events` | Stands in for a campaign: PPV = K (D / W^0.5)^-beta with lognormal noise. Saved as CSV. |
| 3 | `bp.read_events` | Maps your column names to the canonical ones and converts to SI (m, kg, mm/s, Hz). |
| 4 | `attenuation.fit` | Log-log least squares. Gives K, beta, sigma and the upper bound `k_upper = K * 10^(2 sigma)`. |
| 5 | `attenuation.max_charge` | Largest charge per delay at 300 m that keeps the PPV under 12.7 mm/s, on the upper bound. Prints the extrapolation warnings, if any. |
| 6 | `criteria.evaluate` | One verdict per event against `USBM-RI8507-1980` / `appendix-b-drywall` (frequency curve) and `PE-MEM-1995` / `distance-table`. |
| 7 | `plotting` | `site_law.png` (events, median and upper bound) and `compliance.png` (events against the limit curve). |

The chart draws on matplotlib's current figure, so `plt.savefig` right after each call saves it.
A profile is a transcription of limit values with their citation; it never replaces the standard.
