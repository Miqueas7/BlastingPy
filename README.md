# BlastingPy

Open-source Python library, with a C++20 core, to analyze the effects of blasting in mining: ground vibration,
airblast, flyrock and near-field vibration.

The central use case: a blasting engineer ends a monitoring campaign with a table of events (distance, maximum charge per
delay, PPV, frequency). BlastingPy fits the attenuation law of the site, computes the maximum charge per delay and
checks the result against a regulatory criterion, stating the edition, table, magnitude and frequency method used.

**Status:** in development, unpublished (version 0.1.0). Sprint 1 of 5 is complete: events table, attenuation law,
maximum charge and compliance verdict. The package is not on PyPI and the repository is private.

## Install (from the repository)

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

A C++ compiler and CMake are required to build from source. Extras: `io` (read and write `.xlsx`), `viz` (charts).

## Example

```python
import blastingpy as bp
from blastingpy import attenuation, criteria, plotting

events = bp.read_events("campaign.csv", units="metric", ppv_kind="component_max",
                        columns={"distance_m": "Distance", "charge_kg": "Charge", "ppv_mm_s": "PPV",
                                 "frequency_hz": "Freq"})
law = attenuation.fit(events, law="sqrt")                  # PPV = K (D / sqrt(W))^-beta
result = attenuation.max_charge(law, 300.0, 12.7)          # kg per delay at 300 m, PPV <= 12.7 mm/s
print(result.value, result.warnings)
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
print(verdict["status"].value_counts())
plotting.plot_site_law(law, events)
plotting.plot_compliance(events, "USBM-RI8507-1980", "appendix-b-drywall")
```

`write_events_template("events.csv")` writes the blank table. A complete, runnable version with synthetic data is in
[examples/01-site-law-and-compliance](examples/01-site-law-and-compliance/).

## Conventions

- **Units.** Inside the library everything is SI: m, kg, mm/s, Hz. `read_events(units="imperial")` (or a dict per
  quantity) converts at input; `blastingpy.units` has `to_si` and `from_si`.
- **Upper bound of the site law.** `k_upper = k * 10^(z * sigma)` with `z = 2` by default, as in USBM RI 8507 and OSMRE
  BIVDEP practice. For a one-sided 95 % bound use `attenuation.fit(events, z=attenuation.normal_quantile(0.95))`
  (z = 1.645). `max_charge` and `predict_ppv` use the upper bound by default; pass `attenuation.Level.Median` for
  the median.
- **Warnings.** Fits with fewer than 30 events, `beta <= 0` and predictions outside the fitted range of distance or
  charge come with warnings (`law.warnings`, `result.warnings`). A law with `beta <= 0` is refused for predictions.

## Site laws

| `law` | Form (SI) | Source | Mark |
|---|---|---|---|
| `"sqrt"` | PPV = K (D / W^0.5)^-beta | USBM Bulletin 656, eq. 4.20 and 6.1 | V1 |
| `"cbrt"` | PPV = K (D / W^(1/3))^-beta | USBM Bulletin 656, eq. 4.13 with alpha = 1/3 (reported there as other authors' proposal) | V2 |
| `"general"` | PPV = K W^a D^-beta | USBM Bulletin 656, eq. 4.10-4.13 | V1 |

## Profiles

`criteria.list_profiles()` returns the ids. Marks: **V1** values read in the primary source; **V2** values read in a
reliable secondary source, not in the standard.

| Profile | Criterion ids | Quantity | Mark |
|---|---|---|---|
| `USBM-RI8507-1980` | `appendix-b-drywall`, `appendix-b-plaster`, `table-13-drywall`, `table-13-plaster` | component maximum, frequency curve | V1 |
| `OSMRE-30CFR816.67` | `distance-table` | component maximum, distance table (scaled distance) | V1 |
| `PE-MEM-1995` | `distance-table` | component maximum, distance table (guide of Peru's MEM) | V1 |
| `BS-7385-2-1993` | `line-1`, `line-2` | component maximum, frequency curve | V2 |
| `DIN-4150-3-2016` | `line-2`, `line-3` (foundation only) | component maximum, frequency curve | V2 |

**A profile never replaces the standard.** It is a transcription of limit values with their citation. The two V2
profiles (BS 7385-2 and DIN 4150-3) were transcribed from secondary sources and must be checked against the purchased
standard before anyone relies on them. DIN 4150-3 values were read from a figure at labelled grid lines.

To see a profile's reservations, which the verdict table shows only as a mark:

```python
c = criteria.load_profile("DIN-4150-3-2016").criteria["line-2"]
print(c.verification, c.reference, c.note, sep="\n")
```

(The `note` texts of the DIN and BS criteria are currently in Spanish only; they will be made bilingual in a later change.)

## What the verdict means

`criteria.evaluate(events, profile, criterion)` returns one row per event with `status`, `limit_mm_s`, `margin_mm_s`,
`margin_pct`, `governing`, `verification` and `reason`. The four statuses:

- `complies`: every candidate value was evaluated and is at or below its limit.
- `exceeds`: at least one evaluated value is above its limit (the `reason` lists what could not be evaluated).
- `not_comparable`: the quantity the profile asks for is missing or incomplete, a value is bad data, or the
  frequency (or distance) the criterion needs is missing.
- `indeterminate`: a frequency or distance falls outside the criterion's domain.

"Complies" requires the quantity the profile asks for to be fully known. For component-maximum profiles that means the
three components (`ppv_t_mm_s`, `ppv_v_mm_s`, `ppv_l_mm_s`) or a peak declared with `ppv_kind="component_max"`. A
`vector_sum` PPV is never compared with a component limit. Accepted limitation: a `component_max` peak with its
frequency is taken as the event's maximum component; smaller components at lower frequencies that the event does not
carry are not checked.

## Your own profile

A user-defined profile is the way to apply a limit you have from a standard that is not included, or a limit set by a
permit or a contract. Pass it as a dict (or the path to a JSON file); its criteria get the mark `"user"`:

```python
mine = {"profile_id": "MY-LIMIT", "quantity": "component_max", "units": {"velocity": "mm/s"},
        "criteria": [{"id": "flat-10", "kind": "flat", "limit": 10.0, "reference": "Contract clause 4"}]}
criteria.evaluate(events, mine, "flat-10")
```

A profile's `quantity` must be one the events carry; otherwise every verdict is `not_comparable` and `reason` says what
is missing. Criterion kinds: `flat` (shown above), `frequency_curve` and `distance_table`; the shipped JSON files in
`python/blastingpy/profiles/` illustrate the last two.

## Not included yet

Laws Langefors-Kihlstrom, Ghosh-Daemen and CMRI, and the standards UNE 22381, AS 2187.2, BS 6472-2, SS 4604866,
SN 640312a and NBR 9653 are missing because no source with their formulas or values has been read. DIN 4150-3 line 1
and the upper-floor criteria, the OSMRE Figure 1 curve and BS 7385-2 line 1 below 4 Hz are also out. Details in
[docs/CHANGELOG.md](docs/CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
