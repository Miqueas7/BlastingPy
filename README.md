# BlastingPy

Open-source Python library, with a C++20 core, to analyze the effects of blasting in mining: ground vibration,
airblast, flyrock and near-field vibration.

The central use case: a blasting engineer ends a monitoring campaign with a table of events (distance, maximum charge per
delay, PPV, frequency). BlastingPy fits the attenuation law of the site, computes the maximum charge per delay and
checks the result against a regulatory criterion, stating the edition, table, magnitude and frequency method used.

**Status:** in development, unpublished (version 0.1.0). Sprints 1 and 2 of 5 are complete: events table, attenuation law,
maximum charge and compliance verdict, and reading seismograph records (waveform, PPV, vector sum, frequency). The package is not on PyPI and the repository is private.

## Install (from the repository)

```bash
python -m pip install ".[test,io,viz]"
python -m pytest tests/python -q
```

A C++ compiler and CMake are required to build from source. Extras: `io` (read and write `.xlsx`), `viz` (charts), `seismic` (ObsPy readers).

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

## From a seismograph record

`blastingpy.io` reads the waveform a seismograph exports; `blastingpy.waveform` turns it into peaks, vector sum and
frequency, and into a row of the events table of the sections above. The facts and their evidence are in
[docs/investigacion/FUENTES_SPRINT2.md](docs/investigacion/FUENTES_SPRINT2.md).

```python
from blastingpy import attenuation, criteria, io, waveform

records = [io.read_instantel_ascii(p, sample_rate_hz=1024, pretrigger_s=0.25) for p in paths]
events = waveform.events_from_records(records, distance_m=[100, 150, 200], charge_kg=[20, 50, 30])
law = attenuation.fit(events, law="sqrt")
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
```

A complete, runnable version with synthetic data is in [examples/02-waveform-to-events](examples/02-waveform-to-events/).

| Reader | `format_evidence` | Notes |
|---|---|---|
| `read_instantel_ascii`, minimal (`Tran Vert Long MicL` and rows) | `real-file` | No metadata: pass `sample_rate_hz` (and `pretrigger_s`). |
| `read_instantel_ascii`, THOR | `third-party-parser` | **Not validated against a real file**: built from a third-party parser's description. |
| `read_instantel_ascii`, Blastware with header (table or blocks) | `manual-only` | **Not validated against a real file**: built from the manual; the record carries a warning. Only the default separators: header in double quotes, block values separated by commas. Header units are a hint: `units=` wins, with a warning. |
| `read_instantel_histogram` | `manual-only` | **Not validated against a real file**: built from the manual; returns a DataFrame with a warning in `attrs`. |
| `read_waveform_csv` | `user` | Any mark or renamed channels (`columns=`, `units=`, `sep="\t"`); scientific notation accepted. |
| `read_obspy` (extra `seismic`) | `user` | miniSEED, SAC and other formats ObsPy reads. |

A real export of each unvalidated format would settle it. Meanwhile, check the first record you read against what your
seismograph's own software shows.

- **The sample rate is never guessed.** If the file lacks it and you do not pass `sample_rate_hz`, the reader raises an
  error. `read_waveform_csv` with a time column fits the rate by least squares and refuses a non-uniform column.
- **Units.** Inside everything is SI (mm/s, Pa, s). Units are inferred from the resolution only for the Micromate and
  Minimate Pro steps (0.00788 mm/s, 0.0156 Pa), with a warning; otherwise pass `units=`.
- **Renamed channels** are not recognized by the Instantel readers: use
  `read_waveform_csv(..., columns=..., units=..., sep="\t")` (`units=` is required there).
- **Zero-crossing frequency.** `analyze(zc_method="interpolated")` (default) interpolates the crossings; `"instantel"`
  counts whole samples and flags a frequency above 100 fs / 1024 Hz. The frequency is NaN when the record edge cuts the
  half cycle of the peak or, with `"instantel"`, above that limit. The verdict is then `not_comparable` against a
  frequency-dependent curve, never a guess.
- **Vector sum.** `pvs_mm_s` is the real vector sum (maximum of the sum of squares sample by sample); the pseudo vector
  sum is the square root of the sum of the squares of the three component peaks, an upper bound of the real one.
- **FFT.** `fft_frequency` takes the dominant frequency of the whole record and lists the peaks within 6 dB of the largest.
- **Acceleration and displacement.** They depend on the precision of the exported file: with three significant figures or
  fewer the summary warns. Displacement integrates each half cycle as the sum of V times the time step and halves the
  largest, as the Instantel manual describes.

## Airblast

`blastingpy.airblast` fits the site law to the peak overpressure, checks it against the airblast limits and carries the
published equations. The facts, pages and marks are in
[docs/investigacion/FUENTES_SPRINT3.md](docs/investigacion/FUENTES_SPRINT3.md).

```python
from blastingpy import airblast, waveform

events = waveform.events_from_records(records, distance_m=[150, 200, 250], charge_kg=[40, 80, 60])
law = airblast.fit(events)                       # cube-root law on airblast_pa
limit = airblast.limit_db("OSMRE-30CFR816.67", system="2Hz")
charge = airblast.max_charge(law, 400.0, limit.pressure_pa)
verdict = airblast.evaluate(events, "OSMRE-30CFR816.67", system="2Hz")
```

A complete, runnable version with synthetic data is in [examples/03-airblast](examples/03-airblast/).

- **Units.** Inside everything is Pa. `airblast.to_db` / `from_db` convert with P0 = 20 uPa; `read_events(...,
  units={"pressure": "Pa" | "kPa" | "psi" | "dB"})` and `units.to_si` accept Pa, kPa and psi. Events carry
  `airblast_pa` and `airblast_freq_hz`; `waveform.events_from_records` fills them from the microphone.
- **Site law.** `airblast.fit` is the regression of the vibration law with the cube-root form,
  P = K (D / W^(1/3))^-beta, on `airblast_pa`: same statistics, upper bound K 10^(z sigma) (RI 8485, Table B-1, fits
  airblast this way). `predict`, `max_charge` and `distance_for` work as in `attenuation`. `outliers` and `residuals`
  refer to the rows used in the fit.
- **An airblast-only campaign** is passed as a DataFrame with `distance_m`, `charge_kg` and `airblast_pa`.
- **The measuring system is required.** The library refuses to compare against a limit without it.
- **A C-weighted slow level is not a linear peak:** with `system="c-slow"` every verdict is `not_comparable`, and so is a C-slow published equation against any limit. A published equation is a median: `predict_published` and `evaluate_prediction` always warn that about half of the blasts exceed it.

| System | USBM RI 8485 (1980) | OSMRE 30 CFR 816.67 |
|---|---|---|
| `0.1Hz` (lower cutoff 0.1 Hz or lower) | 134 dB | 134 dB, only if the regulatory authority approves |
| `2Hz` | 133 dB | 133 dB |
| `6Hz` (a 5-Hz system uses this row) | 129 dB | 129 dB |
| `c-slow` (C-weighted, slow) | 105 dB, events up to 2 s | 105 dB, only if the regulatory authority approves |

`airblast.system_from_cutoff(low_cutoff_hz)` gives the system of a linear microphone. The limit depends on the lower
cutoff because a level in dB is a peak of what the microphone passes: the lower the cutoff, the more low-frequency
pressure it counts, so each system has its own limit.

Published equations (`list_equations`, `published`, `predict_published`) are medians, without an upper bound, tied to
the system they were fitted with. Fourteen come from RI 8485 Table B-1, two from the OSMRE BLEP and one from AS 2187.2.

| Equation id | System | Mark |
|---|---|---|
| `USBM-RI8485/coal-highwall/{0.1Hz,2Hz,5Hz,c-slow}` | as named | V1 |
| `USBM-RI8485/coal-parting/{0.1Hz,2Hz,5Hz,c-slow}` | as named | V1 |
| `USBM-RI8485/metal-mine-highwall/0.1Hz` | 0.1 Hz | V1 |
| `USBM-RI8485/quarry/0.1Hz`, `USBM-RI8485/quarry-direction-of-initiation/0.1Hz`, `USBM-RI8485/quarry-behind-face/0.1Hz`, `USBM-RI8485/quarry-opposite-initiation/0.1Hz`, `USBM-RI8485/quarry-front-of-face/0.1Hz` | 0.1 Hz | V1 |
| `OSMRE-BLEP/highwall`, `OSMRE-BLEP/parting` | not stated | V1 |
| `AS2187.2-2006/free-face` | not stated | V2 |

BLEP's two equations come from 0.1 Hz fits (RI 8485 states that fit; BLEP itself does not state a system), but BLEP compares them with the 2 Hz limit; `evaluate_prediction` warns when
the equation's system differs from the limit's. AS 2187.2 is [V2], from a single secondary source.

Not included: wind correction, the Australian comfort limits and AS 2187.2 from the standard itself.

## Flyrock

`blastingpy.flyrock` carries the published rock-projection models, a per-hole comparison table and the clearance
distance. The facts, pages and marks are in
[docs/investigacion/FUENTES_SPRINT4.md](docs/investigacion/FUENTES_SPRINT4.md).

```python
import pandas as pd
from blastingpy import flyrock

holes = pd.DataFrame({"diameter_mm": [165.0, 165.0], "burden_m": [4.0, 3.2], "stemming_m": [3.0, 2.5],
                      "linear_charge_kg_m": [18.2, 18.2], "hole_angle_from_vertical_deg": [10.0, 10.0]})
table = flyrock.compare(holes, k=27.0)               # ranges in m, one row per hole
worst = table["rm_max_m"].max()
clear = flyrock.clearance(worst, target="people", regulation="PE-DS024-2016-EM")
print(clear.distance_m, clear.governs)
```

A complete, runnable version with a synthetic hole table is in [examples/04-flyrock](examples/04-flyrock/).

- **No model carries a safety margin; `clearance` adds it:** factor 2 for equipment and 4 for people (Richards & Moore), or your own factor >= 1.
- **Peruvian minimum:** DS 024-2016-EM art. 267 a), as amended by DS 034-2023-EM: a 500 m radius when rock can be projected through 360 degrees, and in bench blasts at least 500 m in the direction the blast throws. It applies to the evacuation of workers (people, not equipment); the result says which of the model and the regulation governs (`governs`). Reporting the larger of the factored range and 500 m is a design decision, not the text of the regulation.
- **`k` is required** (Richards & Moore site constant): 13.5 for soft competent rock, 27 for hard competent rock; the published data range is 15 to 37.
- **Charge column shorter than 1 m:** `compare` uses the actual charge mass (kg) instead of the linear charge, as Richards & Moore prescribe, and warns.
- **Hole angle** is measured from the vertical. A vertical hole gives 0 m of stemming ejection, with a warning (the source says vertical holes need an adjustment).
- **Diameters are in mm** (the Lundborg source uses inches; the library converts and rejects values that look like inches or metres).
- The models are different estimates, not a consensus: nothing averages them or picks one.

| Model | Estimates | Inputs | Mark | Source read |
|---|---|---|---|---|
| Ballistics, `FLY-BALLISTIC` (`range_level`, `max_range_level`, `range_to_height`, `max_height`, `velocity_for_range`) | Range and height without drag | Velocity, launch angle | V1 | Roth 1979 eqs. (1)-(3), (5); Richards & Moore eqs. [4]-[5] |
| Richards & Moore, `FLY-RM-FACE`, `FLY-RM-CRATER`, `FLY-RM-STEMMING` (`rm_face_burst`, `rm_cratering`, `rm_stemming_ejection`, `rm_min_confinement`) | Maximum range by face burst, cratering and stemming ejection; minimum burden or stemming | `k`, charge (kg/m), burden, stemming, hole angle | V1 | Richards & Moore, ISEE, eqs. [6], [7a]-[7c] |
| Lundborg, `FLY-LUNDBORG-MAX`, `FLY-LUNDBORG-Q` (`lundborg_max_range`, `lundborg_optimal_fragment`, `lundborg_range`) | Maximum range and optimal fragment size; range by powder factor | Diameter (mm), powder factor | V2 | As restated by Richards & Moore, McKenzie 2018 and Szendrei & Tose 2022; originals not read |
| McKenzie 2018, `FLY-MCKENZIE-2018` (`mckenzie_max_range`) | Maximum range | SDoB (you supply it), diameter | V1 | McKenzie 2018 summary, slide 11, version 2018 only |
| Roth, `FLY-ROTH-GURNEY`, `FLY-ROTH-CAL` (`roth_charge_mass_ratio`, `roth_gurney_velocity`, `roth_launch_velocity`, `roth_calibrated_velocity`, `roth_reference_gurney_velocity`) | Launch velocity from c/m and the Gurney velocity; calibrated equations | Linear charge, rock density, burden, detonation velocity | V1 | Roth 1979, USBM OFR 77-81, eqs. (7)-(10), (13), (13a), (14), (17) |

`flyrock.info(flyrock.Model.X)` gives the id, mark and citation of each one, and `compare` carries them in `attrs`.

**Published critiques** (Szendrei & Tose 2022, JSAIMM 122(12), pp. 726-731, `attrs["critiques"]`): models without drag can overestimate the range by factors of 2 to 8; Lundborg holds for spherical fragments of the optimal size; McKenzie inherits Lundborg's calibration.

**Not included, and why:** McKenzie (2009) coefficient and the SDoB definition (no source read defines it, so SDoB is yours to give), McKenzie's fragment size and minimum stemming, Roth's graphical curves and bench correlation, scaling Roth's calibrated equations to other explosives (the source does not state the rule and its printed factor looks inverted), drag trajectories, the shape of the clearance zone, and a hole-table file reader (pass a DataFrame; pandas already reads CSV and Excel).

## Conventions

- **Units.** Inside the library everything is SI: m, kg, mm/s, Hz. `read_events(units="imperial")` (or a dict per
  quantity) converts at input; `blastingpy.units` has `to_si` and `from_si`.
- **Upper bound of the site law.** `k_upper = k * 10^(z * sigma)` with `z = 2` by default, as in USBM RI 8507 and OSMRE
  BIVDEP practice. For a one-sided 95 % bound use `attenuation.fit(events, z=attenuation.normal_quantile(0.95))`
  (z = 1.645). `max_charge` and `predict_ppv` use the upper bound by default; pass `attenuation.Level.Median` for
  the median.
- **Warnings.** Ground-vibration fits with fewer than 30 events (the airblast fit does not warn about it), `beta <= 0` and predictions outside the fitted range of distance or
  charge come with warnings (`law.warnings`, `result.warnings`). A law with `beta <= 0` is refused for predictions.
  A law with `sigma == 0` or `z == 0` (for example `make_site_law` without `sigma`) has an upper bound equal to the
  median: its predictions at the upper bound warn that the result has no safety margin.
- **One quantity per law.** A law predicts the PPV quantity it was fitted on (`ppv_kind`), so compare it only with
  limits of that quantity; `attenuation.fit` refuses events that mix kinds.

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
print(c.verification, c.reference, c.note["en"], sep="\n")
```

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

`reason` is empty for `complies`, except when a frequency-curve verdict rests on that peak without the three
components: then it says `peak-only check: components at other frequencies are not verified`. The verdict frame
carries what it was checked against in `verdict.attrs` (`profile_id`, `criterion`, `citation`, `edition`, `quantity`,
`frequency_methods`, `verification`, `reference`), and `criteria.candidates(events, profile, criterion)` lists, per
event, the values the verdict compared with the limit (the compliance chart draws exactly those).

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
`python/blastingpy/profiles/` illustrate the last two. Every limit, frequency and distance in a profile must be a
finite number > 0 (`f_min` may be 0); anything else is refused when the profile is loaded.

## Not included yet

Laws Langefors-Kihlstrom, Ghosh-Daemen and CMRI, and the standards UNE 22381, AS 2187.2, BS 6472-2, SS 4604866,
SN 640312a and NBR 9653 are missing because no source with their formulas or values has been read. DIN 4150-3 line 1
and the upper-floor criteria, the OSMRE Figure 1 curve and BS 7385-2 line 1 below 4 Hz are also out. Details in
[docs/CHANGELOG.md](docs/CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
