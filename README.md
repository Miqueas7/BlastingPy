# BlastingPy

Open-source Python library, with a C++20 core, to analyze the effects of blasting in mining: ground vibration,
airblast, flyrock and near-field vibration.

The central use case: a blasting engineer ends a monitoring campaign with a table of events (distance, maximum charge per
delay, PPV, frequency). BlastingPy fits the attenuation law of the site, computes the maximum charge per delay and
checks the result against a regulatory criterion, stating the edition, table, magnitude and frequency method used.

**Documentation: <https://miqueas.dev/blastingpy>** (guide, methods, worked examples and API reference, in English
and Spanish).

**Status:** version 0.2.0. Its six blocks: events table, attenuation law, maximum charge and compliance verdict;
reading seismograph records (waveform, PPV, vector sum, frequency); airblast; flyrock (published range models, per-hole
comparison and clearance distance); near-field vibration (Holmberg-Persson, Heelan, whole-blast simulation and its
Monte Carlo); and the bench pattern (burden by several methods, hole layout, firing sequence, charge per hole and DXF).

## Install

```bash
python -m pip install "blastingpy[io,viz,dxf]"
```

Wheels are published for Python 3.10 to 3.14 on Linux, Windows and macOS. Extras: `io` (read and write `.xlsx`), `viz`
(charts), `dxf` (read and write `.dxf`), `seismic` (ObsPy readers). To build from the source distribution on PyPI (needs a C++ compiler and CMake):

```bash
python -m pip install --no-binary blastingpy "blastingpy[io,viz,dxf]"
```

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
[the worked case 01](https://miqueas.dev/blastingpy/en/casos/01-ley-del-sitio/).

## From a seismograph record

`blastingpy.io` reads the waveform a seismograph exports; `blastingpy.waveform` turns it into peaks, vector sum and
frequency, and into a row of the events table of the sections above. The facts and their evidence are in
[the waveform method page](https://miqueas.dev/blastingpy/en/metodos/forma-de-onda/).

```python
from blastingpy import attenuation, criteria, io, waveform

records = [io.read_instantel_ascii(p, sample_rate_hz=1024, pretrigger_s=0.25) for p in paths]
events = waveform.events_from_records(records, distance_m=[100, 150, 200], charge_kg=[20, 50, 30])
law = attenuation.fit(events, law="sqrt")
verdict = criteria.evaluate(events, "USBM-RI8507-1980", "appendix-b-drywall")
```

A complete, runnable version with synthetic data is in [the worked case 02](https://miqueas.dev/blastingpy/en/casos/02-registros-a-eventos/).

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
  error. `read_waveform_csv` with a time column fits the rate by least squares and refuses a non-uniform column; a
  fitted rate within 1e-6 of a whole number, and a pretrigger within 1e-3 of a sample period of a whole sample, are
  taken as those values (times rounded to the microsecond give 1024.0000003 Hz). SAC stores the sample interval
  rounded, so a 1024 Hz SAC reads as about 1023.54 Hz: `read_obspy` keeps the file's rate, adds a warning that names
  the whole number, and takes `sample_rate_hz=` (it must agree with the file within 0.1 %).
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
  When the dominant peak is in the two lowest bins (f <= 2 / T, T the record length), as with several holes that
  interfere, the spectrum is dominated by the record's envelope or a trend, not by an oscillation: `fft_frequency`
  warns (`UserWarning`) and `analyze` adds it to its warnings. Use the zero-crossing frequency then.
- **Acceleration and displacement.** They depend on the precision of the exported file: with three significant figures or
  fewer the summary warns. Displacement integrates each half cycle as the sum of V times the time step and halves the
  largest, as the Instantel manual describes.

## Airblast

`blastingpy.airblast` fits the site law to the peak overpressure, checks it against the airblast limits and carries the
published equations. The facts, pages and marks are in
[the airblast method page](https://miqueas.dev/blastingpy/en/metodos/airblast/).

```python
from blastingpy import airblast, waveform

events = waveform.events_from_records(records, distance_m=[150, 200, 250], charge_kg=[40, 80, 60])
law = airblast.fit(events)                       # cube-root law on airblast_pa
limit = airblast.limit_db("OSMRE-30CFR816.67", system="2Hz")
charge = airblast.max_charge(law, 400.0, limit.pressure_pa)
verdict = airblast.evaluate(events, "OSMRE-30CFR816.67", system="2Hz")
```

A complete, runnable version with synthetic data is in [the worked case 03](https://miqueas.dev/blastingpy/en/casos/03-airblast/).

- **Units.** Inside everything is Pa. `airblast.to_db` / `from_db` convert with P0 = 20 uPa; `read_events(...,
  units={"pressure": "Pa" | "kPa" | "psi" | "dB"})` and `units.to_si` accept Pa, kPa and psi. Events carry
  `airblast_pa` and `airblast_freq_hz`; `waveform.events_from_records` fills them from the microphone.
- **Site law.** `airblast.fit` is the regression of the vibration law with the cube-root form,
  P = K (D / W^(1/3))^-beta, on `airblast_pa`: same statistics, upper bound K 10^(z sigma) (RI 8485, Table B-1, fits
  airblast this way). `predict`, `max_charge` and `distance_for` work as in `attenuation`. `outliers` and `residuals`
  refer to the rows used in the fit.
- **An airblast-only campaign** is passed as a DataFrame with `distance_m`, `charge_kg` and `airblast_pa`.
- **The measuring system is required.** The library refuses to compare against a limit without it.
- **Verdicts as in `criteria.evaluate`:** `airblast.evaluate` gives `not_comparable` to an event without an airblast value (missing or NaN `airblast_pa`) and `indeterminate` to a value outside the domain (<= 0 or infinite).
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
[the flyrock method page](https://miqueas.dev/blastingpy/en/metodos/flyrock/).

```python
import pandas as pd
from blastingpy import flyrock

holes = pd.DataFrame({"diameter_mm": [165.0, 165.0], "burden_m": [4.0, 3.2], "stemming_m": [3.0, 2.5],
                      "linear_charge_kg_m": [18.2, 18.2], "hole_angle_from_vertical_deg": [10.0, 10.0]})
table = flyrock.compare(holes, k=27.0)               # ranges in m, one row per hole
worst = table["max_model_m"].max()                   # the worst hole, largest hole-specific range
clear = flyrock.clearance(worst, target="people", regulation="PE-DS024-2016-EM")
print(clear.distance_m, clear.governs)
```

A complete, runnable version with a synthetic hole table is in [the worked case 04](https://miqueas.dev/blastingpy/en/casos/04-flyrock/).

- **No model carries a safety margin; `clearance` adds it:** factor 2 for equipment and 4 for people (Richards & Moore), or your own factor >= 1.
- **Peruvian minimum:** DS 024-2016-EM art. 267 a), as amended by DS 034-2023-EM: a 500 m radius when rock can be projected through 360 degrees, and in bench blasts at least 500 m in the direction the blast throws. It applies to the evacuation of workers (people, not equipment); the result says which of the model and the regulation governs (`governs`). Reporting the larger of the factored range and 500 m is a design decision, not the text of the regulation.
- **`k` is required** (Richards & Moore site constant): 13.5 for soft competent rock, 27 for hard competent rock; the published data range is 15 to 37. A `k` outside [13.5, 37] gives a warning (the range goes with k²).
- **Hole table:** required columns `diameter_mm`, `burden_m`, `stemming_m`, `linear_charge_kg_m` (or `explosive_density_kg_m3`, 300-2000 kg/m3, to compute it for a full hole); optional `charge_length_m`, `hole_angle_from_vertical_deg`, `powder_factor_kg_m3` (Lundborg's range by powder factor) and `sdob_m_kg13` (McKenzie, in m/kg^(1/3)). A NaN in an optional column means "not given for that row".
- **`max_model_m`** is the largest range across the hole-specific models of each hole (`rm_face_burst_m`, `rm_cratering_m`, `rm_stemming_ejection_m`, `lundborg_q_m`, `mckenzie_2018_m`); `compare` warns about the holes where it is more than twice `rm_max_m`. **`lundborg_max_m`** stays apart: it depends only on the diameter (Lundborg's practical maximum, for the optimal fragment), so it is the same for every hole of one diameter and does not tell which hole is the worst; it is the diameter-only upper envelope, and `compare` warns about the holes where it is more than twice `max_model_m`. The blast's clearance is governed by its worst hole, and choosing the model is the engineer's decision: `clearance` takes one range and says so in its notes.
- **Warnings where the result can come out too short:** Lundborg's powder-factor range below its maximum (in `compare`, below half of it, a design threshold; a powder-factor correlation tends to underestimate, Roth 1979), an SDoB above 2.4 (ft/lb^(1/3) instead of m/kg^(1/3)?), a vertical hole, Roth's calibrated equations near their minimum c/m, and a linear charge implying a density below 300 kg/m3. A density in g/cc is an error.
- **Charge column shorter than 1 m:** `compare` uses the actual charge mass (kg) instead of the linear charge, as Richards & Moore prescribe, and warns.
- **Hole angle** is measured from the vertical. A vertical hole gives 0 m of stemming ejection, with a warning (the source says vertical holes need an adjustment).
- **Diameters are in mm** (the Lundborg source uses inches; the library converts and rejects values that look like inches or metres).
- The models are different estimates, not a consensus: nothing averages them or picks one.

| Model | Estimates | Inputs | Mark | Source read |
|---|---|---|---|---|
| Ballistics, `FLY-BALLISTIC` (`range_level`, `max_range_level`, `range_to_height`, `max_height`, `velocity_for_range`) | Range and height without drag | Velocity, launch angle | V1 | Roth 1979 eqs. (1)-(3), (5); Richards & Moore eqs. [4]-[5]; Moore & Richards 2005 eqs. [7], [9], [10] |
| Richards & Moore, `FLY-RM-FACE`, `FLY-RM-CRATER`, `FLY-RM-STEMMING` (`rm_face_burst`, `rm_cratering`, `rm_stemming_ejection`, `rm_min_confinement`) | Maximum range by face burst, cratering and stemming ejection; minimum burden or stemming | `k`, charge (kg/m), burden, stemming, hole angle | V1 | Richards & Moore, ISEE, eqs. [6], [7a]-[7c] |
| Lundborg, `FLY-LUNDBORG-MAX`, `FLY-LUNDBORG-Q` (`lundborg_max_range`, `lundborg_optimal_fragment`, `lundborg_range`) | Maximum range and optimal fragment size; range by powder factor | Diameter (mm), powder factor | V2 | As restated by Richards & Moore, McKenzie 2018 and Szendrei & Tose 2022; originals not read |
| McKenzie 2018, `FLY-MCKENZIE-2018` (`mckenzie_max_range`) | Maximum range | SDoB (you supply it), diameter | V1 | McKenzie 2018 summary, slide 11, version 2018 only |
| Roth, `FLY-ROTH-GURNEY`, `FLY-ROTH-CAL` (`roth_charge_mass_ratio`, `roth_gurney_velocity`, `roth_launch_velocity`, `roth_calibrated_velocity`, `roth_reference_gurney_velocity`) | Launch velocity from c/m and the Gurney velocity; calibrated equations | Linear charge, rock density, burden, detonation velocity | V1 | Roth 1979, USBM OFR 77-81, eqs. (7)-(10), (13), (13a), (14), (17) |

`flyrock.info(flyrock.Model.X)` gives the id, mark and citation of each one, and `compare` carries them in `attrs`.

**Published critiques** (`attrs["critiques"]`): models without drag can overestimate the range by factors of 2 to 8; Lundborg holds for spherical fragments of the optimal size; McKenzie inherits Lundborg's calibration (Szendrei & Tose 2022, JSAIMM 122(12), pp. 726-731); a powder-factor correlation tends to underestimate the range (Roth 1979, pp. 10-11, 29-31).

**Not included, and why:** McKenzie (2009) coefficient and the SDoB definition (no source read defines it, so SDoB is yours to give), McKenzie's fragment size and minimum stemming, Roth's graphical curves and bench correlation, scaling Roth's calibrated equations to other explosives (the source does not state the rule and its printed factor looks inverted), drag trajectories, the shape of the clearance zone, and a hole-table file reader (pass a DataFrame; pandas already reads CSV and Excel).

## Near field

`blastingpy.nearfield` estimates the vibration close to a charge (Holmberg-Persson, Heelan) and simulates a whole blast
by superposing a seed waveform, with a Monte Carlo over the delays and the amplitudes. The facts, pages and marks are in
[the near-field method page](https://miqueas.dev/blastingpy/en/metodos/campo-cercano/).

```python
import pandas as pd
from blastingpy import attenuation, datasets, nearfield

# Holmberg-Persson around a 3 m charge of 1 kg/m. K, alpha and beta are yours to give.
charge = dict(k=700.0, alpha=0.7, beta=1.5, linear_charge_kg_m=1.0, z_start=-1.5, z_end=1.5)
critical = nearfield.ppv_critical(10e6, 5000.0, 50e9)       # T Vp / E with the static E: 1000 mm/s
extent = nearfield.hp_damage_extent(nearfield.HpModel.Niosh, **charge, z=0.0, ppv_limit_mm_s=critical)

# Heelan around a 6 m column primed at the bottom.
source = nearfield.PressureSource(vn_pressure_pa=5e9, n=6, gamma_per_s=8000.0)
medium = nearfield.Medium(p_velocity_m_s=5000.0, s_velocity_m_s=2900.0, density_kg_m3=2700.0)
column = nearfield.Column(hole_radius_m=0.05, z_bottom=0.0, z_top=6.0, elements=41, primer_element=1,
                          vod_m_s=4500.0, charge_exponent=1.0)
peak = nearfield.heelan_ppv(source, medium, column, 5.0, 3.0)        # at r = 5 m, z = 3 m
print(peak.ppv_mm_s, peak.warnings)

# A whole blast: hole table, seed (the record of a single-hole shot) and site law.
holes = pd.DataFrame({"x_m": [0.0, 5.0, 10.0], "y_m": 0.0, "charge_kg": 80.0, "firing_time_s": [0.0, 0.017, 0.034]})
seed = (datasets.synthetic_record(pretrigger_s=0.0, mic_pa=None), "vertical")   # stands in for your record
blast = dict(law=attenuation.fit(datasets.synthetic_events(), law="sqrt"), wave_velocity_m_s=3000.0)
result = nearfield.simulate_blast(holes, (5.0, 120.0), seed, **blast)
print(result.ppv_mm_s, result.arrivals, result.warnings)
mc = nearfield.monte_carlo_blast(holes, (5.0, 120.0), seed, delay_sigma_s=0.002, amplitude_cv=0.25,
                                 simulations=2000, rng_seed=1, **blast)
print(mc.percentile(95), mc.probability_exceeding(25.0))
```

A complete, runnable version with synthetic data is in [the worked case 05](https://miqueas.dev/blastingpy/en/casos/05-campo-cercano/).

| Model | Estimates | Inputs | Mark | Source read |
|---|---|---|---|---|
| Holmberg-Persson, original integral form (`hp_point`, `hp_original`, `hp_grid("original", ...)`) | Static PPV around a long charge, K [q ∫ dz / (r² + z²)^(β/(2α))]^α | K, α, β (yours), linear charge (kg/m), charged span and point (m) | V1, through NIOSH | Iverson, Kerkering & Hustrulid (2008), NIOSH, eqs. 1 and 5. Holmberg & Persson's own papers were not read |
| Holmberg-Persson, closed form for β = 2α (`hp_term`) | The same integral as an arctangent; `hp_original` uses it when β = 2α | Linear charge, charged span, point | V2 | Onederra & Esen (2004), Fragblast 8(2), eq. 2 |
| NIOSH modified form (`hp_mean_distance`, `hp_niosh`, `hp_grid("niosh", ...)`) | K (q L)^α / R̄^β, with R̄ the mean distance from the charge to the point | The same as the original form | V1 | NIOSH (2008), eqs. 8-10 |
| Critical PPV (`ppv_critical`) and damage extent (`hp_damage_extent`) | T Vp / E, and the radial distance at which either form falls to a PPV limit | Tensile strength and static Young's modulus (Pa), P velocity (m/s); form, constants, charge, height z, limit (mm/s) | V2; the extent takes the mark of the form used | Onederra & Esen (2004), eq. 11, citing Persson, Holmberg & Lee (1994, p. 247). That the modulus is the static one is inferred |
| Heelan element (`heelan_element`, with `pressure_function`, `g_function`, `g_peak`, `scaled_frequency`, `heelan_amplitude`) | P and SV velocity waveform radiated by a short cylindrical charge | `PressureSource` (von Neumann pressure in Pa, n, decay γ in 1/s), `Medium` (Vp and Vs in m/s, density in kg/m³), hole radius, half length, point, times | V1 | Heelan (1953), Geophysics 18(3), eqs. 12-13, in the form of Blair (2007), Geophysics 72(2), eqs. 8 and 13-16 |
| Heelan column (`Column`, `column_weights`, `heelan_column`, `vector_peak`, `heelan_ppv`) | Waveform and vector PPV of a column initiated at the detonation velocity, linear (A = 1) or with non-linear weights (A < 1) | Source, medium, `Column` (hole radius, span, odd number of elements, primer element, detonation velocity in m/s, charge exponent A), point | V1 | Blair (2008), IJRMMS 45, eqs. 1-3 and 22. Inferred: the initiation time of each element and the mirrored weights when the primer is above mid-column |
| Seed superposition (`superpose`, `simulate_blast`, `delay_sweep`) | Waveform of a whole blast at a monitor: one seed per hole, delayed and scaled | Hole table, monitor, seed, site law, wave velocity | V1 | Hinzen (1988), IJRMMS 25(6), eqs. 3-7; Blair (1999), Fragblast 3(4), eq. 1a |
| Full-blast weights (`screening_counts`, `screening_factor`, `charge_scaling_weights`, `damage_weights`) | Weight of each hole by screening, charge scaling or damage | Positions, charges, firing times, mean spacing (m) | V1 | Blair (1999), eqs. 10b-10c; Blair (2008), eqs. 7-12 and 16-21 |
| Monte Carlo (`monte_carlo_peaks`, `monte_carlo_blast`) | Distribution of the peak with scattered delays and amplitudes | The same as the simulation, plus `delay_sigma_s` (s), `amplitude_cv`, `simulations`, `rng_seed` | V1 for the equations; the delay distribution is yours | Blair (1999), eqs. 1b and 7 |
| Charge per window (`max_charge_per_window`, `max_charge_8ms_kg`) | Largest charge fired within 8 ms | Firing times (s), charges (kg) | V2 | Duvall et al. (1963), as reported by Blair (1999, p. 337) |

- **Units.** Lengths, coordinates and spacing in m; charges in kg and linear charge in kg/m; firing times, delays and their scatter in **seconds, not milliseconds**; wave and detonation velocities in m/s; pressures and moduli in Pa; density in kg/m³; sample rate in Hz. Every PPV and waveform comes out in mm/s (the `hp_*` functions return the unit K carries: mm/s with constants fitted in mm/s, kg and m).
- **K, α and β of Holmberg-Persson are required.** No function has a default for them. 700 / 0.7 / 1.5 are the values NIOSH reports for surface blasts ([V2], attributed to Lundborg et al. 1978; no rock type is named); they are offered as data in `HOLMBERG_PERSSON_SURFACE`, with their citation, and are used only if you pass them. Fit the constants of your site.
- **Holmberg-Persson is static, and its integral has a known flaw.** It is time-independent and ignores the detonation velocity (Onederra & Esen 2004, p. 62, summarizing Blair & Minchinton 1996 and 1997). Going from the sum over charge elements to the integral moves the exponent α from inside the sum to outside the integral, and the exponent does not commute with the sum (Hustrulid & Lu 2002, as reported by NIOSH, PDF p. 7; Blair 2008, p. 235, states the same defect). NIOSH's modified form answers it with a mean distance. The original form diverges on the charged axis (`hp_original` raises there, `hp_grid` gives NaN); the modified form is finite everywhere.
- **Heelan holds within limits, and the library warns outside them.** Blair (2007, pp. E37-E40): the scaled frequency Ω_A = c γ a / Vp must be at most about 0.1 (above it Heelan overestimates the exact solution at any distance); the point is in the near field when Ω_A r / a < π; and the half length of an element should not exceed about twice the radius. `a` is the hole radius, the convention of Blair (2007); Heelan (1953, p. 695) defines it as the radius of the equivalent cavity. `heelan_ppv` refuses nothing on those grounds: each limit that is not met goes in `warnings` with the computed number, next to the standing caveats (P and SV are added linearly where they overlap; the non-linear column weights have no experimental validation).
- **Time grid of `heelan_ppv`.** The PPV is always the peak of the waveform. Without `sample_rate_hz` and `duration_s` the grid is automatic: a step of 1 / (40 γ) until the last arrival plus 40 / γ (or plus (n + 4√n + 2) / γ if longer). With your own grid it warns when the record ends before the last arrival plus (n + 4√n + 2) / γ, the length of the pulse (the peak may be truncated) and when the step is longer than 1 / (10 γ) (the pulse is under-resolved; the sample rate of a seismograph is usually far too low here).
- **Whole-blast simulation.** The amplitude of each hole is the **median** prediction of the site law (not its upper bound) for its charge and its distance to the monitor, times a weight; each hole arrives at its firing time plus distance / wave velocity; the result is one component, the seed's. The seed is normalized to unit peak and used from its first sample: trim the pretrigger. Inspect `arrivals` before trusting the peak. An arrival between two samples is spread linearly over them, which lowers its peak by up to sin²(π f / fs) for a seed of dominant frequency f (9.5 % at f = fs / 10): sample the seed at ten times its dominant frequency or more.
- **Three weightings, each with its author's caveat** (they travel in `warnings`). `"linear"` (Hinzen 1988; Blair 1999), optionally with the screening of Blair (1999), whose g was fitted to one set of field data and whose floor of 0.2 the author sets for lack of near-field data. `"charge-scaling"` (Blair 2008, eq. 12): its author reports an almost linear, unrealistic decay of the amplitude with time and does not recommend it over the damage model. `"damage"` (Blair 2008, eq. 21): η = 0.2 is the author's first estimate and 0.3 fitted his single field case. Blair's (2008) models have no free faces and do not change the shape of the waveform. `charge_exponent` defaults to the law's own (β/2 for a square-root law, β/3 for a cube-root law, the fitted exponent for a general law); `SiteLaw.charge_exponent` is not that number for the scaled-distance laws.
- **Firing times are compared with a tolerance of 1 ns.** Holes within 1 ns of each other are simultaneous: they do not count as already fired for each other, and two holes nominally 8 ms apart never share a window because of rounding.
- **The 8 ms window.** `max_charge_8ms_kg` is the largest charge fired within any 8 ms, by firing times. The window comes from Duvall et al. (1963), as reported by Blair (1999, p. 337), who adds that there is nothing special about 8 ms: whether two holes separate depends on the seed.
- **What the Monte Carlo randomizes.** The arrival of each hole, with a normal scatter of standard deviation `delay_sigma_s` that you give (a design choice: Blair 1999 publishes neither the type nor the deviation of the delay distribution), and its amplitude, multiplied by (1 + c z) with z standard normal and c = `amplitude_cv` (Blair 1999, eq. 7). It does not randomize the shape of the seed (Blair's fluctuating seed), so the variability of the waveform is under-represented, and the weights come from the nominal firing times. The same `rng_seed` gives the same peaks (across platforms, to about 1e-9 relative). The default `amplitude_cv = 0` ignores the scatter of the site law: when the law has σ > 0 a warning gives the coefficient of variation of a lognormal with that σ, √(exp((σ ln 10)²) − 1), for you to pass.
- **Plausibility warnings.** Firing times that span more than 20 s or with more than 1 s between consecutive holes, and a delay above 1 s in `delay_sweep`, look like milliseconds and are flagged; so is a `wave_velocity_m_s` below 100 (it looks like km/s). A `delay_sigma_s` above 1 s is an error; so is a waveform of more than 2 000 000 samples. A seed taken from the microphone channel, a seed whose dominant frequency is above a tenth of its sample rate, and an argument the chosen weighting does not use are flagged too.
- **Warnings that a lower PPV would otherwise hide.** `simulate_blast` and `monte_carlo_blast` always say that the amplitudes are the site law's median (about half of the holes exceed it). With `weighting="damage"`, holes with a screening factor S_n below 0.2 or closer than 99 m to the monitor are listed: S_n (Blair 2008, eq. 20) has no floor and g was fitted with monitors at 99 m or more (Blair 1999, Fig. 7b), so their weights extrapolate the source. These warnings (units, the median, `amplitude_cv`, the near field of the damage model and the under-sampled seed) go in `warnings` and are also emitted as `UserWarning`, so they show even if you never read `warnings`.
- **Not validated against field data.** The functions reproduce the equations and worked numbers of their sources; no simulation here has been compared with a measured blast.

**Not included, and the paper that would unlock it:** the scaled Heelan model and its constant γ_n (Blair & Minchinton 2006, Fragblast-8, pp. 152-159; the column here is the standard Heelan element with Blair's 2008 weights); the exact solution for the short charge (Blair 2007 and Meredith 1990; read, out of scope); the fluctuating seed of the Monte Carlo (its normalization is derived in Blair 1993a, Int. J. Numer. Anal. Methods Geomech. 17: 95-118); the composition of the longitudinal, transverse and vertical components (Blair 1999, eqs. 15a-15c: the sign convention depends on a figure); the statistical charge and its spectrum (Blair 1999, eqs. 17-23: read, not implemented; eq. 23 as printed carries a factor that its own limit does not give); Heelan's energy partition and his SV/P ratio of 1.6 (the printed formulas are inconsistent); the PPV at the end of the crushing zone (Onederra & Esen 2004, eq. 5, with Liu & Katsabanis 1993 and Esen et al. 2003: an exponent could not be verified and it needs non-ideal detonation pressures); and Holmberg & Persson's own papers (1979) and Persson, Holmberg & Lee (1994, p. 247), with which the integral, the surface constants and the critical PPV would be [V1]. Their 700-1000 mm/s damage limit is for Swedish igneous rocks, as reported by NIOSH, and is not offered as a general criterion.

## Bench pattern

`blastingpy.pattern` computes the burden with four methods side by side (Konya [V1, FHWA], López Jimeno and Ash [V2];
Langefors-Kihlström is listed but not implemented, because the two secondary sources that give its equation disagree,
pending the 1978 book), lays the holes inside a blast outline, assigns firing times (row or V sequence), computes the
charge per hole and writes a DXF (outline, crest on its own layer `CRESTA`, holes by row or by delay, labels and tie-in).
The DXF needs the optional extra `dxf`. The facts and marks are on [the documentation site](https://miqueas.dev/blastingpy/en/).

```python
import numpy as np
from blastingpy import pattern

methods = pattern.burden_methods(diameter_mm=165.0, bench_height_m=10.0, rock_density_kg_m3=2700.0,
                                 explosive_density_kg_m3=850.0, initiation="delayed", ucs_mpa=90.0,
                                 explosive_class="low_density")          # one row per method
outline = np.array([[0.0, 0.0], [48.0, 0.0], [48.0, 26.0], [0.0, 26.0]])
crest = np.array([[0.0, 0.0], [48.0, 0.0]])
konya = methods.loc["konya"]
holes = pattern.layout(outline, crest, burden_m=4.2, spacing_m=4.9, arrangement="rectangular")
holes = pattern.sequence(holes, scheme="v", hole_delay_ms=17.0, row_delay_ms=42.0)
holes = pattern.charge(holes, bench_height_m=10.0, subdrill_m=konya.subdrill_m, stemming_m=konya.stemming_m,
                       diameter_mm=165.0, explosive_density_kg_m3=850.0)
pattern.to_dxf(holes, "malla.dxf", outline=outline, crest=crest, diameter_mm=165.0)   # needs blastingpy[dxf]
```

A complete, runnable version with an invented outline, ending in the simulated PPV, is in `examples/06-bench-pattern/`
of the source distribution.

**Front row and flyrock.** `layout` puts the design burden in `burden_m` on every hole, and adds `crest_distance_m`: the
distance from each front-row hole to the real crest, NaN in the other rows. That distance is the real burden of the
front row. `flyrock.compare` reads `burden_m`, and the face-burst range grows as B^-2.6 (Richards & Moore), so a front
burden 20 % short gives about 1.8 times the range. Put the real burden in the front row before comparing:

```python
front = holes["row"] == 0
holes.loc[front, "burden_m"] = holes.loc[front, "crest_distance_m"]
```

**Limits.** Vertical holes, one free face (the crest) and one explosive per hole (a full column, no decks). The
edge-hole rule and the row rules (rows parallel to the crest line, empty rows skipped) are design decisions without a
published source. No partial row is added at the back: the outline is the limit of the rock to blast, and a rear gap over B only gives a warning. Coordinates are in
metres: a pattern over 1,000,000 holes or rows is refused (usually an outline drawn in mm), and
`io.read_dxf_outline` warns when the DXF header (`$INSUNITS`) gives other units.

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

The five blocks planned for version 0.1 are in; what follows is not part of it. Each block lists what it leaves out,
and why, at the end of its section ([Airblast](#airblast), [Flyrock](#flyrock), [Near field](#near-field)). In ground
vibration: the laws Langefors-Kihlstrom, Ghosh-Daemen and CMRI, and the standards UNE 22381, AS 2187.2, BS 6472-2,
SS 4604866, SN 640312a and NBR 9653 are not included because no source with their formulas or values has been read.
DIN 4150-3 line 1 and the upper-floor criteria, the OSMRE Figure 1 curve and BS 7385-2 line 1 below 4 Hz are also out.
Three seismograph formats (THOR, Blastware with header, histogram) are not validated against a real file. Details in
[the status page](https://miqueas.dev/blastingpy/en/fundamento/estado/).

## How to cite

If BlastingPy helps your work, please cite it (the metadata is in `CITATION.cff`, in the
source distribution):

> Quintanilla Ventura, M. (2026). *BlastingPy: blast vibration, airblast, flyrock and near-field analysis for mining*
> (version 0.2.0) [Computer software]. https://miqueas.dev/blastingpy

Each result also records the method and source it follows (`MethodInfo`); cite that original source as well.

## License

MIT (the `LICENSE` file travels in the package).
