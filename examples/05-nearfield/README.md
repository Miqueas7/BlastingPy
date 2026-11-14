# Near field

Holmberg-Persson around a charge -> Heelan around a column -> a whole blast by seed superposition -> its Monte Carlo ->
a delay sweep. Synthetic data only: the rock, the explosive, the hole table, the seed waveform and the site law are
invented, and nothing here comes from a real site.

```bash
python -m pip install ".[viz]"         # from the repository root
cd examples/05-nearfield
python nearfield.py                    # or: python nearfield.py some_folder
```

The script writes `holmberg_persson.png` in the given folder (default: the current one), prints a summary and never
opens a window. It runs in about a second.

| Step | Code | What it does |
|---|---|---|
| 1 | `hp_original`, `hp_niosh`, `hp_grid`, `ppv_critical`, `hp_damage_extent` | PPV around a 3 m charge of 1 kg/m with the original integral and with NIOSH's modified form, the critical PPV of an invented rock (T Vp / E) and the damage extent at mid-charge. K, alpha and beta are required: the script passes `HOLMBERG_PERSSON_SURFACE` explicitly, the values NIOSH reports for surface blasts, which are data and not a default. The chart shows both contours with the critical PPV. |
| 2 | `PressureSource`, `Medium`, `Column`, `heelan_ppv` | Vector PPV at three heights, 5 m from a 6 m column primed at the bottom, linear (A = 1) and with Blair's non-linear weights (A = 0.7), on the automatic time grid. The warnings of the result are printed: here the points are in the near field (Omega_A r / a below pi). |
| 3 | `datasets.synthetic_record`, `datasets.synthetic_events`, `attenuation.fit`, `simulate_blast` | Twelve holes in two rows. The seed stands in for the record of a single-hole shot and the site law is fitted to synthetic events. The table of arrivals shows the distance, arrival time, amplitude (the law's median) and weight of each hole. The three weightings are run, each with the caveats of its author. |
| 4 | `monte_carlo_blast` | 2000 simulations with a delay scatter of 2 ms and 25 % on the amplitude: mean, standard deviation, 95th percentile and the probability of exceeding an invented level. The scatter of the delays is yours to give; the same `rng_seed` gives the same peaks. |
| 5 | `delay_sweep` | Peak of two holes against the delay between them, over the peak of both fired together. The best delay belongs to that seed. |

Holmberg-Persson is static (no detonation velocity, no arrival times); the Heelan column is not, which is why its peak
is larger above a bottom-primed column than below it. The simulation is of one component, the seed's. None of this has
been compared with a measured blast: read the section "Near field" of the main README for what each model assumes and
what is left out.
