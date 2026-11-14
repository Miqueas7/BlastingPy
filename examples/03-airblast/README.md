# Example 3: airblast (overpressure)

Eight synthetic seismograph records (different distances and charges) become the events table with `airblast_pa`, the
cube-root site law of the overpressure is fitted, and the events are checked against a limit. Synthetic data only
(`datasets.synthetic_record`, seeded); nothing here comes from a real site. The microphone peak follows an invented law
P = 20000 (D / W^(1/3))^-1.2 Pa with mild scatter.

```bash
python -m pip install .               # from the repository root
cd examples/03-airblast
python airblast.py                     # writes no files
```

The measuring system is `"2Hz"`: a linear microphone with a 2 Hz lower cutoff, typical of a blasting seismograph. The
library refuses to compare against a limit without it.

| Step | Code | What it does |
|---|---|---|
| 1 | `datasets.synthetic_record(mic_pa=...)` | One record per blast; the microphone peak follows the invented law. |
| 2 | `waveform.events_from_records` | One events row per record, with `airblast_pa` and `airblast_freq_hz` from the microphone. |
| 3 | `airblast.fit` | Site law on `airblast_pa` (cube root); upper bound K 10^(z sigma). Eight events fit but warn (fewer than 30). |
| 4 | `airblast.predict`, `airblast.max_charge` | Upper-bound pressure and maximum charge per delay at a receiver 400 m away. |
| 5 | `airblast.evaluate(..., system="2Hz")` | A verdict per event against `OSMRE-30CFR816.67` (133 dB for 2 Hz). |
| 6 | `airblast.evaluate_prediction` | A published median (`USBM-RI8485/quarry/0.1Hz`) against the same limit; it warns that its measuring system (0.1 Hz) differs from the limit's. |
