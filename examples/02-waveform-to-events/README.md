# Example 2: from seismograph records to the events table

Six synthetic seismograph records (different distances and charges) are saved as Instantel ASCII, read back, analyzed
and turned into the events table of example 1, then fitted and checked against a standard. Synthetic data only
(`datasets.synthetic_record`, seeded); nothing here comes from a real site.

```bash
python -m pip install ".[io,viz]"      # from the repository root
cd examples/02-waveform-to-events
python waveform_to_events.py           # or: python waveform_to_events.py some_folder
```

The script writes `record_1.txt` ... `record_6.txt` and `record.png` in the given folder (default: the current one),
prints a short summary and never opens a window.

| Step | Code | What it does |
|---|---|---|
| 1 | `datasets.synthetic_record(quantize=True)` | One record per blast. Its PPV follows an invented law PPV = 1000 (D / W^0.5)^-1.6 with scatter; values are rounded to the sensor steps (0.00788 mm/s, 0.0156 Pa). |
| 2 | file write | The minimal Instantel ASCII: a row `Tran Vert Long MicL` and one row per sample. |
| 3 | `io.read_instantel_ascii` | The file has no sample rate or pretrigger, so you pass `sample_rate_hz=1024, pretrigger_s=0.25`. Units are inferred from the resolution (with a warning). |
| 4 | `waveform.analyze` | Peak, vector sum, zero-crossing frequency and airblast level of the first record. |
| 5 | `waveform.events_from_records` | One events row per record: largest component as `ppv_mm_s` (`component_max`) and its zero-crossing frequency. The records' warnings travel in `events.attrs["warnings"]` (one `UserWarning` sums them up); the script prints each distinct one. |
| 6 | `attenuation.fit`, `criteria.evaluate` | The site law and a verdict per event against `USBM-RI8507-1980` / `appendix-b-drywall`. Six events fit but warn (fewer than 30). |
| 7 | `plotting.plot_record` | `record.png`: one panel per channel with its peak marked. |
