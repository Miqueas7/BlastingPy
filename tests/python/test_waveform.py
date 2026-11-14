import copy
import math
import pickle
import warnings
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import blastingpy
from blastingpy import attenuation, criteria, waveform
from blastingpy._blastingpy_core import _waveform as core
from blastingpy.events import _ORDER, read_events
from blastingpy.io import _make_record, from_arrays

FS = 1024.0
N = 2048
GM = {"geophone": "mm/s", "mic": "Pa"}
G = {"geophone": "mm/s"}
FREQS = {"transverse": 20.0, "vertical": 35.0, "longitudinal": 12.0, "mic": 8.0}
START = datetime(2026, 1, 15, 10, 30)
PRECISION = "significant figures"


def _burst(amp, f, center_s=1.0, width_s=0.15, n=N, fs=FS):
    # Seno con envolvente gaussiana: la envolvente es positiva, así que los cruces por cero son los del seno.
    t = np.arange(n) / fs
    x = np.exp(-(((t - center_s) / width_s) ** 2)) * np.sin(2 * np.pi * f * t)
    return amp * x / np.max(np.abs(x))


def _edge(amp, f, tau_s=0.1, n=N, fs=FS):
    # Coseno amortiguado: el pico es la muestra 0 y su medio ciclo toca el borde con muestras distintas de cero.
    t = np.arange(n) / fs
    return amp * np.exp(-t / tau_s) * np.cos(2 * np.pi * f * t)


def _channels(amps=None, freqs=FREQS):
    amps = amps or {"transverse": 3.0, "vertical": 5.0, "longitudinal": 2.0, "mic": 100.0}
    return {k: _burst(a, freqs[k]) for k, a in amps.items()}


def _record(channels=None, units=GM, pretrigger_s=0.25, start=START):
    return from_arrays(_channels() if channels is None else channels, FS, units=units, pretrigger_s=pretrigger_s,
                       start=start)


# --- Análisis del registro ----------------------------------------------------------------------------------------

def test_exported_as_submodule():
    assert blastingpy.waveform is waveform


def test_peaks_frequencies_and_mic_of_a_synthetic_record():
    ch = _channels()
    rec = _record(ch)
    s = waveform.analyze(rec)
    assert list(s.channels) == ["transverse", "vertical", "longitudinal", "mic"]
    t = rec.time_s()
    for name, x in ch.items():
        c = s.channels[name]
        assert c.name == name
        assert c.peak == np.max(np.abs(x))
        assert c.peak_time_s == t[int(np.argmax(np.abs(x)))]
        assert c.zc_status == "ok"
        assert c.zc_frequency_hz == pytest.approx(FREQS[name], rel=0.005)
        assert c.fft_frequency_hz == pytest.approx(FREQS[name], abs=FS / N)
        assert c.fft_peaks_6db_hz[0] == c.fft_frequency_hz
    assert s.mic_pspl_pa == 100.0
    assert s.mic_db_l == pytest.approx(133.98, abs=0.005)
    assert s.zc_method == "interpolated"
    assert s.start == START
    assert not any(PRECISION in w for w in s.warnings)


def test_acceleration_displacement_and_vector_sums_come_from_the_core():
    ch = _channels()
    rec = _record(ch)
    s = waveform.analyze(rec)
    for name in ("transverse", "vertical", "longitudinal"):
        x = ch[name]
        c = s.channels[name]
        a = core.peak_acceleration(x, FS, core.DiffMethod.Central).value / core.standard_gravity_mm_s2
        assert c.peak_acceleration_g == pytest.approx(a, rel=1e-12)
        assert c.peak_displacement_mm == pytest.approx(core.peak_displacement(x, FS).value, rel=1e-12)
    forward = waveform.analyze(rec, acceleration="forward").channels["vertical"].peak_acceleration_g
    a_fwd = core.peak_acceleration(ch["vertical"], FS, core.DiffMethod.Forward).value / core.standard_gravity_mm_s2
    assert forward == pytest.approx(a_fwd, rel=1e-12)
    mic = s.channels["mic"]
    assert math.isnan(mic.peak_acceleration_g) and math.isnan(mic.peak_displacement_mm)
    pvs = core.vector_sum_peak(ch["transverse"], ch["vertical"], ch["longitudinal"])
    assert s.pvs_mm_s == pytest.approx(pvs.value, rel=1e-12)
    assert s.pvs_time_s == rec.time_s()[pvs.index]
    assert s.pseudo_vector_sum_mm_s == pytest.approx(math.sqrt(3.0 ** 2 + 5.0 ** 2 + 2.0 ** 2), rel=1e-12)
    assert s.pseudo_vector_sum_mm_s >= s.pvs_mm_s


def test_summary_is_frozen_and_carries_the_record_warnings_first():
    rec = _record(pretrigger_s=None)
    s = waveform.analyze(rec)
    assert s.warnings[: len(rec.warnings)] == rec.warnings
    with pytest.raises(AttributeError):
        s.pvs_mm_s = 1.0
    with pytest.raises(TypeError):
        s.channels["mic"] = None
    with pytest.raises(AttributeError):
        s.channels["vertical"].peak = 1.0


def test_instantel_method_counts_whole_samples_and_above_limit_has_nan_frequency():
    ch = _channels({"transverse": 1.0, "vertical": 0.8, "longitudinal": 0.5},
                   freqs={"transverse": 150.0, "vertical": 35.0, "longitudinal": 12.0})
    rec = _record(ch, units=G)
    s = waveform.analyze(rec, zc_method="instantel")
    assert s.zc_method == "instantel"
    t = s.channels["transverse"]
    assert t.zc_status == "above_limit" and math.isnan(t.zc_frequency_hz)
    v = s.channels["vertical"]
    expected = core.zero_crossing_frequency(ch["vertical"], FS, int(np.argmax(np.abs(ch["vertical"]))),
                                            core.ZcMethod.WholeSamples)
    assert v.zc_status == "ok" and v.zc_frequency_hz == expected.frequency_hz
    row = waveform.to_event_row(s, distance_m=100.0, charge_kg=50.0)
    assert math.isnan(row["frequency_hz"]) and math.isnan(row["freq_t_hz"])
    verdict = criteria.evaluate(pd.DataFrame([row]), "USBM-RI8507-1980", "appendix-b-drywall")
    assert verdict["status"].iloc[0] == "not_comparable"


def test_bad_methods_and_record_are_domain_errors():
    rec = _record()
    with pytest.raises(ValueError, match="parameter 'zc_method'"):
        waveform.analyze(rec, zc_method="whole")
    with pytest.raises(ValueError, match="parameter 'acceleration'"):
        waveform.analyze(rec, acceleration="backward")
    with pytest.raises(ValueError, match="parameter 'record'"):
        waveform.analyze({"vertical": np.ones(4)})


# --- Review Focus 3: pico en el borde ------------------------------------------------------------------------------

def test_peak_at_the_record_edge_is_incomplete_and_not_comparable():
    ch = {"transverse": _edge(2.0, 20.0), "vertical": _edge(1.0, 35.0)[::-1].copy(),
          "longitudinal": _edge(0.5, 12.0)}
    assert int(np.argmax(np.abs(ch["vertical"]))) == N - 1
    s = waveform.analyze(_record(ch, units=G))
    for c in s.channels.values():
        assert c.zc_status == "incomplete" and math.isnan(c.zc_frequency_hz)
    row = waveform.to_event_row(s, distance_m=150.0, charge_kg=40.0)
    assert row["ppv_mm_s"] == 2.0
    for col in ("frequency_hz", "freq_t_hz", "freq_v_hz", "freq_l_hz"):
        assert math.isnan(row[col])
    verdict = criteria.evaluate(pd.DataFrame([row]), "USBM-RI8507-1980", "appendix-b-drywall")
    assert verdict["status"].iloc[0] == "not_comparable"


# --- Review Focus 4: canal en cero, sin micrófono, sin un geófono ---------------------------------------------------

def test_all_zero_channel_gives_zero_peak_and_nan_without_error():
    ch = _channels()
    ch["longitudinal"] = np.zeros(N)
    s = waveform.analyze(_record(ch))
    c = s.channels["longitudinal"]
    assert c.peak == 0.0 and c.zc_status == "not_computable" and math.isnan(c.zc_frequency_hz)
    assert math.isnan(c.fft_frequency_hz) and c.fft_peaks_6db_hz == ()
    assert c.peak_acceleration_g == 0.0 and math.isnan(c.peak_displacement_mm)
    assert np.isfinite(s.pvs_mm_s) and np.isfinite(s.pseudo_vector_sum_mm_s)
    row = waveform.to_event_row(s, distance_m=100.0, charge_kg=50.0)
    assert row["ppv_l_mm_s"] == 0.0 and math.isnan(row["freq_l_hz"])


def test_without_mic_the_mic_values_are_nan():
    ch = _channels()
    del ch["mic"]
    s = waveform.analyze(_record(ch, units=G))
    assert "mic" not in s.channels
    assert math.isnan(s.mic_pspl_pa) and math.isnan(s.mic_db_l)


def test_zero_mic_gives_nan_db_not_minus_infinity():
    ch = _channels()
    ch["mic"] = np.zeros(N)
    s = waveform.analyze(_record(ch))
    assert s.mic_pspl_pa == 0.0 and math.isnan(s.mic_db_l)


def test_without_one_geophone_the_vector_sums_are_nan():
    ch = _channels()
    del ch["longitudinal"]
    s = waveform.analyze(_record(ch))
    assert math.isnan(s.pvs_mm_s) and math.isnan(s.pvs_time_s) and math.isnan(s.pseudo_vector_sum_mm_s)
    row = waveform.to_event_row(s, distance_m=100.0, charge_kg=50.0)
    assert row["ppv_mm_s"] == 5.0
    assert math.isnan(row["ppv_l_mm_s"]) and math.isnan(row["freq_l_hz"]) and math.isnan(row["pvs_mm_s"])


def test_event_row_without_geophone_is_a_domain_error():
    s = waveform.analyze(_record({"mic": _burst(50.0, 8.0)}, units={"mic": "Pa"}))
    with pytest.raises(ValueError, match="parameter 'summary'"):
        waveform.to_event_row(s, distance_m=100.0, charge_kg=50.0)


# --- Fila de eventos ------------------------------------------------------------------------------------------------

def test_event_row_has_the_canonical_columns_and_values():
    s = waveform.analyze(_record())
    row = waveform.to_event_row(s, distance_m=120.0, charge_kg=60.0, point="P1")
    assert list(row) == list(_ORDER)
    assert row["distance_m"] == 120.0 and row["charge_kg"] == 60.0
    assert row["ppv_mm_s"] == 5.0 and row["ppv_kind"] == "component_max"
    assert row["frequency_hz"] == s.channels["vertical"].zc_frequency_hz
    assert (row["ppv_t_mm_s"], row["ppv_v_mm_s"], row["ppv_l_mm_s"]) == (3.0, 5.0, 2.0)
    assert row["freq_t_hz"] == s.channels["transverse"].zc_frequency_hz
    assert row["freq_l_hz"] == s.channels["longitudinal"].zc_frequency_hz
    assert row["pvs_mm_s"] == s.pvs_mm_s
    assert row["date"] == START and row["point"] == "P1"
    other = datetime(2026, 2, 1)
    assert waveform.to_event_row(s, distance_m=120.0, charge_kg=60.0, date=other)["date"] == other


def test_event_row_airblast_from_the_microphone():
    ch = _channels({"transverse": 3.0, "vertical": 5.0, "longitudinal": 2.0, "mic": 50.0})
    s = waveform.analyze(_record(ch))
    row = waveform.to_event_row(s, distance_m=120.0, charge_kg=60.0)
    assert row["airblast_pa"] == pytest.approx(50.0, rel=0.01)
    assert row["airblast_pa"] == s.mic_pspl_pa
    assert row["airblast_freq_hz"] == s.channels["mic"].zc_frequency_hz and math.isfinite(row["airblast_freq_hz"])
    geo = {k: v for k, v in ch.items() if k != "mic"}
    s = waveform.analyze(_record(geo, units=G))
    row = waveform.to_event_row(s, distance_m=120.0, charge_kg=60.0)
    assert math.isnan(row["airblast_pa"]) and math.isnan(row["airblast_freq_hz"])


def test_event_row_ties_go_to_transverse_then_vertical():
    ch = _channels({"transverse": 4.0, "vertical": 4.0, "longitudinal": 1.0})
    s = waveform.analyze(_record(ch, units=G, start=None))
    row = waveform.to_event_row(s, distance_m=120.0, charge_kg=60.0)
    assert row["frequency_hz"] == s.channels["transverse"].zc_frequency_hz
    assert math.isnan(row["date"]) and math.isnan(row["point"])


@pytest.mark.parametrize("name,value", [("distance_m", 0.0), ("distance_m", math.nan), ("charge_kg", -1.0),
                                        ("charge_kg", math.inf), ("distance_m", True)])
def test_event_row_needs_positive_distance_and_charge(name, value):
    s = waveform.analyze(_record())
    kwargs = {"distance_m": 100.0, "charge_kg": 50.0, name: value}
    with pytest.raises(ValueError, match=f"parameter '{name}'"):
        waveform.to_event_row(s, **kwargs)


# --- Integración: registros -> tabla de eventos -> ley de atenuación y veredicto ------------------------------------

def _site_records():
    # Ley sintética v = 1000 (D/√W)^-1.6 con W = 100 kg y una dispersión fija; la vertical es la mayor componente.
    distances = [40.0, 100.0, 200.0, 600.0, 1200.0, 2000.0]
    scatter = [1.1, 1.1, 0.9, 1.05, 0.95, 1.0]
    records = []
    for i, (d, k) in enumerate(zip(distances, scatter)):
        ppv = 1000.0 * (d / 10.0) ** -1.6 * k
        ch = _channels({"transverse": 0.7 * ppv, "vertical": ppv, "longitudinal": 0.5 * ppv, "mic": 20.0})
        records.append(_record(ch, start=datetime(2026, 3, 1 + i, 9, 0)))
    return records, distances


def test_events_from_records_feed_attenuation_and_criteria():
    records, distances = _site_records()
    events = waveform.events_from_records(records, distance_m=distances, charge_kg=[100.0] * 6,
                                          point=[f"P{i + 1}" for i in range(6)])
    assert list(events.columns) == list(_ORDER)
    assert len(events) == 6 and (events["ppv_kind"] == "component_max").all()
    assert events["frequency_hz"].to_numpy() == pytest.approx([FREQS["vertical"]] * 6, rel=0.005)
    pd.testing.assert_frame_equal(events, read_events(events, columns={c: c for c in _ORDER}))
    law = attenuation.fit(events, "sqrt")
    assert law.beta == pytest.approx(1.6, abs=0.15)
    verdict = criteria.evaluate(events, "PE-MEM-1995", "distance-table")
    # 40 m: 120 > 31.75 (fila 0); 100 m: 27.6 > 25.4 (fila 1); el resto cumple su fila.
    assert list(verdict["status"]) == ["exceeds", "exceeds", "complies", "complies", "complies", "complies"]


def test_events_from_records_needs_one_item_per_record():
    records, distances = _site_records()
    with pytest.raises(ValueError, match="parameter 'distance_m'"):
        waveform.events_from_records(records, distance_m=distances[:5], charge_kg=[100.0] * 6)
    with pytest.raises(ValueError, match="parameter 'charge_kg'"):
        waveform.events_from_records(records, distance_m=distances, charge_kg=100.0)
    with pytest.raises(ValueError, match="parameter 'point'"):
        waveform.events_from_records(records, distance_m=distances, charge_kg=[100.0] * 6, point=["P1"])


def test_events_from_records_keeps_warnings_and_evidence_and_warns_once():
    records, distances = _site_records()
    records = records[:3]
    records[0] = _record(pretrigger_s=None)
    records[2] = _make_record(_channels(), FS, units=GM, pretrigger_s=None, start=None, metadata=None, source="csv",
                              format_evidence="manual-only", warnings=["caller note"])
    with pytest.warns(UserWarning) as caught:
        events = waveform.events_from_records(records, distance_m=distances[:3], charge_kg=[100.0] * 3)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "no pretrigger given: time starts at 0 (records 0, 2)" in message
    assert "caller note (record 2)" in message
    assert events.attrs["warnings"] == ["record 0: no pretrigger given: time starts at 0", "record 2: caller note",
                                        "record 2: no pretrigger given: time starts at 0"]
    assert events.attrs["format_evidence"] == ["user", "user", "manual-only"]


def test_events_from_records_without_warnings_is_silent():
    records, distances = _site_records()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        events = waveform.events_from_records(records, distance_m=distances, charge_kg=[100.0] * 6)
    assert events.attrs["warnings"] == []
    assert events.attrs["format_evidence"] == ["user"] * 6


def test_summary_pickles_and_deep_copies():
    s = waveform.analyze(_record(pretrigger_s=None))
    for c in (pickle.loads(pickle.dumps(s)), copy.deepcopy(s)):
        assert type(c) is waveform.RecordSummary
        assert repr(c) == repr(s)  # mismo contenido (los NaN no se comparan con ==)
        assert isinstance(c.channels, MappingProxyType)
        with pytest.raises(TypeError):
            c.channels["mic"] = None


def test_events_from_records_passes_the_zc_method():
    records, distances = _site_records()
    events = waveform.events_from_records(records[:2], distance_m=distances[:2], charge_kg=[100.0] * 2,
                                          zc_method="instantel")
    expected = waveform.analyze(records[0], zc_method="instantel").channels["vertical"].zc_frequency_hz
    assert events["frequency_hz"].iloc[0] == expected


# --- Frecuencia por FFT ---------------------------------------------------------------------------------------------

def _two_tones(a10, a30, n=1024, fs=1024.0):
    t = np.arange(n) / fs
    return a10 * np.sin(2 * np.pi * 10.0 * t) + a30 * np.sin(2 * np.pi * 30.0 * t)


def test_fft_frequency_keeps_peaks_within_6_db():
    assert waveform.fft_frequency(_two_tones(1.0, 0.6), 1024.0) == (10.0, (10.0, 30.0))
    assert waveform.fft_frequency(_two_tones(1.0, 0.4), 1024.0) == (10.0, (10.0,))
    assert waveform.fft_frequency(_two_tones(0.6, 1.0), 1024.0) == (30.0, (30.0, 10.0))


def test_fft_frequency_of_a_flat_signal_is_nan():
    for x in (np.zeros(64), np.full(64, 0.1)):
        f, peaks = waveform.fft_frequency(x, 1024.0)
        assert math.isnan(f) and peaks == ()


@pytest.mark.parametrize("x", [[5.0, 5.0 + 1e-15, 5.0], [1e16, 1e16 + 2, 1e16, 1e16]])
def test_fft_frequency_of_a_rounding_level_signal_is_nan(x):
    f, peaks = waveform.fft_frequency(np.array(x), 1024.0)
    assert math.isnan(f) and peaks == ()


def test_fft_frequency_domain_errors():
    with pytest.raises(ValueError, match="parameter 'sample_rate_hz'"):
        waveform.fft_frequency(np.ones(8), 0.0)
    with pytest.raises(ValueError, match="parameter 'x'"):
        waveform.fft_frequency(np.array([1.0, np.nan, 2.0]), 1024.0)
    with pytest.raises(ValueError, match="parameter 'x'"):
        waveform.fft_frequency(np.array([1.0]), 1024.0)


# --- Aviso de precisión ---------------------------------------------------------------------------------------------

def _round_sig(x, digits):
    return np.array([float(f"{v:.{digits}g}") for v in x])


def test_precision_warning_with_three_significant_figures_only():
    ch = _channels()
    rounded = {k: (_round_sig(v, 3) if k != "mic" else v) for k, v in ch.items()}
    s = waveform.analyze(_record(rounded))
    assert sum(PRECISION in w for w in s.warnings) == 1
    six = {k: (_round_sig(v, 6) if k != "mic" else v) for k, v in ch.items()}
    s = waveform.analyze(_record(six))
    assert not any(PRECISION in w for w in s.warnings)


def test_fft_peak_in_the_lowest_bins_warns_to_use_the_zero_crossing_frequency():
    # R3: tres pozos de 25 Hz cada 17 ms interfieren y el espectro lo domina la envolvente: 0.5 Hz en un registro de 2 s.
    rec = blastingpy.datasets.synthetic_record(holes=3, delay_ms=17, frequency_hz=25)
    with pytest.warns(UserWarning, match=r"envelope or a trend.*zero-crossing frequency"):
        f, _ = waveform.fft_frequency(rec.channels["transverse"], rec.sample_rate_hz)
    assert f == 0.5
    summary = waveform.analyze(rec)
    low = [w for w in summary.warnings if "envelope or a trend" in w]
    assert len(low) == 1 and "'transverse'" in low[0] and "0.5 Hz" in low[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # una oscilación limpia no avisa
        assert waveform.fft_frequency(_burst(1.0, 20.0), FS)[0] == pytest.approx(20.0, abs=1.0)
    assert not any("envelope" in w for w in waveform.analyze(blastingpy.datasets.synthetic_record()).warnings)
