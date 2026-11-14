import pytest

from blastingpy import attenuation
from blastingpy.datasets import synthetic_events

COLUMNS = ["distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz"]


def test_same_seed_same_table_other_seed_differs():
    a, b, c = synthetic_events(seed=1), synthetic_events(seed=1), synthetic_events(seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_canonical_columns_and_positive():
    ev = synthetic_events(n=25)
    assert list(ev.columns) == COLUMNS and len(ev) == 25
    assert (ev.ppv_kind == "component_max").all()
    assert (ev[["distance_m", "charge_kg", "ppv_mm_s", "frequency_hz"]] > 0).all().all()
    assert ev.distance_m.between(50, 2000).all() and ev.frequency_hz.between(5, 60).all()


@pytest.mark.parametrize("law", ["sqrt", "cbrt"])
def test_noiseless_fit_recovers_parameters(law):
    f = attenuation.fit(synthetic_events(law, sigma=0.0, k=800.0, beta=1.4), law)
    assert f.k == pytest.approx(800.0, rel=1e-6) and f.beta == pytest.approx(1.4, rel=1e-6)


@pytest.mark.parametrize("law", ["sqrt", "cbrt"])
def test_noisy_fit_recovers_beta_and_sigma(law):
    f = attenuation.fit(synthetic_events(law, n=400, sigma=0.15, seed=3), law)
    assert f.beta == pytest.approx(1.6, abs=0.05) and f.sigma == pytest.approx(0.15, abs=0.02)


@pytest.mark.parametrize("kwargs", [
    dict(law="general"), dict(n=0), dict(k=0.0), dict(beta=-1.0), dict(sigma=-0.1),
    dict(distance_range=(10.0, 10.0)), dict(charge_range=(0.0, 5.0)), dict(frequency_range=(60.0, 5.0)),
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        synthetic_events(**kwargs)


# --------------------------------------------------------------------------- synthetic_record
import numpy as np  # noqa: E402

from blastingpy import io, waveform  # noqa: E402
from blastingpy.datasets import synthetic_record  # noqa: E402

GEO = ("transverse", "vertical", "longitudinal")


def test_record_same_seed_same_record_other_seed_differs():
    a, b, c = synthetic_record(seed=3, noise_mm_s=0.1), synthetic_record(seed=3, noise_mm_s=0.1), synthetic_record(seed=4)
    for ch in a.channels:
        assert np.array_equal(a.channels[ch], b.channels[ch])
    assert not np.array_equal(a.channels["vertical"], c.channels["vertical"])


def test_record_shape_units_and_metadata():
    r = synthetic_record(sample_rate_hz=1000.0, duration_s=1.5, pretrigger_s=0.2)
    assert r.n_samples == 1500 and r.sample_rate_hz == 1000.0 and r.pretrigger_s == 0.2
    assert set(r.channels) == {*GEO, "mic"}
    assert r.source == "arrays" and r.format_evidence == "user" and r.warnings == ()
    assert r.metadata["generator"] == "synthetic_record" and r.metadata["seed"] == "0"
    assert r.time_s()[0] == pytest.approx(-0.2)


def test_record_exact_ppv_per_geophone_even_with_noise():
    r = synthetic_record(ppv_mm_s=12.5, mic_pa=30.0, noise_mm_s=0.5, seed=7)
    for ch in GEO:
        assert np.abs(r.channels[ch]).max() == pytest.approx(12.5, rel=1e-12)
    assert np.abs(r.channels["mic"]).max() == pytest.approx(30.0, rel=1e-12)


def test_record_noise_is_in_mm_s_on_the_geophones_only():
    # R19: el ruido (mm/s) se suma a los geófonos ya escalados; el pretrigger (sin señal) muestra su desviación.
    kw = dict(seed=5, sample_rate_hz=4096.0, duration_s=1.0, pretrigger_s=0.5, ppv_mm_s=10.0, mic_pa=30.0)
    clean, noisy = synthetic_record(**kw), synthetic_record(noise_mm_s=0.05, **kw)
    before = noisy.time_s() < 0
    for ch in GEO:
        assert np.abs(noisy.channels[ch]).max() == pytest.approx(10.0, rel=1e-12)
        assert np.std(noisy.channels[ch][before]) == pytest.approx(0.05, rel=0.1)
    np.testing.assert_array_equal(noisy.channels["mic"], clean.channels["mic"])


def test_record_without_mic():
    assert "mic" not in synthetic_record(mic_pa=None).channels


def test_record_quantized_infers_units_and_peak_within_half_step():
    r = synthetic_record(quantize=True, ppv_mm_s=10.0, mic_pa=50.0)
    units, _ = io.infer_units(r.channels)
    assert units == {"geophone": "mm/s", "mic": "Pa"}
    for ch in GEO:
        assert abs(np.abs(r.channels[ch]).max() - 10.0) <= 0.00788 / 2 + 1e-12
    assert abs(np.abs(r.channels["mic"]).max() - 50.0) <= 0.0156 / 2 + 1e-12


def test_record_analyze_recovers_frequency_single_hole():
    s = waveform.analyze(synthetic_record(frequency_hz=20.0, sample_rate_hz=4096.0))
    for ch in GEO:
        assert s.channels[ch].zc_status == "ok"
        assert s.channels[ch].zc_frequency_hz == pytest.approx(20.0, rel=0.03)
        assert s.channels[ch].peak == pytest.approx(10.0)


def test_record_four_holes_envelope_peaks_spaced_by_delay():
    fs, delay = 4096.0, 25.0
    r = synthetic_record(holes=4, delay_ms=delay, frequency_hz=100.0, damping=0.5, sample_rate_hz=fs, duration_s=1.0)
    t, x = r.time_s(), np.abs(r.channels["vertical"])
    times = []
    for k in range(4):
        lo, hi = k * delay / 1000.0, (k + 1) * delay / 1000.0
        w = np.flatnonzero((t >= lo - 1e-12) & (t < hi - 1e-12))
        times.append(t[w[np.argmax(x[w])]])
    assert np.diff(times) == pytest.approx(delay / 1000.0, abs=1.0 / fs)


@pytest.mark.parametrize("kw", [dict(holes=0), dict(holes=1.5), dict(sample_rate_hz=0.0), dict(sample_rate_hz=float("nan")),
                                dict(duration_s=0.2), dict(duration_s=0.3, holes=4, delay_ms=100.0),
                                dict(pretrigger_s=-1.0), dict(delay_ms=0.0), dict(frequency_hz=-1.0),
                                dict(ppv_mm_s=0.0), dict(damping=-0.1), dict(mic_pa=0.0), dict(noise_mm_s=-1.0),
                                dict(seed=1.5), dict(quantize=1)])
def test_record_domain_errors(kw):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter '"):
        synthetic_record(**kw)
