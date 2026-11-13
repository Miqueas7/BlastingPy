import numpy as np
import pytest

from blastingpy._blastingpy_core import _waveform as wf


def test_peak_abs_accepts_numpy_and_reads_fields():
    p = wf.peak_abs(np.array([1.0, -7.0, 3.0]))
    assert (p.value, p.index) == (7.0, 1)


def test_peak_fields_are_read_only():
    p = wf.peak_abs(np.array([1.0, -2.0]))
    with pytest.raises(AttributeError):
        p.value = 1.0


def test_vector_sum_peak_and_pseudo_sum():
    t = np.array([5.0, 0.0, 0.0])
    v = np.array([0.0, 5.0, 0.0])
    l = np.array([0.0, 0.0, 5.0])
    assert wf.vector_sum_peak(t, v, l).value == pytest.approx(5.0)
    assert wf.pseudo_vector_sum(5.0, 5.0, 5.0) == pytest.approx(8.660254, abs=1e-6)


def test_domain_errors_name_the_parameter():
    with pytest.raises(ValueError, match="parameter 'x'"):
        wf.peak_abs(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="parameter 'v'"):
        wf.vector_sum_peak(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="parameter 'peak_l'"):
        wf.pseudo_vector_sum(1.0, 1.0, -1.0)


def _mid_peak_sine(f, fs, n=2048):
    t_mid = (n // 2) / fs
    phi = np.pi / 2 - 2 * np.pi * f * t_mid
    return 10.0 * np.sin(2 * np.pi * f * np.arange(n) / fs + phi)


def test_zero_crossing_frequency_and_enum_values_exposed():
    x = _mid_peak_sine(40.0, 1024.0)
    r = wf.zero_crossing_frequency(x, 1024.0, 1024, wf.ZcMethod.Interpolated)
    assert r.status == wf.ZcStatus.Ok
    assert r.frequency_hz == pytest.approx(40.0, rel=0.005)
    assert r.half_period_s == pytest.approx(0.0125, rel=0.005)
    assert {s.name for s in wf.ZcStatus.__members__.values()} == {"Ok", "AboveLimit", "Incomplete", "NotComputable"}
    assert {m.name for m in wf.ZcMethod.__members__.values()} == {"WholeSamples", "Interpolated"}
    assert wf.instantel_zc_limit_hz(2048.0) == pytest.approx(200.0)
    above = wf.zero_crossing_frequency(_mid_peak_sine(150.0, 1024.0), 1024.0, 1024, wf.ZcMethod.WholeSamples)
    assert above.status == wf.ZcStatus.AboveLimit


def test_zero_crossing_incomplete_has_nan_and_fields_are_read_only():
    x = np.cos(2 * np.pi * 10.0 * np.arange(512) / 1024.0)
    r = wf.zero_crossing_frequency(x, 1024.0, 0, wf.ZcMethod.WholeSamples)
    assert r.status == wf.ZcStatus.Incomplete
    assert np.isnan(r.frequency_hz) and np.isnan(r.half_period_s)
    with pytest.raises(AttributeError):
        r.frequency_hz = 1.0


def test_zero_crossing_domain_errors():
    x = np.array([-1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="parameter 'sample_rate_hz'"):
        wf.zero_crossing_frequency(x, 0.0, 1, wf.ZcMethod.WholeSamples)
    with pytest.raises(ValueError, match="parameter 'peak_index'"):
        wf.zero_crossing_frequency(x, 1024.0, 3, wf.ZcMethod.WholeSamples)
