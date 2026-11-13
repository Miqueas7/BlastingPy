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
