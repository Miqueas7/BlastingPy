import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from blastingpy import units


@given(st.floats(1e-3, 1e6), st.sampled_from(["ft", "lb", "in/s", "in"]))
def test_round_trip(value, unit):
    assert units.from_si(units.to_si(value, unit), unit) == pytest.approx(value, rel=1e-12)


def test_known_factors():
    assert units.to_si(1.0, "ft") == 0.3048
    assert units.to_si(1.0, "lb") == 0.45359237
    assert units.to_si(1.0, "in/s") == 25.4
    assert units.to_si(1.0, "in") == 25.4
    assert units.from_si(0.3048, "ft") == pytest.approx(1.0, rel=1e-15)


@pytest.mark.parametrize("unit", ["m", "kg", "mm/s", "mm"])
def test_si_units_are_identity(unit):
    assert units.to_si(3.5, unit) == 3.5
    assert units.from_si(3.5, unit) == 3.5


def test_scalar_returns_float_and_array_returns_array():
    assert isinstance(units.to_si(2.0, "ft"), float)
    arr = np.array([1.0, 2.0, 3.0])
    out = units.to_si(arr, "ft")
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    np.testing.assert_allclose(out, arr * 0.3048, rtol=1e-15)
    np.testing.assert_allclose(units.from_si(out, "ft"), arr, rtol=1e-12)


def test_unknown_unit():
    with pytest.raises(ValueError, match="unit"):
        units.to_si(1.0, "yd")
    with pytest.raises(ValueError, match="unit"):
        units.from_si(1.0, "yd")


def test_core_functions_are_reexported():
    assert units.scaled_distance(100.0, 25.0, 0.5) == 20.0
    assert units.convert_k_to_si(160.0, 1.6, 0.5) == pytest.approx(1143.0, abs=0.5)
    with pytest.raises(ValueError, match="Domain error"):
        units.scaled_distance(0.0, 25.0, 0.5)
