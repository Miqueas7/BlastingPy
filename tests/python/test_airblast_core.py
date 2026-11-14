import math

import pytest

from blastingpy import _blastingpy_core as core
from blastingpy import units

ab = core._airblast


def test_constants():
    assert ab.reference_pressure_pa == 20e-6
    assert core._units.pa_per_psi == 6894.757


def test_pa_to_db_and_back():
    assert ab.pa_to_db(20e-6) == 0.0
    assert ab.pa_to_db(100.0) == pytest.approx(133.9794, abs=1e-4)
    assert ab.db_to_pa(ab.pa_to_db(6900.0)) == pytest.approx(6900.0, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_pa_to_db_domain(bad):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        ab.pa_to_db(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_db_to_pa_domain(bad):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        ab.db_to_pa(bad)


def test_convert_k_to_si():
    k = ab.convert_k_to_si(0.162, 0.794)
    d, w = 100 * units.to_si(1.0, "ft"), units.to_si(1.0, "lb")
    assert k * (d / w ** (1 / 3)) ** -0.794 == pytest.approx(0.162 * 100**-0.794 * 6894.757, rel=1e-12)
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        ab.convert_k_to_si(0.0, 0.794)


def test_pressure_units():
    assert units.to_si(1, "psi") == 6894.757
    assert units.to_si(2, "Pa") == 2
    assert units.from_si(1000, "kPa") == 1
