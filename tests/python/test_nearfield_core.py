import math

import pytest

from blastingpy import _blastingpy_core as core

nf = core._nearfield


def test_hp_published_values():
    assert nf.hp_mean_distance(-1.5, 1.5, 2, 0) == pytest.approx(2.174, rel=1e-4)
    assert nf.hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, 2, 0) == pytest.approx(471, rel=1e-3)
    assert nf.hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 0.5, 0) == pytest.approx(2230, rel=0.02)
    assert math.log(nf.hp_term(4.98, 0, 29, 17.3, 14.5)) == pytest.approx(-0.9129, abs=1e-3)
    assert nf.hp_point(700, 0.7, 1.5, 3, 500) > 0


def test_damage_extent_and_critical():
    r = nf.hp_damage_extent(nf.HpModel.Original, 456, 1.12, 2.24, 4.98, 0, 29, 14.5, 848)
    assert r == pytest.approx(6.5, abs=0.1)
    r = nf.hp_damage_extent(model=nf.HpModel.Niosh, k=171, alpha=1.4, beta=1, linear_charge_kg_m=1.27,
                            z_start=-1.85, z_end=1.85, z=0, ppv_limit_mm_s=1000)
    assert r == pytest.approx(1.1, abs=0.06)
    assert nf.ppv_critical(4.7e6, 3716, 9.6e9) == pytest.approx(1819, rel=1e-3)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_domain(bad):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        nf.hp_niosh(bad, 0.7, 1.5, 1, -1.5, 1.5, 2, 0)


def test_axis_inside_charge_is_domain_error():
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        nf.hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 0, 0)
