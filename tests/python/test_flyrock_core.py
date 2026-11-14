import math

import pytest

from blastingpy import _blastingpy_core as core

fr = core._flyrock


def test_ballistics_and_rm():
    assert fr.standard_gravity == 9.80665
    assert fr.velocity_for_range(95, 45, 9.8) == pytest.approx(30.5, rel=0.002)
    assert fr.range_to_height(31.30, 75, 10) == pytest.approx(47.1, rel=0.005)
    assert fr.max_height(30.51, 45, 9.8) == pytest.approx(23.7, rel=0.01)
    assert fr.rm_face_burst(27, 11, 2.6, g=9.8) == pytest.approx(140, rel=0.01)
    assert fr.rm_cratering(13.5, 21.4, 1.6, 9.8) == pytest.approx(295, rel=0.01)
    assert fr.rm_stemming_ejection(27, 11, 3.0, 45, g=9.8) == pytest.approx(96.5, abs=0.5)
    assert fr.rm_stemming_ejection(27, 11, 3.0, 0) == 0.0
    assert fr.rm_min_confinement(13.5, 38.8, 50, 9.8) == pytest.approx(4.3, abs=0.05)
    v = fr.rm_launch_velocity(27, 11, 2.6)
    assert fr.max_range_level(v, 9.8) == pytest.approx(fr.rm_face_burst(27, 11, 2.6, 9.8), rel=1e-12)
    assert fr.range_level(v, 30) == pytest.approx(v * v * math.sin(math.radians(60)) / fr.standard_gravity)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_domain(bad):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        fr.rm_face_burst(27, 11, bad)


def test_domain_angles_and_height():
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        fr.range_level(10, 90)
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        fr.range_to_height(10, 45, 100)
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        fr.rm_stemming_ejection(27, 11, 3, 90)
