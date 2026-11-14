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


def test_lundborg_mckenzie_roth():
    assert fr.lundborg_max_range(127) == pytest.approx(761, rel=0.005)
    assert fr.lundborg_optimal_fragment(127) == pytest.approx(0.293, rel=0.005)
    assert fr.lundborg_range(101.6, 1.35) == pytest.approx(658, rel=0.005)
    assert fr.mckenzie_max_range(0.596, 127) == pytest.approx(fr.lundborg_max_range(127), rel=0.03)
    g = fr.roth_gurney_velocity(4273, fr.Explosive.Anfo)
    assert g == pytest.approx(1880, rel=0.001)
    cm = fr.roth_charge_mass_ratio(10, 2700, 3)
    assert fr.roth_launch_velocity(g, cm) == pytest.approx(g * math.sqrt(cm))
    rock = fr.RothCalibration.Granite
    assert fr.roth_calibrated_velocity(rock, 5e-4) == pytest.approx(math.sqrt(3.487e6 * 5e-4 - 584), rel=1e-12)
    assert fr.roth_reference_gurney_velocity(rock) == 2300
    assert math.isnan(fr.roth_reference_gurney_velocity(fr.RothCalibration.GraniteAllData))


def test_lundborg_domain():
    for bad in (5, 0.127):
        with pytest.raises(ValueError, match="millimetres"):
            fr.lundborg_max_range(bad)
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        fr.lundborg_range(100, 0.2)
    with pytest.raises(ValueError, match="minimum c/m"):
        fr.roth_calibrated_velocity(fr.RothCalibration.Granite, 1e-4)


def test_info():
    ids = {m: fr.info(m) for m in fr.Model.__members__.values()}
    assert len(ids) == 9
    assert fr.info(fr.Model.LundborgMaxRange).verification == core.Verification.Secondary
    assert fr.info(fr.Model.RothCalibrated).method_id == "FLY-ROTH-CAL"
    assert "version 2018, coefficient 10" in fr.info(fr.Model.McKenzie2018).source
