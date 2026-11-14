import math

import numpy as np
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


def test_heelan_factors():
    assert [nf.mean_frequency_factor(n) for n in (4, 6, 8)] == pytest.approx([0.5, 0.375, 0.3125], abs=5e-4)
    assert nf.g_peak(6) == pytest.approx(21.9602, rel=1e-4)
    assert nf.g_function(n=6, tau=-1.0) == 0.0


def test_heelan_element_from_python():
    rock = nf.Medium(p_velocity_m_s=3000 * math.sqrt(3), s_velocity_m_s=3000, density_kg_m3=2700)
    source = nf.PressureSource(vn_pressure_pa=5e9, n=6, gamma_per_s=20000)
    assert (rock.s_velocity_m_s, source.n) == (3000, 6)
    with pytest.raises(AttributeError):
        source.n = 4
    assert nf.pressure_function(source, 6 / 20000) == pytest.approx(5e9, rel=1e-12)
    r, a, half = 10.0, 0.05, 0.1
    gamma, vp = 20000, rock.p_velocity_m_s
    t = np.arange(r / vp - 1e-4, r / vp + 30 / gamma, 0.002 / gamma)
    w = nf.heelan_element(source, rock, hole_radius_m=a, half_length_m=half, r=r, z=0.0, t_s=t)
    assert len(w.radial_mm_s) == len(w.axial_mm_s) == len(t)
    want = nf.heelan_amplitude(source, rock, a, half) * nf.g_peak(6) * a / r
    assert max(abs(v) for v in w.radial_mm_s) == pytest.approx(want, rel=1e-3)
    assert nf.scaled_frequency(source, rock, a) > 0


def test_heelan_domain():
    rock = nf.Medium(5000, 3000, 2700)
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        nf.heelan_element(nf.PressureSource(5e9, 6, 2e4), rock, 0.05, 0.1, 0.0, 1.0, [0.0, 1e-3])
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        nf.g_peak(21)
