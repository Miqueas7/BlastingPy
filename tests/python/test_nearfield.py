import math

import numpy as np
import pandas as pd
import pytest

import blastingpy as bp
from blastingpy import airblast, datasets
from blastingpy import attenuation as at
from blastingpy import nearfield as nfl
from blastingpy._blastingpy_core import _nearfield as core

DOMAIN = r"Domain error \[BlastingPy\]: parameter"
FS = 1000.0
LAW = at.make_site_law(at.Law.SquareRoot, 1000.0, 1.6)
ROCK = nfl.Medium(p_velocity_m_s=3000 * math.sqrt(3), s_velocity_m_s=3000, density_kg_m3=2700)


def seed_wave(n=64, scale=3.7):
    t = np.arange(n) / FS
    return scale * np.exp(-60 * t) * np.sin(2 * np.pi * 50 * t)


def table(x, y, charge, firing, **extra):
    return pd.DataFrame({"x_m": x, "y_m": y, "charge_kg": charge, "firing_time_s": firing, **extra})


def median(distance, charge, law=LAW):
    return at.predict_ppv(law, distance, charge, at.Level.Median).value


MEDIAN = "amplitudes are the site law's median: about half of the holes exceed it"


def loud(func, *args, emits=(), **kw):
    """Call a whole-blast function capturing its UserWarnings: the median one always comes (R25), every emitted warning
    is also in the result (R24), and each text of ``emits`` is in exactly one emitted warning."""
    with pytest.warns(UserWarning) as caught:
        res = func(*args, **kw)
    emitted = [str(w.message) for w in caught]
    notes = res.attrs["warnings"] if isinstance(res, pd.DataFrame) else res.warnings
    assert set(emitted) <= set(notes)
    if not isinstance(res, pd.DataFrame):
        assert emitted.count(MEDIAN) == 1
    for text in emits:
        assert sum(text in m for m in emitted) == 1, text
    return res


def simulate(holes, monitor=(100.0, 0.0), **kw):  # no warning capture: for the calls that raise
    kw.setdefault("law", LAW)
    kw.setdefault("wave_velocity_m_s", 2500.0)
    return nfl.simulate_blast(holes, monitor, seed_wave(), sample_rate_hz=FS, **kw)


def blast(holes, monitor=(100.0, 0.0), emits=(), **kw):
    return loud(simulate, holes, monitor, emits=emits, **kw)


def source(gamma=2000.0):
    return nfl.PressureSource(vn_pressure_pa=5e9, n=6, gamma_per_s=gamma)


def column(elements=41, charge_exponent=1.0):
    return nfl.Column(hole_radius_m=0.05, z_bottom=0.0, z_top=5.0, elements=elements, primer_element=1,
                      vod_m_s=5000.0, charge_exponent=charge_exponent)


# --- Module and Holmberg-Persson grids --------------------------------------------------------------------------------

def test_module_and_reexports():
    assert bp.nearfield is nfl
    for name in dir(core):
        if not name.startswith("_"):
            assert getattr(nfl, name) is getattr(core, name) and name in nfl.__all__
    hp = nfl.HOLMBERG_PERSSON_SURFACE
    assert (hp["k"], hp["alpha"], hp["beta"], hp["verification"]) == (700.0, 0.7, 1.5, "V2")
    assert "NIOSH" in hp["citation"] and "Lundborg" in hp["citation"]
    notes = " ".join(hp["notes"])
    assert "not a default" in notes and "rock" in notes
    assert nfl.BLAST_COLUMNS == ("x_m", "y_m", "charge_kg", "firing_time_s")


def test_hp_grid_matches_the_core_point_by_point():
    r, z = [0.0, 0.5, 2.0, 10.0], [-2.0, 0.0, 1.0]
    args = dict(k=700.0, alpha=0.7, beta=1.5, linear_charge_kg_m=1.0, z_start=-1.5, z_end=1.5)
    niosh = nfl.hp_grid("niosh", r=r, z=z, **args)
    original = nfl.hp_grid("original", r=np.array(r), z=np.array(z), **args)
    assert niosh.shape == original.shape == (3, 4)
    for i, zi in enumerate(z):
        for j, rj in enumerate(r):
            assert niosh[i, j] == core.hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, rj, zi)
            if rj == 0 and -1.5 <= zi <= 1.5:
                assert math.isnan(original[i, j])  # on the charged axis the original form diverges
            else:
                assert original[i, j] == core.hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, rj, zi)
    assert niosh[1, 2] == pytest.approx(471, rel=1e-3)  # HP2


def test_hp_grid_axis_is_finite_when_beta_is_below_alpha():
    grid = nfl.hp_grid("original", k=171.0, alpha=1.4, beta=1.0, linear_charge_kg_m=1.0, z_start=-1.85, z_end=1.85,
                       r=[0.0], z=[0.0])
    assert grid[0, 0] == core.hp_original(171, 1.4, 1, 1, -1.85, 1.85, 0, 0)


def test_hp_grid_domain():
    args = dict(k=700.0, alpha=0.7, beta=1.5, linear_charge_kg_m=1.0, z_start=-1.5, z_end=1.5)
    with pytest.raises(ValueError, match=DOMAIN + " 'model'"):
        nfl.hp_grid("modified", r=[1.0], z=[0.0], **args)
    with pytest.raises(ValueError, match=DOMAIN + " 'r'"):
        nfl.hp_grid("niosh", r=[[1.0, 2.0]], z=[0.0], **args)
    with pytest.raises(ValueError, match=DOMAIN + " 'z'"):
        nfl.hp_grid("niosh", r=[1.0], z=0.0, **args)
    with pytest.raises(ValueError, match=DOMAIN + " 'r'"):
        nfl.hp_grid("niosh", r=[-1.0], z=[0.0], **args)
    with pytest.raises(ValueError, match=DOMAIN + " 'k'"):  # constants are checked even if every point diverges
        nfl.hp_grid("original", r=[0.0], z=[0.0], **{**args, "k": -1.0})


# --- Heelan PPV around a hole ----------------------------------------------------------------------------------------

def test_heelan_ppv_is_the_peak_of_the_column_waveform():
    res = nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5)
    wave = core.heelan_column(source(), ROCK, column(), 50.0, 2.5, res.time_s.tolist())
    assert res.ppv_mm_s == core.vector_peak(wave) > 0
    assert res.radial_mm_s.tolist() == wave.radial_mm_s and res.axial_mm_s.tolist() == wave.axial_mm_s
    assert res.scaled_frequency == core.scaled_frequency(source(), ROCK, 0.05)
    assert res.ppv_mm_s == pytest.approx(np.hypot(res.radial_mm_s, res.axial_mm_s).max())
    # far and at low frequency: only the note about the linear sum of P and SV
    assert len(res.warnings) == 1 and "added linearly" in res.warnings[0] and "E37" in res.warnings[0]


def test_heelan_ppv_automatic_time_grid():
    res = nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5)
    gamma = 2000.0
    assert res.time_s[0] == 0 and np.diff(res.time_s) == pytest.approx(1 / (40 * gamma))
    centres = (np.arange(41) + 0.5) * 5.0 / 41
    last = np.max(np.abs(centres - centres[0]) / 5000.0 + np.hypot(50.0, 2.5 - centres) / 3000.0)
    assert res.time_s[-1] == pytest.approx(last + 40 / gamma, abs=1 / (40 * gamma))
    fine = nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5, sample_rate_hz=400 * gamma, duration_s=0.08)
    assert res.ppv_mm_s == pytest.approx(fine.ppv_mm_s, rel=5e-3)  # the automatic grid does not cut the peak
    assert not any("truncated" in w for w in fine.warnings)


def test_heelan_ppv_short_duration_warns_of_truncation():
    full = nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5)
    short = nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5, duration_s=0.010)  # the P wave arrives at 9.6 ms
    assert 0 < short.ppv_mm_s < full.ppv_mm_s
    w = [m for m in short.warnings if "truncated" in m]
    assert len(w) == 1 and "duration_s = 0.01 s" in w[0]
    assert short.time_s[1] == pytest.approx(1 / (40 * 2000.0))  # the step stays automatic


def test_heelan_ppv_truncation_threshold_covers_the_pulse_of_a_large_n():
    # g(tau) lasts about n + 4 sqrt(n) in tau: 10 / gamma after the last arrival cuts the S pulse of n = 14, which
    # dominates at 45 degrees from the column (50, 52.5): the peak falls to 0.73 of the true one
    src, gamma = nfl.PressureSource(vn_pressure_pa=5e9, n=14, gamma_per_s=2000.0), 2000.0
    centres = (np.arange(41) + 0.5) * 5.0 / 41
    last = np.max(np.abs(centres - centres[0]) / 5000.0 + np.hypot(50.0, 52.5 - centres) / 3000.0)
    full = nfl.heelan_ppv(src, ROCK, column(), 50.0, 52.5)
    assert full.time_s[-1] == pytest.approx(last + 40 / gamma, abs=1 / (40 * gamma))  # 40 > 14 + 4 sqrt(14) + 2
    short = nfl.heelan_ppv(src, ROCK, column(), 50.0, 52.5, duration_s=last + 11 / gamma)
    assert short.ppv_mm_s == pytest.approx(0.733 * full.ppv_mm_s, rel=1e-2)  # the silent loss the warning is for
    threshold = last + (14 + 4 * math.sqrt(14) + 2) / gamma
    w = [m for m in short.warnings if "truncated" in m]
    assert len(w) == 1 and f"{threshold:.4g} s" in w[0] and "n + 4 sqrt(n) + 2" in w[0]
    enough = nfl.heelan_ppv(src, ROCK, column(), 50.0, 52.5, duration_s=threshold + 1e-6)
    assert not any("truncated" in m for m in enough.warnings)
    assert enough.ppv_mm_s == pytest.approx(full.ppv_mm_s, rel=1e-6)


def test_heelan_ppv_warns_when_the_time_step_is_too_coarse():
    fine = nfl.heelan_ppv(source(8000.0), ROCK, column(), 50.0, 2.5)
    coarse = nfl.heelan_ppv(source(8000.0), ROCK, column(), 50.0, 2.5, sample_rate_hz=1024.0)
    note = [m for m in coarse.warnings if "under-resolved" in m]
    assert len(note) == 1 and "7.81 / gamma" in note[0]  # 8000 / 1024
    assert coarse.ppv_mm_s < 0.99 * fine.ppv_mm_s
    assert not any("under-resolved" in m for m in fine.warnings)
    at_the_limit = nfl.heelan_ppv(source(8000.0), ROCK, column(), 50.0, 2.5, sample_rate_hz=80000.0)  # 1 / (10 gamma)
    assert not any("under-resolved" in m for m in at_the_limit.warnings)


def test_heelan_ppv_validity_warnings_carry_their_number():
    high = nfl.heelan_ppv(source(60000.0), ROCK, column(), 50.0, 2.5)
    omega = core.scaled_frequency(source(60000.0), ROCK, 0.05)
    w = [m for m in high.warnings if "overestimates" in m]
    assert omega > 0.1 and len(w) == 1 and f"{omega:.3g}" in w[0]
    assert not any("near field" in m for m in high.warnings)

    near = nfl.heelan_ppv(source(), ROCK, column(), 5.0, 2.5)
    number = core.scaled_frequency(source(), ROCK, 0.05) * 5.0 / 0.05
    w = [m for m in near.warnings if "near field" in m]
    assert number < math.pi and len(w) == 1 and f"{number:.3g}" in w[0]
    assert not any("overestimates" in m for m in near.warnings)

    coarse = nfl.heelan_ppv(source(), ROCK, column(elements=5), 50.0, 2.5)
    w = [m for m in coarse.warnings if "half-length" in m]
    assert len(w) == 1 and "= 10" in w[0] and "at least 25 elements" in w[0]  # 0.5 m / 0.05 m; 5 m / (4 * 0.05 m)

    nonlinear = nfl.heelan_ppv(source(), ROCK, column(charge_exponent=0.7), 50.0, 2.5)
    w = [m for m in nonlinear.warnings if "experimental validation" in m]
    assert len(w) == 1 and "0.7" in w[0]
    assert all("added linearly" in " ".join(r.warnings) for r in (high, near, coarse, nonlinear))


def test_heelan_ppv_domain():
    with pytest.raises(ValueError, match=DOMAIN + " 'r'"):
        nfl.heelan_ppv(source(), ROCK, column(), 0.0, 2.5)  # a point on the hole axis
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz'.*explicit"):
        nfl.heelan_ppv(source(1e9), ROCK, column(), 50.0, 2.5)  # 40 gamma: more than 2 000 000 samples
    with pytest.raises(ValueError, match=DOMAIN + " 'duration_s'"):
        nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5, duration_s=-1.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz'"):
        nfl.heelan_ppv(source(), ROCK, column(), 50.0, 2.5, sample_rate_hz=math.nan)
    with pytest.raises(ValueError, match=DOMAIN + " 'vod_m_s'"):
        bad = nfl.Column(hole_radius_m=0.05, z_bottom=0.0, z_top=5.0, elements=41, primer_element=1, vod_m_s=0.0,
                         charge_exponent=1.0)
        nfl.heelan_ppv(source(), ROCK, bad, 50.0, 2.5)


# --- Whole blast: seed superposition from the hole table --------------------------------------------------------------

def test_one_hole_gives_the_median_of_the_law():
    res = blast(table([0.0], [0.0], [50.0], [0.3]))
    assert res.ppv_mm_s == pytest.approx(median(100.0, 50.0), rel=1e-9)
    assert res.time_s[0] == 0 and res.time_s[1] == pytest.approx(1 / FS)
    assert res.velocity_mm_s.shape == res.time_s.shape and res.ppv_mm_s == np.abs(res.velocity_mm_s).max()
    assert res.max_charge_8ms_kg == 50.0 and res.warnings == (MEDIAN,)
    assert res.arrivals["arrival_s"].tolist() == [0.0]  # referred to the first arrival


def test_two_equal_simultaneous_holes_double_the_peak():
    res = blast(table([-5.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.0]), monitor=(0.0, 100.0))
    assert res.ppv_mm_s == pytest.approx(2 * median(math.hypot(5, 100), 50.0), rel=1e-9)


def test_arrivals_table():
    holes = table([0.0, 40.0], [0.0, 0.0], [50.0, 20.0], [0.0, 0.010])
    holes.index = ["H1", "H2"]
    res = blast(holes)
    arr = res.arrivals
    assert list(arr.columns) == ["distance_m", "arrival_s", "amplitude_mm_s", "weight"]
    assert list(arr.index) == ["H1", "H2"]
    assert arr["distance_m"].tolist() == [100.0, 60.0] and arr["weight"].tolist() == [1.0, 1.0]
    # H1: 0 + 100/2500 = 0.040 s; H2: 0.010 + 60/2500 = 0.034 s, the first arrival
    assert arr["arrival_s"].tolist() == pytest.approx([0.006, 0.0], abs=1e-12)
    assert arr["amplitude_mm_s"].tolist() == pytest.approx([median(100, 50), median(60, 20)], rel=1e-12)


def test_max_charge_uses_firing_times_not_arrivals():
    # fired 5 ms apart (one 8 ms window: 30 kg) but arriving 45 ms apart, and a third hole far in time
    res = blast(table([0.0, -100.0, 0.0], [0.0, 0.0, 5.0], [10.0, 20.0, 5.0], [0.0, 0.005, 0.020]))
    assert res.max_charge_8ms_kg == 30.0


def test_damage_without_damage_or_screening_is_linear():
    # fired away from the monitor: no fired hole lies between a hole and the monitor, so nothing screens
    holes = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 60.0, 70.0], [0.0, 0.025, 0.050])
    linear = blast(holes)
    damage = blast(holes, weighting="damage", spacing_m=5.0, eta=0.0)
    assert damage.velocity_mm_s == pytest.approx(linear.velocity_mm_s, rel=1e-12)
    assert damage.arrivals["weight"].tolist() == [1.0, 1.0, 1.0]
    w = " ".join(damage.warnings)
    assert "eta = 0" in w and "23337" in w and "single source" in w and "free faces" in w
    damaged = blast(holes, weighting="damage", spacing_m=5.0)  # eta = 0.2, the author's first estimate
    assert damaged.arrivals["weight"].tolist() == pytest.approx(core.damage_weights(
        [0, 5, 10], [0, 0, 0], [50, 60, 70], [0, 0.025, 0.05], 100, 0, 5, 0.2))
    assert damaged.arrivals["weight"][2] < 1 and damaged.ppv_mm_s < linear.ppv_mm_s


def test_damage_warns_in_the_near_field():
    # the hole nearer to the monitor fires first and screens the one behind it, 35 m away: S = 1 / (1 + 23337 / 35^2)
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.025, 0.0])
    res = blast(holes, monitor=(35.0, 0.0), weighting="damage", spacing_m=5.0, eta=0.0, emits=("near field",))
    s = 1 / (1 + 23337 / 35**2)
    assert s < 0.2 and res.arrivals["weight"].tolist() == pytest.approx([s, 1.0], rel=1e-12)  # no floor (eq. 20)
    w = [m for m in res.warnings if "near field" in m]
    assert len(w) == 1 and "below 0.2 (rows [0])" in w[0] and "closer than 99 m to the monitor (rows [0, 1])" in w[0]
    assert "extrapolation of the source's screening term" in w[0]
    unscreened = blast(holes.assign(firing_time_s=[0.0, 0.025]), monitor=(35.0, 0.0), weighting="damage",
                       spacing_m=5.0, emits=("near field",))  # nothing in the way, only near
    w = [m for m in unscreened.warnings if "near field" in m]
    assert len(w) == 1 and "below 0.2" not in w[0] and "(rows [0, 1])" in w[0]
    far = blast(holes, monitor=(500.0, 0.0), weighting="damage", spacing_m=5.0)
    assert not any("near field" in m for m in far.warnings)
    assert not any("near field" in m for m in blast(holes, monitor=(35.0, 0.0)).warnings)  # linear: no damage weights


def test_charge_scaling_weight_is_c_over_w_to_the_a():
    # FUENTES_SPRINT5: charges 2, 3, 5 kg, A = 0.7, P = 4, r = inf, 0.5, 2 -> C = 2^0.7, ..., 3.017024 (eq. 10).
    # r_n = (h_n / s_P)(mean / w_n) with s_P = 1 and mean = 10/3: h_2 = 0.45 m, h_3 = 3 m. Rows are not in firing order.
    holes = table([3.45, 0.0, 0.45], [0.0, 0.0, 0.0], [5.0, 2.0, 3.0], [0.050, 0.0, 0.025])
    res = blast(holes, monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0, charge_exponent=0.7)
    weight = res.arrivals["weight"].tolist()
    assert weight[1] == pytest.approx(1.0, rel=1e-12)  # first hole to fire: C_1 = w_1^A
    assert weight[0] == pytest.approx(3.017024 / 5.0**0.7, rel=1e-6)
    c = core.charge_scaling_weights([2.0, 3.0, 5.0], [math.inf, 0.45, 3.0], 1.0, 0.7)
    assert weight == pytest.approx([c[2] / 5**0.7, c[0] / 2**0.7, c[1] / 3**0.7], rel=1e-12)
    d = [math.hypot(x, 500.0) for x in (3.45, 0.0, 0.45)]
    assert res.arrivals["amplitude_mm_s"].tolist() == pytest.approx(
        [median(di, wi) * ci for di, wi, ci in zip(d, (5.0, 2.0, 3.0), weight)], rel=1e-12)
    w = " ".join(res.warnings)
    assert "unrealistic" in w and "does not recommend" in w and "free faces" in w


def test_charge_scaling_ignores_holes_fired_at_the_same_time():
    holes = table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.025])
    res = blast(holes, monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0, charge_exponent=0.7)
    c = core.charge_scaling_weights([5.0, 5.0, 5.0], [math.inf, math.inf, 1.0], 1.0, 0.7)
    assert res.arrivals["weight"].tolist() == pytest.approx([ci / 5**0.7 for ci in c], rel=1e-12)


def test_charge_scaling_takes_a_nominal_tie_as_a_tie():
    # 0.1 + 0.2 = 0.30000000000000004 and 0.3 are the same firing time (core time tolerance, 1e-9 s)
    kw = dict(monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0, charge_exponent=0.7)
    noisy = blast(table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.1 + 0.2, 0.3, 0.325]), **kw)
    exact = blast(table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.3, 0.3, 0.325]), **kw)
    assert noisy.arrivals["weight"].tolist() == pytest.approx(exact.arrivals["weight"].tolist(), rel=1e-12)
    assert noisy.arrivals["weight"].tolist()[:2] == pytest.approx([1.0, 1.0], rel=1e-12)


def test_charge_scaling_orders_ties_with_the_time_tolerance():
    # 0.1 + 0.2 is one ulp above 0.3: the same firing time, so the order of the table decides, as in the exact tie
    kw = dict(monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0)

    def weights(second, third):
        holes = table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 3.0, 8.0], [0.0, second, third])
        return blast(holes, **kw).arrivals["weight"].tolist()

    exact = weights(0.3, 0.3)
    c = core.charge_scaling_weights([5.0, 3.0, 8.0], [math.inf, 1.0, 2.0], 1.0, 0.8)
    assert exact == pytest.approx([ci / wi**0.8 for ci, wi in zip(c, (5.0, 3.0, 8.0))], rel=1e-12)
    assert weights(0.1 + 0.2, 0.3) == exact  # a plain sort would fire the third row before the second
    assert weights(0.3, 0.1 + 0.2) == exact


def test_charge_scaling_warns_that_ties_depend_on_the_table_order():
    kw = dict(monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0)
    tied = table([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [5.0, 3.0, 8.0, 4.0], [0.0, 0.3, 0.1 + 0.2, 0.4])
    tied.index = ["H1", "H2", "H3", "H4"]
    res = blast(tied, **kw)
    w = [m for m in res.warnings if "order in the table" in m]
    assert len(w) == 1 and "['H2', 'H3']" in w[0]
    turned = blast(tied.iloc[::-1], **kw)  # the same blast with the rows reversed: the tied weights change
    assert turned.arrivals["weight"]["H2"] != pytest.approx(res.arrivals["weight"]["H2"], rel=1e-3)
    assert turned.arrivals["weight"]["H1"] == res.arrivals["weight"]["H1"]
    apart = blast(table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 3.0, 8.0], [0.0, 0.3, 0.325]), **kw)
    assert not any("order in the table" in m for m in apart.warnings)
    assert not any("order in the table" in m for m in blast(tied, weighting="damage", spacing_m=1.0).warnings)


def test_charge_scaling_takes_the_exponent_from_the_law():
    holes = table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 3.0, 8.0], [0.0, 0.025, 0.050])
    kw = dict(monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0)
    w = np.array([5.0, 3.0, 8.0])
    laws = ((LAW, 0.8),  # square root, beta 1.6: A = beta / 2
            (at.make_site_law(at.Law.CubeRoot, 1000.0, 1.5), 0.5),  # cube root: A = beta / 3
            (at.make_site_law(at.Law.General, 1000.0, 1.6, charge_exponent=0.7), 0.7))  # general: A = a
    for law, a in laws:
        res = blast(holes, law=law, **kw)
        c = np.array(core.charge_scaling_weights(w.tolist(), [math.inf, 1.0, 1.0], 1.0, a))
        assert res.arrivals["weight"].tolist() == pytest.approx((c / w**a).tolist(), rel=1e-12)
        assert not any("charge_exponent" in m for m in res.warnings)
        same = blast(holes, law=law, charge_exponent=a * (1 + 1e-9), **kw)  # the law's own value: no warning
        assert same.warnings == res.warnings and same.ppv_mm_s == pytest.approx(res.ppv_mm_s, rel=1e-8)
    with pytest.raises(ValueError, match=DOMAIN + r" 'law'.*A = 1\.2"):  # A > 1 has no physical meaning
        simulate(holes, law=at.make_site_law(at.Law.SquareRoot, 1000.0, 2.4), **kw)


def test_charge_scaling_warns_when_the_exponent_is_not_the_one_of_the_law():
    # the trap: SiteLaw.charge_exponent of a square-root law is 0.5 (the exponent inside the scaled distance), not A
    holes = table([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [5.0, 3.0, 8.0], [0.0, 0.025, 0.050])
    kw = dict(monitor=(0.0, 500.0), weighting="charge-scaling", spacing_m=1.0)
    assert LAW.charge_exponent == 0.5
    res = blast(holes, charge_exponent=LAW.charge_exponent, **kw)
    note = [m for m in res.warnings if "charge_exponent" in m]
    assert len(note) == 1 and "charge_exponent = 0.5" in note[0] and "A = 0.8" in note[0]
    w = np.array([5.0, 3.0, 8.0])
    c = np.array(core.charge_scaling_weights(w.tolist(), [math.inf, 1.0, 1.0], 1.0, 0.5))
    assert res.arrivals["weight"].tolist() == pytest.approx((c / w**0.5).tolist(), rel=1e-12)  # the given value is used
    assert res.ppv_mm_s != pytest.approx(blast(holes, **kw).ppv_mm_s, rel=1e-3)


def test_arguments_the_weighting_does_not_use_are_reported():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    for kw, name in ((dict(spacing_m=5.0), "spacing_m"), (dict(eta=0.3), "eta"),
                     (dict(charge_exponent=0.8), "charge_exponent"),
                     (dict(weighting="damage", spacing_m=5.0, charge_exponent=0.8), "charge_exponent"),
                     (dict(weighting="charge-scaling", spacing_m=5.0, eta=0.3), "eta")):
        note = [m for m in blast(holes, **kw).warnings if "ignored" in m]
        assert len(note) == 1 and note[0].startswith(name + " is ignored")
    both = [m for m in blast(holes, spacing_m=5.0, eta=0.3).warnings if "ignored" in m]
    assert len(both) == 1 and both[0].startswith("spacing_m and eta are ignored")
    for kw in (dict(), dict(screening=True, spacing_m=5.0), dict(weighting="damage", spacing_m=5.0, eta=0.3)):
        assert not any("ignored" in m for m in blast(holes, **kw).warnings)


def test_firing_times_that_look_like_milliseconds_warn():
    typed_in_ms = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [0.0, 17.0, 34.0])
    res = blast(typed_in_ms, emits=("milliseconds",))
    note = [m for m in res.warnings if "milliseconds" in m]
    assert len(note) == 1 and "34 s" in note[0] and "must be in seconds" in note[0]
    in_seconds = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [0.0, 0.017, 0.034])
    assert blast(in_seconds).warnings == (MEDIAN,)
    assert res.ppv_mm_s != pytest.approx(blast(in_seconds).ppv_mm_s, rel=1e-2)  # the holes no longer overlap
    one_second_apart = table([0.0] * 21, [0.0] * 21, [50.0] * 21, [float(i) for i in range(21)])  # 20 s, 1 s steps
    assert blast(one_second_apart).warnings == (MEDIAN,)
    longer = table([0.0] * 42, [0.0] * 42, [50.0] * 42, [0.5 * i for i in range(42)])  # 0.5 s steps over 20.5 s
    assert any("milliseconds" in m for m in blast(longer, emits=("milliseconds",)).warnings)
    mc = loud(nfl.monte_carlo_blast, typed_in_ms, (100.0, 0.0), seed_wave(), sample_rate_hz=FS, law=LAW,
              wave_velocity_m_s=2500.0, delay_sigma_s=0.001, simulations=2, emits=("milliseconds",))
    assert sum("milliseconds" in m for m in mc.warnings) == 1


def test_a_small_blast_with_firing_times_in_milliseconds_warns():
    # 0, 2 and 4 ms typed as seconds: the span, 4 s, is below 20 s, but a step of 2 s between holes is no delay
    typed_in_ms = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [0.0, 2.0, 4.0])
    res = blast(typed_in_ms, emits=("firing times look like milliseconds",))
    note = [m for m in res.warnings if "milliseconds" in m]
    assert len(note) == 1 and "2 s" in note[0] and "must be in seconds" in note[0]
    in_seconds = blast(table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [0.0, 0.002, 0.004]))
    assert res.ppv_mm_s < 0.5 * in_seconds.ppv_mm_s  # the silent loss the warning is for
    assert in_seconds.warnings == (MEDIAN,)


def test_a_wave_velocity_in_km_per_s_warns():
    holes = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [0.0, 0.002, 0.004])
    res = blast(holes, wave_velocity_m_s=3.0, emits=("wave velocity looks like km/s",))
    note = [m for m in res.warnings if "km/s" in m]
    assert len(note) == 1 and "wave_velocity_m_s = 3" in note[0] and "must be in m/s" in note[0]
    in_m_s = blast(holes, wave_velocity_m_s=3000.0)
    assert res.ppv_mm_s < 0.5 * in_m_s.ppv_mm_s and in_m_s.warnings == (MEDIAN,)
    assert blast(holes, wave_velocity_m_s=100.0).warnings == (MEDIAN,)  # the threshold itself does not warn
    mc = loud(nfl.monte_carlo_blast, holes, (100.0, 0.0), seed_wave(), sample_rate_hz=FS, law=LAW,
              wave_velocity_m_s=3.0, delay_sigma_s=0.001, simulations=2, emits=("km/s",))
    assert sum("km/s" in m for m in mc.warnings) == 1


def test_screening_with_the_linear_weighting():
    # the hole nearer to the monitor fires first and screens the one behind it
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.025, 0.0])
    res = blast(holes, screening=True, spacing_m=5.0)
    arr = res.arrivals
    assert arr["screening_count"].tolist() == [1, 0]
    assert arr["weight"].tolist() == pytest.approx([1 / (1 + 23337 / 100**2), 1.0])  # Blair 1999 eq. 10c
    assert arr["amplitude_mm_s"][0] == pytest.approx(median(100, 50) * arr["weight"][0], rel=1e-12)
    assert "screening" in " ".join(res.warnings) and "screening_count" not in blast(holes).arrivals


def test_screening_is_only_defined_with_the_linear_weighting():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    with pytest.raises(ValueError, match=DOMAIN + " 'screening'.*damage"):
        simulate(holes, weighting="damage", spacing_m=5.0, screening=True)
    with pytest.raises(ValueError, match=DOMAIN + " 'screening'.*charge-scaling"):
        simulate(holes, weighting="charge-scaling", spacing_m=5.0, charge_exponent=0.7, screening=True)


def test_simulate_blast_required_arguments():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    with pytest.raises(ValueError, match=DOMAIN + " 'spacing_m'"):
        simulate(holes, weighting="damage")
    with pytest.raises(ValueError, match=DOMAIN + " 'spacing_m'"):
        simulate(holes, screening=True)
    with pytest.raises(ValueError, match=DOMAIN + " 'charge_exponent'"):
        simulate(holes, weighting="charge-scaling", spacing_m=5.0, charge_exponent=1.2)  # A > 1: no physical meaning
    with pytest.raises(ValueError, match=DOMAIN + " 'charge_exponent'"):
        simulate(holes, weighting="charge-scaling", spacing_m=5.0, charge_exponent="0.8")
    with pytest.raises(ValueError, match=DOMAIN + " 'weighting'"):
        simulate(holes, weighting="nonlinear")
    with pytest.raises(ValueError, match=DOMAIN + " 'wave_velocity_m_s'"):
        simulate(holes, wave_velocity_m_s=0.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'monitor'"):
        simulate(holes, monitor=(math.nan, 0.0))
    with pytest.raises(ValueError, match=DOMAIN + " 'monitor'.*row"):
        simulate(holes, monitor=(5.0, 0.0))  # on a hole
    twice = table([0.0, 0.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    twice.index = ["H1", "H2"]
    for kw in (dict(weighting="damage"), dict(weighting="charge-scaling"), dict(screening=True)):
        with pytest.raises(ValueError, match=DOMAIN + r" 'holes'.*\['H1', 'H2'\].*same point"):
            simulate(twice, spacing_m=5.0, **kw)
    assert blast(twice).ppv_mm_s > 0  # two decks of one hole: fine with the plain linear sum


def test_simulate_blast_table_errors_name_column_and_row():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    with pytest.raises(ValueError, match=DOMAIN + " 'holes' is missing the column 'firing_time_s'"):
        simulate(holes.drop(columns="firing_time_s"))
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'"):
        simulate(holes.iloc[:0])
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'"):
        simulate(holes.to_dict("list"))
    bad = holes.copy()
    bad.index = ["H1", "H2"]
    bad.loc["H2", "charge_kg"] = -1.0
    with pytest.raises(ValueError, match=DOMAIN + r" 'holes'.*'charge_kg'.*\['H2'\]"):
        simulate(bad)
    bad = holes.copy()
    bad.loc[0, "x_m"] = math.nan
    with pytest.raises(ValueError, match=DOMAIN + r" 'holes'.*'x_m'.*\[0\]"):
        simulate(bad)
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'.*seconds"):
        simulate(table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 25000.0]))  # 25 000 s: a 25-million-sample record


def test_simulate_blast_needs_a_vibration_law():
    holes = table([0.0], [0.0], [50.0], [0.0])
    for law in (airblast.make_site_law(2000.0, 1.2), airblast.published(airblast.list_equations()[0]).law):
        with pytest.raises(ValueError, match=DOMAIN + " 'law'.*airblast"):
            simulate(holes, law=law)
    with pytest.raises(ValueError, match=DOMAIN + " 'law'"):
        simulate(holes, law="sqrt")


def test_simulate_blast_repeats_the_warnings_of_the_prediction():
    fitted = at.fit(datasets.synthetic_events(seed=3))  # distances from 50 m to 2000 m
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    res = blast(holes, monitor=(20.0, 0.0), law=fitted)
    assert len(res.warnings) == 2 and res.warnings[1] == MEDIAN  # one per cause, not one per hole
    assert res.warnings[0].startswith("distance is outside the fitted range [")
    assert res.warnings[0].endswith("m: extrapolation (rows [0, 1])")
    assert res.ppv_mm_s > 0
    both = blast(table([0.0, 5.0, 60.0], [0.0, 0.0, 0.0], [50.0, 1.0, 1.0], [0.0, 0.025, 0.05]), monitor=(-50.0, 0.0),
                 law=fitted)  # only the first hole is too near; the second and the third are too light
    assert len(both.warnings) == 3 and both.warnings[0].startswith("distance is outside the fitted range [")
    assert fitted.distance_min_m > 50.0 and both.warnings[0].endswith("m: extrapolation (rows [0])")
    assert both.warnings[1].startswith("charge is outside the fitted range [")
    assert both.warnings[1].endswith("kg: extrapolation (rows [1, 2])")


# --- Seed -------------------------------------------------------------------------------------------------------------

def test_seed_from_a_record_equals_the_same_seed_as_an_array():
    record = datasets.synthetic_record(seed=1, sample_rate_hz=1024.0, duration_s=0.5, pretrigger_s=0.0, mic_pa=None)
    holes = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 60.0, 70.0], [0.0, 0.017, 0.042])
    kw = dict(law=LAW, wave_velocity_m_s=2500.0)
    from_record = loud(nfl.simulate_blast, holes, (100.0, 0.0), (record, "vertical"), **kw)
    from_array = loud(nfl.simulate_blast, holes, (100.0, 0.0), np.array(record.channels["vertical"]),
                      sample_rate_hz=record.sample_rate_hz, **kw)
    assert from_record.velocity_mm_s.tolist() == from_array.velocity_mm_s.tolist()
    assert from_record.ppv_mm_s == from_array.ppv_mm_s and from_record.time_s[1] == 1 / 1024.0
    same_rate = loud(nfl.simulate_blast, holes, (100.0, 0.0), (record, "vertical"), sample_rate_hz=1024.0, **kw)
    assert same_rate.ppv_mm_s == from_record.ppv_mm_s
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz'.*1024"):
        nfl.simulate_blast(holes, (100.0, 0.0), (record, "vertical"), sample_rate_hz=2048.0, **kw)
    with pytest.raises(ValueError, match=DOMAIN + " 'seed'.*'mic'"):
        nfl.simulate_blast(holes, (100.0, 0.0), (record, "mic"), **kw)


def test_seed_as_a_list_and_the_mic_channel():
    record = datasets.synthetic_record(seed=1, sample_rate_hz=1024.0, duration_s=0.5, pretrigger_s=0.0)
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 60.0], [0.0, 0.017])
    kw = dict(law=LAW, wave_velocity_m_s=2500.0)
    as_tuple = loud(nfl.simulate_blast, holes, (100.0, 0.0), (record, "vertical"), **kw)
    as_list = loud(nfl.simulate_blast, holes, (100.0, 0.0), [record, "vertical"], **kw)
    assert as_list.velocity_mm_s.tolist() == as_tuple.velocity_mm_s.tolist() and as_tuple.warnings == (MEDIAN,)
    mic = loud(nfl.simulate_blast, holes, (100.0, 0.0), (record, "mic"), **kw)
    assert len(mic.warnings) == 2 and "'mic'" in mic.warnings[0] and "pressure" in mic.warnings[0]
    sweep = nfl.delay_sweep([record, "mic"], delays_s=[0.0, 0.004])
    assert len(sweep.attrs["warnings"]) == 1 and "pressure" in sweep.attrs["warnings"][0]


def test_a_seed_under_sampled_for_the_superposition_warns():
    fs = 2048.0
    t = np.arange(256) / fs

    def seed(f):
        return np.exp(-150 * t) * np.sin(2 * np.pi * f * t)

    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 60.0], [0.0, 0.017])
    kw = dict(sample_rate_hz=fs, law=LAW, wave_velocity_m_s=2500.0)
    res = loud(nfl.simulate_blast, holes, (100.0, 0.0), seed(500.0), emits=("dominant frequency",), **kw)
    w = [m for m in res.warnings if "dominant frequency" in m]
    assert len(w) == 1 and "204.8 Hz" in w[0] and "higher sample rate" in w[0]  # fs / 10
    # what it warns of: an arrival mid-sample is spread over two samples and the peak of a 500 Hz seed drops
    assert np.abs(nfl.superpose(seed(500.0).tolist(), fs, [0.5 / fs], [1.0])).max() < 0.9 * np.abs(seed(500.0)).max()
    mc = loud(nfl.monte_carlo_blast, holes, (100.0, 0.0), seed(500.0), delay_sigma_s=0.001, simulations=2,
              emits=("dominant frequency",), **kw)
    assert sum("dominant frequency" in m for m in mc.warnings) == 1
    with pytest.warns(UserWarning, match="dominant frequency") as caught:
        sweep = nfl.delay_sweep(seed(500.0), sample_rate_hz=fs, delays_s=[0.0, 0.004])
    assert len(caught) == 1 and str(caught[0].message) in sweep.attrs["warnings"]
    assert loud(nfl.simulate_blast, holes, (100.0, 0.0), seed(150.0), **kw).warnings == (MEDIAN,)  # below fs / 10
    assert nfl.delay_sweep(seed(150.0), sample_rate_hz=fs, delays_s=[0.0]).attrs["warnings"] == ()


def test_seed_is_normalized_to_unit_peak():
    holes = table([0.0], [0.0], [50.0], [0.0])
    big = loud(nfl.simulate_blast, holes, (100.0, 0.0), seed_wave(scale=1000.0), sample_rate_hz=FS, law=LAW,
               wave_velocity_m_s=2500.0)
    assert big.ppv_mm_s == pytest.approx(blast(holes).ppv_mm_s, rel=1e-12)


def test_seed_domain():
    holes = table([0.0], [0.0], [50.0], [0.0])
    kw = dict(law=LAW, wave_velocity_m_s=2500.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz' is required"):
        nfl.simulate_blast(holes, (100.0, 0.0), seed_wave(), **kw)
    with pytest.raises(ValueError, match=DOMAIN + " 'seed'.*zero"):
        nfl.simulate_blast(holes, (100.0, 0.0), np.zeros(16), sample_rate_hz=FS, **kw)
    for bad in ([], [[1.0, 2.0]], [1.0, math.nan], "seed"):
        with pytest.raises(ValueError, match=DOMAIN + " 'seed'"):
            nfl.simulate_blast(holes, (100.0, 0.0), bad, sample_rate_hz=FS, **kw)
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz'"):
        nfl.simulate_blast(holes, (100.0, 0.0), seed_wave(), sample_rate_hz=0.0, **kw)


# --- Monte Carlo ------------------------------------------------------------------------------------------------------

HOLES = table([0.0, 5.0, 10.0, 15.0], [0.0, 0.0, 0.0, 0.0], [50.0, 60.0, 70.0, 80.0], [0.0, 0.017, 0.034, 0.051])


def raw_monte_carlo(holes=HOLES, **kw):  # no warning capture: for the calls that raise
    kw.setdefault("law", LAW)
    kw.setdefault("wave_velocity_m_s", 2500.0)
    return nfl.monte_carlo_blast(holes, (100.0, 0.0), seed_wave(), sample_rate_hz=FS, **kw)


def monte_carlo(holes=HOLES, emits=(), **kw):
    return loud(raw_monte_carlo, holes, emits=emits, **kw)


def test_monte_carlo_without_scatter_is_the_deterministic_peak():
    res = monte_carlo(delay_sigma_s=0.0, simulations=50)
    deterministic = blast(HOLES).ppv_mm_s
    assert res.deterministic_ppv_mm_s == deterministic
    assert res.peaks_mm_s.shape == (50,) and res.peaks_mm_s == pytest.approx(deterministic, rel=1e-12)
    assert res.mean == pytest.approx(deterministic, rel=1e-12) and res.std == pytest.approx(0.0, abs=1e-9)


def test_monte_carlo_summary_is_coherent_with_the_peaks():
    res = monte_carlo(delay_sigma_s=0.002, amplitude_cv=0.25, simulations=500, rng_seed=7)
    peaks = res.peaks_mm_s
    assert peaks.std() > 0 and res.mean == peaks.mean() and res.std == peaks.std(ddof=1)
    assert res.percentile(50) == np.percentile(peaks, 50) and res.percentile(100) == peaks.max()
    level = float(np.sort(peaks)[400])
    assert res.probability_exceeding(level) == pytest.approx(99 / 500)  # strictly above
    assert res.probability_exceeding(peaks.max()) == 0.0 and res.probability_exceeding(0.0) == 1.0
    again = monte_carlo(delay_sigma_s=0.002, amplitude_cv=0.25, simulations=500, rng_seed=7)
    assert again.peaks_mm_s.tolist() == peaks.tolist()
    assert monte_carlo(delay_sigma_s=0.002, amplitude_cv=0.25, simulations=500, rng_seed=8).mean != res.mean
    w = " ".join(res.warnings)
    assert "delay" in w and "distribution" in w and "fluctuating seed" in w
    with pytest.raises(ValueError, match=DOMAIN + " 'q'"):
        res.percentile(101)
    with pytest.raises(ValueError, match=DOMAIN + " 'level_mm_s'"):
        res.probability_exceeding(math.nan)


def test_the_amplitudes_are_said_to_be_the_median_once():
    res = blast(HOLES)  # four holes, one note
    assert res.warnings.count(MEDIAN) == 1
    assert monte_carlo(delay_sigma_s=0.002, simulations=2).warnings.count(MEDIAN) == 1


def test_monte_carlo_without_amplitude_scatter_warns_with_the_equivalent_cv():
    law = at.make_site_law(at.Law.SquareRoot, 1000.0, 1.6, sigma=0.15)
    cv = math.sqrt(math.exp((0.15 * math.log(10)) ** 2) - 1)  # lognormal: 0.356
    blind = monte_carlo(law=law, delay_sigma_s=0.002, simulations=400, rng_seed=2, emits=("amplitude_cv = 0",))
    w = [m for m in blind.warnings if "amplitude_cv = 0" in m]
    assert len(w) == 1 and "sigma = 0.15" in w[0] and f"amplitude_cv={cv:.3g}" in w[0]
    scattered = monte_carlo(law=law, delay_sigma_s=0.002, amplitude_cv=cv, simulations=400, rng_seed=2)
    assert not any("amplitude_cv = 0" in m for m in scattered.warnings)
    assert scattered.percentile(95) > 1.1 * blind.percentile(95)  # the scatter the warning is about
    assert not any("amplitude_cv = 0" in m for m in monte_carlo(delay_sigma_s=0.002, simulations=2).warnings)  # sigma 0


def test_monte_carlo_one_simulation_has_no_standard_deviation():
    res = monte_carlo(delay_sigma_s=0.002, simulations=1)
    assert res.peaks_mm_s.shape == (1,) and math.isnan(res.std) and res.mean == res.peaks_mm_s[0]


def test_monte_carlo_delay_sigma_forms():
    kw = dict(amplitude_cv=0.1, simulations=40, rng_seed=3)
    scalar = monte_carlo(delay_sigma_s=0.003, **kw).peaks_mm_s.tolist()
    assert monte_carlo(delay_sigma_s=np.full(4, 0.003), **kw).peaks_mm_s.tolist() == scalar
    per_hole = [0.001, 0.002, 0.003, 0.004]
    by_array = monte_carlo(delay_sigma_s=per_hole, **kw).peaks_mm_s.tolist()
    by_column = monte_carlo(HOLES.assign(sigma_s=per_hole), delay_sigma_s="sigma_s", **kw).peaks_mm_s.tolist()
    assert by_column == by_array != scalar
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'.*1 s"):
        raw_monte_carlo(delay_sigma_s=1.5)
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        raw_monte_carlo(delay_sigma_s=[0.001, 0.002])
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        raw_monte_carlo(delay_sigma_s=-0.001)
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        raw_monte_carlo(delay_sigma_s="sigma_s")  # no such column
    with pytest.raises(ValueError, match=DOMAIN + " 'simulations'"):
        raw_monte_carlo(delay_sigma_s=0.001, simulations=0)
    with pytest.raises(ValueError, match=DOMAIN + " 'rng_seed'"):
        raw_monte_carlo(delay_sigma_s=0.001, rng_seed=-1)
    with pytest.raises(TypeError, match=r"monte_carlo_blast\(\) got an unexpected keyword argument 'spacing'"):
        raw_monte_carlo(delay_sigma_s=0.001, spacing=5.0)


def test_monte_carlo_uses_the_weights_of_the_deterministic_simulation():
    kw = dict(weighting="damage", spacing_m=5.0)
    res = monte_carlo(delay_sigma_s=0.0, simulations=3, **kw)
    assert res.deterministic_ppv_mm_s == blast(HOLES, **kw).ppv_mm_s
    assert res.peaks_mm_s == pytest.approx(res.deterministic_ppv_mm_s, rel=1e-12)
    assert "single source" in " ".join(res.warnings)
    with pytest.raises(ValueError, match=DOMAIN + " 'screening'"):
        raw_monte_carlo(delay_sigma_s=0.0, screening=True, **kw)


# --- Delay sweep ------------------------------------------------------------------------------------------------------

def test_delay_sweep():
    out = nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[0.0, 0.002, 0.2])
    assert list(out.columns) == ["delay_s", "peak", "ratio_to_simultaneous"]
    assert out["delay_s"].tolist() == [0.0, 0.002, 0.2]
    assert out["peak"].tolist() == pytest.approx([2.0, out["peak"][1], 1.0])
    assert out["ratio_to_simultaneous"][0] == pytest.approx(1.0)
    assert out["ratio_to_simultaneous"][2] == pytest.approx(0.5)  # delay longer than the seed: one seed at a time
    assert 0.5 < out["ratio_to_simultaneous"][1] < 1.0  # 2 ms apart: the two seeds overlap almost in phase
    half = nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[0.0, 0.2], amplitude_ratio=0.5)
    assert half["peak"].tolist() == pytest.approx([1.5, 1.0])
    assert half["ratio_to_simultaneous"].tolist() == pytest.approx([1.0, 1 / 1.5])


def test_delay_sweep_warns_when_delays_look_like_milliseconds():
    out = nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[0.0, 5.0, 13.0])  # 5 and 13 ms typed as s
    note = out.attrs["warnings"]
    assert len(note) == 1 and "milliseconds" in note[0] and "13 s" in note[0] and "must be in seconds" in note[0]
    assert out["ratio_to_simultaneous"].tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[0.0, 0.005, 1.0]).attrs["warnings"] == ()


def test_delay_sweep_from_a_record_and_domain():
    record = datasets.synthetic_record(seed=1, sample_rate_hz=1024.0, duration_s=0.5, pretrigger_s=0.0, mic_pa=None)
    a = nfl.delay_sweep((record, "vertical"), delays_s=[0.0, 0.004, 0.013])
    b = nfl.delay_sweep(np.array(record.channels["vertical"]), sample_rate_hz=1024.0, delays_s=[0.0, 0.004, 0.013])
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match=DOMAIN + " 'delays_s'"):
        nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[-0.001])
    with pytest.raises(ValueError, match=DOMAIN + " 'delays_s'"):
        nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[1e9])
    with pytest.raises(ValueError, match=DOMAIN + " 'amplitude_ratio'"):
        nfl.delay_sweep(seed_wave(), sample_rate_hz=FS, delays_s=[0.0], amplitude_ratio=0.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz' is required"):
        nfl.delay_sweep(seed_wave(), delays_s=[0.0])
