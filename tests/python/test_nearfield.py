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


def blast(holes, monitor=(100.0, 0.0), **kw):
    kw.setdefault("law", LAW)
    kw.setdefault("wave_velocity_m_s", 2500.0)
    return nfl.simulate_blast(holes, monitor, seed_wave(), sample_rate_hz=FS, **kw)


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
    assert res.max_charge_8ms_kg == 50.0 and res.warnings == ()
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
        blast(holes, weighting="damage", spacing_m=5.0, screening=True)
    with pytest.raises(ValueError, match=DOMAIN + " 'screening'.*charge-scaling"):
        blast(holes, weighting="charge-scaling", spacing_m=5.0, charge_exponent=0.7, screening=True)


def test_simulate_blast_required_arguments():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    with pytest.raises(ValueError, match=DOMAIN + " 'spacing_m'"):
        blast(holes, weighting="damage")
    with pytest.raises(ValueError, match=DOMAIN + " 'spacing_m'"):
        blast(holes, screening=True)
    with pytest.raises(ValueError, match=DOMAIN + " 'charge_exponent'"):
        blast(holes, weighting="charge-scaling", spacing_m=5.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'charge_exponent'"):
        blast(holes, weighting="charge-scaling", spacing_m=5.0, charge_exponent=1.2)  # A > 1: no physical meaning
    with pytest.raises(ValueError, match=DOMAIN + " 'weighting'"):
        blast(holes, weighting="nonlinear")
    with pytest.raises(ValueError, match=DOMAIN + " 'wave_velocity_m_s'"):
        blast(holes, wave_velocity_m_s=0.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'monitor'"):
        blast(holes, monitor=(math.nan, 0.0))
    with pytest.raises(ValueError, match=DOMAIN + " 'monitor'.*row"):
        blast(holes, monitor=(5.0, 0.0))  # on a hole
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'.*same point"):
        blast(table([0.0, 0.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025]), weighting="damage", spacing_m=5.0)


def test_simulate_blast_table_errors_name_column_and_row():
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    with pytest.raises(ValueError, match=DOMAIN + " 'holes' is missing the column 'firing_time_s'"):
        blast(holes.drop(columns="firing_time_s"))
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'"):
        blast(holes.iloc[:0])
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'"):
        blast(holes.to_dict("list"))
    bad = holes.copy()
    bad.index = ["H1", "H2"]
    bad.loc["H2", "charge_kg"] = -1.0
    with pytest.raises(ValueError, match=DOMAIN + r" 'holes'.*'charge_kg'.*\['H2'\]"):
        blast(bad)
    bad = holes.copy()
    bad.loc[0, "x_m"] = math.nan
    with pytest.raises(ValueError, match=DOMAIN + r" 'holes'.*'x_m'.*\[0\]"):
        blast(bad)
    with pytest.raises(ValueError, match=DOMAIN + " 'holes'.*seconds"):
        blast(table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 25000.0]))  # 25 000 s: a 25-million-sample record


def test_simulate_blast_needs_a_vibration_law():
    holes = table([0.0], [0.0], [50.0], [0.0])
    for law in (airblast.make_site_law(2000.0, 1.2), airblast.published(airblast.list_equations()[0]).law):
        with pytest.raises(ValueError, match=DOMAIN + " 'law'.*airblast"):
            blast(holes, law=law)
    with pytest.raises(ValueError, match=DOMAIN + " 'law'"):
        blast(holes, law="sqrt")


def test_simulate_blast_repeats_the_warnings_of_the_prediction():
    fitted = at.fit(datasets.synthetic_events(seed=3))  # distances from 50 m to 2000 m
    holes = table([0.0, 5.0], [0.0, 0.0], [50.0, 50.0], [0.0, 0.025])
    res = blast(holes, monitor=(20.0, 0.0), law=fitted)
    assert len(res.warnings) == 2 and all("extrapolation" in w for w in res.warnings)
    assert "(rows [0])" in res.warnings[0] and "(rows [1])" in res.warnings[1]
    assert res.ppv_mm_s > 0


# --- Seed -------------------------------------------------------------------------------------------------------------

def test_seed_from_a_record_equals_the_same_seed_as_an_array():
    record = datasets.synthetic_record(seed=1, sample_rate_hz=1024.0, duration_s=0.5, pretrigger_s=0.0, mic_pa=None)
    holes = table([0.0, 5.0, 10.0], [0.0, 0.0, 0.0], [50.0, 60.0, 70.0], [0.0, 0.017, 0.042])
    kw = dict(law=LAW, wave_velocity_m_s=2500.0)
    from_record = nfl.simulate_blast(holes, (100.0, 0.0), (record, "vertical"), **kw)
    from_array = nfl.simulate_blast(holes, (100.0, 0.0), np.array(record.channels["vertical"]),
                                    sample_rate_hz=record.sample_rate_hz, **kw)
    assert from_record.velocity_mm_s.tolist() == from_array.velocity_mm_s.tolist()
    assert from_record.ppv_mm_s == from_array.ppv_mm_s and from_record.time_s[1] == 1 / 1024.0
    same_rate = nfl.simulate_blast(holes, (100.0, 0.0), (record, "vertical"), sample_rate_hz=1024.0, **kw)
    assert same_rate.ppv_mm_s == from_record.ppv_mm_s
    with pytest.raises(ValueError, match=DOMAIN + " 'sample_rate_hz'.*1024"):
        nfl.simulate_blast(holes, (100.0, 0.0), (record, "vertical"), sample_rate_hz=2048.0, **kw)
    with pytest.raises(ValueError, match=DOMAIN + " 'seed'.*'mic'"):
        nfl.simulate_blast(holes, (100.0, 0.0), (record, "mic"), **kw)


def test_seed_is_normalized_to_unit_peak():
    holes = table([0.0], [0.0], [50.0], [0.0])
    big = nfl.simulate_blast(holes, (100.0, 0.0), seed_wave(scale=1000.0), sample_rate_hz=FS, law=LAW,
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


def monte_carlo(holes=HOLES, **kw):
    kw.setdefault("law", LAW)
    kw.setdefault("wave_velocity_m_s", 2500.0)
    return nfl.monte_carlo_blast(holes, (100.0, 0.0), seed_wave(), sample_rate_hz=FS, **kw)


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
        monte_carlo(delay_sigma_s=1.5)
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        monte_carlo(delay_sigma_s=[0.001, 0.002])
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        monte_carlo(delay_sigma_s=-0.001)
    with pytest.raises(ValueError, match=DOMAIN + " 'delay_sigma_s'"):
        monte_carlo(delay_sigma_s="sigma_s")  # no such column
    with pytest.raises(ValueError, match=DOMAIN + " 'simulations'"):
        monte_carlo(delay_sigma_s=0.001, simulations=0)
    with pytest.raises(ValueError, match=DOMAIN + " 'rng_seed'"):
        monte_carlo(delay_sigma_s=0.001, rng_seed=-1)
    with pytest.raises(TypeError, match=r"monte_carlo_blast\(\) got an unexpected keyword argument 'spacing'"):
        monte_carlo(delay_sigma_s=0.001, spacing=5.0)


def test_monte_carlo_uses_the_weights_of_the_deterministic_simulation():
    kw = dict(weighting="damage", spacing_m=5.0)
    res = monte_carlo(delay_sigma_s=0.0, simulations=3, **kw)
    assert res.deterministic_ppv_mm_s == blast(HOLES, **kw).ppv_mm_s
    assert res.peaks_mm_s == pytest.approx(res.deterministic_ppv_mm_s, rel=1e-12)
    assert "single source" in " ".join(res.warnings)
    with pytest.raises(ValueError, match=DOMAIN + " 'screening'"):
        monte_carlo(delay_sigma_s=0.0, screening=True, **kw)


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
