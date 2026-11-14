import math
import warnings

import numpy as np
import pandas as pd
import pytest

import blastingpy as bp
from blastingpy import flyrock as fl


def holes(**extra):
    df = pd.DataFrame({
        "diameter_mm": [165.0, 165.0, 165.0], "burden_m": [2.6, 2.6, 2.6], "stemming_m": [1.6, 3.0, 3.0],
        "linear_charge_kg_m": [35.6, 11.0, 11.0], "charge_length_m": [0.6, 5.0, 5.0],
        "hole_angle_from_vertical_deg": [0.0, 20.0, 0.0]})
    for k, v in extra.items():
        df[k] = v
    return df


def test_reexports_and_module():
    assert bp.flyrock is fl and fl.rm_face_burst(27, 11, 2.6, 9.8) == pytest.approx(140, rel=0.01)
    assert fl.info(fl.Model.McKenzie2018).method_id == "FLY-MCKENZIE-2018"


def test_linear_charge():
    assert fl.linear_charge(165, 1150) == pytest.approx(24.59, rel=1e-3)
    with pytest.raises(ValueError, match="parameter 'diameter_mm'"):
        fl.linear_charge(-1, 1150)


def test_compare_table():
    out = fl.compare(holes(), k=13.5, g=9.8)
    assert out["rm_face_burst_m"][1] == pytest.approx(fl.rm_face_burst(13.5, 11, 2.6, 9.8))
    assert out["rm_charge"][0] == pytest.approx(21.36)
    assert out["rm_cratering_m"][0] == pytest.approx(295, rel=0.01)  # E3: actual mass
    assert out["rm_stemming_ejection_m"][2] == 0 and out["rm_stemming_ejection_m"][1] > 0
    assert out["rm_max_m"][2] == max(out["rm_face_burst_m"][2], out["rm_cratering_m"][2])
    assert out["rm_max_m"][1] >= out["rm_stemming_ejection_m"][1]
    assert out["lundborg_max_m"][0] == pytest.approx(fl.lundborg_max_range(165))
    assert "mckenzie_2018_m" not in out and "lundborg_q_m" not in out
    w = " ".join(out.attrs["warnings"])
    assert "shorter than 1 m" in w and "vertical" in w
    assert out.attrs["k"] == 13.5 and out.attrs["g"] == 9.8 and out.attrs["critiques"]
    meth = out.attrs["methods"]
    assert set(meth) == {"rm_face_burst_m", "rm_cratering_m", "rm_stemming_ejection_m", "lundborg_max_m"}
    assert meth["lundborg_max_m"]["verification"] == "V2" and meth["rm_face_burst_m"]["verification"] == "V1"
    assert all(v["citation"] and v["id"] for v in meth.values())
    assert "2 to 8" in " ".join(out.attrs["critiques"])


def test_compare_optional_models():
    out = fl.compare(holes(sdob_m_kg13=0.6, powder_factor_kg_m3=1.0), k=13.5)
    assert out["mckenzie_2018_m"][0] == pytest.approx(fl.mckenzie_max_range(0.6, 165))
    with pytest.warns(UserWarning, match="underestimate"):  # 743 m, below Lundborg's 905 m maximum
        assert out["lundborg_q_m"][0] == pytest.approx(fl.lundborg_range(165, 1.0))
    assert {"mckenzie_2018_m", "lundborg_q_m"} <= set(out.attrs["methods"])


def test_compare_defaults_and_derived_charge():
    df = holes().drop(columns=["charge_length_m", "hole_angle_from_vertical_deg", "linear_charge_kg_m"])
    df["explosive_density_kg_m3"] = 1150.0
    out = fl.compare(df, k=13.5)
    assert out["linear_charge_kg_m"][0] == pytest.approx(24.59, rel=1e-3)
    assert (out["rm_stemming_ejection_m"] == 0).all()
    assert any("computed" in w for w in out.attrs["warnings"])


def test_compare_errors():
    with pytest.raises(ValueError, match="missing the column 'burden_m'"):
        fl.compare(holes().drop(columns=["burden_m"]), k=13.5)
    with pytest.raises(ValueError, match="missing the column 'linear_charge_kg_m'"):
        fl.compare(holes().drop(columns=["linear_charge_kg_m"]), k=13.5)
    bad = holes()
    bad.loc[1, "burden_m"] = -1.0
    with pytest.raises(ValueError, match=r"'burden_m', row 1"):
        fl.compare(bad, k=13.5)
    bad = holes()
    bad.loc[2, "diameter_mm"] = 0.165
    with pytest.raises(ValueError, match=r"millimetres.*row 2"):
        fl.compare(bad, k=13.5)
    bad = holes()
    bad.loc[0, "stemming_m"] = float("nan")
    with pytest.raises(ValueError, match="stemming_m"):
        fl.compare(bad, k=13.5)
    with pytest.raises(ValueError, match="parameter 'k'"):
        fl.compare(holes(), k=-1)


def test_clearance_defaults():
    p = fl.clearance(95, target="people")
    assert (p.distance_m, p.governs, p.factor, p.regulation_minimum_m) == (380, "model", 4, None)
    assert fl.clearance(95, target="equipment").distance_m == 190  # E15
    with pytest.raises(TypeError):
        fl.clearance(95)


def test_clearance_regulation():
    r = fl.clearance(95, target="people", regulation="PE-DS024-2016-EM")
    assert (r.distance_m, r.governs, r.regulation_minimum_m) == (500, "regulation", 500)
    assert "DS 024-2016-EM, art. 267 a), modificado por DS 034-2023-EM" in r.citation
    assert any("360" in n for n in r.notes) and any("no safety margin" in n for n in r.notes)
    assert fl.clearance(200, target="people", regulation="PE-DS024-2016-EM").distance_m == 800
    assert fl.clearance(125, target="people", regulation="PE-DS024-2016-EM").governs == "model"  # tie
    e = fl.clearance(95, target="equipment", regulation="PE-DS024-2016-EM")
    assert e.distance_m == 190 and e.governs == "model" and e.regulation_minimum_m is None
    assert any("ignored" in n for n in e.notes)


def test_clearance_user_factor_and_errors():
    c = fl.clearance(100, target="people", factor=1.2)
    assert c.distance_m == pytest.approx(120) and "user factor" in c.notes
    for kw in ({"target": "x"}, {"target": "people", "factor": 0.9}, {"target": "people", "regulation": "x"}):
        with pytest.raises(ValueError, match="Domain error"):
            fl.clearance(95, **kw)
    for r in (0, -5, math.nan):
        with pytest.raises(ValueError, match="range_m"):
            fl.clearance(r, target="people")


def quiet(func, *args, **kwargs):
    """Calls func and fails if it emits any warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


def test_linear_charge_density_in_g_cc_is_an_error():
    for bad in (0.85, 1.15, 250.0, 2500.0):
        with pytest.raises(ValueError, match=r"'explosive_density_kg_m3'.*kg/m3 \(g/cc by mistake\?\)"):
            fl.linear_charge(165, bad)
    assert fl.linear_charge(165, 300) > 0 and fl.linear_charge(165, 2000) > 0
    df = holes().drop(columns=["linear_charge_kg_m"])
    df["explosive_density_kg_m3"] = [850.0, 0.85, 850.0]
    with pytest.raises(ValueError, match=r"g/cc by mistake.*'explosive_density_kg_m3', row 1"):
        fl.compare(df, k=13.5)


def test_compare_implied_density():
    with pytest.raises(ValueError, match=r"'linear_charge_kg_m'.*\[1\].*2000 kg/m3"):
        fl.compare(holes(linear_charge_kg_m=[35.6, 60.0, 11.0]), k=13.5)  # 2806 kg/m3 in 165 mm: impossible
    out = fl.compare(holes(linear_charge_kg_m=[35.6, 5.0, 11.0]), k=13.5)  # 234 kg/m3: decoupled, warning only
    w = [x for x in out.attrs["warnings"] if "300 kg/m3" in x]
    assert len(w) == 1 and "[1]" in w[0]
    assert not any("300 kg/m3" in x for x in fl.compare(holes(), k=13.5).attrs["warnings"])


def test_compare_nan_in_optional_columns():
    df = holes(sdob_m_kg13=[0.6, math.nan, 0.6], powder_factor_kg_m3=[1.0, 1.0, math.nan])
    df.loc[0, "charge_length_m"] = math.nan
    df.loc[1, "hole_angle_from_vertical_deg"] = math.nan
    out = fl.compare(df, k=13.5)
    assert math.isnan(out["mckenzie_2018_m"][1]) and out["mckenzie_2018_m"][0] > 0
    assert math.isnan(out["lundborg_q_m"][2]) and out["lundborg_q_m"][0] > 0
    assert out["rm_charge"][0] == pytest.approx(35.6)  # per-metre charge
    assert out["rm_stemming_ejection_m"][1] == 0
    vertical = [x for x in out.attrs["warnings"] if x.startswith("vertical hole")]
    assert len(vertical) == 1 and "[0, 1, 2]" in vertical[0]  # rows 0 and 2 are 0 degrees, row 1 is NaN
    assert np.isfinite(out["max_model_m"]).all()
    for column, value, row in (("sdob_m_kg13", math.inf, 1), ("charge_length_m", -1.0, 0),
                               ("powder_factor_kg_m3", 0.0, 2), ("hole_angle_from_vertical_deg", math.inf, 0)):
        bad = holes(sdob_m_kg13=0.6, powder_factor_kg_m3=1.0)
        bad.loc[row, column] = value
        with pytest.raises(ValueError, match=rf"'{column}', row {row}"):
            fl.compare(bad, k=13.5)


def test_stemming_ejection_vertical_warns():
    with pytest.warns(UserWarning, match="vertical"):
        assert fl.rm_stemming_ejection(27, 11, 3.0, 0) == 0.0
    assert quiet(fl.rm_stemming_ejection, 27, 11, 3.0, 20, g=9.8) == pytest.approx(62, abs=0.5)


def test_rm_k_outside_published_range_warns():
    calls = ((fl.rm_launch_velocity, (11, 2.6)), (fl.rm_face_burst, (11, 2.6)), (fl.rm_cratering, (11, 3.0)),
             (fl.rm_stemming_ejection, (11, 3.0, 20)), (fl.rm_min_confinement, (11, 100)))
    for func, args in calls:
        for k in (2.7, 40):
            with pytest.warns(UserWarning, match=r"\[13.5, 37\]") as rec:
                func(k, *args)
            assert rec[0].filename == __file__  # points at the caller, not at the wrapper
        for k in (13.5, 27, 37):
            quiet(func, k, *args)
    assert any("[13.5, 37]" in x for x in fl.compare(holes(), k=2.7).attrs["warnings"])
    assert not any("[13.5, 37]" in x for x in fl.compare(holes(), k=27).attrs["warnings"])


def test_roth_calibrated_near_minimum_warns():
    rock = fl.RothCalibration.Granite
    cm_min = fl.roth_minimum_charge_mass_ratio(rock)
    assert cm_min == pytest.approx(584 / 3.487e6)
    with pytest.warns(UserWarning, match="underestimate"):
        assert fl.roth_calibrated_velocity(rock, 1.01 * cm_min) == pytest.approx(2.42, rel=0.01)
    assert quiet(fl.roth_calibrated_velocity, rock, 5e-4) == pytest.approx(math.sqrt(3.487e6 * 5e-4 - 584))


def test_lundborg_powder_factor_underestimates():
    with pytest.warns(UserWarning, match="underestimate"):
        assert fl.lundborg_range(165, 0.25) == pytest.approx(46.4, rel=0.01)
    assert quiet(fl.lundborg_range, 101.6, 1.4) > fl.lundborg_max_range(101.6)
    out = fl.compare(holes(powder_factor_kg_m3=[0.25, 1.4, 1.4]), k=13.5)
    w = [x for x in out.attrs["warnings"] if "lundborg_q_m" in x]
    assert len(w) == 1 and "[0]" in w[0]
    assert any("Roth" in c and "underestimate" in c for c in out.attrs["critiques"])


def test_mckenzie_sdob_in_feet_warns():
    with pytest.warns(UserWarning, match=r"m/kg\^\(1/3\), not ft/lb\^\(1/3\)"):
        fl.mckenzie_max_range(2.52, 165)
    quiet(fl.mckenzie_max_range, 2.4, 165)
    out = fl.compare(holes(sdob_m_kg13=[0.6, 0.6, 2.52]), k=13.5)
    w = [x for x in out.attrs["warnings"] if "ft/lb^(1/3)" in x]
    assert len(w) == 1 and "[2]" in w[0]


def test_compare_max_model_and_disagreement():
    out = fl.compare(holes(sdob_m_kg13=[0.6, math.nan, 0.6], powder_factor_kg_m3=1.0), k=13.5)
    # R5: solo los modelos propios del pozo; lundborg_max_m depende solo del diámetro y queda aparte
    cols = ["rm_face_burst_m", "rm_cratering_m", "rm_stemming_ejection_m", "lundborg_q_m", "mckenzie_2018_m"]
    assert out["max_model_m"].tolist() == pytest.approx(out[cols].max(axis=1).tolist())
    over = list(out.index[out["max_model_m"] > 2 * out["rm_max_m"]])
    w = [x for x in out.attrs["warnings"] if "more than twice" in x]
    assert over and len(w) == 1 and str(over) in w[0]
    assert out["rm_max_m"].tolist() == np.maximum(out["rm_face_burst_m"], out["rm_cratering_m"]).tolist()


def test_compare_angle_not_given_and_up_front_checks():
    out = fl.compare(holes().drop(columns=["hole_angle_from_vertical_deg"]), k=13.5)
    assert any("angle not given: assumed vertical" in x for x in out.attrs["warnings"])
    bad = holes()
    bad.loc[0, "burden_m"] = -1.0
    for kw, name in (({"k": -1}, "k"), ({"k": True}, "k"), ({"k": 13.5, "g": 0}, "g")):
        with pytest.raises(ValueError, match=rf"parameter '{name}'") as e:
            fl.compare(bad, **kw)
        assert "column" not in str(e.value)


def test_clearance_one_model_note_and_regulation_ignored():
    for target in ("people", "equipment"):
        c = fl.clearance(95, target=target)
        assert "range_m comes from one model: see compare() for the others" in c.notes
    e = fl.clearance(95, target="equipment", regulation="PE-DS024-2016-EM")
    assert e.regulation is None
    for kw in ({"range_m": True}, {"range_m": 95, "factor": True}):
        with pytest.raises(ValueError, match="Domain error"):
            fl.clearance(target="people", **kw)


def test_compare_lundborg_envelope_is_apart_from_max_model():
    out = fl.compare(holes(), k=13.5)
    assert out["max_model_m"].tolist() == out["rm_max_m"].tolist()  # sin columnas opcionales: solo Richards y Moore
    assert (out["lundborg_max_m"] > out["max_model_m"]).all()
    over = list(out.index[out["lundborg_max_m"] > 2 * out["max_model_m"]])
    w = [x for x in out.attrs["warnings"] if x.startswith("Lundborg's diameter-only maximum")]
    assert over and len(w) == 1 and "905 m" in w[0] and "engineer's" in w[0] and str(over) in w[0]
    assert not any("more than twice rm_max_m" in x for x in out.attrs["warnings"])
    # el aviso del factor de carga solo cuando lundborg_q_m < 0.5 lundborg_max_m (743 m contra 905 m: no avisa)
    out = fl.compare(holes(powder_factor_kg_m3=[1.0, 1.0, 0.25]), k=13.5)
    w = [x for x in out.attrs["warnings"] if x.startswith("lundborg_q_m")]
    assert len(w) == 1 and "[2]" in w[0] and "[0" not in w[0]
