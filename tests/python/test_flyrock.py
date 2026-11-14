import math

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
