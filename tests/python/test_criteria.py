import importlib.resources
import json
import math

import pandas as pd
import pytest

from blastingpy._blastingpy_core import _criteria_curve as cc
from blastingpy.criteria import Profile, evaluate, list_profiles, load_profile, max_charge_unmonitored

NAN = float("nan")
COLUMNS = ["status", "limit_mm_s", "margin_mm_s", "margin_pct", "governing", "verification", "reason"]


def _ev(**kw):  # un evento canónico
    base = dict(distance_m=100.0, charge_kg=50.0, ppv_mm_s=10.0, ppv_kind="component_max", frequency_hz=10.0)
    return pd.DataFrame([{**base, **kw}])


def test_ri8507_curve():
    r = evaluate(_ev(ppv_mm_s=19.05, frequency_hz=10.0), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "complies" and r.limit_mm_s == pytest.approx(19.05) and r.margin_mm_s == pytest.approx(0.0)
    assert r.governing == "segment-1" and r.verification == "V1" and r.reason == ""
    r = evaluate(_ev(ppv_mm_s=30.0, frequency_hz=50.0), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.limit_mm_s == pytest.approx(50.8) and r.margin_pct == pytest.approx(40.945, abs=1e-3)
    r = evaluate(_ev(ppv_mm_s=13.0, frequency_hz=5.0), "USBM-RI8507-1980", "appendix-b-plaster").iloc[0]
    assert r.status == "exceeds" and r.limit_mm_s == pytest.approx(12.7) and r.margin_mm_s == pytest.approx(-0.3)


def test_osmre_distance_table():
    for d_ft, limit in [(200, 31.75), (300, 31.75), (1000, 25.4), (6000, 19.05)]:
        r = evaluate(_ev(distance_m=d_ft * 0.3048), "OSMRE-30CFR816.67", "distance-table").iloc[0]
        assert r.limit_mm_s == pytest.approx(limit)
    assert r.governing == "row-2"
    assert max_charge_unmonitored("OSMRE-30CFR816.67", 304.8) == pytest.approx(149.948, abs=1e-2)  # (1000/55)² lb


def test_not_comparable_and_indeterminate():
    r = evaluate(_ev(ppv_kind="vector_sum"), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "not_comparable" and "component" in r.reason
    assert math.isnan(r.limit_mm_s) and math.isnan(r.margin_pct) and r.governing == ""
    r = evaluate(_ev(frequency_hz=float("nan")), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "not_comparable" and "frequency" in r.reason
    r = evaluate(_ev(frequency_hz=500.0), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "indeterminate" and math.isnan(r.limit_mm_s) and "500" in r.reason


def test_appendix_a_published_case():
    # RI 8507, Apéndice A (p. 71): 0,008 in equivale a 0,5 in/s a 10 Hz y 0,25 in/s a 5 Hz (impresos redondeados).
    r = evaluate(_ev(frequency_hz=10.0), "USBM-RI8507-1980", "appendix-b-plaster").iloc[0]
    assert r.limit_mm_s == pytest.approx(12.7, abs=0.1) and r.governing == "segment-2"
    only = [cc.Segment(cc.SegmentKind.ConstantDisplacement, 1.0, 100.0, 0.008)]
    assert cc.limit_at(only, 5.0).limit == pytest.approx(0.25, rel=0.01)


def test_table_13_step_at_40_hz():
    p = "USBM-RI8507-1980"
    assert evaluate(_ev(frequency_hz=39.9), p, "table-13-drywall").iloc[0].limit_mm_s == pytest.approx(19.05)
    assert evaluate(_ev(frequency_hz=40.0), p, "table-13-drywall").iloc[0].limit_mm_s == pytest.approx(50.8)
    assert evaluate(_ev(frequency_hz=39.9), p, "table-13-plaster").iloc[0].limit_mm_s == pytest.approx(12.7)
    assert evaluate(_ev(frequency_hz=5000.0), p, "table-13-plaster").iloc[0].limit_mm_s == pytest.approx(50.8)


def test_components_governing_is_smallest_margin():
    ev = _ev(ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=15.0, freq_v_hz=50.0, ppv_l_mm_s=12.0, freq_l_hz=10.0)
    r = evaluate(ev, "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "complies" and r.governing == "longitudinal/segment-1"
    assert r.limit_mm_s == pytest.approx(19.05) and r.margin_mm_s == pytest.approx(7.05)
    # Una componente excede en otra frecuencia: manda ella.
    r = evaluate(_ev(ppv_t_mm_s=20.0, freq_t_hz=10.0, ppv_v_mm_s=40.0, freq_v_hz=50.0),
                 "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "exceeds" and r.governing == "transverse/segment-1"


def test_components_missing_frequency_or_out_of_domain():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    r = evaluate(_ev(ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=12.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "freq_v_hz" in r.reason and r.governing == ""
    r = evaluate(_ev(ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=12.0, freq_v_hz=300.0), p, c).iloc[0]
    assert r.status == "indeterminate" and r.governing == "transverse/segment-1"
    assert r.limit_mm_s == pytest.approx(19.05) and "300" in r.reason
    r = evaluate(_ev(ppv_t_mm_s=30.0, freq_t_hz=10.0, ppv_v_mm_s=12.0, freq_v_hz=300.0), p, c).iloc[0]
    assert r.status == "exceeds"
    # Tabla por distancia: no pide frecuencia; manda la componente mayor.
    r = evaluate(_ev(ppv_t_mm_s=10.0, ppv_v_mm_s=12.0, frequency_hz=NAN), "OSMRE-30CFR816.67", "distance-table").iloc[0]
    assert r.status == "complies" and r.governing == "vertical/row-1" and r.margin_mm_s == pytest.approx(25.4 - 12.0)


def test_recorded_peak_above_the_components_present_is_evaluated():
    # ppv_mm_s (component_max) es el pico del evento: si supera a las componentes presentes, también se evalúa.
    ev = _ev(ppv_mm_s=30.0, ppv_t_mm_s=10.0, freq_t_hz=10.0)
    r = evaluate(ev, "OSMRE-30CFR816.67", "distance-table").iloc[0]
    assert r.status == "exceeds" and r.governing == "row-1" and r.limit_mm_s == pytest.approx(25.4)
    r = evaluate(ev, "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "exceeds" and r.governing == "segment-1" and r.margin_mm_s == pytest.approx(19.05 - 30.0)
    r = evaluate(_ev(ppv_mm_s=30.0, ppv_t_mm_s=10.0, freq_t_hz=10.0, frequency_hz=NAN),
                 "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "not_comparable" and "frequency_hz" in r.reason
    # ppv_mm_s igual a la componente mayor y sin frequency_hz: no agrega nada.
    full = _ev(ppv_mm_s=15.0, frequency_hz=NAN, ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=15.0, freq_v_hz=50.0,
               ppv_l_mm_s=12.0, freq_l_hz=10.0)
    assert evaluate(full, "USBM-RI8507-1980", "appendix-b-drywall").iloc[0].status == "complies"
    # Componentes todas vacías: se usa ppv_mm_s.
    r = evaluate(_ev(ppv_t_mm_s=NAN, ppv_v_mm_s=NAN, ppv_l_mm_s=NAN, freq_t_hz=NAN),
                 "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "complies" and r.governing == "segment-1"


def test_complies_only_when_the_maximum_component_is_known():
    p, c, t = "USBM-RI8507-1980", "appendix-b-drywall", "OSMRE-30CFR816.67"
    # ppv_kind vertical: ppv_mm_s es la componente vertical (con frequency_hz si no hay freq_v_hz).
    ev = _ev(ppv_kind="vertical", ppv_mm_s=30.0, ppv_t_mm_s=10.0, freq_t_hz=10.0)
    r = evaluate(ev, p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "vertical/segment-1" and "longitudinal" in r.reason
    r = evaluate(ev, t, "distance-table").iloc[0]
    assert r.status == "exceeds" and r.governing == "vertical/row-1"
    # ppv_kind vector_sum nunca es candidato de component_max; faltan dos componentes.
    r = evaluate(_ev(ppv_kind="vector_sum", ppv_mm_s=40.0, ppv_t_mm_s=10.0, freq_t_hz=10.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "missing components: vertical, longitudinal" in r.reason
    # Una componente y la vertical por ppv_mm_s, ambas cumplen: falta la longitudinal.
    r = evaluate(_ev(ppv_kind="vertical", ppv_mm_s=12.0, ppv_t_mm_s=10.0, freq_t_hz=10.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "missing components: longitudinal" in r.reason
    # Una componente que excede, las demás faltan: excede y el reason dice cuáles faltan.
    r = evaluate(_ev(ppv_kind="vector_sum", ppv_t_mm_s=30.0, freq_t_hz=10.0), p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "transverse/segment-1"
    assert "missing components: vertical, longitudinal" in r.reason
    # Tres componentes con ppv_kind vector_sum: se evalúan las componentes.
    full = dict(ppv_kind="vector_sum", ppv_mm_s=40.0, ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=15.0,
                freq_v_hz=50.0, ppv_l_mm_s=12.0, freq_l_hz=10.0)
    r = evaluate(_ev(**full), p, c).iloc[0]
    assert r.status == "complies" and r.governing == "longitudinal/segment-1"
    r = evaluate(_ev(**{**full, "ppv_l_mm_s": 20.0}), p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "longitudinal/segment-1"


def test_recorded_value_with_its_own_frequency_is_evaluated():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    # Empate entre el pico y la componente presente, a otra frecuencia: el pico se evalúa con frequency_hz.
    r = evaluate(_ev(ppv_mm_s=20.0, frequency_hz=4.0, ppv_t_mm_s=20.0, freq_t_hz=50.0), p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "segment-1" and r.limit_mm_s == pytest.approx(19.05)
    r = evaluate(_ev(ppv_mm_s=20.0, frequency_hz=50.0, ppv_t_mm_s=20.0, freq_t_hz=50.0), p, c).iloc[0]
    assert r.status == "complies"
    # ppv_mm_s vertical junto a ppv_v_mm_s: también se evalúa.
    comps = dict(ppv_t_mm_s=5.0, freq_t_hz=10.0, ppv_v_mm_s=5.0, freq_v_hz=10.0, ppv_l_mm_s=5.0, freq_l_hz=10.0)
    r = evaluate(_ev(ppv_kind="vertical", ppv_mm_s=30.0, **comps), p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "vertical/segment-1"
    r = evaluate(_ev(ppv_kind="vertical", ppv_mm_s=0.0, **comps), p, c).iloc[0]
    assert r.status == "not_comparable" and "ppv_mm_s" in r.reason
    # Perfil vertical: ppv_mm_s vertical sin ppv_v_mm_s usa freq_v_hz.
    user = {"profile_id": "U", "quantity": "vertical", "units": {"velocity": "mm/s"},
            "criteria": [{"id": "f", "kind": "frequency_curve",
                          "segments": [{"kind": "constant_velocity", "f_min": 0.0, "f_max": 40.0, "value": 10.0},
                                       {"kind": "constant_velocity", "f_max": None, "value": 20.0}]}]}
    r = evaluate(_ev(ppv_kind="vertical", ppv_mm_s=15.0, freq_v_hz=50.0, frequency_hz=NAN), user, "f").iloc[0]
    assert r.status == "complies" and r.limit_mm_s == pytest.approx(20.0) and r.governing == "segment-1"


def test_vector_sum_and_vertical_profiles_evaluate_both_values():
    for quantity, column in [("vector_sum", "pvs_mm_s"), ("vertical", "ppv_v_mm_s")]:
        user = {"profile_id": "U", "quantity": quantity, "units": {"velocity": "mm/s"},
                "criteria": [{"id": "f", "kind": "flat", "limit": 20.0}]}
        r = evaluate(_ev(ppv_kind=quantity, ppv_mm_s=30.0, **{column: 10.0}), user, "f").iloc[0]
        assert r.status == "exceeds" and r.margin_mm_s == pytest.approx(-10.0)
        r = evaluate(_ev(ppv_kind=quantity, ppv_mm_s=15.0, **{column: 10.0}), user, "f").iloc[0]
        assert r.status == "complies" and r.margin_mm_s == pytest.approx(5.0)


def test_bad_ppv_values_are_not_comparable():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    r = evaluate(_ev(ppv_t_mm_s=-30.0, ppv_v_mm_s=5.0, freq_v_hz=10.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "ppv_t_mm_s" in r.reason and math.isnan(r.limit_mm_s)
    r = evaluate(_ev(ppv_mm_s=NAN), p, c).iloc[0]
    assert r.status == "not_comparable" and "ppv_mm_s" in r.reason
    r = evaluate(_ev(ppv_mm_s=0.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "ppv_mm_s" in r.reason


def test_exceedance_wins_over_a_component_without_frequency():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    r = evaluate(_ev(ppv_t_mm_s=30.0, freq_t_hz=10.0, ppv_v_mm_s=5.0), p, c).iloc[0]
    assert r.status == "exceeds" and r.governing == "transverse/segment-1" and "freq_v_hz" in r.reason
    r = evaluate(_ev(ppv_t_mm_s=10.0, freq_t_hz=10.0, ppv_v_mm_s=5.0), p, c).iloc[0]
    assert r.status == "not_comparable" and "freq_v_hz" in r.reason and r.governing == ""


def test_max_charge_at_row_boundaries():
    kg_per_lb = 0.45359237
    assert max_charge_unmonitored("OSMRE-30CFR816.67", 300 * 0.3048) == pytest.approx(36.0 * kg_per_lb)
    assert max_charge_unmonitored("OSMRE-30CFR816.67", 5000 * 0.3048) == pytest.approx((5000 / 55) ** 2 * kg_per_lb)
    assert (5000 / 55) ** 2 == pytest.approx(8264.46, abs=1e-2)
    user = {"profile_id": "U", "quantity": "component_max", "units": {"velocity": "mm/s", "distance": "m",
                                                                      "charge": "kg"},
            "criteria": [{"id": "t", "kind": "distance_table",
                          "rows": [{"d_max": 50.0, "limit": 20.0, "scaled_distance": 10.0},
                                   {"d_max": None, "limit": 10.0}]}]}
    with pytest.raises(ValueError, match="scaled_distance"):
        max_charge_unmonitored(user, 20.0, "t")


def test_distance_table_needs_distance_and_stops_at_last_row():
    with pytest.raises(ValueError, match="'distance_m'"):
        evaluate(_ev().drop(columns="distance_m"), "OSMRE-30CFR816.67", "distance-table")
    user = {"profile_id": "U", "quantity": "component_max", "units": {"velocity": "mm/s", "distance": "m"},
            "criteria": [{"id": "t", "kind": "distance_table", "rows": [{"d_max": 50.0, "limit": 20.0}]}]}
    r = evaluate(_ev(distance_m=60.0), user, "t").iloc[0]
    assert r.status == "indeterminate" and "60" in r.reason
    r = evaluate(_ev(distance_m=NAN), user, "t").iloc[0]
    assert r.status == "not_comparable" and "distance" in r.reason


def test_vector_sum_and_vertical_profiles_never_use_another_quantity():
    for quantity, kind, column in [("vector_sum", "vector_sum", "pvs_mm_s"), ("vertical", "vertical", "ppv_v_mm_s")]:
        user = {"profile_id": "U", "quantity": quantity, "units": {"velocity": "mm/s"},
                "criteria": [{"id": "f", "kind": "flat", "limit": 20.0}]}
        r = evaluate(_ev(ppv_kind="component_max"), user, "f").iloc[0]
        assert r.status == "not_comparable" and column in r.reason
        r = evaluate(_ev(ppv_kind=kind, ppv_mm_s=25.0), user, "f").iloc[0]
        assert r.status == "exceeds" and r.governing == "" and r.margin_mm_s == pytest.approx(-5.0)
        r = evaluate(_ev(ppv_kind="component_max", **{column: 15.0}), user, "f").iloc[0]
        assert r.status == "complies" and r.margin_pct == pytest.approx(25.0)


def test_pe_mem_1995_is_a_guide_with_osmre_values():
    p = load_profile("PE-MEM-1995")
    assert isinstance(p, Profile) and p.force == "guide" and p.frequency_methods == ()
    for d_ft, limit in [(300, 31.75), (5000, 25.4), (5001, 19.05)]:
        r = evaluate(_ev(distance_m=d_ft * 0.3048), p, "distance-table").iloc[0]
        assert r.limit_mm_s == pytest.approx(limit) and r.verification == "V1"


def test_user_profile_dict_and_path(tmp_path):
    user = {"profile_id": "MY-SITE", "name": {"en": "Site", "es": "Sitio"}, "quantity": "component_max",
            "units": {"velocity": "in/s", "displacement": "in"},
            "criteria": [{"id": "c", "kind": "frequency_curve", "verification": "V1", "reference": "x",
                          "segments": [{"kind": "constant_displacement", "f_min": 1.0, "f_max": "intersection",
                                        "value": 0.008},
                                       {"kind": "constant_velocity", "f_max": None, "value": 1.0}]}]}
    out = evaluate(_ev(ppv_mm_s=25.4, frequency_hz=100.0), user, "c")
    assert list(out.columns) == COLUMNS and out.iloc[0].status == "complies" and out.iloc[0].verification == "user"
    path = tmp_path / "site.json"
    path.write_text(json.dumps(user), encoding="utf-8")
    assert load_profile(path).criteria["c"].verification == "user"
    assert load_profile(str(path)).profile_id == "MY-SITE"
    for key in ("quantity", "units", "criteria"):
        broken = {k: v for k, v in user.items() if k != key}
        with pytest.raises(ValueError, match=f"'{key}'"):
            load_profile(broken)
    broken = json.loads(json.dumps(user))
    del broken["criteria"][0]["segments"]
    with pytest.raises(ValueError, match="'segments'"):
        load_profile(broken)


def test_result_frame_keeps_index_and_columns():
    ev = pd.concat([_ev(), _ev(ppv_mm_s=30.0)])
    ev.index = ["a", "b"]
    out = evaluate(ev, "USBM-RI8507-1980", "table-13-drywall")
    assert list(out.columns) == COLUMNS and list(out.index) == ["a", "b"]
    assert list(out.status) == ["complies", "exceeds"]
    with pytest.raises(ValueError, match="'ppv_kind'"):
        evaluate(ev.drop(columns="ppv_kind"), "USBM-RI8507-1980", "table-13-drywall")


def test_list_profiles_and_unknown_names():
    assert list_profiles() == ["OSMRE-30CFR816.67", "PE-MEM-1995", "USBM-RI8507-1980"]
    with pytest.raises(ValueError, match="USBM-RI8507-1980"):
        load_profile("NOPE")
    with pytest.raises(ValueError, match="appendix-b-drywall"):
        evaluate(_ev(), "USBM-RI8507-1980", "nope")
    with pytest.raises(ValueError, match="scaled_distance"):
        max_charge_unmonitored("USBM-RI8507-1980", 100.0, "table-13-drywall")


def test_shipped_profiles_schema():
    files = [f for f in importlib.resources.files("blastingpy").joinpath("profiles").iterdir()
             if f.name.endswith(".json")]
    assert len(files) == 3
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["source"]["citation"] and data["quantity"] and data["units"]
        assert set(data["name"]) == {"en", "es"}
        for c in data["criteria"]:
            assert c["verification"] in {"V1", "V2"} and c["reference"]
        assert load_profile(data["profile_id"]).profile_id == data["profile_id"]
