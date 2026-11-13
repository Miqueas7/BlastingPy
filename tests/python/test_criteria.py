import importlib.resources
import json
import math

import pandas as pd
import pytest

from blastingpy._blastingpy_core import _criteria_curve as cc
from blastingpy.criteria import Profile, evaluate, list_profiles, load_profile, max_charge_unmonitored

NAN = float("nan")
COLUMNS = ["status", "limit_mm_s", "margin_mm_s", "margin_pct", "governing", "verification", "reason"]
PEAK_ONLY = "peak-only check: components at other frequencies are not verified"


def _ev(**kw):  # un evento canónico
    base = dict(distance_m=100.0, charge_kg=50.0, ppv_mm_s=10.0, ppv_kind="component_max", frequency_hz=10.0)
    return pd.DataFrame([{**base, **kw}])


def test_ri8507_curve():
    r = evaluate(_ev(ppv_mm_s=19.05, frequency_hz=10.0), "USBM-RI8507-1980", "appendix-b-drywall").iloc[0]
    assert r.status == "complies" and r.limit_mm_s == pytest.approx(19.05) and r.margin_mm_s == pytest.approx(0.0)
    assert r.governing == "segment-1" and r.verification == "V1" and r.reason == PEAK_ONLY
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


def test_vertical_value_is_checked_at_every_frequency_of_its_group():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    tl = dict(ppv_t_mm_s=5.0, freq_t_hz=10.0, ppv_l_mm_s=5.0, freq_l_hz=10.0)
    ev = dict(ppv_kind="vertical", ppv_mm_s=20.0, frequency_hz=4.0, **tl)
    assert evaluate(_ev(**ev), p, c).iloc[0].status == "exceeds"
    r = evaluate(_ev(**ev, freq_v_hz=50.0), p, c).iloc[0]  # agregar freq_v_hz no puede volverlo «cumple»
    assert r.status == "exceeds" and r.governing == "vertical/segment-1"
    ev = dict(ppv_kind="vertical", ppv_mm_s=20.0, freq_v_hz=4.0, frequency_hz=50.0, **tl)
    assert evaluate(_ev(**ev), p, c).iloc[0].status == "exceeds"
    assert evaluate(_ev(**ev, ppv_v_mm_s=18.0), p, c).iloc[0].status == "exceeds"


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


def _flat(limit):
    return {"profile_id": "U", "quantity": "component_max", "units": {"velocity": "mm/s"},
            "criteria": [{"id": "f", "kind": "flat", "limit": limit}]}


def _table(**row):
    rows = [{"d_max": 50.0, "limit": 20.0, "scaled_distance": 10.0}, {"d_max": None, "limit": 10.0}]
    rows[0].update(row)
    return {"profile_id": "U", "quantity": "component_max",
            "units": {"velocity": "mm/s", "distance": "m", "charge": "kg"},
            "criteria": [{"id": "t", "kind": "distance_table", "rows": rows}]}


def _curve(**segment):
    segments = [{"kind": "linear", "f_min": 1.0, "f_max": 10.0, "value": 5.0, "value_max": 10.0},
                {"kind": "constant_velocity", "f_max": None, "value": 10.0}]
    segments[0].update(segment)
    return {"profile_id": "U", "quantity": "component_max", "units": {"velocity": "mm/s"},
            "criteria": [{"id": "c", "kind": "frequency_curve", "segments": segments}]}


BAD_NUMBERS = [NAN, float("inf"), -float("inf"), 0.0, -5.0, "12", None, True, [1.0]]
FIELDS = [("limit", lambda v: _flat(v)), ("limit", lambda v: _table(limit=v)),
          ("scaled_distance", lambda v: _table(scaled_distance=v)), ("d_max", lambda v: _table(d_max=v)),
          ("value", lambda v: _curve(value=v)), ("value_max", lambda v: _curve(value_max=v)),
          ("f_max", lambda v: _curve(f_max=v))]
NULLABLE = ("scaled_distance", "d_max", "f_max")  # null: sin distancia escalada, última fila o tramo abierto


@pytest.mark.parametrize("field, build, bad", [(f, b, v) for f, b in FIELDS for v in BAD_NUMBERS
                                               if not (v is None and f in NULLABLE)])
def test_profile_numbers_must_be_finite_and_positive(field, build, bad):
    # Un límite NaN o infinito daba «cumple» a cualquier evento; un 0 daba ZeroDivisionError.
    profile = build(bad)
    with pytest.raises(ValueError, match=rf"parameter 'profile' \(profile 'U', criterion '[ftc]'\).*'{field}' must be"):
        load_profile(profile)


def test_profile_numbers_may_be_numpy_scalars():
    import numpy as np
    assert load_profile(_flat(np.float64(20.0))).criteria["f"].limit_mm_s == 20.0
    assert load_profile(_table(limit=np.int64(20))).criteria["t"].rows[0].limit_mm_s == 20.0


@pytest.mark.parametrize("bad", [NAN, float("inf"), -1.0, "0", None])
def test_profile_f_min_must_be_finite_and_not_negative(bad):
    with pytest.raises(ValueError, match=r"\(profile 'U', criterion 'c'\) segment 0: 'f_min' must be .*>= 0"):
        load_profile(_curve(f_min=bad))
    later = _curve()
    later["criteria"][0]["segments"][1]["f_min"] = bad
    with pytest.raises(ValueError, match=r"segment 1: 'f_min' must be"):
        load_profile(later)
    assert load_profile(_curve(f_min=0.0, kind="constant_velocity")).criteria["c"].segments[0].f_min == 0.0


def test_profile_d_max_must_increase():
    profile = _table()
    profile["criteria"][0]["rows"] = [{"d_max": 50.0, "limit": 20.0}, {"d_max": 50.0, "limit": 10.0}]
    with pytest.raises(ValueError, match=r"row 1: 'd_max' must be greater than the previous row's"):
        load_profile(profile)


def test_profile_json_rejects_nan_and_infinity_tokens(tmp_path):
    for token in ("NaN", "Infinity", "-Infinity"):
        path = tmp_path / f"{token}.json"
        path.write_text(json.dumps(_flat(1.0)).replace("1.0", token), encoding="utf-8")
        with pytest.raises(ValueError, match=rf"parameter 'profile'.*{token}"):
            load_profile(path)


def test_a_non_finite_limit_never_complies():
    # Defensa en profundidad: un Criterion armado a mano (sin pasar por el cargador) con límite NaN no da «cumple».
    from blastingpy.criteria import Criterion
    good = load_profile(_flat(20.0))
    for limit in (NAN, float("inf"), 0.0):
        c = Criterion(criterion_id="f", kind="flat", structure={}, verification="user", reference="", note="",
                      limit_mm_s=limit)
        p = Profile(**{**good.__dict__, "criteria": {"f": c}})
        r = evaluate(_ev(ppv_mm_s=500.0), p, "f").iloc[0]
        assert r.status == "not_comparable" and "limit" in r.reason and math.isnan(r.limit_mm_s)


def test_peak_only_compliance_against_a_curve_says_so():
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    comps = dict(ppv_t_mm_s=5.0, freq_t_hz=10.0, ppv_v_mm_s=5.0, freq_v_hz=10.0, ppv_l_mm_s=5.0, freq_l_hz=10.0)
    # El pico con su frecuencia, sin las tres componentes: cumple, pero lo dice en reason.
    r = evaluate(_ev(ppv_mm_s=5.0, ppv_t_mm_s=4.0, freq_t_hz=50.0), p, c).iloc[0]
    assert r.status == "complies" and r.reason == PEAK_ONLY
    # Con las tres componentes, o contra una tabla por distancia, el reason queda vacío.
    assert evaluate(_ev(ppv_mm_s=5.0, **comps), p, c).iloc[0].reason == ""
    assert evaluate(_ev(ppv_kind="vector_sum", **comps), p, c).iloc[0].reason == ""
    r = evaluate(_ev(ppv_mm_s=5.0), "OSMRE-30CFR816.67", "distance-table").iloc[0]
    assert r.status == "complies" and r.reason == ""
    # Solo en «cumple»: un exceso del pico no lleva la nota.
    r = evaluate(_ev(ppv_mm_s=30.0), p, c).iloc[0]
    assert r.status == "exceeds" and r.reason == ""


def test_verdict_frame_describes_itself():
    out = evaluate(_ev(), "USBM-RI8507-1980", "appendix-b-drywall")
    assert out.attrs == {
        "profile_id": "USBM-RI8507-1980", "criterion": "appendix-b-drywall",
        "citation": load_profile("USBM-RI8507-1980").source["citation"], "edition": "1980",
        "quantity": "component_max", "frequency_methods": ("zero_crossing", "fourier"), "verification": "V1",
        "reference": "Appendix B, Fig. B-1 and text, pp. 73-74 (PDF pp. 82-83)"}
    user = evaluate(_ev(), _flat(20.0), "f").attrs
    assert user["verification"] == "user" and user["citation"] == "" and user["edition"] == ""


def test_candidates_are_the_ones_the_verdict_used():
    from blastingpy.criteria import Candidate, candidates
    p, c = "USBM-RI8507-1980", "appendix-b-drywall"
    ev = pd.concat([_ev(ppv_mm_s=30.0, ppv_t_mm_s=10.0, freq_t_hz=4.0),  # pico y transversal
                    _ev(ppv_kind="vector_sum"),  # ningún candidato
                    _ev(ppv_kind="vertical", ppv_mm_s=12.0, frequency_hz=500.0)], ignore_index=True)
    got = candidates(ev, p, c)
    assert len(got) == 3 and got[1] == []
    t, peak = got[0]
    assert isinstance(t, Candidate) and t.label == "transverse" and t.frequency_hz == 4.0
    assert t.limit_mm_s == pytest.approx(19.05) and t.ppv_mm_s == 10.0 and not t.exceeds
    assert peak.label == "" and peak.ppv_mm_s == 30.0 and peak.exceeds
    (v,) = got[2]
    assert v.label == "vertical" and math.isnan(v.limit_mm_s) and not v.exceeds  # 500 Hz: fuera del dominio
    status = evaluate(ev, p, c).status.tolist()
    assert status == ["exceeds", "not_comparable", "not_comparable"]
    with pytest.raises(ValueError, match="appendix-b-drywall"):
        candidates(ev, p, "nope")


def test_result_frame_keeps_index_and_columns():
    ev = pd.concat([_ev(), _ev(ppv_mm_s=30.0)])
    ev.index = ["a", "b"]
    out = evaluate(ev, "USBM-RI8507-1980", "table-13-drywall")
    assert list(out.columns) == COLUMNS and list(out.index) == ["a", "b"]
    assert list(out.status) == ["complies", "exceeds"]
    with pytest.raises(ValueError, match="'ppv_kind'"):
        evaluate(ev.drop(columns="ppv_kind"), "USBM-RI8507-1980", "table-13-drywall")


def test_list_profiles_and_unknown_names():
    assert list_profiles() == ["BS-7385-2-1993", "DIN-4150-3-2016", "OSMRE-30CFR816.67", "PE-MEM-1995",
                               "USBM-RI8507-1980"]
    with pytest.raises(ValueError, match="USBM-RI8507-1980"):
        load_profile("NOPE")
    with pytest.raises(ValueError, match="appendix-b-drywall"):
        evaluate(_ev(), "USBM-RI8507-1980", "nope")
    with pytest.raises(ValueError, match="scaled_distance"):
        max_charge_unmonitored("USBM-RI8507-1980", 100.0, "table-13-drywall")


def test_shipped_profiles_schema():
    files = [f for f in importlib.resources.files("blastingpy").joinpath("profiles").iterdir()
             if f.name.endswith(".json")]
    assert len(files) == 5
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert data["source"]["citation"] and data["quantity"] and data["units"]
        assert set(data["name"]) == {"en", "es"}
        for c in data["criteria"]:
            assert c["verification"] in {"V1", "V2"} and c["reference"]
            assert set(c["note"]) == {"en", "es"} and all(c["note"].values()), (data["profile_id"], c["id"])
        assert load_profile(data["profile_id"]).profile_id == data["profile_id"]
    bs = load_profile("BS-7385-2-1993")
    assert set(bs.source["note"]) == {"en", "es"} and "final draft" in bs.source["note"]["en"]
    assert "borrador final" in bs.criteria["line-1"].note["es"]


def test_criterion_note_is_text_or_bilingual():
    for note in ("Clause 4", {"en": "Clause 4", "es": "Cláusula 4"}):
        profile = _flat(10.0)
        profile["criteria"][0]["note"] = note
        assert load_profile(profile).criteria["f"].note == note
    profile["criteria"][0]["note"] = 4
    with pytest.raises(ValueError, match=r"\(profile 'U', criterion 'f'\) 'note' must be"):
        load_profile(profile)


def _limit(profile, criterion, f):
    r = evaluate(_ev(ppv_mm_s=1.0, frequency_hz=f), profile, criterion).iloc[0]
    return r.status, r.limit_mm_s, r.verification


def _check(profile, criterion, cases):
    for f, expected, tol in cases:
        status, limit, verification = _limit(profile, criterion, f)
        assert verification == "V2", (criterion, f)
        if expected is None:
            assert status == "indeterminate" and math.isnan(limit), (criterion, f)
        else:
            assert limit == pytest.approx(expected, abs=tol), (criterion, f, limit)


def test_bs_7385_2_profile_metadata():
    p = load_profile("BS-7385-2-1993")
    assert p.quantity == "component_max" and p.frequency_methods == ("zero_crossing", "fourier")
    assert list(p.criteria) == ["line-1", "line-2"]
    assert all(c.verification == "V2" and "Malam" in c.reference for c in p.criteria.values())


def test_bs_7385_2_line_1():
    # Malam (1993), Tabla 4 y Fig. 9, p. 145: 50 mm/s de 4 a 250 Hz; bajo 4 Hz la fuente no da valor.
    _check("BS-7385-2-1993", "line-1",
           [(3.0, None, 0), (4.0, 50.0, 1e-9), (100.0, 50.0, 1e-9), (250.0, 50.0, 1e-9), (300.0, None, 0)])


def test_bs_7385_2_line_2():
    # Malam (1993), Tabla 4 y Fig. 9, p. 145. Uniones log-log según Miller et al. (2024), p. 281.
    _check("BS-7385-2-1993", "line-2", [
        (0.5, None, 0),
        (1.0, 2 * math.pi * 1.0 * 0.6, 1e-9),  # desplazamiento constante de 0,6 mm, 1 a 4 Hz
        (2.0, 7.540, 1e-3),
        (4.0, 15.0, 1e-9),  # inicio del tramo log-log: 15 mm/s a 4 Hz
        (8.0, 17.443, 1e-3),
        (15.0, 20.0, 1e-9),
        (25.0, 32.23, 1e-2),
        (40.0, 50.0, 1e-9),
        (100.0, 50.0, 1e-9),
        (250.0, 50.0, 1e-9),
        (300.0, None, 0)])


def test_din_4150_3_profile_metadata():
    p = load_profile("DIN-4150-3-2016")
    assert p.quantity == "component_max" and p.frequency_methods == ()
    assert list(p.criteria) == ["line-2", "line-3"]
    assert all(c.verification == "V2" and "Fig. 1a" in c.reference and "foundation" in c.structure["en"]
               for c in p.criteria.values())


def test_din_4150_3_line_2_foundation():
    # Miller, Duschlbauer y Spagnol (2024), Fig. 1a, p. 282; rectas en ejes lineales (p. 281).
    _check("DIN-4150-3-2016", "line-2", [
        (0.5, None, 0), (1.0, 5.0, 1e-9), (5.0, 5.0, 1e-9), (10.0, 5.0, 1e-9),
        (30.0, 10.0, 1e-9), (50.0, 15.0, 1e-9), (75.0, 17.5, 1e-9), (100.0, 20.0, 1e-9), (120.0, None, 0)])


def test_din_4150_3_line_3_foundation():
    _check("DIN-4150-3-2016", "line-3", [
        (0.5, None, 0), (1.0, 3.0, 1e-9), (5.0, 3.0, 1e-9), (10.0, 3.0, 1e-9),
        (30.0, 5.5, 1e-9), (50.0, 8.0, 1e-9), (75.0, 9.0, 1e-9), (100.0, 10.0, 1e-9), (120.0, None, 0)])
