import importlib.resources
import json
import math

import numpy as np
import pandas as pd
import pytest

from blastingpy import _blastingpy_core as core
from blastingpy import airblast
from blastingpy.airblast import Level

FT, LB, PSI = core._units.m_per_ft, core._units.kg_per_lb, core._units.pa_per_psi
DOMAIN = r"Domain error \[BlastingPy\]: parameter"


def _psi_at_sd100(equation_id):
    return airblast.predict_published(equation_id, 100 * FT, LB).value / PSI


# Tabla B-1 de RI 8485 a SD = 100 ft/lb^⅓ (FUENTES_SPRINT3.md, «Ejemplos para pruebas»): 5 cifras → rel 1e-4.
@pytest.mark.parametrize("equation_id, psi", [
    ("USBM-RI8485/coal-highwall/0.1Hz", 0.0041833),
    ("USBM-RI8485/coal-highwall/2Hz", 0.0032988),
    ("USBM-RI8485/coal-parting/0.1Hz", 0.095915),
    ("USBM-RI8485/coal-parting/2Hz", 0.055142),
    ("USBM-RI8485/metal-mine-highwall/0.1Hz", 0.015036),
    ("USBM-RI8485/quarry/0.1Hz", 0.0093097),
])
def test_table_b1_at_sd_100(equation_id, psi):
    assert _psi_at_sd100(equation_id) == pytest.approx(psi, rel=1e-4)


@pytest.mark.parametrize("equation_id", airblast.list_equations())
def test_every_equation_is_k_sd_minus_beta(equation_id):
    """K·SD^−β exacto con la K y la β del propio JSON, en sus unidades: una conversión mal hecha no se esconde."""
    eq = airblast.published(equation_id)
    if eq.units_original == "psi; ft/lb^(1/3)":
        distance, charge, factor = 100 * FT, LB, PSI
    else:
        assert eq.units_original == "kPa; m/kg^(1/3)"
        distance, charge, factor = 100.0, 1.0, 1000.0
    expected = eq.k_original * 100.0 ** -eq.beta * factor
    assert airblast.predict_published(equation_id, distance, charge).value == pytest.approx(expected, rel=1e-9)
    assert eq.law.k == airblast.predict_published(equation_id, 1.0, 1.0).value  # SD = 1: K en Pa-m-kg^⅓
    assert eq.law.sigma == 0 and eq.law.beta == eq.beta


# BLEP v2.0, hoja de análisis (PDF p. 39): dB redondeado al entero → ±0,5 dB.
@pytest.mark.parametrize("equation_id, sd, db", [
    ("OSMRE-BLEP/highwall", 214, 118), ("OSMRE-BLEP/highwall", 188, 119),
    ("OSMRE-BLEP/highwall", 171, 120), ("OSMRE-BLEP/highwall", 162, 120),
    ("OSMRE-BLEP/parting", 214, 140), ("OSMRE-BLEP/parting", 188, 142),
    ("OSMRE-BLEP/parting", 171, 143), ("OSMRE-BLEP/parting", 162, 144),
])
def test_blep_published_example(equation_id, sd, db):
    pa = airblast.predict_published(equation_id, sd * FT, LB).value
    assert airblast.to_db(pa) == pytest.approx(db, abs=0.5)


def test_as2187_free_face():
    assert airblast.predict_published("AS2187.2-2006/free-face", 100.0, 1.0).value == pytest.approx(649.6, rel=1e-3)
    eq = airblast.published("AS2187.2-2006/free-face")
    assert (eq.k_original, eq.beta, eq.verification) == (516, 1.45, "V2")


def test_catalog():
    ids = airblast.list_equations()
    assert len(ids) == 17 and len(set(ids)) == 17
    assert sum(i.startswith("USBM-RI8485/") for i in ids) == 14
    for equation_id in ids:
        eq = airblast.published(equation_id)
        assert eq.id == equation_id
        assert eq.citation and eq.verification in ("V1", "V2")
        assert eq.system is not None or any("system" in note for note in eq.notes)
    c_slow = airblast.published("USBM-RI8485/coal-highwall/c-slow")
    assert c_slow.se_db is None and any("erratum" in note for note in c_slow.notes)
    row1 = airblast.published("USBM-RI8485/coal-highwall/0.1Hz")
    assert (row1.k_original, row1.beta, row1.r, row1.se_db, row1.n, row1.system) == (0.162, 0.794, 0.739, 5.5, 115,
                                                                                   "0.1Hz")
    lo, hi = row1.scaled_distance_range_si
    k = FT / LB ** (1 / 3)
    assert (lo, hi) == pytest.approx((10 * k, 900 * k), rel=1e-12)
    assert airblast.published("USBM-RI8485/quarry-front-of-face/0.1Hz").scaled_distance_range_si is None
    blep = airblast.published("OSMRE-BLEP/parting")
    assert (blep.k_original, blep.beta, blep.system) == (169, 1.62, None)


def test_unknown_equation():
    with pytest.raises(ValueError, match=DOMAIN + " 'equation_id'"):
        airblast.published("USBM-RI8485/coal-highwall/6Hz")


def test_json_is_installed_with_the_package():
    text = importlib.resources.files("blastingpy").joinpath("airblast_data/equations.json").read_text("utf-8")
    assert len(json.loads(text)["equations"]) == 17


# Review Focus 4: avisos de predict_published.
def test_predict_published_extrapolation():
    p = airblast.predict_published("USBM-RI8485/coal-parting/0.1Hz", 100 * FT, LB)  # SD 100, rango ~400-1700
    assert any("extrapolation" in w for w in p.warnings)
    inside = airblast.predict_published("USBM-RI8485/coal-highwall/0.1Hz", 100 * FT, LB)
    assert not any("extrapolation" in w for w in inside.warnings)


def test_predict_published_range_not_published():
    p = airblast.predict_published("USBM-RI8485/quarry-front-of-face/0.1Hz", 100 * FT, LB)
    assert any("range not published" in w for w in p.warnings)


def test_predict_published_carries_notes():
    p = airblast.predict_published("USBM-RI8485/quarry-opposite-initiation/0.1Hz", 100 * FT, LB)
    assert any("r = 0.050: no predictive value" in w for w in p.warnings)
    assert not any("upper bound equals the median" in w for w in p.warnings)  # mediana: sin el aviso de cota superior


def test_to_db_and_from_db():
    assert airblast.to_db(100.0) == pytest.approx(133.98, abs=0.005)
    out = airblast.to_db(np.array([100.0, np.nan]))
    assert out[0] == pytest.approx(133.98, abs=0.005) and math.isnan(out[1])
    assert math.isnan(airblast.to_db(math.nan))
    back = airblast.from_db(airblast.to_db(np.array([20e-6, 6900.0, np.nan])))
    assert back[:2] == pytest.approx([20e-6, 6900.0], rel=1e-12) and math.isnan(back[2])
    assert isinstance(airblast.from_db(0.0), float)
    for bad in (0.0, -1.0, np.array([100.0, 0.0])):
        with pytest.raises(ValueError, match=DOMAIN):
            airblast.to_db(bad)


def _events(k=5000.0, beta=1.2):
    d = np.array([100.0, 200.0, 400.0, 800.0, 1600.0])
    w = np.array([50.0, 120.0, 80.0, 300.0, 25.0])
    return pd.DataFrame({"distance_m": d, "charge_kg": w, "airblast_pa": k * (d / np.cbrt(w)) ** -beta})


def test_fit_recovers_the_law():
    law = airblast.fit(_events())
    assert law.law == airblast.Law.CubeRoot
    assert law.k == pytest.approx(5000.0, rel=1e-9) and law.beta == pytest.approx(1.2, rel=1e-9)


def test_fit_skips_rows_without_airblast():
    events = _events()
    events.loc[len(events)] = [300.0, 40.0, np.nan]
    events.loc[len(events)] = [500.0, 40.0, np.inf]
    with pytest.warns(UserWarning, match="2 events"):
        law = airblast.fit(events)
    assert law.n == 5 and law.k == pytest.approx(5000.0, rel=1e-9)


def test_fit_needs_three_rows():
    with pytest.raises(ValueError, match=DOMAIN + " 'airblast_pa'.*at least 3"):
        airblast.fit(_events().iloc[:2])


def test_fit_rejects_non_positive_pressure_with_airblast_wording():
    events = _events()
    events.loc[0, "airblast_pa"] = 0.0
    with pytest.raises(ValueError, match=DOMAIN + " 'airblast_pa'"):
        airblast.fit(events)


def test_fit_missing_column():
    with pytest.raises(ValueError, match=DOMAIN + " 'events'.*'airblast_pa'"):
        airblast.fit(_events().drop(columns="airblast_pa"))


def test_inverses():
    law = airblast.fit(_events())
    for level in (Level.Median, Level.Upper):
        p = airblast.predict(law, 350.0, 90.0, level)
        assert airblast.max_charge(law, 350.0, p.value, level).value == pytest.approx(90.0, rel=1e-9)
        assert airblast.distance_for(law, 90.0, p.value, level).value == pytest.approx(350.0, rel=1e-9)


def test_limit_errors_use_airblast_wording():
    law = airblast.fit(_events())
    with pytest.raises(ValueError, match=DOMAIN + " 'limit_pa'"):
        airblast.max_charge(law, 350.0, 0.0)
    with pytest.raises(ValueError, match=DOMAIN + " 'limit_pa'"):
        airblast.distance_for(law, 90.0, -1.0)


# (K, β) de las 17 ecuaciones, literales de FUENTES_SPRINT3.md («Ecuaciones publicadas»): un cambio del JSON se nota.
CATALOG = {
    "USBM-RI8485/coal-highwall/0.1Hz": (0.162, 0.794), "USBM-RI8485/coal-highwall/2Hz": (0.146, 0.823),
    "USBM-RI8485/coal-highwall/5Hz": (0.087, 0.725), "USBM-RI8485/coal-highwall/c-slow": (0.015, 0.885),
    "USBM-RI8485/coal-parting/0.1Hz": (169, 1.623), "USBM-RI8485/coal-parting/2Hz": (49.6, 1.477),
    "USBM-RI8485/coal-parting/5Hz": (194, 1.666), "USBM-RI8485/coal-parting/c-slow": (41.6, 1.785),
    "USBM-RI8485/metal-mine-highwall/0.1Hz": (0.401, 0.713), "USBM-RI8485/quarry/0.1Hz": (0.246, 0.711),
    "USBM-RI8485/quarry-direction-of-initiation/0.1Hz": (0.979, 1.120),
    "USBM-RI8485/quarry-behind-face/0.1Hz": (0.056, 0.515),
    "USBM-RI8485/quarry-opposite-initiation/0.1Hz": (0.028, 0.098),
    "USBM-RI8485/quarry-front-of-face/0.1Hz": (1.317, 0.966),
    "OSMRE-BLEP/highwall": (0.162, 0.79), "OSMRE-BLEP/parting": (169, 1.62),
    "AS2187.2-2006/free-face": (516, 1.45),
}


def test_catalog_values_are_the_sources():
    assert airblast.list_equations() == list(CATALOG)
    for equation_id, (k, beta) in CATALOG.items():
        eq = airblast.published(equation_id)
        assert (eq.k_original, eq.beta) == (k, beta) and isinstance(eq.k_original, float)


def test_range_edges():
    # Fila 1: rango ~10 a ~900 ft/lb^⅓; el borde cuenta como dentro.
    eq = "USBM-RI8485/coal-highwall/0.1Hz"
    for edge in (10, 900):
        assert not any("extrapolation" in w for w in airblast.predict_published(eq, edge * FT, LB).warnings)
    assert any("extrapolation" in w for w in airblast.predict_published(eq, 9.9 * FT, LB).warnings)


def test_fit_info_cites_ri8485():
    info = airblast.fit(_events()).info
    assert info.method_id == "AIR-CBRT-USBM8485"
    assert info.verification == core.Verification.Primary
    assert "RI 8485" in info.source and "Table B-1" in info.source


def _no_vibration_words(texts):
    for text in texts:
        assert "PPV" not in text and "ppv" not in text and "BIVDEP" not in text and "30" not in text, text


def test_fit_warnings_speak_of_airblast():
    noisy = _events()
    noisy["airblast_pa"] *= [1.1, 0.9, 1.05, 0.95, 1.0]
    law = airblast.fit(noisy)
    assert law.n < 30 and law.sigma > 0
    _no_vibration_words(law.warnings)
    rising = _events(beta=1.2)
    rising["airblast_pa"] = [10.0, 20.0, 40.0, 80.0, 160.0]
    rising["charge_kg"] = 100.0
    law = airblast.fit(rising)
    assert any("overpressure does not decrease with distance" in w for w in law.warnings)
    _no_vibration_words(law.warnings)
    for bad in (_events().iloc[:2], _events().assign(airblast_pa=[1.0, 0.0, 1.0, 1.0, 1.0])):
        with pytest.raises(ValueError) as e:
            airblast.fit(bad)
        _no_vibration_words([str(e.value)])


@pytest.mark.parametrize("equation_id, verification", [
    ("USBM-RI8485/quarry/0.1Hz", "Primary"), ("OSMRE-BLEP/parting", "Primary"),
    ("AS2187.2-2006/free-face", "Secondary"),
])
def test_published_law_info(equation_id, verification):
    eq = airblast.published(equation_id)
    assert eq.law.info.method_id == equation_id
    assert eq.law.info.verification == getattr(core.Verification, verification)
    assert eq.law.info.source == eq.citation


def test_vibration_fit_unchanged():
    from blastingpy import attenuation

    d = np.array([100.0, 200.0, 400.0, 800.0])
    v = 1000.0 * (d / 10.0) ** -1.6 * np.array([1.1, 0.9, 1.05, 0.95])
    law = attenuation.fit(pd.DataFrame({"distance_m": d, "charge_kg": 100.0, "ppv_mm_s": v}), law="cbrt")
    assert (law.info.method_id, law.info.verification) == ("ATT-CBRT-USBM656", core.Verification.Secondary)
    assert law.warnings == ["fewer than 30 events: OSMRE BIVDEP (2019) asks for a minimum of 30"]
    made = attenuation.make_site_law(attenuation.Law.CubeRoot, 1000.0, 1.6)
    assert (made.info.method_id, made.info.verification) == ("ATT-CBRT-USBM656", core.Verification.UserDefined)


def test_fit_three_rows_one_nan():
    events = _events().iloc[:3].copy()
    events.loc[events.index[1], "airblast_pa"] = np.nan
    with pytest.warns(UserWarning), pytest.raises(ValueError, match=DOMAIN + " 'airblast_pa'.*at least 3"):
        airblast.fit(events)


def test_outliers_index_the_rows_used():
    d = np.linspace(100.0, 2000.0, 20)
    p = 5000.0 * (d / np.cbrt(50.0)) ** -1.2 * np.where(np.arange(20) % 2, 1.02, 0.98)
    p[12] *= 10.0  # atípico sembrado
    events = pd.DataFrame({"distance_m": d, "charge_kg": 50.0, "airblast_pa": p}, index=[f"e{i}" for i in range(20)])
    events.loc[["e3", "e7"], "airblast_pa"] = np.nan  # sin micrófono
    with pytest.warns(UserWarning, match=r"refer to the 18 rows used"):
        law = airblast.fit(events)
    used = events[np.isfinite(events["airblast_pa"])]
    assert list(used.index[law.outliers]) == ["e12"]
    assert len(law.residuals) == len(used)


def test_submodule_is_exposed():
    import blastingpy

    assert blastingpy.airblast is airblast


# --- Revisión final del sprint 3 ---------------------------------------------------------------------------------

def test_median_warning_with_and_without_se():
    w = airblast.predict_published("USBM-RI8485/coal-highwall/2Hz", 300.0, 100.0).warnings
    assert any("median of the published equation" in x and "no safety margin" in x for x in w)
    assert any("published standard error: 4.9 dB" in x for x in w)
    w = airblast.predict_published("OSMRE-BLEP/highwall", 300.0, 100.0).warnings
    assert any("median of the published equation" in x for x in w)
    assert not any("standard error" in x for x in w)


def test_cslow_equation_warns_about_the_quantity():
    w = airblast.predict_published("USBM-RI8485/coal-parting/c-slow", 300.0, 100.0).warnings
    assert any("C-weighted slow level" in x for x in w)
    assert not any("C-weighted slow level" in x
                   for x in airblast.predict_published("USBM-RI8485/coal-parting/2Hz", 300.0, 100.0).warnings)


def test_make_site_law_airblast():
    law = airblast.make_site_law(20000.0, 1.2, sigma=0.05)
    assert law.info.method_id == "AIR-CBRT-USER" and law.info.verification == core.Verification.UserDefined
    assert "8485" in law.info.source and law.sigma == 0.05 and law.z == 2.0
    assert airblast.predict(law, 400.0, 100.0, Level.Median).value == pytest.approx(
        20000.0 * (400.0 / 100.0 ** (1 / 3)) ** -1.2)
    assert airblast.max_charge(law, 400.0, 100.0).value > 0


def test_vibration_law_is_refused():
    from blastingpy import attenuation

    ev = pd.DataFrame({"distance_m": [100.0, 200, 300, 400, 500], "charge_kg": [10.0, 20, 30, 40, 50],
                       "ppv_mm_s": [20.0, 15, 9, 8, 5]})
    law = attenuation.fit(ev, "cbrt")
    for call in (lambda: airblast.predict(law, 300.0, 30.0), lambda: airblast.max_charge(law, 300.0, 5.0),
                 lambda: airblast.distance_for(law, 30.0, 5.0)):
        with pytest.raises(ValueError, match=DOMAIN + " 'law'"):
            call()
    airblast.predict(airblast.published("OSMRE-BLEP/highwall").law, 300.0, 30.0)  # catalog law is airblast
