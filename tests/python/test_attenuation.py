import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import blastingpy
from blastingpy import attenuation as att

LAW = att.make_site_law(att.Law.SquareRoot, 1000.0, 1.6)
SIGMA_LAW = att.make_site_law(att.Law.SquareRoot, 1000.0, 1.6, sigma=0.115470)


def _events(ds=(50.0, 100.0, 200.0, 500.0), k=1000.0, beta=1.6):
    rows = [(d, w, k * (d / w**0.5) ** -beta) for d in ds for w in (25.0, 100.0, 400.0)]
    return pd.DataFrame(rows, columns=["distance_m", "charge_kg", "ppv_mm_s"])


def test_module_is_reachable_from_the_package():
    assert blastingpy.attenuation is att
    for name in ("SiteLaw", "Law", "Level", "Prediction", "make_site_law", "normal_quantile", "predict_ppv",
                 "max_charge", "distance_for_ppv", "fit"):
        assert hasattr(att, name)
    for term in ("base-10", "RI 8507", "Phi(z)", "Student-t", "normal_quantile(0.95)"):
        assert term in att.__doc__


def test_brief_values_and_read_only_prediction():
    p = att.predict_ppv(LAW, 100.0, 25.0)
    assert p.value == pytest.approx(8.2864, abs=1e-3)
    assert p.warnings == []
    with pytest.raises(AttributeError):
        p.value = 1.0
    assert att.max_charge(LAW, 100.0, 8.2864).value == pytest.approx(25.0, abs=1e-2)
    assert att.distance_for_ppv(LAW, 25.0, 8.2864).value == pytest.approx(100.0, abs=1e-2)


def test_upper_is_1_7019_times_median_and_is_the_default():
    median = att.predict_ppv(SIGMA_LAW, 100.0, 25.0, att.Level.Median).value
    upper = att.predict_ppv(SIGMA_LAW, 100.0, 25.0).value
    assert upper / median == pytest.approx(1.7019, abs=1e-3)
    assert upper == att.predict_ppv(SIGMA_LAW, 100.0, 25.0, level=att.Level.Upper).value


def test_published_bulletin_656_scaled_distance():
    # USBM Bulletin 656 (Nicholls, Johnson & Duvall 1971), pp. 70-71: W = (D / SD)**2, so D = 500 and 1000 with SD = 50
    # give 100 and 400 kg, and with SD = 20 give 625 and 2500 kg.
    v50 = att.predict_ppv(LAW, 500.0, 100.0, att.Level.Median).value
    assert att.max_charge(LAW, 500.0, v50, att.Level.Median).value == pytest.approx(100.0)
    assert att.max_charge(LAW, 1000.0, v50, att.Level.Median).value == pytest.approx(400.0)
    v20 = att.predict_ppv(LAW, 500.0, 625.0, att.Level.Median).value
    assert att.max_charge(LAW, 500.0, v20, att.Level.Median).value == pytest.approx(625.0)
    assert att.max_charge(LAW, 1000.0, v20, att.Level.Median).value == pytest.approx(2500.0)


def test_fitted_range_warnings_and_errors_from_the_core():
    law = att.fit(_events())
    assert any("outside the fitted range" in w for w in att.predict_ppv(law, 1000.0, 100.0).warnings)
    assert att.predict_ppv(law, 200.0, 100.0).warnings == []
    assert att.predict_ppv(LAW, 1e6, 1e-3).warnings == []  # make_site_law: no range, never warns
    with pytest.raises(ValueError, match="parameter 'distance_m'"):
        att.predict_ppv(LAW, 0.0, 25.0)
    with pytest.raises(ValueError, match="parameter 'ppv_limit_mm_s'"):
        att.max_charge(LAW, 100.0, float("nan"))


def test_law_with_non_positive_beta_cannot_predict():
    bad = att.fit(pd.DataFrame({"distance_m": [100.0, 200.0, 400.0, 800.0], "charge_kg": [100.0] * 4,
                                "ppv_mm_s": [1.0, 2.0, 4.0, 8.0]}))
    assert bad.beta < 0
    for call in (lambda: att.predict_ppv(bad, 100.0, 25.0), lambda: att.max_charge(bad, 100.0, 5.0),
                 lambda: att.distance_for_ppv(bad, 25.0, 5.0)):
        with pytest.raises(ValueError, match="parameter 'beta'.*must not be used for prediction"):
            call()


def test_fit_recovers_the_three_laws_by_name_or_enum():
    df = _events()
    f = att.fit(df)
    assert f.law == att.Law.SquareRoot
    assert f.k == pytest.approx(1000.0, rel=1e-6)
    assert f.beta == pytest.approx(1.6, rel=1e-9)
    assert att.fit(df, law=att.Law.CubeRoot).law == att.Law.CubeRoot
    assert att.fit(df, law="cbrt").law == att.Law.CubeRoot
    g = att.fit(df, law="general", z=att.normal_quantile(0.95))
    assert g.law == att.Law.General
    assert g.coverage == pytest.approx(0.95)


def test_fit_rejects_unknown_law_and_missing_column():
    with pytest.raises(ValueError, match="parameter 'law'"):
        att.fit(_events(), law="otra")
    with pytest.raises(ValueError, match="charge_kg"):
        att.fit(_events().drop(columns="charge_kg"))


@pytest.mark.parametrize("law", ["sqrt", "cbrt", "general"])
def test_round_trip_in_python(law):
    f = att.fit(_events(), law=law)
    w = att.max_charge(f, 150.0, 12.0).value
    assert att.predict_ppv(f, 150.0, w).value == pytest.approx(12.0)
    d = att.distance_for_ppv(f, 60.0, 12.0).value
    assert att.predict_ppv(f, d, 60.0).value == pytest.approx(12.0)


_dist = st.floats(10.0, 5000.0)
_ppv = st.floats(0.1, 100.0)


@settings(max_examples=200, deadline=None)
@given(d1=_dist, d2=_dist, v=_ppv)
def test_max_charge_grows_with_distance(d1, d2, v):
    lo, hi = sorted((d1, d2))
    assert att.max_charge(SIGMA_LAW, lo, v).value <= att.max_charge(SIGMA_LAW, hi, v).value


@settings(max_examples=200, deadline=None)
@given(d=_dist, v1=_ppv, v2=_ppv)
def test_max_charge_grows_with_the_limit(d, v1, v2):
    lo, hi = sorted((v1, v2))
    assert att.max_charge(SIGMA_LAW, d, lo).value <= att.max_charge(SIGMA_LAW, d, hi).value
    assert math.isfinite(att.max_charge(SIGMA_LAW, d, hi).value)
