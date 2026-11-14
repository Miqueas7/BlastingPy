import math

import numpy as np
import pandas as pd
import pytest

from blastingpy import airblast

DOMAIN = r"Domain error \[BlastingPy\]: parameter"
USBM, CFR = "USBM-RI8485-1980", "OSMRE-30CFR816.67"


@pytest.mark.parametrize("standard", [USBM, CFR])
@pytest.mark.parametrize("system, db", [("0.1Hz", 134), ("2Hz", 133), ("6Hz", 129), ("c-slow", 105)])
def test_eight_values(standard, system, db):
    lim = airblast.limit_db(standard, system=system)
    assert lim.level_db == db and lim.standard == standard and lim.system == system
    assert lim.citation and lim.wording and lim.verification == "V1"


def test_list_standards_and_systems():
    assert airblast.list_standards() == [USBM, CFR]
    assert airblast.SYSTEMS == ("0.1Hz", "2Hz", "6Hz", "c-slow")


def test_pressure_pa():
    assert airblast.limit_db(USBM, system="0.1Hz").pressure_pa == pytest.approx(100.24, rel=1e-4)
    assert airblast.limit_db(USBM, system="c-slow").pressure_pa is None


def test_cfr_approval_notes():
    for s in ("0.1Hz", "c-slow"):
        assert any("only when approved by the regulatory authority" in n.lower()
                   for n in airblast.limit_db(CFR, system=s).notes)
    assert not airblast.limit_db(CFR, system="2Hz").notes
    assert any("2" in n and "s" in n for n in airblast.limit_db(USBM, system="c-slow").notes)
    assert airblast.limit_db(USBM, system="6Hz").wording.startswith("5- or 6-Hz")


def test_5hz_alias():
    assert airblast.limit_db(USBM, system="5Hz").level_db == 129


@pytest.mark.parametrize("cutoff, system", [(0.05, "0.1Hz"), (0.1, "0.1Hz"), (0.5, "2Hz"), (2.0, "2Hz"),
                                            (2.1, "6Hz"), (5.0, "6Hz"), (6.0, "6Hz")])
def test_system_from_cutoff(cutoff, system):
    assert airblast.system_from_cutoff(cutoff) == system


@pytest.mark.parametrize("bad", [8.0, 0.0, -1.0, float("nan"), float("inf")])
def test_system_from_cutoff_errors(bad):
    with pytest.raises(ValueError, match=DOMAIN):
        airblast.system_from_cutoff(bad)


def _events(*pa):
    return pd.DataFrame({"airblast_pa": list(pa)}, index=list("abcdefg")[:len(pa)])


def test_system_is_required():
    ev = _events(100.0)
    with pytest.raises(TypeError):
        airblast.evaluate(ev, USBM)
    with pytest.raises(TypeError):
        airblast.limit_db(USBM)
    with pytest.raises(TypeError):
        airblast.evaluate_prediction("OSMRE-BLEP/highwall", 300, 100, CFR)
    for bad in (None, "3Hz"):
        for call in (lambda: airblast.evaluate(ev, USBM, system=bad), lambda: airblast.limit_db(USBM, system=bad)):
            with pytest.raises(ValueError, match=DOMAIN) as e:
                call()
            assert "'0.1Hz', '2Hz', '6Hz', 'c-slow'" in str(e.value)


def test_unknown_standard():
    with pytest.raises(ValueError, match=DOMAIN) as e:
        airblast.limit_db("nope", system="2Hz")
    assert USBM in str(e.value)


def test_verdicts():
    exact = float(airblast.from_db(133.0))
    ev = _events(100.0, exact, 50.0, np.nan, 0.0, -3.0, np.inf)
    before = ev.copy()
    out = airblast.evaluate(ev, USBM, system="2Hz")
    pd.testing.assert_frame_equal(ev, before)
    assert list(out.index) == list(ev.index)
    assert list(out["status"]) == ["exceeds", "complies", "complies"] + ["indeterminate"] * 4
    assert out["airblast_db"].iloc[0] == pytest.approx(133.98, abs=0.01)
    assert out["margin_db"].iloc[0] == pytest.approx(-0.98, abs=0.01)
    assert (out["limit_db"] == 133).all()
    assert not np.isinf(out["airblast_db"]).any()
    assert all(out["reason"].iloc[3:])
    assert airblast.evaluate(ev.iloc[:1], USBM, system="0.1Hz")["status"].iloc[0] == "complies"


def test_c_slow_not_comparable():
    out = airblast.evaluate(_events(1.0, 500.0), CFR, system="c-slow")
    assert list(out["status"]) == ["not_comparable"] * 2 and all(out["reason"])
    assert out.attrs["limit_db"] == 105


def test_attrs_and_notes():
    out = airblast.evaluate(_events(10.0), CFR, system="0.1Hz")
    for k in ("standard", "system", "limit_db", "citation", "verification", "notes"):
        assert k in out.attrs
    assert any("approved by the regulatory authority" in n for n in out.attrs["notes"])


def test_missing_column():
    with pytest.raises(ValueError, match=DOMAIN):
        airblast.evaluate(pd.DataFrame({"x": [1.0]}), USBM, system="2Hz")


def test_evaluate_prediction_system_mismatch():
    r = airblast.evaluate_prediction("OSMRE-BLEP/highwall", 300.0, 100.0, CFR, system="2Hz")
    assert set(r) == {"level_db", "limit_db", "margin_db", "status", "equation_system", "limit_system", "warnings"}
    assert r["limit_db"] == 133 and r["margin_db"] == pytest.approx(133 - r["level_db"])
    assert r["equation_system"] is None
    assert any("measuring system" in w for w in r["warnings"])


def test_evaluate_prediction_same_system():
    r = airblast.evaluate_prediction("USBM-RI8485/coal-highwall/2Hz", 300.0, 100.0, USBM, system="2Hz")
    assert r["equation_system"] == "2Hz" and r["limit_system"] == "2Hz"
    assert not any("measuring system" in w for w in r["warnings"])
    assert r["status"] in ("complies", "exceeds")


def test_evaluate_prediction_5hz_equation_and_cslow():
    eq5 = next(e for e in airblast.list_equations() if airblast.published(e).system == "5Hz")
    r = airblast.evaluate_prediction(eq5, 300.0, 100.0, USBM, system="6Hz")
    assert r["equation_system"] == "6Hz"
    assert not any("measuring system" in w for w in r["warnings"])
    r = airblast.evaluate_prediction(eq5, 300.0, 100.0, CFR, system="c-slow")
    assert r["status"] == "not_comparable" and r["margin_db"] is None
    assert any("approved by the regulatory authority" in w for w in r["warnings"])
