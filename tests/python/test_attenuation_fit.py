import math
from pathlib import Path

import pandas as pd
import pytest

from blastingpy import units
from blastingpy._blastingpy_core import Verification
from blastingpy._blastingpy_core import _attenuation as att

DATA = Path(__file__).parent / "data" / "bivdep_example.csv"
FUENTES = Path(__file__).parents[2] / "docs" / "investigacion" / "FUENTES_SPRINT1.md"


@pytest.fixture(scope="module")
def bivdep():
    df = pd.read_csv(DATA, comment="#")
    assert list(df.columns) == ["distance_ft", "charge_lb", "ppv_in_s"]
    return df


def test_bivdep_published_example_imperial(bivdep):
    # OSMRE (2019), BIVDEP v2.0, PDF p. 14: A = 37.18, B = -1.16 (beta = +1.16), R2 = 0.8328, cota A = 112.57.
    f = att.fit_site_law(bivdep.distance_ft.tolist(), bivdep.charge_lb.tolist(), bivdep.ppv_in_s.tolist(),
                         att.Law.SquareRoot)
    assert f.n == 184
    assert f.dof == 182
    assert f.k == pytest.approx(37.18, abs=0.005)
    assert f.beta == pytest.approx(1.16, abs=0.005)
    assert f.r_squared == pytest.approx(0.8328, abs=5e-5)
    assert f.sigma == pytest.approx(0.2405, abs=5e-5)
    assert f.z == 2.0
    assert f.k_upper == pytest.approx(112.57, abs=0.01)
    assert f.warnings == []


def test_bivdep_in_si_matches_converted_k(bivdep):
    imp = att.fit_site_law(bivdep.distance_ft.tolist(), bivdep.charge_lb.tolist(), bivdep.ppv_in_s.tolist(),
                           att.Law.SquareRoot)
    si = att.fit_site_law(units.to_si(bivdep.distance_ft.to_numpy(), "ft").tolist(),
                          units.to_si(bivdep.charge_lb.to_numpy(), "lb").tolist(),
                          units.to_si(bivdep.ppv_in_s.to_numpy(), "in/s").tolist(), att.Law.SquareRoot)
    assert si.beta == pytest.approx(imp.beta, rel=1e-9)
    assert si.k == pytest.approx(units.convert_k_to_si(imp.k, imp.beta, 0.5), rel=1e-9)
    assert si.sigma == pytest.approx(imp.sigma, rel=1e-9)
    assert si.distance_min_m == pytest.approx(units.to_si(bivdep.distance_ft.min(), "ft"))


def _source_in_fuentes(law_name):
    for line in FUENTES.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| `{law_name}` |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return cells[-1], cells[-2]
    raise AssertionError(law_name)


@pytest.mark.parametrize(
    "law, method_id, verification",
    [
        (att.Law.SquareRoot, "ATT-SQRT-USBM656", Verification.Primary),
        (att.Law.CubeRoot, "ATT-CBRT-USBM656", Verification.Secondary),
        (att.Law.General, "ATT-GENERAL-USBM656", Verification.Primary),
    ],
)
def test_method_info_matches_fuentes(law, method_id, verification):
    d = [50.0, 100.0, 200.0, 400.0] * 2
    w = [25.0] * 4 + [100.0] * 4
    v = [500.0 * wi**0.4 * di**-1.5 for di, wi in zip(d, w)]
    f = att.fit_site_law(d, w, v, law)
    source, mark = _source_in_fuentes(law.name)
    assert f.info.method_id == method_id
    assert f.info.verification == verification
    assert f.info.source == source
    assert mark.startswith("[V1]" if verification == Verification.Primary else "[V2]")


def test_bindings_defaults_and_read_only():
    f = att.make_site_law(att.Law.SquareRoot, 1000.0, 1.6)
    assert f.charge_exponent == 0.5
    assert f.z == 2.0
    assert f.sigma == 0.0
    assert f.info.verification == Verification.UserDefined
    assert math.isnan(f.r_squared)
    with pytest.raises(AttributeError):
        f.k = 1.0
    g = att.make_site_law(att.Law.General, 500.0, 1.5, charge_exponent=0.7, sigma=0.1, z=att.normal_quantile(0.95))
    assert g.k_upper == pytest.approx(500.0 * 10 ** (1.6448536 * 0.1), rel=1e-6)
    assert att.normal_quantile(0.975) == pytest.approx(1.9599640, abs=1e-6)


def test_domain_errors_are_value_errors():
    with pytest.raises(ValueError, match="parameter 'charge_exponent'"):
        att.make_site_law(att.Law.General, 500.0, 1.5)
    with pytest.raises(ValueError, match="parameter 'ppv_mm_s'"):
        att.fit_site_law([100.0, 200.0, 400.0], [100.0] * 3, [5.0, 0.0, 1.0], att.Law.SquareRoot)
    with pytest.raises(ValueError, match="scaled-distance law"):
        att.fit_site_law([100.0, 200.0, 400.0, 800.0], [50.0] * 4, [9.0, 4.0, 2.0, 1.0], att.Law.General)
    with pytest.raises(ValueError, match="parameter 'z'"):
        att.fit_site_law([100.0, 200.0, 400.0], [100.0] * 3, [5.0, 2.0, 1.0], att.Law.SquareRoot, z=-1.0)


def test_warnings_and_outliers_are_python_lists():
    d = [100.0, 200.0, 400.0, 800.0]
    f = att.fit_site_law(d, [100.0] * 4, [1.0, 2.0, 4.0, 8.0], att.Law.SquareRoot)
    assert any("not positive" in w for w in f.warnings)
    assert "fewer than 30 events: OSMRE BIVDEP (2019) asks for a minimum of 30" in f.warnings
    assert f.outliers == []
