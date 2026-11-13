"""De punta a punta con números: el ejemplo publicado de BIVDEP (OSMRE 2019, 184 eventos artificiales) en pies, libras y
pulgadas por segundo → read_events(units="imperial") → fit → max_charge → evaluate contra 30 CFR 816.67."""

import csv
from pathlib import Path

import pytest

from blastingpy import attenuation, criteria, read_events, units

DATA = Path(__file__).parent / "data" / "bivdep_example.csv"
COLUMNS = {"distance_m": "Distance (ft)", "charge_kg": "Charge (lb)", "ppv_mm_s": "PPV (in/s)"}


@pytest.fixture(scope="module")
def raw():
    with open(DATA, encoding="utf-8") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    assert rows[0] == ["distance_ft", "charge_lb", "ppv_in_s"]
    return [tuple(float(x) for x in r) for r in rows[1:]]


@pytest.fixture(scope="module")
def events(raw, tmp_path_factory):
    path = tmp_path_factory.mktemp("bivdep") / "campaign.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(COLUMNS.values()))
        w.writerows(raw)
    return read_events(path, columns=COLUMNS, units="imperial", ppv_kind="component_max")


def test_campaign_to_max_charge(events):
    assert len(events) == 184
    law = attenuation.fit(events)
    # BIVDEP publica beta = 1,16 y la cota 112,57 en in/s-ft-lb: la lectura en imperiales llegó bien a SI.
    assert law.beta == pytest.approx(1.16, abs=0.005)
    assert law.k_upper == pytest.approx(units.convert_k_to_si(112.57, law.beta, 0.5), rel=1e-4)
    result = attenuation.max_charge(law, 304.8, 25.4)  # 1000 ft, 1 in/s, sobre la cota superior
    by_hand = (25.4 * 304.8 ** law.beta / law.k_upper) ** (1.0 / (0.5 * law.beta))  # W = (v D^beta / K)^(1/(beta/2))
    assert result.value == pytest.approx(by_hand, rel=1e-12)
    assert result.value == pytest.approx(130.871, abs=1e-3)  # 288,52 lb
    assert result.warnings == []


def test_campaign_against_osmre_distance_table(raw, events):
    out = criteria.evaluate(events, "OSMRE-30CFR816.67", "distance-table")
    # Conteo independiente en las unidades de la fuente: 1,25 in/s hasta 300 ft, 1,00 hasta 5000 ft, 0,75 después.
    by_hand = sum(v > (1.25 if d <= 300 else 1.00 if d <= 5000 else 0.75) for d, _, v in raw)
    assert by_hand == 47
    assert (out.status == "exceeds").sum() == 47 and (out.status == "complies").sum() == 184 - 47
    assert out.attrs["profile_id"] == "OSMRE-30CFR816.67"
