import math

import numpy as np
import pandas as pd
import pytest

import blastingpy as bp
from blastingpy import attenuation as at
from blastingpy import flyrock, nearfield
from blastingpy import pattern as pt
from blastingpy._blastingpy_core import _pattern as core

# Datos inventados: un rectángulo de 40 × 20 m con la cresta en el lado y = 0.
RECT = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
CREST = np.array([[0.0, 0.0], [40.0, 0.0]])
REAR_GAP = "ignore:rear gap"  # 20 m es múltiplo exacto de B = 4: el hueco trasero es B y el aviso salta (§4.2)
ALL = dict(diameter_mm=165.0, bench_height_m=10.0, rock_density_kg_m3=2700.0, explosive_density_kg_m3=850.0,
           initiation="delayed", ucs_mpa=100.0, explosive_class="low_density")
COLUMNS = ["burden_m", "spacing_m", "stemming_m", "subdrill_m", "method_id", "verification", "source", "note"]
NUMBERS = COLUMNS[:4]


def test_burden_methods_table_shape():
    t = pt.burden_methods(**ALL)
    assert tuple(t.index) == pt.BURDEN_METHODS == ("konya", "lopez_jimeno", "ash", "langefors")
    assert t.index.name == "method" and list(t.columns) == COLUMNS
    assert t.loc["konya", "verification"] == "Primary"
    assert (t.loc[["lopez_jimeno", "ash"], "verification"] == "Secondary").all()
    k = core.konya(165.0, 10.0, 2700.0, 850.0, core.Initiation.Delayed)
    assert t.loc["konya", "burden_m"] == pytest.approx(k.burden_m)
    assert t.loc["konya", "method_id"] == k.info.method_id and "p. 110" in t.loc["konya", "source"]
    assert t.loc[["konya", "lopez_jimeno", "ash"], NUMBERS].notna().all().all()
    assert t.loc["langefors", NUMBERS].isna().all()
    assert t.loc["langefors", "note"] == ("not implemented: the secondary sources disagree; pending Langefors & "
                                          "Kihlström (1978)")


def test_burden_methods_missing_input_gives_empty_row():
    t = pt.burden_methods(**{**ALL, "ucs_mpa": None})
    assert t.loc["lopez_jimeno", NUMBERS].isna().all() and "ucs_mpa" in t.loc["lopez_jimeno", "note"]
    assert t.loc[["konya", "ash"], NUMBERS].notna().all().all()
    t = pt.burden_methods(**{**ALL, "explosive_class": None})
    assert t.loc["ash", NUMBERS].isna().all() and "explosive_class" in t.loc["ash", "note"]


def test_burden_methods_requires_initiation():
    kw = {k: v for k, v in ALL.items() if k != "initiation"}
    with pytest.raises(TypeError):
        pt.burden_methods(**kw)
    with pytest.raises(ValueError, match="initiation"):
        pt.burden_methods(**kw, initiation="sometimes")
    with pytest.raises(ValueError, match="explosive_class"):
        pt.burden_methods(**{**ALL, "explosive_class": "heavy"})


@pytest.mark.filterwarnings(REAR_GAP)
def test_layout_default_staggered_spacing():
    holes = pt.layout(RECT, CREST, burden_m=4.0)
    assert list(holes.columns) == ["hole_id", "row", "x_m", "y_m", "along_m", "edge", "burden_m", "spacing_m"]
    assert list(holes["spacing_m"]) == pytest.approx([1.15 * 4.0] * len(holes)) and (holes["burden_m"] == 4.0).all()
    assert list(holes["hole_id"]) == list(range(1, len(holes) + 1)) and holes["row"].min() == 0
    with pytest.raises(ValueError, match="spacing_m"):
        pt.layout(RECT, CREST, burden_m=4.0, arrangement="rectangular")
    with pytest.raises(ValueError, match="outline"):
        pt.layout(RECT[:, 0], CREST, burden_m=4.0)


@pytest.mark.filterwarnings(REAR_GAP)
def test_layout_warns_irregular_crest():
    bumpy = np.array([[0.0, 0.0], [20.0, 2.5], [40.0, 0.0]])  # la cresta real se mete 2,5 m: la fila 0 queda cerca
    with pytest.warns(UserWarning, match="front row"):
        pt.layout(RECT, bumpy, burden_m=4.0)


def test_layout_warns_rear_gap():
    with pytest.warns(UserWarning, match="rear gap"):
        pt.layout(RECT, CREST, burden_m=4.0)


@pytest.mark.filterwarnings(REAR_GAP)
def test_sequence_adds_delays_and_rejects_missing_front_row():
    holes = pt.layout(RECT, CREST, burden_m=4.0)
    out = pt.sequence(holes, scheme="row", hole_delay_ms=17.0, row_delay_ms=100.0)
    assert "delay_ms" not in holes and (out["firing_time_s"] == out["delay_ms"] / 1000).all()
    front = out[out["row"] == 0].sort_values("along_m")
    assert list(front["delay_ms"]) == [17.0 * i for i in range(len(front))]
    with pytest.raises(ValueError, match="row"):
        pt.sequence(holes[holes["row"] != 0], scheme="row", hole_delay_ms=17.0, row_delay_ms=100.0)
    with pytest.raises(ValueError, match="scheme"):
        pt.sequence(holes, scheme="diagonal", hole_delay_ms=17.0, row_delay_ms=100.0)


@pytest.mark.filterwarnings("ignore::UserWarning")  # el hueco trasero y la mediana de la ley: no son el tema aquí
def test_chain_into_simulate_blast_and_flyrock():
    holes = pt.layout(RECT, CREST, burden_m=4.0)
    holes = pt.sequence(holes, scheme="v", hole_delay_ms=17.0, row_delay_ms=100.0)
    holes = pt.charge(holes, bench_height_m=10.0, subdrill_m=1.2, stemming_m=2.8, diameter_mm=165.0,
                      explosive_density_kg_m3=850.0)
    assert set(nearfield.BLAST_COLUMNS) | set(flyrock.HOLE_COLUMNS) <= set(holes.columns)
    assert holes["charge_kg"].iloc[0] == pytest.approx(flyrock.linear_charge(165.0, 850.0) * (10.0 + 1.2 - 2.8))
    t = np.arange(64) / 1000.0
    seed = np.exp(-60 * t) * np.sin(2 * math.pi * 50 * t)
    law = at.make_site_law(at.Law.SquareRoot, 1000.0, 1.6)
    res = nearfield.simulate_blast(holes, (200.0, 10.0), seed, sample_rate_hz=1000.0, law=law,
                                   wave_velocity_m_s=2500.0)
    assert res.max_charge_8ms_kg > 0
    assert flyrock.compare(holes, k=20.0)["rm_max_m"].notna().all()


@pytest.mark.filterwarnings(REAR_GAP)
def test_charge_rejects_no_column():
    holes = pt.layout(RECT, CREST, burden_m=4.0)
    with pytest.raises(ValueError, match="stemming_m"):
        pt.charge(holes, bench_height_m=10.0, subdrill_m=1.0, stemming_m=11.0, diameter_mm=165.0,
                  explosive_density_kg_m3=850.0)


def test_module_is_exported():
    assert bp.pattern is pt
