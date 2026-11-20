import sys

import numpy as np
import pytest

from blastingpy import io
from blastingpy import pattern as pt

ezdxf = pytest.importorskip("ezdxf")

# Datos inventados: rectángulo de 40 × 20 m con la cresta en y = 0.
RECT = np.array([[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [0.0, 20.0]])
CREST = np.array([[0.0, 0.0], [40.0, 0.0]])
pytestmark = pytest.mark.filterwarnings("ignore:rear gap")


def _holes():
    h = pt.layout(RECT, CREST, burden_m=4.0)
    h = pt.sequence(h, scheme="row", hole_delay_ms=17.0, row_delay_ms=42.0)
    return pt.charge(h, bench_height_m=10.0, subdrill_m=1.0, stemming_m=3.0, diameter_mm=165.0,
                     explosive_density_kg_m3=850.0)


def test_dxf_round_trip(tmp_path):
    h = _holes()
    f = tmp_path / "malla.dxf"
    pt.to_dxf(h, f, outline=RECT, crest=CREST)
    msp = ezdxf.readfile(f).modelspace()
    circles = msp.query("CIRCLE")
    assert len(circles) == len(h)
    got = sorted((c.dxf.center.x, c.dxf.center.y) for c in circles)
    want = sorted(zip(h["x_m"], h["y_m"]))
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert all(abs(c.dxf.radius - 0.0825) < 1e-12 for c in circles)
    layers = {la.dxf.name for la in ezdxf.readfile(f).layers}
    assert {"CONTORNO", "FILA_01", "TEXTO", "AMARRE"} <= layers
    outline, crest = io.read_dxf_outline(f, outline_layer="CONTORNO", crest_layer="CRESTA")
    np.testing.assert_allclose(outline, RECT, atol=1e-9)
    np.testing.assert_allclose(crest, CREST, atol=1e-9)
    assert any("/" in t.dxf.text and "ms" in t.dxf.text for t in msp.query("TEXT"))


def test_no_sequence_no_tie_layer(tmp_path):
    h = pt.layout(RECT, CREST, burden_m=4.0)
    pt.to_dxf(h, tmp_path / "a.dxf", outline=RECT, crest=CREST, diameter_mm=165.0)
    doc = ezdxf.readfile(tmp_path / "a.dxf")
    assert not doc.modelspace().query("LWPOLYLINE[layer=='AMARRE']")
    assert all("ms" not in t.dxf.text for t in doc.modelspace().query("TEXT"))


def test_diameter_required(tmp_path):
    h = pt.layout(RECT, CREST, burden_m=4.0)
    with pytest.raises(ValueError, match="diameter_mm"):
        pt.to_dxf(h, tmp_path / "a.dxf", outline=RECT, crest=CREST)


def _write(path, polylines):
    doc = ezdxf.new("R2010")
    for layer, pts, close in polylines:
        doc.modelspace().add_lwpolyline(pts, close=close, dxfattribs={"layer": layer})
    doc.saveas(path)


def test_read_outline_accepts_closed_by_repeat(tmp_path):
    f = tmp_path / "r.dxf"
    _write(f, [("C", [*RECT.tolist(), RECT[0].tolist()], False), ("K", CREST.tolist(), False)])
    outline, crest = io.read_dxf_outline(f, outline_layer="C", crest_layer="K")
    assert outline.shape == (4, 2) and crest.shape == (2, 2)


def test_read_outline_rejects_open_outline(tmp_path):
    f = tmp_path / "r.dxf"
    _write(f, [("C", RECT.tolist(), False), ("K", CREST.tolist(), False)])
    with pytest.raises(ValueError, match="closed"):
        io.read_dxf_outline(f, outline_layer="C", crest_layer="K")


def test_read_outline_rejects_two_polylines(tmp_path):
    f = tmp_path / "r.dxf"
    _write(f, [("C", RECT.tolist(), True), ("C", (RECT + 100).tolist(), True), ("K", CREST.tolist(), False)])
    with pytest.raises(ValueError, match="has 2 polylines"):
        io.read_dxf_outline(f, outline_layer="C", crest_layer="K")


def test_read_outline_warns_non_metric_units(tmp_path):
    f = tmp_path / "mm.dxf"
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # milímetros
    doc.modelspace().add_lwpolyline(RECT.tolist(), close=True, dxfattribs={"layer": "C"})
    doc.modelspace().add_lwpolyline(CREST.tolist(), dxfattribs={"layer": "K"})
    doc.saveas(f)
    with pytest.warns(UserWarning, match=r"\$INSUNITS = 4"):
        io.read_dxf_outline(f, outline_layer="C", crest_layer="K")


def test_read_outline_in_world_coordinates_when_mirrored(tmp_path):
    # extrusión (0,0,-1): la x del OCS es la -x del mundo; leer el OCS espejaba la malla
    f = tmp_path / "m.dxf"
    doc = ezdxf.new("R2010")
    ocs = (RECT * [-1.0, 1.0]).tolist()
    doc.modelspace().add_lwpolyline(ocs, close=True, dxfattribs={"layer": "C", "extrusion": (0, 0, -1)})
    doc.modelspace().add_lwpolyline((CREST * [-1.0, 1.0]).tolist(), dxfattribs={"layer": "K", "extrusion": (0, 0, -1)})
    doc.saveas(f)
    outline, crest = io.read_dxf_outline(f, outline_layer="C", crest_layer="K")
    assert np.allclose(outline, RECT) and np.allclose(crest, CREST)


def test_read_outline_follows_arcs(tmp_path):
    # lado derecho en semicírculo de radio 10 hacia afuera: debe llegar a x = 50, no quedarse en la cuerda x = 40
    f = tmp_path / "a.dxf"
    doc = ezdxf.new("R2010")
    pts = [(0, 0, 0), (40, 0, 1), (40, 20, 0), (0, 20, 0)]
    doc.modelspace().add_lwpolyline(pts, format="xyb", close=True, dxfattribs={"layer": "C"})
    doc.modelspace().add_lwpolyline(CREST.tolist(), dxfattribs={"layer": "K"})
    doc.saveas(f)
    outline, _ = io.read_dxf_outline(f, outline_layer="C", crest_layer="K")
    assert outline[:, 0].max() == pytest.approx(50.0, abs=0.05)
    assert not np.allclose(outline[0], outline[-1])  # sin el punto de cierre repetido


def test_read_outline_layer_names_ignore_case(tmp_path):
    f = tmp_path / "c.dxf"
    _write(f, [("CONTORNO", RECT.tolist(), True), ("CRESTA", CREST.tolist(), False)])
    outline, _ = io.read_dxf_outline(f, outline_layer="contorno", crest_layer="cresta")
    assert outline.shape == (4, 2)


def _ties(f):
    return [[tuple(p[:2]) for p in e.get_points()]
            for e in ezdxf.readfile(f).modelspace().query("LWPOLYLINE[layer=='AMARRE']")]


def test_tie_in_row_scheme_one_polyline_per_row(tmp_path):
    f = tmp_path / "row.dxf"
    h = _holes()
    pt.to_dxf(h, f, outline=RECT, crest=CREST)
    ties = _ties(f)
    assert len(ties) == h["row"].nunique()
    for line in ties:  # en el orden de encendido: along_m creciente desde la izquierda
        xs = [x for x, _ in line]
        assert xs == sorted(xs)


def test_tie_in_v_scheme_goes_out_from_the_apex(tmp_path):
    f = tmp_path / "v.dxf"
    h = pt.layout(RECT, CREST, burden_m=4.0, spacing_m=4.0, arrangement="square")
    h = pt.sequence(h, scheme="v", hole_delay_ms=17.0, row_delay_ms=42.0)
    pt.to_dxf(h, f, outline=RECT, crest=CREST, diameter_mm=165.0)
    ties = _ties(f)
    assert len(ties) == 2 * h["row"].nunique()  # dos ramas por fila
    for r, g in h.groupby("row"):
        apex = g.loc[g["firing_time_s"].idxmin()]
        branches = [line for line in ties if np.isclose(line[0][1], apex["y_m"])]
        assert len(branches) == 2
        for line in branches:
            assert np.allclose(line[0], (apex["x_m"], apex["y_m"]))  # empieza en el vértice
            d = [abs(x - apex["x_m"]) for x, _ in line]
            assert d == sorted(d) and len(set(np.sign([x - apex["x_m"] for x, _ in line[1:]]))) == 1
        assert sum(len(line) for line in branches) == len(g) + 1  # el vértice está en las dos


def test_layers_by_delay(tmp_path):
    f = tmp_path / "d.dxf"
    pt.to_dxf(_holes(), f, outline=RECT, crest=CREST, layers_by="delay")
    names = {la.dxf.name for la in ezdxf.readfile(f).layers}
    assert {"RETARDO_0000", "RETARDO_0017"} <= names


def test_layer_names_override(tmp_path):
    f = tmp_path / "n.dxf"
    pt.to_dxf(_holes(), f, outline=RECT, crest=CREST, layer_names={"outline": "LIMITE", "prefix": "R_"})
    names = {la.dxf.name for la in ezdxf.readfile(f).layers}
    assert {"LIMITE", "R_01"} <= names


def test_missing_ezdxf_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ezdxf", None)
    with pytest.raises(ImportError, match=r"blastingpy\[dxf\]"):
        pt.to_dxf(_holes(), tmp_path / "x.dxf", outline=RECT, crest=CREST)
    with pytest.raises(ImportError, match=r"blastingpy\[dxf\]"):
        io.read_dxf_outline(tmp_path / "x.dxf", outline_layer="C", crest_layer="K")
