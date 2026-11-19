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
