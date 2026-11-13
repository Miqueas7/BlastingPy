import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from blastingpy import attenuation, criteria  # noqa: E402
from blastingpy.datasets import synthetic_events  # noqa: E402
from blastingpy.plotting import plot_compliance, plot_site_law  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_site_law_axes():
    ev = synthetic_events(n=60, seed=4)
    fit = attenuation.fit(ev, "sqrt")
    ax = plot_site_law(fit, ev)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert len(ax.lines) >= 2 and len(ax.collections) >= 1
    assert ax.get_xlabel() == "Scaled distance (m/kg^0.5)" and ax.get_ylabel() == "PPV (mm/s)"
    labels = " ".join(t.get_text() for t in ax.get_legend().get_texts())
    assert "beta" in labels and "z=2" in labels


def test_site_law_outliers_cbrt_and_given_ax():
    ev = synthetic_events("cbrt", n=60, seed=5)
    ev.loc[3, "ppv_mm_s"] *= 50
    fit = attenuation.fit(ev, "cbrt")
    assert 3 in list(fit.outliers)
    _, ax = plt.subplots()
    assert plot_site_law(fit, ev, ax) is ax
    assert "Outliers" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert ax.get_xlabel() == "Scaled distance (m/kg^(1/3))"


def test_site_law_general_unsupported():
    ev = synthetic_events(n=30)
    with pytest.raises(ValueError, match="Supported"):
        plot_site_law(attenuation.fit(ev, "general"), ev)


def test_compliance_frequency_curve():
    ev = synthetic_events(n=12, seed=6)
    ev["ppv_mm_s"] = ev["ppv_mm_s"].clip(upper=40)
    ax = plot_compliance(ev, "USBM-RI8507-1980", "appendix-b-drywall")
    assert ax.get_xscale() == "log" and ax.get_title() == "USBM-RI8507-1980: appendix-b-drywall"
    assert len(ax.lines) == 1
    assert sum(len(c.get_offsets()) for c in ax.collections) == 12
    assert any("[V1]" in t.get_text() for t in ax.get_legend().get_texts())
    # la curva dibujada coincide con el veredicto
    x, y = ax.lines[0].get_data()
    probe = pd.DataFrame([dict(distance_m=1.0, ppv_mm_s=1.0, ppv_kind="component_max", frequency_hz=float(x[100]))])
    assert criteria.evaluate(probe, "USBM-RI8507-1980", "appendix-b-drywall").limit_mm_s[0] == pytest.approx(y[100])


def test_compliance_components_add_points():
    ev = synthetic_events(n=5, seed=7)
    ev["ppv_t_mm_s"], ev["freq_t_hz"] = 1.0, 10.0
    ax = plot_compliance(ev, criteria.load_profile("USBM-RI8507-1980"), "appendix-b-drywall")
    assert sum(len(c.get_offsets()) for c in ax.collections) == 10


def _points(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets())]


def _legend(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def _color(c):
    return matplotlib.colors.to_hex(c.get_edgecolor()[0])


GREEN, RED, GRAY = (matplotlib.colors.to_hex(x) for x in ("tab:green", "tab:red", "tab:gray"))


def test_compliance_distance_table():
    ev = synthetic_events(n=9, seed=8)
    ax = plot_compliance(ev, "OSMRE-30CFR816.67", "distance-table")
    assert ax.get_xlabel() == "Distance (m)"
    assert sum(len(c.get_offsets()) for c in _points(ax)) == 9


def _bs_event(**kw):
    base = dict(distance_m=100.0, charge_kg=50.0, ppv_mm_s=5.0, ppv_kind="component_max", frequency_hz=10.0,
                ppv_t_mm_s=80.0, freq_t_hz=10.0)
    return pd.DataFrame([{**base, **kw}])


def test_point_colour_is_its_own_comparison():
    ax = plot_compliance(_bs_event(), "BS-7385-2-1993", "line-2")
    by_marker = {c.get_paths()[0].vertices.shape: _color(c) for c in _points(ax)}
    peak, transverse = [_color(c) for c in _points(ax)]  # el pico se dibuja primero
    assert peak == GREEN and transverse == RED
    assert any("1 exceeds" in t for t in _legend(ax)) and by_marker
    assert not any("Peak: exceeds" in t for t in _legend(ax))


def test_incomplete_event_is_hollow_and_within():
    ev = pd.DataFrame([dict(distance_m=1.0, charge_kg=1.0, ppv_mm_s=1.0, ppv_kind="vector_sum", frequency_hz=10.0,
                            ppv_t_mm_s=2.0, freq_t_hz=10.0)])
    ax = plot_compliance(ev, "BS-7385-2-1993", "line-2")
    assert criteria.evaluate(ev, "BS-7385-2-1993", "line-2").status[0] == "not_comparable"
    (c,) = _points(ax)
    assert len(c.get_facecolor()) == 0 and _color(c) == GREEN
    assert any("1 not comparable" in t for t in _legend(ax))


def test_point_outside_domain_has_no_limit_colour_and_not_drawn_count():
    ev = pd.concat([_bs_event(frequency_hz=5000.0, ppv_t_mm_s=float("nan")),
                    _bs_event(frequency_hz=float("nan"), ppv_t_mm_s=float("nan"))], ignore_index=True)
    ax = plot_compliance(ev, "BS-7385-2-1993", "line-2")
    assert [_color(c) for c in _points(ax)] == [GRAY]
    assert any("not drawn: 1" in t for t in _legend(ax))


def test_step_at_4_hz_is_drawn():
    ax = plot_compliance(_bs_event(), "BS-7385-2-1993", "line-2")
    x, y = ax.lines[0].get_data()
    assert y[(x < 4.0)][-1] == pytest.approx(15.08, abs=0.01) and y[x >= 4.0][0] == pytest.approx(15.0, abs=0.01)


def test_single_open_ended_table_draws():
    prof = {"profile_id": "X", "quantity": "component_max", "units": {"velocity": "mm/s", "distance": "m"},
            "criteria": [{"id": "t", "kind": "distance_table", "rows": [{"d_max": None, "limit": 10.0}]}]}
    ax = plot_compliance(synthetic_events(n=6, seed=9), prof, "t")
    assert len(ax.collections) >= 2 and sum(len(c.get_offsets()) for c in _points(ax)) == 6


def test_compliance_unsupported_criterion():
    flat = {"profile_id": "X", "quantity": "component_max", "units": {"velocity": "mm/s"},
            "criteria": [{"id": "f", "kind": "flat", "limit": 10.0}]}
    with pytest.raises(ValueError, match="Supported"):
        plot_compliance(synthetic_events(n=3), flat, "f")


def test_missing_matplotlib_message(monkeypatch):
    ev = synthetic_events(n=30)
    fit = attenuation.fit(ev, "sqrt")
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"pip install blastingpy\[viz\]"):
        plot_site_law(fit, ev)
    with pytest.raises(ImportError, match=r"pip install blastingpy\[viz\]"):
        plot_compliance(ev, "USBM-RI8507-1980", "appendix-b-drywall")


def test_import_without_matplotlib():
    code = ("import sys; sys.modules['matplotlib'] = None; import blastingpy, blastingpy.plotting; "
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.stdout.strip() == "ok", r.stderr
