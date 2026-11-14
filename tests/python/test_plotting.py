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


SHIPPED = [(p, c) for p in criteria.list_profiles() for c in criteria.load_profile(p).criteria]


@pytest.mark.parametrize("profile, criterion", SHIPPED)
def test_every_shipped_criterion_draws(profile, criterion):
    # table-13-* empieza en 0 Hz: geomspace no admite 0 y el gráfico se caía.
    ev = synthetic_events(n=15, seed=10)
    ax = plot_compliance(ev, profile, criterion)
    assert sum(len(c.get_offsets()) for c in _points(ax)) == 15


@pytest.mark.parametrize("frequency_range", [(0.2, 300.0), (5.0, 30.0)])
def test_curve_starting_at_zero_hz_is_sampled_from_a_positive_frequency(frequency_range):
    ev = synthetic_events(n=5, seed=11, frequency_range=frequency_range)
    ax = plot_compliance(ev, "USBM-RI8507-1980", "table-13-plaster")
    x, _ = ax.lines[0].get_data()
    f = ev["frequency_hz"]
    assert x.min() == pytest.approx(min(f.min(), 1.0)) and x.max() == pytest.approx(max(f.max() * 1.5, 100.0))


def _counts(ax):
    pts = _points(ax)
    return sum(len(c.get_offsets()) for c in pts), sum(len(c.get_offsets()) for c in pts if _color(c) == RED)


def test_chart_draws_exactly_the_candidates_of_the_verdict():
    # ppv_kind vertical: el veredicto usa ppv_mm_s como componente vertical; el gráfico no la dibujaba.
    ev = synthetic_events(n=20, seed=12)
    ev["ppv_kind"] = "vertical"
    out = criteria.evaluate(ev, "USBM-RI8507-1980", "appendix-b-drywall")
    drawn, red = _counts(plot_compliance(ev, "USBM-RI8507-1980", "appendix-b-drywall"))
    assert (out.status == "exceeds").sum() > 0
    assert drawn == 20 and red == (out.status == "exceeds").sum()
    # vector_sum nunca es candidato de un perfil de componente máxima: no se dibuja contra la tabla.
    ev["ppv_kind"] = "vector_sum"
    ax = plot_compliance(ev, "OSMRE-30CFR816.67", "distance-table")
    assert _counts(ax) == (0, 0) and any("not drawn: 20" in t for t in _legend(ax))


def test_components_without_a_peak_frequency_column():
    # Sin columna frequency_hz el pico no tiene par; antes se enmascaraba la transversal en su lugar.
    ev = pd.DataFrame([dict(distance_m=100.0, charge_kg=10.0, ppv_mm_s=8.0, ppv_kind="vertical", ppv_t_mm_s=30.0,
                            freq_t_hz=10.0)])
    ax = plot_compliance(ev, "USBM-RI8507-1980", "appendix-b-drywall")
    assert _counts(ax) == (1, 1)
    # Sin ningún par dibujable: no hay IndexError, la leyenda cuenta el evento no dibujado.
    ev = pd.DataFrame([dict(distance_m=100.0, charge_kg=10.0, ppv_mm_s=8.0, ppv_kind="component_max")])
    ax = plot_compliance(ev, "USBM-RI8507-1980", "appendix-b-drywall")
    assert _counts(ax) == (0, 0) and any("not drawn: 1" in t for t in _legend(ax))


def test_curve_without_any_defined_limit_is_a_domain_error(monkeypatch):
    import numpy as np
    from blastingpy import plotting
    monkeypatch.setattr(plotting, "_curve_frequencies", lambda segments, f_data: np.array([1e6, 2e6]))
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter 'criterion'"):
        plot_compliance(synthetic_events(n=3), "USBM-RI8507-1980", "appendix-b-drywall")


def test_plotting_uses_no_private_criteria_helpers():
    import inspect
    from blastingpy import plotting
    source = inspect.getsource(plotting)
    for name in ("_get_criterion", "_COMPONENTS", "_cc", "_REL_TOL"):
        assert f"criteria.{name}" not in source


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


def test_plot_record_one_axes_per_channel_in_order_and_labels():
    from blastingpy.datasets import synthetic_record
    from blastingpy.plotting import plot_record
    rec = synthetic_record(mic_pa=50.0)
    fig = plot_record(rec)
    assert len(fig.axes) == 4
    assert [a.get_title() for a in fig.axes] == ["transverse", "vertical", "longitudinal", "mic"]
    assert [a.get_ylabel() for a in fig.axes] == ["Velocity (mm/s)"] * 3 + ["Pressure (Pa)"]
    assert fig.axes[-1].get_xlabel() == "Time (s)"
    assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[3])
    assert len(fig.axes[0].lines) >= 2  # trazo y marca del pico


def test_plot_record_without_mic_and_into_given_figure():
    from blastingpy.datasets import synthetic_record
    from blastingpy.plotting import plot_record
    fig = plt.figure()
    out = plot_record(synthetic_record(mic_pa=None), fig=fig)
    assert out is fig and len(fig.axes) == 3
