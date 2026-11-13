import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
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


def test_compliance_distance_table():
    ev = synthetic_events(n=9, seed=8)
    ax = plot_compliance(ev, "OSMRE-30CFR816.67", "distance-table")
    assert ax.get_xlabel() == "Distance (m)" and len(ax.collections) >= 2  # hlines + puntos
    assert sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, "get_offsets")
               and c.get_offsets().ndim == 2 and c.get_offsets().shape[1] == 2
               and len(c.get_offsets()) in range(1, 10)) >= 9


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
