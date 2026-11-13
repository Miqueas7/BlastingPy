"""The two standard charts: the site law regression and the events against a standard's limit.

matplotlib is an optional dependency (``pip install blastingpy[viz]``); it is imported when a function is called.
The functions draw on ``ax`` or create a figure when ``ax`` is None; they never call ``plt.show()``.
"""

from __future__ import annotations

import math

import numpy as np

from . import criteria
from .attenuation import Law

__all__ = ["plot_site_law", "plot_compliance"]

_LABELS = {Law.SquareRoot: "m/kg^0.5", Law.CubeRoot: "m/kg^(1/3)"}
_COLORS = {"complies": "tab:green", "exceeds": "tab:red", "indeterminate": "tab:orange", "not_comparable": "tab:gray"}
_SAMPLES = 400


def _axes(ax):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install blastingpy[viz]") from None
    return plt.subplots()[1] if ax is None else ax


def _unsupported(name: str, supported: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' is not supported for plotting. "
                      f"Supported: {supported}")


def plot_site_law(fit, events, ax=None):
    """Events, median line (``k``) and upper-bound line (``k_upper``) against the scaled distance; returns the Axes.

    Only the scaled-distance laws (``sqrt`` and ``cbrt``) are supported. ``fit.outliers`` get their own marker.
    """
    if fit.law not in _LABELS:
        raise _unsupported("fit", "laws sqrt and cbrt (scaled distance)")
    ax = _axes(ax)
    sd = events["distance_m"].to_numpy(float) / events["charge_kg"].to_numpy(float) ** fit.charge_exponent
    ppv = events["ppv_mm_s"].to_numpy(float)
    out = np.zeros(len(sd), dtype=bool)
    out[list(fit.outliers)] = True  # posiciones de fila
    ax.scatter(sd[~out], ppv[~out], s=18, color="tab:blue", label="Events")
    if out.any():
        ax.scatter(sd[out], ppv[out], s=40, marker="x", color="tab:red", label="Outliers")
    x = np.geomspace(sd.min(), sd.max(), 50)
    ax.plot(x, fit.k * x ** -fit.beta, color="black", label=f"Median: K={fit.k:.4g}, beta={fit.beta:.3g}")
    ax.plot(x, fit.k_upper * x ** -fit.beta, color="tab:red", linestyle="--",
            label=f"Upper bound (z={fit.z:.3g}): K={fit.k_upper:.4g}")
    ax.set(xscale="log", yscale="log", xlabel=f"Scaled distance ({_LABELS[fit.law]})", ylabel="PPV (mm/s)",
           title="Site attenuation law")
    ax.legend()
    return ax


def _curve_frequencies(segments, f_data) -> np.ndarray:
    lo, hi = segments[0].f_min, segments[-1].f_max
    if not math.isfinite(hi):
        hi = max(max(f_data, default=0.0), lo * 100.0)
    # Los bordes de tramo, y un punto justo antes, para que un escalón se dibuje vertical.
    edges = np.array([s.f_max for s in segments if lo < s.f_max < hi])
    return np.sort(np.concatenate([np.geomspace(lo, hi, _SAMPLES), edges, edges * (1 - 1e-9)]))


def plot_compliance(events, profile, criterion, ax=None):
    """Events against the limit of ``criterion`` (frequency curve or distance table); returns the Axes.

    Points are coloured by the status ``evaluate`` gives each event; the limit is drawn with the same evaluation.
    """
    profile, c = criteria._get_criterion(profile, criterion)
    if c.kind not in ("frequency_curve", "distance_table"):
        raise _unsupported("criterion", "frequency_curve and distance_table criteria")
    status = criteria.evaluate(events, profile, criterion)["status"].to_numpy()
    ax = _axes(ax)
    label = f"{profile.profile_id} {criterion} [{c.verification}]"
    ppv = events["ppv_mm_s"].to_numpy(float)
    if c.kind == "frequency_curve":
        pairs = [("Peak", "o", "ppv_mm_s", "frequency_hz")]
        pairs += [(name.capitalize(), "^", v, f) for name, v, f in criteria._COMPONENTS]
        pts = [(name, marker, events[v].to_numpy(float), events[f].to_numpy(float))
               for name, marker, v, f in pairs if v in events and f in events]
        freqs = _curve_frequencies(c.segments, [x for *_, f in pts for x in f[np.isfinite(f)]])
        limits = [(x, criteria._cc.limit_at(list(c.segments), float(x))) for x in freqs]
        defined = [(x, r.limit) for x, r in limits if r.defined]
        ax.plot(*zip(*defined), color="black", label=label)
        xlabel = "Frequency (Hz)"
    else:
        dist = events["distance_m"].to_numpy(float)
        finite = [r.d_max_m for r in c.rows if math.isfinite(r.d_max_m)]
        left, x_end = min(dist.min(), finite[0]) / 2, max(dist.max() * 1.5, finite[-1] * 2)
        for i, r in enumerate(c.rows):
            right = r.d_max_m if math.isfinite(r.d_max_m) else x_end
            ax.hlines(r.limit_mm_s, left, right, color="black", label=label if i == 0 else None)
            left = right
        pts = [("Peak", "o", ppv, dist)]
        xlabel = "Distance (m)"
    for name, marker, values, xs in pts:
        ok = np.isfinite(values) & np.isfinite(xs)
        for s in dict.fromkeys(status):  # las componentes se pintan con el estado del evento
            m = ok & (status == s)
            if m.any():
                ax.scatter(xs[m], values[m], s=22, marker=marker, color=_COLORS[s], label=f"{name}: {s}")
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="PPV (mm/s)", title=f"{profile.profile_id}: {criterion}")
    ax.legend()
    return ax
