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
    return plt.subplots(figsize=(9, 5))[1] if ax is None else ax


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


_WITHIN, _ABOVE, _NONE = "within limit", "above limit", "no limit at this coordinate"
_POINT_COLORS = {_WITHIN: "tab:green", _ABOVE: "tab:red", _NONE: "tab:gray"}
_VERDICTS = ("complies", "exceeds", "indeterminate", "not_comparable")


def _point_limit(c, x: float) -> float:
    """Limit at the coordinate ``x`` (NaN if there is none), with the same evaluation as the verdict."""
    if not (math.isfinite(x) and x > 0):
        return math.nan
    if c.kind == "frequency_curve":
        r = criteria._cc.limit_at(list(c.segments), x)
        return r.limit if r.defined else math.nan
    return next((r.limit_mm_s for r in c.rows if x <= r.d_max_m * (1 + criteria._REL_TOL)), math.nan)


def plot_compliance(events, profile, criterion, ax=None):
    """Events against the limit of ``criterion`` (frequency curve or distance table); returns the Axes.

    The limit is drawn with the same evaluation as :func:`criteria.evaluate`. Each point is coloured by its own
    comparison with the limit at its coordinate (within, above, or no limit there); a point of an event whose verdict
    is ``not_comparable`` or ``indeterminate`` is hollow. The legend counts the events by verdict.
    """
    profile, c = criteria._get_criterion(profile, criterion)
    if c.kind not in ("frequency_curve", "distance_table"):
        raise _unsupported("criterion", "frequency_curve and distance_table criteria")
    status = criteria.evaluate(events, profile, criterion)["status"].to_numpy()
    created = ax is None
    ax = _axes(ax)
    from matplotlib.lines import Line2D
    label = f"{profile.profile_id} {criterion} [{c.verification}]"
    if c.kind == "frequency_curve":
        pairs = [("Peak", "o", "ppv_mm_s", "frequency_hz")]
        pairs += [(name.capitalize(), "^", v, f) for name, v, f in criteria._COMPONENTS]
        pts = [(name, marker, events[v].to_numpy(float), events[f].to_numpy(float))
               for name, marker, v, f in pairs if v in events and f in events]
        if profile.quantity == "component_max":  # un pico de otra magnitud no se compara con este límite
            pts[0] = (*pts[0][:2], np.where(events["ppv_kind"] == "component_max", pts[0][2], np.nan), pts[0][3])
        freqs = _curve_frequencies(c.segments, [x for *_, f in pts for x in f[np.isfinite(f)]])
        defined = [(x, r.limit) for x in freqs if (r := criteria._cc.limit_at(list(c.segments), float(x))).defined]
        ax.plot(*zip(*defined), color="black", label=label)
        xlabel = "Frequency (Hz)"
    else:
        dist = events["distance_m"].to_numpy(float)
        finite = [r.d_max_m for r in c.rows if math.isfinite(r.d_max_m)]
        left = min([dist.min(), *finite]) / 2
        x_end = max([dist.max() * 1.5, *(d * 2 for d in finite)])
        for i, r in enumerate(c.rows):
            right = r.d_max_m if math.isfinite(r.d_max_m) else x_end
            ax.hlines(r.limit_mm_s, left, right, color="black", label=label if i == 0 else None)
            left = right
        pts = [("Peak", "o", events["ppv_mm_s"].to_numpy(float), dist)]
        xlabel = "Distance (m)"
    hollow = np.isin(status, ("not_comparable", "indeterminate"))
    drawn = np.zeros(len(status), dtype=bool)
    seen = set()
    for _, marker, values, xs in pts:
        ok = np.isfinite(values) & np.isfinite(xs)
        drawn |= ok
        limits = np.array([_point_limit(c, x) if k else math.nan for x, k in zip(xs, ok)])
        kind = np.where(np.isnan(limits), _NONE, np.where(values <= limits * (1 + criteria._REL_TOL), _WITHIN, _ABOVE))
        for k in _POINT_COLORS:
            for h in (False, True):
                m = ok & (kind == k) & (hollow == h)
                if not m.any():
                    continue
                name = k + (" (event not fully evaluated)" if h else "")
                colors = dict(facecolors="none", edgecolors=_POINT_COLORS[k]) if h else dict(color=_POINT_COLORS[k])
                ax.scatter(xs[m], values[m], s=22, marker=marker, label=None if name in seen else name, **colors)
                seen.add(name)
    handles, labels = ax.get_legend_handles_labels()
    for name, marker, *_ in pts:
        if len(pts) > 1:
            handles.append(Line2D([], [], linestyle="none", marker=marker, color="0.4"))
            labels.append(name)
    counts = ", ".join(f"{(status == v).sum()} {v.replace('_', ' ')}" for v in _VERDICTS if (status == v).any())
    text = f"events: {counts}" + (f"; not drawn: {int((~drawn).sum())}" if not drawn.all() else "")
    handles.append(Line2D([], [], linestyle="none", label=text))
    labels.append(text)
    ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel="PPV (mm/s)", title=f"{profile.profile_id}: {criterion}")
    # Fuera de los ejes, para no tapar puntos; con ax propio, el usuario ajusta el margen (p. ej. bbox_inches="tight").
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize="small")
    if created:
        ax.figure.tight_layout()
    return ax
