"""The two standard charts: the site law regression and the events against a standard's limit.

matplotlib is an optional dependency (``pip install blastingpy[viz]``); it is imported when a function is called.
The functions draw on ``ax`` or create a figure when ``ax`` is None; they never call ``plt.show()``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import criteria
from .attenuation import Law

__all__ = ["plot_site_law", "plot_compliance", "plot_record"]

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
    """Frequencies where the curve is drawn: its domain, made finite and positive from the data when it is open."""
    lo, hi = segments[0].f_min, segments[-1].f_max
    if not math.isfinite(hi):
        hi = max([f * 1.5 for f in f_data] + [100.0])
    if lo <= 0:  # geomspace no admite 0: se empieza en la menor frecuencia de los datos o en 1 Hz
        lo = min([*f_data, 1.0, hi / 10.0])
    # Los bordes de tramo, y un punto justo antes, para que un escalón se dibuje vertical.
    edges = np.array([s.f_max for s in segments if lo < s.f_max < hi])
    return np.sort(np.concatenate([np.geomspace(lo, hi, _SAMPLES), edges, edges * (1 - 1e-9)]))


def _curve_limits(profile, criterion, frequencies) -> np.ndarray:
    """The limit at each frequency, from the verdict's own candidates (NaN where the curve has none)."""
    probe = pd.DataFrame({"ppv_mm_s": 1.0, "ppv_kind": profile.quantity, "frequency_hz": frequencies})
    return np.array([cands[0].limit_mm_s for cands in criteria.candidates(probe, profile, criterion)])


_WITHIN, _ABOVE, _NONE = "within limit", "above limit", "no limit at this coordinate"
_POINT_COLORS = {_WITHIN: "tab:green", _ABOVE: "tab:red", _NONE: "tab:gray"}
_VERDICTS = ("complies", "exceeds", "indeterminate", "not_comparable")
# Series del gráfico por etiqueta de candidato: el valor propio del evento y cada componente.
_SERIES = {"": ("Peak", "o"), "transverse": ("Transverse", "^"), "vertical": ("Vertical", "^"),
           "longitudinal": ("Longitudinal", "^")}


def plot_compliance(events, profile, criterion, ax=None):
    """Events against the limit of ``criterion`` (frequency curve or distance table); returns the Axes.

    The chart draws exactly the candidates :func:`criteria.evaluate` used (see :func:`criteria.candidates`), and the
    limit with the same evaluation. Each point is coloured by its own comparison with its limit (within, above, or no
    limit at its coordinate); a point of an event whose verdict is ``not_comparable`` or ``indeterminate`` is hollow.
    The legend counts the events by verdict and the events with no point drawn.
    """
    profile = criteria.load_profile(profile)
    per_event = criteria.candidates(events, profile, criterion)  # valida el criterio y las columnas
    c = profile.criteria[criterion]
    if c.kind not in ("frequency_curve", "distance_table"):
        raise _unsupported("criterion", "frequency_curve and distance_table criteria")
    status = criteria.evaluate(events, profile, criterion)["status"].to_numpy()
    created = ax is None
    ax = _axes(ax)
    from matplotlib.lines import Line2D
    label = f"{profile.profile_id} {criterion} [{c.verification}]"
    if c.kind == "frequency_curve":
        coords = [[cand.frequency_hz for cand in cands] for cands in per_event]
        freqs = _curve_frequencies(c.segments, [x for xs in coords for x in xs if math.isfinite(x) and x > 0])
        limits = _curve_limits(profile, criterion, freqs)
        defined = np.isfinite(limits)
        if not defined.any():
            raise ValueError(f"Domain error [BlastingPy]: parameter 'criterion' {criterion!r} has no limit at any "
                             f"frequency to draw")
        ax.plot(freqs[defined], limits[defined], color="black", label=label)
        xlabel = "Frequency (Hz)"
    else:
        dist = events["distance_m"].to_numpy(float)
        coords = [[d] * len(cands) for d, cands in zip(dist, per_event)]
        known = [d for d in dist if math.isfinite(d) and d > 0]
        finite = [r.d_max_m for r in c.rows if math.isfinite(r.d_max_m)]
        left = min([*known, *finite], default=1.0) / 2
        x_end = max([*(d * 1.5 for d in known), *(d * 2 for d in finite)], default=left * 200)
        for i, r in enumerate(c.rows):
            right = r.d_max_m if math.isfinite(r.d_max_m) else x_end
            ax.hlines(r.limit_mm_s, left, right, color="black", label=label if i == 0 else None)
            left = right
        xlabel = "Distance (m)"
    hollow = np.isin(status, ("not_comparable", "indeterminate"))
    drawn = np.zeros(len(status), dtype=bool)
    points = {key: [] for key in _SERIES}  # serie -> [(x, ppv, color, hueco)]
    for i, (cands, xs) in enumerate(zip(per_event, coords)):
        for cand, x in zip(cands, xs):
            if not (math.isfinite(x) and x > 0):
                continue
            drawn[i] = True
            kind = _NONE if math.isnan(cand.limit_mm_s) else (_ABOVE if cand.exceeds else _WITHIN)
            points[cand.label].append((x, cand.ppv_mm_s, kind, bool(hollow[i])))
    seen = set()
    for key, pts in points.items():
        for k in _POINT_COLORS:
            for h in (False, True):
                sel = [(x, v) for x, v, kind, hol in pts if kind == k and hol == h]
                if not sel:
                    continue
                name = k + (" (event not fully evaluated)" if h else "")
                colors = dict(facecolors="none", edgecolors=_POINT_COLORS[k]) if h else dict(color=_POINT_COLORS[k])
                xs, values = zip(*sel)
                ax.scatter(xs, values, s=22, marker=_SERIES[key][1], label=None if name in seen else name, **colors)
                seen.add(name)
    handles, labels = ax.get_legend_handles_labels()
    series = [_SERIES[key] for key, pts in points.items() if pts]
    for name, marker in series:
        if len(series) > 1:
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


_CHANNEL_ORDER = ("transverse", "vertical", "longitudinal", "mic")


def plot_record(record, *, fig=None):
    """One panel per channel of ``record`` (transverse, vertical, longitudinal, mic), sharing the time axis; returns the figure.

    Each panel marks the channel's peak ``|x|`` (value and time in the legend). Draws into ``fig`` (adding its
    subplots) when given, else creates a figure; never calls ``plt.show()``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting: pip install blastingpy[viz]") from None
    names = [c for c in _CHANNEL_ORDER if c in record.channels]
    if fig is None:
        fig = plt.figure(figsize=(9, 2.2 * len(names) + 0.8))
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    t = record.time_s()
    for ax, name in zip(axes, names):
        x = np.asarray(record.channels[name], dtype=float)
        i = int(np.argmax(np.abs(x)))
        unit = "Pa" if name == "mic" else "mm/s"
        ax.plot(t, x, color="tab:blue", linewidth=0.8)
        ax.plot([t[i]], [x[i]], linestyle="none", marker="o", color="tab:red",
                label=f"Peak {abs(x[i]):.4g} {unit} at {t[i]:.4g} s")
        ax.set(title=name, ylabel="Pressure (Pa)" if name == "mic" else "Velocity (mm/s)")
        ax.legend(loc="upper right", fontsize="small")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig
