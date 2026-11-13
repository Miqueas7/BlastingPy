"""Seeded synthetic blast events, for examples and tests. The repository never contains real blast data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import io

__all__ = ["synthetic_events", "synthetic_record"]

_EXPONENT = {"sqrt": 0.5, "cbrt": 1.0 / 3.0}


def _bad(name: str, message: str, value) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {message}. Received: {value!r}")


def _range(name: str, value) -> tuple[float, float]:
    try:
        lo, hi = (float(x) for x in value)
    except (TypeError, ValueError):
        raise _bad(name, "must be a (min, max) pair of numbers", value) from None
    if not (0.0 < lo < hi and np.isfinite(hi)):
        raise _bad(name, "must be finite, positive and increasing (0 < min < max)", value)
    return lo, hi


def synthetic_events(law: str = "sqrt", *, k: float = 1000.0, beta: float = 1.6, sigma: float = 0.15, n: int = 40,
                     seed: int = 0, distance_range=(50.0, 2000.0), charge_range=(20.0, 500.0),
                     frequency_range=(5.0, 60.0)) -> pd.DataFrame:
    """Events with ``PPV = k (D / W**e)**-beta * 10**(sigma * N(0, 1))``, canonical columns, ``ppv_kind="component_max"``.

    ``law`` is ``"sqrt"`` (``e = 1/2``) or ``"cbrt"`` (``e = 1/3``). Distances, charges and frequencies are log-uniform
    in their ranges (m, kg, Hz); the noise is lognormal on the PPV (normal in log10 with standard deviation ``sigma``).
    The same ``seed`` gives the same table.
    """
    if law not in _EXPONENT:
        raise _bad("law", f"must be one of {sorted(_EXPONENT)}", law)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise _bad("n", "must be an integer >= 1", n)
    for name, value, ok in (("k", k, lambda x: x > 0), ("beta", beta, lambda x: x > 0),
                            ("sigma", sigma, lambda x: x >= 0)):
        if not (np.isfinite(value) and ok(value)):
            raise _bad(name, "must be finite and " + (">= 0" if name == "sigma" else "> 0"), value)
    ranges = [_range(*p) for p in (("distance_range", distance_range), ("charge_range", charge_range),
                                   ("frequency_range", frequency_range))]
    rng = np.random.default_rng(seed)
    d, w, f = (10.0 ** rng.uniform(np.log10(lo), np.log10(hi), n) for lo, hi in ranges)
    ppv = k * (d / w ** _EXPONENT[law]) ** -beta * 10.0 ** (sigma * rng.standard_normal(n))
    return pd.DataFrame({"distance_m": d, "charge_kg": w, "ppv_mm_s": ppv, "ppv_kind": "component_max",
                         "frequency_hz": f})


_GEOPHONES = ("transverse", "vertical", "longitudinal")
_STEP_GEOPHONE_MM_S, _STEP_MIC_PA = 0.00788, 0.0156  # resolución de los sensores (ver io.infer_units)


def _positive_finite(name: str, value, *, zero_ok: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value)
            or value < 0 or (value == 0 and not zero_ok)):
        raise _bad(name, "must be a finite number " + (">= 0" if zero_ok else "> 0"), value)
    return float(value)


def _burst(t: np.ndarray, tau: float, f: float, zeta: float, phase: float) -> np.ndarray:
    """Damped sinusoid that starts at ``tau`` (zero before)."""
    s = np.maximum(t - tau, 0.0)
    return np.where(t >= tau, np.exp(-zeta * 2 * np.pi * f * s) * np.sin(2 * np.pi * f * s + phase), 0.0)


def synthetic_record(*, seed: int = 0, sample_rate_hz: float = 1024.0, duration_s: float = 2.0,
                     pretrigger_s: float = 0.25, holes: int = 1, delay_ms: float = 25.0,
                     frequency_hz: float = 20.0, ppv_mm_s: float = 10.0, damping: float = 0.15,
                     mic_pa: float | None = 50.0, noise_mm_s: float = 0.0, quantize: bool = False) -> io.Record:
    """Seeded synthetic seismograph record: one damped sinusoid per blasthole on each channel.

    Hole ``k`` starts ``k * delay_ms`` after the trigger (``t = 0``) and contributes
    ``exp(-damping * 2 pi f (t - tau_k)) * sin(2 pi f (t - tau_k) + phase)``. Each geophone has its own phase
    (between -30 and 0 degrees, so the peak is a whole lobe, not one cut by the onset), drawn from ``seed``. Each
    channel is scaled so that its largest ``|x|`` is exactly ``ppv_mm_s`` (``mic_pa`` for the microphone, which is
    left out with ``mic_pa=None``). ``noise_mm_s`` adds Gaussian noise in mm/s to the scaled geophones only (never to
    the microphone); each geophone is then rescaled so that its largest ``|x|`` is still exactly ``ppv_mm_s``, so the
    noise standard deviation is approximately ``noise_mm_s``. With ``quantize=True`` the values are rounded to
    multiples of 0.00788 mm/s and 0.0156 Pa, the sensor resolutions, so the peak is within half a step of the target.
    The same ``seed`` gives the same record. Returns a :class:`blastingpy.io.Record` (``source="arrays"``).
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise _bad("seed", "must be an integer", seed)
    if isinstance(holes, bool) or not isinstance(holes, (int, np.integer)) or holes < 1:
        raise _bad("holes", "must be an integer >= 1", holes)
    if not isinstance(quantize, (bool, np.bool_)):
        raise _bad("quantize", "must be a bool", quantize)
    fs = _positive_finite("sample_rate_hz", sample_rate_hz)
    dur = _positive_finite("duration_s", duration_s)
    pre = _positive_finite("pretrigger_s", pretrigger_s, zero_ok=True)
    delay = _positive_finite("delay_ms", delay_ms)
    f = _positive_finite("frequency_hz", frequency_hz)
    ppv = _positive_finite("ppv_mm_s", ppv_mm_s)
    zeta = _positive_finite("damping", damping, zero_ok=True)
    noise = _positive_finite("noise_mm_s", noise_mm_s, zero_ok=True)
    mic = None if mic_pa is None else _positive_finite("mic_pa", mic_pa)
    if dur <= pre + (holes - 1) * delay / 1000.0:
        raise _bad("duration_s", "must exceed pretrigger_s + (holes - 1) * delay_ms / 1000", duration_s)
    n = round(dur * fs)
    if n < 2:
        raise _bad("duration_s", "must span at least 2 samples at sample_rate_hz", duration_s)
    rng = np.random.default_rng(seed)
    # Fases en (-pi/6, 0): con fase positiva el pico cae en el primer lóbulo, truncado por el arranque, y el cruce por
    # cero deja de medir la frecuencia; con fase negativa el primer lóbulo es pequeño y el pico es un lóbulo completo.
    # Tres geófonos y el micrófono.
    phases = rng.uniform(-np.pi / 6, 0.0, len(_GEOPHONES) + 1)
    t = np.arange(n) / fs - pre
    channels = {}
    for name, target, phase, step in (*((c, ppv, p, _STEP_GEOPHONE_MM_S) for c, p in zip(_GEOPHONES, phases)),
                                      ("mic", mic, phases[-1], _STEP_MIC_PA)):
        if target is None:
            continue
        x = sum(_burst(t, k * delay / 1000.0, f, zeta, phase) for k in range(int(holes)))
        x = x * (target / np.abs(x).max())
        if noise > 0 and name != "mic":  # ruido en mm/s, solo en los geófonos ya escalados (R19)
            x = x + noise * rng.standard_normal(n)
            x = x * (target / np.abs(x).max())  # el pico vuelve a ser exactamente ppv_mm_s
        channels[name] = np.round(x / step) * step if quantize else x
    return io.from_arrays(channels, fs, units={"geophone": "mm/s", "mic": "Pa"}, pretrigger_s=pre,
                          metadata={"generator": "synthetic_record", "seed": str(seed)})
