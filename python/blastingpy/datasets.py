"""Seeded synthetic blast events, for examples and tests. The repository never contains real blast data."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["synthetic_events"]

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
