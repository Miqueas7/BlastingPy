"""Site attenuation law: fit it to blast monitoring events, then predict PPV, maximum charge and distance.

Units are SI: distance in m, charge per delay in kg, PPV in mm/s (see :mod:`blastingpy.units` to convert).

Conventions
-----------
* All logarithms are base-10. The residual of an event is ``log10(observed PPV) - log10(median predicted PPV)`` and
  ``sigma = sqrt(SSE / dof)`` is computed on those residuals (``dof = n - 2`` for the scaled-distance laws,
  ``n - 3`` for ``Law.General``).
* The upper bound is ``k_upper = k * 10**(z * sigma)`` with ``z = 2`` by default (USBM RI 8507, p. 16, and OSMRE
  BIVDEP practice). Predictions and ``max_charge`` use the upper bound unless ``Level.Median`` is requested.
* ``coverage`` is ``Phi(z)``, the standard normal CDF at ``z``. It assumes sigma is known: it does not include the
  uncertainty of the fitted parameters nor a Student-t correction, so with few events the real coverage is lower.
* ``normal_quantile(0.95)`` gives ``z = 1.645``, the one-sided 95 % bound: pass it as ``fit(..., z=normal_quantile(0.95))``.
* A law with ``beta <= 0`` (or ``a <= 0`` in ``Law.General``) can be fitted, with a warning, but ``predict_ppv``,
  ``max_charge`` and ``distance_for_ppv`` refuse it with a ``ValueError``.
* Predictions return a :class:`Prediction` whose ``warnings`` list says when an input, or the computed charge or
  distance, is outside the range of the data the law was fitted on (extrapolation).
* A law predicts the PPV quantity it was fitted on (``ppv_kind``); :func:`fit` refuses events that mix kinds.
"""

from __future__ import annotations

import pandas as pd

from ._blastingpy_core import _attenuation as _core

SiteLaw = _core.SiteLaw
Law = _core.Law
Level = _core.Level
Prediction = _core.Prediction
make_site_law = _core.make_site_law
normal_quantile = _core.normal_quantile
predict_ppv = _core.predict_ppv
max_charge = _core.max_charge
distance_for_ppv = _core.distance_for_ppv

__all__ = [
    "fit",
    "SiteLaw",
    "Law",
    "Level",
    "Prediction",
    "make_site_law",
    "normal_quantile",
    "predict_ppv",
    "max_charge",
    "distance_for_ppv",
]

_LAWS = {"sqrt": Law.SquareRoot, "cbrt": Law.CubeRoot, "general": Law.General}
_COLUMNS = ("distance_m", "charge_kg", "ppv_mm_s")


def _law(law) -> Law:
    if isinstance(law, Law):
        return law
    try:
        return _LAWS[law]
    except (KeyError, TypeError):
        raise ValueError(
            f"Domain error [BlastingPy]: parameter 'law' must be one of {sorted(_LAWS)} or an attenuation.Law. "
            f"Received: {law!r}"
        ) from None


def fit(events: pd.DataFrame, law: str | Law = "sqrt", z: float = 2.0, outlier_z: float = 3.0) -> SiteLaw:
    """Fit the site law to the events in ``events`` (columns ``distance_m``, ``charge_kg``, ``ppv_mm_s``).

    ``law`` is ``"sqrt"`` (``v = K (D / W**0.5)**-beta``), ``"cbrt"`` (``W**(1/3)``) or ``"general"``
    (``v = K W**a D**-beta``); an :class:`attenuation.Law` value is also accepted. ``z`` sets the upper bound
    ``k * 10**(z * sigma)``; events with ``|residual| > outlier_z * sigma`` are listed in ``outliers`` but kept.

    The law predicts the quantity it was fitted on (the ``ppv_kind`` of the events: peak component, vector sum or
    vertical), so compare its predictions only with limits of that quantity. A ``ppv_kind`` column with more than one
    kind is refused: fit one law per kind.
    """
    law = _law(law)
    missing = [c for c in _COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(
            f"Domain error [BlastingPy]: parameter 'events' is missing the column {missing[0]!r}. "
            f"Expected columns: {list(_COLUMNS)}"
        )
    if "ppv_kind" in events.columns:
        kinds = sorted(str(k) for k in events["ppv_kind"].dropna().unique())
        if len(kinds) > 1:
            raise ValueError(
                f"Domain error [BlastingPy]: parameter 'events' mixes the PPV kinds {kinds} in 'ppv_kind'; "
                f"a law predicts the quantity it was fitted on, so fit one law per kind"
            )
    d, w, v = (events[c].to_numpy(dtype=float).tolist() for c in _COLUMNS)
    return _core.fit_site_law(d, w, v, law, z, outlier_z)
