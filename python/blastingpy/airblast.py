"""Airblast (overpressure): the site law fitted to peak pressure, and the catalog of published equations.

Units are SI: distance in m, charge per delay in kg, peak linear overpressure in Pa (see :mod:`blastingpy.units`
for psi and kPa). Levels in dB are re 20 uPa (USBM RI 8485, p. 8).

* The site law is the regression of :mod:`blastingpy.attenuation` with the cube-root law, ``P = K (D / W**(1/3))**-beta``,
  applied to ``airblast_pa`` instead of the PPV: K is in Pa-m-kg^(1/3). RI 8485 (Table B-1) fits airblast with this form.
* Published equations (:func:`list_equations`, :func:`published`) are medians (best fit), stored in their source's units
  in ``airblast_data/equations.json`` and converted to SI once. Their scaled-distance ranges are read from figures and
  approximate. :func:`predict_published` warns when the scaled distance is outside that range (or the range is not
  published) and repeats the equation's notes.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.resources
import json
import warnings

import numpy as np
import pandas as pd

from ._blastingpy_core import _airblast as _ab
from ._blastingpy_core import _attenuation as _core
from ._blastingpy_core import _units
from .units import _apply

SiteLaw = _core.SiteLaw
Law = _core.Law
Level = _core.Level
Prediction = _core.Prediction

__all__ = ["to_db", "from_db", "fit", "predict", "max_charge", "distance_for", "PublishedEquation",
           "list_equations", "published", "predict_published", "SiteLaw", "Law", "Level", "Prediction"]

_COLUMNS = ("distance_m", "charge_kg", "airblast_pa")
_FT_LB = "psi; ft/lb^(1/3)"
_M_KG = "kPa; m/kg^(1/3)"


def _vectorized(func, values):
    # NaN pasa como NaN (celda sin micrófono); el resto lo valida el núcleo.
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    out[ok] = [func(v) for v in x[ok]]
    return _apply(out)


def to_db(pressure_pa):
    """Level in dB re 20 uPa of ``pressure_pa`` (float or numpy array). NaN gives NaN; a value <= 0 is an error."""
    return _vectorized(_ab.pa_to_db, pressure_pa)


def from_db(level_db):
    """Pressure in Pa of ``level_db`` (dB re 20 uPa; float or numpy array). NaN gives NaN."""
    return _vectorized(_ab.db_to_pa, level_db)


def _core_call(func, *args, **renames):
    """Call the core and name the airblast parameter in its domain errors (the core speaks of PPV)."""
    try:
        return func(*args)
    except ValueError as e:
        message = str(e)
        for old, new in renames.items():
            message = message.replace(f"parameter '{old}'", f"parameter '{new}'")
        raise ValueError(message) from None


def fit(events: pd.DataFrame, *, z: float = 2.0, outlier_z: float = 3.0) -> SiteLaw:
    """Fit ``P = K (D / W**(1/3))**-beta`` to ``events`` (columns ``distance_m``, ``charge_kg``, ``airblast_pa``).

    Rows whose ``airblast_pa`` is not finite (no microphone) are skipped with a ``UserWarning``. K is in Pa-m-kg^(1/3);
    ``z`` and ``outlier_z`` work as in :func:`blastingpy.attenuation.fit`. The law's ``info`` and ``warnings`` come from
    the shared regression (they cite the cube-root law of ground vibration).
    """
    missing = [c for c in _COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(
            f"Domain error [BlastingPy]: parameter 'events' is missing the column {missing[0]!r}. "
            f"Expected columns: {list(_COLUMNS)}"
        )
    p = events["airblast_pa"].to_numpy(dtype=float)
    used = np.isfinite(p)
    if not used.all():
        warnings.warn(f"{int((~used).sum())} events without a finite airblast_pa were skipped", UserWarning,
                      stacklevel=2)
    d, w = (events[c].to_numpy(dtype=float)[used].tolist() for c in _COLUMNS[:2])
    return _core_call(_core.fit_site_law, d, w, p[used].tolist(), Law.CubeRoot, z, outlier_z, ppv_mm_s="airblast_pa")


def predict(law: SiteLaw, distance_m: float, charge_kg: float, level: Level = Level.Upper) -> Prediction:
    """Peak overpressure (Pa) at ``distance_m`` (m) with ``charge_kg`` (kg per delay)."""
    return _core.predict_ppv(law, distance_m, charge_kg, level)


def max_charge(law: SiteLaw, distance_m: float, limit_pa: float, level: Level = Level.Upper) -> Prediction:
    """Maximum charge per delay (kg) at ``distance_m`` (m) that keeps the overpressure at or below ``limit_pa``."""
    return _core_call(_core.max_charge, law, distance_m, limit_pa, level, ppv_limit_mm_s="limit_pa")


def distance_for(law: SiteLaw, charge_kg: float, limit_pa: float, level: Level = Level.Upper) -> Prediction:
    """Minimum distance (m) at which ``charge_kg`` per delay keeps the overpressure at or below ``limit_pa``."""
    return _core_call(_core.distance_for_ppv, law, charge_kg, limit_pa, level, ppv_limit_mm_s="limit_pa")


@dataclasses.dataclass(frozen=True)
class PublishedEquation:
    """A published airblast equation: the law in SI (median, sigma 0) and what the source prints about it."""

    id: str
    law: SiteLaw  # Pa, m, kg^(1/3); mediana
    system: str | None  # "0.1Hz", "2Hz", "5Hz", "c-slow"; None si la fuente no lo dice
    k_original: float
    beta: float
    units_original: str  # "psi; ft/lb^(1/3)" o "kPa; m/kg^(1/3)"
    n: int | None
    r: float | None
    se_db: float | None  # error estándar impreso, en dB; None si es errata o no hay
    scaled_distance_range_si: tuple[float, float] | None  # m/kg^(1/3), aproximado, leído de figura
    verification: str
    citation: str
    notes: tuple[str, ...]


def _equation(raw: dict) -> PublishedEquation:
    k, beta, units = raw["k"], raw["beta"], raw["units"]
    if units == _FT_LB:
        k_si, sd_factor = _ab.convert_k_to_si(k, beta), _units.m_per_ft / _units.kg_per_lb ** (1 / 3)
    elif units == _M_KG:
        k_si, sd_factor = k * 1000.0, 1.0
    else:
        raise ValueError(f"Domain error [BlastingPy]: parameter 'units' of {raw['id']!r} must be {_FT_LB!r} or "
                         f"{_M_KG!r}. Received: {units!r}")
    rng = raw["scaled_distance_range"]
    return PublishedEquation(
        id=raw["id"], law=_core.make_site_law(Law.CubeRoot, k_si, beta), system=raw["system"], k_original=k,
        beta=beta, units_original=units, n=raw["n"], r=raw["r"], se_db=raw["se_db"],
        scaled_distance_range_si=None if rng is None else (rng[0] * sd_factor, rng[1] * sd_factor),
        verification=raw["verification"], citation=raw["citation"], notes=tuple(raw["notes"]))


@functools.cache
def _catalog() -> dict:
    text = importlib.resources.files("blastingpy").joinpath("airblast_data", "equations.json").read_text("utf-8")
    return {raw["id"]: _equation(raw) for raw in json.loads(text)["equations"]}


def list_equations() -> list[str]:
    """Ids of the published equations, in catalog order (RI 8485 Table B-1, BLEP, AS 2187.2)."""
    return list(_catalog())


def published(equation_id: str) -> PublishedEquation:
    """The published equation ``equation_id`` (see :func:`list_equations`)."""
    try:
        return _catalog()[equation_id]
    except (KeyError, TypeError):
        raise ValueError(f"Domain error [BlastingPy]: parameter 'equation_id' must be one of {list_equations()}. "
                         f"Received: {equation_id!r}") from None


def predict_published(equation_id: str, distance_m: float, charge_kg: float) -> Prediction:
    """Median peak overpressure (Pa) of a published equation at ``distance_m`` (m) with ``charge_kg`` (kg).

    ``warnings`` says when D / W^(1/3) is outside the equation's scaled-distance range (approximate, read from a
    figure), or that the range is not published, and repeats the equation's notes.
    """
    eq = published(equation_id)
    p = _core.predict_ppv(eq.law, distance_m, charge_kg, Level.Median)
    sd = distance_m / charge_kg ** (1 / 3)
    out = list(p.warnings)
    if eq.scaled_distance_range_si is None:
        out.append("scaled-distance range not published for this equation: extrapolation cannot be checked")
    else:
        lo, hi = eq.scaled_distance_range_si
        if not lo <= sd <= hi:
            out.append(f"scaled distance {sd:.4g} m/kg^(1/3) is outside the equation's range [{lo:.4g}, {hi:.4g}] "
                       f"m/kg^(1/3) (approximate, read from a figure): extrapolation")
    return Prediction(p.value, out + list(eq.notes))
