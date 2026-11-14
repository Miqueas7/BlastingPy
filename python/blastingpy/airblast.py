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
import math
import warnings

import numpy as np
import pandas as pd

from ._blastingpy_core import _airblast as _ab
from ._blastingpy_core import _attenuation as _core
from ._blastingpy_core import MethodInfo, Verification, _units
from .units import _apply

SiteLaw = _core.SiteLaw
Law = _core.Law
Level = _core.Level
Prediction = _core.Prediction

__all__ = ["to_db", "from_db", "fit", "make_site_law", "predict", "max_charge", "distance_for", "PublishedEquation",
           "list_equations", "published", "predict_published", "SiteLaw", "Law", "Level", "Prediction",
           "SYSTEMS", "Limit", "list_standards", "system_from_cutoff", "limit_db", "evaluate", "evaluate_prediction"]

_COLUMNS = ("distance_m", "charge_kg", "airblast_pa")
_FT_LB = "psi; ft/lb^(1/3)"
_M_KG = "kPa; m/kg^(1/3)"
_VERIFICATION = {"V1": Verification.Primary, "V2": Verification.Secondary}
_TOL = 1e-9  # dB: a level equal to the limit complies
_USER_ID = "AIR-CBRT-USER"


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
    """Call the core and name the airblast limit in its domain errors (the core's inverses speak of PPV)."""
    try:
        return func(*args)
    except ValueError as e:
        message = str(e)
        for old, new in renames.items():
            message = message.replace(f"parameter '{old}'", f"parameter '{new}'")
        raise ValueError(message) from None


def fit(events: pd.DataFrame, *, z: float = 2.0, outlier_z: float = 3.0) -> SiteLaw:
    """Fit ``P = K (D / W**(1/3))**-beta`` to ``events`` (columns ``distance_m``, ``charge_kg``, ``airblast_pa``).

    Rows whose ``airblast_pa`` is not finite (no microphone) are skipped with a ``UserWarning``; then ``outliers`` and
    ``residuals`` refer to the rows used, ``events[np.isfinite(events["airblast_pa"])]`` (for example
    ``used.index[law.outliers]``). K is in Pa-m-kg^(1/3); ``z`` and ``outlier_z`` work as in
    :func:`blastingpy.attenuation.fit`. ``info`` cites USBM RI 8485 (Table B-1).
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
        skipped = int((~used).sum())
        what = "1 event without a finite airblast_pa was" if skipped == 1 else             f"{skipped} events without a finite airblast_pa were"
        warnings.warn(f"{what} skipped; outliers and residuals refer to the {int(used.sum())} rows used: "
                      "events[np.isfinite(events['airblast_pa'])]", UserWarning, stacklevel=2)
    d, w = (events[c].to_numpy(dtype=float)[used].tolist() for c in _COLUMNS[:2])
    return _core.fit_site_law(d, w, p[used].tolist(), Law.CubeRoot, z, outlier_z, _core.Quantity.Airblast)


def make_site_law(k_pa: float, beta: float, *, sigma: float = 0.0, z: float = 2.0) -> SiteLaw:
    """Hand-made airblast law ``P = K (D / W**(1/3))**-beta`` (K in Pa-m-kg^(1/3)); user defined, cites RI 8485's form.

    ``sigma`` (log10 scatter) and ``z`` work as in :func:`blastingpy.attenuation.make_site_law`: with ``sigma == 0``
    the upper bound is the median and predictions at it warn that there is no safety margin.
    """
    info = MethodInfo(_USER_ID, Verification.UserDefined,
                      "User-defined airblast law; form P = K (D / W^(1/3))^-beta of USBM RI 8485 (Table B-1)")
    return _core.make_site_law(Law.CubeRoot, k_pa, beta, sigma=sigma, z=z, info=info)


def _airblast_law(law: SiteLaw) -> SiteLaw:
    """Refuse a ground-vibration law: its K is in mm/s, not Pa (airblast ids: fit, make_site_law and the catalog)."""
    mid = getattr(getattr(law, "info", None), "method_id", None)
    if not (isinstance(mid, str) and (mid.startswith("AIR-") or mid in _catalog())):
        raise _error("law", f"must be an airblast law (from airblast.fit, airblast.make_site_law or "
                            f"airblast.published); this one is not. Received method id: {mid!r}")
    return law


def predict(law: SiteLaw, distance_m: float, charge_kg: float, level: Level = Level.Upper) -> Prediction:
    """Peak overpressure (Pa) at ``distance_m`` (m) with ``charge_kg`` (kg per delay)."""
    return _core.predict_ppv(_airblast_law(law), distance_m, charge_kg, level)


def max_charge(law: SiteLaw, distance_m: float, limit_pa: float, level: Level = Level.Upper) -> Prediction:
    """Maximum charge per delay (kg) at ``distance_m`` (m) that keeps the overpressure at or below ``limit_pa``."""
    return _core_call(_core.max_charge, _airblast_law(law), distance_m, limit_pa, level, ppv_limit_mm_s="limit_pa")


def distance_for(law: SiteLaw, charge_kg: float, limit_pa: float, level: Level = Level.Upper) -> Prediction:
    """Minimum distance (m) at which ``charge_kg`` per delay keeps the overpressure at or below ``limit_pa``."""
    return _core_call(_core.distance_for_ppv, _airblast_law(law), charge_kg, limit_pa, level,
                      ppv_limit_mm_s="limit_pa")


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
    info = MethodInfo(raw["id"], _VERIFICATION[raw["verification"]], raw["citation"])
    return PublishedEquation(
        id=raw["id"], law=_core.make_site_law(Law.CubeRoot, k_si, beta, info=info), system=raw["system"],
        k_original=float(k),
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
    """Median overpressure (Pa) of a published equation at ``distance_m`` (m) with ``charge_kg`` (kg).

    For a linear system the value is the peak overpressure; for the ``c-slow`` equations it is the source's C-slow
    level expressed as a pressure (the source prints them in psi), not a linear peak. ``warnings`` always says that
    this is a median (about half of the blasts exceed it; with the published standard error when there is one) and,
    for a C-slow equation, what the value is; it says when D / W^(1/3) is outside the equation's scaled-distance
    range (approximate, read from a figure), or that the range is not published, and repeats the equation's notes.
    """
    eq = published(equation_id)
    p = _core.predict_ppv(eq.law, distance_m, charge_kg, Level.Median)
    sd = distance_m / charge_kg ** (1 / 3)
    out = list(p.warnings)
    se = f" (published standard error: {eq.se_db:g} dB)" if eq.se_db is not None else ""
    out.append(f"median of the published equation: about half of the blasts exceed it; no safety margin{se}")
    if eq.system == _SLOW:
        out.append("this equation predicts a C-weighted slow level, expressed as a pressure; it is not a linear peak")
    if eq.scaled_distance_range_si is None:
        out.append("scaled-distance range not published for this equation: extrapolation cannot be checked")
    else:
        lo, hi = eq.scaled_distance_range_si
        if not lo <= sd <= hi:
            out.append(f"scaled distance {sd:.4g} m/kg^(1/3) is outside the equation's range [{lo:.4g}, {hi:.4g}] "
                       f"m/kg^(1/3) (approximate, read from a figure): extrapolation")
    return Prediction(p.value, out + list(eq.notes))


# --- Limits and verdict -------------------------------------------------------------------------------------------

SYSTEMS = ("0.1Hz", "2Hz", "6Hz", "c-slow")
_LINEAR = SYSTEMS[:3]
_SLOW = SYSTEMS[3]


@dataclasses.dataclass(frozen=True)
class Limit:
    """An airblast limit for one measuring system: a linear peak level in dB re 20 uPa (or C-slow)."""

    standard: str  # "USBM-RI8485-1980" | "OSMRE-30CFR816.67"
    system: str
    level_db: float
    pressure_pa: float | None  # None en "c-slow": no es un pico lineal
    wording: str  # cómo nombra la fuente al sistema
    notes: tuple[str, ...]
    citation: str
    verification: str


def _error(name: str, detail: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {detail}")


@functools.cache
def _limits() -> dict:
    text = importlib.resources.files("blastingpy").joinpath("airblast_data", "limits.json").read_text("utf-8")
    out = {}
    for r in json.loads(text)["limits"]:
        linear = r["system"] in _LINEAR
        out[r["standard"], r["system"]] = Limit(
            r["standard"], r["system"], float(r["level_db"]), float(from_db(r["level_db"])) if linear else None,
            r["wording"], tuple(r["notes"]), r["citation"], r["verification"])
    return out


def list_standards() -> list[str]:
    """Ids of the standards with airblast limits, in file order."""
    return list(dict.fromkeys(std for std, _ in _limits()))


def _five_to_six(system):
    return "6Hz" if system == "5Hz" else system  # RI 8485: "5- or 6-Hz"; CFR: "6 Hz or lower"


def _system(system) -> str:
    system = _five_to_six(system)
    if not isinstance(system, str) or system not in SYSTEMS:
        raise _error("system", f"is required (the measuring system of the microphone) and must be one of "
                               f"{list(SYSTEMS)}. Received: {system!r}")
    return system


def system_from_cutoff(low_cutoff_hz: float) -> str:
    """Measuring system of a linear microphone from its lower -3 dB cutoff: <= 0.1 Hz, <= 2 Hz or <= 6 Hz.

    A cutoff above 6 Hz has no limit (inferred from "or lower" in the sources) and is an error.
    """
    f = low_cutoff_hz
    if not (isinstance(f, (int, float, np.number)) and math.isfinite(f) and f > 0):
        raise _error("low_cutoff_hz", f"must be finite and > 0. Received: {f!r}")
    if f > 6:
        raise _error("low_cutoff_hz", f"is above 6 Hz: no airblast limit applies to a linear system that cuts "
                                      f"off there. Received: {f!r}")
    return "0.1Hz" if f <= 0.1 else "2Hz" if f <= 2 else "6Hz"


def limit_db(standard: str, *, system: str) -> Limit:
    """Limit of ``standard`` (see :func:`list_standards`) for the measuring ``system`` (one of :data:`SYSTEMS`).

    ``system`` is required: a level in dB means nothing without it. ``"5Hz"`` is read as ``"6Hz"``.
    """
    system = _system(system)
    try:
        return _limits()[standard, system]
    except (KeyError, TypeError):
        raise _error("standard", f"must be one of {list_standards()}. Received: {standard!r}") from None


def evaluate(events: pd.DataFrame, standard: str, *, system: str) -> pd.DataFrame:
    """Verdict of each event's ``airblast_pa`` against ``standard`` for the measuring ``system``.

    Returns a copy of ``events`` plus ``airblast_db``, ``limit_db``, ``margin_db`` (limit - measured), ``status``
    (``complies``, ``exceeds``, ``indeterminate`` or ``not_comparable``) and ``reason``, with the meanings of
    :func:`blastingpy.criteria.evaluate`: a missing (NaN) pressure is ``not_comparable`` and a pressure outside the
    domain (<= 0 or infinite) is ``indeterminate``; with ``system="c-slow"`` every row is ``not_comparable`` (a
    linear peak is not a C-slow level). ``attrs`` keeps the frame's own (``warnings``, ``notes``, ``format_evidence``) and adds the
    standard, system, limit, citation, verification and the limit's notes (``limit_notes``).
    """
    lim = limit_db(standard, system=system)
    if "airblast_pa" not in events.columns:
        raise _error("events", "is missing the column 'airblast_pa'")
    out = events.copy()
    p = out["airblast_pa"].to_numpy(dtype=float)
    ok = np.isfinite(p) & (p > 0)
    db = np.full(p.shape, np.nan)
    db[ok] = to_db(p[ok])
    out["airblast_db"] = db
    out["limit_db"] = lim.level_db
    out["margin_db"] = lim.level_db - db
    status = np.where(db > lim.level_db + _TOL, "exceeds", "complies").astype(object)
    reason = np.full(p.shape, "", dtype=object)
    missing = np.isnan(p)
    status[~ok] = "indeterminate"
    reason[~ok] = "airblast_pa is outside the domain (<= 0 or infinite)"
    status[missing] = "not_comparable"
    reason[missing] = "no airblast value (airblast_pa is missing)"
    if lim.pressure_pa is None:
        status[:] = "not_comparable"
        reason[:] = "a linear peak pressure is not a C-slow level: it cannot be compared with this limit"
        out["margin_db"] = np.nan
    out["status"], out["reason"] = status, reason
    out.attrs.update({"standard": lim.standard, "system": lim.system, "limit_db": lim.level_db,
                      "citation": lim.citation, "verification": lim.verification, "limit_notes": lim.notes})
    return out


def evaluate_prediction(equation_id: str, distance_m: float, charge_kg: float, standard: str, *, system: str) -> dict:
    """Median level of a published equation at ``distance_m`` / ``charge_kg`` against the limit of ``system``.

    Returns ``level_db``, ``limit_db``, ``margin_db``, ``status``, ``equation_system``, ``limit_system``, ``level``
    (always ``"median"``), ``reason`` and ``warnings`` (the prediction's, which says it is a median, the limit's notes,
    and one when the equation's measuring system differs from the limit's or is not stated). The status is
    ``not_comparable`` (margin None, ``reason`` says why) when exactly one of the equation and the limit is C-slow,
    when the equation's system is not stated and the limit is C-slow, and when both are: the source prints the C-slow
    equations in psi without saying how that pressure maps to the dBC level of the limit. ``level_db`` is then
    20 log10(P / P0) of the predicted value, for the user to judge.
    """
    lim = limit_db(standard, system=system)
    pred = predict_published(equation_id, distance_m, charge_kg)
    eq_sys = _five_to_six(published(equation_id).system)
    level = float(to_db(pred.value))
    warns = list(pred.warnings) + list(lim.notes)
    if eq_sys != lim.system:
        warns.append(f"the equation's measuring system ({eq_sys or 'not stated by the source'}) differs from the "
                     f"limit's ({lim.system}): the comparison mixes measuring systems")
    eq_slow, lim_slow = eq_sys == _SLOW, lim.system == _SLOW
    reason = ""
    if eq_slow and lim_slow:
        reason = ("the source prints the C-slow equations in psi and does not say how that pressure maps to the dBC "
                  "level of the limit")
    elif eq_slow:
        reason = "the equation predicts a C-slow level and the limit is a linear peak: different quantities"
    elif lim_slow:
        reason = "the equation predicts a linear peak (or its system is not stated) and the limit is a C-slow level"
    if reason:
        status, margin = "not_comparable", None
    else:
        margin = lim.level_db - level
        status = "exceeds" if level > lim.level_db + _TOL else "complies"
    return {"level_db": level, "limit_db": lim.level_db, "margin_db": margin, "status": status,
            "equation_system": eq_sys, "limit_system": lim.system, "warnings": warns, "level": "median",
            "reason": reason}
