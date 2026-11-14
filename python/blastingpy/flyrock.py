"""Flyrock (rock projection): ballistics, the published range models, a per-hole comparison and the clearance distance.

Units are SI: lengths in m, linear charge in kg/m, velocity in m/s, angles in degrees (hole angles from the vertical,
launch angles from the horizontal), hole diameter in mm, powder factor in kg/m3, SDoB in m/kg^(1/3).

* The core functions are re-exported with their names (:func:`rm_face_burst`, :func:`lundborg_max_range`, ...).
* :func:`compare` puts the Richards & Moore, Lundborg and McKenzie ranges of each hole side by side. They are different
  models, not a consensus: nothing here averages them or picks one.
* :func:`clearance` turns a range into a distance with the safety factors of Richards & Moore and, optionally, the
  Peruvian regulatory minimum. The models carry no safety margin of their own.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from ._blastingpy_core import _flyrock as _fr
from ._blastingpy_core import Verification

range_level = _fr.range_level
max_range_level = _fr.max_range_level
range_to_height = _fr.range_to_height
max_height = _fr.max_height
velocity_for_range = _fr.velocity_for_range
rm_launch_velocity = _fr.rm_launch_velocity
rm_face_burst = _fr.rm_face_burst
rm_cratering = _fr.rm_cratering
rm_stemming_ejection = _fr.rm_stemming_ejection
rm_min_confinement = _fr.rm_min_confinement
lundborg_max_range = _fr.lundborg_max_range
lundborg_optimal_fragment = _fr.lundborg_optimal_fragment
lundborg_range = _fr.lundborg_range
mckenzie_max_range = _fr.mckenzie_max_range
roth_charge_mass_ratio = _fr.roth_charge_mass_ratio
roth_gurney_velocity = _fr.roth_gurney_velocity
roth_launch_velocity = _fr.roth_launch_velocity
roth_calibrated_velocity = _fr.roth_calibrated_velocity
roth_reference_gurney_velocity = _fr.roth_reference_gurney_velocity
Explosive = _fr.Explosive
RothCalibration = _fr.RothCalibration
Model = _fr.Model
info = _fr.info
standard_gravity = _fr.standard_gravity

HOLE_COLUMNS = ("diameter_mm", "burden_m", "stemming_m", "linear_charge_kg_m")
REGULATIONS = ("PE-DS024-2016-EM",)

__all__ = ["range_level", "max_range_level", "range_to_height", "max_height", "velocity_for_range",
           "rm_launch_velocity", "rm_face_burst", "rm_cratering", "rm_stemming_ejection", "rm_min_confinement",
           "lundborg_max_range", "lundborg_optimal_fragment", "lundborg_range", "mckenzie_max_range",
           "roth_charge_mass_ratio", "roth_gurney_velocity", "roth_launch_velocity", "roth_calibrated_velocity",
           "roth_reference_gurney_velocity", "Explosive", "RothCalibration", "Model", "info", "standard_gravity",
           "HOLE_COLUMNS", "REGULATIONS", "linear_charge", "compare", "Clearance", "clearance"]

_MARK = {Verification.Primary: "V1", Verification.Secondary: "V2"}
_REG_MINIMUM_M = 500.0
_REG_CITATION = "DS 024-2016-EM, art. 267 a), modificado por DS 034-2023-EM (El Peruano, 30-12-2023)"
_RM_CITATION = ("Richards & Moore, Flyrock Control - By Chance or Design, 30th ISEE (p. 31); "
                "Moore & Richards (2005), p. 14")
_FACTORS = {"people": 4.0, "equipment": 2.0}
_NO_MARGIN = ("the models carry no safety margin of their own: with the real burden and stemming the predicted range "
              "can be exceeded (Moore & Richards 2005, p. 5)")
_CRITIQUES = (
    "Models without drag (ballistics, Richards & Moore) can overestimate the range by factors of 2 to 8 "
    "(Szendrei & Tose 2022, JSAIMM 122(12), pp. 727, 731).",
    "Lundborg's ranges hold only for spherical rock of the optimal size; other shapes fly shorter, and the authors "
    "consider the model an overestimate (Szendrei & Tose 2022, JSAIMM 122(12), pp. 726, 729, 731).",
    "McKenzie calibrated his velocities against Lundborg's, so the model inherits that problem; in his own validation "
    "the measured ranges averaged 4.8 times below the calculated ones (Szendrei & Tose 2022, JSAIMM 122(12), p. 727).",
    "No model here has a safety margin: with the real burden and stemming the range can be exceeded "
    "(Moore & Richards 2005, p. 5); apply a factor with clearance().",
)


def _error(name: str, detail: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {detail}")


def linear_charge(diameter_mm: float, explosive_density_kg_m3: float) -> float:
    """Linear charge of a full hole in kg/m: density * pi * d^2 / 4 (geometry; no coupling or decking)."""
    for name, v in (("diameter_mm", diameter_mm), ("explosive_density_kg_m3", explosive_density_kg_m3)):
        if not (isinstance(v, (int, float, np.number)) and math.isfinite(v) and v > 0):
            raise _error(name, f"must be finite and > 0. Received: {v!r}")
    return explosive_density_kg_m3 * math.pi * (diameter_mm / 1000.0) ** 2 / 4.0


def _column(df: pd.DataFrame, name: str, *, allow_zero: bool = False) -> np.ndarray:
    """Column as floats; finite and > 0 (>= 0 with ``allow_zero``), else an error naming column and row."""
    try:
        x = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise _error("holes", f"column {name!r} must be numeric") from None
    for i, v in enumerate(x):
        if not (math.isfinite(v) and (v >= 0 if allow_zero else v > 0)):
            need = "finite and >= 0" if allow_zero else "finite and > 0"
            raise _error("holes", f"column {name!r}, row {df.index[i]!r}: must be {need}. Received: {v!r}")
    return x


def _per_row(df: pd.DataFrame, column: str, func, *arrays) -> np.ndarray:
    """func over the rows; a core domain error gets the column and row appended."""
    out = np.empty(len(df))
    for i, args in enumerate(zip(*arrays)):
        try:
            out[i] = func(*args)
        except ValueError as e:
            raise ValueError(f"{e} (column {column!r}, row {df.index[i]!r})") from None
    return out


def _method(model) -> dict:
    m = info(model)
    return {"id": m.method_id, "verification": _MARK[m.verification], "citation": m.source}


def compare(holes: pd.DataFrame, *, k: float, g: float = standard_gravity) -> pd.DataFrame:
    """Per-hole ranges (m) of the published models, side by side; returns a copy of ``holes`` plus the model columns.

    Required columns: :data:`HOLE_COLUMNS` (``linear_charge_kg_m`` may be replaced by ``explosive_density_kg_m3``,
    full-hole geometry, with a warning). Optional: ``charge_length_m``, ``hole_angle_from_vertical_deg`` (default 0),
    ``sdob_m_kg13``, ``powder_factor_kg_m3``. ``k`` is the Richards & Moore site constant (you supply it).

    Added columns: ``rm_charge`` (kg/m, or the actual mass in kg when ``charge_length_m < 1``: Richards & Moore),
    ``rm_face_burst_m``, ``rm_cratering_m``, ``rm_stemming_ejection_m`` (0 for a vertical hole), ``rm_max_m`` (larger of
    face burst and cratering, plus stemming ejection only when the hole angle is > 0), ``lundborg_max_m``,
    ``lundborg_q_m`` (with ``powder_factor_kg_m3``), ``mckenzie_2018_m`` (with ``sdob_m_kg13``).
    ``attrs`` gets ``k``, ``g``, ``methods`` (column -> id, verification, citation), ``warnings`` and ``critiques``.
    """
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    warns: list[str] = []
    out = holes.copy()
    miss = [c for c in HOLE_COLUMNS if c not in out.columns]
    if miss == ["linear_charge_kg_m"] and "explosive_density_kg_m3" in out.columns:
        dens = _column(out, "explosive_density_kg_m3")
        diam = _column(out, "diameter_mm")
        out["linear_charge_kg_m"] = [linear_charge(d, r) for d, r in zip(diam, dens)]
        warns.append("linear_charge_kg_m computed from diameter_mm and explosive_density_kg_m3 (full-hole geometry)")
    elif miss:
        raise _error("holes", f"is missing the column {miss[0]!r}. Required columns: {list(HOLE_COLUMNS)}")
    d, b, sh, lin = (_column(out, c) for c in HOLE_COLUMNS)
    _per_row(out, "diameter_mm", lundborg_max_range, d)  # millimetre range check, row named on error
    clen = _column(out, "charge_length_m") if "charge_length_m" in out.columns else np.full(len(out), np.inf)
    angle = (_column(out, "hole_angle_from_vertical_deg", allow_zero=True)
             if "hole_angle_from_vertical_deg" in out.columns else np.zeros(len(out)))

    short = clen < 1.0
    m = np.where(short, lin * np.where(short, clen, 1.0), lin)
    if short.any():
        warns.append("charge column shorter than 1 m: rm_charge is the actual charge mass (kg), as Richards & Moore "
                     f"prescribe; rows {list(out.index[short])}")
    vertical = angle == 0.0
    if vertical.any():
        warns.append("vertical hole: rm_stemming_ejection_m is 0, but the source says vertical holes need an "
                     "adjustment because stemming can be ejected at a flatter angle; "
                     f"rows {list(out.index[vertical])}")

    out["rm_charge"] = m
    out["rm_face_burst_m"] = _per_row(out, "burden_m", lambda *a: rm_face_burst(k, a[0], a[1], g), m, b)
    out["rm_cratering_m"] = _per_row(out, "stemming_m", lambda *a: rm_cratering(k, a[0], a[1], g), m, sh)
    out["rm_stemming_ejection_m"] = _per_row(
        out, "hole_angle_from_vertical_deg", lambda *a: rm_stemming_ejection(k, a[0], a[1], a[2], g), m, sh, angle)
    rm_max = np.maximum(out["rm_face_burst_m"], out["rm_cratering_m"])
    out["rm_max_m"] = np.where(angle > 0, np.maximum(rm_max, out["rm_stemming_ejection_m"]), rm_max)
    out["lundborg_max_m"] = [lundborg_max_range(x) for x in d]
    methods = {c: _method(mod) for c, mod in (
        ("rm_face_burst_m", Model.RichardsMooreFaceBurst), ("rm_cratering_m", Model.RichardsMooreCratering),
        ("rm_stemming_ejection_m", Model.RichardsMooreStemmingEjection), ("lundborg_max_m", Model.LundborgMaxRange))}
    if "powder_factor_kg_m3" in out.columns:
        q = _column(out, "powder_factor_kg_m3")
        out["lundborg_q_m"] = _per_row(out, "powder_factor_kg_m3", lundborg_range, d, q)
        methods["lundborg_q_m"] = _method(Model.LundborgRange)
    if "sdob_m_kg13" in out.columns:
        s = _column(out, "sdob_m_kg13")
        out["mckenzie_2018_m"] = _per_row(out, "sdob_m_kg13", mckenzie_max_range, s, d)
        methods["mckenzie_2018_m"] = _method(Model.McKenzie2018)
    out.attrs.update({"k": k, "g": g, "methods": methods, "warnings": tuple(warns), "critiques": _CRITIQUES})
    return out


@dataclasses.dataclass(frozen=True)
class Clearance:
    """Clearance distance: the model range times a safety factor, against an optional regulatory minimum."""

    distance_m: float
    governs: str  # "model" | "regulation"
    range_m: float
    factor: float
    target: str  # "people" | "equipment"
    regulation: str | None
    regulation_minimum_m: float | None
    citation: str
    notes: tuple[str, ...]


def clearance(range_m: float, *, target: str, factor: float | None = None, regulation: str | None = None) -> Clearance:
    """Clearance distance for a predicted maximum ``range_m`` (m).

    ``target`` is required: ``"people"`` (default factor 4) or ``"equipment"`` (factor 2), Richards & Moore. An explicit
    ``factor`` (>= 1) replaces the default. ``regulation="PE-DS024-2016-EM"`` applies only to people: the distance is
    then ``max(factor * range_m, 500)`` and ``governs`` says which one decided (a tie goes to ``"model"``). For
    equipment the regulation is ignored with a note (it is about evacuating workers). The models carry no safety margin.
    """
    if target not in _FACTORS:
        raise _error("target", f"is required and must be one of {list(_FACTORS)}. Received: {target!r}")
    if regulation is not None and regulation not in REGULATIONS:
        raise _error("regulation", f"must be None or one of {list(REGULATIONS)}. Received: {regulation!r}")
    if not (isinstance(range_m, (int, float, np.number)) and math.isfinite(range_m) and range_m > 0):
        raise _error("range_m", f"must be finite and > 0. Received: {range_m!r}")
    notes = [_NO_MARGIN]
    if factor is None:
        factor = _FACTORS[target]
    else:
        if not (isinstance(factor, (int, float, np.number)) and math.isfinite(factor) and factor >= 1):
            raise _error("factor", f"must be finite and >= 1. Received: {factor!r}")
        notes.append("user factor")
    model_m = float(factor) * float(range_m)
    cite = _RM_CITATION
    minimum = None
    governs, distance = "model", model_m
    if regulation is not None and target == "equipment":
        notes.append("regulation ignored: it covers the evacuation of workers, not equipment")
    elif regulation is not None:
        minimum = _REG_MINIMUM_M
        cite += "; " + _REG_CITATION
        notes.append("regulation: 500 m safety radius when rock can be projected through 360 degrees; in bench blasts "
                     "at least 500 m in the direction the blast throws")
        if minimum > model_m:
            governs, distance = "regulation", minimum
    return Clearance(distance, governs, float(range_m), float(factor), target, regulation, minimum, cite, tuple(notes))
