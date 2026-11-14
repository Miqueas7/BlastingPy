"""Flyrock (rock projection): ballistics, the published range models, a per-hole comparison and the clearance distance.

Units are SI: lengths in m, linear charge in kg/m, velocity in m/s, angles in degrees (hole angles from the vertical,
launch angles from the horizontal), hole diameter in mm, powder factor in kg/m3, SDoB in m/kg^(1/3).

* The core functions are re-exported with their names (:func:`rm_face_burst`, :func:`lundborg_max_range`, ...). Some
  are thin wrappers that add a ``UserWarning`` where the result can come out too short or an input looks like a unit
  mistake: ``k`` outside [13.5, 37] in the ``rm_*`` functions, a vertical hole in :func:`rm_stemming_ejection`,
  :func:`lundborg_range` below Lundborg's maximum, an SDoB above 2.4 in :func:`mckenzie_max_range` and a c/m near the
  minimum in :func:`roth_calibrated_velocity`.
* :func:`compare` puts the Richards & Moore, Lundborg and McKenzie ranges of each hole side by side. They are different
  models, not a consensus: nothing here averages them or picks one; ``max_model_m`` is the largest of them.
* :func:`clearance` turns a range into a distance with the safety factors of Richards & Moore and, optionally, the
  Peruvian regulatory minimum. The models carry no safety margin of their own.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd

from ._blastingpy_core import _flyrock as _fr
from ._blastingpy_core import Verification

range_level = _fr.range_level
max_range_level = _fr.max_range_level
range_to_height = _fr.range_to_height
max_height = _fr.max_height
velocity_for_range = _fr.velocity_for_range
lundborg_max_range = _fr.lundborg_max_range
lundborg_optimal_fragment = _fr.lundborg_optimal_fragment
roth_charge_mass_ratio = _fr.roth_charge_mass_ratio
roth_gurney_velocity = _fr.roth_gurney_velocity
roth_launch_velocity = _fr.roth_launch_velocity
roth_minimum_charge_mass_ratio = _fr.roth_minimum_charge_mass_ratio
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
           "roth_minimum_charge_mass_ratio", "roth_reference_gurney_velocity", "Explosive", "RothCalibration", "Model",
           "info", "standard_gravity", "HOLE_COLUMNS", "REGULATIONS", "linear_charge", "compare", "Clearance",
           "clearance"]

_MARK = {Verification.Primary: "V1", Verification.Secondary: "V2"}
_REG_MINIMUM_M = 500.0
_REG_CITATION = "DS 024-2016-EM, art. 267 a), modificado por DS 034-2023-EM (El Peruano, 30-12-2023)"
_RM_CITATION = ("Richards & Moore, Flyrock Control - By Chance or Design, 30th ISEE (p. 31); "
                "Moore & Richards (2005), p. 14")
_FACTORS = {"people": 4.0, "equipment": 2.0}
_NO_MARGIN = ("the models carry no safety margin of their own: with the real burden and stemming the predicted range "
              "can be exceeded (Moore & Richards 2005, p. 5)")
_ONE_MODEL = "range_m comes from one model: see compare() for the others"
_K_RANGE = (13.5, 37.0)  # Richards y Moore, p. 25
# 300-2000 kg/m3: decisión de diseño (no de la fuente), holgada para ANFO y emulsiones; fuera de ahí, casi seguro g/cc
_DENSITY_KG_M3 = (300.0, 2000.0)
_SDOB_SCALE_END = 2.4  # fin de la escala de SDoB que muestra McKenzie (2018), diaps. 7 y 12
# Roth: aviso bajo 2·(b/a); el 2 es decisión de diseño (inferido): la fuente solo dice que cerca de b/a subestima
_ROTH_NEAR_MINIMUM = 2.0
_POWDER_SHARE = 0.5  # compare avisa cuando lundborg_q_m < 0,5 · lundborg_max_m (elección de diseño, R5)
_POWDER_FACTOR = ("a powder-factor formula tends to underestimate the range (Roth 1979, USBM OFR 77-81, pp. 10-11, "
                  "29-31)")
_SDOB_UNIT = ("SDoB must be in m/kg^(1/3), not ft/lb^(1/3): 2.4 is the end of the scaled-depth-of-burial scale shown "
              "by McKenzie (2018, slides 7 and 12), and a value in ft/lb^(1/3) is about 2.5 times larger, which gives a "
              "range about 7 times shorter")
_VERTICAL = ("the source says vertical holes need an adjustment because stemming can be ejected at a flatter angle "
             "(Richards & Moore, p. 30)")
_CRITIQUES = (
    "Models without drag (ballistics, Richards & Moore) can overestimate the range by factors of 2 to 8 "
    "(Szendrei & Tose 2022, JSAIMM 122(12), pp. 727, 731).",
    "Lundborg's ranges hold only for spherical rock of the optimal size; other shapes fly shorter, and the authors "
    "consider the model an overestimate (Szendrei & Tose 2022, JSAIMM 122(12), pp. 726, 729, 731).",
    "McKenzie calibrated his velocities against Lundborg's, so the model inherits that problem; in his own validation "
    "the measured ranges averaged 4.8 times below the calculated ones (Szendrei & Tose 2022, JSAIMM 122(12), p. 727).",
    "No model here has a safety margin: with the real burden and stemming the range can be exceeded "
    "(Moore & Richards 2005, p. 5); apply a factor with clearance().",
    "There is no universal correlation between powder factor and range, and a powder-factor correlation tends to "
    "underestimate the range (Roth 1979, USBM OFR 77-81, pp. 10-11, 29-31): this applies to Lundborg's powder-factor "
    "range (lundborg_q_m).",
)


def _error(name: str, detail: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {detail}")


def _number(v) -> bool:
    """A finite real number (a bool is not one)."""
    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def _k_message(k) -> str:
    return (f"k = {k:g} is outside the published range [13.5, 37] (Richards & Moore, p. 25); the range goes with "
            "k^2, so check the value")


def _with_k(func, k, *args):
    """Core Richards & Moore function; warns when k is outside the published range."""
    out = func(k, *args)
    if not _K_RANGE[0] <= k <= _K_RANGE[1]:
        warnings.warn(_k_message(k), UserWarning, stacklevel=3)  # 3: la línea que llamó al envoltorio público
    return out


def rm_launch_velocity(k: float, charge_kg_m: float, confinement_m: float) -> float:
    """Richards & Moore face velocity k (sqrt(m)/B)^1.3, m/s. Warns when k is outside [13.5, 37]."""
    return _with_k(_fr.rm_launch_velocity, k, charge_kg_m, confinement_m)


def rm_face_burst(k: float, charge_kg_m: float, burden_m: float, g: float = standard_gravity) -> float:
    """Richards & Moore face-burst maximum range (k^2/g)(sqrt(m)/B)^2.6, m. Warns when k is outside [13.5, 37]."""
    return _with_k(_fr.rm_face_burst, k, charge_kg_m, burden_m, g)


def rm_cratering(k: float, charge_kg_m: float, stemming_m: float, g: float = standard_gravity) -> float:
    """Richards & Moore cratering maximum range (k^2/g)(sqrt(m)/SH)^2.6, m. Warns when k is outside [13.5, 37]."""
    return _with_k(_fr.rm_cratering, k, charge_kg_m, stemming_m, g)


def rm_stemming_ejection(k: float, charge_kg_m: float, stemming_m: float, hole_angle_from_vertical_deg: float,
                         g: float = standard_gravity) -> float:
    """Richards & Moore stemming-ejection range: cratering * sin(2 theta), m; hole angle from the vertical, [0, 90).

    A vertical hole gives 0 with a warning: the source says vertical holes need an adjustment because stemming can be
    ejected at a flatter angle. Warns when k is outside [13.5, 37].
    """
    out = _with_k(_fr.rm_stemming_ejection, k, charge_kg_m, stemming_m, hole_angle_from_vertical_deg, g)
    if hole_angle_from_vertical_deg == 0:
        warnings.warn(f"vertical hole: the stemming-ejection range is 0, but {_VERTICAL}", UserWarning, stacklevel=2)
    return out


def rm_min_confinement(k: float, charge_kg_m: float, range_m: float, g: float = standard_gravity) -> float:
    """Burden or stemming that limits the maximum range to range_m, m. Warns when k is outside [13.5, 37]."""
    return _with_k(_fr.rm_min_confinement, k, charge_kg_m, range_m, g)


def lundborg_range(diameter_mm: float, powder_factor_kg_m3: float) -> float:
    """Lundborg range 143 d (q - 0.2) in m (hole diameter in mm; q > 0.2 kg/m3, inferred). Secondary source.

    Warns when the result is below Lundborg's maximum range for the diameter: a powder-factor formula tends to
    underestimate the range (Roth 1979, pp. 10-11, 29-31).
    """
    out = _fr.lundborg_range(diameter_mm, powder_factor_kg_m3)
    cap = _fr.lundborg_max_range(diameter_mm)
    if out < cap:
        warnings.warn(f"lundborg_range gives {out:.0f} m, below Lundborg's maximum range for this diameter "
                      f"({cap:.0f} m): {_POWDER_FACTOR}", UserWarning, stacklevel=2)
    return out


def mckenzie_max_range(sdob_m_kg13: float, diameter_mm: float) -> float:
    """McKenzie (2018) maximum range 10 SDoB^-2.167 d^0.667 in m; SDoB in m/kg^(1/3), supplied by the user.

    Warns when SDoB is above 2.4, the end of the scale McKenzie (2018) shows: an SDoB in ft/lb^(1/3) is about 2.5 times
    larger and gives a range about 7 times shorter.
    """
    out = _fr.mckenzie_max_range(sdob_m_kg13, diameter_mm)
    if sdob_m_kg13 > _SDOB_SCALE_END:
        warnings.warn(f"SDoB = {sdob_m_kg13:g}: {_SDOB_UNIT}", UserWarning, stacklevel=2)
    return out


def roth_calibrated_velocity(rock: RothCalibration, charge_mass_ratio: float) -> float:
    """Roth calibrated velocity sqrt(a c/m - b), m/s; valid for roth_reference_gurney_velocity(rock).

    Raises if c/m is at or below b/a (:func:`roth_minimum_charge_mass_ratio`). Near b/a the fit predicts near-zero
    velocities while Roth's data show finite ones (Roth 1979, p. 21): it underestimates there, so this warns when c/m is
    below 2 b/a (the factor 2 is a library choice, inferred, not from the source).
    """
    out = _fr.roth_calibrated_velocity(rock, charge_mass_ratio)
    minimum = _fr.roth_minimum_charge_mass_ratio(rock)
    if charge_mass_ratio < _ROTH_NEAR_MINIMUM * minimum:
        warnings.warn(f"c/m = {charge_mass_ratio:.3g} is below twice the minimum c/m of this calibration "
                      f"({minimum:.3g}): near that minimum the fitted line predicts near-zero velocities while Roth's "
                      "data show finite ones (Roth 1979, p. 21), so it is likely to underestimate the velocity; the "
                      "factor of 2 is a library choice (inferred)", UserWarning, stacklevel=2)
    return out


def linear_charge(diameter_mm: float, explosive_density_kg_m3: float) -> float:
    """Linear charge of a full hole in kg/m: density * pi * d^2 / 4 (geometry; no coupling or decking).

    The density must be within 300-2000 kg/m3: a library guard against a density given in g/cc, not a source value.
    """
    for name, v in (("diameter_mm", diameter_mm), ("explosive_density_kg_m3", explosive_density_kg_m3)):
        if not (_number(v) and v > 0):
            raise _error(name, f"must be finite and > 0. Received: {v!r}")
    lo, hi = _DENSITY_KG_M3
    if not lo <= explosive_density_kg_m3 <= hi:
        raise _error("explosive_density_kg_m3",
                     f"must be within [{lo:g}, {hi:g}] kg/m3 (g/cc by mistake?). Received: {explosive_density_kg_m3!r}")
    return explosive_density_kg_m3 * math.pi * (diameter_mm / 1000.0) ** 2 / 4.0


def _column(df: pd.DataFrame, name: str, *, allow_zero: bool = False, allow_nan: bool = False) -> np.ndarray:
    """Column as floats; finite and > 0 (>= 0 with ``allow_zero``; NaN kept with ``allow_nan``), else an error naming
    column and row."""
    try:
        x = pd.to_numeric(df[name], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise _error("holes", f"column {name!r} must be numeric") from None
    for i, v in enumerate(x):
        if allow_nan and math.isnan(v):
            continue
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


def _nan_through(func):
    """func, except that a NaN argument (an optional value not given for that row) gives NaN."""
    return lambda *a: math.nan if any(math.isnan(x) for x in a) else func(*a)


def _method(model) -> dict:
    m = info(model)
    return {"id": m.method_id, "verification": _MARK[m.verification], "citation": m.source}


def compare(holes: pd.DataFrame, *, k: float, g: float = standard_gravity) -> pd.DataFrame:
    """Per-hole ranges (m) of the published models, side by side; returns a copy of ``holes`` plus the model columns.

    Required columns: :data:`HOLE_COLUMNS` (``linear_charge_kg_m`` may be replaced by ``explosive_density_kg_m3``,
    300-2000 kg/m3, full-hole geometry, with a warning). Optional: ``charge_length_m``, ``hole_angle_from_vertical_deg``
    (default 0), ``sdob_m_kg13`` (m/kg^(1/3)), ``powder_factor_kg_m3``. A NaN in an optional column means «not given
    for that row»: NaN in that model's column, the per-metre charge, or a vertical hole. ``k`` is the Richards & Moore
    site constant (you supply it; a warning when it is outside [13.5, 37]).

    Added columns: ``rm_charge`` (kg/m, or the actual mass in kg when ``charge_length_m < 1``: Richards & Moore),
    ``rm_face_burst_m``, ``rm_cratering_m``, ``rm_stemming_ejection_m`` (0 for a vertical hole), ``rm_max_m`` (larger of
    face burst and cratering; the cratering range bounds the stemming ejection, since sin 2 theta <= 1),
    ``lundborg_max_m``, ``lundborg_q_m`` (with ``powder_factor_kg_m3``), ``mckenzie_2018_m`` (with ``sdob_m_kg13``) and
    ``max_model_m``: the largest of the hole-specific ranges of the row (``rm_face_burst_m``, ``rm_cratering_m``,
    ``rm_stemming_ejection_m``, ``lundborg_q_m``, ``mckenzie_2018_m``; NaN ignored). ``lundborg_max_m`` is not in it:
    it depends only on the diameter (Lundborg's practical maximum, for the optimal fragment), so it is the same for
    every hole of one diameter and does not tell which hole is the worst; it is the diameter-only upper envelope.
    ``attrs`` gets ``k``, ``g``, ``methods`` (column -> id, verification, citation), ``warnings`` (among them the rows
    where ``max_model_m`` is more than twice ``rm_max_m``, where ``lundborg_max_m`` is more than twice
    ``max_model_m``, where ``lundborg_q_m`` is below half of ``lundborg_max_m`` and where the SDoB looks like
    ft/lb^(1/3)) and ``critiques``.
    """
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    for name, v in (("k", k), ("g", g)):
        if not (_number(v) and v > 0):
            raise _error(name, f"must be finite and > 0. Received: {v!r}")
    warns: list[str] = []
    if not _K_RANGE[0] <= k <= _K_RANGE[1]:
        warns.append(_k_message(k))
    out = holes.copy()
    miss = [c for c in HOLE_COLUMNS if c not in out.columns]
    derived = miss == ["linear_charge_kg_m"] and "explosive_density_kg_m3" in out.columns
    if derived:
        dens = _column(out, "explosive_density_kg_m3")
        diam = _column(out, "diameter_mm")
        out["linear_charge_kg_m"] = _per_row(out, "explosive_density_kg_m3", linear_charge, diam, dens)
        warns.append("linear_charge_kg_m computed from diameter_mm and explosive_density_kg_m3 (full-hole geometry)")
    elif miss:
        raise _error("holes", f"is missing the column {miss[0]!r}. Required columns: {list(HOLE_COLUMNS)}")
    d, b, sh, lin = (_column(out, c) for c in HOLE_COLUMNS)
    _per_row(out, "diameter_mm", _fr.lundborg_max_range, d)  # rango en milímetros; nombra la fila si falla
    if not derived:
        lo, hi = _DENSITY_KG_M3
        density = lin / (math.pi / 4.0 * (d / 1000.0) ** 2)
        high = density > hi
        if high.any():
            raise _error("holes", f"column 'linear_charge_kg_m', rows {list(out.index[high])}: the implied explosive "
                                  f"density linear_charge_kg_m / (pi/4 d^2) is above {hi:g} kg/m3, which is impossible "
                                  "(check the units of linear_charge_kg_m and diameter_mm)")
        low = density < lo
        if low.any():
            warns.append(f"linear_charge_kg_m implies an explosive density below {lo:g} kg/m3 (a decoupled or decked "
                         f"charge, or a unit mistake?); rows {list(out.index[low])}")
    clen = np.full(len(out), np.inf)
    if "charge_length_m" in out.columns:
        clen = _column(out, "charge_length_m", allow_nan=True)
        clen = np.where(np.isnan(clen), np.inf, clen)  # NaN: se usa la carga por metro
    if "hole_angle_from_vertical_deg" in out.columns:
        angle = _column(out, "hole_angle_from_vertical_deg", allow_zero=True, allow_nan=True)
        vertical = (angle == 0.0) | np.isnan(angle)
        angle = np.where(np.isnan(angle), 0.0, angle)
        if vertical.any():
            warns.append("vertical hole (a missing angle is taken as 0): rm_stemming_ejection_m is 0, but "
                         f"{_VERTICAL}; rows {list(out.index[vertical])}")
    else:
        angle = np.zeros(len(out))
        warns.append("hole angle not given: assumed vertical for every row, so rm_stemming_ejection_m is 0, but "
                     + _VERTICAL)

    short = clen < 1.0
    m = np.where(short, lin * np.where(short, clen, 1.0), lin)
    if short.any():
        warns.append("charge column shorter than 1 m: rm_charge is the actual charge mass (kg), as Richards & Moore "
                     f"prescribe; rows {list(out.index[short])}")

    out["rm_charge"] = m
    out["rm_face_burst_m"] = _per_row(out, "burden_m", lambda *a: _fr.rm_face_burst(k, a[0], a[1], g), m, b)
    out["rm_cratering_m"] = _per_row(out, "stemming_m", lambda *a: _fr.rm_cratering(k, a[0], a[1], g), m, sh)
    out["rm_stemming_ejection_m"] = _per_row(
        out, "hole_angle_from_vertical_deg", lambda *a: _fr.rm_stemming_ejection(k, a[0], a[1], a[2], g), m, sh, angle)
    # el cráter acota la eyección de taco (sen 2θ ≤ 1), así que basta el mayor de frente y cráter
    out["rm_max_m"] = np.maximum(out["rm_face_burst_m"], out["rm_cratering_m"])
    out["lundborg_max_m"] = [_fr.lundborg_max_range(x) for x in d]
    methods = {c: _method(mod) for c, mod in (
        ("rm_face_burst_m", Model.RichardsMooreFaceBurst), ("rm_cratering_m", Model.RichardsMooreCratering),
        ("rm_stemming_ejection_m", Model.RichardsMooreStemmingEjection), ("lundborg_max_m", Model.LundborgMaxRange))}
    if "powder_factor_kg_m3" in out.columns:
        q = _column(out, "powder_factor_kg_m3", allow_nan=True)
        out["lundborg_q_m"] = _per_row(out, "powder_factor_kg_m3", _nan_through(_fr.lundborg_range), d, q)
        methods["lundborg_q_m"] = _method(Model.LundborgRange)
        # umbral de diseño (R5): por debajo de la mitad de lundborg_max_m; con solo «por debajo» avisaba en casi todo pozo
        below = out["lundborg_q_m"] < _POWDER_SHARE * out["lundborg_max_m"]
        if below.any():
            warns.append(f"lundborg_q_m is below half of lundborg_max_m: {_POWDER_FACTOR}; "
                         f"rows {list(out.index[below])}")
    if "sdob_m_kg13" in out.columns:
        s = _column(out, "sdob_m_kg13", allow_nan=True)
        out["mckenzie_2018_m"] = _per_row(out, "sdob_m_kg13", _nan_through(_fr.mckenzie_max_range), s, d)
        methods["mckenzie_2018_m"] = _method(Model.McKenzie2018)
        feet = s > _SDOB_SCALE_END
        if feet.any():
            warns.append(f"sdob_m_kg13 above {_SDOB_SCALE_END}: {_SDOB_UNIT}; rows {list(out.index[feet])}")
    ranges = [c for c in ("rm_face_burst_m", "rm_cratering_m", "rm_stemming_ejection_m", "lundborg_q_m",
                          "mckenzie_2018_m") if c in out.columns]
    out["max_model_m"] = out[ranges].max(axis=1)
    envelope = out["lundborg_max_m"] > 2.0 * out["max_model_m"]
    if envelope.any():
        sizes = ", ".join(f"{v:.0f}" for v in sorted(set(out.loc[envelope, "lundborg_max_m"])))
        warns.append(f"Lundborg's diameter-only maximum, {sizes} m, exceeds the hole-specific models (more than twice "
                     f"max_model_m): the choice of model is the engineer's; rows {list(out.index[envelope])}")
    wide = out["max_model_m"] > 2.0 * out["rm_max_m"]
    if wide.any():
        warns.append("max_model_m is more than twice rm_max_m: the models disagree widely, and choosing one is the "
                     f"engineer's decision; rows {list(out.index[wide])}")
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
    regulation: str | None  # None when not given or ignored (equipment)
    regulation_minimum_m: float | None
    citation: str
    notes: tuple[str, ...]


def clearance(range_m: float, *, target: str, factor: float | None = None, regulation: str | None = None) -> Clearance:
    """Clearance distance for a predicted maximum ``range_m`` (m).

    ``target`` is required: ``"people"`` (default factor 4) or ``"equipment"`` (factor 2), Richards & Moore. An explicit
    ``factor`` (>= 1) replaces the default. ``regulation="PE-DS024-2016-EM"`` applies only to people: the distance is
    then ``max(factor * range_m, 500)`` and ``governs`` says which one decided (a tie goes to ``"model"``). For
    equipment the regulation is ignored with a note (it is about evacuating workers) and ``regulation`` is ``None``.
    The models carry no safety margin, and ``range_m`` comes from one model: :func:`compare` shows the others.
    """
    if target not in _FACTORS:
        raise _error("target", f"is required and must be one of {list(_FACTORS)}. Received: {target!r}")
    if regulation is not None and regulation not in REGULATIONS:
        raise _error("regulation", f"must be None or one of {list(REGULATIONS)}. Received: {regulation!r}")
    if not (_number(range_m) and range_m > 0):
        raise _error("range_m", f"must be finite and > 0. Received: {range_m!r}")
    notes = [_NO_MARGIN, _ONE_MODEL]
    if factor is None:
        factor = _FACTORS[target]
    else:
        if not (_number(factor) and factor >= 1):
            raise _error("factor", f"must be finite and >= 1. Received: {factor!r}")
        notes.append("user factor")
    model_m = float(factor) * float(range_m)
    cite = _RM_CITATION
    minimum = None
    governs, distance = "model", model_m
    if regulation is not None and target == "equipment":
        notes.append("regulation ignored: it covers the evacuation of workers, not equipment")
        regulation = None
    elif regulation is not None:
        minimum = _REG_MINIMUM_M
        cite += "; " + _REG_CITATION
        notes.append("regulation: 500 m safety radius when rock can be projected through 360 degrees; in bench blasts "
                     "at least 500 m in the direction the blast throws")
        if minimum > model_m:
            governs, distance = "regulation", minimum
    return Clearance(distance, governs, float(range_m), float(factor), target, regulation, minimum, cite, tuple(notes))
