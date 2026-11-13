"""Normative profiles (limit criteria with their citation) and the per-event compliance verdict.

A profile is a JSON file: the values stay in the source's units and the loader converts them once to SI (velocity in
mm/s, displacement in mm, distance in m). The profiles shipped in the package keep the verification mark of each
criterion (``"V1"`` primary source, ``"V2"`` reliable secondary); a profile loaded from a dict or a path is ``"user"``.

Criterion kinds: ``frequency_curve`` (piecewise limit against the dominant frequency), ``distance_table`` (a row applies
to ``d <= d_max``; ``d_max`` null is the open last row) and ``flat`` (one limit).

Verdict of :func:`evaluate`, one row per event:

An event gives one or more candidate PPVs of the quantity the profile asks for (never another quantity), and it complies
only when that quantity is fully known:

* ``quantity == "component_max"``: each component column present (``ppv_t/v/l_mm_s``) is a candidate with its own
  frequency (``freq_t/v/l_hz``). With ``ppv_kind == "vertical"`` and no ``ppv_v_mm_s``, ``ppv_mm_s`` is the vertical
  component (``freq_v_hz``, else ``frequency_hz``). Otherwise a ``ppv_mm_s`` of kind ``component_max`` (or ``vertical``,
  next to ``ppv_v_mm_s``) is evaluated whenever it can add information: it is larger than every component present (or
  than ``ppv_v_mm_s``), or ``frequency_hz`` is present; it is then evaluated with ``frequency_hz``. A ``vector_sum``
  ``ppv_mm_s`` is never a candidate here. The maximum component is known only when the three components
  are present or ``ppv_kind == "component_max"`` with a valid ``ppv_mm_s``; otherwise the missing components count as
  a candidate that could not be evaluated.
* ``quantity == "vector_sum"`` / ``"vertical"``: ``pvs_mm_s`` / ``ppv_v_mm_s`` if present, and ``ppv_mm_s`` if
  ``ppv_kind`` is that quantity; both are evaluated when both are present. A vertical ``ppv_mm_s`` uses ``freq_v_hz``
  (else ``frequency_hz``) when there is no ``ppv_v_mm_s``, and ``frequency_hz`` when there is.

A candidate complies when ``ppv <= limit * (1 + 1e-9)`` (equality complies). The status, in this order:

* ``exceeds`` if any candidate that could be evaluated exceeds its limit (``reason`` lists the ones not evaluated);
* ``not_comparable`` if any candidate could not be evaluated: the quantity is missing or incomplete, a PPV is not finite
  and > 0, a frequency curve has no frequency, or a distance table has no distance;
* ``indeterminate`` if any candidate's frequency or distance is outside the criterion's domain;
* ``complies`` otherwise: every candidate was evaluated and complies.

Limit, margins and ``governing`` are those of the candidate with the smallest margin (mm/s) among the exceeding ones, or
among all the evaluated ones; NaN and empty for ``not_comparable``.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import json
import math
import os
from typing import NamedTuple

import numpy as np
import pandas as pd

from ._blastingpy_core import _criteria_curve as _cc
from .units import from_si, to_si

__all__ = ["list_profiles", "load_profile", "evaluate", "max_charge_unmonitored", "Profile"]

_PREFIX = "Domain error [BlastingPy]"
_QUANTITIES = ("component_max", "vector_sum", "vertical")
_KINDS = ("frequency_curve", "distance_table", "flat")
_SEGMENT_KINDS = {"constant_velocity": _cc.SegmentKind.ConstantVelocity,
                  "constant_displacement": _cc.SegmentKind.ConstantDisplacement,
                  "linear": _cc.SegmentKind.Linear, "loglog": _cc.SegmentKind.LogLog}
_REL_TOL = 1e-9
_COMPONENTS = (("transverse", "ppv_t_mm_s", "freq_t_hz"), ("vertical", "ppv_v_mm_s", "freq_v_hz"),
               ("longitudinal", "ppv_l_mm_s", "freq_l_hz"))
_RESULT = ["status", "limit_mm_s", "margin_mm_s", "margin_pct", "governing", "verification", "reason"]


class _Row(NamedTuple):
    d_max_m: float  # inf en la última fila abierta
    limit_mm_s: float
    scaled_distance: float | None  # en las unidades de la fuente (distancia / carga^½)


@dataclasses.dataclass(frozen=True)
class Criterion:
    """One limit criterion of a profile, already in SI."""

    criterion_id: str
    kind: str
    structure: dict
    verification: str
    reference: str
    note: str
    segments: tuple = ()
    rows: tuple = ()
    limit_mm_s: float = math.nan


@dataclasses.dataclass(frozen=True)
class Profile:
    """A normative profile: its source, the quantity it limits and its criteria (by id)."""

    profile_id: str
    name: dict
    quantity: str
    frequency_methods: tuple
    source: dict
    criteria: dict
    jurisdiction: str = ""
    force: str = ""
    quantity_basis: str = ""
    units: dict = dataclasses.field(default_factory=dict)


def _error(name: str, message: str) -> ValueError:
    return ValueError(f"{_PREFIX}: parameter '{name}' {message}")


def _key(data: dict, key: str, where: str):
    if not isinstance(data, dict) or key not in data:
        raise _error("profile", f"{where}is missing the key '{key}'")
    return data[key]


def _profile_files() -> dict:
    """Shipped profiles: {profile_id: resource}."""
    folder = importlib.resources.files("blastingpy").joinpath("profiles")
    files = [f for f in folder.iterdir() if f.name.endswith(".json")]
    return {json.loads(f.read_text(encoding="utf-8"))["profile_id"]: f for f in files}


def list_profiles() -> list[str]:
    """Ids of the profiles shipped with the package, sorted."""
    return sorted(_profile_files())


def _unit(units: dict, quantity: str, where: str) -> str:
    return _key(units, quantity, f"{where}'units' ")


def _segments(raw: list, units: dict, where: str) -> tuple:
    if not isinstance(raw, list) or not raw:
        raise _error("profile", f"{where}'segments' must be a non-empty list")
    kinds, v1, v2 = [], [], []
    for i, s in enumerate(raw):
        at = f"{where}segment {i} "
        kind = _key(s, "kind", at)
        if kind not in _SEGMENT_KINDS:
            raise _error("profile", f"{at}has an unknown kind {kind!r}. Valid kinds: {list(_SEGMENT_KINDS)}")
        quantity = "displacement" if kind == "constant_displacement" else "velocity"
        kinds.append(kind)
        v1.append(to_si(float(_key(s, "value", at)), _unit(units, quantity, where)))
        v2.append(to_si(float(_key(s, "value_max", at)), _unit(units, "velocity", where))
                  if kind in ("linear", "loglog") else math.nan)
    f_max = []
    for i, s in enumerate(raw):
        value = _key(s, "f_max", f"{where}segment {i} ")
        if value == "intersection":
            pair = {kinds[i], kinds[i + 1]} if i + 1 < len(raw) else set()
            if pair != {"constant_velocity", "constant_displacement"}:
                raise _error("profile", f"{where}segment {i}: 'f_max': 'intersection' needs a constant_velocity and a "
                                        f"constant_displacement neighbour as the next segment")
            vel, disp = (v1[i], v1[i + 1]) if kinds[i] == "constant_velocity" else (v1[i + 1], v1[i])
            f_max.append(_cc.intersection_frequency(vel, disp))
        else:
            f_max.append(math.inf if value is None else float(value))
    segments = []
    for i, s in enumerate(raw):
        f_min = float(_key(s, "f_min", f"{where}segment 0 ")) if i == 0 else float(s.get("f_min", f_max[i - 1]))
        segments.append(_cc.Segment(_SEGMENT_KINDS[kinds[i]], f_min, f_max[i], v1[i], v2[i]))
    _cc.limit_at(segments, 1.0)  # valida la curva al cargarla (tramos ordenados, sin huecos)
    return tuple(segments)


def _rows(raw: list, units: dict, where: str) -> tuple:
    if not isinstance(raw, list) or not raw:
        raise _error("profile", f"{where}'rows' must be a non-empty list")
    rows = []
    for i, r in enumerate(raw):
        at = f"{where}row {i} "
        d_max = _key(r, "d_max", at)
        if d_max is None and i + 1 != len(raw):
            raise _error("profile", f"{at}'d_max' may be null only on the last row")
        d_max_m = math.inf if d_max is None else to_si(float(d_max), _unit(units, "distance", where))
        if rows and d_max_m <= rows[-1].d_max_m:
            raise _error("profile", f"{at}'d_max' must be greater than the previous row's")
        sd = r.get("scaled_distance")
        rows.append(_Row(d_max_m, to_si(float(_key(r, "limit", at)), _unit(units, "velocity", where)),
                         None if sd is None else float(sd)))
    return tuple(rows)


def _criterion(raw: dict, units: dict, user: bool) -> Criterion:
    cid = _key(raw, "id", "a criterion ")
    where = f"criterion {cid!r} "
    kind = _key(raw, "kind", where)
    if kind not in _KINDS:
        raise _error("profile", f"{where}has an unknown kind {kind!r}. Valid kinds: {list(_KINDS)}")
    fields = dict(criterion_id=cid, kind=kind, structure=raw.get("structure", {}),
                  verification="user" if user else _key(raw, "verification", where),
                  reference=raw.get("reference", "") if user else _key(raw, "reference", where),
                  note=raw.get("note", ""))
    if kind == "frequency_curve":
        fields["segments"] = _segments(_key(raw, "segments", where), units, where)
    elif kind == "distance_table":
        fields["rows"] = _rows(_key(raw, "rows", where), units, where)
    else:
        fields["limit_mm_s"] = to_si(float(_key(raw, "limit", where)), _unit(units, "velocity", where))
    return Criterion(**fields)


def _build(data: dict, user: bool) -> Profile:
    profile_id = _key(data, "profile_id", "")
    quantity = _key(data, "quantity", "")
    if quantity not in _QUANTITIES:
        raise _error("profile", f"'quantity' must be one of {list(_QUANTITIES)}. Received: {quantity!r}")
    units = _key(data, "units", "")
    criteria = {}
    for raw in _key(data, "criteria", ""):
        c = _criterion(raw, units, user)
        if c.criterion_id in criteria:
            raise _error("profile", f"has the criterion id {c.criterion_id!r} twice")
        criteria[c.criterion_id] = c
    return Profile(profile_id=profile_id, name=data.get("name", {}), quantity=quantity,
                   frequency_methods=tuple(data.get("frequency_methods", ())), source=data.get("source", {}),
                   criteria=criteria, jurisdiction=data.get("jurisdiction", ""), force=data.get("force", ""),
                   quantity_basis=data.get("quantity_basis", ""), units=dict(units))


def load_profile(profile: str | os.PathLike | dict) -> Profile:
    """Load a shipped profile by id, a user profile from a JSON path, or a user profile from a dict.

    User profiles (path or dict) get ``verification == "user"`` on every criterion.
    """
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, dict):
        return _build(profile, user=True)
    if isinstance(profile, (str, os.PathLike)):
        files = _profile_files()
        if isinstance(profile, str) and profile in files:
            return _build(json.loads(files[profile].read_text(encoding="utf-8")), user=False)
        if os.path.isfile(profile):
            with open(profile, encoding="utf-8") as f:
                return _build(json.load(f), user=True)
    raise _error("profile", f"must be one of {sorted(_profile_files())}, a path to a JSON file or a dict. "
                            f"Received: {profile!r}")


def _get_criterion(profile, criterion: str) -> tuple[Profile, Criterion]:
    profile = load_profile(profile)
    if criterion not in profile.criteria:
        raise _error("criterion", f"must be one of {list(profile.criteria)} for profile {profile.profile_id!r}. "
                                  f"Received: {criterion!r}")
    return profile, profile.criteria[criterion]


def _missing(value) -> bool:
    return not (isinstance(value, (int, float, np.number)) and math.isfinite(value) and value > 0)


def _limit(c: Criterion, frequency, distance, freq_name: str) -> tuple:
    """(status or None, limit, governing, reason): status is set only when no limit can be compared."""
    if c.kind == "flat":
        return None, c.limit_mm_s, "", ""
    if c.kind == "distance_table":
        if _missing(distance):
            return "not_comparable", math.nan, "", "distance_m is missing or not positive"
        for i, row in enumerate(c.rows):
            if distance <= row.d_max_m * (1 + _REL_TOL):
                return None, row.limit_mm_s, f"row-{i}", ""
        return ("indeterminate", math.nan, "",
                f"distance {distance:g} m is beyond the last row of the table ({c.rows[-1].d_max_m:g} m)")
    if _missing(frequency):
        return "not_comparable", math.nan, "", f"no dominant frequency ({freq_name}) for the frequency curve"
    r = _cc.limit_at(list(c.segments), float(frequency))
    if not r.defined:
        lo, hi = c.segments[0].f_min, c.segments[-1].f_max
        return ("indeterminate", math.nan, "",
                f"frequency {frequency:g} Hz ({freq_name}) is outside the curve's domain [{lo:g}, {hi:g}] Hz")
    return None, r.limit, f"segment-{r.segment}", ""


def _value(row: dict, column: str) -> float:
    value = row.get(column, math.nan)
    return math.nan if value is None or pd.isna(value) else float(value)


def _ppv(row: dict, column: str, problems: list) -> float:
    """The PPV in ``column``; NaN when empty. A value that is not finite and > 0 is bad data: noted in ``problems``."""
    value = _value(row, column)
    if not math.isnan(value) and _missing(value):
        problems.append(f"{column} is not a finite value > 0 ({value:g})")
        return math.nan
    return value


def _vertical_frequency(row: dict) -> str:
    """Frequency column of a vertical ``ppv_mm_s`` that stands for the vertical component."""
    return "freq_v_hz" if not math.isnan(_value(row, "freq_v_hz")) else "frequency_hz"


def _magnitudes(row: dict, quantity: str) -> tuple[list, list]:
    """Candidates [(component label, ppv, frequency, frequency column)] and the problems that block comparing."""
    kind, problems, candidates = row.get("ppv_kind"), [], []
    if quantity == "component_max":
        absent, peak = [], math.nan
        for label, p, f in _COMPONENTS:
            value = _ppv(row, p, problems)
            if not math.isnan(value):
                candidates.append((label, value, _value(row, f), f))
            elif math.isnan(_value(row, p)):
                absent.append(label)
        if kind in ("component_max", "vertical"):
            peak = _ppv(row, "ppv_mm_s", problems)
            if kind == "vertical" and "vertical" in absent:
                if not math.isnan(peak):  # el PPV del evento es la componente vertical
                    absent.remove("vertical")
                    f = _vertical_frequency(row)
                    candidates.append(("vertical", peak, _value(row, f), f))
            elif not math.isnan(peak):
                label = "" if kind == "component_max" else "vertical"
                top = max((cand[1] for cand in candidates if kind == "component_max" or cand[0] == "vertical"),
                          default=0.0)
                # Se evalúa si puede agregar algo: supera a lo presente o trae su propia frecuencia.
                if peak > top * (1 + _REL_TOL) or not math.isnan(_value(row, "frequency_hz")):
                    candidates.append((label, peak, _value(row, "frequency_hz"), "frequency_hz"))
        complete = not absent or (kind == "component_max" and not math.isnan(peak))
        if candidates and not complete:  # solo se cumple si se conoce la componente máxima
            problems.append(f"missing components: {', '.join(absent)}")
        missing = (f"the profile needs the peak component PPV (ppv_kind 'component_max' or columns "
                   f"ppv_t/v/l_mm_s); the event has ppv_kind {kind!r}")
    else:
        column, freq = ("pvs_mm_s", "frequency_hz") if quantity == "vector_sum" else ("ppv_v_mm_s", "freq_v_hz")
        # freq_v_hz es de ppv_v_mm_s si está; si no, sirve al ppv_mm_s vertical.
        own = _vertical_frequency(row) if quantity == "vertical" and math.isnan(_value(row, column)) else "frequency_hz"
        sources = [(column, freq)] + ([("ppv_mm_s", own)] if kind == quantity else [])
        for p, f in sources:  # ambos se evalúan si están los dos
            value = _ppv(row, p, problems)
            if not math.isnan(value):
                candidates.append(("", value, _value(row, f), f))
        missing = f"the profile needs {column} (or ppv_kind {quantity!r}); the event has ppv_kind {kind!r}"
    if not candidates and not problems:
        problems.append("ppv_mm_s is empty" if kind == quantity else missing)
    return candidates, problems


def _verdict(row: dict, c: Criterion, quantity: str) -> list:
    """Precedence: exceeds > not_comparable > indeterminate > complies (complies only if every candidate complies)."""
    nan = math.nan
    candidates, problems = _magnitudes(row, quantity)
    evaluated, out_of_domain = [], []
    for label, ppv, freq, freq_name in candidates:
        status, limit, governing, reason = _limit(c, freq, _value(row, "distance_m"), freq_name)
        if status == "not_comparable":
            if label and c.kind == "frequency_curve":
                reason = f"the {label} component has a PPV but {reason}"
            problems.append(reason)
        elif status == "indeterminate":
            out_of_domain.append(reason)
        else:
            prefix = f"{label}/" if label and governing else ""
            evaluated.append((limit - ppv, limit, prefix + governing, ppv))
    exceeding = [e for e in evaluated if e[3] > e[1] * (1 + _REL_TOL)]
    if exceeding:
        status, (margin, limit, governing, _) = "exceeds", min(exceeding, key=lambda e: e[0])
        reason = "; ".join(f"not evaluated: {r}" for r in problems + out_of_domain)
    elif problems:
        return ["not_comparable", nan, nan, nan, "", c.verification, "; ".join(problems)]
    elif not evaluated:
        return ["indeterminate", nan, nan, nan, "", c.verification, "; ".join(out_of_domain)]
    else:
        margin, limit, governing, _ = min(evaluated, key=lambda e: e[0])
        status = "indeterminate" if out_of_domain else "complies"
        reason = "; ".join(out_of_domain)
    return [status, limit, margin, 100.0 * margin / limit, governing, c.verification, reason]


def evaluate(events: pd.DataFrame, profile: Profile | str, criterion: str) -> pd.DataFrame:
    """Verdict of each event against ``criterion`` of ``profile`` (a :class:`Profile`, a shipped id, a path or a dict).

    ``events`` needs ``ppv_mm_s`` and ``ppv_kind`` (and ``distance_m`` for a distance table); the other canonical
    columns are used when present. Returns one row per event, same index, with the columns ``status``,
    ``limit_mm_s``, ``margin_mm_s`` (limit - PPV), ``margin_pct`` (of the limit), ``governing``, ``verification`` and
    ``reason``. See the module docstring for the rules.
    """
    profile, c = _get_criterion(profile, criterion)
    required = ["ppv_mm_s", "ppv_kind"] + (["distance_m"] if c.kind == "distance_table" else [])
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise _error("events", f"is missing the column {missing[0]!r}. Required for this criterion: {required}")
    rows = [_verdict(row, c, profile.quantity) for row in events.to_dict("records")]
    out = pd.DataFrame(rows, columns=_RESULT, index=events.index)
    return out.astype({"limit_mm_s": float, "margin_mm_s": float, "margin_pct": float})


def max_charge_unmonitored(profile: Profile | str, distance_m: float, criterion: str = "distance-table") -> float:
    """Charge (kg) allowed without monitoring at ``distance_m``: W = (D / Ds)², in the source's units.

    D is converted to the profile's distance unit, Ds is the scaled-distance factor of the row that applies, and W is
    converted back to kg. In 30 CFR 816.67 and the MEM guide, W is the charge detonated in any 8 ms period.
    """
    profile, c = _get_criterion(profile, criterion)
    if c.kind != "distance_table" or any(r.scaled_distance is None for r in c.rows):
        raise _error("criterion", f"{criterion!r} has no scaled_distance on every row; it cannot give a charge")
    if _missing(distance_m):
        raise _error("distance_m", f"must be finite and > 0. Received: {distance_m!r}")
    row = next((r for r in c.rows if distance_m <= r.d_max_m * (1 + _REL_TOL)), None)
    if row is None:
        raise _error("distance_m", f"{distance_m:g} m is beyond the last row of the table ({c.rows[-1].d_max_m:g} m)")
    d = from_si(distance_m, _unit(profile.units, "distance", ""))
    return to_si((d / row.scaled_distance) ** 2, _unit(profile.units, "charge", ""))
