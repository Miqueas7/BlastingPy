"""Normative profiles (limit criteria with their citation) and the per-event compliance verdict.

A profile is a JSON file: the values stay in the source's units and the loader converts them once to SI (velocity in
mm/s, displacement in mm, distance in m). The profiles shipped in the package keep the verification mark of each
criterion (``"V1"`` primary source, ``"V2"`` reliable secondary); a profile loaded from a dict or a path is ``"user"``.

Criterion kinds: ``frequency_curve`` (piecewise limit against the dominant frequency), ``distance_table`` (a row applies
to ``d <= d_max``; ``d_max`` null is the open last row; ``d_max`` strictly increasing) and ``flat`` (one limit). Every
number of a criterion (limits, ``value``, ``value_max``, ``d_max``, ``scaled_distance``, ``f_max``) must be finite and
> 0, and ``f_min`` finite and >= 0; a JSON file with the tokens ``NaN`` or ``Infinity`` is refused.

Verdict of :func:`evaluate`, one row per event:

An event gives candidate (value, frequency) pairs of the quantity the profile asks for (never another quantity). Values
and frequencies come in groups, and every value of a group is paired with every frequency of the group, so filling in a
column can only add pairs (the verdict is monotonic: it never turns ``exceeds`` into ``complies``). A group with values
but no frequency gives candidates that cannot be evaluated against a curve; ``distance_table`` and ``flat`` criteria
ignore the frequency. The event complies only when the quantity is fully known:

* vertical group: values ``ppv_v_mm_s``, plus ``ppv_mm_s`` when ``ppv_kind == "vertical"``; frequencies ``freq_v_hz``,
  plus ``frequency_hz`` when ``ppv_kind == "vertical"``.
* ``quantity == "component_max"``: transverse (``ppv_t_mm_s``, ``freq_t_hz``), the vertical group and longitudinal
  (``ppv_l_mm_s``, ``freq_l_hz``), with ``governing`` prefixed by the component. With ``ppv_kind == "component_max"``,
  ``ppv_mm_s`` (the peak, no prefix) is a candidate with ``frequency_hz`` when that frequency is present, or without a
  frequency when it is larger than every component present; otherwise it adds nothing. A ``vector_sum`` ``ppv_mm_s`` is
  never a candidate here. The maximum component is known only when the transverse, vertical and longitudinal values are
  present or ``ppv_kind == "component_max"`` with a valid ``ppv_mm_s``; otherwise the missing components count as a
  candidate that could not be evaluated. Known limitation (accepted convention): a ``component_max`` peak with its
  ``frequency_hz`` is taken as the event's maximum component; smaller components at lower frequencies that the event
  does not carry are not checked.
* ``quantity == "vertical"``: the vertical group.
* ``quantity == "vector_sum"``: values ``pvs_mm_s``, plus ``ppv_mm_s`` when ``ppv_kind == "vector_sum"``; frequency
  ``frequency_hz``.

A value or a frequency that is present but not finite and > 0 is bad data and makes the event ``not_comparable``.

A candidate complies when ``ppv <= limit * (1 + 1e-9)`` (equality complies). The status, in this order:

* ``exceeds`` if any candidate that could be evaluated exceeds its limit (``reason`` lists the ones not evaluated);
* ``not_comparable`` if any candidate could not be evaluated: the quantity is missing or incomplete, a value is bad
  data, a frequency curve has no frequency, or a distance table has no distance;
* ``indeterminate`` if any candidate's frequency or distance is outside the criterion's domain;
* ``complies`` otherwise: every candidate was evaluated and complies.

Limit, margins and ``governing`` are those of the candidate with the smallest margin (mm/s) among the exceeding ones, or
among all the evaluated ones; NaN and empty for ``not_comparable``. ``reason`` is empty for ``complies``, except
against a ``frequency_curve`` when the maximum component is known only through the ``component_max`` peak (some of the
three components are absent): then it carries the note ``peak-only check: components at other frequencies are not
verified`` (the accepted limitation above). :func:`candidates` returns, per event, the candidates the verdict used.
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
_PEAK_ONLY = "peak-only check: components at other frequencies are not verified"
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
    note: str | dict  # como en el JSON: texto o {"en": ..., "es": ...}
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


def _number(raw: dict, key: str, at: str, *, allow_zero: bool = False) -> float:
    """``raw[key]`` as a float; it must be a finite number > 0 (>= 0 with ``allow_zero``)."""
    value = _key(raw, key, at)
    ok = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
          and (value >= 0 if allow_zero else value > 0))
    if not ok:
        bound = ">= 0" if allow_zero else "> 0"
        raise _error("profile", f"{at}'{key}' must be a finite number {bound}. Received: {value!r}")
    return float(value)


def _reject_constant(token: str):
    raise _error("profile", f"contains the JSON token {token}; every number of a profile must be finite")


def _json(text: str) -> dict:
    return json.loads(text, parse_constant=_reject_constant)  # json acepta NaN e Infinity sin esto


def _profile_files() -> dict:
    """Shipped profiles: {profile_id: resource}."""
    folder = importlib.resources.files("blastingpy").joinpath("profiles")
    files = [f for f in folder.iterdir() if f.name.endswith(".json")]
    return {_json(f.read_text(encoding="utf-8"))["profile_id"]: f for f in files}


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
        at = f"{where}segment {i}: "
        kind = _key(s, "kind", at)
        if kind not in _SEGMENT_KINDS:
            raise _error("profile", f"{at}has an unknown kind {kind!r}. Valid kinds: {list(_SEGMENT_KINDS)}")
        quantity = "displacement" if kind == "constant_displacement" else "velocity"
        kinds.append(kind)
        v1.append(to_si(_number(s, "value", at), _unit(units, quantity, where)))
        v2.append(to_si(_number(s, "value_max", at), _unit(units, "velocity", where))
                  if kind in ("linear", "loglog") else math.nan)
    f_max = []
    for i, s in enumerate(raw):
        at = f"{where}segment {i}: "
        value = _key(s, "f_max", at)
        if value == "intersection":
            pair = {kinds[i], kinds[i + 1]} if i + 1 < len(raw) else set()
            if pair != {"constant_velocity", "constant_displacement"}:
                raise _error("profile", f"{at}'f_max': 'intersection' needs a constant_velocity and a "
                                        f"constant_displacement neighbour as the next segment")
            vel, disp = (v1[i], v1[i + 1]) if kinds[i] == "constant_velocity" else (v1[i + 1], v1[i])
            f_max.append(_cc.intersection_frequency(vel, disp))
        else:
            f_max.append(math.inf if value is None else _number(s, "f_max", at))
    segments = []
    for i, s in enumerate(raw):
        at = f"{where}segment {i}: "
        f_min = _number(s, "f_min", at, allow_zero=True) if i == 0 or "f_min" in s else f_max[i - 1]
        segments.append(_cc.Segment(_SEGMENT_KINDS[kinds[i]], f_min, f_max[i], v1[i], v2[i]))
    _cc.limit_at(segments, 1.0)  # valida la curva al cargarla (tramos ordenados, sin huecos)
    return tuple(segments)


def _rows(raw: list, units: dict, where: str) -> tuple:
    if not isinstance(raw, list) or not raw:
        raise _error("profile", f"{where}'rows' must be a non-empty list")
    rows = []
    for i, r in enumerate(raw):
        at = f"{where}row {i}: "
        d_max = _key(r, "d_max", at)
        if d_max is None and i + 1 != len(raw):
            raise _error("profile", f"{at}'d_max' may be null only on the last row")
        d_max_m = math.inf if d_max is None else to_si(_number(r, "d_max", at), _unit(units, "distance", where))
        if rows and d_max_m <= rows[-1].d_max_m:
            raise _error("profile", f"{at}'d_max' must be greater than the previous row's")
        sd = None if r.get("scaled_distance") is None else _number(r, "scaled_distance", at)
        rows.append(_Row(d_max_m, to_si(_number(r, "limit", at), _unit(units, "velocity", where)), sd))
    return tuple(rows)


def _criterion(raw: dict, units: dict, user: bool, profile_id) -> Criterion:
    cid = _key(raw, "id", f"(profile {profile_id!r}) a criterion ")
    where = f"(profile {profile_id!r}, criterion {cid!r}) "
    kind = _key(raw, "kind", where)
    if kind not in _KINDS:
        raise _error("profile", f"{where}has an unknown kind {kind!r}. Valid kinds: {list(_KINDS)}")
    note = raw.get("note", "")
    if not (isinstance(note, str) or (isinstance(note, dict) and all(isinstance(v, str) for v in note.values()))):
        raise _error("profile", f"{where}'note' must be a text or a dict of texts by language "
                                f"({{'en': ..., 'es': ...}}). Received: {note!r}")
    fields = dict(criterion_id=cid, kind=kind, structure=raw.get("structure", {}),
                  verification="user" if user else _key(raw, "verification", where),
                  reference=raw.get("reference", "") if user else _key(raw, "reference", where),
                  note=note)
    if kind == "frequency_curve":
        fields["segments"] = _segments(_key(raw, "segments", where), units, where)
    elif kind == "distance_table":
        fields["rows"] = _rows(_key(raw, "rows", where), units, where)
    else:
        fields["limit_mm_s"] = to_si(_number(raw, "limit", where), _unit(units, "velocity", where))
    return Criterion(**fields)


def _build(data: dict, user: bool) -> Profile:
    profile_id = _key(data, "profile_id", "")
    quantity = _key(data, "quantity", "")
    if quantity not in _QUANTITIES:
        raise _error("profile", f"'quantity' must be one of {list(_QUANTITIES)}. Received: {quantity!r}")
    units = _key(data, "units", "")
    criteria = {}
    for raw in _key(data, "criteria", ""):
        c = _criterion(raw, units, user, profile_id)
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
            return _build(_json(files[profile].read_text(encoding="utf-8")), user=False)
        if os.path.isfile(profile):
            with open(profile, encoding="utf-8") as f:
                return _build(_json(f.read()), user=True)
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


def _read(row: dict, column: str, problems: list) -> float:
    """The value in ``column``; NaN when empty. A value that is not finite and > 0 is bad data: noted in ``problems``."""
    value = _value(row, column)
    if not math.isnan(value) and _missing(value):
        problems.append(f"{column} is not a finite value > 0 ({value:g})")
        return math.nan
    return value


def _group(row: dict, label: str, values: list, freqs: list, candidates: list, problems: list) -> bool:
    """Add every (value, frequency) pair of a group; True if the group has a value.

    Without any frequency each value is one candidate with NaN frequency (not evaluable against a curve).
    """
    vals = [v for v in (_read(row, c, problems) for c in values) if not math.isnan(v)]
    fs = [(f, c) for c in freqs if not math.isnan(f := _read(row, c, problems))] or [(math.nan, " or ".join(freqs))]
    pairs = {}
    for v in vals:
        for f, name in fs:
            pairs.setdefault((v, f), name)  # sin pares repetidos
    candidates.extend((label, v, f, name) for (v, f), name in pairs.items())
    return bool(vals)


def _magnitudes(row: dict, quantity: str) -> tuple[list, list, bool]:
    """Candidates [(component label, ppv, frequency, frequency column)], the problems that block comparing, and whether
    the maximum component is known only through the recorded peak (some of the three components are absent).

    Each value of the quantity is paired with every frequency the event gives for it, so a new column only adds pairs.
    """
    kind, problems, candidates, peak_only = row.get("ppv_kind"), [], [], False
    vertical = (["ppv_v_mm_s"] + (["ppv_mm_s"] if kind == "vertical" else []),
                ["freq_v_hz"] + (["frequency_hz"] if kind == "vertical" else []))
    if quantity == "component_max":
        groups = {"transverse": (["ppv_t_mm_s"], ["freq_t_hz"]), "vertical": vertical,
                  "longitudinal": (["ppv_l_mm_s"], ["freq_l_hz"])}
        absent = [label for label, (values, freqs) in groups.items()
                  if not _group(row, label, values, freqs, candidates, problems)]
        peak = _read(row, "ppv_mm_s", problems) if kind == "component_max" else math.nan
        if not math.isnan(peak):
            top = max((cand[1] for cand in candidates), default=0.0)
            f = _read(row, "frequency_hz", problems)
            # El pico con su frecuencia se evalúa siempre; sin ella, solo si supera a las componentes presentes.
            if not math.isnan(f) or peak > top * (1 + _REL_TOL):
                candidates.append(("", peak, f, "frequency_hz"))
        if candidates and absent and math.isnan(peak):  # solo se cumple si se conoce la componente máxima
            problems.append(f"missing components: {', '.join(absent)}")
        peak_only = bool(absent) and not math.isnan(peak)
        missing = (f"the profile needs the peak component PPV (ppv_kind 'component_max' or columns "
                   f"ppv_t/v/l_mm_s); the event has ppv_kind {kind!r}")
    elif quantity == "vertical":
        _group(row, "", *vertical, candidates, problems)
        missing = f"the profile needs ppv_v_mm_s (or ppv_kind 'vertical'); the event has ppv_kind {kind!r}"
    else:
        values = ["pvs_mm_s"] + (["ppv_mm_s"] if kind == "vector_sum" else [])
        _group(row, "", values, ["frequency_hz"], candidates, problems)
        missing = f"the profile needs pvs_mm_s (or ppv_kind 'vector_sum'); the event has ppv_kind {kind!r}"
    if not candidates and not problems:
        problems.append("ppv_mm_s is empty" if kind == quantity else missing)
    return candidates, problems, peak_only


class _Evaluated(NamedTuple):
    label: str
    ppv: float
    frequency: float
    status: str | None  # None: se comparó con un límite
    limit: float
    governing: str
    reason: str


def _row_candidates(row: dict, c: Criterion, quantity: str) -> tuple[list, list, bool]:
    """The candidates of one event, each with its limit (or why it has none); the single source of the verdict and of
    the chart. Returns ([_Evaluated], problems of the event, peak_only)."""
    pairs, problems, peak_only = _magnitudes(row, quantity)
    distance, out = _value(row, "distance_m"), []
    for label, ppv, freq, freq_name in pairs:
        status, limit, governing, reason = _limit(c, freq, distance, freq_name)
        if status is None and _missing(limit):  # defensa: un límite no finito o <= 0 nunca da «cumple»
            reason = f"the criterion's limit is not a finite value > 0 ({limit:g})"
            status, limit, governing = "not_comparable", math.nan, ""
        elif status == "not_comparable" and label and c.kind == "frequency_curve":
            reason = f"the {label} component has a PPV but {reason}"
        out.append(_Evaluated(label, ppv, freq, status, limit, governing, reason))
    return out, problems, peak_only


def _verdict(row: dict, c: Criterion, quantity: str) -> list:
    """Precedence: exceeds > not_comparable > indeterminate > complies (complies only if every candidate complies)."""
    nan = math.nan
    candidates, problems, peak_only = _row_candidates(row, c, quantity)
    evaluated, out_of_domain = [], []
    for e in candidates:
        if e.status == "not_comparable":
            problems.append(e.reason)
        elif e.status == "indeterminate":
            out_of_domain.append(e.reason)
        else:
            prefix = f"{e.label}/" if e.label and e.governing else ""
            evaluated.append((e.limit - e.ppv, e.limit, prefix + e.governing, e.ppv))
    exceeding = [e for e in evaluated if e[3] > e[1] * (1 + _REL_TOL)]
    if exceeding:
        status, (margin, limit, governing, _) = "exceeds", min(exceeding, key=lambda e: e[0])
        reason = "; ".join(f"not evaluated: {r}" for r in dict.fromkeys(problems + out_of_domain))
    elif problems:
        return ["not_comparable", nan, nan, nan, "", c.verification, "; ".join(dict.fromkeys(problems))]
    elif not evaluated:
        return ["indeterminate", nan, nan, nan, "", c.verification, "; ".join(out_of_domain)]
    else:
        margin, limit, governing, _ = min(evaluated, key=lambda e: e[0])
        status = "indeterminate" if out_of_domain else "complies"
        reason = "; ".join(out_of_domain)
        if status == "complies" and peak_only and c.kind == "frequency_curve":
            reason = _PEAK_ONLY  # la limitación aceptada viaja con el resultado
    return [status, limit, margin, 100.0 * margin / limit, governing, c.verification, reason]


def _prepare(events: pd.DataFrame, profile, criterion: str) -> tuple[Profile, Criterion, list]:
    profile, c = _get_criterion(profile, criterion)
    required = ["ppv_mm_s", "ppv_kind"] + (["distance_m"] if c.kind == "distance_table" else [])
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise _error("events", f"is missing the column {missing[0]!r}. Required for this criterion: {required}")
    return profile, c, events.to_dict("records")


class Candidate(NamedTuple):
    """One value of an event that the verdict compared (or tried to compare) with the criterion's limit."""

    label: str  # "transverse", "vertical", "longitudinal", or "" for the event's own value of the quantity
    ppv_mm_s: float
    frequency_hz: float  # NaN when the event gives no frequency for this value
    limit_mm_s: float  # NaN when the value could not be compared (no frequency or distance, outside the domain)
    exceeds: bool  # above its limit, with the verdict's tolerance


def candidates(events: pd.DataFrame, profile: Profile | str, criterion: str) -> list[list[Candidate]]:
    """Per event, the candidates :func:`evaluate` used for its verdict (same rules, same order).

    For charts and audits: :func:`blastingpy.plotting.plot_compliance` draws exactly these. Values that are bad data,
    and components that are missing, give no candidate (they show up in the verdict's ``reason``).
    """
    profile, c, rows = _prepare(events, profile, criterion)
    out = []
    for row in rows:
        cands = _row_candidates(row, c, profile.quantity)[0]
        out.append([Candidate(e.label, e.ppv, e.frequency, e.limit,
                              e.status is None and e.ppv > e.limit * (1 + _REL_TOL)) for e in cands])
    return out


def evaluate(events: pd.DataFrame, profile: Profile | str, criterion: str) -> pd.DataFrame:
    """Verdict of each event against ``criterion`` of ``profile`` (a :class:`Profile`, a shipped id, a path or a dict).

    ``events`` needs ``ppv_mm_s`` and ``ppv_kind`` (and ``distance_m`` for a distance table); the other canonical
    columns are used when present. Returns one row per event, same index, with the columns ``status``,
    ``limit_mm_s``, ``margin_mm_s`` (limit - PPV), ``margin_pct`` (of the limit), ``governing``, ``verification`` and
    ``reason``. ``attrs`` names what the frame was checked against: ``profile_id``, ``criterion``, ``citation``,
    ``edition``, ``quantity``, ``frequency_methods``, ``verification`` and ``reference``. See the module docstring for
    the rules.
    """
    profile, c, rows = _prepare(events, profile, criterion)
    out = pd.DataFrame([_verdict(row, c, profile.quantity) for row in rows], columns=_RESULT, index=events.index)
    out = out.astype({"limit_mm_s": float, "margin_mm_s": float, "margin_pct": float})
    source = profile.source if isinstance(profile.source, dict) else {}
    out.attrs = {"profile_id": profile.profile_id, "criterion": c.criterion_id,
                 "citation": source.get("citation", ""), "edition": source.get("edition", ""),
                 "quantity": profile.quantity, "frequency_methods": profile.frequency_methods,
                 "verification": c.verification, "reference": c.reference}
    return out


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
