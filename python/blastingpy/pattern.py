"""Bench drill pattern: burden and spacing by several methods, the hole layout, the firing sequence and the charge.

Units are SI: lengths in m, hole diameter in mm, densities in kg/m3, UCS in MPa, delays in ms, firing times in s.

* :func:`burden_methods` puts the burden, spacing, stemming and subdrill of each method side by side. They are
  different methods, not a consensus: nothing here averages them or picks one. Every row is in the drilled-pattern
  frame (``burden_m`` x ``spacing_m`` is the grid you drill).
* :func:`layout` places the holes inside an outline, with rows parallel to the crest (the free face).
* :func:`sequence` adds the nominal firing times, row by row or in a V. You supply the delays: there is no default.
* :func:`charge` adds the charge per hole. The result has the columns of :data:`blastingpy.nearfield.BLAST_COLUMNS`
  and :data:`blastingpy.flyrock.HOLE_COLUMNS`, so it goes straight into ``nearfield.simulate_blast`` and
  ``flyrock.compare``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from ._blastingpy_core import _pattern as _pt
from .flyrock import linear_charge

BURDEN_METHODS = ("konya", "lopez_jimeno", "ash", "langefors")

__all__ = ["BURDEN_METHODS", "burden_methods", "layout", "sequence", "charge", "to_dxf"]

_INITIATIONS = {"delayed": _pt.Initiation.Delayed, "instantaneous": _pt.Initiation.Instantaneous}
_CLASSES = {"low_density": _pt.ExplosiveClass.LowDensity, "average": _pt.ExplosiveClass.Average,
            "dense": _pt.ExplosiveClass.Dense}
_ARRANGEMENTS = {"square": _pt.Arrangement.Square, "rectangular": _pt.Arrangement.Rectangular,
                 "staggered": _pt.Arrangement.Staggered}
_SCHEMES = {"row": _pt.Scheme.Row, "v": _pt.Scheme.V}
_SIDES = {"left": _pt.StartSide.Left, "right": _pt.StartSide.Right}
_STAGGERED_RATIO = 1.15  # S = 1,15·B por omisión en tresbolillo (tesis, vía López Jimeno; diseño §4.2)
_LANGEFORS = "not implemented: the secondary sources disagree; pending Langefors & Kihlström (1978)"
_NUMBERS = ("burden_m", "spacing_m", "stemming_m", "subdrill_m")


def _error(name: str, detail: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {detail}")


def _number(v) -> bool:
    """A finite real number (a bool is not one)."""
    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def _choice(name: str, value, options: dict):
    if not isinstance(value, str) or value not in options:
        raise _error(name, f"must be one of {list(options)}. Received: {value!r}")
    return options[value]


def _row(design) -> dict:
    """A table row: the core result, or NaN numbers and a note (a str) saying why there is none."""
    if isinstance(design, str):
        return {**dict.fromkeys(_NUMBERS, math.nan), "method_id": None, "verification": None, "source": None,
                "note": design}
    return {**{c: getattr(design, c) for c in _NUMBERS}, "method_id": design.info.method_id,
            "verification": design.info.verification.name, "source": design.info.source, "note": design.note}


def burden_methods(*, diameter_mm: float, bench_height_m: float, rock_density_kg_m3: float,
                   explosive_density_kg_m3: float, initiation: str, ucs_mpa: float | None = None,
                   explosive_class: str | None = None, konya_kd: float = 1.0, konya_ks: float = 1.0,
                   konya_kr: float = 1.0) -> pd.DataFrame:
    """Burden, spacing, stemming and subdrill (m) of each method in :data:`BURDEN_METHODS`, one row each.

    ``initiation`` is ``"delayed"`` or ``"instantaneous"`` and has no default: Konya's spacing and Ash's K_S depend on
    it. ``explosive_class`` (``"low_density"``, ``"average"`` or ``"dense"``: K_B = 25, 30, 35) is needed by Ash and
    ``ucs_mpa`` by López Jimeno. A method whose input is missing is not computed: its row has NaN numbers and a
    ``note`` saying what is missing (no rock property is ever defaulted). ``konya_kd``, ``konya_ks`` and ``konya_kr``
    are Konya's corrections (FHWA Tables 6.2-6.4); 1 is the published uncorrected formula, and the note says so.

    Columns: ``burden_m``, ``spacing_m``, ``stemming_m``, ``subdrill_m``, ``method_id``, ``verification``
    (``"Primary"`` or ``"Secondary"``), ``source`` (with pages) and ``note``; the index is ``method``. Every row is in
    the drilled-pattern frame: with delays, Konya and Ash give the drilled row distance as ``burden_m``.
    Langefors-Kihlström is listed but not implemented: the secondary sources disagree on the equation.
    """
    init = _choice("initiation", initiation, _INITIATIONS)
    cls = None if explosive_class is None else _choice("explosive_class", explosive_class, _CLASSES)
    designs = {
        "konya": _pt.konya(diameter_mm, bench_height_m, rock_density_kg_m3, explosive_density_kg_m3, init, konya_kd,
                           konya_ks, konya_kr),
        "lopez_jimeno": "needs ucs_mpa" if ucs_mpa is None else _pt.lopez_jimeno(diameter_mm, ucs_mpa),
        "ash": "needs explosive_class" if cls is None else _pt.ash(diameter_mm, cls, init),
        "langefors": _LANGEFORS,
    }
    return pd.DataFrame([_row(designs[m]) for m in BURDEN_METHODS], index=pd.Index(BURDEN_METHODS, name="method"))


def _points(name: str, values) -> list:
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise _error(name, "must be an (N, 2) array of numbers") from None
    if a.ndim != 2 or a.shape[1] != 2:
        raise _error(name, f"must be an (N, 2) array of numbers. Received shape: {a.shape}")
    return [_pt.Point(x, y) for x, y in a.tolist()]


def layout(outline, crest, *, burden_m: float, spacing_m: float | None = None, arrangement: str = "staggered",
           front_tolerance: float = 0.20) -> pd.DataFrame:
    """Holes of a bench pattern inside ``outline``, one row per hole.

    ``outline`` (closed polygon, either orientation, closed or open) and ``crest`` (polyline of the free face) are
    (N, 2) arrays of plan-view coordinates in m (UTM or local). Rows are parallel to the line from the first to the
    last crest point, the first one ``burden_m`` inside, the others every ``burden_m`` back. ``arrangement`` is
    ``"square"`` (needs ``spacing_m == burden_m``), ``"rectangular"`` or ``"staggered"`` (odd rows shifted S/2). Only
    the staggered pattern has a default spacing: 1.15 ``burden_m``.

    A hole is kept when its collar is strictly inside the outline. At each end of a row, if the free length to the
    outline is more than S/2, an edge hole goes at its midpoint (``edge`` True).

    Columns: ``hole_id`` (1, 2, ... from the crest), ``row`` (0 = front row), ``x_m``, ``y_m``, ``along_m`` (position
    along the rows), ``edge``, ``burden_m``, ``spacing_m`` and ``crest_distance_m``. ``burden_m`` is the design burden
    on every hole; ``crest_distance_m`` is the distance from each front-row hole to the real crest (its real burden),
    NaN in the other rows. For flyrock, put ``crest_distance_m`` in place of ``burden_m`` in the front row before
    ``flyrock.compare``: face-burst range grows as the burden shrinks, so the design burden underestimates it where
    the crest comes closer. Warnings: the front-row holes whose distance to the real crest differs from ``burden_m``
    by more than ``front_tolerance`` (a fraction), and a rear gap larger than ``burden_m`` between the last row
    and the outline (the outline is the limit of the rock to blast, so a gap of one burden is normal; no partial row
    is added there). Raises if no hole fits, and before generating anything if the
    pattern would pass 1,000,000 holes or rows (outline area / (B S) or depth / B): usually an outline in mm.
    """
    arr = _choice("arrangement", arrangement, _ARRANGEMENTS)
    if not (_number(burden_m) and burden_m > 0):
        raise _error("burden_m", f"must be finite and > 0. Received: {burden_m!r}")
    if spacing_m is None:
        if arrangement != "staggered":
            raise _error("spacing_m", f"is required for a {arrangement} pattern (only staggered has a default)")
        spacing_m = _STAGGERED_RATIO * burden_m
    res = _pt.layout(_points("outline", outline), _points("crest", crest), burden_m, spacing_m, arr, front_tolerance)
    hs = res.holes  # cada lectura de res.holes copia el vector entero
    holes = pd.DataFrame({"hole_id": [h.id for h in hs], "row": [h.row for h in hs], "x_m": [h.x for h in hs],
                          "y_m": [h.y for h in hs], "along_m": [h.along_m for h in hs], "edge": [h.edge for h in hs]})
    holes["burden_m"] = float(burden_m)
    holes["spacing_m"] = float(spacing_m)
    holes["crest_distance_m"] = res.crest_distance_m
    if res.front_off_ids:
        warnings.warn(f"front row: holes {list(res.front_off_ids)} are more than {front_tolerance:.0%} off the design "
                      f"burden ({burden_m:g} m) from the real crest", UserWarning, stacklevel=2)
    if res.rear_gap_m > burden_m * (1 + 1e-9):  # el contorno es el límite del material: un hueco de B es lo normal
        warnings.warn(f"rear gap: {res.rear_gap_m:.2f} m between the last row and the back of the outline, more than "
                      "burden_m; no partial row is added there", UserWarning, stacklevel=2)
    return holes


def _ints(holes: pd.DataFrame, name: str) -> list[int]:
    try:
        x = pd.to_numeric(holes[name], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise _error("holes", f"column {name!r} must be numeric") from None
    bad = ~(np.isfinite(x) & (x == np.round(x)))
    if bad.any():
        raise _error("holes", f"column {name!r}, rows {list(holes.index[bad])}: must be whole numbers")
    return [int(v) for v in x]


def sequence(holes: pd.DataFrame, *, scheme: str, hole_delay_ms: float, row_delay_ms: float, start: str = "left",
             apex_hole: int | None = None) -> pd.DataFrame:
    """Nominal firing times; returns a copy of ``holes`` plus ``delay_ms`` and ``firing_time_s`` (= delay_ms / 1000).

    ``holes`` needs the columns ``hole_id``, ``row`` (0 = front row, numbered 0, 1, 2... without gaps) and
    ``along_m``, as :func:`layout` gives them. ``scheme`` is ``"row"`` (t = r row_delay + i hole_delay, i counted from
    ``start``, ``"left"`` or ``"right"`` seen from the free face) or ``"v"`` (t = r row_delay + |i - i_apex|
    hole_delay; the apex is ``apex_hole``, a front-row hole id, by default the central one, the lower of the two for
    an even count). The delays have no default: you supply them. Holes symmetric about the apex share a nominal time
    on purpose; the charge fired within 8 ms is evaluated by ``nearfield.simulate_blast`` (``max_charge_8ms_kg``).
    """
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    miss = [c for c in ("hole_id", "row", "along_m") if c not in holes.columns]
    if miss:
        raise _error("holes", f"is missing the column {miss[0]!r}. Required columns: ['hole_id', 'row', 'along_m']")
    sch = _choice("scheme", scheme, _SCHEMES)
    side = _choice("start", start, _SIDES)
    if apex_hole is not None and not (isinstance(apex_hole, (int, np.integer)) and apex_hole >= 0):
        raise _error("apex_hole", f"must be the id of a front-row hole (an integer >= 0). Received: {apex_hole!r}")
    try:
        along = pd.to_numeric(holes["along_m"], errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError):
        raise _error("holes", "column 'along_m' must be numeric") from None
    core_holes = [_pt.Hole(i, r, 0.0, 0.0, a, False)
                  for i, r, a in zip(_ints(holes, "hole_id"), _ints(holes, "row"), along.tolist())]
    t = _pt.firing_times_ms(core_holes, sch, hole_delay_ms, row_delay_ms, side, -1 if apex_hole is None else apex_hole)
    out = holes.copy()
    out["delay_ms"] = t
    out["firing_time_s"] = out["delay_ms"] / 1000.0
    return out


def charge(holes: pd.DataFrame, *, bench_height_m: float, subdrill_m: float, stemming_m: float, diameter_mm: float,
           explosive_density_kg_m3: float) -> pd.DataFrame:
    """Charge per hole; returns a copy of ``holes`` plus ``diameter_mm``, ``stemming_m``, ``linear_charge_kg_m`` and
    ``charge_kg``.

    The charge column is H + J - T (``bench_height_m + subdrill_m - stemming_m``), fully charged at
    ``flyrock.linear_charge(diameter_mm, explosive_density_kg_m3)`` (no decking or decoupling). Raises when the column
    would be zero or negative.

    The result goes into ``flyrock.compare``, which reads ``burden_m``. From :func:`layout` that is the design burden
    on every hole; in the front row the real burden is ``crest_distance_m``, so replace it there first, e.g.
    ``front = holes["row"] == 0; holes.loc[front, "burden_m"] = holes.loc[front, "crest_distance_m"]``.
    """
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    if not (_number(bench_height_m) and bench_height_m > 0):
        raise _error("bench_height_m", f"must be finite and > 0. Received: {bench_height_m!r}")
    for name, v in (("subdrill_m", subdrill_m), ("stemming_m", stemming_m)):
        if not (_number(v) and v >= 0):
            raise _error(name, f"must be finite and >= 0. Received: {v!r}")
    column_m = bench_height_m + subdrill_m - stemming_m
    if column_m <= 0:
        raise _error("stemming_m", f"leaves no charge column: bench_height_m + subdrill_m - stemming_m = {column_m:g} m")
    lin = linear_charge(diameter_mm, explosive_density_kg_m3)
    out = holes.copy()
    out["diameter_mm"] = float(diameter_mm)
    out["stemming_m"] = float(stemming_m)
    out["linear_charge_kg_m"] = lin
    out["charge_kg"] = lin * column_m
    return out


def _points_xy(name: str, values) -> list:
    return [(p.x, p.y) for p in _points(name, values)]


def to_dxf(holes: pd.DataFrame, path, *, outline, crest, diameter_mm: float | None = None, layers_by: str = "row",
           text_height_m: float = 0.5, layer_names: dict | None = None) -> None:
    """Write the pattern to a DXF R2010 file, in the same coordinates as ``holes`` (m).

    Layers: ``CONTORNO`` (outline, closed), ``CRESTA`` (crest), one per row ``FILA_01``... (or per nominal delay
    ``RETARDO_0017``... with ``layers_by="delay"``, needs ``delay_ms``), ``TEXTO`` (``"id / delay ms"``, or only the
    id without a sequence) and ``AMARRE`` (the tie-in, only with a sequence: one polyline per row joining the holes by
    increasing ``firing_time_s``; for a V, two per row, from the apex outward on each side). ``layer_names`` renames
    them: keys ``outline``, ``crest``, ``text``, ``tie`` and ``prefix`` (the ``FILA_`` / ``RETARDO_`` prefix). Each
    hole is a circle of ``diameter_mm`` (the column of that name, else the argument; an error if neither). Needs
    ``pip install blastingpy[dxf]``.
    """
    from .io import _require_ezdxf
    ezdxf = _require_ezdxf()
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    miss = [c for c in ("hole_id", "row", "x_m", "y_m") if c not in holes.columns]
    if miss:
        raise _error("holes", f"is missing the column {miss[0]!r}")
    if layers_by not in ("row", "delay"):
        raise _error("layers_by", f"must be 'row' or 'delay'. Received: {layers_by!r}")
    seq = "delay_ms" in holes.columns
    if layers_by == "delay" and not seq:
        raise _error("layers_by", "'delay' needs the column 'delay_ms' (run sequence first)")
    if not (_number(text_height_m) and text_height_m > 0):
        raise _error("text_height_m", f"must be finite and > 0. Received: {text_height_m!r}")
    names = {"outline": "CONTORNO", "crest": "CRESTA", "text": "TEXTO", "tie": "AMARRE",
             "prefix": "FILA_" if layers_by == "row" else "RETARDO_", **(layer_names or {})}
    if "diameter_mm" in holes.columns:
        diam = holes["diameter_mm"].to_numpy(dtype=float)
    elif diameter_mm is not None and _number(diameter_mm) and diameter_mm > 0:
        diam = np.full(len(holes), float(diameter_mm))
    else:
        raise _error("diameter_mm", "is required when holes has no 'diameter_mm' column (finite and > 0)")
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_lwpolyline(_points_xy("outline", outline), close=True, dxfattribs={"layer": names["outline"]})
    msp.add_lwpolyline(_points_xy("crest", crest), dxfattribs={"layer": names["crest"]})
    for (_, h), d in zip(holes.iterrows(), diam.tolist()):
        layer = (f"{names['prefix']}{int(h['row']) + 1:02d}" if layers_by == "row"
                 else f"{names['prefix']}{int(round(h['delay_ms'])):04d}")
        x, y = float(h["x_m"]), float(h["y_m"])
        msp.add_circle((x, y), d / 2000.0, dxfattribs={"layer": layer})
        label = f"{int(h['hole_id'])} / {h['delay_ms']:g} ms" if seq else f"{int(h['hole_id'])}"
        msp.add_text(label, height=text_height_m, dxfattribs={"layer": names["text"], "insert": (x + d / 2000.0, y)})
    if seq and "firing_time_s" in holes.columns:
        for _, g in holes.groupby("row"):
            # El DXF no sabe el esquema: si el primero en salir no está en un extremo de la fila, es una V.
            by_along = g.sort_values("along_m", kind="stable") if "along_m" in g.columns else None
            apex = None if by_along is None else int(np.argmin(by_along["firing_time_s"].to_numpy()))
            if apex is not None and 0 < apex < len(g) - 1:
                branches = [by_along.iloc[apex::-1], by_along.iloc[apex:]]
            else:
                branches = [g.sort_values("firing_time_s", kind="stable")]
            for b in branches:
                if len(b) > 1:
                    msp.add_lwpolyline(list(zip(b["x_m"], b["y_m"])), dxfattribs={"layer": names["tie"]})
    # ezdxf does not create the layer table entries by itself; sorted so the same pattern gives the same file
    for name in sorted({e.dxf.layer for e in msp}):
        if name not in doc.layers:
            doc.layers.add(name)
    doc.saveas(path)
