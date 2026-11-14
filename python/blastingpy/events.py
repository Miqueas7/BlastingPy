"""Events table: read the blast monitoring spreadsheet (CSV, Excel or DataFrame) into canonical SI columns.

The result has the columns that were mapped, in this order: ``distance_m``, ``charge_kg``, ``ppv_mm_s`` (required),
``ppv_kind`` (``"component_max"``, ``"vector_sum"`` or ``"vertical"``; from the ``ppv_kind`` argument or a column) and the
optional ``frequency_hz``, ``date``, ``point``, ``ppv_t_mm_s``, ``ppv_v_mm_s``, ``ppv_l_mm_s``, ``freq_t_hz``,
``freq_v_hz``, ``freq_l_hz``, ``pvs_mm_s``, ``airblast_pa`` (peak overpressure, linear) and ``airblast_freq_hz``. Other
columns of the file are dropped. Distances, charges, velocities and pressures are converted to m, kg, mm/s and Pa;
frequencies are Hz and never converted; ``date`` and ``point`` are kept as read.

Decimal mark, when ``decimal`` is None: in a ``;`` CSV it is the comma (a point is a thousands mark there and is
rejected); in a ``,`` CSV it is the point (a comma inside a quoted number is a thousands mark and is rejected); tab CSVs,
Excel cells stored as text and DataFrame string cells accept either mark, except the shape ``1,250`` / ``1.250`` (one to
three digits, a mark, three digits), which could be a decimal or a thousands mark and is rejected as ambiguous. Every
rejected cell names its column and row and asks for ``decimal=``.

Row handling: a row with every mapped cell empty is skipped silently; a row with an empty PPV is skipped and counted in a
single ``UserWarning``; a row with a PPV but no distance or charge is a ``ValueError``. Cells listed in ``na_values``
count as empty. Row numbers in errors are the ones the user sees in the sheet (header = 1), or ``row index N`` (position)
for a DataFrame source. The rule is the vibration one: a row without PPV is skipped even if it has an
overpressure. An airblast-only campaign is built as a DataFrame with ``distance_m``, ``charge_kg`` and ``airblast_pa``
(no ``read_events`` call).
"""

from __future__ import annotations

import csv
import importlib.resources
import io
import math
import os
import re
import warnings

import numpy as np
import pandas as pd

from ._blastingpy_core import _airblast as _core
from .units import to_si

__all__ = ["read_events", "write_events_template"]

_REQUIRED = ("distance_m", "charge_kg", "ppv_mm_s")
_NUMERIC_OPTIONAL = ("frequency_hz", "ppv_t_mm_s", "ppv_v_mm_s", "ppv_l_mm_s", "freq_t_hz", "freq_v_hz", "freq_l_hz",
                     "pvs_mm_s", "airblast_pa", "airblast_freq_hz")
# Orden de salida de las columnas.
_ORDER = ("distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz", "date", "point", "ppv_t_mm_s",
          "ppv_v_mm_s", "ppv_l_mm_s", "freq_t_hz", "freq_v_hz", "freq_l_hz", "pvs_mm_s", "airblast_pa", "airblast_freq_hz")
_KINDS = ("component_max", "vector_sum", "vertical")
# Magnitud de `units` -> (unidades aceptadas, unidad métrica); "mm" de to_si no sirve: se queda en mm, no pasa a m.
_UNITS = {"distance": ("m", "ft"), "charge": ("kg", "lb"), "velocity": ("mm/s", "in/s"),
          "pressure": ("Pa", "kPa", "psi", "dB")}
_IMPERIAL = {"distance": "ft", "charge": "lb", "velocity": "in/s", "pressure": "psi"}
# Magnitud de cada columna convertida a SI.
_QUANTITY = {"distance_m": "distance", "charge_kg": "charge", "ppv_mm_s": "velocity", "ppv_t_mm_s": "velocity",
             "ppv_v_mm_s": "velocity", "ppv_l_mm_s": "velocity", "pvs_mm_s": "velocity",
             "airblast_pa": "pressure"}
_NUMBER = {None: re.compile(r"^-?\d+([.,]\d+)?$"), ",": re.compile(r"^-?\d+(,\d+)?$"),
           ".": re.compile(r"^-?\d+(\.\d+)?$")}
# Opt-in de quien lee formas de onda (read_waveform_csv): los registradores escriben 1e-05; read_events lo sigue rechazando.
_NUMBER_SCI = {None: re.compile(r"^-?\d+([.,]\d+)?([eE][+-]?\d+)?$"), ",": re.compile(r"^-?\d+(,\d+)?([eE][+-]?\d+)?$"),
               ".": re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")}
# Donde se aceptan las dos marcas, "1,250" o "1.250" puede ser 1.25 o 1250: se rechaza y se pide decimal=.
_AMBIGUOUS = re.compile(r"^-?[1-9]\d{0,2}[.,]\d{3}$")
_PREFIX = "Domain error [BlastingPy]"
_IO_HINT = "openpyxl is required to read or write .xlsx files: pip install blastingpy[io]"


def _other_mark(text: str) -> str:
    """Why a number written with the other mark is rejected: thousands separators are never accepted."""
    mark = "," if "," in text else "."
    plain = f" (write {text.replace(mark, '')})" if _AMBIGUOUS.match(text) else ""
    return (f"thousands separators are not accepted{plain}; if the {'comma' if mark == ',' else 'point'} is the "
            f"decimal mark, pass decimal='{mark}'")


def _param_error(name: str, message: str) -> ValueError:
    return ValueError(f"{_PREFIX}: parameter '{name}' {message}")


def _require_openpyxl() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(_IO_HINT) from None


def _unit_names(units) -> dict[str, str]:
    if units == "metric":
        return {q: u[0] for q, u in _UNITS.items()}
    if units == "imperial":
        return dict(_IMPERIAL)
    if not isinstance(units, dict):
        raise _param_error("units", f"must be 'metric', 'imperial' or a dict with keys {sorted(_UNITS)}. "
                                    f"Received: {units!r}")
    names = {q: u[0] for q, u in _UNITS.items()}
    for quantity, unit in units.items():
        if quantity not in _UNITS:
            raise _param_error("units", f"has an unknown key {quantity!r}. Valid keys: {sorted(_UNITS)}")
        if unit not in _UNITS[quantity]:
            raise _param_error("units", f"has an unknown unit {unit!r} for {quantity!r}. "
                                        f"Valid units: {list(_UNITS[quantity])}")
        names[quantity] = unit
    return names


def _check_arguments(columns, ppv_kind, decimal) -> None:
    if not isinstance(columns, dict):
        raise _param_error("columns", "must be a dict {canonical name: column name in the file}")
    unknown = [k for k in columns if k not in _ORDER]
    if unknown:
        raise _param_error("columns", f"has an unknown key {unknown[0]!r}. Valid keys: {list(_ORDER)}")
    missing = [k for k in _REQUIRED if k not in columns]
    if missing:
        raise _param_error("columns", f"must map {missing[0]!r} (required: {list(_REQUIRED)})")
    if ppv_kind is not None and ppv_kind not in _KINDS:
        raise _param_error("ppv_kind", f"must be one of {list(_KINDS)}. Received: {ppv_kind!r}")
    if "ppv_kind" not in columns and ppv_kind is None:
        raise _param_error("ppv_kind", f"is required when 'columns' has no 'ppv_kind' column. One of {list(_KINDS)}")
    if "ppv_kind" in columns and ppv_kind is not None:
        raise _param_error("ppv_kind", f"is given as an argument ({ppv_kind!r}) and as the column "
                                       f"{columns['ppv_kind']!r}; pass only one of them")
    if decimal not in _NUMBER:
        raise _param_error("decimal", f"must be ',' or '.'. Received: {decimal!r}")


def _decode(path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8-sig")  # Excel escribe BOM
    except UnicodeDecodeError:
        return data.decode("cp1252")  # "CSV (delimitado por comas)" de Excel en español


def _read_csv(source, sep) -> tuple[pd.DataFrame, str]:
    """Parse the CSV as text; every non-blank line must have as many fields as the header."""
    text = _decode(source)
    if sep is None:
        header = text.splitlines()[0] if text.strip() else ""
        sep = max((";", ",", "\t"), key=header.count)
    lines = list(csv.reader(io.StringIO(text), delimiter=sep))
    names = lines[0] if lines else []
    rows = []
    for i, fields in enumerate(lines[1:]):
        if not fields:  # línea en blanco: fila vacía, pero cuenta para la numeración
            fields = [""] * len(names)
        elif len(fields) != len(names):
            raise ValueError(
                f"{_PREFIX}: row {i + 2} has {len(fields)} fields but the header has {len(names)}. Check for an "
                f"unquoted decimal comma in a comma-separated file, or pass sep= (detected: {sep!r})")
        rows.append(fields)
    return pd.DataFrame(rows, columns=names, dtype=object), sep


def _load(source, sheet, sep) -> tuple[pd.DataFrame, str | None]:
    """Return the table as objects with the user's column names (row ``i`` is position ``i``) and the CSV separator."""
    if isinstance(source, pd.DataFrame):
        return source.reset_index(drop=True), None
    if not isinstance(source, (str, os.PathLike)):
        raise _param_error("source", f"must be a path to a .csv or .xlsx file or a DataFrame. Received: {source!r}")
    suffix = os.path.splitext(os.fspath(source))[1].lower()
    if suffix == ".csv":
        return _read_csv(source, sep)
    if suffix == ".xlsx":
        _require_openpyxl()
        table = pd.read_excel(source, sheet_name=0 if sheet is None else sheet, dtype=object, keep_default_na=False)
        return table, None
    raise _param_error("source", f"must be a .csv or .xlsx file. Received: {os.fspath(source)!r}")


def _is_empty(value, na_values) -> bool:
    if isinstance(value, str):
        text = value.strip()
        return text == "" or text in na_values
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _number(value, decimal, scientific=False):
    """Return the value as a float, or None when it is not a number (``scientific=True`` accepts ``1e-05``)."""
    if isinstance(value, str):
        text = value.strip()
        return float(text.replace(",", ".")) if (_NUMBER_SCI if scientific else _NUMBER)[decimal].match(text) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_events(source, *, columns: dict[str, str], units: str | dict[str, str] = "metric",
                ppv_kind: str | None = None, sheet: str | int | None = None, sep: str | None = None,
                decimal: str | None = None, na_values: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read the events table from a ``.csv``/``.xlsx`` path or a DataFrame into canonical SI columns.

    ``columns`` maps canonical names (``distance_m``, ``charge_kg`` and ``ppv_mm_s`` are required) to the column names of
    the file. ``units`` is ``"metric"`` (m, kg, mm/s), ``"imperial"`` (ft, lb, in/s, psi) or a dict with any of the keys
    ``distance`` (``m``/``ft``), ``charge`` (``kg``/``lb``) and ``velocity`` (``mm/s``/``in/s``) and ``pressure`` (``Pa``/``kPa``/``psi``/``dB``, the last one a peak dB(L) level re 20
    uPa, any finite number); missing keys are metric (``"imperial"`` implies ``pressure="psi"``).
    ``ppv_kind`` applies to the whole table; give it either as this argument or as a mapped ``ppv_kind`` column, not
    both. ``sep`` (CSV, detected from the header line when ``None``) and ``decimal`` (``","`` or ``"."``) override the
    detection; with ``decimal=None`` the mark follows the rules in the module docstring. Excel numbers stored as text are
    converted. A mapped column whose name appears twice in the header is an error. See the module docstring for the row
    handling.
    """
    _check_arguments(columns, ppv_kind, decimal)
    unit_names = _unit_names(units)
    table, sep = _load(source, sheet, sep)
    if sep == ";" and decimal is None:
        decimal = ","  # con ";" el punto es marca de miles (1.250 = 1250): nunca se lee como decimal
    elif sep == "," and decimal is None:
        decimal = "."  # con "," la coma es marca de miles ("1,250" = 1250): nunca se lee como decimal
    for canonical, name in columns.items():
        if name not in table.columns:
            raise _param_error("columns", f"maps {canonical!r} to {name!r}, which is not in the file. "
                                          f"Available columns: {[str(c) for c in table.columns]}")
        if (table.columns == name).sum() > 1:
            raise _param_error("source", f"has the column {name!r} twice in its header; the column mapped to "
                                         f"{canonical!r} must be unique")
    names = {c: columns[c] for c in _ORDER if c in columns}
    na_values = tuple(na_values)
    is_file = not isinstance(source, pd.DataFrame)

    def where(i):
        return f"row {i + 2}" if is_file else f"row index {i}"

    def cell_error(canonical, i, problem):
        return ValueError(f"{_PREFIX}: column {names[canonical]!r} ({canonical}), {where(i)}: {problem}")

    cells = {c: list(table[n]) for c, n in names.items()}
    empty = {c: [_is_empty(v, na_values) for v in vals] for c, vals in cells.items()}
    keep, skipped = [], 0
    for i in range(len(table)):
        if all(empty[c][i] for c in names):
            continue
        if empty["ppv_mm_s"][i]:
            skipped += 1
            continue
        keep.append(i)
    if skipped:
        warnings.warn(f"{skipped} row{'s' if skipped > 1 else ''} skipped: empty ppv_mm_s", UserWarning, stacklevel=2)

    out = {}
    for canonical in _ORDER:
        if canonical == "ppv_kind" and canonical not in names:
            out[canonical] = [ppv_kind] * len(keep)
        elif canonical not in names:
            continue
        elif canonical in ("date", "point"):
            out[canonical] = [np.nan if empty[canonical][i] else cells[canonical][i] for i in keep]
        elif canonical == "ppv_kind":
            values = []
            for i in keep:
                value = None if empty[canonical][i] else str(cells[canonical][i]).strip()
                if value not in _KINDS:
                    raise cell_error(canonical, i, f"expected one of {list(_KINDS)}, received {cells[canonical][i]!r}")
                values.append(value)
            out[canonical] = values
        else:
            values = []
            for i in keep:
                if empty[canonical][i]:
                    if canonical in _REQUIRED:
                        raise cell_error(canonical, i, "the cell is empty but the row has a PPV")
                    values.append(np.nan)
                    continue
                cell = cells[canonical][i]
                text = cell.strip() if isinstance(cell, str) else None
                if decimal is None and text is not None and _AMBIGUOUS.match(text):
                    raise cell_error(canonical, i, f"{text!r} is ambiguous: {_other_mark(text)}")
                number = _number(cell, decimal)
                if number is None and text is not None and _NUMBER[None].match(text):  # la otra marca
                    raise cell_error(canonical, i, f"{text!r} is not a number with decimal={decimal!r}: "
                                                   f"{_other_mark(text)}")
                if number is None or not np.isfinite(number):
                    raise cell_error(canonical, i, f"expected a number, received {cells[canonical][i]!r}. "
                                                f"If the decimal mark is not what you expect, pass decimal=',' or '.'")
                is_db = canonical == "airblast_pa" and unit_names["pressure"] == "dB"
                if (canonical in _REQUIRED or (canonical == "airblast_pa" and not is_db)) and number <= 0:
                    raise cell_error(canonical, i, f"expected a positive value, received {cells[canonical][i]!r}")
                if is_db:
                    try:
                        number = _core.db_to_pa(number)
                    except ValueError:
                        number = math.nan
                    if not np.isfinite(number):
                        raise cell_error(canonical, i, f"the level {cells[canonical][i]!r} dB is out of range "
                                                       f"(its pressure in Pa is not finite)")
                values.append(number)
            array = np.array(values, dtype=float)
            quantity = _QUANTITY.get(canonical)
            if quantity and not (canonical == "airblast_pa" and unit_names["pressure"] == "dB"):
                array = to_si(array, unit_names[quantity])
            out[canonical] = array
    return pd.DataFrame(out, columns=[c for c in _ORDER if c in out])


def write_events_template(path) -> None:
    """Write an empty-of-real-data events table template (headers and two invented rows) to ``.csv`` or ``.xlsx``."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    template = importlib.resources.files("blastingpy").joinpath("templates", "events_template.csv")
    if suffix == ".csv":
        with open(path, "wb") as f:
            f.write(template.read_bytes())
    elif suffix == ".xlsx":
        _require_openpyxl()
        pd.read_csv(io.BytesIO(template.read_bytes())).to_excel(path, index=False)
    else:
        raise _param_error("path", f"must end in .csv or .xlsx. Received: {os.fspath(path)!r}")
