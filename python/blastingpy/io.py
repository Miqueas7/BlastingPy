"""Seismograph records: the immutable :class:`Record`, unit handling and file readers.

A :class:`Record` holds one blast record in SI units: geophone channels in mm/s, the
microphone in Pa and time in s. Channel names are canonical (``"transverse"``,
``"vertical"``, ``"longitudinal"``, ``"mic"``). Every reader of this module builds its
``Record`` through :func:`_make_record`, so validation and unit conversion live in one place.

The sample rate is never guessed: it must come from the file or from the caller.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from types import MappingProxyType

import numpy as np
import pandas as pd

from ._blastingpy_core import _units
from .events import _AMBIGUOUS, _NUMBER, _decode, _number, _other_mark, _param_error, _read_csv

__all__ = ["Record", "from_arrays", "infer_units", "read_instantel_ascii", "read_instantel_histogram",
           "read_waveform_csv", "read_obspy", "read_dxf_outline"]

CHANNEL_NAMES = ("transverse", "vertical", "longitudinal", "mic")
SOURCES = ("arrays", "instantel-thor", "instantel-blastware-table", "instantel-blastware-blocks",
           "instantel-minimal", "csv", "obspy")
FORMAT_EVIDENCE = ("user", "real-file", "third-party-parser", "manual-only")

MM_PER_IN = 25.4
PA_PER_PSI = _units.pa_per_psi

# Alias (minúsculas, sin espacios) -> nombre canónico. Los sufijos 1 son los nombres de la primera
# unidad de los equipos de 8 canales.
_ALIASES = {
    "transverse": "transverse", "tran": "transverse", "tran1": "transverse",
    "vertical": "vertical", "vert": "vertical", "vert1": "vertical",
    "longitudinal": "longitudinal", "long": "longitudinal", "long1": "longitudinal",
    "mic": "mic", "micl": "mic", "mic1": "mic", "micl1": "mic",
}
_SECOND_SENSOR = {"tran2", "vert2", "long2", "micl2", "mic2"}
_A_WEIGHTED = {"mica", "mica1"}

_GEOPHONE_UNITS = {"mm/s": 1.0, "in/s": MM_PER_IN}
_MIC_UNITS = {"pa": 1.0, "psi": PA_PER_PSI, "db": None}  # dB no es una forma de onda de presión

# Resoluciones del sensor (FUENTES_SPRINT2.md, «Resolución y rangos», [V1]): geófono 0,00788 mm/s
# (0,00031 in/s) en Micromate y Minimate Pro; micrófono lineal 0,0156 Pa (Micromate) y 0,0155 Pa
# (Minimate Pro), ambos = 0,015625 Pa, cubiertos por la tolerancia de ±3 %.
_RES_TOLERANCE = 0.03
# Tolerancia de la frecuencia de muestreo de un archivo frente a la que da el usuario (elección de diseño, R1): SAC
# redondea el paso al microsegundo, y a 1024 Hz eso es 0,05 %.
_RATE_TOLERANCE = 1e-3
# Redondeo del ajuste de una columna de tiempo (R2, elección de diseño): por debajo de 1e-6 relativo en fs y de 1e-3
# periodos de muestreo en el predisparo (a 1024 sps, ~1 µs, la resolución de tiempos con 6 decimales) la diferencia
# no se ve en el archivo y no se avisa.
_SNAP = 1e-6
_SNAP_SAMPLE = 1e-3
_RES_GEOPHONE = (("mm/s", 0.00788), ("in/s", 0.00031))
_RES_MIC = (("Pa", 0.0156),)


def _normalize_name(name) -> str:
    return "".join(str(name).split()).lower()


def _smallest_nonzero(arrays) -> float | None:
    values = np.concatenate([np.abs(np.asarray(a, dtype=float)).ravel() for a in arrays])
    values = values[values > 0]
    return float(values.min()) if values.size else None


def _match_resolution(smallest, table):
    if smallest is None:
        return None
    for unit, res in table:
        if abs(smallest - res) <= _RES_TOLERANCE * res:
            return unit, res
    return None


def infer_units(channels: Mapping[str, np.ndarray]) -> tuple[dict[str, str], list[str]]:
    """Infer the raw units of canonical channels from the sensor resolution.

    The smallest nonzero ``|x|`` of the geophone channels (pooled) must equal 0.00788
    (``mm/s``) or 0.00031 (``in/s``) within 3 %; the microphone's must equal 0.0156 (``Pa``).
    It is a match against the resolution itself, not a multiple of it. Returns
    ``(units, warnings)``; every inference adds a warning. If a present kind does not match,
    raises ``ValueError`` asking for ``units=``.
    """
    units: dict[str, str] = {}
    warns: list[str] = []
    geo = [v for k, v in channels.items() if k != "mic"]
    if geo:
        found = _match_resolution(_smallest_nonzero(geo), _RES_GEOPHONE)
        if found is None:
            raise _param_error("units", "cannot be inferred for the geophone channels: the smallest nonzero "
                               "value does not match 0.00788 mm/s or 0.00031 in/s; pass units=")
        units["geophone"] = found[0]
        warns.append(f"geophone units inferred as {found[0]} from the sensor resolution {found[1]} {found[0]} "
                     "(the rule holds for Micromate / Minimate Pro resolutions)")
    if "mic" in channels:
        found = _match_resolution(_smallest_nonzero([channels["mic"]]), _RES_MIC)
        if found is None:
            raise _param_error("units", "cannot be inferred for the mic channel: the smallest nonzero "
                               "value does not match 0.0156 Pa; pass units=")
        units["mic"] = found[0]
        warns.append(f"mic units inferred as {found[0]} from the sensor resolution {found[1]} {found[0]} "
                     "(the rule holds for Micromate / Minimate Pro resolutions)")
    return units, warns


def _canonical_channels(channels) -> tuple[dict[str, object], list[str]]:
    if not isinstance(channels, Mapping):
        raise _param_error("channels", "must be a mapping of channel name to samples")
    kept: dict[str, object] = {}
    warns: list[str] = []
    for name, data in channels.items():
        key = _normalize_name(name)
        if key in _A_WEIGHTED:
            warns.append(f"channel '{name}' dropped: A-weighted sound level is not a pressure waveform")
            continue
        if key in _SECOND_SENSOR:
            raise _param_error("channels", f"has '{name}': a second geophone or microphone is not supported "
                               "in this version")
        canon = _ALIASES.get(key)
        if canon is None:
            raise _param_error("channels", f"has unknown channel name '{name}'; found "
                               f"{[str(n) for n in channels]}, accepted: transverse (Tran), vertical (Vert), "
                               "longitudinal (Long), mic (MicL, Mic)")
        if canon in kept:
            raise _param_error("channels", f"has two names for the same channel '{canon}'")
        kept[canon] = data
    return kept, warns


def _check_units(units) -> dict[str, str]:
    if units is None:
        return {}
    if not isinstance(units, Mapping):
        raise _param_error("units", "must be a mapping with keys 'geophone' and 'mic'")
    clean: dict[str, str] = {}
    for key, value in units.items():
        table = {"geophone": _GEOPHONE_UNITS, "mic": _MIC_UNITS}.get(key)
        if table is None:
            raise _param_error("units", f"has unknown key '{key}'; use 'geophone' and 'mic'")
        text = str(value).strip().rstrip(".").strip().lower()
        if text not in table:
            raise _param_error("units", f"has unsupported {key} unit '{value}'; accepted: "
                               + ", ".join({"geophone": ("mm/s", "in/s"), "mic": ("Pa", "psi", "dB")}[key]))
        clean[key] = text
    return clean


def _check_array(name, data) -> np.ndarray:
    try:
        arr = np.array(data, dtype=float)  # copia: el arreglo del usuario no se toca
    except (TypeError, ValueError):
        raise _param_error("channels", f"channel '{name}' must be numeric") from None
    if arr.ndim != 1:
        raise _param_error("channels", f"channel '{name}' must be one-dimensional")
    if arr.size < 2:
        raise _param_error("channels", f"channel '{name}' needs at least 2 samples")
    if not np.all(np.isfinite(arr)):
        raise _param_error("channels", f"channel '{name}' has non-finite values")
    return arr


@dataclass(frozen=True, eq=False)
class Record:
    """One seismograph record, immutable, in SI units.

    ``channels`` maps canonical names to read-only arrays (geophones in mm/s, ``"mic"`` in Pa).
    ``start`` is the start of the record as given by the source: the event (trigger) time for
    Instantel exports, the first sample's time for ObsPy, whatever the caller passed for arrays,
    and None when the source gives none. ``source`` and ``format_evidence`` say how the record was
    built and how far the format is validated; ``warnings`` collects everything the builder had to
    assume. A record can be pickled and deep-copied; the copy is frozen and read-only as well.
    """

    channels: Mapping[str, np.ndarray]
    sample_rate_hz: float
    pretrigger_s: float | None
    start: datetime | None
    metadata: Mapping[str, str]
    source: str
    format_evidence: str
    warnings: tuple[str, ...]

    def __reduce__(self):
        # MappingProxyType no se serializa: se guarda como dict y se vuelve a congelar al reconstruir.
        return (_rebuild_record, (dict(self.channels), self.sample_rate_hz, self.pretrigger_s, self.start,
                                  dict(self.metadata), self.source, self.format_evidence, self.warnings))

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.channels.values())).size)

    def time_s(self) -> np.ndarray:
        """Time axis in seconds: ``i / fs - pretrigger`` (``i / fs`` without pretrigger)."""
        t = np.arange(self.n_samples) / self.sample_rate_hz
        return t - self.pretrigger_s if self.pretrigger_s is not None else t


def _rebuild_record(channels, sample_rate_hz, pretrigger_s, start, metadata, source, format_evidence,
                    warnings) -> Record:
    """Inverse of ``Record.__reduce__``: freeze the plain dicts again, with read-only array copies."""
    frozen = {}
    for name, data in channels.items():
        arr = np.array(data, dtype=float)
        arr.flags.writeable = False
        frozen[name] = arr
    return Record(channels=MappingProxyType(frozen), sample_rate_hz=sample_rate_hz, pretrigger_s=pretrigger_s,
                  start=start, metadata=MappingProxyType(dict(metadata)), source=source,
                  format_evidence=format_evidence, warnings=tuple(warnings))


def _make_record(channels, sample_rate_hz, *, units, pretrigger_s, start, metadata, source,
                 format_evidence, warnings) -> Record:
    """Validate, rename, convert to SI and freeze. Single constructor for every reader.

    ``warnings`` are the caller's own; they go first, followed by the ones added here.
    """
    if source not in SOURCES:
        raise _param_error("source", f"must be one of {', '.join(SOURCES)}")
    if format_evidence not in FORMAT_EVIDENCE:
        raise _param_error("format_evidence", f"must be one of {', '.join(FORMAT_EVIDENCE)}")
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, (int, float, np.number)) \
            or not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise _param_error("sample_rate_hz", "must be a finite number > 0")
    if pretrigger_s is not None and (isinstance(pretrigger_s, bool) or not math.isfinite(pretrigger_s)
                                     or pretrigger_s < 0):
        raise _param_error("pretrigger_s", "must be a finite number >= 0 or None")
    if start is not None and not isinstance(start, datetime):
        raise _param_error("start", "must be a datetime or None")
    meta = {} if metadata is None else {str(k): str(v) for k, v in dict(metadata).items()}
    unit_map = _check_units(units)

    warns = list(warnings)
    kept, dropped = _canonical_channels(channels)
    warns += dropped
    if unit_map.get("mic") == "db" and "mic" in kept:
        del kept["mic"]
        warns.append("mic channel dropped: units 'dB' are a sound level, not a pressure waveform")
    if not kept:
        raise _param_error("channels", "has no usable channel left")
    arrays = {name: _check_array(name, data) for name, data in kept.items()}
    if len({a.size for a in arrays.values()}) > 1:
        raise _param_error("channels", "must all have the same length; got "
                           + ", ".join(f"{k}={a.size}" for k, a in arrays.items()))

    present = {"mic" if k == "mic" else "geophone" for k in arrays}
    missing = present - unit_map.keys()
    if missing:
        subset = {k: v for k, v in arrays.items() if ("mic" if k == "mic" else "geophone") in missing}
        inferred, extra = infer_units(subset)
        unit_map.update({k: v.lower() for k, v in inferred.items()})
        warns += extra

    out = {}
    for name in CHANNEL_NAMES:
        if name not in arrays:
            continue
        kind = "mic" if name == "mic" else "geophone"
        factor = (_MIC_UNITS if kind == "mic" else _GEOPHONE_UNITS)[unit_map[kind]]
        data = arrays[name] * factor if factor != 1.0 else arrays[name]
        data.flags.writeable = False
        out[name] = data

    n = next(iter(out.values())).size
    if pretrigger_s is None:
        warns.append("no pretrigger given: time starts at 0")
    elif pretrigger_s >= n / sample_rate_hz:
        warns.append(f"pretrigger_s = {pretrigger_s:g} s is not shorter than the record ({n} samples / "
                     f"{sample_rate_hz:g} Hz = {n / sample_rate_hz:g} s): every sample is before the trigger")
    return Record(channels=MappingProxyType(out), sample_rate_hz=float(sample_rate_hz),
                  pretrigger_s=None if pretrigger_s is None else float(pretrigger_s), start=start,
                  metadata=MappingProxyType(meta), source=source, format_evidence=format_evidence,
                  warnings=tuple(warns))


def from_arrays(channels: Mapping[str, object], sample_rate_hz: float, *,
                units: Mapping[str, str] | None = None, pretrigger_s: float | None = None,
                start: datetime | None = None, metadata: Mapping[str, str] | None = None) -> Record:
    """Build a :class:`Record` from arrays.

    ``channels`` uses the canonical names or the instrument aliases (``Tran``, ``Vert``,
    ``Long``, ``MicL``). ``units`` is ``{"geophone": "mm/s" | "in/s", "mic": "Pa" | "psi"}``;
    a kind left out is inferred from the sensor resolution (see :func:`infer_units`). A ``MicA``
    channel or ``dB`` microphone units drop the mic channel with a warning. Values are converted
    to mm/s and Pa exactly once; the caller's arrays are copied, never modified.
    """
    return _make_record(channels, sample_rate_hz, units=units, pretrigger_s=pretrigger_s, start=start,
                        metadata=metadata, source="arrays", format_evidence="user", warnings=[])


# --------------------------------------------------------------------------- texto tabular
# Ayudantes pequeños de lectura de texto: los usa read_instantel_ascii y los reusará read_waveform_csv.

def _read_lines(path, strip: bool = True) -> list[tuple[int, str]]:
    """Non-empty lines of a text file as ``(1-based line number, text)``; BOM and CRLF accepted.

    With ``strip=False`` the text keeps its leading and trailing blanks (a leading tab is an empty first field).
    Lines end only at ``\\n`` (a trailing ``\\r`` is dropped), so U+0085 or U+2028 inside a line never splits it.
    """
    lines = (line[:-1] if line.endswith("\r") else line for line in _decode(path).split("\n"))
    return [(i, line.strip() if strip else line) for i, line in enumerate(lines, start=1) if line.strip()]


def _guess_sep(line: str) -> str | None:
    """Field separator of a line: tab, else comma, else runs of spaces (``None``)."""
    if "\t" in line:
        return "\t"
    return "," if "," in line else None


def _split_fields(line: str, sep: str | None) -> list[str]:
    return line.split() if sep is None else [f.strip() for f in line.split(sep)]


def _check_decimal(decimal, sep: str | None) -> str:
    if decimal is None:
        return "."
    if decimal not in (".", ","):
        raise _param_error("decimal", f"must be '.' or ',' (got {decimal!r})")
    if decimal == "," and sep == ",":
        raise _param_error("decimal", "cannot be ',' when the fields are separated by commas")
    return decimal


def _looks_numeric(text: str) -> bool:
    try:
        float(text.replace(",", "."))
    except ValueError:
        return False
    return True


def _parse_number(text: str, decimal: str, lineno: int) -> float:
    """A data value, checked against the number patterns of ``events`` (scientific notation accepted).

    With ``decimal=","`` a point-decimal number is accepted too (ruling R11).
    """
    if decimal == "." and "," in text:
        raise _param_error("decimal", f"is '.' but line {lineno} has a comma inside the number '{text}'; "
                           "if the file uses a decimal comma, pass decimal=\",\"")
    number = _number(text, None if decimal == "," else ".", scientific=True)
    if number is None or not math.isfinite(number):
        raise _param_error("path", f"has a value that is not a finite number at line {lineno}: '{text}'")
    return number


def _parse_table(rows, n_fields: int, sep: str | None, decimal: str) -> list[np.ndarray]:
    """Parse data rows (one sample per row) into one array per column; every row needs ``n_fields`` fields."""
    if not rows:
        raise _param_error("path", "has no data rows after the channel names")
    data = []
    for lineno, line in rows:
        fields = _split_fields(line, sep)
        if len(fields) != n_fields:
            raise _param_error("path", f"has {len(fields)} fields at line {lineno}, expected {n_fields} "
                               "(one per channel)")
        data.append([_parse_number(f, decimal, lineno) for f in fields])
    return list(np.array(data, dtype=float).T)


# --------------------------------------------------------------------------- ASCII de Instantel
# Formatos y evidencia: docs/investigacion/FUENTES_SPRINT2.md, «Formatos de exportación» (A, B, C).

# Solo claves propias de THOR (README) cuentan para detectarlo: las genéricas (fs, unidades, fecha, hora) también
# pueden venir en un encabezado de Blastware. Las genéricas igual se leen en un archivo THOR. Con comillas, la clave
# tiene que estar escrita como en el README (sin quitar espacios ni guiones): «Geo Range» es de Blastware (R17).
_THOR_README_KEYS = ("EventType", "Pre-triggerLength", "RecordStopMode", "AutoStopTime", "GeoTriggerLevel",
                     "MicTriggerLevel", "MicTriggerLeveldB", "GeoRange", "MicRange")
_THOR_LITERAL_KEYS = frozenset(k.lower() for k in _THOR_README_KEYS)
_THOR_SPECIFIC_KEYS = frozenset(re.sub(r"[\s\-_]", "", k).lower() for k in _THOR_README_KEYS)
_KEY_VALUE = re.compile(r'^"?\s*([^":]+?)\s*:\s*(.*?)\s*"?$')
# El número completo (con coma o punto) y sin dígitos pegados a la izquierda: «1,024 sps» no se lee como 24.
_SPS = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*sps\b", re.IGNORECASE)
_SEC = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*sec\b", re.IGNORECASE)
_GEO_UNIT = re.compile(r"(?<![a-z/])(mm/s|in/s)(?![a-z/])", re.IGNORECASE)
_MIC_UNIT = re.compile(r"(?<![a-z/])(pa|psi|db)(?![a-z/])", re.IGNORECASE)
_NOT_UNIT_LINE = re.compile(r"trigger|level|range", re.IGNORECASE)
_TIME = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?::(\d{1,3}))?(?::(\d{1,3}))?\s*([AP]M)?$", re.IGNORECASE)
_DEFAULT_BLOCK_ORDER = ("Tran", "Vert", "Long", "MicL")
_RENAMED_HINT = ("; if the channels were renamed, read the file with "
                 "read_waveform_csv(path, columns=..., units=..., sep=\"\\t\")")
_THOR_WARNING = "THOR ASCII layout taken from a third-party parser, not validated against a real file"
_BLASTWARE_WARNING = "Blastware ASCII layout built from the manual only, not validated against a real file"


@dataclass
class _Parsed:
    names: list[str]
    columns: list[np.ndarray]
    source: str
    format_evidence: str
    sample_rate_hz: float | None = None
    units: dict[str, str] = field(default_factory=dict)  # declaradas por el archivo (THOR): chocan con units=
    unit_hints: dict[str, str] = field(default_factory=dict)  # sugeridas (Blastware): units= manda, con aviso
    pretrigger_s: float | None = None
    start: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize_key(key: str) -> str:
    return re.sub(r"[\s\-_]", "", key).lower()


def _is_channel_token(token: str) -> bool:
    key = _normalize_name(token)
    return key in _ALIASES or key in _A_WEIGHTED or key in _SECOND_SENSOR


def _quoted_header_end(lines) -> int:
    """Index of the first line that is not a double-quoted header line."""
    k = 0
    while k < len(lines) and lines[k][1].startswith('"'):
        k += 1
    return k


def _has_blocks_body(lines) -> bool:
    """True when the first line after the quoted header is a comma-separated list of numbers (no row of names)."""
    k = _quoted_header_end(lines)
    return k < len(lines) and all(_looks_numeric(f) for f in _split_fields(lines[k][1], ","))


def _is_thor(lines) -> bool:
    """THOR detection, ruling R17: marker; else README keys spelled literally, or unquoted keys normalized.

    Without the marker, a blocks body (one comma-separated line per channel) is always Blastware.
    """
    if any("waveform data" in " ".join(s.lower().split()) for _, s in lines):
        return True
    if _has_blocks_body(lines):
        return False
    literal, unquoted = set(), set()
    for _, s in lines:
        m = _KEY_VALUE.match(s)
        if m is None:
            continue
        if m.group(1).lower() in _THOR_LITERAL_KEYS:
            literal.add(m.group(1).lower())
        if not s.startswith('"') and _normalize_key(m.group(1)) in _THOR_SPECIFIC_KEYS:
            unquoted.add(_normalize_key(m.group(1)))
    return len(literal) >= 2 or len(unquoted) >= 2


def _thor_units(text: str, warns: list[str]) -> dict[str, str]:
    out = {}
    for kind, part in zip(("geophone", "mic"), re.split(r"\s+and\s+", text.strip(), flags=re.IGNORECASE)):
        unit = part.strip().rstrip(".").strip().lower()
        if unit in (_GEOPHONE_UNITS if kind == "geophone" else _MIC_UNITS):
            out[kind] = unit
        else:
            warns.append(f"Units '{text}': {kind} unit '{part}' not understood, ignored")
    return out


def _thor_start(date_text: str | None, time_text: str | None, warns: list[str]) -> datetime | None:
    if date_text is None and time_text is None:
        return None
    try:
        if date_text is None or time_text is None:
            raise ValueError
        day = datetime.strptime(date_text.strip(), "%Y-%m-%d").date()
        m = _TIME.match(time_text.strip())
        if m is None:
            raise ValueError
        hour, minute, second = int(m[1]), int(m[2]), int(m[3])
        micro = int(m[4] or 0) * 1000 + int(m[5] or 0)
        if m[6]:  # reloj de 12 h
            if not 1 <= hour <= 12:
                raise ValueError
            hour = hour % 12 + (12 if m[6].upper() == "PM" else 0)
        return datetime.combine(day, dt_time(hour, minute, second, micro))
    except ValueError:
        warns.append(f"start not read: EventDate {date_text!r} and EventTime {time_text!r} not understood "
                     "(expected YYYY-MM-DD and HH:MM:SS[:mmm[:uuu]] [AM|PM]); start is None")
        return None


def _header_number(number: str, key: str, text: str, decimal, warns: list[str]) -> float | None:
    """A number taken from a header value; a comma counts as decimal mark only with ``decimal=","``."""
    if "," in number:
        if decimal != ",":
            warns.append(f"{key} '{text}' has a comma in the number (decimal comma or thousands separator), "
                         "ignored; pass decimal=\",\" if it is a decimal comma")
            return None
        number = number.replace(",", ".")
    return float(number)


def _number_before(regex, key: str, text: str | None, decimal, warns: list[str]) -> float | None:
    if text is None:
        return None
    m = regex.search(text)
    if m is None:
        warns.append(f"{key} '{text}' not understood, ignored")
        return None
    return _header_number(m[1], key, text, decimal, warns)


def _parse_thor(lines, decimal) -> _Parsed:
    warns = [_THOR_WARNING]
    marker = next((k for k, (_, s) in enumerate(lines) if "waveformdatachannels" in _normalize_key(s)), None)
    if marker is not None:
        header, names_at = lines[:marker], marker + 1
        if names_at >= len(lines):
            raise _param_error("path", "has no channel-name row after 'Waveform Data Channels'")
    else:
        names_at = next((k for k, (_, s) in enumerate(lines)
                         if all(_is_channel_token(f) for f in _split_fields(s, _guess_sep(s)))), None)
        if names_at is None:
            raise _param_error("path", "looks like a THOR export but no row of channel names was found "
                               "(Tran, Vert, Long, MicL)" + _RENAMED_HINT)
        header = lines[:names_at]

    meta: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for _, s in header:
        m = _KEY_VALUE.match(s)
        if m:
            meta.setdefault(m[1], m[2])
            by_key.setdefault(_normalize_key(m[1]), m[2])

    names_line = lines[names_at][1]
    sep = _guess_sep(names_line)
    names = _split_fields(names_line, sep)
    rows = []
    for lineno, s in lines[names_at + 1:]:
        if "waveform data" in " ".join(s.lower().split()):
            warns.append(f"section '{s.strip(chr(34))}' at line {lineno} ignored: its content is not documented")
            break
        rows.append((lineno, s))
    columns = _parse_table(rows, len(names), sep, _check_decimal(decimal, sep))

    units = _thor_units(by_key["units"], warns) if "units" in by_key else {}
    return _Parsed(names, columns, "instantel-thor", "third-party-parser",
                   sample_rate_hz=_number_before(_SPS, "SampleRate", by_key.get("samplerate"), decimal, warns),
                   units=units,
                   pretrigger_s=_number_before(_SEC, "Pre-triggerLength", by_key.get("pretriggerlength"), decimal,
                                               warns),
                   start=_thor_start(by_key.get("eventdate"), by_key.get("eventtime"), warns),
                   metadata=meta, warnings=warns)


def _header_channel_names(texts: list[str]) -> list[str]:
    """Channel names of the header line that names the most channels (first such line on ties)."""
    best: list[str] = []
    for text in texts:
        found, seen = [], set()
        for token in re.findall(r"[A-Za-z]+\d*", text):
            key = _normalize_name(token)
            if _is_channel_token(token) and _ALIASES.get(key, key) not in seen:
                seen.add(_ALIASES.get(key, key))
                found.append(token)
        if len(found) > len(best):
            best = found
    return best


def _header_unit(regex, kind: str, texts: list[str], warns: list[str]) -> str | None:
    found = sorted({m.lower() for t in texts for m in regex.findall(t)})
    if len(found) > 1:
        warns.append(f"header names two {kind} units ({', '.join(found)}): the header gives no {kind} unit")
        return None
    return found[0] if found else None


def _parse_blastware(lines, decimal) -> _Parsed:
    warns = [_BLASTWARE_WARNING]
    k = _quoted_header_end(lines)
    texts = [s.strip('"').strip() for _, s in lines[:k]]
    body = lines[k:]
    if not body:
        raise _param_error("path", "has a header but no waveform data")

    rates = sorted({r for t in texts for r in (_header_number(m, "sample rate", t, decimal, warns)
                                                for m in _SPS.findall(t)) if r is not None})
    if len(rates) > 1:
        warns.append(f"header names two sample rates ({', '.join(f'{r:g}' for r in rates)} sps): ignored")
    # Unidades: solo una pista (R18). Las líneas de disparo y rango nombran unidades de otra cosa («120 dB(L)»).
    unit_texts = [t for t in texts if not _NOT_UNIT_LINE.search(t)]
    hints = {}
    for kind, regex in (("geophone", _GEO_UNIT), ("mic", _MIC_UNIT)):
        unit = _header_unit(regex, kind, unit_texts, warns)
        if unit is not None:
            hints[kind] = unit

    first = body[0][1]
    sep = _guess_sep(first)
    fields = _split_fields(first, sep)
    if not all(_looks_numeric(f) for f in fields):  # fila de nombres: tabla
        names, source = fields, "instantel-blastware-table"
        columns = _parse_table(body[1:], len(names), sep, _check_decimal(decimal, sep))
    else:  # una línea por canal, muestras separadas por coma
        source = "instantel-blastware-blocks"
        dec = _check_decimal(decimal, ",")
        names = _header_channel_names(texts)
        if not names:
            names = list(_DEFAULT_BLOCK_ORDER)
            warns.append("no channel names in the header: blocks read in the default order "
                         + ", ".join(_DEFAULT_BLOCK_ORDER))
        if len(body) != len(names):
            raise _param_error("path", f"has {len(body)} data lines (one per channel) but {len(names)} channels "
                               f"({', '.join(names)})")
        columns = [np.array([_parse_number(f, dec, lineno) for f in _split_fields(s, ",")])
                   for lineno, s in body]

    return _Parsed(names, columns, source, "manual-only", sample_rate_hz=rates[0] if len(rates) == 1 else None,
                   unit_hints=hints, metadata={f"header_{j}": t for j, t in enumerate(texts, start=1)},
                   warnings=warns)


def _parse_minimal(lines, decimal) -> _Parsed:
    names_line = lines[0][1]
    sep = _guess_sep(names_line)
    names = _split_fields(names_line, sep)
    if all(_looks_numeric(f) for f in names):
        raise _param_error("path", "has no row of channel names: the first line must hold them "
                           "(Tran, Vert, Long, MicL)" + _RENAMED_HINT)
    columns = _parse_table(lines[1:], len(names), sep, _check_decimal(decimal, sep))
    return _Parsed(names, columns, "instantel-minimal", "real-file")


def _merge_value(name: str, from_file: float | None, given, unit: str):
    """The argument fills only what the file lacks; both present and different is an error."""
    if given is None:
        return from_file
    if from_file is None:
        return given
    same = isinstance(given, (int, float, np.number)) and not isinstance(given, bool) \
        and math.isclose(float(given), from_file, rel_tol=1e-9, abs_tol=1e-12)
    if not same:
        raise _param_error(name, f"is {given!r} but the file says {from_file:g} {unit}; "
                           "drop the argument or fix the file")
    return from_file


def _merge_units(from_file: dict[str, str], given) -> dict[str, str]:
    clean = _check_units(given)
    for kind in clean.keys() & from_file.keys():
        if clean[kind] != from_file[kind]:
            raise _param_error("units", f"has {kind} '{given[kind]}' but the file says '{from_file[kind]}'; "
                               "drop the argument or fix the file")
    return {**clean, **from_file}


def read_instantel_ascii(path, *, sample_rate_hz: float | None = None, units: Mapping[str, str] | None = None,
                         pretrigger_s: float | None = None, decimal: str | None = None) -> Record:
    """Read an Instantel waveform exported to ASCII into a :class:`Record`.

    Three layouts are detected, in this order:

    * **THOR** (``"Key : Value"`` header lines, quotes optional, then ``Waveform Data Channels``,
      a row of channel names and one row per sample). Detected by a ``Waveform Data`` line; without
      it, by two THOR-only keys (``EventType``, ``Pre-triggerLength``, ``GeoRange``...) spelled as in
      THOR on any header line, or with other spacing on unquoted lines; a quoted ``Geo Range: ...``
      is Blastware, and so is a blocks body. Sample rate, units, pretrigger and start (the event
      time, ``EventDate`` + ``EventTime``) come from the header; the whole header goes to
      ``metadata``. ``format_evidence`` is ``"third-party-parser"``: built from a third-party
      parser's description, not validated against a real file.
    * **Blastware with header** (double-quoted text lines), followed by a row of channel names and
      a table (tab, comma or spaces), or by one comma-separated line per channel (blocks). The
      sample rate is searched in the header; the units found there (not on trigger, level or range
      lines) are only a hint: an explicit ``units=`` wins, with a warning if they differ. Only the
      default separators are read: double quotes around the header and commas between block values
      (other String Delimiter or separator settings are not supported). ``start`` is None. Built
      from the manual only, not validated against a real file: ``format_evidence`` is
      ``"manual-only"``.
    * **Minimal**: a row of channel names (``Tran``, ``Vert``, ``Long``, ``MicL``) and one row per
      sample, with no metadata (``start`` is None). ``format_evidence`` is ``"real-file"``.

    ``sample_rate_hz``, ``units`` and ``pretrigger_s`` fill only what the file lacks; if both give
    a value and they differ, ``ValueError`` shows the two (Blastware units excepted, see above). The
    sample rate is never guessed. The decimal mark is a point; pass ``decimal=","`` for a decimal
    comma (not allowed when the fields are comma-separated; point-decimal values are still read).
    Values follow the number patterns of ``read_events`` plus scientific notation; anything else
    (``nan``, ``1_0``, ``+.5``) is a ``ValueError`` with its line. Renamed channels are not
    recognized: use ``read_waveform_csv``.
    """
    _check_decimal(decimal, None)  # valor inválido: error antes de abrir el archivo
    lines = _read_lines(path)
    if not lines:
        raise _param_error("path", "is an empty file")
    if _is_thor(lines):
        parsed = _parse_thor(lines, decimal)
    elif lines[0][1].startswith('"'):
        parsed = _parse_blastware(lines, decimal)
    else:
        parsed = _parse_minimal(lines, decimal)

    fs = _merge_value("sample_rate_hz", parsed.sample_rate_hz, sample_rate_hz, "sps")
    if fs is None:
        raise _param_error("sample_rate_hz", "is required: the file does not give the sample rate; "
                           "pass sample_rate_hz=")
    pre = _merge_value("pretrigger_s", parsed.pretrigger_s, pretrigger_s, "s")
    unit_map = _merge_units(parsed.units, units)
    for kind, hint in parsed.unit_hints.items():  # Blastware: el encabezado sugiere, units= manda (R18)
        if kind not in unit_map:
            unit_map[kind] = hint
        elif unit_map[kind] != hint:
            parsed.warnings.append(f"the header suggests {kind} unit '{hint}' but units= gives '{units[kind]}': "
                                   "the argument is used")
    if len(set(parsed.names)) != len(parsed.names):
        raise _param_error("channels", f"has a repeated channel name in {parsed.names}")

    try:
        return _make_record(dict(zip(parsed.names, parsed.columns)), fs, units=unit_map, pretrigger_s=pre,
                            start=parsed.start, metadata=parsed.metadata, source=parsed.source,
                            format_evidence=parsed.format_evidence, warnings=parsed.warnings)
    except ValueError as exc:
        if "unknown channel name" not in str(exc):
            raise
        raise ValueError(f"{exc}{_RENAMED_HINT}") from None


# --------------------------------------------------------------------------- miniSEED y SAC (ObsPy)

def read_obspy(path, *, channels: Mapping[str, str], units: Mapping[str, str],
               counts_per_unit: Mapping[str, float] | None = None,
               pretrigger_s: float | None = None, sample_rate_hz: float | None = None) -> Record:
    """Read a miniSEED or SAC file (anything ``obspy.read`` understands) into a :class:`Record`.

    Needs the optional dependency ObsPy: ``pip install blastingpy[seismic]``.

    ``channels`` maps each canonical channel name (``"transverse"``, ``"vertical"``,
    ``"longitudinal"``, ``"mic"``) to the trace channel code (``stats.channel``, e.g. ``"HHZ"``);
    a code that is absent from the file or matches more than one trace is an error. ``units`` is
    ``{"geophone": ..., "mic": ...}`` as in :func:`from_arrays`. With ``counts_per_unit`` (the
    sensitivity of each named channel, counts per unit), that channel is divided by it before the
    conversion to SI; without it the samples are taken as already in ``units``.

    All selected traces must share the sample rate (relative 1e-9), the start time (within half a
    sample period) and the length. ``Record.start`` is the first sample's time of the first selected
    trace (not the trigger time), as a naive ``datetime`` in UTC. ``format_evidence`` is ``"user"``: the calibration is the caller's.

    The sample rate is never guessed. SAC stores the sample interval rounded, so a 1024 Hz SAC reads as about
    1023.54 Hz: when the file's rate is within 0.1 % of a whole number but not equal to it, the record carries a
    warning that names the whole number. Pass ``sample_rate_hz`` (the instrument's rate) to use it instead; it must
    agree with the file within 0.1 %, else ``ValueError``.
    """
    try:
        import obspy
    except ImportError:
        raise ImportError("read_obspy needs ObsPy: pip install blastingpy[seismic]") from None
    if not isinstance(channels, Mapping) or not channels:
        raise _param_error("channels", "must be a non-empty mapping of channel name to trace channel code")
    sens = {} if counts_per_unit is None else dict(counts_per_unit)
    for key, value in sens.items():
        if key not in channels:
            raise _param_error("counts_per_unit", f"has key '{key}' that is not in channels")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
                or not math.isfinite(value) or value <= 0:
            raise _param_error("counts_per_unit", f"must be finite numbers > 0; got {value!r} for '{key}'")

    stream = obspy.read(str(path))
    picked = {}
    for name, code in channels.items():
        found = [tr for tr in stream if tr.stats.channel == code]
        if len(found) != 1:
            raise _param_error("channels", f"code '{code}' for '{name}' matches {len(found)} traces in the file "
                               f"(codes present: {sorted({tr.stats.channel for tr in stream})}); need exactly one")
        picked[name] = found[0]

    first_name, first = next(iter(picked.items()))
    fs = float(first.stats.sampling_rate)
    for name, tr in picked.items():
        if abs(tr.stats.sampling_rate - fs) > 1e-9 * fs:
            raise _param_error("channels", f"must share one sample rate; '{first_name}' has {fs:g} Hz "
                               f"and '{name}' has {tr.stats.sampling_rate:g} Hz")
        if abs(tr.stats.starttime - first.stats.starttime) > 0.5 / fs:
            raise _param_error("channels", f"must share the same start; '{name}' starts "
                               f"{tr.stats.starttime - first.stats.starttime:+g} s from '{first_name}'")
        if tr.stats.npts != first.stats.npts:
            raise _param_error("channels", f"must all have the same length; '{first_name}' has "
                               f"{first.stats.npts} samples and '{name}' has {tr.stats.npts}")

    warns: list[str] = []
    whole = round(fs)
    if sample_rate_hz is not None:
        if not _real(sample_rate_hz) or abs(sample_rate_hz - fs) > _RATE_TOLERANCE * fs:
            raise _param_error("sample_rate_hz", f"is {sample_rate_hz!r} Hz but the file gives {fs:g} Hz; they must "
                               "agree within 0.1 %")
        fs = float(sample_rate_hz)
    elif fs != whole and abs(fs - whole) <= _RATE_TOLERANCE * fs:
        warns.append(f"the file's sample rate is {fs:.6g} Hz, close to {whole} Hz: SAC stores the sample interval "
                     f"rounded; pass sample_rate_hz={whole} if that is the instrument's rate")
    data = {name: np.asarray(tr.data, dtype=np.float64) / sens.get(name, 1.0) for name, tr in picked.items()}
    return _make_record(data, fs, units=units, pretrigger_s=pretrigger_s, start=first.stats.starttime.datetime,
                        metadata=None, source="obspy", format_evidence="user", warnings=warns)


# --------------------------------------------------------------------------- histograma de Instantel
# Formato y evidencia: docs/investigacion/FUENTES_SPRINT2.md, «Formatos de exportación», D (solo el manual).

_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                       "dec"), start=1)}
_HIST_DATE = re.compile(r"^(\d{1,2})-([A-Za-z]{3})-(\d{2})$")
_HIST_TIME = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")
_HIST_MARKER = re.compile(r"^(N/A|[<>]\s*\d+(?:\.\d+)?)$", re.IGNORECASE)
_HIST_WARNING = "manual-only: histogram layout built from the Blastware manual, not validated against a real file"
_OTHER_UNITS = {"hz": "Hz", "db(a)": "dB(A)", "db(l)": "dB(L)", "db": "dB"}


def _hist_unit(unit: str) -> tuple[str, float]:
    """Unit text of a histogram column -> (SI unit text, factor to SI); unknown units are kept as written."""
    key = unit.strip().rstrip(".").strip().lower()
    if key in _GEOPHONE_UNITS:
        return "mm/s", _GEOPHONE_UNITS[key]
    if _MIC_UNITS.get(key) is not None:
        return "Pa", _MIC_UNITS[key]
    return _OTHER_UNITS.get(key, unit.strip()), 1.0


def _hist_date(text: str):
    """The date of a ``D-Mon-YY`` text (English month, 2-digit year as in ``strptime``) or None."""
    m = _HIST_DATE.match(text)
    month = _MONTHS.get(m[2].lower()) if m else None
    if month is None:
        return None
    year = int(m[3])
    try:
        return datetime(2000 + year if year < 69 else 1900 + year, month, int(m[1])).date()
    except ValueError:
        return None


def read_instantel_histogram(path) -> pd.DataFrame:
    """Read a Blastware interval histogram exported to ASCII into a DataFrame.

    Layout (built from the manual only, not validated against a real file): three header lines
    (channel, quantity, unit; ``Time`` labels the first column on any of them), tab- or comma-separated
    (decided by the first header line); a line holding only a date (``15-Feb-13``); then one row per
    interval starting with the time ``H:MM:SS``. A later date-only line changes the date (midnight).

    Columns: ``time`` (date + time, intervals need not be equal) and one per channel-quantity pair named
    ``<channel>_<quantity>`` in lowercase with spaces removed (``Mic A`` + ``Lmax`` -> ``mica_lmax``); there is
    no fixed list of quantities. A cell ``N/A``, ``<x`` or ``>x`` becomes NaN and its text goes to
    ``<column>_flag`` (that column exists only for columns with at least one marker; other cells are None).
    Any other non-numeric cell is a ``ValueError``. ``df.attrs["units"]`` maps each numeric column to its unit
    after conversion (``in/s`` -> mm/s, ``psi`` -> Pa, converted once) and ``df.attrs["warnings"]`` carries the
    "manual-only" warning.
    """
    lines = _read_lines(path, strip=False)
    if len(lines) < 3:
        raise _param_error("path", "has fewer than the three header lines (channel, quantity, unit)")
    first = lines[0][1]
    sep = "\t" if "\t" in first else "," if "," in first else None
    if sep is None:
        raise _param_error("path", "has no tab or comma separator in the first header line")

    head = [[f.strip() for f in text.split(sep)] for _, text in lines[:3]]
    for h in head:
        while h and h[-1] == "":
            h.pop()
    n = max(len(h) for h in head)
    head = [h + [""] * (n - len(h)) for h in head]
    labels = [h[0] for h in head if h[0]]
    if n < 2 or len(labels) != 1 or labels[0].lower() != "time":
        raise _param_error("path", "must have the label 'Time' on one of the three header lines, in the first "
                           "column, and at least one data column")
    names, units, factors = [], [], []
    for j in range(1, n):
        channel, quantity, unit = head[0][j], head[1][j], head[2][j]
        if not (channel and quantity and unit):
            raise _param_error("path", f"has an incomplete header in column {j + 1} (channel, quantity and unit "
                               f"are needed; got '{channel}', '{quantity}', '{unit}')")
        name = f"{_normalize_name(channel)}_{_normalize_name(quantity)}"
        if name in names:
            raise _param_error("path", f"has two columns named '{name}' (channel and quantity repeat)")
        si_unit, factor = _hist_unit(unit)
        names.append(name)
        units.append(si_unit)
        factors.append(factor)

    date = None
    times: list[datetime] = []
    values: list[list[float]] = [[] for _ in names]
    flags: list[list[str | None]] = [[] for _ in names]
    for lineno, text in lines[3:]:
        fields = [f.strip() for f in text.split(sep)]
        while len(fields) > n and fields[-1] == "":
            fields.pop()
        filled = [f for f in fields if f]
        if len(filled) == 1 and _HIST_DATE.match(filled[0]):
            date = _hist_date(filled[0])
            if date is None:
                raise _param_error("path", f"has an invalid date '{filled[0]}' at line {lineno} "
                                   "(expected D-Mon-YY with an English month, e.g. 15-Feb-13)")
            continue
        m = _HIST_TIME.match(fields[0])
        if m is None:
            raise _param_error("path", f"has '{fields[0]}' at line {lineno}: expected a time H:MM:SS or a date "
                               "D-Mon-YY")
        if date is None:
            raise _param_error("path", f"has a data row at line {lineno} before any date line (D-Mon-YY)")
        if len(fields) != n:
            raise _param_error("path", f"has {len(fields)} fields at line {lineno}, expected {n}")
        try:
            times.append(datetime.combine(date, dt_time(int(m[1]), int(m[2]), int(m[3]))))
        except ValueError:
            raise _param_error("path", f"has an invalid time '{fields[0]}' at line {lineno}") from None
        for j, cell in enumerate(fields[1:]):
            if _HIST_MARKER.match(cell):
                values[j].append(np.nan)
                flags[j].append(cell)
                continue
            number = _number(cell, ".")
            if number is None or not np.isfinite(number):
                raise _param_error("path", f"has a value that is not a number at line {lineno}, column "
                                   f"'{names[j]}': '{cell}' (a number, N/A, <x or >x is expected)")
            values[j].append(number * factors[j])
            flags[j].append(None)
    if not times:
        raise _param_error("path", "has no data rows after the header")

    out: dict[str, object] = {"time": pd.Series(times)}
    for j, name in enumerate(names):
        out[name] = np.array(values[j], dtype=float)
        if any(f is not None for f in flags[j]):
            out[f"{name}_flag"] = pd.Series(flags[j], dtype=object)
    df = pd.DataFrame(out)
    df.attrs["units"] = dict(zip(names, units))
    df.attrs["warnings"] = [_HIST_WARNING]
    return df


# --------------------------------------------------------------------------- CSV genérico

def _csv_column(cells, label: str, name: str, decimal) -> np.ndarray:
    """A CSV column as floats, with the number rules of ``read_events`` (row 1 is the header)."""
    out = np.empty(len(cells))
    for i, cell in enumerate(cells):
        text = str(cell).strip()
        where = f"column '{name}' ({label}), row {i + 2}"
        if text == "":
            raise _param_error("path", f"has an empty cell in {where}")
        if decimal is None and _AMBIGUOUS.match(text):
            raise _param_error("path", f"has '{text}' in {where}, which is ambiguous: {_other_mark(text)}")
        number = _number(text, decimal, scientific=True)
        if number is None and _NUMBER[None].match(text):  # escrito con la otra marca
            raise _param_error("path", f"has '{text}' in {where}, which is not a number with decimal={decimal!r}: "
                               f"{_other_mark(text)}")
        if number is None or not np.isfinite(number):
            raise _param_error("path", f"has a value that is not a number in {where}: received '{text}'; if the "
                               "decimal mark is not what you expect, pass decimal=',' or '.'")
        out[i] = number
    return out


def _real(value) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(value)


def _time_axis(t: np.ndarray, sample_rate_hz, pretrigger_s) -> tuple[float, float, list[str]]:
    """Sample rate, pretrigger and warnings from a time column; explicit arguments must agree with it."""
    if t.size < 2:
        raise _param_error("time_column", "needs at least 2 rows to give the sample rate")
    idx = np.arange(t.size, dtype=float)
    step, intercept = np.polyfit(idx, t, 1)  # t = intercept + step * i
    step = float(step)
    if not step > 0:
        raise _param_error("time_column", "must be increasing: the fitted time step is not positive")
    resid = np.abs(t - (intercept + step * idx))
    k = int(np.argmax(resid))
    if resid[k] > 0.25 * step:
        raise _param_error("time_column", f"must have a uniform step: the time at row {k + 2} is "
                           f"{resid[k] / step:.2g} steps away from the fitted straight line (tolerance 0.25 step)")
    fs = 1.0 / step
    if abs(fs - round(fs)) <= _SNAP * fs:  # R2: tiempos redondeados dan 1024.0000003 Hz; el resto es ruido del ajuste
        fs = float(round(fs))
    if sample_rate_hz is not None:
        if not _real(sample_rate_hz) or abs(sample_rate_hz - fs) > 1e-3 * fs:
            raise _param_error("sample_rate_hz", f"is {sample_rate_hz!r} but the time column gives {fs:g} Hz; "
                               "drop the argument or fix the file")
    t0 = float(intercept)
    half = step / 2
    warns: list[str] = []
    if t0 < -half:
        pre = -t0
    else:
        pre = 0.0
        if t0 > half:
            warns.append(f"the first sample is at t = {t0:g} s: the time axis is shifted to start at 0 and "
                         "pretrigger_s is 0.0")
    whole = round(pre * fs)
    if abs(pre * fs - whole) <= _SNAP_SAMPLE:  # R2: a la muestra entera si está a menos de 1e-3 de periodo
        pre = whole / fs
    if pretrigger_s is not None:
        if not _real(pretrigger_s) or abs(pretrigger_s - pre) > half:
            raise _param_error("pretrigger_s", f"is {pretrigger_s!r} s but the time column gives {pre:g} s; "
                               "drop the argument or fix the file")
    return fs, pre, warns


def read_waveform_csv(path, *, columns: Mapping[str, str], units: Mapping[str, str],
                      sample_rate_hz: float | None = None, time_column: str | None = None,
                      pretrigger_s: float | None = None, sep: str | None = None,
                      decimal: str | None = None) -> Record:
    """Read any brand's waveform exported as CSV into a :class:`Record` (renamed or unknown channels included).

    ``columns`` maps a canonical channel name (``transverse``, ``vertical``, ``longitudinal``, ``mic``, or an
    alias such as ``Tran``) to its column in the file; a mapped column that is missing is an error. ``units`` is
    ``{"geophone": "mm/s" | "in/s", "mic": "Pa" | "psi"}`` (a kind left out is inferred, with a warning).

    With ``time_column`` a straight line ``t = t0 + i / fs`` is fitted by least squares (so times rounded to a few
    decimals are fine) and every time must lie within a quarter step of it, else ``ValueError``; a negative fitted
    ``t0`` gives ``pretrigger_s = -t0``, ``t0`` near 0 gives 0.0 and ``t0 > 0`` gives 0.0 with a warning. A fitted
    rate within 1e-6 (relative) of a whole number is taken as that number, and a pretrigger within 1e-3 of a sample
    period of a whole number of samples as that time (times rounded to the microsecond give 1024.0000003 Hz). Numbers may
    use scientific notation (``1e-05``). An explicit ``sample_rate_hz`` or
    ``pretrigger_s`` must agree with the time column. Without it, ``sample_rate_hz`` is required (never guessed).

    Separator and decimal follow ``read_events``: with ``;`` the decimal mark is the comma, with ``,`` it is the
    point, and with a tab either one is accepted except the ambiguous shape ``1,250``; ``sep`` and ``decimal``
    override the detection. The record has ``source="csv"`` and ``format_evidence="user"``.
    """
    if not isinstance(columns, Mapping) or not columns:
        raise _param_error("columns", "must be a non-empty mapping of canonical channel name to column name")
    if decimal not in _NUMBER:
        raise _param_error("decimal", f"must be ',' or '.' (got {decimal!r})")
    if time_column is None and sample_rate_hz is None:
        raise _param_error("sample_rate_hz", "is required when there is no time_column: the sample rate is "
                           "never guessed")
    table, found = _read_csv(path, sep)
    if decimal is None and found == ";":
        decimal = ","  # con ";" el punto es marca de miles: nunca se lee como decimal (igual que read_events)
    elif decimal is None and found == ",":
        decimal = "."

    wanted = {name: str(file_col) for name, file_col in columns.items()}
    if time_column is not None:
        wanted["time"] = time_column
    for label, file_col in wanted.items():
        if file_col not in table.columns:
            raise _param_error("columns" if label != "time" else "time_column",
                               f"maps '{label}' to '{file_col}', which is not in the file; available columns: "
                               f"{[str(c) for c in table.columns]}")
        if (table.columns == file_col).sum() > 1:
            raise _param_error("columns", f"has the column '{file_col}' twice in the header")
    cells = {label: list(table[file_col]) for label, file_col in wanted.items()}
    used = len(table)
    while used and all(str(c[used - 1]).strip() == "" for c in cells.values()):
        used -= 1  # líneas en blanco al final del archivo
    data = {label: _csv_column(c[:used], label, wanted[label], decimal) for label, c in cells.items()}

    warns: list[str] = []
    pretrigger = pretrigger_s
    if time_column is not None:
        sample_rate_hz, pretrigger, warns = _time_axis(data.pop("time"), sample_rate_hz, pretrigger_s)
    return _make_record(data, sample_rate_hz, units=units, pretrigger_s=pretrigger, start=None, metadata=None,
                        source="csv", format_evidence="user", warnings=warns)


_DXF_HINT = "ezdxf is required to read or write .dxf files: pip install blastingpy[dxf]"


def _require_ezdxf():
    try:
        import ezdxf
    except ImportError:
        raise ImportError(_DXF_HINT) from None
    return ezdxf


_ARC_TOLERANCE_M = 0.05  # máxima distancia entre un arco (bulge) y las cuerdas que lo reemplazan


def _one_polyline(msp, layer: str):
    key = layer.casefold()  # en DXF los nombres de capa no distinguen mayúsculas
    found = [e for e in msp if e.dxftype() in ("LWPOLYLINE", "POLYLINE") and e.dxf.layer.casefold() == key]
    if len(found) != 1:
        raise ValueError(f"layer '{layer}' has {len(found)} polylines; expected 1")
    e = found[0]
    from ezdxf import path
    # make_path pasa del OCS de la entidad al mundo (dibujos espejados) y convierte los arcos en cuerdas
    pts = [(v.x, v.y) for v in path.make_path(e).flattening(_ARC_TOLERANCE_M)]
    pts = np.array(pts, dtype=float)
    if e.is_closed and len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]  # make_path repite el primer punto al cerrar
    return pts, bool(e.is_closed)


def read_dxf_outline(path, *, outline_layer: str, crest_layer: str) -> tuple[np.ndarray, np.ndarray]:
    """Outline and crest of a bench from a DXF file, as (N, 2) arrays in the file's units (m for a pattern).

    Each layer must hold exactly one LWPOLYLINE or POLYLINE (layer names are case-insensitive, as in DXF). Points are
    returned in world coordinates (mirrored drawings with extrusion (0, 0, -1) are handled) and arcs (bulges) are
    replaced by chords within 0.05 m. The outline must be closed, by the closed flag or by a last point equal to the
    first (that repeated point is dropped); the crest may be open or closed. Warns
    when the header ``$INSUNITS`` gives units other than metres (6) or unitless (0). Needs
    ``pip install blastingpy[dxf]``.
    """
    ezdxf = _require_ezdxf()
    doc = ezdxf.readfile(path)
    units = doc.header.get("$INSUNITS", 0)
    if units not in (0, 6):  # 0 sin unidades, 6 metros
        warnings.warn(f"DXF $INSUNITS = {units}: the drawing is not in metres (6) nor unitless (0), but the "
                      "coordinates are read as metres; check units (mm vs m?)", UserWarning, stacklevel=2)
    msp = doc.modelspace()
    outline, closed = _one_polyline(msp, outline_layer)
    if len(outline) > 1 and np.allclose(outline[0], outline[-1]):
        outline, closed = outline[:-1], True
    if not closed:
        raise ValueError(f"layer '{outline_layer}': the outline polyline is not closed (set the closed flag or "
                         "repeat the first point at the end)")
    return outline, _one_polyline(msp, crest_layer)[0]
