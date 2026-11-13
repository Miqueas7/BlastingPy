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
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .events import _param_error

CHANNEL_NAMES = ("transverse", "vertical", "longitudinal", "mic")
SOURCES = ("arrays", "instantel-thor", "instantel-blastware-table", "instantel-blastware-blocks",
           "instantel-minimal", "csv", "obspy")
FORMAT_EVIDENCE = ("user", "real-file", "third-party-parser", "manual-only")

MM_PER_IN = 25.4
PA_PER_PSI = 6894.757

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
    ``source`` and ``format_evidence`` say how the record was built and how far the format is
    validated; ``warnings`` collects everything the builder had to assume.
    """

    channels: Mapping[str, np.ndarray]
    sample_rate_hz: float
    pretrigger_s: float | None
    start: datetime | None
    metadata: Mapping[str, str]
    source: str
    format_evidence: str
    warnings: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.channels.values())).size)

    def time_s(self) -> np.ndarray:
        """Time axis in seconds: ``i / fs - pretrigger`` (``i / fs`` without pretrigger)."""
        t = np.arange(self.n_samples) / self.sample_rate_hz
        return t - self.pretrigger_s if self.pretrigger_s is not None else t


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

    if pretrigger_s is None:
        warns.append("no pretrigger given: time starts at 0")
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

def _read_lines(path) -> list[tuple[int, str]]:
    """Non-empty lines of a text file as ``(1-based line number, stripped text)``; BOM and CRLF accepted."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), start=1) if line.strip()]


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
    if decimal == ",":
        text = text.replace(",", ".")
    elif "," in text:
        raise _param_error("decimal", f"is '.' but line {lineno} has a comma inside the number '{text}'; "
                           "if the file uses a decimal comma, pass decimal=\",\"")
    try:
        return float(text)
    except ValueError:
        raise _param_error("path", f"has a value that is not a number at line {lineno}: '{text}'") from None


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
# pueden venir en un encabezado de Blastware. Las genéricas igual se leen en un archivo THOR.
_THOR_SPECIFIC_KEYS = frozenset({"eventtype", "pretriggerlength", "recordstopmode", "autostoptime", "geotriggerlevel",
                                 "mictriggerlevel", "mictriggerleveldb", "georange", "micrange"})
_KEY_VALUE = re.compile(r'^"?\s*([^":]+?)\s*:\s*(.*?)\s*"?$')
# El número completo (con coma o punto) y sin dígitos pegados a la izquierda: «1,024 sps» no se lee como 24.
_SPS = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*sps\b", re.IGNORECASE)
_SEC = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s*sec\b", re.IGNORECASE)
_GEO_UNIT = re.compile(r"(?<![a-z/])(mm/s|in/s)(?![a-z/])", re.IGNORECASE)
_MIC_UNIT = re.compile(r"(?<![a-z/])(pa|psi|db)(?![a-z/])", re.IGNORECASE)
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
    units: dict[str, str] = field(default_factory=dict)
    pretrigger_s: float | None = None
    start: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _normalize_key(key: str) -> str:
    return re.sub(r"[\s\-_]", "", key).lower()


def _is_channel_token(token: str) -> bool:
    key = _normalize_name(token)
    return key in _ALIASES or key in _A_WEIGHTED or key in _SECOND_SENSOR


def _is_thor(lines) -> bool:
    if any("waveform data" in " ".join(s.lower().split()) for _, s in lines):
        return True
    keys = set()
    for _, s in lines:
        m = _KEY_VALUE.match(s)
        if m and _normalize_key(m.group(1)) in _THOR_SPECIFIC_KEYS:
            keys.add(_normalize_key(m.group(1)))
    return len(keys) >= 2


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
    k = 0
    while k < len(lines) and lines[k][1].startswith('"'):
        k += 1
    texts = [s.strip('"').strip() for _, s in lines[:k]]
    body = lines[k:]
    if not body:
        raise _param_error("path", "has a header but no waveform data")

    rates = sorted({r for t in texts for r in (_header_number(m, "sample rate", t, decimal, warns)
                                                for m in _SPS.findall(t)) if r is not None})
    if len(rates) > 1:
        warns.append(f"header names two sample rates ({', '.join(f'{r:g}' for r in rates)} sps): ignored")
    units = {}
    for kind, regex in (("geophone", _GEO_UNIT), ("mic", _MIC_UNIT)):
        unit = _header_unit(regex, kind, texts, warns)
        if unit is not None:
            units[kind] = unit

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
                   units=units, metadata={f"header_{j}": t for j, t in enumerate(texts, start=1)},
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
      a row of channel names and one row per sample). Sample rate, units, pretrigger and start
      come from the header; the whole header goes to ``metadata``. ``format_evidence`` is
      ``"third-party-parser"``.
    * **Blastware with header** (double-quoted text lines), followed by a row of channel names and
      a table (tab, comma or spaces), or by one comma-separated line per channel (blocks). The
      sample rate and units are searched in the header. Built from the manual only:
      ``format_evidence`` is ``"manual-only"``.
    * **Minimal**: a row of channel names (``Tran``, ``Vert``, ``Long``, ``MicL``) and one row per
      sample, with no metadata. ``format_evidence`` is ``"real-file"``.

    ``sample_rate_hz``, ``units`` and ``pretrigger_s`` fill only what the file lacks; if both give
    a value and they differ, ``ValueError`` shows the two. The sample rate is never guessed.
    The decimal mark is a point; pass ``decimal=","`` for a decimal comma (not allowed when the
    fields are comma-separated). Renamed channels are not recognized: use ``read_waveform_csv``.
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
