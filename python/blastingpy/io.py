"""Seismograph records: the immutable :class:`Record` and unit handling.

A :class:`Record` holds one blast record in SI units: geophone channels in mm/s, the
microphone in Pa and time in s. Channel names are canonical (``"transverse"``,
``"vertical"``, ``"longitudinal"``, ``"mic"``). Every reader of this module builds its
``Record`` through :func:`_make_record`, so validation and unit conversion live in one place.

The sample rate is never guessed: it must come from the file or from the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
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
