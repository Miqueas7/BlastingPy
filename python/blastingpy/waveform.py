"""Analysis of a seismograph record and its row in the events table.

:func:`analyze` takes a :class:`blastingpy.io.Record` (geophones in mm/s, microphone in Pa) and returns a
:class:`RecordSummary`: per channel the peak and its time, the zero-crossing frequency at the peak, the FFT dominant
frequency and, for the geophones, the peak acceleration (g) and displacement (mm); for the record the true vector sum
(PVS), the pseudo vector sum and the microphone peak in Pa and dB(L). :func:`to_event_row` and
:func:`events_from_records` turn summaries into rows of the events table of :mod:`blastingpy.events`, ready for
:func:`blastingpy.attenuation.fit` and :func:`blastingpy.criteria.evaluate`.

Conventions
-----------
* ``zc_method="interpolated"`` interpolates the zero crossings linearly (a design variant, no limit);
  ``"instantel"`` counts whole samples and flags ``above_limit`` over ``100 * fs / 1024`` Hz. The frequency is NaN
  unless ``zc_status == "ok"``, so an event row never carries a frequency the method could not stand behind.
* The FFT frequency is the largest bin of ``|rfft(x - mean)|`` (bin 0 excluded); ``fft_peaks_6db_hz`` lists the local
  maxima with at least half the largest amplitude (6 dB), largest first.
* dB(L) is ``20 * log10(PSPL / 20e-6 Pa)``; a zero PSPL gives NaN, not minus infinity.
* The event row uses ``ppv_kind = "component_max"``: ``ppv_mm_s`` is the largest geophone peak and ``frequency_hz``
  its zero-crossing frequency (ties: transverse, vertical, longitudinal).
* When at least 90 % of the nonzero geophone samples have 3 significant figures or fewer, a warning says that the
  acceleration and the displacement depend on the precision the signal was exported with.
"""

from __future__ import annotations

import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pandas as pd

from ._blastingpy_core import _waveform as _core
from .events import _ORDER, _param_error
from .io import Record

__all__ = ["ChannelSummary", "RecordSummary", "fft_frequency", "analyze", "to_event_row", "events_from_records"]

_GEOPHONES = ("transverse", "vertical", "longitudinal")
_ROW_SUFFIX = {"transverse": "t", "vertical": "v", "longitudinal": "l"}
_ZC_METHODS = {"interpolated": _core.ZcMethod.Interpolated, "instantel": _core.ZcMethod.WholeSamples}
_DIFF_METHODS = {"central": _core.DiffMethod.Central, "forward": _core.DiffMethod.Forward}
_ZC_STATUS = {_core.ZcStatus.Ok: "ok", _core.ZcStatus.AboveLimit: "above_limit",
              _core.ZcStatus.Incomplete: "incomplete", _core.ZcStatus.NotComputable: "not_computable"}
_P_REF_PA = 20e-6  # presión de referencia del dB(L)
_PRECISION_SHARE = 0.9
_PRECISION_WARNING = ("at least 90 % of the nonzero geophone samples have 3 significant figures or fewer: the peak "
                      "acceleration and displacement depend on the precision the signal was exported with")


@dataclass(frozen=True)
class ChannelSummary:
    """Peak, frequencies and derived peaks of one channel (geophone in mm/s, ``"mic"`` in Pa)."""

    name: str
    peak: float
    peak_time_s: float
    zc_frequency_hz: float  # NaN salvo zc_status == "ok"
    zc_status: str  # "ok", "above_limit", "incomplete", "not_computable"
    fft_frequency_hz: float
    fft_peaks_6db_hz: tuple[float, ...]
    peak_acceleration_g: float  # NaN en el micrófono
    peak_displacement_mm: float  # NaN en el micrófono


@dataclass(frozen=True)
class RecordSummary:
    """Analysis of one record; ``warnings`` lists the record's own warnings first, then the analysis ones.

    It can be pickled and deep-copied; ``channels`` stays read-only in the copy.
    """

    channels: Mapping[str, ChannelSummary]
    pvs_mm_s: float  # suma vectorial real; NaN si falta un geófono
    pvs_time_s: float
    pseudo_vector_sum_mm_s: float
    mic_pspl_pa: float  # NaN sin micrófono
    mic_db_l: float
    zc_method: str
    warnings: tuple[str, ...]
    start: datetime | None

    def __reduce__(self):
        # MappingProxyType no se serializa: se guarda como dict y se vuelve a congelar al reconstruir.
        state = {f.name: getattr(self, f.name) for f in fields(self)}
        state["channels"] = dict(self.channels)
        return (_rebuild_summary, (state,))


def _rebuild_summary(state: dict) -> RecordSummary:
    """Inverse of ``RecordSummary.__reduce__``."""
    return RecordSummary(**{**state, "channels": MappingProxyType(dict(state["channels"]))})


def _is_real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _choice(name: str, value, table: dict):
    try:
        return table[value]
    except (KeyError, TypeError):
        raise _param_error(name, f"must be one of {list(table)}. Received: {value!r}") from None


def fft_frequency(x, sample_rate_hz: float) -> tuple[float, tuple[float, ...]]:
    """Dominant frequency (Hz) of ``x`` and the spectral peaks within 6 dB of it, largest first.

    The spectrum is ``|numpy.fft.rfft(x - mean(x))|`` without bin 0. A local maximum is a bin larger than its left
    neighbour and not smaller than its right one (the last bin compares only to the left); those with at least half
    the largest amplitude are kept. A flat signal, or one that varies only at rounding level so that no
    peak is kept, has no frequency: ``(nan, ())``.
    """
    if not _is_real(sample_rate_hz) or sample_rate_hz <= 0:
        raise _param_error("sample_rate_hz", f"must be a finite number > 0. Received: {sample_rate_hz!r}")
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise _param_error("x", "must be a numeric array") from None
    if x.ndim != 1 or x.size < 2:
        raise _param_error("x", f"must be a one-dimensional signal with at least 2 samples. Received shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise _param_error("x", "must not contain NaN or infinite values")
    if np.ptp(x) == 0.0:  # señal plana (también toda en cero): x − media no siempre da cero exacto
        return math.nan, ()
    amp = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sample_rate_hz)
    i = np.arange(1, amp.size)
    right = np.append(amp[2:], -np.inf)  # el último bin solo se compara con el izquierdo
    local = i[(amp[1:] > amp[:-1]) & (amp[1:] >= right)]
    kept = local[amp[local] >= 0.5 * amp[1:].max()]
    if kept.size == 0:  # variación al nivel del redondeo: el espectro no tiene un máximo local que valga
        return math.nan, ()
    kept = kept[np.argsort(-amp[kept], kind="stable")]
    peaks = tuple(float(f) for f in freqs[kept])
    return peaks[0], peaks


def _has_three_figures(x: np.ndarray) -> np.ndarray:
    """True where ``x`` (nonzero) is unchanged when rounded to 3 significant figures (relative 1e-9)."""
    with np.errstate(over="ignore", invalid="ignore"):  # un subnormal desborda la escala: cuenta como «más cifras»
        scale = 10.0 ** (2 - np.floor(np.log10(np.abs(x))))
        return np.abs(np.round(x * scale) / scale - x) <= 1e-9 * np.abs(x)


def _channel(name: str, x: np.ndarray, record: Record, zc_method, diff_method) -> ChannelSummary:
    fs = record.sample_rate_hz
    peak = _core.peak_abs(x)
    zc = _core.zero_crossing_frequency(x, fs, peak.index, zc_method)
    status = _ZC_STATUS[zc.status]
    fft_f, fft_peaks = fft_frequency(x, fs)
    if name == "mic":
        accel_g = disp_mm = math.nan
    else:
        accel_g = _core.peak_acceleration(x, fs, diff_method).value / _core.standard_gravity_mm_s2
        disp_mm = _core.peak_displacement(x, fs).value
    return ChannelSummary(name=name, peak=peak.value, peak_time_s=float(record.time_s()[peak.index]),
                          zc_frequency_hz=zc.frequency_hz if status == "ok" else math.nan, zc_status=status,
                          fft_frequency_hz=fft_f, fft_peaks_6db_hz=fft_peaks, peak_acceleration_g=accel_g,
                          peak_displacement_mm=disp_mm)


def analyze(record: Record, *, zc_method: str = "interpolated", acceleration: str = "central") -> RecordSummary:
    """Analyze every channel of ``record`` and the record as a whole (see the module docstring).

    ``zc_method`` is ``"interpolated"`` or ``"instantel"`` (whole samples with Instantel's limits); ``acceleration``
    is ``"central"`` or ``"forward"`` differences. The vector sums need the three geophones (NaN otherwise); the mic
    values need the microphone (NaN otherwise).
    """
    if not isinstance(record, Record):
        raise _param_error("record", f"must be a blastingpy.io.Record. Received: {type(record).__name__}")
    zc = _choice("zc_method", zc_method, _ZC_METHODS)
    diff = _choice("acceleration", acceleration, _DIFF_METHODS)
    channels = {name: _channel(name, x, record, zc, diff) for name, x in record.channels.items()}

    nan = math.nan
    pvs = pvs_time = pseudo = nan
    if all(g in record.channels for g in _GEOPHONES):
        t, v, l = (record.channels[g] for g in _GEOPHONES)
        vs = _core.vector_sum_peak(t, v, l)
        pvs, pvs_time = vs.value, float(record.time_s()[vs.index])
        pseudo = _core.pseudo_vector_sum(*(channels[g].peak for g in _GEOPHONES))
    pspl = channels["mic"].peak if "mic" in channels else nan
    db_l = 20.0 * math.log10(pspl / _P_REF_PA) if pspl > 0 else nan  # NaN > 0 es False

    warns = list(record.warnings)
    geo = [record.channels[g] for g in _GEOPHONES if g in record.channels]
    if geo:
        pooled = np.concatenate(geo)
        pooled = pooled[pooled != 0.0]
        if pooled.size and np.mean(_has_three_figures(pooled)) >= _PRECISION_SHARE:
            warns.append(_PRECISION_WARNING)
    return RecordSummary(channels=MappingProxyType(channels), pvs_mm_s=pvs, pvs_time_s=pvs_time,
                         pseudo_vector_sum_mm_s=pseudo, mic_pspl_pa=pspl, mic_db_l=db_l, zc_method=zc_method,
                         warnings=tuple(warns), start=record.start)


def _positive(name: str, value) -> float:
    if not _is_real(value) or value <= 0:
        raise _param_error(name, f"must be a finite number > 0. Received: {value!r}")
    return float(value)


def to_event_row(summary: RecordSummary, *, distance_m: float, charge_kg: float, date=None, point=None) -> dict:
    """One row of the events table (keys in the canonical column order) from a record summary.

    ``ppv_mm_s`` is the largest geophone peak (``ppv_kind = "component_max"``) and ``frequency_hz`` its zero-crossing
    frequency; ``airblast_pa`` and ``airblast_freq_hz`` are the microphone's peak pressure and zero-crossing frequency;
    ``date`` defaults to the record's start. Values that do not exist are NaN. A summary without geophones
    is refused: the events table needs a PPV.
    """
    if not isinstance(summary, RecordSummary):
        raise _param_error("summary", f"must be a RecordSummary. Received: {type(summary).__name__}")
    distance_m = _positive("distance_m", distance_m)
    charge_kg = _positive("charge_kg", charge_kg)
    geo = [summary.channels[g] for g in _GEOPHONES if g in summary.channels]
    if not geo:
        raise _param_error("summary", "has no geophone channel: the events table needs a PPV")
    top = max(geo, key=lambda c: c.peak)  # max devuelve el primero de los empatados: T, V, L
    mic = summary.channels.get("mic")
    if date is None:
        date = summary.start
    values = {"distance_m": distance_m, "charge_kg": charge_kg, "ppv_mm_s": top.peak, "ppv_kind": "component_max",
              "frequency_hz": top.zc_frequency_hz, "date": math.nan if date is None else date,
              "point": math.nan if point is None else point, "pvs_mm_s": summary.pvs_mm_s,
              "airblast_pa": summary.mic_pspl_pa,
              "airblast_freq_hz": mic.zc_frequency_hz if mic else math.nan}
    for g, s in _ROW_SUFFIX.items():
        c = summary.channels.get(g)
        values[f"ppv_{s}_mm_s"] = c.peak if c else math.nan
        values[f"freq_{s}_hz"] = c.zc_frequency_hz if c else math.nan
    return {col: values[col] for col in _ORDER}


def _per_record(name: str, values, n: int) -> list:
    if isinstance(values, (str, bytes)) or np.ndim(values) != 1:
        raise _param_error(name, f"must be a sequence with one item per record ({n}). Received: {values!r}")
    if len(values) != n:
        raise _param_error(name, f"must have one item per record: {n} records, {len(values)} items")
    return list(values)


def events_from_records(records: Sequence[Record], *, distance_m: Sequence[float], charge_kg: Sequence[float],
                        point: Sequence | None = None, zc_method: str = "interpolated") -> pd.DataFrame:
    """The events table (canonical columns, in order) of a list of records, one row per record.

    ``distance_m`` and ``charge_kg`` (and ``point`` when given) have one item per record. Each record is analyzed with
    :func:`analyze` and turned into a row with :func:`to_event_row` (``date`` is the record's start).

    Nothing the records carry is lost: ``df.attrs["warnings"]`` lists every warning of every summary (the record's
    own, then the analysis ones) as ``"record <i>: <warning>"`` (``i`` is the position in ``records``), and
    ``df.attrs["format_evidence"]`` lists each record's ``format_evidence``. When there are warnings, a single
    ``UserWarning`` names each distinct one with the records that carry it.
    """
    records = list(records)
    n = len(records)
    distances = _per_record("distance_m", distance_m, n)
    charges = _per_record("charge_kg", charge_kg, n)
    points = [None] * n if point is None else _per_record("point", point, n)
    summaries = [analyze(r, zc_method=zc_method) for r in records]
    rows = [to_event_row(s, distance_m=d, charge_kg=w, point=p)
            for s, d, w, p in zip(summaries, distances, charges, points)]
    df = pd.DataFrame(rows, columns=list(_ORDER))

    where: dict[str, list[int]] = {}  # aviso distinto -> registros que lo traen, en orden de aparición
    for i, s in enumerate(summaries):
        for w in s.warnings:
            where.setdefault(w, [])
            if i not in where[w]:
                where[w].append(i)
    df.attrs["warnings"] = [f"record {i}: {w}" for i, s in enumerate(summaries) for w in s.warnings]
    df.attrs["format_evidence"] = [r.format_evidence for r in records]
    if where:
        parts = [f"{w} (record{'s' if len(idx) > 1 else ''} {', '.join(map(str, idx))})" for w, idx in where.items()]
        warnings.warn(f"{len(where)} distinct warning{'s' if len(where) > 1 else ''} in the records (all of them in "
                      f"df.attrs['warnings']): " + "; ".join(parts), UserWarning, stacklevel=2)
    return df
