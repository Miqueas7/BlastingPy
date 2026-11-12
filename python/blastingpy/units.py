"""Unit conversion to and from SI (m, kg, mm/s, mm) and scaled distance."""

from __future__ import annotations

import numpy as np

from ._blastingpy_core import _units as _core

scaled_distance = _core.scaled_distance
convert_k_to_si = _core.convert_k_to_si

__all__ = ["to_si", "from_si", "scaled_distance", "convert_k_to_si"]

# Factor que multiplica un valor en la unidad dada para pasarlo a SI (m, kg, mm/s, mm).
_TO_SI = {
    "m": 1.0,
    "ft": _core.m_per_ft,
    "kg": 1.0,
    "lb": _core.kg_per_lb,
    "mm/s": 1.0,
    "in/s": _core.mm_per_in,
    "mm": 1.0,
    "in": _core.mm_per_in,
}


def _factor(unit: str) -> float:
    try:
        return _TO_SI[unit]
    except (KeyError, TypeError):
        raise ValueError(
            f"Domain error [BlastingPy]: parameter 'unit' must be one of {sorted(_TO_SI)}. Received: {unit!r}"
        ) from None


def _apply(result):
    # Escalar de entrada -> float de Python; arreglo de entrada -> arreglo.
    return float(result) if result.ndim == 0 else result


def to_si(value, unit: str):
    """Convert ``value`` (float or numpy array) from ``unit`` to SI (m, kg, mm/s, mm)."""
    return _apply(np.asarray(value, dtype=float) * _factor(unit))


def from_si(value, unit: str):
    """Convert ``value`` (float or numpy array) from SI (m, kg, mm/s, mm) to ``unit``."""
    return _apply(np.asarray(value, dtype=float) / _factor(unit))
