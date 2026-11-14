"""Near field: PPV close to a charge (Holmberg-Persson, Heelan) and a whole blast simulated by seed superposition.

Units are SI: lengths in m, charge in kg (kg/m along a hole), time in s, wave and detonation velocities in m/s, pressure
in Pa. Every PPV and waveform is in mm/s (the Holmberg-Persson functions return the unit that K carries: mm/s with the
published constants, kg and m).

Geometry
--------
* Around one hole the charge lies on the z axis, between ``z_start`` and ``z_end`` (``z_bottom`` and ``z_top`` in
  :class:`Column`); ``r`` is the radial distance to that axis and ``z`` the height of the point.
* A whole blast is in plan view: holes at (``x_m``, ``y_m``) and the monitor at (x, y), all taken at the same level.

Models (the core functions are re-exported with their names) and how far each one was verified
---------------------------------------------------------------------------------------------
* Holmberg-Persson. :func:`hp_original` is the integral form, ``K [l int dz / (r^2 + z^2)^(beta / (2 alpha))]^alpha``
  (NIOSH 2008, eq. 5) [V1]; its closed form for ``beta = 2 alpha`` is :func:`hp_term` (Onederra & Esen 2004, eq. 2)
  [V2]. :func:`hp_niosh` is the NIOSH modified form with the mean distance (eqs. 8-9) [V1], which answers the
  mathematical objection to the integral (the exponent alpha does not commute with the sum; Hustrulid & Lu 2002 as
  reported by NIOSH). Both are static: no detonation velocity, no arrival times. :func:`hp_grid` evaluates either on a
  grid; :data:`HOLMBERG_PERSSON_SURFACE` holds the constants that NIOSH reports for surface blasts [V2].
  :func:`ppv_critical` is ``T Vp / E`` (Onederra & Esen 2004, eq. 11, citing Persson, Holmberg & Lee 1994) [V2]; the
  static E is inferred.
* Heelan. :func:`heelan_element` is the radiation of a short cylindrical charge (Heelan 1953, eqs. 12-13) with the
  explicit pressure function of Blair (2007, eqs. 8 and 13-16) [V1]. :func:`heelan_column` adds the elements of a
  column initiated at the detonation velocity, each with its non-linear weight (Blair 2008, eqs. 1-3 and 22) [V1].
  Inferred, not read in the source: the initiation time of each element (the paper describes it without a formula)
  and mirroring the weights when the primer is above mid-column (the paper only treats a primer nearer the bottom).
  :func:`heelan_ppv` returns the peak of that waveform with the validity limits of Blair (2007) as warnings. It is
  not a far-field solution in the strict sense.
* Whole blast. :func:`simulate_blast` superposes one seed waveform per hole (Hinzen 1988, eqs. 3-7; Blair 1999,
  eq. 1a) [V1], with the amplitude of each hole taken from the site law of :mod:`blastingpy.attenuation` and,
  optionally, the screening of Blair (1999, eqs. 10b-10c) [V1] or the non-linear weights of Blair (2008): charge
  scaling (eqs. 7-12) or damage (eqs. 16-21) [V1]. Inferred, not read in the sources: holes fired at the same time
  do not count as already fired for each other; the screening rectangle of Blair (1999) is centred on the line from
  the hole to the monitor; holes and monitor are at the same level; g is in m^2; an arrival between two samples is
  spread linearly over them. :func:`monte_carlo_blast` scatters delays and amplitudes (Blair 1999, eqs. 1b and 7)
  [V1]; the normal delay scatter is a design choice. :func:`delay_sweep` is the systematic variation of the delay
  between two holes of Hinzen (1988). The 8 ms window of ``max_charge_8ms_kg`` is Duvall et al. (1963) as reported
  by Blair (1999, p. 337) [V2].

Not included (not verified, or out of scope), and the paper that would unlock it
---------------------------------------------------------------------------------
* The scaled Heelan model and its constant gamma_n: Blair & Minchinton (2006), Fragblast-8, pp. 152-159. The column
  here is the standard Heelan element with Blair's (2008) weights, not that model.
* The exact solution for the short charge (discrete wavenumber): Blair (2007) and Meredith (1990); out of scope.
* The fluctuating seed of the Monte Carlo (Blair 1999, eqs. 2-4 and 14): its normalization is derived in Blair
  (1993a), Int. J. Numer. Anal. Methods Geomech. 17: 95-118.
* The distribution of the delay scatter: Blair (1999) publishes neither its type nor its deviation; the user gives it.
* The broadening of the seed with Q (Blair 1990, Fragblast-3) and the time scaling of the analytic seed of Blair
  (2008, eqs. 23-24), which the paper does not state.
* The composition of the longitudinal, transverse and vertical components (Blair 1999, eqs. 15a-15c): the sign
  convention depends on a figure. A simulation here is of one component, the seed's.
* Airblast screening (Blair 1999, eq. 10d): the printed exponent is ambiguous.
* Holmberg & Persson's own papers (1979, Tunnelling '79, pp. 270-283) and Persson, Holmberg & Lee (1994, p. 247):
  with them the integral, the surface constants and the critical PPV would be [V1]. Their 700-1000 mm/s damage limit
  is for Swedish igneous rocks (as reported by NIOSH) and is not offered as a general criterion.
* The objection of Blair & Minchinton to Holmberg-Persson in their own words (1996, Fragblast-5, pp. 121-130; 1997,
  Fragblast 1(1): 59-72): it is known here only as summarized by Onederra & Esen (2004) and Blair (2007, 2008).
* The PPV at the end of the crushing zone (Onederra & Esen 2004, eq. 5), with the equilibrium pressure of Liu &
  Katsabanis (1993) and the crushing radius of Esen et al. (2003): an exponent could not be verified and they need
  non-ideal detonation pressures, which the library does not compute.
* Heelan's energy partition and his SV/P ratio of 1.6: the printed formulas are inconsistent.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd

from . import airblast
from ._blastingpy_core import _nearfield as _nf
from .attenuation import Law, Level, SiteLaw, predict_ppv
from .io import Record
from .waveform import _fft

HpModel = _nf.HpModel
hp_point = _nf.hp_point
hp_term = _nf.hp_term
hp_original = _nf.hp_original
hp_mean_distance = _nf.hp_mean_distance
hp_niosh = _nf.hp_niosh
hp_damage_extent = _nf.hp_damage_extent
ppv_critical = _nf.ppv_critical
Medium = _nf.Medium
PressureSource = _nf.PressureSource
Waveform = _nf.Waveform
pressure_function = _nf.pressure_function
g_function = _nf.g_function
g_peak = _nf.g_peak
mean_frequency_factor = _nf.mean_frequency_factor
scaled_frequency = _nf.scaled_frequency
heelan_amplitude = _nf.heelan_amplitude
heelan_element = _nf.heelan_element
Column = _nf.Column
column_weights = _nf.column_weights
heelan_column = _nf.heelan_column
vector_peak = _nf.vector_peak
superpose = _nf.superpose
screening_counts = _nf.screening_counts
screening_factor = _nf.screening_factor
charge_scaling_weights = _nf.charge_scaling_weights
damage_weights = _nf.damage_weights
max_charge_per_window = _nf.max_charge_per_window
monte_carlo_peaks = _nf.monte_carlo_peaks

__all__ = ["HpModel", "hp_point", "hp_term", "hp_original", "hp_mean_distance", "hp_niosh", "hp_damage_extent",
           "ppv_critical", "Medium", "PressureSource", "Waveform", "pressure_function", "g_function", "g_peak",
           "mean_frequency_factor", "scaled_frequency", "heelan_amplitude", "heelan_element", "Column",
           "column_weights", "heelan_column", "vector_peak", "superpose", "screening_counts", "screening_factor",
           "charge_scaling_weights", "damage_weights", "max_charge_per_window", "monte_carlo_peaks",
           "HOLMBERG_PERSSON_SURFACE", "hp_grid", "HeelanResult", "heelan_ppv", "BLAST_COLUMNS", "WEIGHTINGS",
           "BlastResult", "simulate_blast", "MonteCarloResult", "monte_carlo_blast", "delay_sweep"]

HOLMBERG_PERSSON_SURFACE = {
    "k": 700.0, "alpha": 0.7, "beta": 1.5,
    "citation": "Lundborg et al. (1978), as reported by Iverson, Kerkering & Hustrulid (2008), Application of the "
                "NIOSH-Modified Holmberg-Persson Approach to Perimeter Blast Design, NIOSH (PDF pp. 3 and 31)",
    "verification": "V2",
    "notes": (
        "PPV in mm/s, charge in kg, distance in m.",
        "NIOSH attributes them to a large set of USBM measurements of surface blasts and repeats them as the values "
        "suggested by Holmberg and Persson. It names no rock type: they come from surface blasting, not from a rock.",
        "They are not a default: no function here uses them unless you pass them. Fit the constants of your site.",
    ),
}
BLAST_COLUMNS = ("x_m", "y_m", "charge_kg", "firing_time_s")
WEIGHTINGS = ("linear", "charge-scaling", "damage")

_HP_MODELS = {"original": _nf.hp_original, "niosh": _nf.hp_niosh}
_MAX_SAMPLES = 2_000_000  # tope de muestras de una onda: evita reservar vectores enormes por un dato mal puesto
_MAX_SIGMA_S = 1.0
# Umbrales de plausibilidad (avisos, no errores). Son elecciones de diseño, no valores de una fuente: una voladura
# dura segundos y un retardo entre pozos, milisegundos; por encima de esto lo más probable es que estén en ms.
_FIRING_SPAN_S = 20.0
_LONG_DELAY_S = 1.0
_STEPS_PER_PULSE = 10.0  # heelan_ppv avisa si el paso pasa de 1/(10 gamma): elección de diseño (el automático es 1/40)
# Más umbrales de diseño (revisión final; no vienen de una fuente). Una onda sísmica en roca no viaja a menos de
# 100 m/s: por debajo, lo más probable es que esté en km/s. Con fs ≥ 10 veces la frecuencia dominante de la semilla,
# el reparto lineal de una llegada entre dos muestras resta a lo más sin²(π/10) ≈ 9,5 % del pico.
_SLOW_WAVE_M_S = 100.0
_SAMPLES_PER_PERIOD = 10.0
_MIN_SCREENING = 0.2  # el piso de screening_factor (Blair 1999, p. 346): por debajo, S_n de la ec. 20 no tiene datos
_G_NEAREST_MONITOR_M = 99.0  # el monitor más cercano con que se ajustó g (Blair 1999, Fig. 7b)
_MEDIAN = "amplitudes are the site law's median: about half of the holes exceed it"
_ETA = 0.2  # primera estimación de eta de Blair (2008); el valor por omisión de simulate_blast
_OUTSIDE = " is outside the fitted range"  # texto del aviso de extrapolación del núcleo (attenuation.hpp)
_G = 23337.0  # g de Blair (1999), Fig. 7c; el valor por omisión del núcleo
_LINEAR_SUM = ("the P and SV pulses are added linearly where they overlap in time; Blair (2007, p. E37) questions that "
               "sum: the Heelan model only makes sense where the two waves do not overlap")
_FULL_BLAST_LIMITS = ("Blair's (2008) models have no free faces and apply the non-linearity as weights on waveforms "
                      "of a fixed shape; the shape itself does not change (pp. 237, 246-247)")
_CHARGE_SCALING = ("weighting='charge-scaling' (Blair 2008, eq. 12): its author reports that this model predicts an "
                   "almost linear, unrealistic decay of the amplitude with time and does not recommend it over the "
                   "damage model (pp. 243, 246)")
_SCREENING = (f"screening (Blair 1999, eqs. 10b-10c): g = {_G:g} m^2 was fitted to one set of field data (its unit is "
              "inferred) and the factor has a floor of 0.2 that the author sets for lack of near-field data (p. 346)")
_DELAY_DISTRIBUTION = ("the delay scatter is the user's: a normal distribution with delay_sigma_s is a design choice, "
                       "because Blair (1999) publishes neither the type nor the deviation of the delay distribution")
_FIXED_SEED = ("the fluctuating seed of Blair (1999) is not modelled: every hole radiates the same seed, so the "
               "variability of the waveform shape is under-represented")


class _Loud(str):
    """A note of the final review of sprint 5: it goes in ``warnings`` and is also emitted as a UserWarning."""


def _emitted(notes) -> tuple[str, ...]:
    """The result's ``warnings``; the loud notes are also emitted, pointing at the line that called the public
    function (stacklevel 3: this helper, the public function, its caller)."""
    for note in notes:
        if isinstance(note, _Loud):
            warnings.warn(str(note), UserWarning, stacklevel=3)
    return tuple(str(note) for note in notes)


def _error(name: str, detail: str) -> ValueError:
    return ValueError(f"Domain error [BlastingPy]: parameter '{name}' {detail}")


def _number(v) -> bool:
    """A finite real number (a bool is not one)."""
    return isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_)) and math.isfinite(v)


def _positive(name: str, v, why: str = "") -> float:
    if not (_number(v) and v > 0):
        raise _error(name, f"{why}must be finite and > 0. Received: {v!r}")
    return float(v)


def _axis(name: str, values) -> np.ndarray:
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise _error(name, f"must be a 1-D array of numbers. Received: {values!r}") from None
    if a.ndim != 1:
        raise _error(name, f"must be a 1-D array. Received shape: {a.shape}")
    return a


# --- Holmberg-Persson on a grid ---------------------------------------------------------------------------------------

def hp_grid(model: str, *, k: float, alpha: float, beta: float, linear_charge_kg_m: float, z_start: float,
            z_end: float, r, z) -> np.ndarray:
    """Holmberg-Persson PPV on a grid: ``r`` and ``z`` are 1-D arrays (m) and the result has shape (len(z), len(r)).

    ``model`` is ``"original"`` (:func:`hp_original`, the integral of NIOSH eq. 5) or ``"niosh"`` (:func:`hp_niosh`,
    the modified form with the mean distance). The charge of ``linear_charge_kg_m`` (kg/m) lies on the axis between
    ``z_start`` and ``z_end``. With ``"original"`` the points on the charged axis (``r == 0`` within the charge, with
    ``beta >= alpha``) are NaN: the integral diverges there. The modified form is finite everywhere. The result is in
    the unit of ``k`` (mm/s with constants fitted in mm/s, kg and m).
    """
    if not isinstance(model, str) or model not in _HP_MODELS:
        raise _error("model", f"must be one of {list(_HP_MODELS)}. Received: {model!r}")
    rr, zz = _axis("r", r), _axis("z", z)
    _nf.hp_niosh(k, alpha, beta, linear_charge_kg_m, z_start, z_end, 1.0, z_start)  # valida constantes y tramo
    func = _HP_MODELS[model]
    out = np.empty((zz.size, rr.size))
    for i, zi in enumerate(zz):
        for j, rj in enumerate(rr):
            diverges = model == "original" and rj == 0.0 and beta >= alpha and z_start <= zi <= z_end
            out[i, j] = math.nan if diverges else func(k, alpha, beta, linear_charge_kg_m, z_start, z_end, rj, zi)
    return out


# --- Heelan PPV around a hole ----------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, eq=False)
class HeelanResult:
    """Vector PPV of a charge column at one point, with its waveform and the validity warnings of the model."""

    ppv_mm_s: float  # pico de sqrt(radial^2 + axial^2), tomado de la onda
    time_s: np.ndarray  # desde la iniciación del cebo
    radial_mm_s: np.ndarray
    axial_mm_s: np.ndarray
    scaled_frequency: float  # Omega_A = c gamma a / Vp
    warnings: tuple[str, ...]


def heelan_ppv(source, medium, column, r: float, z: float, *, sample_rate_hz: float | None = None,
               duration_s: float | None = None) -> HeelanResult:
    """Peak vector velocity (mm/s) radiated by a charge column at the point (``r``, ``z``), Heelan's solution.

    ``source`` is a :class:`PressureSource`, ``medium`` a :class:`Medium` and ``column`` a :class:`Column` (its
    ``hole_radius_m`` is the radius of the hole, the convention of Blair 2007; Heelan defines it as the radius of the
    equivalent cavity). ``r`` must be > 0: the solution is not defined on the axis of the hole. The PPV is always the
    peak of the waveform of :func:`heelan_column`, never ``G_n a / r``.

    Time starts when the primer initiates. Without ``sample_rate_hz`` the step is ``1 / (40 gamma)``; without
    ``duration_s`` the record lasts until the last arrival (initiation of the element at the detonation velocity plus
    its S travel time) plus ``40 / gamma``, or plus ``(n + 4 sqrt(n) + 2) / gamma`` if that is longer (the pulse of
    g(tau) lasts about ``n + 4 sqrt(n)`` in ``tau = gamma t``). A grid of more than 2 000 000 samples is an error. The
    pulse lasts a few ``1 / gamma``, so the sample rate of a seismograph is usually far too low here: leave
    ``sample_rate_hz`` out unless you need a particular grid.

    ``warnings`` always says that the P and SV pulses are added linearly where they overlap, and adds, with the
    computed number, each limit of Blair (2007, pp. E37-E40) that is not met: ``Omega_A > 0.1`` (Heelan overestimates
    the exact solution), ``Omega_A r / a < pi`` (near field) and element half-length / radius > 2; that the non-linear
    column weights have no experimental validation when ``column.charge_exponent < 1``; that the peak may be
    truncated when ``duration_s`` ends before the last arrival plus ``(n + 4 sqrt(n) + 2) / gamma`` (with n = 14
    and 11 / gamma after the last arrival the peak can fall to about 0.73 of the true one); and that it may be
    under-resolved when the time step is longer than ``1 / (10 gamma)``. Nothing is refused on those grounds: the
    result is yours to judge.
    """
    _nf.heelan_column(source, medium, column, r, z, [])  # sin instantes: solo valida (fuente, medio, columna y r > 0)
    gamma, a = source.gamma_per_s, column.hole_radius_m
    half = (column.z_top - column.z_bottom) / (2.0 * column.elements)
    centres = column.z_bottom + (2.0 * np.arange(column.elements) + 1.0) * half
    last = float(np.max(np.abs(centres - centres[column.primer_element - 1]) / column.vod_m_s
                        + np.hypot(r, z - centres) / medium.s_velocity_m_s))
    pulse = source.n + 4.0 * math.sqrt(source.n) + 2.0  # duración de g(tau) en tau, con margen (elección de diseño)
    fs = 40.0 * gamma if sample_rate_hz is None else _positive("sample_rate_hz", sample_rate_hz)
    duration = last + max(40.0, pulse) / gamma if duration_s is None else _positive("duration_s", duration_s)
    samples = math.ceil(duration * fs) + 1
    if samples > _MAX_SAMPLES:
        raise _error("sample_rate_hz", f"and duration_s give a time grid of {samples} samples (at most "
                                       f"{_MAX_SAMPLES}): pass a lower sample_rate_hz or a shorter duration_s "
                                       f"explicitly. Grid: {fs:g} Hz for {duration:g} s")
    time = np.arange(samples) / fs
    wave = _nf.heelan_column(source, medium, column, r, z, time.tolist())

    omega = _nf.scaled_frequency(source, medium, a)
    notes = []
    if omega > 0.1:
        notes.append(f"scaled frequency Omega_A = {omega:.3g} is above 0.1: Heelan overestimates the exact solution "
                     "at any distance (Blair 2007, pp. E37-E40)")
    if omega * r / a < math.pi:
        notes.append(f"near field: Omega_A r / a = {omega * r / a:.3g} is below pi; the P and S waves are not "
                     "uncoupled there (they uncouple only above about 5) and the far-field condition of the solution, "
                     "Omega r / a large, is not met (Blair 2007, pp. E37-E40)")
    if half / a > 2.0:
        needed = math.ceil((column.z_top - column.z_bottom) / (4.0 * a) - 1e-9) | 1  # impar
        notes.append(f"element half-length / hole radius = {half / a:.3g} is above 2: Heelan agrees with the exact "
                     f"solution up to about 2 and is off by about 20 % at 10 (Blair 2007, pp. E37-E40); use at least "
                     f"{needed} elements")
    notes.append(_LINEAR_SUM)
    if column.charge_exponent < 1.0:
        notes.append(f"charge_exponent = {column.charge_exponent:g} is below 1: the non-linear weights of the column "
                     "(Blair 2008, eqs. 1-3 and 22) have no experimental validation, as their author says "
                     "(pp. 237, 246-247)")
    if duration_s is not None and duration < last + pulse / gamma:
        notes.append(f"duration_s = {duration:g} s ends before the last arrival plus (n + 4 sqrt(n) + 2) / gamma "
                     f"({last + pulse / gamma:.4g} s), the length of the pulse: the peak may be truncated")
    if fs < _STEPS_PER_PULSE * gamma:  # paso mayor que 1/(10 gamma)
        notes.append(f"time step = {gamma / fs:.3g} / gamma is longer than 1 / ({_STEPS_PER_PULSE:g} gamma): the "
                     "pulse is under-resolved and the peak may be cut short; leave sample_rate_hz out for the "
                     "automatic step, 1 / (40 gamma)")
    return HeelanResult(_nf.vector_peak(wave), time, np.asarray(wave.radial_mm_s), np.asarray(wave.axial_mm_s),
                        omega, tuple(notes))


# --- Whole blast: seed superposition from the hole table --------------------------------------------------------------

def _seed(seed, sample_rate_hz) -> tuple[list[float], float, list[str]]:
    """The seed normalized to unit peak (Blair 1999), its sample rate and the warnings about it.

    ``seed`` is an array or a (Record, channel name) pair (a tuple or a list).
    """
    notes = []
    if isinstance(seed, (tuple, list)) and len(seed) == 2 and isinstance(seed[0], Record):
        record, name = seed
        if not isinstance(name, str) or name not in record.channels:
            raise _error("seed", f"names the channel {name!r}, which is not in the record. "
                                 f"Channels: {list(record.channels)}")
        fs = float(record.sample_rate_hz)
        if sample_rate_hz is not None and not (_number(sample_rate_hz) and math.isclose(sample_rate_hz, fs)):
            raise _error("sample_rate_hz", f"disagrees with the sample rate of the record ({fs:g} Hz); leave it out. "
                                           f"Received: {sample_rate_hz!r}")
        values = record.channels[name]
        if name == "mic":
            notes.append("the seed is the 'mic' channel of the record: that is air pressure (Pa), not ground vibration")
    else:
        if sample_rate_hz is None:
            raise _error("sample_rate_hz", "is required when the seed is an array: the sample rate is never guessed")
        fs, values = _positive("sample_rate_hz", sample_rate_hz), seed
    try:
        s = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        s = np.empty((0, 0))
    if s.ndim != 1 or s.size == 0 or not np.isfinite(s).all():
        raise _error("seed", "must be a non-empty 1-D array of finite numbers or a (Record, channel name) pair")
    peak = float(np.abs(s).max())
    if peak == 0.0:
        raise _error("seed", "is zero everywhere: it cannot be normalized to unit peak")
    dominant = _fft(s, fs)[0] if s.size > 1 else math.nan  # NaN (semilla plana o de una muestra): sin aviso
    if dominant > fs / _SAMPLES_PER_PERIOD:
        loss = math.sin(math.pi * dominant / fs) ** 2  # peor caso: el pico pasa a cos²(π f/fs)
        notes.append(_Loud(f"the seed's dominant frequency, {dominant:.4g} Hz, is above a tenth of its sample rate "
                           f"({fs / _SAMPLES_PER_PERIOD:g} Hz): an arrival between two samples is spread linearly over "
                           f"them, which lowers its peak by up to sin^2(pi f / fs) = {loss:.0%} when it falls "
                           f"mid-sample; use a seed recorded at a higher sample rate (at least "
                           f"{_SAMPLES_PER_PERIOD * dominant:.0f} Hz)"))
    return (s / peak).tolist(), fs, notes


def _law_charge_exponent(law: SiteLaw) -> float:
    """A of ``v = K W^A d^-B``. ``SiteLaw.charge_exponent`` is A only for a general law: for the scaled-distance laws
    it is the exponent of W inside the scaled distance (1/2 or 1/3), so A = beta times it (Blair 2008, eqs. 14-15)."""
    return law.charge_exponent if law.law == Law.General else law.beta * law.charge_exponent


def _vibration_law(law) -> SiteLaw:
    """Refuse an airblast law (its K is in Pa, not mm/s), with the check of :mod:`blastingpy.airblast`."""
    if not isinstance(law, SiteLaw):
        raise _error("law", f"must be an attenuation.SiteLaw (attenuation.fit or attenuation.make_site_law). "
                            f"Received: {law!r}")
    try:
        airblast._airblast_law(law)
    except ValueError:
        return law
    raise _error("law", f"must be a ground-vibration law, and this one is an airblast law (overpressure in Pa). "
                        f"Received method id: {law.info.method_id!r}")


def _column(holes: pd.DataFrame, name: str, *, positive: bool = False) -> np.ndarray:
    try:
        a = holes[name].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise _error("holes", f"column {name!r} must be numeric") from None
    bad = ~np.isfinite(a) | ((a <= 0) if positive else False)
    if bad.any():
        raise _error("holes", f"column {name!r}, rows {list(holes.index[bad])}: must be finite"
                              + (" and > 0" if positive else ""))
    return a


def _check_span(name: str, last_s: float, fs: float, seed_samples: int, hint: str) -> None:
    samples = math.ceil(last_s * fs) + seed_samples + 1
    if samples > _MAX_SAMPLES:
        raise _error(name, f"gives a waveform of {samples} samples at {fs:g} Hz (at most {_MAX_SAMPLES}): {hint}")


def _prepare(holes, monitor, seed, sample_rate_hz, law, wave_velocity_m_s, *, weighting="linear", spacing_m=None,
             charge_exponent=None, eta=_ETA, screening=False):
    """Validated inputs of a blast: (seed, fs, arrivals table, firing times, charges, warnings)."""
    if not isinstance(holes, pd.DataFrame):
        raise _error("holes", "must be a pandas DataFrame")
    missing = [c for c in BLAST_COLUMNS if c not in holes.columns]
    if missing:
        raise _error("holes", f"is missing the column {missing[0]!r}. Required columns: {list(BLAST_COLUMNS)}")
    if holes.empty:
        raise _error("holes", "must have at least one row")
    x, y, t = (_column(holes, c) for c in ("x_m", "y_m", "firing_time_s"))
    w = _column(holes, "charge_kg", positive=True)
    try:
        mx, my = monitor
    except (TypeError, ValueError):
        mx = my = None
    if not (_number(mx) and _number(my)):
        raise _error("monitor", f"must be an (x, y) pair of finite numbers, in m. Received: {monitor!r}")
    mx, my = float(mx), float(my)
    samples, fs, notes = _seed(seed, sample_rate_hz)
    law = _vibration_law(law)
    velocity = _positive("wave_velocity_m_s", wave_velocity_m_s)
    if velocity < _SLOW_WAVE_M_S:
        notes.append(_Loud(f"wave velocity looks like km/s: wave_velocity_m_s = {velocity:g}, and it must be in m/s"))
    if not isinstance(weighting, str) or weighting not in WEIGHTINGS:
        raise _error("weighting", f"must be one of {list(WEIGHTINGS)}. Received: {weighting!r}")
    if not isinstance(screening, (bool, np.bool_)):
        raise _error("screening", f"must be a bool. Received: {screening!r}")
    if screening and weighting != "linear":
        raise _error("screening", f"is defined only with weighting='linear' (Blair 1999, eq. 10c). The damage model "
                                  f"carries its own screening and no source combines screening with the charge-scaling "
                                  f"model. Received: screening=True with weighting={weighting!r}")
    if weighting != "linear" or screening:
        spacing = _positive("spacing_m", spacing_m, "is required (the mean spacing between holes, m) and ")
        if holes.duplicated(["x_m", "y_m"]).any():  # el núcleo lo rechaza igual, pero nombra posiciones, no filas
            raise _error("holes", f"rows {list(holes.index[holes.duplicated(['x_m', 'y_m'], keep=False)])} are at the "
                                  "same point: screening and the non-linear weightings need the distance between "
                                  "holes to be > 0")
    ignored = [name for name, given, used in (
        ("spacing_m", spacing_m is not None, weighting != "linear" or screening),
        ("eta", eta != _ETA, weighting == "damage"),
        ("charge_exponent", charge_exponent is not None, weighting == "charge-scaling")) if given and not used]
    if ignored:
        one = len(ignored) == 1
        notes.append(f"{' and '.join(ignored)} {'is' if one else 'are'} ignored: weighting={weighting!r}"
                     f"{' without screening' if weighting == 'linear' else ''} does not use {'it' if one else 'them'}")
    span = float(t.max() - t.min())
    step = float(np.diff(np.sort(t)).max(initial=0.0))  # el mayor intervalo entre pozos consecutivos
    if span > _FIRING_SPAN_S or step > _LONG_DELAY_S:
        notes.append(_Loud(f"firing times look like milliseconds: firing_time_s spans {span:g} s, with up to "
                           f"{step:g} s between consecutive holes, and it must be in seconds"))

    d = np.hypot(x - mx, y - my)
    if (d == 0.0).any():
        raise _error("monitor", f"coincides with the hole of row {list(holes.index[d == 0.0])}: the distance to the "
                                "monitor must be > 0")
    arrival = t + d / velocity
    arrival -= arrival.min()
    _check_span("holes", float(arrival.max()), fs, len(samples),
                "check that firing_time_s is in seconds, or lower the sample rate of the seed")

    median = np.empty(len(holes))
    from_law: dict[str, list] = {}
    for i, label in enumerate(holes.index):
        prediction = predict_ppv(law, d[i], w[i], Level.Median)
        median[i] = prediction.value
        for message in prediction.warnings:
            # una nota por causa y no por pozo: se quita el valor del pozo («distance 20 m is…» → «distance is…»)
            head, outside, tail = message.partition(_OUTSIDE)
            cause = head.rsplit(" ", 2)[0] + outside + tail if outside else message
            from_law.setdefault(cause, []).append(label)
    notes += [f"{cause} (rows {rows})" for cause, rows in from_law.items()]

    weight, counts = np.ones(len(holes)), None
    if weighting == "charge-scaling":
        own = _law_charge_exponent(law)
        if charge_exponent is None:
            if not 0.0 < own <= 1.0:
                raise _error("law", f"has the charge exponent A = {own:g}, outside (0, 1]: weighting='charge-scaling' "
                                    "is not defined for it (A > 1 has no physical meaning, Blair 2008, p. 240)")
            exponent = own
        elif not _number(charge_exponent):
            raise _error("charge_exponent", f"must be a number within (0, 1], or None to take it from the law. "
                                            f"Received: {charge_exponent!r}")
        else:
            exponent = float(charge_exponent)
            if not math.isclose(exponent, own, rel_tol=1e-6):
                notes.append(f"charge_exponent = {exponent:g} is not the charge exponent of the law, A = {own:g} (for "
                             "a square-root or cube-root law, beta times SiteLaw.charge_exponent): the given value is "
                             "used, but the weight C_n / w_n^A is eq. 12 of Blair (2008) only with the law's own A")
        # «Ya inició» con la tolerancia de tiempos del núcleo: los pozos a menos de 1 ns son simultáneos, no cuentan
        # entre sí para el pozo ya iniciado más cercano y conservan el orden de la tabla (orden estable sobre los
        # tiempos redondeados a la tolerancia).
        tolerance = _nf._time_tolerance_s
        nearest, tied = np.full(len(holes), math.inf), []
        for i, label in enumerate(holes.index):
            fired = t < t[i] - tolerance
            if fired.any():
                nearest[i] = np.hypot(x[fired] - x[i], y[fired] - y[i]).min()
            if (np.abs(t - t[i]) <= tolerance).sum() > 1:
                tied.append(label)
        order = np.argsort(np.round(t / tolerance), kind="stable")
        c = np.empty(len(holes))
        c[order] = _nf.charge_scaling_weights(w[order].tolist(), nearest[order].tolist(), spacing, exponent)
        weight = c / w ** exponent  # la amplitud de la ley ya trae w^A: la ec. 12 lo cambia por C_n
        if tied:
            notes.append(f"holes fired at the same time (rows {tied}): with weighting='charge-scaling' the weights of "
                         "tied holes depend on their order in the table; Blair (2008) does not say how to order them")
        notes += [_CHARGE_SCALING, _FULL_BLAST_LIMITS]
    elif weighting == "damage":
        weight = np.asarray(_nf.damage_weights(x.tolist(), y.tolist(), w.tolist(), t.tolist(), mx, my, spacing, eta))
        notes += [f"weighting='damage' (Blair 2008, eq. 21): eta = {eta:g} and g = {_G:g} m^2 are estimates from a "
                  "single source (0.2 is the author's first estimate of eta and 0.3 fitted his one field case, an "
                  "acceleration envelope; g comes from one set of field data in Blair 1999 and its unit is inferred); "
                  "holes and monitor are taken at the same level", _FULL_BLAST_LIMITS]
        # con eta = 0, D_n = 1 y el peso es S_n (ec. 20)
        shield = np.asarray(_nf.damage_weights(x.tolist(), y.tolist(), w.tolist(), t.tolist(), mx, my, spacing, 0.0))
        low, near = list(holes.index[shield < _MIN_SCREENING]), list(holes.index[d < _G_NEAREST_MONITOR_M])
        if low or near:
            found = ([f"the screening factor S_n is below {_MIN_SCREENING:g} (rows {low})"] if low else []) + (
                [f"holes are closer than {_G_NEAREST_MONITOR_M:g} m to the monitor (rows {near})"] if near else [])
            notes.append(_Loud(
                f"weighting='damage' in the near field: {' and '.join(found)}. S_n (Blair 2008, eq. 20) has no floor, "
                f"unlike the screening of Blair (1999, p. 346), floored at {_MIN_SCREENING:g} for lack of near-field "
                f"data, and g was fitted with monitors at {_G_NEAREST_MONITOR_M:g} m or more (Blair 1999, Fig. 7b): "
                "these weights are an extrapolation of the source's screening term"))
    elif screening:
        counts = np.asarray(_nf.screening_counts(x.tolist(), y.tolist(), t.tolist(), mx, my, spacing))
        weight = np.array([_nf.screening_factor(di, int(n)) for di, n in zip(d, counts)])
        notes.append(_SCREENING)

    table = pd.DataFrame({"distance_m": d, "arrival_s": arrival, "amplitude_mm_s": median * weight, "weight": weight},
                         index=holes.index)
    if counts is not None:
        table["screening_count"] = counts
    notes.append(_Loud(_MEDIAN))
    return samples, fs, table, t.tolist(), w.tolist(), notes


def _superposed(samples, fs, table) -> np.ndarray:
    return np.asarray(_nf.superpose(samples, fs, table["arrival_s"].tolist(), table["amplitude_mm_s"].tolist()))


@dataclasses.dataclass(frozen=True, eq=False)
class BlastResult:
    """Simulated waveform of a whole blast at the monitor, in the component of the seed."""

    ppv_mm_s: float  # pico de |v|
    time_s: np.ndarray  # desde la primera llegada
    velocity_mm_s: np.ndarray
    arrivals: pd.DataFrame  # por pozo: distance_m, arrival_s, amplitude_mm_s, weight (y screening_count)
    max_charge_8ms_kg: float  # mayor carga que inicia en una ventana de 8 ms, por tiempos de iniciación
    warnings: tuple[str, ...]


def simulate_blast(holes: pd.DataFrame, monitor: tuple[float, float], seed, *, sample_rate_hz: float | None = None,
                   law: SiteLaw, wave_velocity_m_s: float, weighting: str = "linear", spacing_m: float | None = None,
                   charge_exponent: float | None = None, eta: float = 0.2, screening: bool = False) -> BlastResult:
    """Waveform of a whole blast at ``monitor``: one seed per hole, delayed and scaled (Hinzen 1988; Blair 1999).

    ``holes`` has one row per hole with the columns :data:`BLAST_COLUMNS` (plan-view coordinates in m, charge in kg,
    firing time in s) and ``monitor`` is the (x, y) of the monitoring point.

    Units: ``x_m``, ``y_m``, ``monitor`` and ``spacing_m`` in m; ``charge_kg`` in kg; ``firing_time_s`` in s, not ms;
    ``wave_velocity_m_s`` in m/s; ``sample_rate_hz`` in Hz. Inspect ``arrivals`` before trusting the peak: it shows
    the distance, arrival time, amplitude and weight each hole ended up with, and a unit mistake shows there.

    ``seed`` is the waveform of a single hole: a 1-D array (then ``sample_rate_hz`` is required) or a pair
    ``(record, channel_name)`` with a :class:`blastingpy.io.Record`, whose sample rate is used. It should be the record
    of a single-hole shot in one component, and the simulation is of that component only. It is normalized to unit
    peak (Blair 1999), so only its shape matters, and it is used from its first sample: trim the pretrigger.

    ``law`` is a ground-vibration :class:`blastingpy.attenuation.SiteLaw`, ideally fitted to single-hole shots. The
    amplitude of each hole is the law's **median** prediction with the distance from that hole to the monitor and its
    charge, times a weight. Each hole arrives at ``firing_time_s + distance / wave_velocity_m_s``; times are referred
    to the first arrival.

    ``weighting``:

    * ``"linear"`` (Hinzen 1988; Blair 1999, eq. 1a): weight 1. With ``screening=True`` the weight is the screening
      factor of Blair (1999, eq. 10c), ``1 / (1 + g d^-2 sqrt(N_S))`` with a floor of 0.2, where N_S counts the holes
      fired earlier inside the rectangle of width ``spacing_m`` between the hole and the monitor.
    * ``"charge-scaling"`` (Blair 2008, eq. 12): weight ``C_n / w_n**A`` with C_n of eq. 10. Eq. 12 replaces
      ``w_n**A`` by C_n, and the amplitude of the law already contains ``w_n**A``, so A has to be the law's own charge
      exponent in ``v = K W**A d**-B``. With ``charge_exponent=None`` (the default) it is taken from the law:
      ``beta / 2`` for a square-root law and ``beta / 3`` for a cube-root law (eqs. 14-15), and the fitted exponent
      of the charge for a general law. Mind the name: ``SiteLaw.charge_exponent`` is that A only for a general law;
      for the other two it is the 1/2 or 1/3 inside the scaled distance, so do not pass it here. An explicit
      ``charge_exponent`` that differs from the law's is used, with a warning that gives both. A must be within
      (0, 1]. P = 4 (how sharp the transition of eq. 8 is) and D = 1 (eq. 7) are fixed design choices: the paper asks
      only for a positive integer P and takes D = 1. Holes are taken in firing order; holes fired within 1e-9 s of
      each other are simultaneous: they do not count as already fired for each other and keep the order of the
      table, on which their weights then depend (a warning lists them). Requires ``spacing_m``. Its own author does
      not recommend it over the damage model.
    * ``"damage"`` (Blair 2008, eq. 21): weight ``S_n / D_n``, the accumulated damage of the holes fired earlier and
      their screening. Requires ``spacing_m``; ``eta`` is the damage constant (0.2 and 0.3 are the author's estimates).
      With ``eta=0`` and no hole in the way it equals ``"linear"``.

    ``screening=True`` with ``"damage"`` (which has its own) or ``"charge-scaling"`` is an error: no source defines
    those combinations.

    ``max_charge_8ms_kg`` is the largest charge fired within any 8 ms window (by firing times, not arrivals). The
    8 ms come from Duvall et al. (1963) as reported by Blair (1999, p. 337), who adds that there is nothing special
    about 8 ms: whether two holes separate depends on the seed. ``warnings`` repeats those of the law's predictions
    (extrapolation: one per cause, with the rows) and the caveats of the model chosen, and says when an input looks
    wrong: firing times that span more than 20 s or with more than 1 s between consecutive holes (they look like
    milliseconds), ``wave_velocity_m_s`` below 100 (it looks like km/s), an argument that the chosen weighting does
    not use (``spacing_m``, ``eta``, ``charge_exponent``), a seed taken from the ``"mic"`` channel and a seed whose
    dominant frequency is above a tenth of its sample rate (an arrival between two samples is spread linearly over
    them, which lowers its peak by up to ``sin^2(pi f / fs)``). It always says that the amplitudes are the law's
    median, so about half of the holes exceed them. With ``"damage"`` it warns, with the rows, when a screening factor
    S_n is below 0.2 or a hole is closer than 99 m to the monitor: S_n has no floor and g was fitted with monitors
    at 99 m or more (Blair 1999, Fig. 7b), so those weights extrapolate the source. These last warnings (units, the
    median, the near field of ``"damage"`` and the seed's frequency) are also emitted as ``UserWarning``.
    """
    samples, fs, table, firing, charges, notes = _prepare(
        holes, monitor, seed, sample_rate_hz, law, wave_velocity_m_s, weighting=weighting, spacing_m=spacing_m,
        charge_exponent=charge_exponent, eta=eta, screening=screening)
    v = _superposed(samples, fs, table)
    return BlastResult(float(np.abs(v).max()), np.arange(v.size) / fs, v, table,
                       _nf.max_charge_per_window(firing, charges), _emitted(notes))


@dataclasses.dataclass(frozen=True, eq=False)
class MonteCarloResult:
    """Peaks of the simulations of a blast with scattered delays and amplitudes."""

    peaks_mm_s: np.ndarray  # un pico de |v| por simulación
    mean: float
    std: float  # desviación estándar muestral (ddof = 1); NaN con una sola simulación
    deterministic_ppv_mm_s: float  # sin dispersión: el de simulate_blast
    warnings: tuple[str, ...]

    def percentile(self, q: float) -> float:
        """Percentile ``q`` (0 to 100) of the peaks, mm/s."""
        if not (_number(q) and 0 <= q <= 100):
            raise _error("q", f"must be within [0, 100]. Received: {q!r}")
        return float(np.percentile(self.peaks_mm_s, q))

    def probability_exceeding(self, level_mm_s: float) -> float:
        """Fraction of the simulations whose peak is strictly above ``level_mm_s``."""
        if not _number(level_mm_s):
            raise _error("level_mm_s", f"must be a finite number. Received: {level_mm_s!r}")
        return float(np.mean(self.peaks_mm_s > level_mm_s))


def monte_carlo_blast(holes: pd.DataFrame, monitor: tuple[float, float], seed, *, sample_rate_hz: float | None = None,
                      law: SiteLaw, wave_velocity_m_s: float, delay_sigma_s, amplitude_cv: float = 0.0,
                      simulations: int = 1000, rng_seed: int = 0, **weights) -> MonteCarloResult:
    """Monte Carlo of :func:`simulate_blast` over the delay scatter and the scatter of the site law (Blair 1999).

    The blast is set up exactly as in :func:`simulate_blast` (same arguments and units: m, kg, s, m/s and Hz;
    ``**weights`` takes ``weighting``, ``spacing_m``, ``charge_exponent``, ``eta`` and ``screening``). Run
    :func:`simulate_blast` with the same inputs first and inspect its ``arrivals``. Each simulation then draws, for
    every hole, an arrival ``d_j + delay_sigma_s * z`` and an amplitude ``w_j (1 + amplitude_cv * z')`` with z and z'
    standard normal (the amplitude is eq. 7), and keeps the peak of the waveform.

    * ``delay_sigma_s`` (s): one number for every hole, an array with one value per hole, or the name of a column of
      ``holes``. It is yours to give: the source publishes no delay distribution, and the normal one is a design
      choice. A value above 1 s is an error.
    * ``amplitude_cv`` is the coefficient of variation sigma / mean of the single-hole law (0.25 in Blair's example).
      The default, 0, ignores the scatter of ``law`` (its ``sigma``, in log10), and the nominal amplitudes are the
      law's median, which about half of the holes exceed: with 0 the percentiles and the probability of exceeding a
      level come out too low. When ``law.sigma > 0`` and ``amplitude_cv`` is 0 a warning gives the coefficient of
      variation of a lognormal with that sigma, ``sqrt(exp((sigma ln 10)**2) - 1)`` (0.356 for sigma = 0.15), for
      you to pass.
    * The same ``rng_seed`` gives the same peaks.

    Of what Blair (1999) randomizes, this covers the firing times and the amplitude; the shape of the seed (his
    fluctuating seed) and the polarity of the transverse component are not modelled. The weights and screening counts
    come from the nominal firing times (a design choice: in the source the screening count changes in each
    simulation, because the firing order does).

    ``warnings`` carries those of :func:`simulate_blast` (the median one always) and the caveats of the Monte Carlo;
    the median, ``amplitude_cv``, unit, near-field and seed warnings are also emitted as ``UserWarning``.
    """
    unknown = set(weights) - {"weighting", "spacing_m", "charge_exponent", "eta", "screening"}
    if unknown:
        raise TypeError(f"monte_carlo_blast() got an unexpected keyword argument {sorted(unknown)[0]!r}")
    samples, fs, table, _, _, notes = _prepare(holes, monitor, seed, sample_rate_hz, law, wave_velocity_m_s, **weights)
    if isinstance(delay_sigma_s, str):
        if delay_sigma_s not in holes.columns:
            raise _error("delay_sigma_s", f"names the column {delay_sigma_s!r}, which is not in holes")
        sigma = holes[delay_sigma_s]
    else:
        sigma = delay_sigma_s
    try:
        sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (len(holes),))
    except (TypeError, ValueError):
        raise _error("delay_sigma_s", f"must be a number, an array with one value per hole ({len(holes)}) or the name "
                                      f"of a column of holes. Received: {delay_sigma_s!r}") from None
    if not (np.isfinite(sigma) & (sigma >= 0) & (sigma <= _MAX_SIGMA_S)).all():
        raise _error("delay_sigma_s", f"must be finite, >= 0 and at most {_MAX_SIGMA_S:g} s (it is the scatter of a "
                                      f"blasting delay, in seconds). Received: {delay_sigma_s!r}")
    for name, value, least in (("simulations", simulations, 1), ("rng_seed", rng_seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < least:
            raise _error(name, f"must be an integer >= {least}. Received: {value!r}")
    if amplitude_cv == 0 and law.sigma > 0:
        cv = math.sqrt(math.expm1((law.sigma * math.log(10.0)) ** 2))  # CV de una lognormal (inferido)
        notes.append(_Loud(f"amplitude_cv = 0 leaves out the scatter of the site law (sigma = {law.sigma:g} in log10): "
                           "the peaks scatter only with the delays, so the percentiles and the probability of "
                           "exceeding a level come out too low. The coefficient of variation of a lognormal with that "
                           f"sigma is sqrt(exp((sigma ln 10)^2) - 1) = {cv:.3g}: pass amplitude_cv={cv:.3g} to "
                           "include it"))
    peaks = np.asarray(_nf.monte_carlo_peaks(samples, fs, table["arrival_s"].tolist(), table["amplitude_mm_s"].tolist(),
                                             sigma.tolist(), amplitude_cv, int(simulations), int(rng_seed)))
    return MonteCarloResult(peaks, float(peaks.mean()), float(peaks.std(ddof=1)) if peaks.size > 1 else math.nan,
                            float(np.abs(_superposed(samples, fs, table)).max()),
                            _emitted(notes + [_DELAY_DISTRIBUTION, _FIXED_SEED]))


def delay_sweep(seed, *, sample_rate_hz: float | None = None, delays_s, amplitude_ratio: float = 1.0) -> pd.DataFrame:
    """Peak of two seeds against the delay between them: the systematic delay variation of Hinzen (1988, p. 441).

    For each delay of ``delays_s`` (s, >= 0) the seed (as in :func:`simulate_blast`, normalized to unit peak) is added
    to a copy of itself delayed and scaled by ``amplitude_ratio`` (> 0). Returns ``delay_s``, ``peak`` (in units of
    the peak of one seed) and ``ratio_to_simultaneous`` (``peak`` over the peak with both fired together,
    ``1 + amplitude_ratio``): 1 at delay 0, and ``max(1, amplitude_ratio) / (1 + amplitude_ratio)`` once the delay
    exceeds the length of the seed. The best delay belongs to that seed, that is, to its site and distance.

    ``attrs["warnings"]`` is a tuple, empty unless an input looks wrong: a delay above 1 s (it looks like
    milliseconds), a seed taken from the ``"mic"`` channel or a seed whose dominant frequency is above a tenth of its
    sample rate (an arrival between two samples is spread linearly over them and its peak drops). The delays in
    milliseconds and the seed's frequency are also emitted as ``UserWarning``.
    """
    samples, fs, notes = _seed(seed, sample_rate_hz)
    ratio = _positive("amplitude_ratio", amplitude_ratio)
    delays = _axis("delays_s", delays_s)
    if not (np.isfinite(delays) & (delays >= 0)).all():
        raise _error("delays_s", f"must be finite and >= 0. Received: {delays_s!r}")
    if delays.size:
        _check_span("delays_s", float(delays.max()), fs, len(samples), "check that the delays are in seconds")
        if delays.max() > _LONG_DELAY_S:
            notes.append(_Loud(f"delays look like milliseconds: the longest delay is {delays.max():g} s and "
                               "delays_s must be in seconds"))
    peaks = [float(np.abs(_nf.superpose(samples, fs, [0.0, delay], [1.0, ratio])).max()) for delay in delays.tolist()]
    out = pd.DataFrame({"delay_s": delays, "peak": peaks, "ratio_to_simultaneous": np.array(peaks) / (1.0 + ratio)})
    out.attrs["warnings"] = _emitted(notes)
    return out
