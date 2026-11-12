/**
 * @file criteria_curve.hpp
 * @brief Curva límite de PPV por frecuencia, hecha de tramos (velocidad constante, desplazamiento constante, recta
 *        lineal o log-log), y su evaluación.
 *
 * Convenciones:
 * - cada tramo cubre [f_min, f_max); el último incluye su f_max; en un quiebre manda el tramo superior;
 * - desplazamiento constante d: v = 2π·f·d (v1 guarda d);
 * - las unidades de v1 y v2 son las de quien llama: la función no convierte nada (el perfil pasa a SI antes);
 * - los quiebres suelen salir de intersection_frequency, por eso la continuidad entre tramos se compara con
 *   tolerancia relativa de 1e-9.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "blastingpy/validation.hpp"

namespace blastingpy {

enum class SegmentKind { ConstantVelocity, ConstantDisplacement, Linear, LogLog };

struct Segment {
    SegmentKind kind;
    double f_min;
    double f_max;  // +inf solo en el último tramo y solo con ConstantVelocity
    double v1;     // velocidad en f_min (o el desplazamiento en ConstantDisplacement)
    double v2 = validation::unset;  // velocidad en f_max; solo Linear y LogLog
};

/// segment es el índice (base 0) del tramo usado; sin tramo, defined = false, limit = NaN y segment = −1.
struct CurveLimit {
    bool defined;
    double limit;
    int segment;
};

namespace criteria_curve_detail {

inline constexpr double continuity_rel_tol = 1e-9;

/// Valida la curva completa en cada llamada (las listas son pequeñas). Todo error nombra `segments`.
inline void validate(const std::vector<Segment>& segments) {
    namespace v = validation;
    v::require(!segments.empty(), "segments", "must contain at least one segment", 0.0);
    for (std::size_t i = 0; i < segments.size(); ++i) {
        const Segment& s = segments[i];
        const std::string at = "segment " + std::to_string(i) + ": ";
        const bool last = i + 1 == segments.size();
        v::require(std::isfinite(s.f_min) && s.f_min >= 0.0, "segments", at + "f_min must be finite and >= 0", s.f_min);
        v::require(!std::isnan(s.f_max) && s.f_max > s.f_min, "segments", at + "f_max must be greater than f_min",
                   s.f_max);
        if (std::isinf(s.f_max))
            v::require(last && s.kind == SegmentKind::ConstantVelocity, "segments",
                       at + "f_max may be infinite only on the last segment and only for ConstantVelocity", s.f_max);
        v::require(std::isfinite(s.v1) && s.v1 > 0.0, "segments", at + "v1 must be finite and > 0", s.v1);
        if (s.kind == SegmentKind::Linear || s.kind == SegmentKind::LogLog)
            v::require(std::isfinite(s.v2) && s.v2 > 0.0, "segments", at + "v2 must be finite and > 0", s.v2);
        if (s.kind == SegmentKind::LogLog)
            v::require(s.f_min > 0.0, "segments", at + "f_min must be > 0 for LogLog", s.f_min);
        if (i > 0) {
            const double prev = segments[i - 1].f_max;
            const double tol = continuity_rel_tol * std::max(std::fabs(prev), std::fabs(s.f_min));
            v::require(s.f_min >= prev - tol, "segments",
                       at + "f_min is below the previous segment's f_max (segments overlap or are not sorted)",
                       s.f_min);
            v::require(s.f_min <= prev + tol, "segments",
                       at + "f_min is above the previous segment's f_max (gap between segments)", s.f_min);
        }
    }
}

inline double value(const Segment& s, double f) {
    switch (s.kind) {
        case SegmentKind::ConstantVelocity:
            return s.v1;
        case SegmentKind::ConstantDisplacement:
            return 2.0 * std::numbers::pi * f * s.v1;
        case SegmentKind::Linear:
            return s.v1 + (s.v2 - s.v1) * (f - s.f_min) / (s.f_max - s.f_min);
        default:  // LogLog: recta en (log f, log v)
            return std::exp(std::log(s.v1) +
                            std::log(s.v2 / s.v1) * std::log(f / s.f_min) / std::log(s.f_max / s.f_min));
    }
}

}  // namespace criteria_curve_detail

/// Límite de la curva a `frequency_hz` (Hz), en las unidades de los tramos. Fuera de todos los tramos: defined = false.
inline CurveLimit limit_at(const std::vector<Segment>& segments, double frequency_hz) {
    validation::require_positive(frequency_hz, "frequency_hz");
    criteria_curve_detail::validate(segments);
    if (frequency_hz < segments.front().f_min || frequency_hz > segments.back().f_max) return {false, validation::unset, -1};
    for (std::size_t i = 0; i < segments.size(); ++i)
        if (frequency_hz < segments[i].f_max || i + 1 == segments.size())
            return {true, criteria_curve_detail::value(segments[i], frequency_hz), static_cast<int>(i)};
    return {false, validation::unset, -1};  // inalcanzable
}

/// Frecuencia (Hz) en que una velocidad constante y un desplazamiento constante dan el mismo límite: v / (2π·d).
inline double intersection_frequency(double velocity, double displacement) {
    validation::require_positive(velocity, "velocity");
    validation::require_positive(displacement, "displacement");
    return velocity / (2.0 * std::numbers::pi * displacement);
}

}  // namespace blastingpy
