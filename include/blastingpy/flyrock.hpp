/**
 * @file flyrock.hpp
 * @brief Proyección de rocas: balística sin arrastre y ecuaciones de Richards y Moore.
 *
 * Procedencia (docs/investigacion/FUENTES_SPRINT4.md, «Roth (Gurney)» y «Richards y Moore»), todo [V1]:
 * - Alcance a nivel L = v²·sen(2θ)/g: Roth (1979) ec. (1), p. 5; Richards y Moore ec. [4], p. 23.
 * - Alcance máximo L_max = v²/g (θ = 45°): Roth ec. (2), p. 5; Richards y Moore ec. [5], p. 23.
 * - Alcance hasta una altura H sobre el lanzamiento: Moore y Richards (2005) ec. [7], p. 11; Roth ec. (3), p. 6
 *   (misma cinemática; H < 0 = el punto de llegada está por debajo).
 * - Altura máxima h = v²·sen²θ/(2g): Roth ec. (5), p. 6. (La ec. (4) de Roth tiene errata y no se usa.)
 * - V0 = k·(√m/B)^1,3: Richards y Moore ec. [6], p. 24. Frente: ec. [7a]; cráter: ec. [7b]; eyección de taco:
 *   ec. [7c] (= [7b]·sen 2θ), las tres en p. 30. m es la carga por metro (kg/m), o la masa real (kg) si la columna
 *   mide menos de 1 m (p. 28). g = 9,8 en la fuente; aquí por omisión 9,80665.
 * - rm_min_confinement invierte [7a]/[7b] (álgebra, inferido).
 * - Ángulo del taladro de [7c]: se mide desde la vertical aquí; sen 2θ es igual desde la horizontal (inferido).
 *   Con 0° (vertical) la fórmula da 0 (Tabla 4, p. 30).
 *
 * Ángulos en grados. Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <cmath>

#include "blastingpy/validation.hpp"

namespace blastingpy::flyrock {

inline constexpr double standard_gravity = 9.80665;

namespace detail {
inline constexpr double pi = 3.14159265358979323846;
inline double rad(double deg) { return deg * pi / 180.0; }
inline void launch_angle(double deg) {
    validation::require_finite(deg, "launch_angle_deg");
    validation::require(deg > 0.0 && deg < 90.0, "launch_angle_deg", "must be within (0, 90)", deg);
}
inline void scale(double charge_kg_m, double confinement_m, const char* confinement_name) {
    validation::require_positive(charge_kg_m, "charge_kg_m");
    validation::require_positive(confinement_m, confinement_name);
}
}  // namespace detail

/// L = v²·sen(2θ)/g.
inline double range_level(double velocity_m_s, double launch_angle_deg, double g = standard_gravity) {
    validation::require_positive(velocity_m_s, "velocity_m_s");
    detail::launch_angle(launch_angle_deg);
    validation::require_positive(g, "g");
    return velocity_m_s * velocity_m_s * std::sin(2.0 * detail::rad(launch_angle_deg)) / g;
}

/// L_max = v²/g.
inline double max_range_level(double velocity_m_s, double g = standard_gravity) {
    validation::require_positive(velocity_m_s, "velocity_m_s");
    validation::require_positive(g, "g");
    return velocity_m_s * velocity_m_s / g;
}

/// Alcance horizontal hasta un punto height_m sobre el lanzamiento (negativo = debajo).
inline double range_to_height(double velocity_m_s, double launch_angle_deg, double height_m,
                              double g = standard_gravity) {
    validation::require_positive(velocity_m_s, "velocity_m_s");
    detail::launch_angle(launch_angle_deg);
    validation::require_finite(height_m, "height_m");
    validation::require_positive(g, "g");
    const double th = detail::rad(launch_angle_deg);
    const double vy = velocity_m_s * std::sin(th);
    const double disc = vy * vy - 2.0 * g * height_m;
    validation::require(disc >= 0.0, "height_m", "is unreachable at this velocity and angle", height_m);
    return velocity_m_s * std::cos(th) * (vy + std::sqrt(disc)) / g;
}

/// h = v²·sen²θ/(2g).
inline double max_height(double velocity_m_s, double launch_angle_deg, double g = standard_gravity) {
    validation::require_positive(velocity_m_s, "velocity_m_s");
    detail::launch_angle(launch_angle_deg);
    validation::require_positive(g, "g");
    const double s = std::sin(detail::rad(launch_angle_deg));
    return velocity_m_s * velocity_m_s * s * s / (2.0 * g);
}

/// v = √(L·g/sen 2θ).
inline double velocity_for_range(double range_m, double launch_angle_deg, double g = standard_gravity) {
    validation::require_positive(range_m, "range_m");
    detail::launch_angle(launch_angle_deg);
    validation::require_positive(g, "g");
    return std::sqrt(range_m * g / std::sin(2.0 * detail::rad(launch_angle_deg)));
}

/// V0 = k·(√m/B)^1,3 (Richards y Moore ec. [6]).
inline double rm_launch_velocity(double k, double charge_kg_m, double confinement_m) {
    validation::require_positive(k, "k");
    detail::scale(charge_kg_m, confinement_m, "confinement_m");
    return k * std::pow(std::sqrt(charge_kg_m) / confinement_m, 1.3);
}

/// Frente: L_max = (k²/g)·(√m/B)^2,6 (ec. [7a]).
inline double rm_face_burst(double k, double charge_kg_m, double burden_m, double g = standard_gravity) {
    validation::require_positive(k, "k");
    detail::scale(charge_kg_m, burden_m, "burden_m");
    validation::require_positive(g, "g");
    return k * k / g * std::pow(std::sqrt(charge_kg_m) / burden_m, 2.6);
}

/// Cráter: L_max = (k²/g)·(√m/SH)^2,6 (ec. [7b]).
inline double rm_cratering(double k, double charge_kg_m, double stemming_m, double g = standard_gravity) {
    validation::require_positive(k, "k");
    detail::scale(charge_kg_m, stemming_m, "stemming_m");
    validation::require_positive(g, "g");
    return k * k / g * std::pow(std::sqrt(charge_kg_m) / stemming_m, 2.6);
}

/// Eyección de taco: ec. [7c]; ángulo del taladro desde la vertical, en [0, 90). Con 0° devuelve 0.
inline double rm_stemming_ejection(double k, double charge_kg_m, double stemming_m,
                                   double hole_angle_from_vertical_deg, double g = standard_gravity) {
    validation::require_in_range(hole_angle_from_vertical_deg, 0.0, 90.0, "hole_angle_from_vertical_deg");
    validation::require(hole_angle_from_vertical_deg < 90.0, "hole_angle_from_vertical_deg",
                        "must be within [0, 90)", hole_angle_from_vertical_deg);
    return rm_cratering(k, charge_kg_m, stemming_m, g) * std::sin(2.0 * detail::rad(hole_angle_from_vertical_deg));
}

/// Burden o taco mínimo para que L_max no pase de range_m: √m / (range·g/k²)^(1/2,6).
inline double rm_min_confinement(double k, double charge_kg_m, double range_m, double g = standard_gravity) {
    validation::require_positive(k, "k");
    validation::require_positive(charge_kg_m, "charge_kg_m");
    validation::require_positive(range_m, "range_m");
    validation::require_positive(g, "g");
    return std::sqrt(charge_kg_m) / std::pow(range_m * g / (k * k), 1.0 / 2.6);
}

}  // namespace blastingpy::flyrock
