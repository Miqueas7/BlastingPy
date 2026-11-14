/**
 * @file flyrock.hpp
 * @brief Proyección de rocas: balística sin arrastre, Richards y Moore (frente, cráter, eyección de taco), Lundborg,
 *        McKenzie 2018 y Roth (Gurney y ecuaciones calibradas).
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
 *   Con 0° (vertical) la fórmula da 0 (Tabla 4, p. 30); la API de Python avisa, porque la fuente dice que los
 *   verticales necesitan un ajuste.
 *
 * Lundborg, McKenzie y Roth (FUENTES_SPRINT4.md, «Lundborg», «McKenzie», «Roth (Gurney)»):
 * - Lundborg [V2], solo por secundarias (Richards y Moore ecs. [1]-[3], p. 22; McKenzie 2018 diaps. 3-4; Szendrei y
 *   Tose 2022 ecs. [1]-[2], p. 726). L_max = 260·d^(2/3) m, fragmento óptimo 0,1·d^(2/3) m, L = 143·d·(q − 0,2) m, con d
 *   en PULGADAS en la fuente (aquí mm, se convierte). La condición impresa de L3 («q ≤ 0,2») es errata: q > 0,2 es
 *   inferido (con q ≤ 0,2 el alcance sería ≤ 0).
 * - McKenzie, versión 2018 [V1] (diap. 11, «Flyrock Model Summary»): Range_max = 10·SDoB^−2,167·∅^0,667, ∅ en mm. La
 *   definición de SDoB no está en ninguna fuente leída [NV]: la da el usuario. El 11 y el 9,74 de 2009 son [NV].
 * - Roth (1979), USBM OFR 77-81 [V1]: ec. (7) y (8) p. 7, ecs. (9) y (10) p. 10; calibradas: ec. (13) granito p. 19,
 *   (13a) p. 21, (14) dolomita y caliza p. 21, (17) arenisca (cráter) p. 35, todas υ0² = a·(c/m) − b en (m/s)².
 *   La ec. (13a) es la recta «basada en todos los datos» según la fuente, pero deja fuera los de dos referencias y dos
 *   puntos de una tercera de la Tabla 2 (la de (13) omite tres puntos); r = 0,971, y no dice a qué √(2E') se normalizó
 *   (por eso GraniteAllData no tiene velocidad de Gurney de referencia: NaN).
 * - Las calibradas valen para la velocidad de Gurney con la que se normalizaron (roth_reference_gurney_velocity). La
 *   fuente no dice cómo escalarlas a otro explosivo (el factor impreso en § 4.1 parece invertido), así que
 *   roth_calibrated_velocity no recibe explosivo.
 *
 * Ángulos en grados. Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <cmath>
#include <limits>
#include <string>

#include "blastingpy/method_info.hpp"
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
    validation::require_finite(hole_angle_from_vertical_deg, "hole_angle_from_vertical_deg");
    validation::require(hole_angle_from_vertical_deg >= 0.0 && hole_angle_from_vertical_deg < 90.0,
                        "hole_angle_from_vertical_deg", "must be within [0, 90)", hole_angle_from_vertical_deg);
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

namespace detail {
inline constexpr double mm_per_in = 25.4;
/// Diámetro de taladro en pulgadas desde mm (20 a 500); 5 (pulgadas) o 0,127 (metros) por error se rechazan.
inline double diameter_in(double diameter_mm) {
    validation::require_finite(diameter_mm, "diameter_mm");
    validation::require(diameter_mm >= 20.0 && diameter_mm <= 500.0, "diameter_mm",
                        "must be a hole diameter in millimetres within [20, 500] (inches or metres by mistake?)",
                        diameter_mm);
    return diameter_mm / mm_per_in;
}
}  // namespace detail

/// Lundborg [V2]: L_max = 260·d_in^(2/3), m.
inline double lundborg_max_range(double diameter_mm) {
    return 260.0 * std::pow(detail::diameter_in(diameter_mm), 2.0 / 3.0);
}

/// Lundborg [V2]: tamaño óptimo de la roca que llega a L_max, 0,1·d_in^(2/3), m.
inline double lundborg_optimal_fragment(double diameter_mm) {
    return 0.1 * std::pow(detail::diameter_in(diameter_mm), 2.0 / 3.0);
}

/// Lundborg [V2]: L = 143·d_in·(q − 0,2), m, con q en kg/m³. q > 0,2 es inferido (la fuente imprime «≤», errata).
inline double lundborg_range(double diameter_mm, double powder_factor_kg_m3) {
    const double d_in = detail::diameter_in(diameter_mm);
    validation::require_finite(powder_factor_kg_m3, "powder_factor_kg_m3");
    validation::require(powder_factor_kg_m3 > 0.2, "powder_factor_kg_m3", "must be greater than 0.2 kg/m3",
                        powder_factor_kg_m3);
    return 143.0 * d_in * (powder_factor_kg_m3 - 0.2);
}

/// McKenzie 2018 [V1]: 10·SDoB^−2,167·∅^0,667, m; SDoB en m/kg^(1/3) lo da el usuario, ∅ en mm.
inline double mckenzie_max_range(double sdob_m_kg13, double diameter_mm) {
    validation::require_positive(sdob_m_kg13, "sdob_m_kg13");
    detail::diameter_in(diameter_mm);  // solo valida; la fuente usa mm
    return 10.0 * std::pow(sdob_m_kg13, -2.167) * std::pow(diameter_mm, 0.667);
}

enum class Explosive { Anfo, Other };
enum class RothCalibration { Granite, GraniteAllData, DolomiteLimestone, SandstoneCrater };

/// c/m = (W/ℓ)/(ρ·b²·tan(α/2)), Roth ec. (7); α = ángulo de rotura en la cara libre (≈ 90°).
inline double roth_charge_mass_ratio(double linear_charge_kg_m, double rock_density_kg_m3, double burden_m,
                                     double breakout_angle_deg = 90.0) {
    validation::require_positive(linear_charge_kg_m, "linear_charge_kg_m");
    validation::require_positive(rock_density_kg_m3, "rock_density_kg_m3");
    validation::require_positive(burden_m, "burden_m");
    validation::require_finite(breakout_angle_deg, "breakout_angle_deg");
    validation::require(breakout_angle_deg > 0.0 && breakout_angle_deg < 180.0, "breakout_angle_deg",
                        "must be within (0, 180)", breakout_angle_deg);
    return linear_charge_kg_m /
           (rock_density_kg_m3 * burden_m * burden_m * std::tan(detail::rad(breakout_angle_deg) / 2.0));
}

/// √(2E') = 0,44·D (ANFO, ec. 9) o D/3 (los demás, ec. 10).
inline double roth_gurney_velocity(double detonation_velocity_m_s, Explosive explosive) {
    validation::require_positive(detonation_velocity_m_s, "detonation_velocity_m_s");
    return explosive == Explosive::Anfo ? 0.44 * detonation_velocity_m_s : detonation_velocity_m_s / 3.0;
}

/// υ0 = √(2E')·√(c/m), ec. (8).
inline double roth_launch_velocity(double gurney_velocity_m_s, double charge_mass_ratio) {
    validation::require_positive(gurney_velocity_m_s, "gurney_velocity_m_s");
    validation::require_positive(charge_mass_ratio, "charge_mass_ratio");
    return gurney_velocity_m_s * std::sqrt(charge_mass_ratio);
}

namespace detail {
struct RothLine {
    double a, b;  // υ0² = a·(c/m) − b, (m/s)²
};
inline RothLine roth_line(RothCalibration rock) {
    switch (rock) {
        case RothCalibration::Granite: return {3.487e6, 584.0};
        case RothCalibration::GraniteAllData: return {3.66e6, 518.0};
        case RothCalibration::DolomiteLimestone: return {3.0e6, 200.0};
        default: return {2.266e6, 475.0};
    }
}
}  // namespace detail

/// c/m mínimo de cada calibración, b/a: ahí la recta da velocidad nula.
inline double roth_minimum_charge_mass_ratio(RothCalibration rock) {
    const auto line = detail::roth_line(rock);
    return line.b / line.a;
}

/// Ecuaciones calibradas υ0² = a·(c/m) − b (ecs. 13, 13a, 14, 17). Error si c/m ≤ b/a (velocidad nula o imaginaria).
/// Cerca del mínimo la recta subestima: predice velocidad casi nula y los datos tienen velocidad finita (p. 21).
inline double roth_calibrated_velocity(RothCalibration rock, double charge_mass_ratio) {
    const auto line = detail::roth_line(rock);
    validation::require_positive(charge_mass_ratio, "charge_mass_ratio");
    // se compara con b/a y no el signo de a·(c/m) − b: con contracción FMA ese residuo puede salir positivo en b/a
    const double minimum = line.b / line.a;
    validation::require(charge_mass_ratio > minimum, "charge_mass_ratio",
                        "is at or below the minimum c/m of this calibration (" + validation::detail::num(minimum) +
                            "), where the fitted velocity is zero",
                        charge_mass_ratio);
    return std::sqrt(std::fmax(line.a * charge_mass_ratio - line.b, 0.0));
}

/// √(2E') con el que se normalizó cada calibración: 2300, NaN (la ec. 13a no lo dice), 1880, 1750 m/s.
inline double roth_reference_gurney_velocity(RothCalibration rock) {
    switch (rock) {
        case RothCalibration::Granite: return 2300.0;
        case RothCalibration::GraniteAllData: return std::numeric_limits<double>::quiet_NaN();
        case RothCalibration::DolomiteLimestone: return 1880.0;
        default: return 1750.0;
    }
}

enum class Model {
    Ballistics, RichardsMooreFaceBurst, RichardsMooreCratering, RichardsMooreStemmingEjection,
    LundborgMaxRange, LundborgRange, McKenzie2018, RothGurney, RothCalibrated
};

/// Id, marca y cita de cada modelo. Lundborg es [V2]: cita las secundarias, no los originales.
inline MethodInfo info(Model model) {
    const std::string rm = "Richards, A.B. & Moore, A.J., Flyrock Control - By Chance or Design, 30th ISEE Annual "
                           "Conference, New Orleans (reproduced as Appendix 1 of Moore & Richards 2005), ";
    const std::string roth = "Roth, J. (1979). A Model for the Determination of Flyrock Range as a Function of Shot "
                             "Conditions. USBM OFR 77-81, ";
    const std::string lund = "Lundborg's equations as restated by Richards & Moore (ISEE, eqs. [1]-[3], p. 22), "
                             "McKenzie (2018) and Szendrei & Tose (2022, JSAIMM 122(12), eqs. [1]-[2], p. 726); "
                             "originals not read";
    switch (model) {
        case Model::Ballistics:
            return {"FLY-BALLISTIC", Verification::Primary,
                    roth + "eqs. (1)-(3), (5), pp. 5-6; " + rm + "eqs. [4]-[5], p. 23; Moore, A.J. & Richards, A.B. "
                    "(2005), consulting report on the calibration of a flyrock predictive model, eqs. [7], [9], [10], "
                    "p. 11"};
        case Model::RichardsMooreFaceBurst: return {"FLY-RM-FACE", Verification::Primary, rm + "eq. [7a], p. 30"};
        case Model::RichardsMooreCratering: return {"FLY-RM-CRATER", Verification::Primary, rm + "eq. [7b], p. 30"};
        case Model::RichardsMooreStemmingEjection:
            return {"FLY-RM-STEMMING", Verification::Primary, rm + "eq. [7c], p. 30"};
        case Model::LundborgMaxRange: return {"FLY-LUNDBORG-MAX", Verification::Secondary, lund};
        case Model::LundborgRange: return {"FLY-LUNDBORG-Q", Verification::Secondary, lund + "; q > 0.2 inferred"};
        case Model::McKenzie2018:
            return {"FLY-MCKENZIE-2018", Verification::Primary,
                    "McKenzie, C. (2018). Flyrock Model Validation. ISEE Australia, 4th Annual Conference, "
                    "Fremantle, 8-9 Nov 2018; flyrock model summary, slide 11: version 2018, coefficient 10; SDoB "
                    "supplied by the user in m/kg^(1/3) (its definition is not in any source read)"};
        case Model::RothGurney:
            return {"FLY-ROTH-GURNEY", Verification::Primary, roth + "eqs. (7)-(10), pp. 7 and 10"};
        default:
            return {"FLY-ROTH-CAL", Verification::Primary,
                    roth + "eqs. (13), (13a), (14), (17), pp. 19-35; valid for the Gurney velocity each was "
                           "normalized with"};
    }
}

}  // namespace blastingpy::flyrock
