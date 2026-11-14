/**
 * @file airblast.hpp
 * @brief Sobrepresión (onda aérea): Pa <-> dB y paso de un coeficiente publicado de psi-ft-lb a SI.
 *
 * Procedencia (docs/investigacion/FUENTES_SPRINT3.md, «Conversiones»):
 * - dB = 20·log10(P/P0), con P0 = 20 µPa = 2,9×10⁻⁹ psi (RI 8485, p. 8) [V1].
 * - El factor psi -> Pa (units::pa_per_psi) es una definición que ninguna fuente del sprint imprime (inferido).
 * - K': AB[Pa] = K'·(D[m]/W[kg]^⅓)^−β sale de AB[psi] = K·(D[ft]/W[lb]^⅓)^−β (mismo cambio de variable que
 *   units::convert_k_to_si; el exponente ⅓ es el de la raíz cúbica de las ecuaciones de RI 8485).
 *
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <cmath>

#include "blastingpy/units.hpp"
#include "blastingpy/validation.hpp"

namespace blastingpy::airblast {

inline constexpr double reference_pressure_pa = 20e-6;

/// Nivel en dB: 20·log10(p/p0). p en Pa, finito y > 0.
inline double pa_to_db(double pressure_pa) {
    validation::require_positive(pressure_pa, "pressure_pa");
    return 20.0 * std::log10(pressure_pa / reference_pressure_pa);
}

/// Presión en Pa: p0·10^(dB/20). dB finito.
inline double db_to_pa(double level_db) {
    validation::require_finite(level_db, "level_db");
    return reference_pressure_pa * std::pow(10.0, level_db / 20.0);
}

/// K' = K·pa_per_psi·(m_per_ft / kg_per_lb^⅓)^β. K en psi, β adimensional; ambos finitos y > 0.
inline double convert_k_to_si(double k_psi, double beta) {
    validation::require_positive(k_psi, "k_psi");
    validation::require_positive(beta, "beta");
    return k_psi * units::pa_per_psi * std::pow(units::m_per_ft / std::cbrt(units::kg_per_lb), beta);
}

}  // namespace blastingpy::airblast
