/**
 * @file units.hpp
 * @brief Factores de conversión exactos y distancia escalada.
 *
 * El núcleo trabaja en SI (m, kg, mm/s, mm). Los factores son exactos por definición
 * (pie internacional, libra avoirdupois, pulgada), no valores medidos.
 */

#pragma once

#include <cmath>

#include "blastingpy/validation.hpp"

namespace blastingpy::units {

inline constexpr double m_per_ft = 0.3048;
inline constexpr double kg_per_lb = 0.45359237;
inline constexpr double mm_per_in = 25.4;
/// psi -> Pa: definición (libra-fuerza / pulgada²); ninguna fuente del sprint 3 la imprime (inferido).
inline constexpr double pa_per_psi = 6894.757;

/// Distancia escalada SD = D / W^e, con D en m, W en kg y e el exponente de la carga
/// (0.5 raíz cuadrada, 1/3 raíz cúbica). Unidades de salida: m/kg^e.
inline double scaled_distance(double distance_m, double charge_kg, double charge_exponent) {
    validation::require_positive(distance_m, "distance_m");
    validation::require_positive(charge_kg, "charge_kg");
    validation::require_positive(charge_exponent, "charge_exponent");
    return distance_m / std::pow(charge_kg, charge_exponent);
}

/// Pasa K de v[in/s] = K (D[ft] / W[lb]^e)^-beta a v[mm/s] = K' (D[m] / W[kg]^e)^-beta.
/// Con D_ft = D_m / m_per_ft y W_lb = W_kg / kg_per_lb:  K' = K * mm_per_in * (m_per_ft / kg_per_lb^e)^beta.
inline double convert_k_to_si(double k_imperial, double beta, double charge_exponent) {
    validation::require_positive(k_imperial, "k_imperial");
    validation::require_positive(beta, "beta");
    validation::require_positive(charge_exponent, "charge_exponent");
    return k_imperial * mm_per_in * std::pow(m_per_ft / std::pow(kg_per_lb, charge_exponent), beta);
}

}  // namespace blastingpy::units
