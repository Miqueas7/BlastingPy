/**
 * @file pattern.hpp
 * @brief Malla de banco: burden, espaciamiento, taco y sobreperforación por Konya (FHWA) y Ash (IC 8550).
 *
 * Konya y Walter (1991), FHWA-HI-92-001 [Primary]. Páginas del PDF (la impresa entre paréntesis):
 * - Burden B = (2·SGe/SGr + 1,5)·De, B en pies y De en pulgadas: ec. 6.2, p. 110 (91); ejemplo 6.2 (6,25 ft), p. 111.
 *   De es el diámetro del explosivo; aquí el del taladro (carga acoplada). SGe/SGr = ρe/ρr (ambas en kg/m³).
 * - Correcciones B·Kd·Ks·Kr (pp. 112-114): Kr por número de filas, Tabla 6.2 p. 113 (1,0 una o dos filas; 0,9 tercera
 *   y siguientes o voladura contra escombro); Kd por depósito, Tabla 6.3 p. 113; Ks por estructura, Tabla 6.4 p. 114.
 *   Por omisión valen 1: la fórmula publicada sin corregir.
 * - Taco T = 0,7·B: ec. 6.4, p. 115 (96). Sobreperforación J = 0,3·B: ec. 6.6, p. 118 (99).
 * - Espaciamiento por la relación de rigidez L/B (pp. 157-162): L/B < 4 es banco bajo, ≥ 4 alto (p. 158).
 *   Instantáneo: S = (L + 2B)/3 (ec. 7.1, ejemplo 7.1 p. 159) y S = 2B (ec. 7.2, ejemplo 7.2 p. 160).
 *   Con retardos: S = (L + 7B)/8 (ec. 7.3, ejemplo 7.3 p. 161) y S = 1,4B (ec. 7.4, ejemplo 7.4 p. 162).
 *   La fuente define el banco bajo con 1 < L/B < 4 (p. 158); con L/B ≤ 1 se calcula igual y la nota lo avisa.
 *
 * Ash, por Pugliese (1972), USBM IC 8550 [Secondary: no se leyó a Ash directamente]:
 * - B = K_B·De/12 (pies, De en pulgadas): ec. (1), p. 13; K_B = 30 promedio, 25 explosivo de baja densidad (ANFO),
 *   35 denso (hidrogeles, gelatinas), para roca de ~2,7 g/cm³: p. 14.
 * - Instantáneo: S = K_S·B, K_S = 1,8-2 (p. 14); se usa el promedio 2 de la Tabla 1 (p. 16).
 * - Con retardos: s = K_S·b con b = 1,4·B (malla cuadrada), K_S = 1-1,2 (p. 15); se usa el promedio 1 de la Tabla 1
 *   (p. 16). spacing_m es la separación s en la fila; la distancia perforada entre filas es b = 1,4·B, no B.
 * - J = K_J·B con K_J = 0,3 (preferido, p. 15; promedio de la Tabla 1, p. 16).
 * - T = K_T·B con K_T = 0,7 (p. 16; promedio de la Tabla 1, p. 16). No está en pp. 14-15.
 *
 * Konya y Ash se programan en pies y pulgadas, como se publicaron, y se convierten a SI en el borde.
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <string>

#include "blastingpy/method_info.hpp"
#include "blastingpy/units.hpp"
#include "blastingpy/validation.hpp"

namespace blastingpy::pattern {

enum class Initiation { Delayed, Instantaneous };
enum class ExplosiveClass { LowDensity, Average, Dense };

struct BlastDesign {
    double burden_m, spacing_m, stemming_m, subdrill_m;
    MethodInfo info;
    std::string note;
};

inline BlastDesign konya(double diameter_mm, double bench_height_m, double rock_density_kg_m3,
                         double explosive_density_kg_m3, Initiation initiation, double kd = 1.0, double ks = 1.0,
                         double kr = 1.0) {
    validation::require_positive(diameter_mm, "diameter_mm");
    validation::require_positive(bench_height_m, "bench_height_m");
    validation::require_positive(rock_density_kg_m3, "rock_density_kg_m3");
    validation::require_positive(explosive_density_kg_m3, "explosive_density_kg_m3");
    validation::require_positive(kd, "kd");
    validation::require_positive(ks, "ks");
    validation::require_positive(kr, "kr");

    const double de_in = diameter_mm / units::mm_per_in;
    const double l_ft = bench_height_m / units::m_per_ft;
    const double b_ft = (2.0 * explosive_density_kg_m3 / rock_density_kg_m3 + 1.5) * de_in * kd * ks * kr;
    const double stiffness = l_ft / b_ft;
    double s_ft;
    if (initiation == Initiation::Instantaneous)
        s_ft = stiffness < 4.0 ? (l_ft + 2.0 * b_ft) / 3.0 : 2.0 * b_ft;
    else
        s_ft = stiffness < 4.0 ? (l_ft + 7.0 * b_ft) / 8.0 : 1.4 * b_ft;

    std::string note;
    if (kd == 1.0 && ks == 1.0 && kr == 1.0) note = "uncorrected burden (Kd = Ks = Kr = 1)";
    if (stiffness <= 1.0) {
        if (!note.empty()) note += "; ";
        note += "stiffness ratio L/B <= 1 is outside the FHWA spacing range (1 < L/B)";
    }
    return {b_ft * units::m_per_ft, s_ft * units::m_per_ft, 0.7 * b_ft * units::m_per_ft,
            0.3 * b_ft * units::m_per_ft,
            {"FHWA-HI-92-001", Verification::Primary,
             "Konya & Walter (1991), FHWA-HI-92-001: burden eq. 6.2 p. 110, corrections pp. 112-114, stemming eq. 6.4 "
             "p. 115, subdrill eq. 6.6 p. 118, spacing eqs. 7.1-7.4 pp. 158-162"},
            note};
}

inline BlastDesign ash(double diameter_mm, ExplosiveClass cls, Initiation initiation) {
    validation::require_positive(diameter_mm, "diameter_mm");
    const double kb = cls == ExplosiveClass::LowDensity ? 25.0 : cls == ExplosiveClass::Dense ? 35.0 : 30.0;
    const double b_ft = kb * (diameter_mm / units::mm_per_in) / 12.0;
    const bool delayed = initiation == Initiation::Delayed;
    const double s_ft = delayed ? 1.0 * (1.4 * b_ft) : 2.0 * b_ft;
    return {b_ft * units::m_per_ft, s_ft * units::m_per_ft, 0.7 * b_ft * units::m_per_ft,
            0.3 * b_ft * units::m_per_ft,
            {"USBM-IC8550", Verification::Secondary,
             "Ash's ratios in Pugliese (1972), USBM IC 8550: K_B p. 14, K_S pp. 14-15, K_J p. 15, K_T p. 16 "
             "(averages of Table 1, p. 16)"},
            delayed ? "delayed: spacing is the in-row separation s = K_S*b with b = 1.4*B (square pattern, IC 8550 "
                      "p. 15); rows are drilled b apart"
                    : ""};
}

}  // namespace blastingpy::pattern
