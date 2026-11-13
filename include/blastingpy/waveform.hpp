/**
 * @file waveform.hpp
 * @brief Forma de onda del sismógrafo: pico por canal y suma vectorial.
 *
 * Procedencia (docs/investigacion/FUENTES_SPRINT2.md, «Cálculo de Instantel»):
 * - PPV por canal: la máxima rapidez de la partícula en cada geófono (Minimate Blaster, p. 4-6) [V1]. Que se tome
 *   el mayor |x| sin importar el signo es inferido (la fuente habla de «rapidez» y dibuja un pico positivo).
 * - Suma vectorial (PVS): √(T² + V² + L²) punto por punto de las formas de onda, y se informa el mayor valor
 *   (Minimate Blaster, p. 4-9) [V1]. No coincide necesariamente con el pico de una componente.
 * - Seudo suma vectorial (raíz de los tres picos): no aparece en ninguna fuente de Instantel; es una magnitud de
 *   diseño, una cota superior de la suma vectorial real.
 *
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "blastingpy/validation.hpp"

namespace blastingpy {

/// Máximo de |x| (≥ 0) y la primera muestra donde ocurre.
struct Peak {
    double value;
    std::size_t index;
};

namespace waveform_detail {
inline void require_signal(const std::vector<double>& x, const std::string& name) {
    validation::require(!x.empty(), name, "must not be empty", 0.0);
    for (double xi : x) validation::require_finite(xi, name);
}
}  // namespace waveform_detail

/// Pico de una señal: máximo |x| y primera aparición (los empates se quedan con la primera muestra).
inline Peak peak_abs(const std::vector<double>& x) {
    waveform_detail::require_signal(x, "x");
    Peak best{std::fabs(x[0]), 0};
    for (std::size_t i = 1; i < x.size(); ++i) {
        const double a = std::fabs(x[i]);
        if (a > best.value) best = {a, i};
    }
    return best;
}

/// Mayor √(t² + v² + l²) calculado muestra por muestra (la suma vectorial real, PVS) y su muestra.
inline Peak vector_sum_peak(const std::vector<double>& t, const std::vector<double>& v,
                            const std::vector<double>& l) {
    waveform_detail::require_signal(t, "t");
    waveform_detail::require_signal(v, "v");
    waveform_detail::require_signal(l, "l");
    validation::require(v.size() == t.size(), "v", "must have the same length as t",
                        static_cast<double>(v.size()));
    validation::require(l.size() == t.size(), "l", "must have the same length as t",
                        static_cast<double>(l.size()));
    Peak best{-1.0, 0};
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double s = std::sqrt(t[i] * t[i] + v[i] * v[i] + l[i] * l[i]);
        if (s > best.value) best = {s, i};
    }
    return best;
}

/// Seudo suma vectorial: √(peak_t² + peak_v² + peak_l²) con los tres picos no negativos (cota de la real).
inline double pseudo_vector_sum(double peak_t, double peak_v, double peak_l) {
    validation::require_non_negative(peak_t, "peak_t");
    validation::require_non_negative(peak_v, "peak_v");
    validation::require_non_negative(peak_l, "peak_l");
    return std::sqrt(peak_t * peak_t + peak_v * peak_v + peak_l * peak_l);
}

}  // namespace blastingpy
