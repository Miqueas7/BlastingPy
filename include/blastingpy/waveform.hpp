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
#include <limits>
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

// ---------------------------------------------------------------------------------------------------------------
// Frecuencia por cruce por cero («ZC Freq»)
//
// Procedencia (FUENTES_SPRINT2.md, «Cálculo de Instantel», «Frecuencia por cruce por cero»):
// - La ZC Freq es la frecuencia de la forma de onda en el mayor pico; el período se mide entre cruces por cero y
//   la frecuencia es 1/período (Minimate Blaster, p. 4-6) [V1]; usa el medio período T1, del cruce anterior al pico
//   al cruce siguiente (p. 4-7) [V1]. De ahí f = 1/(2·T½): inferido (combinación de esas dos frases).
// - Topes «>100 / >200 / >400 Hz» a 1024 / 2048 / 4096 sps (p. 4-8) [V1]; la regla 100·fs/1024 para otros
//   muestreos es inferida (reproduce los tres valores leídos).
// - Tabla de error por muestreo (p. 4-8) [V1]: las pruebas usan |f − f_real| ≤ cota.
// - Contar muestras enteras (WholeSamples) es inferido de la explicación del manual; interpolar linealmente el
//   cruce (Interpolated) es una variante de diseño sin respaldo en Instantel, y por eso no tiene tope.
// - «N/A» (onda no capturada entera) ↔ Incomplete y «<1 Hz» (no calculable) ↔ NotComputable: correspondencia
//   inferida. La «ventana aceptable» del manual no está definida y no se codifica [NV].
// ---------------------------------------------------------------------------------------------------------------

/// Cómo se mide el medio período: contando muestras enteras del mismo signo que el pico, o interpolando el cruce.
enum class ZcMethod { WholeSamples, Interpolated };

/// Ok; AboveLimit (WholeSamples con f sobre el tope, f se informa igual); Incomplete (falta un cruce: el registro
/// se corta); NotComputable (el pico vale cero).
enum class ZcStatus { Ok, AboveLimit, Incomplete, NotComputable };

/// Frecuencia (Hz) y medio período (s); ambos NaN en Incomplete y NotComputable.
struct ZcFrequency {
    double frequency_hz;
    double half_period_s;
    ZcStatus status;
};

/// Tope de la ZC Freq que muestra Instantel: 100·fs/1024 (100, 200 y 400 Hz a 1024, 2048 y 4096 sps [V1]; otros: inferido).
inline double instantel_zc_limit_hz(double sample_rate_hz) {
    validation::require_positive(sample_rate_hz, "sample_rate_hz");
    return 100.0 * sample_rate_hz / 1024.0;
}

/// Frecuencia del semiciclo que contiene la muestra `peak_index`: f = 1/(2·T½), con T½ entre los cruces por cero
/// que rodean al pico (una muestra en cero cuenta como cruce).
inline ZcFrequency zero_crossing_frequency(const std::vector<double>& x, double sample_rate_hz,
                                           std::size_t peak_index, ZcMethod method) {
    waveform_detail::require_signal(x, "x");
    validation::require_positive(sample_rate_hz, "sample_rate_hz");
    validation::require(peak_index < x.size(), "peak_index", "must be smaller than the signal length",
                        static_cast<double>(peak_index));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::size_t p = peak_index;
    if (x[p] == 0.0) return {nan, nan, ZcStatus::NotComputable};
    const double s = x[p] > 0.0 ? 1.0 : -1.0;

    // i0: último índice antes del pico con el signo opuesto (o cero); i1: el primero después.
    std::size_t first = p;  // primera muestra del semiciclo (mismo signo que el pico)
    while (first > 0 && s * x[first - 1] > 0.0) --first;
    if (first == 0) return {nan, nan, ZcStatus::Incomplete};
    const std::size_t i0 = first - 1;
    std::size_t i1 = p + 1;
    while (i1 < x.size() && s * x[i1] > 0.0) ++i1;
    if (i1 == x.size()) return {nan, nan, ZcStatus::Incomplete};

    double half_samples;
    if (method == ZcMethod::WholeSamples) {
        half_samples = static_cast<double>(i1 - i0 - 1);
    } else {
        const double t0 = static_cast<double>(i0) + x[i0] / (x[i0] - x[i0 + 1]);
        const double t1 = static_cast<double>(i1 - 1) + x[i1 - 1] / (x[i1 - 1] - x[i1]);
        half_samples = t1 - t0;
    }
    const double half_period = half_samples / sample_rate_hz;
    const double f = 1.0 / (2.0 * half_period);
    const bool above = method == ZcMethod::WholeSamples && f > instantel_zc_limit_hz(sample_rate_hz);
    return {f, half_period, above ? ZcStatus::AboveLimit : ZcStatus::Ok};
}

}  // namespace blastingpy
