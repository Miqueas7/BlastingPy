/**
 * @file nearfield.hpp
 * @brief Campo cercano: Holmberg-Persson (forma integral original y modificada de NIOSH), extensión del daño y PPV crítico.
 *
 * Procedencia (docs/investigacion/FUENTES_SPRINT5.md, «Holmberg-Persson» y «PPV crítico»):
 * - hp_point: PPV = K·Q^α/R^β (NIOSH ec. 1) [V1]. Unidades de los K publicados: PPV en mm/s, Q en kg, R en m; las
 *   funciones hp_* devuelven la unidad que lleve K (con esas constantes, mm/s).
 * - hp_original: K·[ℓ·∫dz'/(r² + (z' − z)²)^(β/(2α))]^α (NIOSH ec. 5; Onederra y Esen ec. 1) [V1]. El exponente del
 *   integrando es β/(2α) sobre la distancia al CUADRADO y α queda FUERA de la integral. Límite lejano: K·(ℓ·L)^α/r^β.
 * - hp_term: forma cerrada con β = 2α (Onederra y Esen ec. 2) [V2]; hp_original usa K·hp_term^α en ese caso.
 * - hp_mean_distance / hp_niosh: R̄ (NIOSH ecs. 8-10, PDF pp. 8-9) [V1]; la primitiva va con asinh (inferido, difiere de
 *   la de NIOSH en una constante y no pierde precisión cerca del eje).
 * - hp_damage_extent: distancia radial donde el PPV baja al límite, medida a la altura z dada (NIOSH Fig. 20 y
 *   Onederra y Esen Fig. 12 miden a media carga) [V1/V2].
 * - ppv_critical: PPVcrit = T·Vp/E (Onederra y Esen ec. 11; Persson, Holmberg y Lee 1994, p. 247) [V2]. Con E estático
 *   (inferido del recálculo de su Tabla 1); entra en Pa y m/s, sale en mm/s.
 *
 * - Condición derivada aquí, no leída en una fuente: con r = 0 sobre el tramo cargado el integrando va como
 *   1/|z' − z|^(β/α); la integral diverge solo si β ≥ α. Con β < α es finita: ∫|u|^(−β/α)du en forma cerrada.
 *
 * Elemento de Heelan (FUENTES_SPRINT5.md, «Radiación de Heelan»):
 * - pressure_function: s(t) = P_VN·(e·γ/n)^n·H(t)·t^n·e^(−γt) (Blair 2007 ec. 8, p. E35) [V1]; máximo P_VN en t = n/γ.
 * - g_function: g(τ) = H(τ)·e^(−τ)·[τ^n − 2n·τ^(n−1) + n(n−1)·τ^(n−2)] = d²/dτ²(τ^n·e^(−τ)) (Blair 2007 ec. 15) [V1].
 * - g_peak: G_n, máximo POSITIVO de g (p. E36) [V1]; 1,8513 / 21,9602 / 680,171 para n = 4, 6, 8 (recalculados; el
 *   680,01 impreso no se reproduce). Para n ≥ 7 el valle negativo es mayor en módulo (n = 8: −767,83): G_n·a/r no es
 *   el PPV, que se toma de la onda.
 * - mean_frequency_factor: c = ω_A/γ (Blair 2007 ecs. 10–12; la fuente da 0,500 / 0,375 / 0,313 para n = 4, 6, 8) [V1];
 *   la forma cerrada c = [1/(n − 1)] / [(√π/2)·Γ(n/2)/Γ((n+1)/2)] es inferida de la ec. 11 y reproduce esos valores.
 * - scaled_frequency: Ω_A = c·γ·a/Vp (Blair 2007 ec. 12) [V1].
 * - heelan_amplitude: F(γ) = γ²·a·L·P_VN·(e/n)^n/(2·μ·Vp) en m/s (Blair 2007 ec. 16) [V1], devuelto en mm/s; μ = ρ·Vs²;
 *   L es la MEDIA longitud del elemento (carga de 2L, Blair 2007 Fig. 1).
 * - heelan_element: P (ec. 13) a lo largo de R = (sen φ, −cos φ) y SV (ec. 14) a lo largo de (cos φ, sen φ) en el marco
 *   (radial, axial) (Heelan ecs. 13a–13b, p. 691) [V1]; φ = arccos(−z/R), ángulo con el eje −z (Heelan p. 690 y Fig. 1,
 *   p. 687) [V1]. a es el radio del taladro (convención de Blair 2007); Heelan (p. 695) lo define como el radio de la
 *   cavidad equivalente.
 * - Validez (Blair 2007, pp. E37–E40) [V1], no se impone: Ω_A ≲ 0,1; P y SV se desacoplan con Ω_A·r/a ≳ 5; L/a ≲ 2.
 *   La suma de P y SV donde se solapan en el tiempo es cuestionable (p. E37).
 *
 * Columna de Heelan (FUENTES_SPRINT5.md, «Heelan escalado y superposición no lineal»; Blair 2008):
 * - La columna se parte en M elementos iguales (M impar, típicamente 41; resultados independientes de M si M ≳ 21,
 *   p. 237) [V1]. Cada elemento es el de Heelan estándar con su presión multiplicada por E_m·w_e^(−A) (p. 237) [V1].
 *   NO es el «Heelan escalado» de Blair y Minchinton (2006): sus ecuaciones y γ_n son [NV] y no entran.
 * - column_weights: E_m = (m·w_e)^A − ((m − 1)·w_e)^A (ec. 1, cebado en el fondo), suma (M·w_e)^A (ec. 2); cebado en
 *   el elemento M1 (ec. 22, p. 245, Fig. 15): el cebo pesa 1, los pares simétricos j = 2…M1 pesan
 *   [(2j − 1)^A − (2j − 3)^A]/2 cada uno, y los sueltos j = M1 + 1 … M − M1 + 1 (hacia el tope, posición M1 + j − 1)
 *   pesan (j + M1 − 1)^A − (j + M1 − 2)^A [V1]. Reflejar la columna cuando el cebo está más cerca del tope es inferido.
 *   A en (0, 1]: A > 1 no tiene sentido físico (p. 240) [V1].
 * - heelan_column: el elemento m inicia en |z_m − z_cebo|/VOD, con z_cebo el centro del elemento del cebo (inferido de
 *   la descripción, la fuente no da la fórmula); las componentes (radial, axial) se suman directamente.
 * - vector_peak: máx √(radial² + axial²) de la onda; el pico se toma siempre de la onda, nunca de G_n.
 *
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "blastingpy/validation.hpp"

namespace blastingpy::nearfield {

enum class HpModel { Original, Niosh };

namespace detail {

/// Un panel Gauss-Kronrod de 15 puntos sobre [a, b].
template <class F>
double gk15(const F& f, double a, double b) {
    static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                  0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                  0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                                  0.207784955007898467600689403773245, 0.0};
    static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                  0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    const double c = 0.5 * (a + b), h = 0.5 * (b - a);
    double k = wgk[7] * f(c);
    for (int j = 0; j < 7; ++j) k += wgk[j] * (f(c - h * xgk[j]) + f(c + h * xgk[j]));
    return k * h;
}

/// ∫_lo^hi (r·cosh t)^q dt para 0 ≤ lo ≤ hi, con q = 1 − β/α. Paneles de ancho ≤ min(1, 1/|q|) (el integrando es
/// analítico y monótono, de escala ~1/|q|); tope duro de 4000 paneles. El exponente se arma con log1p(e^(−2t)) para
/// no desbordar con t grande.
inline double cosh_power_integral(double r, double q, double lo, double hi) {
    if (hi <= lo) return 0.0;
    const double ln_r = std::log(r);
    auto g = [&](double t) { return std::exp(q * (ln_r + t - 0.693147180559945309 + std::log1p(std::exp(-2.0 * t)))); };
    const double width = std::min(1.0, 1.0 / std::max(std::fabs(q), 1e-3));
    const int n = static_cast<int>(std::min(4000.0, std::ceil((hi - lo) / width)));
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += gk15(g, lo + (hi - lo) * i / n, lo + (hi - lo) * (i + 1) / n);
    return sum;
}

/// ∫ dz'/(r² + (z' − z)²)^(β/(2α)) sobre [z_start, z_end] con r > 0, por u = r·sinh t.
inline double original_integral(double alpha, double beta, double z_start, double z_end, double r, double z) {
    const double q = 1.0 - beta / alpha;
    const double t1 = std::asinh((z_start - z) / r), t2 = std::asinh((z_end - z) / r);
    if (t1 < 0.0 && t2 > 0.0) return cosh_power_integral(r, q, 0.0, -t1) + cosh_power_integral(r, q, 0.0, t2);
    if (t1 >= 0.0) return cosh_power_integral(r, q, t1, t2);
    return cosh_power_integral(r, q, -t2, -t1);
}

/// r = 0: ∫|u|^(−β/α) du entre u1 y u2 del mismo lado (0 < a ≤ b en valor absoluto), forma cerrada.
inline double axis_integral(double alpha, double beta, double near_abs, double far_abs) {
    const double e = 1.0 - beta / alpha;
    if (std::fabs(e) < 1e-12) return std::log(far_abs / near_abs);
    return (std::pow(far_abs, e) - std::pow(near_abs, e)) / e;
}

inline void check_constants(double k, double alpha, double beta, double linear_charge) {
    validation::require_positive(k, "k");
    validation::require_positive(alpha, "alpha");
    validation::require_positive(beta, "beta");
    validation::require_positive(linear_charge, "linear_charge_kg_m");
}

inline void check_geometry(double z_start, double z_end, double r, double z) {
    validation::require_finite(z_start, "z_start");
    validation::require_finite(z_end, "z_end");
    validation::require(z_end > z_start, "z_end", "must be greater than z_start", z_end);
    validation::require_non_negative(r, "r");
    validation::require_finite(z, "z");
}

/// R̄ con la primitiva en asinh; las entradas ya vienen validadas.
inline double mean_distance(double z_start, double z_end, double r, double z) {
    auto prim = [r](double u) {
        if (r == 0.0) return 0.5 * u * std::fabs(u);
        return 0.5 * (u * std::sqrt(u * u + r * r) + r * r * std::asinh(u / r));
    };
    return (prim(z_end - z) - prim(z_start - z)) / (z_end - z_start);
}

}  // namespace detail

/// PPV = K·Q^α/R^β con Q en kg y R en m (NIOSH ec. 1). Todos finitos y > 0.
inline double hp_point(double k, double alpha, double beta, double charge_kg, double distance_m) {
    detail::check_constants(k, alpha, beta, charge_kg);
    validation::require_positive(distance_m, "distance_m");
    return k * std::pow(charge_kg, alpha) / std::pow(distance_m, beta);
}

/// a = (ℓ/r)·[atan((z_end − z)/r) + atan((z − z_start)/r)] (Onederra y Esen ec. 2, β = 2α). Exige r > 0.
inline double hp_term(double linear_charge_kg_m, double z_start, double z_end, double r, double z) {
    validation::require_positive(linear_charge_kg_m, "linear_charge_kg_m");
    detail::check_geometry(z_start, z_end, r, z);
    validation::require_positive(r, "r");
    // atan(x) + atan(y) = atan2(x + y, 1 − x·y): no cancellation when the point is far from the span at small r.
    const double x = (z_end - z) / r, y = (z - z_start) / r;
    return linear_charge_kg_m / r * std::atan2(x + y, 1.0 - x * y);
}

/// K·[ℓ·∫ dz'/(r² + (z' − z)²)^(β/(2α))]^α (NIOSH ec. 5). Sobre el eje dentro del tramo cargado diverge: error de dominio.
inline double hp_original(double k, double alpha, double beta, double linear_charge_kg_m, double z_start,
                          double z_end, double r, double z) {
    detail::check_constants(k, alpha, beta, linear_charge_kg_m);
    detail::check_geometry(z_start, z_end, r, z);
    const bool on_charge = r == 0.0 && z >= z_start && z <= z_end;
    validation::require(!(on_charge && beta >= alpha), "r",
                        "must be > 0 when z lies within the charged span and beta >= alpha (the integral diverges)", r);
    if (on_charge) {  // beta < alpha: integrable singularity, closed form of the integral of |u|^(-beta/alpha)
        const double integral = detail::axis_integral(alpha, beta, 0.0, z_end - z) +
                                detail::axis_integral(alpha, beta, 0.0, z - z_start);
        return k * std::pow(linear_charge_kg_m * integral, alpha);
    }
    if (r == 0.0) {  // outside the span (on it, handled above): both ends on the same side of z
        const double u1 = std::fabs(z_start - z), u2 = std::fabs(z_end - z);
        const double integral = detail::axis_integral(alpha, beta, std::min(u1, u2), std::max(u1, u2));
        return k * std::pow(linear_charge_kg_m * integral, alpha);
    }
    if (std::fabs(beta - 2.0 * alpha) <= 1e-12)
        return k * std::pow(hp_term(linear_charge_kg_m, z_start, z_end, r, z), alpha);
    const double integral = detail::original_integral(alpha, beta, z_start, z_end, r, z);
    return k * std::pow(linear_charge_kg_m * integral, alpha);
}

/// R̄ = (1/L)·∫ sqrt((z' − z)² + r²) dz' (NIOSH ec. 9).
inline double hp_mean_distance(double z_start, double z_end, double r, double z) {
    detail::check_geometry(z_start, z_end, r, z);
    return detail::mean_distance(z_start, z_end, r, z);
}

/// K·(ℓ·L)^α/R̄^β (NIOSH ec. 8). Finito también sobre el eje.
inline double hp_niosh(double k, double alpha, double beta, double linear_charge_kg_m, double z_start, double z_end,
                       double r, double z) {
    detail::check_constants(k, alpha, beta, linear_charge_kg_m);
    detail::check_geometry(z_start, z_end, r, z);
    const double rbar = detail::mean_distance(z_start, z_end, r, z);
    return k * std::pow(linear_charge_kg_m * (z_end - z_start), alpha) / std::pow(rbar, beta);
}

/// Distancia radial r donde el PPV baja a ppv_limit, a la altura z (bisección en (1e-9, 1e4] m; tolerancia 1e-9 m).
/// La bisección vale porque el PPV no crece con r en los dos modelos: el integrando (r² + u²)^(−p) baja con r en cada u,
/// y R̄ sube con r.
/// Con Original sobre el tramo cargado y β > α el PPV crece sin tope hacia r → 0, así que hay solución; con β = α la
/// divergencia es logarítmica y un límite muy grande aún puede dar NaN. Con β < α es finito en el eje (puede dar NaN),
/// y fuera del tramo el PPV en r = 0 también es finito.
/// NaN si el PPV no llega al límite (en r → 0 queda por debajo, o a 1e4 m sigue por encima).
inline double hp_damage_extent(HpModel model, double k, double alpha, double beta, double linear_charge_kg_m,
                               double z_start, double z_end, double z, double ppv_limit_mm_s) {
    detail::check_constants(k, alpha, beta, linear_charge_kg_m);
    detail::check_geometry(z_start, z_end, 0.0, z);
    validation::require_positive(ppv_limit_mm_s, "ppv_limit_mm_s");
    auto ppv = [&](double r) {
        return model == HpModel::Niosh ? hp_niosh(k, alpha, beta, linear_charge_kg_m, z_start, z_end, r, z)
                                       : hp_original(k, alpha, beta, linear_charge_kg_m, z_start, z_end, r, z);
    };
    // Niosh, and Original with beta < alpha, are finite on the axis: start the search at r = 0.
    double lo = (model == HpModel::Niosh || beta < alpha) ? 0.0 : 1e-9, hi = 1e4;
    if (ppv(lo) < ppv_limit_mm_s || ppv(hi) > ppv_limit_mm_s) return std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < 200 && hi - lo > 1e-9; ++i) {
        const double mid = 0.5 * (lo + hi);
        (ppv(mid) > ppv_limit_mm_s ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

/// PPVcrit = T·Vp/E en m/s, devuelto en mm/s. T y E en Pa, Vp en m/s; finitos y > 0.
inline double ppv_critical(double tensile_strength_pa, double p_velocity_m_s, double youngs_modulus_pa) {
    validation::require_positive(tensile_strength_pa, "tensile_strength_pa");
    validation::require_positive(p_velocity_m_s, "p_velocity_m_s");
    validation::require_positive(youngs_modulus_pa, "youngs_modulus_pa");
    return 1000.0 * tensile_strength_pa * p_velocity_m_s / youngs_modulus_pa;
}

// ---------------------------------------------------------------------------------------------------------------------
// Elemento de Heelan (Heelan 1953 con la fuente explícita de Blair 2007)

/// Medio elástico: velocidades P y S (m/s) y densidad (kg/m³); μ = ρ·Vs².
struct Medium {
    double p_velocity_m_s;
    double s_velocity_m_s;
    double density_kg_m3;
};

/// s(t) de Blair (2007) ec. 8: presión de von Neumann (Pa), exponente entero n y decaimiento γ (1/s).
struct PressureSource {
    double vn_pressure_pa;
    int n;
    double gamma_per_s;
};

/// Velocidad de partícula (mm/s) en el marco (r, z) del taladro, una muestra por instante.
struct Waveform {
    std::vector<double> radial_mm_s;
    std::vector<double> axial_mm_s;
};

namespace detail {

inline void check_n(int n) { validation::require_in_range(n, 2, 20, "n"); }

inline void check_source(const PressureSource& s) {
    validation::require_positive(s.vn_pressure_pa, "vn_pressure_pa");
    check_n(s.n);
    validation::require_positive(s.gamma_per_s, "gamma_per_s");
}

inline void check_medium(const Medium& m) {
    validation::require_positive(m.p_velocity_m_s, "p_velocity_m_s");
    validation::require_positive(m.s_velocity_m_s, "s_velocity_m_s");
    validation::require_positive(m.density_kg_m3, "density_kg_m3");
    validation::require(m.p_velocity_m_s > m.s_velocity_m_s, "p_velocity_m_s", "must be greater than s_velocity_m_s",
                        m.p_velocity_m_s);
}

/// g(τ) sin validar (ec. 15).
inline double g(int n, double tau) {
    if (tau < 0.0) return 0.0;
    return std::exp(-tau) * std::pow(tau, n - 2) * (tau * tau - 2.0 * n * tau + n * (n - 1.0));
}

}  // namespace detail

/// P_VN·(e·γ/n)^n·t^n·e^(−γt) en Pa; 0 si t < 0. Escrita como P_VN·(γt/n)^n·e^(n − γt) para no desbordar.
inline double pressure_function(const PressureSource& source, double t_s) {
    detail::check_source(source);
    validation::require_finite(t_s, "t_s");
    if (t_s < 0.0) return 0.0;
    const double x = source.gamma_per_s * t_s;
    return source.vn_pressure_pa * std::pow(x / source.n, source.n) * std::exp(source.n - x);
}

/// g(τ) = e^(−τ)·[τ^n − 2n·τ^(n−1) + n(n−1)·τ^(n−2)] (ec. 15); 0 si τ < 0.
inline double g_function(int n, double tau) {
    detail::check_n(n);
    validation::require_finite(tau, "tau");
    return detail::g(n, tau);
}

/// G_n: máximo POSITIVO de g(τ) para τ ≥ 0 (barrido de paso 1e-3 en [0, 4n + 40] y sección áurea alrededor del mejor
/// punto). No es el pico de |g|: para n ≥ 7 el valle negativo es mayor en módulo. Con n = 2 el máximo es g(0) = 2.
inline double g_peak(int n) {
    detail::check_n(n);
    const double h = 1e-3;
    const int steps = static_cast<int>((4.0 * n + 40.0) / h);
    int best = 0;
    for (int i = 1; i <= steps; ++i)
        if (detail::g(n, i * h) > detail::g(n, best * h)) best = i;
    double lo = best > 0 ? (best - 1) * h : 0.0, hi = (best + 1) * h;
    const double phi = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int i = 0; i < 100; ++i) {
        const double x1 = hi - phi * (hi - lo), x2 = lo + phi * (hi - lo);
        if (detail::g(n, x1) < detail::g(n, x2)) lo = x1;
        else hi = x2;
    }
    return std::max(detail::g(n, 0.5 * (lo + hi)), detail::g(n, best * h));
}

/// c = ω_A/γ = [1/(n − 1)] / [(√π/2)·Γ(n/2)/Γ((n+1)/2)] (forma cerrada de la ec. 11, inferida aquí).
inline double mean_frequency_factor(int n) {
    detail::check_n(n);
    const double pi = 3.14159265358979323846;
    return (1.0 / (n - 1.0)) / (0.5 * std::sqrt(pi) * std::tgamma(0.5 * n) / std::tgamma(0.5 * (n + 1.0)));
}

/// Ω_A = c·γ·a/Vp (Blair 2007 ec. 12); Heelan vale con Ω_A ≲ 0,1.
inline double scaled_frequency(const PressureSource& source, const Medium& medium, double hole_radius_m) {
    detail::check_source(source);
    detail::check_medium(medium);
    validation::require_positive(hole_radius_m, "hole_radius_m");
    return mean_frequency_factor(source.n) * source.gamma_per_s * hole_radius_m / medium.p_velocity_m_s;
}

/// F(γ) = γ²·a·L·P_VN·(e/n)^n/(2·μ·Vp) (Blair 2007 ec. 16), en mm/s; L = media longitud del elemento.
inline double heelan_amplitude(const PressureSource& source, const Medium& medium, double hole_radius_m,
                               double half_length_m) {
    detail::check_source(source);
    detail::check_medium(medium);
    validation::require_positive(hole_radius_m, "hole_radius_m");
    validation::require_positive(half_length_m, "half_length_m");
    const double mu = medium.density_kg_m3 * medium.s_velocity_m_s * medium.s_velocity_m_s;
    const double e_over_n = std::exp(1.0) / source.n;
    return 1000.0 * source.gamma_per_s * source.gamma_per_s * hole_radius_m * half_length_m * source.vn_pressure_pa *
           std::pow(e_over_n, source.n) / (2.0 * mu * medium.p_velocity_m_s);
}

/// Velocidad (mm/s) del elemento centrado en el origen en el punto (r, z), para cada instante de t_s (s).
/// P (ec. 13) llega en R/Vp a lo largo de R; SV (ec. 14) llega en R/Vs, perpendicular. Antes de R/Vp todo es 0.
inline Waveform heelan_element(const PressureSource& source, const Medium& medium, double hole_radius_m,
                               double half_length_m, double r, double z, const std::vector<double>& t_s) {
    const double amplitude = heelan_amplitude(source, medium, hole_radius_m, half_length_m);
    validation::require_positive(r, "r");
    validation::require_finite(z, "z");
    for (double t : t_s) validation::require_finite(t, "t_s");
    const double R = std::hypot(r, z);
    const double sin_phi = r / R, cos_phi = -z / R;  // φ = arccos(−z/R), desde el eje −z (Heelan Fig. 1) [V1]
    const double vp = medium.p_velocity_m_s, vs = medium.s_velocity_m_s, gamma = source.gamma_per_s;
    const double base = hole_radius_m * amplitude / R;  // a·F/(r·√(1 + 1/tan²φ)) = a·F/R
    const double p_coef = base * (1.0 - 2.0 * (vs * vs) / (vp * vp) * cos_phi * cos_phi);  // ec. 13
    const double sv_coef = base * (vp / vs) * 2.0 * sin_phi * cos_phi;                   // ec. 14, sen 2φ
    Waveform w{std::vector<double>(t_s.size()), std::vector<double>(t_s.size())};
    for (std::size_t i = 0; i < t_s.size(); ++i) {
        const double vr = p_coef * detail::g(source.n, gamma * (t_s[i] - R / vp));
        const double vphi = sv_coef * detail::g(source.n, gamma * (t_s[i] - R / vs));
        w.radial_mm_s[i] = vr * sin_phi + vphi * cos_phi;
        w.axial_mm_s[i] = -vr * cos_phi + vphi * sin_phi;
    }
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
// Columna de Heelan con velocidad de detonación y superposición no lineal (Blair 2008)

/// Columna cargada entre z_bottom y z_top (m), en `elements` tramos iguales; primer_element = M1 contado desde el fondo.
struct Column {
    double hole_radius_m;
    double z_bottom;
    double z_top;
    int elements;
    int primer_element;
    double vod_m_s;
    double charge_exponent;
};

/// E_m·w_e^(−A) de cada elemento en orden espacial (índice 0 = fondo). Ecs. 1–2 (M1 = 1) y 22 (cualquier M1).
inline std::vector<double> column_weights(int elements, int primer_element, double charge_exponent) {
    validation::require(elements >= 1 && elements % 2 == 1, "elements", "must be an odd integer >= 1", elements);
    validation::require_in_range(primer_element, 1, elements, "primer_element");
    validation::require_finite(charge_exponent, "charge_exponent");
    validation::require(charge_exponent > 0.0 && charge_exponent <= 1.0, "charge_exponent",
                        "must be within (0, 1] (A > 1 has no physical meaning, Blair 2008 p. 240)", charge_exponent);
    const int M = elements;
    const bool mirror = primer_element > (M + 1) / 2;  // cebo más cerca del tope: se calcula reflejado (inferido)
    const int m1 = mirror ? M + 1 - primer_element : primer_element;
    auto p = [charge_exponent](int k) { return std::pow(static_cast<double>(k), charge_exponent); };
    std::vector<double> w(M);
    w[m1 - 1] = 1.0;
    for (int j = 2; j <= m1; ++j) w[m1 - j] = w[m1 + j - 2] = 0.5 * (p(2 * j - 1) - p(2 * j - 3));
    for (int j = m1 + 1; j <= M - m1 + 1; ++j) w[m1 + j - 2] = p(j + m1 - 1) - p(j + m1 - 2);
    if (mirror) std::reverse(w.begin(), w.end());
    return w;
}

/// Pico vectorial: máx √(radial² + axial²) en mm/s (0 si la onda está vacía).
inline double vector_peak(const Waveform& waveform) {
    validation::require(waveform.radial_mm_s.size() == waveform.axial_mm_s.size(), "waveform",
                        "must have radial and axial components of the same length",
                        static_cast<double>(waveform.axial_mm_s.size()));
    double best = 0.0;
    for (std::size_t i = 0; i < waveform.radial_mm_s.size(); ++i)
        best = std::max(best, std::hypot(waveform.radial_mm_s[i], waveform.axial_mm_s[i]));
    return best;
}

/// Velocidad (mm/s) de la columna en (r, z): suma de elementos de Heelan iniciados a la VOD desde el cebo, cada uno
/// con su peso de column_weights.
inline Waveform heelan_column(const PressureSource& source, const Medium& medium, const Column& column, double r,
                              double z, const std::vector<double>& t_s) {
    validation::require_positive(column.hole_radius_m, "hole_radius_m");
    validation::require_finite(column.z_bottom, "z_bottom");
    validation::require_finite(column.z_top, "z_top");
    validation::require(column.z_top > column.z_bottom, "z_top", "must be greater than z_bottom", column.z_top);
    validation::require_positive(column.vod_m_s, "vod_m_s");
    const auto weights = column_weights(column.elements, column.primer_element, column.charge_exponent);
    const double half = (column.z_top - column.z_bottom) / (2.0 * column.elements);
    auto centre = [&](int m) { return column.z_bottom + (2.0 * m + 1.0) * half; };  // m desde 0 (fondo)
    const double z_primer = centre(column.primer_element - 1);
    Waveform total{std::vector<double>(t_s.size(), 0.0), std::vector<double>(t_s.size(), 0.0)};
    std::vector<double> shifted(t_s.size());
    for (int m = 0; m < column.elements; ++m) {
        const double delay = std::fabs(centre(m) - z_primer) / column.vod_m_s;
        for (std::size_t i = 0; i < t_s.size(); ++i) shifted[i] = t_s[i] - delay;
        const auto w = heelan_element(source, medium, column.hole_radius_m, half, r, z - centre(m), shifted);
        for (std::size_t i = 0; i < t_s.size(); ++i) {
            total.radial_mm_s[i] += weights[m] * w.radial_mm_s[i];
            total.axial_mm_s[i] += weights[m] * w.axial_mm_s[i];
        }
    }
    return total;
}

// ---------------------------------------------------------------------------------------------------------------------
// Voladura completa: superposición de la onda semilla, pesos de Blair (2008), apantallamiento y ventana de carga
//
// - superpose: v(t) = Σ_j w_j·s(t − d_j), la semilla s convolucionada con una serie de impulsos (Hinzen 1988 ecs. 3 y
//   5–7, la suma sobre i es inferida porque la ec. 3 se imprime sin sumatoria; Blair 1999 ec. 1a, p. 336) [V1]. Salida
//   desde t = 0 (muestra i en t = i/fs); la llegada entre dos muestras reparte el impulso linealmente (inferido).
// - screening_factor: S = 1/(1 + g·h⁻²·√N_S) con piso 0,2 «por falta de datos de campo cercano» (Blair 1999 ecs. 10b–10c,
//   pp. 345–346) [V1]; g = 23337 de la Fig. 7c; N_S = 0 da S = 1.
// - screening_counts: N_S = pozos ya iniciados cuyo centro cae en el rectángulo de ancho = espaciamiento entre el pozo y
//   el monitor (Blair 1999, Fig. 7a) [V1]; que el rectángulo quede centrado en la recta es inferido del dibujo; «ya
//   iniciado» = tiempo de iniciación estrictamente menor (los de igual tiempo no se cuentan, inferido).
// - charge_scaling_weights: C_n de Blair (2008) ec. 10 tal como está impresa (p. 239) [V1], con r_n (ec. 7), el mismo r_n
//   en ambos términos. NO es la resta F_n − F_{n−1} con cada F con su propio r (da otro valor; ver FUENTES_SPRINT5.md).
//   Primer pozo (h infinito): C_1 = w_1^A. P = 4 por omisión es elección de diseño: la fuente solo pide un entero positivo.
// - damage_weights: E_n = S_n/D_n con D_n (ec. 16), p_ns (ec. 18) y S_n (ec. 20) de Blair (2008, pp. 241–242) [V1], en
//   planta (inferido: pozos y monitor a la misma cota). g va en m² (inferido, para que g/d² sea adimensional).
// - max_charge_per_window: mayor carga que inicia en una ventana [t, t + ventana); 8 ms viene de Duvall et al. (1963)
//   según Blair (1999, p. 337) [V2]; Blair dice que no hay nada especial en 8 ms.

/// v(t) = Σ_j w_j·s(t − d_j) muestreado a fs desde t = 0; largo = ceil(máx llegada·fs) + largo(semilla) + 1.
inline std::vector<double> superpose(const std::vector<double>& seed, double sample_rate_hz,
                                     const std::vector<double>& arrival_times_s, const std::vector<double>& amplitudes) {
    validation::require(!seed.empty(), "seed", "must not be empty", 0.0);
    validation::require_positive(sample_rate_hz, "sample_rate_hz");
    validation::require(arrival_times_s.size() == amplitudes.size(), "amplitudes",
                        "must have the same length as arrival_times_s", static_cast<double>(amplitudes.size()));
    for (double s : seed) validation::require_finite(s, "seed");
    double last = 0.0;
    for (std::size_t j = 0; j < arrival_times_s.size(); ++j) {
        validation::require_non_negative(arrival_times_s[j], "arrival_times_s");
        validation::require_finite(amplitudes[j], "amplitudes");
        last = std::max(last, arrival_times_s[j]);
    }
    std::vector<double> out(static_cast<std::size_t>(std::ceil(last * sample_rate_hz)) + seed.size() + 1, 0.0);
    for (std::size_t j = 0; j < arrival_times_s.size(); ++j) {
        const double pos = arrival_times_s[j] * sample_rate_hz;
        const std::size_t i0 = static_cast<std::size_t>(std::floor(pos));
        const double frac = pos - std::floor(pos);
        for (std::size_t k = 0; k < seed.size(); ++k) {
            out[i0 + k] += amplitudes[j] * (1.0 - frac) * seed[k];
            if (frac > 0.0) out[i0 + k + 1] += amplitudes[j] * frac * seed[k];
        }
    }
    return out;
}

/// S = max(piso, 1/(1 + g·h⁻²·√N_S)) (Blair 1999 ecs. 10b–10c); g en m².
inline double screening_factor(double distance_m, int screening_count, double g = 23337.0, double floor = 0.2) {
    validation::require_positive(distance_m, "distance_m");
    validation::require(screening_count >= 0, "screening_count", "must be >= 0", screening_count);
    validation::require_non_negative(g, "g");
    validation::require_in_range(floor, 0.0, 1.0, "floor");
    return std::max(floor, 1.0 / (1.0 + g / (distance_m * distance_m) * std::sqrt(static_cast<double>(screening_count))));
}

namespace detail {
inline void check_holes(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& t,
                        double monitor_x, double monitor_y) {
    validation::require(x.size() == y.size() && x.size() == t.size(), "y", "x, y and firing_times_s must have the same length",
                        static_cast<double>(y.size()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        validation::require_finite(x[i], "x");
        validation::require_finite(y[i], "y");
        validation::require_finite(t[i], "firing_times_s");
    }
    validation::require_finite(monitor_x, "monitor_x");
    validation::require_finite(monitor_y, "monitor_y");
}
}  // namespace detail

/// N_S de cada pozo (en el orden de entrada): ya iniciados dentro del rectángulo pozo-monitor de ancho `spacing_m`.
inline std::vector<int> screening_counts(const std::vector<double>& x, const std::vector<double>& y,
                                         const std::vector<double>& firing_times_s, double monitor_x, double monitor_y,
                                         double spacing_m) {
    detail::check_holes(x, y, firing_times_s, monitor_x, monitor_y);
    validation::require_positive(spacing_m, "spacing_m");
    std::vector<int> counts(x.size(), 0);
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double ax = monitor_x - x[n], ay = monitor_y - y[n], len = std::hypot(ax, ay);
        validation::require_positive(len, "monitor_x");  // el monitor no puede coincidir con un pozo
        for (std::size_t s = 0; s < x.size(); ++s) {
            if (!(firing_times_s[s] < firing_times_s[n])) continue;
            const double bx = x[s] - x[n], by = y[s] - y[n];
            const double along = (ax * bx + ay * by) / len, across = std::fabs(ax * by - ay * bx) / len;
            if (along >= 0.0 && along <= len && across <= 0.5 * spacing_m) ++counts[n];
        }
    }
    return counts;
}

/// C_n de Blair (2008) ec. 10, con los vectores en orden de iniciación; h = distancia al pozo ya iniciado más cercano
/// (infinito para el primero o si no hay ninguno).
inline std::vector<double> charge_scaling_weights(const std::vector<double>& charges_kg,
                                                  const std::vector<double>& nearest_fired_distance_m, double spacing_m,
                                                  double charge_exponent, double d_factor = 1.0, int p = 4) {
    validation::require(charges_kg.size() == nearest_fired_distance_m.size(), "nearest_fired_distance_m",
                        "must have the same length as charges_kg", static_cast<double>(nearest_fired_distance_m.size()));
    validation::require(!charges_kg.empty(), "charges_kg", "must not be empty", 0.0);
    validation::require_positive(spacing_m, "spacing_m");
    validation::require_positive(d_factor, "d_factor");
    validation::require(p >= 1, "p", "must be an integer >= 1", p);
    validation::require(std::isfinite(charge_exponent) && charge_exponent > 0.0 && charge_exponent <= 1.0, "charge_exponent",
                        "must be within (0, 1] (A > 1 has no physical meaning, Blair 2008 p. 240)", charge_exponent);
    double total = 0.0;
    for (std::size_t n = 0; n < charges_kg.size(); ++n) {
        validation::require_positive(charges_kg[n], "charges_kg");
        const double h = nearest_fired_distance_m[n];
        validation::require(!std::isnan(h) && h > 0.0, "nearest_fired_distance_m", "must be > 0 (infinity allowed)", h);
        total += charges_kg[n];
    }
    const double mean = total / static_cast<double>(charges_kg.size());
    std::vector<double> c(charges_kg.size());
    double before = 0.0;  // Σ_{j<n} w_j
    for (std::size_t n = 0; n < charges_kg.size(); ++n) {
        const double w = charges_kg[n], r = nearest_fired_distance_m[n] / (d_factor * spacing_m) * (mean / w);
        const double frac = 1.0 / (1.0 + std::pow(r, -static_cast<double>(p)));  // r^P/(1 + r^P), sin desbordar con r grande
        const double up = std::pow(before + w, charge_exponent), up_prev = std::pow(before, charge_exponent);
        c[n] = up - up_prev + frac * (std::pow(w, charge_exponent) + up_prev - up);
        before += w;
    }
    return c;
}

/// E_n = S_n/D_n de Blair (2008) ecs. 16 y 20 en planta; resultados en el orden de entrada.
inline std::vector<double> damage_weights(const std::vector<double>& x, const std::vector<double>& y,
                                          const std::vector<double>& charges_kg, const std::vector<double>& firing_times_s,
                                          double monitor_x, double monitor_y, double spacing_m, double eta,
                                          double g = 23337.0) {
    detail::check_holes(x, y, firing_times_s, monitor_x, monitor_y);
    validation::require(charges_kg.size() == x.size(), "charges_kg", "must have the same length as x",
                        static_cast<double>(charges_kg.size()));
    validation::require(!x.empty(), "x", "must not be empty", 0.0);
    validation::require_positive(spacing_m, "spacing_m");
    validation::require_non_negative(eta, "eta");
    validation::require_non_negative(g, "g");
    double mean = 0.0;
    for (double w : charges_kg) {
        validation::require_positive(w, "charges_kg");
        mean += w;
    }
    mean /= static_cast<double>(x.size());
    std::vector<double> e(x.size());
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double ax = monitor_x - x[n], ay = monitor_y - y[n], d = std::hypot(ax, ay);
        validation::require_positive(d, "monitor_x");
        double damage = 0.0, shield = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) {
            if (!(firing_times_s[j] < firing_times_s[n])) continue;
            const double bx = x[j] - x[n], by = y[j] - y[n], h = std::hypot(bx, by);
            validation::require_positive(h, "x");  // dos pozos en el mismo punto
            damage += charges_kg[j] / mean * std::pow(spacing_m / h, 3.0);
            if (ax * bx + ay * by > 0.0) {
                const double perp = std::fabs(ax * by - ay * bx) / d;
                shield += charges_kg[j] / mean * std::exp(-(perp / spacing_m) * (perp / spacing_m));
            }
        }
        e[n] = 1.0 / ((1.0 + g / (d * d) * std::sqrt(shield)) * (1.0 + eta * damage));
    }
    return e;
}

/// Mayor carga que inicia dentro de una ventana [t, t + window_s) con t en cada tiempo de iniciación (0 si no hay pozos).
inline double max_charge_per_window(const std::vector<double>& firing_times_s, const std::vector<double>& charges_kg,
                                    double window_s = 0.008) {
    validation::require(firing_times_s.size() == charges_kg.size(), "charges_kg",
                        "must have the same length as firing_times_s", static_cast<double>(charges_kg.size()));
    validation::require_positive(window_s, "window_s");
    for (std::size_t i = 0; i < charges_kg.size(); ++i) {
        validation::require_finite(firing_times_s[i], "firing_times_s");
        validation::require_non_negative(charges_kg[i], "charges_kg");
    }
    double best = 0.0;
    for (double t0 : firing_times_s) {
        double sum = 0.0;
        for (std::size_t j = 0; j < firing_times_s.size(); ++j)
            if (firing_times_s[j] >= t0 && firing_times_s[j] - t0 < window_s) sum += charges_kg[j];
        best = std::max(best, sum);
    }
    return best;
}

}  // namespace blastingpy::nearfield
