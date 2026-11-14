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

/// Cuadratura adaptativa Gauss-Kronrod G7K15. El integrando debe ser suave en cada subintervalo.
template <class F>
double gk15(const F& f, double a, double b, double abs_tol, int depth) {
    static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                  0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                  0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                                  0.207784955007898467600689403773245, 0.0};
    static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                  0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static const double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                 0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    const double c = 0.5 * (a + b), h = 0.5 * (b - a);
    const double fc = f(c);
    double k = wgk[7] * fc, g = wg[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double v = f(c - h * xgk[j]) + f(c + h * xgk[j]);
        k += wgk[j] * v;
        if (j % 2 == 1) g += wg[j / 2] * v;
    }
    k *= h;
    g *= h;
    if (depth <= 0 || std::fabs(k - g) <= abs_tol) return k;
    return gk15(f, a, c, 0.5 * abs_tol, depth - 1) + gk15(f, c, b, 0.5 * abs_tol, depth - 1);
}

/// Integra f sobre [a, b], partiendo en `split` si cae dentro (el pico del integrando está en z' = z).
/// Tolerancia absoluta 1e-11 × una estimación gruesa de la integral (error relativo ~1e-10).
template <class F>
double integrate(const F& f, double a, double b, double split) {
    const double s = (split > a && split < b) ? split : 0.5 * (a + b);
    const double rough = std::fabs(gk15(f, a, s, 1e300, 0)) + std::fabs(gk15(f, s, b, 1e300, 0));
    const double tol = 1e-11 * rough;
    return gk15(f, a, s, 0.5 * tol, 60) + gk15(f, s, b, 0.5 * tol, 60);
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
    return linear_charge_kg_m / r * (std::atan((z_end - z) / r) + std::atan((z - z_start) / r));
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
        const double e = 1.0 - beta / alpha;
        const double integral = (std::pow(z_end - z, e) + std::pow(z - z_start, e)) / e;
        return k * std::pow(linear_charge_kg_m * integral, alpha);
    }
    if (r > 0.0 && std::fabs(beta - 2.0 * alpha) <= 1e-12)
        return k * std::pow(hp_term(linear_charge_kg_m, z_start, z_end, r, z), alpha);
    const double p = beta / (2.0 * alpha), r2 = r * r;
    auto f = [&](double zp) { return std::pow(r2 + (zp - z) * (zp - z), -p); };
    const double integral = detail::integrate(f, z_start, z_end, z);
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
/// Con Original y β ≥ α el PPV crece sin tope sobre el tramo cargado; con β < α es finito en el eje y también puede dar NaN.
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

}  // namespace blastingpy::nearfield
