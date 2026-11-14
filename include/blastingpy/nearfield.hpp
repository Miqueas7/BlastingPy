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
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <cmath>
#include <limits>

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
    validation::require(!(r == 0.0 && z >= z_start && z <= z_end), "r",
                        "must be > 0 when z lies within the charged span (the integral diverges on the axis)", r);
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
    double lo = 1e-9, hi = 1e4;
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

}  // namespace blastingpy::nearfield
