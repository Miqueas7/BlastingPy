/**
 * @file attenuation.hpp
 * @brief Ley de atenuación del sitio: ajuste de PPV = K·(D/W^e)^(−β) (o K·W^a·D^(−β)) y su cota superior.
 *
 * Convenciones (todas en log10):
 * - residuo = log10(PPV observada) − log10(PPV mediana predicha);
 * - sigma = ols(...).sigma = sqrt(SSE/dof), con dof = n − 2 en las leyes de distancia escalada y n − 3 en General;
 * - r_squared es el de la regresión log-log;
 * - cota superior: k_upper = k·10^(z·sigma), con z = 2 por omisión (RI 8507 p. 16 y BIVDEP de OSMRE usan 2σ);
 *   coverage = Φ(z) = 0.5·erfc(−z/√2), la fracción esperada de eventos bajo la cota;
 * - outliers: índices con |residuo| > outlier_z·sigma; no hay ninguno si sigma es 0;
 * - los rangos son el mínimo y el máximo de las distancias y cargas de entrada.
 * El Bulletin 656 escribe el exponente como pendiente negativa; aquí beta se guarda positivo.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/least_squares.hpp"
#include "blastingpy/method_info.hpp"
#include "blastingpy/units.hpp"
#include "blastingpy/validation.hpp"

namespace blastingpy {

enum class Law { SquareRoot, CubeRoot, General };

/// Nivel de la ley: Median usa k; Upper usa k_upper = k·10^(z·sigma).
enum class Level { Median, Upper };

struct SiteLaw {
    Law law;
    double k;                // mediana (K50), en mm/s-m-kg (o en las unidades de entrada)
    double beta;             // exponente de atenuación, positivo
    double charge_exponent;  // SquareRoot 0.5, CubeRoot 1/3 (exponente de W en la distancia escalada); General: a
    double sigma;            // error estándar de los residuos en log10: sqrt(SSE/dof)
    int n;
    int dof;
    double r_squared;
    double z;                // número de sigmas de la cota superior
    double coverage;         // Φ(z)
    double k_upper;          // k·10^(z·sigma)
    std::string convention;
    double distance_min_m, distance_max_m, charge_min_kg, charge_max_kg;
    std::vector<double> residuals;
    std::vector<int> outliers;
    MethodInfo info;
    std::vector<std::string> warnings;
};

namespace attenuation_detail {

inline constexpr const char* convention = "log10; upper bound = k*10^(z*sigma); sigma = sqrt(SSE/dof)";

inline double fixed_charge_exponent(Law law) { return law == Law::SquareRoot ? 0.5 : 1.0 / 3.0; }

// Cita y marca de FUENTES_SPRINT1.md, `## Leyes de atenuación`. Bytes UTF-8 explícitos (α, raya –): MSVC compila
// sin /utf-8.
inline MethodInfo law_info(Law law) {
    switch (law) {
        case Law::SquareRoot:
            return {"ATT-SQRT-USBM656", Verification::Primary,
                    "Nicholls, H.R., Johnson, C.F. & Duvall, W.I. (1971). Blasting Vibrations and Their Effects on "
                    "Structures. USBM Bulletin 656, eq. 4.20 (p. 49) and eq. 6.1 (p. 70)"};
        case Law::CubeRoot:
            return {"ATT-CBRT-USBM656", Verification::Secondary,
                    "Nicholls, Johnson & Duvall (1971), USBM Bulletin 656, eq. 4.13 (p. 41) with \xCE\xB1 = 1/3; "
                    "cube-root scaling discussed as other investigators' proposal, p. 49"};
        default:
            return {"ATT-GENERAL-USBM656", Verification::Primary,
                    "Nicholls, Johnson & Duvall (1971), USBM Bulletin 656, eqs. 4.10\xE2\x80\x93"
                    "4.12 (p. 40); general form eq. 4.13 (p. 41)"};
    }
}

inline double coverage(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

/// Una ley con beta <= 0 (o a <= 0 en General) puede existir como ajuste, pero no sirve para predecir: la carga máxima
/// saldría infinita o invertida. Se rechaza aquí, antes de cualquier cálculo.
inline void require_usable(const SiteLaw& f) {
    namespace v = validation;
    v::require(f.beta > 0.0, "beta", "is not positive, so this law must not be used for prediction", f.beta);
    if (f.law == Law::General)
        v::require(f.charge_exponent > 0.0, "charge_exponent",
                   "is not positive, so this law must not be used for prediction", f.charge_exponent);
}

inline double level_k(const SiteLaw& f, Level level) { return level == Level::Upper ? f.k_upper : f.k; }

/// Exponente p de W en v = K·W^p·D^(−β): e·β en las leyes de distancia escalada, a en General.
inline double charge_power(const SiteLaw& f) {
    return f.law == Law::General ? f.charge_exponent : f.charge_exponent * f.beta;
}

/// Un rango NaN (ley hecha con make_site_law) significa «sin rango que avisar».
inline void warn_if_outside(std::vector<std::string>& warnings, const std::string& what, const std::string& unit,
                            double value, double lo, double hi) {
    if (std::isnan(lo) || std::isnan(hi) || (value >= lo && value <= hi)) return;
    namespace d = validation::detail;
    warnings.push_back(what + " " + d::num(value) + " " + unit + " is outside the fitted range [" + d::num(lo) + ", " +
                       d::num(hi) + "] " + unit + ": extrapolation");
}

inline double finite_result(double value, const char* name) {
    validation::require(std::isfinite(value) && value > 0.0, name,
                        "gives a result outside the numeric range (not finite or underflows to 0)", value);
    return value;
}

}  // namespace attenuation_detail

/// Resultado de una predicción o inversa, con los avisos de extrapolación.
struct Prediction {
    double value;
    std::vector<std::string> warnings;
};

/// PPV (mm/s) a `distance_m` con `charge_kg` por retardo; Upper usa k_upper, Median usa k.
inline Prediction predict_ppv(const SiteLaw& law, double distance_m, double charge_kg, Level level = Level::Upper) {
    namespace a = attenuation_detail;
    validation::require_positive(distance_m, "distance_m");
    validation::require_positive(charge_kg, "charge_kg");
    a::require_usable(law);
    Prediction r;
    r.value = a::finite_result(a::level_k(law, level) * std::pow(charge_kg, a::charge_power(law)) *
                                   std::pow(distance_m, -law.beta),
                               "distance_m");
    a::warn_if_outside(r.warnings, "distance", "m", distance_m, law.distance_min_m, law.distance_max_m);
    a::warn_if_outside(r.warnings, "charge", "kg", charge_kg, law.charge_min_kg, law.charge_max_kg);
    return r;
}

/// Carga máxima por retardo (kg) a `distance_m` para que la PPV no pase de `ppv_limit_mm_s`.
inline Prediction max_charge(const SiteLaw& law, double distance_m, double ppv_limit_mm_s, Level level = Level::Upper) {
    namespace a = attenuation_detail;
    validation::require_positive(distance_m, "distance_m");
    validation::require_positive(ppv_limit_mm_s, "ppv_limit_mm_s");
    a::require_usable(law);
    Prediction r;
    r.value = a::finite_result(
        std::pow(ppv_limit_mm_s * std::pow(distance_m, law.beta) / a::level_k(law, level), 1.0 / a::charge_power(law)),
        "ppv_limit_mm_s");
    a::warn_if_outside(r.warnings, "distance", "m", distance_m, law.distance_min_m, law.distance_max_m);
    a::warn_if_outside(r.warnings, "computed charge", "kg", r.value, law.charge_min_kg, law.charge_max_kg);
    return r;
}

/// Distancia mínima (m) a la que `charge_kg` por retardo no pasa de `ppv_limit_mm_s`.
inline Prediction distance_for_ppv(const SiteLaw& law, double charge_kg, double ppv_limit_mm_s,
                                   Level level = Level::Upper) {
    namespace a = attenuation_detail;
    validation::require_positive(charge_kg, "charge_kg");
    validation::require_positive(ppv_limit_mm_s, "ppv_limit_mm_s");
    a::require_usable(law);
    Prediction r;
    r.value = a::finite_result(
        std::pow(a::level_k(law, level) * std::pow(charge_kg, a::charge_power(law)) / ppv_limit_mm_s, 1.0 / law.beta),
        "ppv_limit_mm_s");
    a::warn_if_outside(r.warnings, "charge", "kg", charge_kg, law.charge_min_kg, law.charge_max_kg);
    a::warn_if_outside(r.warnings, "computed distance", "m", r.value, law.distance_min_m, law.distance_max_m);
    return r;
}

/// Cuantil de la normal estándar, 0 < p < 1. normal_quantile(0.95) = 1.645 da la cota unilateral del 95 %.
inline double normal_quantile(double p) {
    validation::require_finite(p, "p");
    validation::require(p > 0.0 && p < 1.0, "p", "must be within (0, 1)", p);
    // Bisección sobre Φ(z) = 0.5·erfc(−z/√2); 200 pasos llegan al double vecino.
    double lo = -40.0, hi = 40.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        (attenuation_detail::coverage(mid) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

/// Ajusta la ley del sitio por mínimos cuadrados en log10 (v en mm/s, D en m, W en kg por retardo).
inline SiteLaw fit_site_law(const std::vector<double>& distance_m, const std::vector<double>& charge_kg,
                            const std::vector<double>& ppv_mm_s, Law law, double z = 2.0, double outlier_z = 3.0) {
    namespace v = validation;
    const std::size_t n = ppv_mm_s.size();
    v::require(distance_m.size() == n, "distance_m", "must have the same length as ppv_mm_s",
               static_cast<double>(distance_m.size()));
    v::require(charge_kg.size() == n, "charge_kg", "must have the same length as ppv_mm_s",
               static_cast<double>(charge_kg.size()));
    for (std::size_t i = 0; i < n; ++i) {
        v::require_positive(distance_m[i], "distance_m");
        v::require_positive(charge_kg[i], "charge_kg");
        v::require_positive(ppv_mm_s[i], "ppv_mm_s");
    }
    v::require_non_negative(z, "z");
    v::require_positive(outlier_z, "outlier_z");
    const bool general = law == Law::General;
    const std::size_t params = general ? 3 : 2;
    v::require(n >= params + 1, "ppv_mm_s",
               "needs at least " + std::to_string(params + 1) + " events for this law (parameters + 1)",
               static_cast<double>(n));

    const double e = general ? 0.0 : attenuation_detail::fixed_charge_exponent(law);
    std::vector<double> y(n), x1(n), x2(general ? n : 0);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = std::log10(ppv_mm_s[i]);
        if (general) {
            x1[i] = std::log10(charge_kg[i]);
            x2[i] = std::log10(distance_m[i]);
        } else {
            x1[i] = std::log10(units::scaled_distance(distance_m[i], charge_kg[i], e));
        }
    }

    OlsResult r;
    try {
        r = general ? ols({x1, x2}, y) : ols({x1}, y);
    } catch (const std::invalid_argument&) {
        // Lo único que queda por fallar en ols es una columna constante o colineal: se nombra la causa real.
        auto all_equal = [](const std::vector<double>& a) {
            return std::all_of(a.begin(), a.end(), [&](double x) { return x == a.front(); });
        };
        if (!general)
            v::detail::fail("distance_m", "gives the same scaled distance D / W^e in every event, so beta cannot be fitted",
                            std::pow(10.0, x1.front()));
        if (all_equal(charge_kg))
            v::detail::fail("charge_kg",
                            "is the same in every event, so its exponent cannot be fitted with Law.General; "
                            "use a scaled-distance law (Law.SquareRoot or Law.CubeRoot)",
                            charge_kg.front());
        if (all_equal(distance_m))
            v::detail::fail("distance_m", "is the same in every event, so beta cannot be fitted", distance_m.front());
        v::detail::fail("charge_kg",
                        "is (nearly) collinear with distance_m in log10, so Law.General cannot separate their "
                        "exponents; use a scaled-distance law (Law.SquareRoot or Law.CubeRoot)",
                        static_cast<double>(n));
    }

    SiteLaw f;
    f.law = law;
    f.k = std::pow(10.0, r.coefficients[0]);
    f.beta = -r.coefficients[general ? 2 : 1];
    f.charge_exponent = general ? r.coefficients[1] : e;
    f.sigma = r.sigma;
    f.n = static_cast<int>(n);
    f.dof = r.dof;
    f.r_squared = r.r_squared;
    f.z = z;
    f.coverage = attenuation_detail::coverage(z);
    f.k_upper = f.k * std::pow(10.0, z * f.sigma);
    f.convention = attenuation_detail::convention;
    const auto [dmin, dmax] = std::minmax_element(distance_m.begin(), distance_m.end());
    const auto [wmin, wmax] = std::minmax_element(charge_kg.begin(), charge_kg.end());
    f.distance_min_m = *dmin;
    f.distance_max_m = *dmax;
    f.charge_min_kg = *wmin;
    f.charge_max_kg = *wmax;
    f.residuals = r.residuals;
    if (f.sigma > 0.0)
        for (std::size_t i = 0; i < n; ++i)
            if (std::fabs(f.residuals[i]) > outlier_z * f.sigma) f.outliers.push_back(static_cast<int>(i));
    f.info = attenuation_detail::law_info(law);
    // Recomendación de BIVDEP (p. 4, punto 17), no requisito: por eso es aviso y no error.
    if (n < 30) f.warnings.push_back("fewer than 30 events: OSMRE BIVDEP (2019) asks for a minimum of 30");
    if (f.beta <= 0.0)
        f.warnings.push_back(
            "attenuation exponent beta is not positive (PPV does not decrease with distance): "
            "this law must not be used for prediction");
    // Con a ≤ 0 la inversa W = (v·D^β/K)^(1/a) divide entre cero o convierte el máximo en mínimo.
    if (general && f.charge_exponent <= 0.0)
        f.warnings.push_back(
            "charge exponent a is not positive (PPV does not increase with charge): "
            "this law must not be used for prediction");
    return f;
}

/// Ley dada por el usuario (sin ajuste). En General `charge_exponent` es obligatorio y > 0; en las otras se omite (NaN) o
/// debe ser igual al exponente fijo de la ley.
inline SiteLaw make_site_law(Law law, double k, double beta, double charge_exponent = validation::unset,
                             double sigma = 0.0, double z = 2.0) {
    namespace v = validation;
    v::require_positive(k, "k");
    v::require_positive(beta, "beta");
    v::require_non_negative(sigma, "sigma");
    v::require_non_negative(z, "z");
    if (law == Law::General) {
        v::require(!std::isnan(charge_exponent), "charge_exponent", "is required for Law.General", charge_exponent);
        v::require_positive(charge_exponent, "charge_exponent");
    } else {
        const double fixed = attenuation_detail::fixed_charge_exponent(law);
        if (std::isnan(charge_exponent)) charge_exponent = fixed;
        v::require(std::fabs(charge_exponent - fixed) <= 1e-12, "charge_exponent",
                   "must be omitted or equal to the law's fixed exponent (" + v::detail::num(fixed) + ")",
                   charge_exponent);
    }

    SiteLaw f;
    f.law = law;
    f.k = k;
    f.beta = beta;
    f.charge_exponent = charge_exponent;
    f.sigma = sigma;
    f.n = 0;
    f.dof = 0;
    f.r_squared = v::unset;
    f.z = z;
    f.coverage = attenuation_detail::coverage(z);
    f.k_upper = k * std::pow(10.0, z * sigma);
    f.convention = attenuation_detail::convention;
    f.distance_min_m = f.distance_max_m = f.charge_min_kg = f.charge_max_kg = v::unset;
    f.info = attenuation_detail::law_info(law);
    f.info.verification = Verification::UserDefined;
    return f;
}

}  // namespace blastingpy
