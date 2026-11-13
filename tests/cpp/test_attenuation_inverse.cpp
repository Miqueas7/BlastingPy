#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/attenuation.hpp"

using blastingpy::distance_for_ppv;
using blastingpy::fit_site_law;
using blastingpy::Law;
using blastingpy::Level;
using blastingpy::make_site_law;
using blastingpy::max_charge;
using blastingpy::predict_ppv;
using blastingpy::SiteLaw;

namespace {

const double kNan = std::numeric_limits<double>::quiet_NaN();

std::string error_of(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

bool has(const std::vector<std::string>& warnings, const std::string& text) {
    for (const auto& w : warnings)
        if (w.find(text) != std::string::npos) return true;
    return false;
}

struct Events {
    std::vector<double> d, w, v;
};

// Rejilla sin ruido: D ∈ ds × W ∈ {25,100,400}; General: K·W^a·D^−β; las otras: K·(D/W^e)^−β.
Events grid(const std::vector<double>& ds, double k, double beta, double e_or_a, bool general) {
    Events ev;
    for (double d : ds)
        for (double w : {25.0, 100.0, 400.0}) {
            ev.d.push_back(d);
            ev.w.push_back(w);
            ev.v.push_back(general ? k * std::pow(w, e_or_a) * std::pow(d, -beta)
                                   : k * std::pow(d / std::pow(w, e_or_a), -beta));
        }
    return ev;
}

// Las tres leyes con sigma = 0.115470 (cota 1.7019 veces la mediana).
std::vector<SiteLaw> three_laws() {
    return {make_site_law(Law::SquareRoot, 1000.0, 1.6, kNan, 0.115470),
            make_site_law(Law::CubeRoot, 800.0, 1.4, kNan, 0.115470),
            make_site_law(Law::General, 500.0, 1.5, 0.7, 0.115470)};
}

}  // namespace

TEST(Predict, BriefValues) {
    const auto law = make_site_law(Law::SquareRoot, 1000.0, 1.6);
    EXPECT_NEAR(predict_ppv(law, 100.0, 25.0).value, 8.2864, 1e-3);  // 1000·20^(−1.6)
    EXPECT_NEAR(max_charge(law, 100.0, 8.2864).value, 25.0, 1e-2);
    EXPECT_NEAR(distance_for_ppv(law, 25.0, 8.2864).value, 100.0, 1e-2);
}

TEST(Predict, ThreeLawsFormulas) {
    const auto laws = three_laws();
    EXPECT_NEAR(predict_ppv(laws[1], 100.0, 27.0, Level::Median).value, 800.0 * std::pow(100.0 / 3.0, -1.4), 1e-9);
    EXPECT_NEAR(predict_ppv(laws[2], 100.0, 25.0, Level::Median).value,
                500.0 * std::pow(25.0, 0.7) * std::pow(100.0, -1.5), 1e-9);
}

TEST(Predict, UpperIsMedianTimesTenToZSigma) {
    const auto law = make_site_law(Law::SquareRoot, 1000.0, 1.6, kNan, 0.115470);
    const double median = predict_ppv(law, 100.0, 25.0, Level::Median).value;
    EXPECT_NEAR(predict_ppv(law, 100.0, 25.0, Level::Upper).value / median, 1.7019, 1e-3);
    EXPECT_DOUBLE_EQ(predict_ppv(law, 100.0, 25.0).value, predict_ppv(law, 100.0, 25.0, Level::Upper).value);
    // La carga permitida con la cota es menor que con la mediana, y la distancia necesaria es mayor.
    EXPECT_LT(max_charge(law, 100.0, 10.0, Level::Upper).value, max_charge(law, 100.0, 10.0, Level::Median).value);
    EXPECT_GT(distance_for_ppv(law, 25.0, 5.0, Level::Upper).value,
              distance_for_ppv(law, 25.0, 5.0, Level::Median).value);
}

TEST(Predict, UpperWithoutSigmaOrZWarnsThatThereIsNoMargin) {
    const std::string no_margin = "no safety margin";
    for (const auto& law : {make_site_law(Law::SquareRoot, 1000.0, 1.6),                      // sigma = 0
                            make_site_law(Law::SquareRoot, 1000.0, 1.6, kNan, 0.115470, 0.0)}) {  // z = 0
        EXPECT_TRUE(has(predict_ppv(law, 100.0, 25.0).warnings, no_margin));
        EXPECT_TRUE(has(max_charge(law, 100.0, 5.0).warnings, no_margin));
        EXPECT_TRUE(has(distance_for_ppv(law, 25.0, 5.0).warnings, no_margin));
        EXPECT_TRUE(predict_ppv(law, 100.0, 25.0, Level::Median).warnings.empty());
        EXPECT_TRUE(max_charge(law, 100.0, 5.0, Level::Median).warnings.empty());
        EXPECT_TRUE(distance_for_ppv(law, 25.0, 5.0, Level::Median).warnings.empty());
    }
    const auto with_margin = make_site_law(Law::SquareRoot, 1000.0, 1.6, kNan, 0.115470);
    EXPECT_TRUE(max_charge(with_margin, 100.0, 5.0).warnings.empty());
}

TEST(Predict, RoundTripsInAllThreeLaws) {
    for (const auto& law : three_laws())
        for (Level level : {Level::Median, Level::Upper})
            for (double d : {30.0, 150.0, 700.0})
                for (double v : {1.0, 12.5, 90.0}) {
                    const double w = max_charge(law, d, v, level).value;
                    EXPECT_NEAR(predict_ppv(law, d, w, level).value, v, 1e-9 * v);
                    const double dd = distance_for_ppv(law, w, v, level).value;
                    EXPECT_NEAR(dd, d, 1e-9 * d);
                    EXPECT_NEAR(predict_ppv(law, distance_for_ppv(law, 40.0, v, level).value, 40.0, level).value, v,
                                1e-9 * v);
                }
}

TEST(Predict, PublishedBulletin656ScaledDistance) {
    // USBM Bulletin 656 (Nicholls, Johnson & Duvall 1971), pp. 70-71: con W = (D/SD)^2, la misma vibración
    // exige la misma distancia escalada. SD = 50 y D = 500 -> W = 100; D = 1000 -> W = 400.
    // SD = 20 y D = 500 -> W = 625; D = 1000 -> W = 2500. Vale para cualquier ley de raíz cuadrada.
    const auto law = make_site_law(Law::SquareRoot, 1234.0, 1.37);
    const double v50 = predict_ppv(law, 500.0, 100.0, Level::Median).value;
    EXPECT_NEAR(max_charge(law, 500.0, v50, Level::Median).value, 100.0, 1e-9);
    EXPECT_NEAR(max_charge(law, 1000.0, v50, Level::Median).value, 400.0, 1e-9);
    const double v20 = predict_ppv(law, 500.0, 625.0, Level::Median).value;
    EXPECT_NEAR(max_charge(law, 500.0, v20, Level::Median).value, 625.0, 1e-8);
    EXPECT_NEAR(max_charge(law, 1000.0, v20, Level::Median).value, 2500.0, 1e-7);
}

TEST(Predict, FittedRangeWarnings) {
    const auto ev = grid({50.0, 100.0, 200.0, 500.0}, 1000.0, 1.6, 0.5, false);  // D ∈ [50, 500], W ∈ [25, 400]
    const auto law = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot);
    const std::string out = "outside the fitted range";

    EXPECT_TRUE(has(predict_ppv(law, 1000.0, 100.0).warnings, out));
    EXPECT_TRUE(has(predict_ppv(law, 1000.0, 100.0).warnings, "distance"));
    EXPECT_TRUE(has(predict_ppv(law, 20.0, 100.0).warnings, out));  // por debajo
    EXPECT_TRUE(predict_ppv(law, 200.0, 100.0, Level::Median).warnings.empty());  // dentro de los dos rangos
    // Carga fuera del rango: avisa por la carga, no por la distancia.
    const auto w_only = predict_ppv(law, 200.0, 1000.0).warnings;
    EXPECT_TRUE(has(w_only, "charge"));
    EXPECT_FALSE(has(w_only, "distance"));
    // Dentro de los dos rangos no hay avisos propios de la predicción.
    for (const auto& w : predict_ppv(law, 200.0, 100.0).warnings) EXPECT_EQ(w.find(out), std::string::npos);

    // La carga calculada también cuenta: el PPV de 1000 kg a 200 m da una carga máxima de 1000 kg, fuera de [25, 400].
    const auto big = max_charge(law, 200.0, predict_ppv(law, 200.0, 1000.0).value);
    EXPECT_GT(big.value, 400.0);
    EXPECT_TRUE(has(big.warnings, "charge"));
    EXPECT_FALSE(has(big.warnings, "distance"));
    // La distancia calculada también cuenta.
    const auto far = distance_for_ppv(law, 100.0, 0.01);
    EXPECT_GT(far.value, 500.0);
    EXPECT_TRUE(has(far.warnings, "distance"));
    EXPECT_FALSE(has(far.warnings, "charge"));
    // Distancia de entrada y carga calculada fuera: dos avisos distintos, una por magnitud.
    const auto both = max_charge(law, 1000.0, 50.0, Level::Median);
    EXPECT_TRUE(has(both.warnings, "distance"));
    EXPECT_TRUE(has(both.warnings, "charge"));
    EXPECT_EQ(both.warnings.size(), 2u);
}

TEST(Predict, LawWithoutRangeNeverWarns) {
    const auto law = make_site_law(Law::SquareRoot, 1000.0, 1.6);
    EXPECT_TRUE(std::isnan(law.distance_min_m) && std::isnan(law.distance_max_m));
    EXPECT_TRUE(std::isnan(law.charge_min_kg) && std::isnan(law.charge_max_kg));
    const std::string range = "outside the fitted range";
    for (Level level : {Level::Median, Level::Upper}) {  // con Upper avisa solo de la falta de margen
        EXPECT_FALSE(has(predict_ppv(law, 1e6, 1e-3, level).warnings, range));
        EXPECT_FALSE(has(max_charge(law, 1e6, 1e-3, level).warnings, range));
        EXPECT_FALSE(has(distance_for_ppv(law, 1e6, 1e-3, level).warnings, range));
    }
    EXPECT_TRUE(predict_ppv(law, 1e6, 1e-3, Level::Median).warnings.empty());
    EXPECT_TRUE(max_charge(law, 1e6, 1e-3, Level::Median).warnings.empty());
    EXPECT_TRUE(distance_for_ppv(law, 1e6, 1e-3, Level::Median).warnings.empty());
}

TEST(Predict, RejectsNonPositiveOrNonFiniteInputsByName) {
    const auto law = make_site_law(Law::SquareRoot, 1000.0, 1.6);
    for (double bad : {0.0, -1.0, kNan, static_cast<double>(INFINITY)}) {
        EXPECT_NE(error_of([&] { predict_ppv(law, bad, 25.0); }).find("parameter 'distance_m'"), std::string::npos);
        EXPECT_NE(error_of([&] { predict_ppv(law, 100.0, bad); }).find("parameter 'charge_kg'"), std::string::npos);
        EXPECT_NE(error_of([&] { max_charge(law, bad, 5.0); }).find("parameter 'distance_m'"), std::string::npos);
        EXPECT_NE(error_of([&] { max_charge(law, 100.0, bad); }).find("parameter 'ppv_limit_mm_s'"),
                  std::string::npos);
        EXPECT_NE(error_of([&] { distance_for_ppv(law, bad, 5.0); }).find("parameter 'charge_kg'"),
                  std::string::npos);
        EXPECT_NE(error_of([&] { distance_for_ppv(law, 25.0, bad); }).find("parameter 'ppv_limit_mm_s'"),
                  std::string::npos);
    }
}

TEST(Predict, NonPositiveBetaIsNotUsableForPrediction) {
    const std::vector<double> d = {100, 200, 400, 800}, w(4, 100.0), v = {1, 2, 4, 8};
    const auto law = fit_site_law(d, w, v, Law::SquareRoot);
    ASSERT_LT(law.beta, 0.0);
    const std::string head = "Domain error [BlastingPy]: parameter 'beta'";
    for (const auto& msg : {error_of([&] { predict_ppv(law, 100.0, 25.0); }),
                            error_of([&] { max_charge(law, 100.0, 5.0); }),
                            error_of([&] { distance_for_ppv(law, 25.0, 5.0); })}) {
        EXPECT_EQ(msg.rfind(head, 0), 0u) << msg;
        EXPECT_NE(msg.find("must not be used for prediction"), std::string::npos) << msg;
    }
}

TEST(Predict, NonPositiveChargeExponentIsNotUsableForPrediction) {
    const auto ev = grid({50.0, 100.0, 200.0, 400.0, 800.0}, 500.0, 1.5, -0.4, true);
    const auto law = fit_site_law(ev.d, ev.w, ev.v, Law::General);
    ASSERT_LT(law.charge_exponent, 0.0);
    const std::string head = "Domain error [BlastingPy]: parameter 'charge_exponent'";
    for (const auto& msg : {error_of([&] { predict_ppv(law, 100.0, 25.0); }),
                            error_of([&] { max_charge(law, 100.0, 5.0); }),
                            error_of([&] { distance_for_ppv(law, 25.0, 5.0); })}) {
        EXPECT_EQ(msg.rfind(head, 0), 0u) << msg;
        EXPECT_NE(msg.find("must not be used for prediction"), std::string::npos) << msg;
    }
}

TEST(Predict, NonFiniteResultIsAnError) {
    const auto law = make_site_law(Law::SquareRoot, 1000.0, 1.6);
    EXPECT_THROW(distance_for_ppv(law, 1e300, 1e-300), std::invalid_argument);
    EXPECT_THROW(max_charge(law, 1e300, 1e300), std::invalid_argument);
}
