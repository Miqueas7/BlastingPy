#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/attenuation.hpp"

using blastingpy::fit_site_law;
using blastingpy::Law;
using blastingpy::make_site_law;
using blastingpy::normal_quantile;
using blastingpy::SiteLaw;
using blastingpy::Verification;

namespace {

struct Events {
    std::vector<double> d, w, v;
};

// Rejilla D ∈ {50,100,200,400,800} × W ∈ {25,100,400}, sin ruido.
Events grid(double k, double beta, double e_or_a, bool general) {
    Events ev;
    for (double d : {50.0, 100.0, 200.0, 400.0, 800.0})
        for (double w : {25.0, 100.0, 400.0}) {
            ev.d.push_back(d);
            ev.w.push_back(w);
            ev.v.push_back(general ? k * std::pow(w, e_or_a) * std::pow(d, -beta)
                                   : k * std::pow(d / std::pow(w, e_or_a), -beta));
        }
    return ev;
}

// Cuatro distancias escaladas (W = 100, D = 100..800) con un par ppv·10^(±0.1) cada una: n = 8, dof = 6.
Events pairs() {
    Events ev;
    for (double d : {100.0, 200.0, 400.0, 800.0})
        for (double s : {0.1, -0.1}) {
            ev.d.push_back(d);
            ev.w.push_back(100.0);
            ev.v.push_back(1000.0 * std::pow(d / 10.0, -1.6) * std::pow(10.0, s));
        }
    return ev;
}

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

void expect_exact(const SiteLaw& f, double k, double beta) {
    EXPECT_NEAR(f.k, k, 1e-6);
    EXPECT_NEAR(f.beta, beta, 1e-9);
    EXPECT_LT(f.sigma, 1e-9);
    EXPECT_NEAR(f.r_squared, 1.0, 1e-12);
    EXPECT_NEAR(f.k_upper, f.k, 1e-6);
}

}  // namespace

TEST(FitSiteLaw, RecoversSquareRoot) {
    const auto ev = grid(1000.0, 1.6, 0.5, false);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot);
    expect_exact(f, 1000.0, 1.6);
    EXPECT_EQ(f.law, Law::SquareRoot);
    EXPECT_DOUBLE_EQ(f.charge_exponent, 0.5);
    EXPECT_EQ(f.n, 15);
    EXPECT_EQ(f.dof, 13);
    EXPECT_EQ(f.residuals.size(), 15u);
    EXPECT_TRUE(f.outliers.empty());
    EXPECT_DOUBLE_EQ(f.distance_min_m, 50.0);
    EXPECT_DOUBLE_EQ(f.distance_max_m, 800.0);
    EXPECT_DOUBLE_EQ(f.charge_min_kg, 25.0);
    EXPECT_DOUBLE_EQ(f.charge_max_kg, 400.0);
}

TEST(FitSiteLaw, RecoversCubeRoot) {
    const auto ev = grid(1000.0, 1.6, 1.0 / 3.0, false);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::CubeRoot);
    expect_exact(f, 1000.0, 1.6);
    EXPECT_DOUBLE_EQ(f.charge_exponent, 1.0 / 3.0);
}

TEST(FitSiteLaw, RecoversGeneral) {
    const auto ev = grid(500.0, 1.5, 0.7, true);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::General);
    expect_exact(f, 500.0, 1.5);
    EXPECT_NEAR(f.charge_exponent, 0.7, 1e-9);
    EXPECT_EQ(f.dof, 12);
}

TEST(FitSiteLaw, UpperBoundIsTwoSigmaByDefault) {
    const auto ev = pairs();
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot);
    EXPECT_NEAR(f.k, 1000.0, 1e-6);
    EXPECT_EQ(f.n, 8);
    EXPECT_EQ(f.dof, 6);
    EXPECT_NEAR(f.sigma, 0.115470, 1e-6);  // sqrt(8·0.01 / 6)
    EXPECT_DOUBLE_EQ(f.z, 2.0);
    EXPECT_NEAR(f.coverage, 0.977250, 1e-6);
    EXPECT_NEAR(f.k_upper, 1701.9, 0.5);  // 1000·10^(2·0.115470)
    // residuo = log10(observada) − log10(mediana): el primer punto se armó con +0.1.
    EXPECT_NEAR(f.residuals[0], 0.1, 1e-12);
    EXPECT_NEAR(f.residuals[1], -0.1, 1e-12);

    const auto g = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot, normal_quantile(0.95));
    EXPECT_NEAR(g.k_upper, 1548.6, 0.5);  // 1000·10^(1.6448536·0.115470)
    EXPECT_NEAR(g.coverage, 0.95, 1e-12);
}

TEST(NormalQuantile, KnownValues) {
    EXPECT_NEAR(normal_quantile(0.95), 1.6448536, 1e-6);
    EXPECT_NEAR(normal_quantile(0.975), 1.9599640, 1e-6);
    EXPECT_NEAR(normal_quantile(0.5), 0.0, 1e-9);
    EXPECT_NEAR(normal_quantile(0.05), -1.6448536, 1e-6);
    EXPECT_THROW(normal_quantile(0.0), std::invalid_argument);
    EXPECT_THROW(normal_quantile(1.0), std::invalid_argument);
    EXPECT_THROW(normal_quantile(std::nan("")), std::invalid_argument);
}

TEST(FitSiteLaw, ConventionAndWarnings) {
    const auto ev = grid(1000.0, 1.6, 0.5, false);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot);
    EXPECT_EQ(f.convention.rfind("log10", 0), 0u);
    EXPECT_TRUE(has(f.warnings, "fewer than 30 events: OSMRE BIVDEP (2019) asks for a minimum of 30"));

    Events big;
    for (int i = 0; i < 30; ++i) {
        big.d.push_back(50.0 + 10.0 * i);
        big.w.push_back(100.0);
        big.v.push_back(1000.0 * std::pow(big.d.back() / 10.0, -1.6));
    }
    const auto g = fit_site_law(big.d, big.w, big.v, Law::SquareRoot);
    EXPECT_FALSE(has(g.warnings, "30"));
    EXPECT_TRUE(g.warnings.empty());
}

TEST(FitSiteLaw, OutlierIsFlaggedAndStillCounted) {
    Events ev;
    for (int i = 0; i < 40; ++i) {
        ev.d.push_back(40.0 + 15.0 * i);
        ev.w.push_back(i % 2 ? 100.0 : 225.0);
        ev.v.push_back(1000.0 * std::pow(ev.d.back() / std::sqrt(ev.w.back()), -1.6));
    }
    ev.d.push_back(300.0);
    ev.w.push_back(100.0);
    ev.v.push_back(10.0 * 1000.0 * std::pow(30.0, -1.6));
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::SquareRoot);
    EXPECT_EQ(f.n, 41);
    ASSERT_EQ(f.outliers.size(), 1u);
    EXPECT_EQ(f.outliers[0], 40);
}

TEST(FitSiteLaw, NonPositiveBetaIsAWarningNotAnError) {
    const std::vector<double> d = {100, 200, 400, 800}, w(4, 100.0), v = {1, 2, 4, 8};
    const auto f = fit_site_law(d, w, v, Law::SquareRoot);
    EXPECT_LT(f.beta, 0.0);
    EXPECT_TRUE(has(f.warnings, "not positive"));
}

TEST(FitSiteLaw, NonPositiveChargeExponentIsAWarningNotAnError) {
    // Con a ≤ 0 la inversa W = (v·D^β/K)^(1/a) divide entre cero o vuelve mínimo el máximo.
    const auto ev = grid(500.0, 1.5, -0.4, true);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::General);
    EXPECT_NEAR(f.charge_exponent, -0.4, 1e-9);
    EXPECT_TRUE(has(f.warnings, "charge exponent a is not positive"));
    EXPECT_TRUE(has(f.warnings, "must not be used for prediction"));

    const auto ok = grid(500.0, 1.5, 0.7, true);
    const auto g = fit_site_law(ok.d, ok.w, ok.v, Law::General);
    EXPECT_FALSE(has(g.warnings, "charge exponent"));
}

TEST(FitSiteLaw, RejectsNonPositiveInputsByName) {
    const std::vector<double> d = {100, 200, 400}, w = {100, 100, 100}, v = {5, 2, 1};
    EXPECT_NE(error_of([&] { fit_site_law({100, 0, 400}, w, v, Law::SquareRoot); }).find("'distance_m'"),
              std::string::npos);
    EXPECT_NE(error_of([&] { fit_site_law(d, {100, -1, 100}, v, Law::SquareRoot); }).find("'charge_kg'"),
              std::string::npos);
    EXPECT_NE(error_of([&] { fit_site_law(d, w, {5, 0, 1}, Law::SquareRoot); }).find("'ppv_mm_s'"),
              std::string::npos);
    EXPECT_THROW(fit_site_law(d, {100, 100}, v, Law::SquareRoot), std::invalid_argument);  // largos distintos
}

TEST(FitSiteLaw, RejectsTooFewEvents) {
    EXPECT_THROW(fit_site_law({100, 200}, {100, 100}, {5, 2}, Law::SquareRoot), std::invalid_argument);
    EXPECT_THROW(fit_site_law({100, 200, 400}, {100, 50, 25}, {5, 2, 1}, Law::General), std::invalid_argument);
    EXPECT_NO_THROW(fit_site_law({100, 200, 400}, {100, 100, 100}, {5, 2, 1}, Law::SquareRoot));
}

TEST(FitSiteLaw, RejectsSameScaledDistance) {
    // D / W^½ = 10 en los tres eventos.
    const auto msg = error_of([] { fit_site_law({10, 20, 40}, {1, 4, 16}, {5, 4, 6}, Law::SquareRoot); });
    EXPECT_EQ(msg.rfind("Domain error [BlastingPy]: parameter 'distance_m'", 0), 0u);
    EXPECT_NE(msg.find("same scaled distance"), std::string::npos);
}

TEST(FitSiteLaw, GeneralWithConstantChargeSuggestsAScaledLaw) {
    const auto msg = error_of(
        [] { fit_site_law({100, 200, 400, 800}, std::vector<double>(4, 50.0), {9, 4, 2, 1}, Law::General); });
    EXPECT_EQ(msg.rfind("Domain error [BlastingPy]: parameter 'charge_kg'", 0), 0u);
    EXPECT_NE(msg.find("scaled-distance law"), std::string::npos);

    const auto same_d = error_of(
        [] { fit_site_law(std::vector<double>(4, 100.0), {10, 20, 40, 80}, {1, 2, 3, 4}, Law::General); });
    EXPECT_EQ(same_d.rfind("Domain error [BlastingPy]: parameter 'distance_m'", 0), 0u);
}

TEST(FitSiteLaw, RejectsBadZ) {
    const std::vector<double> d = {100, 200, 400}, w = {100, 100, 100}, v = {5, 2, 1};
    EXPECT_THROW(fit_site_law(d, w, v, Law::SquareRoot, -1.0), std::invalid_argument);
    EXPECT_THROW(fit_site_law(d, w, v, Law::SquareRoot, std::nan("")), std::invalid_argument);
    EXPECT_THROW(fit_site_law(d, w, v, Law::SquareRoot, INFINITY), std::invalid_argument);
    EXPECT_THROW(fit_site_law(d, w, v, Law::SquareRoot, 2.0, 0.0), std::invalid_argument);
}

TEST(FitSiteLaw, MethodInfoMatchesSources) {
    const auto sq = grid(1000.0, 1.6, 0.5, false);
    const auto cb = grid(1000.0, 1.6, 1.0 / 3.0, false);
    const auto ge = grid(500.0, 1.5, 0.7, true);
    const auto a = fit_site_law(sq.d, sq.w, sq.v, Law::SquareRoot).info;
    const auto b = fit_site_law(cb.d, cb.w, cb.v, Law::CubeRoot).info;
    const auto c = fit_site_law(ge.d, ge.w, ge.v, Law::General).info;
    EXPECT_EQ(a.method_id, "ATT-SQRT-USBM656");
    EXPECT_EQ(a.verification, Verification::Primary);
    EXPECT_EQ(a.source,
              "Nicholls, H.R., Johnson, C.F. & Duvall, W.I. (1971). Blasting Vibrations and Their Effects on "
              "Structures. USBM Bulletin 656, eq. 4.20 (p. 49) and eq. 6.1 (p. 70)");
    EXPECT_EQ(b.method_id, "ATT-CBRT-USBM656");
    EXPECT_EQ(b.verification, Verification::Secondary);
    EXPECT_EQ(b.source,
              "Nicholls, Johnson & Duvall (1971), USBM Bulletin 656, eq. 4.13 (p. 41) with \xCE\xB1 = 1/3; "
              "cube-root scaling discussed as other investigators' proposal, p. 49");
    EXPECT_EQ(c.method_id, "ATT-GENERAL-USBM656");
    EXPECT_EQ(c.verification, Verification::Primary);
    EXPECT_EQ(c.source,
              "Nicholls, Johnson & Duvall (1971), USBM Bulletin 656, eqs. 4.10\xE2\x80\x93"
              "4.12 (p. 40); general form eq. 4.13 (p. 41)");
}

TEST(MakeSiteLaw, BuildsAUserDefinedLaw) {
    const auto f = make_site_law(Law::SquareRoot, 1000.0, 1.6, std::numeric_limits<double>::quiet_NaN(), 0.115470);
    EXPECT_DOUBLE_EQ(f.charge_exponent, 0.5);
    EXPECT_NEAR(f.k_upper, 1701.9, 0.5);
    EXPECT_NEAR(f.coverage, 0.977250, 1e-6);
    EXPECT_EQ(f.info.method_id, "ATT-SQRT-USBM656");
    EXPECT_EQ(f.info.verification, Verification::UserDefined);
    EXPECT_EQ(f.n, 0);
    EXPECT_TRUE(f.residuals.empty());

    const auto g = make_site_law(Law::General, 500.0, 1.5, 0.7);
    EXPECT_DOUBLE_EQ(g.charge_exponent, 0.7);
    EXPECT_DOUBLE_EQ(g.k_upper, 500.0);  // sigma = 0
    EXPECT_EQ(g.info.method_id, "ATT-GENERAL-USBM656");
    EXPECT_DOUBLE_EQ(make_site_law(Law::CubeRoot, 1.0, 1.0, 1.0 / 3.0).charge_exponent, 1.0 / 3.0);
}

TEST(MakeSiteLaw, RejectsInvalidParameters) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NE(error_of([] { make_site_law(Law::General, 500.0, 1.5); }).find("'charge_exponent'"), std::string::npos);
    EXPECT_THROW(make_site_law(Law::SquareRoot, 1000.0, 1.6, 0.4), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::SquareRoot, 0.0, 1.6), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::SquareRoot, 1000.0, 0.0), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::SquareRoot, 1000.0, 1.6, nan, -0.1), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::SquareRoot, 1000.0, 1.6, nan, 0.1, -1.0), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::SquareRoot, INFINITY, 1.6), std::invalid_argument);
    EXPECT_THROW(make_site_law(Law::General, 500.0, 1.5, INFINITY), std::invalid_argument);
    EXPECT_EQ(error_of([] { make_site_law(Law::General, 500.0, 1.5, 0.0); })
                  .rfind("Domain error [BlastingPy]: parameter 'charge_exponent'", 0),
              0u);
    EXPECT_EQ(error_of([] { make_site_law(Law::General, 500.0, 1.5, -0.5); })
                  .rfind("Domain error [BlastingPy]: parameter 'charge_exponent'", 0),
              0u);
}

// Sobrepresión (sprint 3): misma regresión, con cita de RI 8485 y avisos sin PPV.
TEST(FitSiteLaw, AirblastCitesRI8485AndSpeaksOfOverpressure) {
    using blastingpy::Quantity;
    const auto ev = grid(5000.0, 1.2, 1.0 / 3.0, false);
    const auto f = fit_site_law(ev.d, ev.w, ev.v, Law::CubeRoot, 2.0, 3.0, Quantity::Airblast);
    EXPECT_EQ(f.info.method_id, "AIR-CBRT-USBM8485");
    EXPECT_EQ(f.info.verification, Verification::Primary);
    EXPECT_NE(f.info.source.find("RI 8485"), std::string::npos);
    EXPECT_TRUE(f.warnings.empty());  // 15 eventos: sin la regla de 30 de BIVDEP

    const std::vector<double> d = {100, 200, 400, 800}, w(4, 100.0), p = {1, 2, 4, 8};
    const auto g = fit_site_law(d, w, p, Law::CubeRoot, 2.0, 3.0, Quantity::Airblast);
    ASSERT_EQ(g.warnings.size(), 1u);
    EXPECT_NE(g.warnings[0].find("overpressure does not decrease"), std::string::npos);
    EXPECT_FALSE(has(g.warnings, "PPV"));

    const auto bad = error_of([&] { fit_site_law(d, w, {1, 0, 4, 8}, Law::CubeRoot, 2.0, 3.0, Quantity::Airblast); });
    EXPECT_EQ(bad.rfind("Domain error [BlastingPy]: parameter 'airblast_pa'", 0), 0u);
    const auto len = error_of([&] { fit_site_law(d, {1, 2}, p, Law::CubeRoot, 2.0, 3.0, Quantity::Airblast); });
    EXPECT_NE(len.find("same length as airblast_pa"), std::string::npos);
    const auto few = error_of([&] { fit_site_law({1, 2}, {1, 1}, {2, 1}, Law::CubeRoot, 2.0, 3.0, Quantity::Airblast); });
    EXPECT_EQ(few.rfind("Domain error [BlastingPy]: parameter 'airblast_pa'", 0), 0u);
    EXPECT_THROW(fit_site_law(d, w, p, Law::SquareRoot, 2.0, 3.0, Quantity::Airblast), std::invalid_argument);
}

TEST(MakeSiteLaw, InfoOverride) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const blastingpy::MethodInfo info{"USBM-RI8485/quarry/0.1Hz", Verification::Primary, "RI 8485, Table B-1"};
    const auto f = make_site_law(Law::CubeRoot, 1000.0, 0.711, nan, 0.0, 2.0, info);
    EXPECT_EQ(f.info.method_id, info.method_id);
    EXPECT_EQ(f.info.verification, Verification::Primary);
    EXPECT_EQ(f.info.source, info.source);
    EXPECT_EQ(make_site_law(Law::CubeRoot, 1000.0, 0.711).info.verification, Verification::UserDefined);
}
