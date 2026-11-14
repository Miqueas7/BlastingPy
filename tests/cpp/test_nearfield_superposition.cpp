#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }
double peak(const std::vector<double>& v) {
    double p = 0.0;
    for (double x : v) p = std::max(p, std::fabs(x));
    return p;
}
const double kInf = std::numeric_limits<double>::infinity();
// Semilla triangular de 20 ms a 1 kHz (21 muestras, pico 1 en el centro).
std::vector<double> triangle() {
    std::vector<double> s(21);
    for (int i = 0; i <= 20; ++i) s[i] = 1.0 - std::fabs(i - 10) / 10.0;
    return s;
}
}  // namespace

TEST(NearfieldSuperpose, SingleImpulseReturnsSeed) {  // Hinzen (c)
    const auto s = triangle();
    const auto v = nf::superpose(s, 1000.0, {0.0}, {1.0});
    ASSERT_EQ(v.size(), s.size() + 1);
    for (std::size_t i = 0; i < s.size(); ++i) EXPECT_NEAR(v[i], s[i], 1e-12);
    EXPECT_EQ(v.back(), 0.0);
}

TEST(NearfieldSuperpose, DoublePeakAndSeparated) {  // Hinzen (a), (b)
    const auto s = triangle();
    EXPECT_NEAR(peak(nf::superpose(s, 1000.0, {0.0, 0.0}, {1.0, 1.0})), 2.0, 1e-12);
    const auto v = nf::superpose(s, 1000.0, {0.0, 0.050}, {1.0, 1.0});
    EXPECT_NEAR(peak(v), 1.0, 1e-12);
    EXPECT_EQ(v.size(), 50u + s.size() + 1);  // última llegada + semilla
}

TEST(NearfieldSuperpose, FractionalArrivalSpreadsLinearlyAndNegativeAmplitude) {
    const auto v = nf::superpose({1.0, 2.0}, 100.0, {0.025}, {-2.0});  // llegada en 2.5 muestras
    ASSERT_EQ(v.size(), 6u);                                          // ceil(2.5) + 2 + 1
    EXPECT_EQ(v[0], 0.0);
    EXPECT_EQ(v[1], 0.0);
    EXPECT_NEAR(v[2], -2.0 * 0.5 * 1.0, 1e-12);
    EXPECT_NEAR(v[3], -2.0 * (0.5 * 1.0 + 0.5 * 2.0), 1e-12);
    EXPECT_NEAR(v[4], -2.0 * 0.5 * 2.0, 1e-12);
    EXPECT_EQ(v[5], 0.0);
}

TEST(NearfieldSuperpose, Domain) {
    EXPECT_THROW(nf::superpose({}, 1000.0, {0.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(nf::superpose({1.0}, 0.0, {0.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(nf::superpose({1.0}, 1000.0, {-1e-3}, {1.0}), std::invalid_argument);
    EXPECT_THROW(nf::superpose({1.0}, 1000.0, {kInf}, {1.0}), std::invalid_argument);
    EXPECT_THROW(nf::superpose({1.0}, 1000.0, {0.0, 1.0}, {1.0}), std::invalid_argument);
    EXPECT_THROW(nf::superpose({1.0}, 1000.0, {0.0}, {std::nan("")}), std::invalid_argument);
}

TEST(NearfieldScreening, FactorBs2Bs3) {
    EXPECT_NEAR(nf::screening_factor(100.0, 1, 23337.0, 0.0), 0.300, 2e-3);  // BS2
    EXPECT_NEAR(nf::screening_factor(100.0, 4, 23337.0, 0.0), 0.176, 2e-3);
    EXPECT_NEAR(nf::screening_factor(100.0, 4), 0.2, 1e-15);  // BS3: piso por defecto
    EXPECT_NEAR(nf::screening_factor(1.0, 100), 0.2, 1e-15);
    EXPECT_EQ(nf::screening_factor(50.0, 0), 1.0);
    EXPECT_THROW(nf::screening_factor(0.0, 1), std::invalid_argument);
    EXPECT_THROW(nf::screening_factor(10.0, -1), std::invalid_argument);
    EXPECT_THROW(nf::screening_factor(10.0, 1, 23337.0, 1.5), std::invalid_argument);
}

TEST(NearfieldScreening, CountsInLine) {
    const std::vector<double> x{0, 5, 10, 15, 20}, y(5, 0.0);
    EXPECT_EQ(nf::screening_counts(x, y, {4, 3, 2, 1, 0}, 1000.0, 0.0, 5.0), (std::vector<int>{4, 3, 2, 1, 0}));
    EXPECT_EQ(nf::screening_counts(x, y, {0, 1, 2, 3, 4}, 1000.0, 0.0, 5.0), (std::vector<int>{0, 0, 0, 0, 0}));
    EXPECT_EQ(nf::screening_counts(x, y, {1, 1, 1, 1, 1}, 1000.0, 0.0, 5.0), (std::vector<int>{0, 0, 0, 0, 0}));
}

TEST(NearfieldScreening, OutsideRectangleDoesNotCount) {
    // hoyo 0 en el origen, monitor a 100 m en x; el otro hoyo inicia antes
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 2.4}, {1, 0}, 100.0, 0.0, 5.0)[0], 1);  // 2.4 <= 2.5
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 2.6}, {1, 0}, 100.0, 0.0, 5.0)[0], 0);
    EXPECT_EQ(nf::screening_counts({0, 150}, {0, 0}, {1, 0}, 100.0, 0.0, 5.0)[0], 0);  // más allá del monitor
    EXPECT_THROW(nf::screening_counts({0}, {0, 1}, {0}, 100.0, 0.0, 5.0), std::invalid_argument);
    EXPECT_THROW(nf::screening_counts({0}, {0}, {0}, 100.0, 0.0, 0.0), std::invalid_argument);
}

TEST(NearfieldWeights, FirstAndLinearLimit) {  // BN4, BN5
    const std::vector<double> w{2, 3, 5};
    const auto c = nf::charge_scaling_weights(w, {kInf, kInf, kInf}, 5.0, 0.7);
    for (int i = 0; i < 3; ++i) expect_rel(c[i], std::pow(w[i], 0.7), 1e-12);
}

TEST(NearfieldWeights, EqualChargeBn3) {
    const double A = 0.7;
    const std::vector<double> w(6, 1.0), h{kInf, 4, 4, 4, 4, 4};
    const auto c = nf::charge_scaling_weights(w, h, 4.0, A);  // r_n = 1
    for (int n = 2; n <= 6; ++n) expect_rel(c[n - 1], 0.5 * (1.0 + std::pow(n, A) - std::pow(n - 1, A)), 1e-12);
}

TEST(NearfieldWeights, FullyNonLinearSumsToTotalPower) {
    const std::vector<double> w{2, 3, 5, 4};
    const auto c = nf::charge_scaling_weights(w, {kInf, 1e-9, 1e-9, 1e-9}, 5.0, 0.7);
    expect_rel(std::accumulate(c.begin(), c.end(), 0.0), std::pow(14.0, 0.7), 1e-6);
}

TEST(NearfieldWeights, Eq10LiteralNotDifferenceOfF) {  // R8: 3.017024, no 3.632
    // w = 2, 3, 5 (media 10/3), s_P = D = 1: h = r·w/w_bar con r = inf, 0.5, 2
    const auto c = nf::charge_scaling_weights({2, 3, 5}, {kInf, 0.45, 3.0}, 1.0, 0.7);
    expect_rel(c[2], 3.017024, 1e-6);
}

TEST(NearfieldWeights, Domain) {
    EXPECT_THROW(nf::charge_scaling_weights({1, 2}, {kInf}, 5.0, 0.7), std::invalid_argument);
    EXPECT_THROW(nf::charge_scaling_weights({1, 2}, {kInf, 0.0}, 5.0, 0.7), std::invalid_argument);
    EXPECT_THROW(nf::charge_scaling_weights({1, 2}, {kInf, 5.0}, 5.0, 1.2), std::invalid_argument);
    EXPECT_THROW(nf::charge_scaling_weights({1, 0}, {kInf, 5.0}, 5.0, 0.7), std::invalid_argument);
    EXPECT_THROW(nf::charge_scaling_weights({1, 2}, {kInf, 5.0}, 5.0, 0.7, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(nf::charge_scaling_weights({1, 2}, {kInf, 5.0}, -1.0, 0.7), std::invalid_argument);
}

TEST(NearfieldDamage, Bn6AndAccumulatedDamage) {
    // monitor lejano y perpendicular a la línea de hoyos: S ~ 1
    const std::vector<double> x{0, 10, 20}, y{0, 0, 0}, w{1, 1, 1}, t{0, 1, 2};
    const auto e0 = nf::damage_weights(x, y, w, t, 10.0, 1e6, 10.0, 0.0);
    EXPECT_EQ(e0[0], 1.0);
    for (double e : e0) expect_rel(e, 1.0, 1e-3);
    const auto e = nf::damage_weights(x, y, w, t, 10.0, 1e6, 10.0, 0.2);
    expect_rel(e[1], 1.0 / 1.2, 1e-3);
    expect_rel(e[2], 1.0 / (1.0 + 0.2 * (1.0 + 0.125)), 1e-3);
}

TEST(NearfieldDamage, ScreeningReducesWeightAndOrderIsPreserved) {
    // el hoyo 1 (x = 5) inicia antes y está entre el hoyo 0 y el monitor (en x)
    const auto e = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {1, 0}, 100.0, 0.0, 5.0, 0.0);
    expect_rel(e[0], 1.0 / (1.0 + 23337.0 / 1e4), 1e-12);  // S = 1/(1 + g/d²·√(1·exp(0)))
    EXPECT_EQ(e[1], 1.0);
    // mismo tiempo: no se cuentan entre sí
    const auto same = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {0, 0}, 100.0, 0.0, 5.0, 0.2);
    EXPECT_EQ(same[0], 1.0);
    EXPECT_EQ(same[1], 1.0);
    EXPECT_THROW(nf::damage_weights({0, 0}, {0, 0}, {1, 1}, {0, 1}, 100.0, 0.0, 5.0, 0.2), std::invalid_argument);
    EXPECT_THROW(nf::damage_weights({0}, {0}, {1}, {0}, 100.0, 0.0, 5.0, -0.1), std::invalid_argument);
}

TEST(NearfieldWindow, MaxCharge) {
    const std::vector<double> q(6, 100.0);
    EXPECT_EQ(nf::max_charge_per_window({0, .005, .010, .015, .020, .025}, q), 200.0);
    EXPECT_EQ(nf::max_charge_per_window({0, .010, .020, .030, .040, .050}, q), 100.0);
    EXPECT_EQ(nf::max_charge_per_window({0.0, 0.008}, {100, 100}), 100.0);     // abierto por arriba
    EXPECT_EQ(nf::max_charge_per_window({0.0, 0.0079}, {100, 100}), 200.0);    // cerrado por abajo
    EXPECT_EQ(nf::max_charge_per_window({0.02, 0.0, 0.001}, {1, 2, 4}), 6.0);  // cualquier orden
    EXPECT_EQ(nf::max_charge_per_window({}, {}), 0.0);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {-1.0}), std::invalid_argument);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {1.0}, 0.0), std::invalid_argument);
}
