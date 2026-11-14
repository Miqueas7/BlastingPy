#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }
// Error de dominio cuyo mensaje contiene `text` (el parámetro y la regla que nombra).
template <class F>
void expect_domain(F&& call, const std::string& text) {
    try {
        call();
        ADD_FAILURE() << "no exception; expected a domain error containing: " << text;
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << e.what();
    }
}
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

TEST(NearfieldSuperpose, ArrivalWithinToleranceOfASampleSnapsToIt) {  // R15
    const auto s = triangle();
    const auto noisy = nf::superpose(s, 1000.0, {9 * 0.001}, {1.0});  // 0.009000000000000001 s: 9.000000000000002 muestras
    const auto exact = nf::superpose(s, 1000.0, {0.009}, {1.0});
    EXPECT_EQ(noisy.size(), exact.size());
    EXPECT_EQ(noisy, exact);
    EXPECT_EQ(nf::superpose(s, 1000.0, {0.009 + 1e-9}, {1.0}).size(), exact.size() + 1);  // 1e-6 muestras: no se ajusta
}

TEST(NearfieldSuperpose, OutputLengthIsBounded) {  // más de 1e8 muestras: error de dominio, no reserva de memoria
    expect_domain([] { nf::superpose({1.0}, 1000.0, {1e9}, {1.0}); }, "parameter 'arrival_times_s'");
    expect_domain([] { nf::superpose({1.0}, 1e300, {1e300}, {1.0}); }, "parameter 'arrival_times_s'");  // producto infinito
    EXPECT_EQ(nf::superpose({1.0}, 1000.0, {100.0}, {1.0}).size(), 100002u);
}

TEST(NearfieldSuperpose, Domain) {
    EXPECT_THROW(nf::superpose({}, 1000.0, {0.0}, {1.0}), std::invalid_argument);
    expect_domain([] { nf::superpose({1.0}, 1000.0, {}, {}); }, "parameter 'arrival_times_s' must not be empty");
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
    // pozo 0 en el origen, monitor a 100 m en x; el otro pozo inicia antes
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 2.4}, {1, 0}, 100.0, 0.0, 5.0)[0], 1);  // 2.4 <= 2.5
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 2.6}, {1, 0}, 100.0, 0.0, 5.0)[0], 0);
    EXPECT_EQ(nf::screening_counts({0, 150}, {0, 0}, {1, 0}, 100.0, 0.0, 5.0)[0], 0);  // más allá del monitor
    EXPECT_THROW(nf::screening_counts({0}, {0}, {0}, 100.0, 0.0, 0.0), std::invalid_argument);
}

TEST(NearfieldScreening, NominalTieIsNotFiredBefore) {  // R15: 0.1 + 0.2 = 0.30000000000000004 frente a 0.3
    // pozo 0 en el origen y pozo 1 entre él y el monitor, con el mismo tiempo nominal armado por dos sumas distintas
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 0}, {0.1 + 0.2, 0.3}, 100.0, 0.0, 5.0), (std::vector<int>{0, 0}));
    EXPECT_EQ(nf::screening_counts({0, 50}, {0, 0}, {0.3, 0.3 - 2e-9}, 100.0, 0.0, 5.0)[0], 1);  // 2 ns antes: sí inició
}

TEST(NearfieldScreening, CoincidencesSayWhatCoincides) {  // R16
    // monitor sobre el pozo 1, con monitor_x = 100: lo que falla no es «> 0»
    expect_domain([] { nf::screening_counts({0, 100}, {0, 0}, {0, 1}, 100.0, 0.0, 5.0); },
                  "parameter 'monitor_x' must not coincide with a hole: the monitor is at the position of hole 1");
    expect_domain([] { nf::damage_weights({0, 100}, {0, 0}, {1, 1}, {0, 1}, 100.0, 0.0, 5.0, 0.2); },
                  "parameter 'monitor_x' must not coincide with a hole: the monitor is at the position of hole 1");
    // dos pozos en el mismo punto: en cualquier orden de iniciación, también empatados
    const std::vector<std::vector<double>> orders{{0, 1, 2}, {2, 1, 0}, {0, 0, 0}};
    for (const auto& t : orders) {
        expect_domain([&] { nf::screening_counts({7, 0, 7}, {3, 0, 3}, t, 100.0, 0.0, 5.0); },
                      "parameter 'x' must not repeat a position: holes 0 and 2");
        expect_domain([&] { nf::damage_weights({7, 0, 7}, {3, 0, 3}, {1, 1, 1}, t, 100.0, 0.0, 5.0, 0.2); },
                      "parameter 'x' must not repeat a position: holes 0 and 2");
    }
}

TEST(NearfieldScreening, EmptyAndLengthMismatchNameTheParameter) {
    expect_domain([] { nf::screening_counts({}, {}, {}, 100.0, 0.0, 5.0); }, "parameter 'x' must not be empty");
    expect_domain([] { nf::screening_counts({0}, {0, 1}, {0}, 100.0, 0.0, 5.0); },
                  "parameter 'y' must have the same length as x");
    expect_domain([] { nf::screening_counts({0}, {0}, {0, 1}, 100.0, 0.0, 5.0); },
                  "parameter 'firing_times_s' must have the same length as x");
    expect_domain([] { nf::damage_weights({0}, {0}, {1}, {0, 1}, 100.0, 0.0, 5.0, 0.2); },
                  "parameter 'firing_times_s' must have the same length as x");
    expect_domain([] { nf::damage_weights({0}, {0}, {1, 1}, {0}, 100.0, 0.0, 5.0, 0.2); },
                  "parameter 'charges_kg' must have the same length as x");
    expect_domain([] { nf::damage_weights({}, {}, {}, {}, 100.0, 0.0, 5.0, 0.2); }, "parameter 'x' must not be empty");
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
    EXPECT_THROW(nf::charge_scaling_weights({}, {}, 5.0, 0.7), std::invalid_argument);
    // «sin sentido físico» es solo de A > 1 (Blair 2008 p. 240); con A <= 0 el mensaje solo da el rango
    expect_domain([] { nf::charge_scaling_weights({1, 2}, {kInf, 5.0}, 5.0, 1.2); }, "no physical meaning");
    try {
        nf::charge_scaling_weights({1, 2}, {kInf, 5.0}, 5.0, 0.0);
        ADD_FAILURE() << "A = 0 must be a domain error";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("parameter 'charge_exponent' must be within (0, 1]"), std::string::npos) << what;
        EXPECT_EQ(what.find("physical"), std::string::npos) << what;
    }
}

TEST(NearfieldDamage, Bn6AndAccumulatedDamage) {
    // monitor lejano y perpendicular a la línea de pozos: S ~ 1
    const std::vector<double> x{0, 10, 20}, y{0, 0, 0}, w{1, 1, 1}, t{0, 1, 2};
    const auto e0 = nf::damage_weights(x, y, w, t, 10.0, 1e6, 10.0, 0.0);
    EXPECT_EQ(e0[0], 1.0);
    for (double e : e0) expect_rel(e, 1.0, 1e-3);
    const auto e = nf::damage_weights(x, y, w, t, 10.0, 1e6, 10.0, 0.2);
    expect_rel(e[1], 1.0 / 1.2, 1e-3);
    expect_rel(e[2], 1.0 / (1.0 + 0.2 * (1.0 + 0.125)), 1e-3);
}

TEST(NearfieldDamage, ScreeningReducesWeightAndOrderIsPreserved) {
    // el pozo 1 (x = 5) inicia antes y está entre el pozo 0 y el monitor (en x)
    const auto e = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {1, 0}, 100.0, 0.0, 5.0, 0.0);
    expect_rel(e[0], 1.0 / (1.0 + 23337.0 / 1e4), 1e-12);  // S = 1/(1 + g/d²·√(1·exp(0)))
    EXPECT_EQ(e[1], 1.0);
    // mismo tiempo: no se cuentan entre sí
    const auto same = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {0, 0}, 100.0, 0.0, 5.0, 0.2);
    EXPECT_EQ(same[0], 1.0);
    EXPECT_EQ(same[1], 1.0);
    EXPECT_THROW(nf::damage_weights({0, 0}, {0, 0}, {1, 1}, {0, 1}, 100.0, 0.0, 5.0, 0.2), std::invalid_argument);
    EXPECT_THROW(nf::damage_weights({0}, {0}, {1}, {0}, 100.0, 0.0, 5.0, -0.1), std::invalid_argument);
    EXPECT_THROW(nf::damage_weights({0, 5}, {0, 0}, {1, 0}, {0, 1}, 100.0, 0.0, 5.0, 0.2), std::invalid_argument);  // carga 0
}

TEST(NearfieldDamage, NominalTieIsNotFiredBefore) {  // R15: 0.1 + 0.2 = 0.30000000000000004 frente a 0.3
    const auto tie = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {0.1 + 0.2, 0.3}, 100.0, 0.0, 5.0, 0.2);
    EXPECT_EQ(tie[0], 1.0);
    EXPECT_EQ(tie[1], 1.0);
    const auto before = nf::damage_weights({0, 5}, {0, 0}, {1, 1}, {0.3, 0.3 - 2e-9}, 100.0, 0.0, 5.0, 0.2);
    EXPECT_LT(before[0], 1.0);  // 2 ns antes: sí inició
}

TEST(NearfieldDamage, UnequalChargesOffAxisScreening) {
    // w̄ = 4; el pozo 1 (6 kg) inició antes, a h = √45 del pozo 0 y a p = 3 m de la recta pozo 0 → monitor (d = 10):
    // D = 1 + 0,25·1,5·(4/√45)³ y S = 1/(1 + (50/100)·√(1,5·e^(−(3/4)²))). Valor de una cuenta aparte en Python.
    const auto e = nf::damage_weights({0, 6}, {0, 3}, {2, 6}, {1, 0}, 10.0, 0.0, 4.0, 0.25, 50.0);
    expect_rel(e[0], 0.6335136052, 1e-9);
    EXPECT_EQ(e[1], 1.0);
}

TEST(NearfieldDamage, HoleBehindDamagesButDoesNotScreen) {
    // el pozo 1 inició antes pero queda detrás del pozo 0 respecto del monitor (a·b < 0): S = 1 y solo queda D
    const auto behind = nf::damage_weights({0, -5}, {0, 0}, {1, 1}, {1, 0}, 100.0, 0.0, 5.0, 0.2);
    expect_rel(behind[0], 1.0 / 1.2, 1e-12);
    EXPECT_EQ(nf::damage_weights({0, -5}, {0, 0}, {1, 1}, {1, 0}, 100.0, 0.0, 5.0, 0.0)[0], 1.0);
    // un pozo ya iniciado MÁS ALLÁ del monitor sí apantalla aquí (a·b > 0, literal de la ec. 20), aunque
    // screening_counts no lo cuenta (su rectángulo termina en el monitor)
    const auto beyond = nf::damage_weights({0, 150}, {0, 0}, {1, 1}, {1, 0}, 100.0, 0.0, 5.0, 0.0);
    expect_rel(beyond[0], 1.0 / (1.0 + 23337.0 / 1e4), 1e-12);
    EXPECT_EQ(nf::screening_counts({0, 150}, {0, 0}, {1, 0}, 100.0, 0.0, 5.0)[0], 0);
}

TEST(NearfieldWindow, MaxCharge) {
    const std::vector<double> q(6, 100.0);
    EXPECT_EQ(nf::max_charge_per_window({0, .005, .010, .015, .020, .025}, q), 200.0);
    EXPECT_EQ(nf::max_charge_per_window({0, .010, .020, .030, .040, .050}, q), 100.0);
    EXPECT_EQ(nf::max_charge_per_window({0.0, 0.008}, {100, 100}), 100.0);     // abierto por arriba
    EXPECT_EQ(nf::max_charge_per_window({0.0, 0.0079}, {100, 100}), 200.0);    // cerrado por abajo
    EXPECT_EQ(nf::max_charge_per_window({0.02, 0.0, 0.001}, {1, 2, 4}), 6.0);  // cualquier orden
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {-1.0}), std::invalid_argument);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {1.0}, 0.0), std::invalid_argument);
    EXPECT_THROW(nf::max_charge_per_window({0.0}, {1.0}, 1e-9), std::invalid_argument);  // ventana <= tolerancia
    expect_domain([] { nf::max_charge_per_window({}, {}); }, "parameter 'firing_times_s' must not be empty");
    expect_domain([] { nf::max_charge_per_window({0.0, 0.1}, {1.0, 0.0}); },
                  "parameter 'charges_kg' must be strictly positive");
}

TEST(NearfieldWindow, UpperBoundIsOpenDespiteRoundingNoise) {  // R15: ventana [t, t + ventana − 1e-9)
    EXPECT_EQ(nf::max_charge_per_window({0.010, 0.018}, {100, 100}), 100.0);  // 0.018 − 0.010 = 0.007999999999999998
    std::vector<double> every8(40), every5(40);
    for (int k = 0; k < 40; ++k) {
        every8[k] = k * 0.008;  // 9 de las 39 diferencias vecinas quedan por debajo de 0.008
        every5[k] = k * 0.005;
    }
    const std::vector<double> q(40, 100.0);
    EXPECT_EQ(nf::max_charge_per_window(every8, q), 100.0);
    EXPECT_EQ(nf::max_charge_per_window(every5, q), 200.0);
    EXPECT_EQ(nf::max_charge_per_window({0.1 + 0.2, 0.3}, {100, 100}), 200.0);  // empate nominal: van juntos
}
