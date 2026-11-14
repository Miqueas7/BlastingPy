#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }

std::vector<double> grid(double t0, double t1, double dt) {
    std::vector<double> t;
    for (double s = t0; s <= t1; s += dt) t.push_back(s);
    return t;
}

double sum(const std::vector<double>& x) { return std::accumulate(x.begin(), x.end(), 0.0); }

const nf::Medium kRock{3000.0 * std::sqrt(3.0), 3000.0, 2700.0};
const nf::PressureSource kSource{5e9, 6, 20000.0};
const double kA = 0.05;
}  // namespace

TEST(NearfieldColumn, WeightsSumBottomPrimed) {  // BN1: ec. 2
    for (int m : {1, 21, 41}) expect_rel(sum(nf::column_weights(m, 1, 0.7)), std::pow(m, 0.7), 1e-12);
    const auto w = nf::column_weights(21, 1, 0.7);  // ec. 1: m^A − (m − 1)^A, del fondo al tope
    for (int m = 1; m <= 21; ++m) expect_rel(w[m - 1], std::pow(m, 0.7) - std::pow(m - 1, 0.7), 1e-12);
}

TEST(NearfieldColumn, WeightsPrimerInside) {  // BN2: ec. 22, M = 15, M1 = 5
    const double A = 0.7;
    const auto w = nf::column_weights(15, 5, A);
    ASSERT_EQ(w.size(), 15u);
    expect_rel(sum(w), 6.656775, 1e-6);
    expect_rel(sum(w), std::pow(15.0, A), 1e-12);
    EXPECT_EQ(w[4], 1.0);                                   // el cebo (posición 5)
    const double second = (std::pow(3.0, A) - 1.0) / 2.0;  // par j = 2: posiciones 4 y 6
    expect_rel(w[3], second, 1e-12);
    expect_rel(w[5], second, 1e-12);
    const double fifth = (std::pow(9.0, A) - std::pow(7.0, A)) / 2.0;  // par j = 5: posiciones 1 y 9
    expect_rel(w[0], fifth, 1e-12);
    expect_rel(w[8], fifth, 1e-12);
    expect_rel(w[9], std::pow(10.0, A) - std::pow(9.0, A), 1e-12);    // j = 6, posición 10
    expect_rel(w[14], std::pow(15.0, A) - std::pow(14.0, A), 1e-12);  // j = 11, el tope
    // Cebo más cerca del tope: la columna se refleja (inferido).
    auto mirrored = nf::column_weights(15, 11, A);
    std::reverse(mirrored.begin(), mirrored.end());
    EXPECT_EQ(mirrored, w);
}

TEST(NearfieldColumn, LinearSuperpositionHasUnitWeights) {
    for (int m1 : {1, 4, 8, 13, 15})
        for (double v : nf::column_weights(15, m1, 1.0)) EXPECT_EQ(v, 1.0);
}

TEST(NearfieldColumn, SingleElementEqualsHeelanElement) {
    const double half = 0.1;
    const nf::Column col{kA, -half, half, 1, 1, 5000.0, 0.7};
    const auto t = grid(0.0, 0.006, 1e-6);
    const auto c = nf::heelan_column(kSource, kRock, col, 7.0, 3.0, t);
    const auto e = nf::heelan_element(kSource, kRock, kA, half, 7.0, 3.0, t);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(c.radial_mm_s[i], e.radial_mm_s[i], 1e-12 * std::fabs(e.radial_mm_s[i]));
        EXPECT_NEAR(c.axial_mm_s[i], e.axial_mm_s[i], 1e-12 * std::fabs(e.axial_mm_s[i]));
    }
}

TEST(NearfieldColumn, FarFieldInstantLinearColumnIsOneElement) {
    // VOD muy grande y A = 1: lejos (r = 200·L, de costado) la columna equivale a un elemento de media longitud L.
    const double L = 1.0, r = 200.0 * L, g = kSource.gamma_per_s;
    const nf::Column col{kA, -L, L, 21, 1, 1e12, 1.0};
    const auto t = grid(r / kRock.p_velocity_m_s - 1e-4, std::hypot(r, L) / kRock.s_velocity_m_s + 30 / g, 0.01 / g);
    const double whole = nf::vector_peak(nf::heelan_element(kSource, kRock, kA, L, r, 0.0, t));
    expect_rel(nf::vector_peak(nf::heelan_column(kSource, kRock, col, r, 0.0, t)), whole, 1e-2);
}

TEST(NearfieldColumn, NonLinearSuperpositionLowersThePeak) {
    const double g = kSource.gamma_per_s, r = 1.0, z = 1.0;
    const auto t = grid(0.0, std::hypot(r, 2.0) / kRock.s_velocity_m_s + 2.0 / 5000.0 + 30 / g, 0.01 / g);
    const nf::Column linear{kA, 0.0, 2.0, 21, 1, 5000.0, 1.0};
    nf::Column scaled = linear;
    scaled.charge_exponent = 0.7;
    EXPECT_LT(nf::vector_peak(nf::heelan_column(kSource, kRock, scaled, r, z, t)),
              nf::vector_peak(nf::heelan_column(kSource, kRock, linear, r, z, t)));
}

TEST(NearfieldColumn, BottomPrimedRadiationIsNotSymmetric) {
    // Con cebado en el fondo la SV se dirige (Blair 2007 p. E33): por encima y por debajo el pico difiere.
    const double g = kSource.gamma_per_s, r = 2.0, vod = 5000.0;
    nf::Column col{kA, 0.0, 2.0, 21, 1, vod, 1.0};  // A = 1: la asimetría viene solo de la VOD
    const auto t = grid(0.0, std::hypot(r, 7.0) / kRock.s_velocity_m_s + 2.0 / vod + 30 / g, 0.01 / g);
    const double below = nf::vector_peak(nf::heelan_column(kSource, kRock, col, r, col.z_bottom - 5.0, t));
    const double above = nf::vector_peak(nf::heelan_column(kSource, kRock, col, r, col.z_top + 5.0, t));
    EXPECT_GT(std::fabs(above - below), 0.05 * std::max(above, below));
    col.vod_m_s = 1e12;  // iniciación casi instantánea: simétrica (los retardos de ~1e-12 s dejan ~1e-8 relativo)
    expect_rel(nf::vector_peak(nf::heelan_column(kSource, kRock, col, r, col.z_top + 5.0, t)),
               nf::vector_peak(nf::heelan_column(kSource, kRock, col, r, col.z_bottom - 5.0, t)), 1e-6);
}

TEST(NearfieldColumn, VectorPeakOfKnownWave) {
    EXPECT_EQ(nf::vector_peak({{0.0, 3.0, -1.0}, {0.0, -4.0, 1.0}}), 5.0);
    EXPECT_THROW(nf::vector_peak({{1.0, 2.0}, {1.0}}), std::invalid_argument);
}

TEST(NearfieldColumn, Domain) {
    EXPECT_THROW(nf::column_weights(2, 1, 0.7), std::invalid_argument);   // M par
    EXPECT_THROW(nf::column_weights(0, 1, 0.7), std::invalid_argument);
    EXPECT_THROW(nf::column_weights(15, 0, 0.7), std::invalid_argument);  // M1 fuera de rango
    EXPECT_THROW(nf::column_weights(15, 16, 0.7), std::invalid_argument);
    EXPECT_THROW(nf::column_weights(15, 5, 1.1), std::invalid_argument);  // A > 1
    EXPECT_THROW(nf::column_weights(15, 5, 0.0), std::invalid_argument);
    const std::vector<double> t{0.0, 1e-3};
    const nf::Column ok{kA, 0.0, 2.0, 21, 1, 5000.0, 0.7};
    nf::Column bad = ok;
    bad.vod_m_s = 0.0;
    EXPECT_THROW(nf::heelan_column(kSource, kRock, bad, 1.0, 1.0, t), std::invalid_argument);
    bad = ok;
    bad.z_top = 0.0;
    EXPECT_THROW(nf::heelan_column(kSource, kRock, bad, 1.0, 1.0, t), std::invalid_argument);
    bad = ok;
    bad.elements = 20;
    EXPECT_THROW(nf::heelan_column(kSource, kRock, bad, 1.0, 1.0, t), std::invalid_argument);
    bad = ok;
    bad.hole_radius_m = -1.0;
    EXPECT_THROW(nf::heelan_column(kSource, kRock, bad, 1.0, 1.0, t), std::invalid_argument);
    EXPECT_THROW(nf::heelan_column(kSource, kRock, ok, 0.0, 1.0, t), std::invalid_argument);
}
