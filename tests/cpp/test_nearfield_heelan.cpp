#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
const double kPi = 3.14159265358979323846;

void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }

std::vector<double> grid(double t0, double t1, double dt) {
    std::vector<double> t;
    for (double s = t0; s <= t1; s += dt) t.push_back(s);
    return t;
}

double vector_peak(const nf::Waveform& w) {
    double best = 0.0;
    for (std::size_t i = 0; i < w.radial_mm_s.size(); ++i)
        best = std::max(best, std::hypot(w.radial_mm_s[i], w.axial_mm_s[i]));
    return best;
}

double abs_peak(const std::vector<double>& x) {
    double best = 0.0;
    for (double v : x) best = std::max(best, std::fabs(v));
    return best;
}

// Vp/Vs = sqrt(3) (sólido de Poisson), mu = rho Vs^2.
const nf::Medium kRock{3000.0 * std::sqrt(3.0), 3000.0, 2700.0};
const nf::PressureSource kSource{5e9, 6, 20000.0};
const double kA = 0.05, kL = 0.1;
}  // namespace

TEST(NearfieldHeelan, MeanFrequencyFactor) {  // HE1
    EXPECT_NEAR(nf::mean_frequency_factor(4), 0.500, 5e-4);
    EXPECT_NEAR(nf::mean_frequency_factor(6), 0.375, 5e-4);
    EXPECT_NEAR(nf::mean_frequency_factor(8), 0.3125, 5e-4);
}

TEST(NearfieldHeelan, GFunctionAndPeak) {  // HE2
    expect_rel(nf::g_peak(4), 1.8513, 1e-4);
    expect_rel(nf::g_peak(6), 21.9602, 1e-4);
    expect_rel(nf::g_peak(8), 680.17, 1e-4);  // el 680,01 impreso no se reproduce
    for (double tau : {0.5, 2.0, 5.0}) {
        const double want = std::exp(-tau) * (std::pow(tau, 6) - 12 * std::pow(tau, 5) + 30 * std::pow(tau, 4));
        expect_rel(nf::g_function(6, tau), want, 1e-12);
    }
    EXPECT_EQ(nf::g_function(6, -0.1), 0.0);
}

TEST(NearfieldHeelan, PressureFunctionPeaksAtVonNeumann) {
    const nf::PressureSource s{4e9, 6, 15000.0};
    const double t_peak = 6 / 15000.0;
    expect_rel(nf::pressure_function(s, t_peak), 4e9, 1e-12);
    for (double f : {0.5, 0.9, 0.999, 1.001, 1.1, 2.0}) EXPECT_LT(nf::pressure_function(s, f * t_peak), 4e9);
    EXPECT_EQ(nf::pressure_function(s, -1e-6), 0.0);
}

TEST(NearfieldHeelan, ScaledFrequency) {  // HE5
    const nf::Medium m{5000.0, 2900.0, 2650.0};
    const double c = nf::mean_frequency_factor(6);
    const nf::PressureSource s{5e9, 6, 0.1 * 5000.0 / (c * 0.1)};
    expect_rel(nf::scaled_frequency(s, m, 0.1), 0.1, 1e-12);
    expect_rel(c * s.gamma_per_s / (2 * kPi), 795.8, 1e-3);
}

TEST(NearfieldHeelan, AmplitudeFormula) {  // ec. 16 x 1000
    const double mu = 2700.0 * 3000.0 * 3000.0, n = 6, e = std::exp(1.0);
    const double want = 20000.0 * 20000.0 * kA * kL * 5e9 * std::pow(e / n, n) / (2 * mu * kRock.p_velocity_m_s);
    expect_rel(nf::heelan_amplitude(kSource, kRock, kA, kL), 1000.0 * want, 1e-12);
}

TEST(NearfieldHeelan, PeakAtRightAngle) {  // HE3: phi = pi/2, solo P
    const double r = 10.0, vp = kRock.p_velocity_m_s, g = kSource.gamma_per_s;
    const auto w = nf::heelan_element(kSource, kRock, kA, kL, r, 0.0, grid(r / vp - 1e-4, r / vp + 30 / g, 0.002 / g));
    const double want = nf::heelan_amplitude(kSource, kRock, kA, kL) * nf::g_peak(6) * kA / r;
    expect_rel(abs_peak(w.radial_mm_s), want, 1e-3);
    for (double v : w.axial_mm_s) EXPECT_EQ(v, 0.0);
}

TEST(NearfieldHeelan, PeakAtThreeQuarterPi) {  // HE4: phi = 3pi/4 (z = r), pico en la SV
    const double r = 10.0, R = r * std::sqrt(2.0), g = kSource.gamma_per_s;
    const double vp = kRock.p_velocity_m_s, vs = kRock.s_velocity_m_s;
    ASSERT_GT(g * R * (1 / vs - 1 / vp), 20.0);  // P y SV no se solapan
    const auto w = nf::heelan_element(kSource, kRock, kA, kL, r, r, grid(0.0, R / vs + 30 / g, 0.002 / g));
    const double want = nf::heelan_amplitude(kSource, kRock, kA, kL) * 15.528 * std::sqrt(3.0) * kA / r;
    expect_rel(vector_peak(w), want, 1e-2);
}

TEST(NearfieldHeelan, SymmetryAboutTheElement) {
    const auto t = grid(0.0, 0.01, 1e-6);
    const auto up = nf::heelan_element(kSource, kRock, kA, kL, 7.0, 4.0, t);
    const auto down = nf::heelan_element(kSource, kRock, kA, kL, 7.0, -4.0, t);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(down.radial_mm_s[i], up.radial_mm_s[i], 1e-12 * std::fabs(up.radial_mm_s[i]));
        EXPECT_NEAR(down.axial_mm_s[i], -up.axial_mm_s[i], 1e-12 * std::fabs(up.axial_mm_s[i]));
    }
}

TEST(NearfieldHeelan, ArrivalTimes) {
    const double r = 6.0, z = 6.0, R = std::hypot(r, z);
    const double vp = kRock.p_velocity_m_s, vs = kRock.s_velocity_m_s;
    const double sin_phi = r / R, cos_phi = -z / R;
    const auto t = grid(0.0, R / vs + 0.002, 1e-6);
    const auto w = nf::heelan_element(kSource, kRock, kA, kL, r, z, t);
    bool p_seen = false, sv_seen = false;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double sv = w.radial_mm_s[i] * cos_phi + w.axial_mm_s[i] * sin_phi;  // componente SV
        if (t[i] < R / vp) {
            EXPECT_EQ(w.radial_mm_s[i], 0.0);
            EXPECT_EQ(w.axial_mm_s[i], 0.0);
        } else if (w.radial_mm_s[i] != 0.0) {
            p_seen = true;
        }
        if (t[i] < R / vs) EXPECT_NEAR(sv, 0.0, 1e-9 * abs_peak(w.radial_mm_s));
        else if (std::fabs(sv) > 0) sv_seen = true;
    }
    EXPECT_TRUE(p_seen);
    EXPECT_TRUE(sv_seen);
}

TEST(NearfieldHeelan, Dimensional) {
    const auto t = grid(0.0, 0.01, 1e-6);
    const auto w1 = nf::heelan_element(kSource, kRock, kA, kL, 7.0, 7.0, t);
    const auto w2 = nf::heelan_element(kSource, kRock, kA, 2 * kL, 7.0, 7.0, t);
    for (std::size_t i = 0; i < t.size(); ++i) {
        EXPECT_NEAR(w2.radial_mm_s[i], 2 * w1.radial_mm_s[i], 1e-12 * std::fabs(w1.radial_mm_s[i]));
        EXPECT_NEAR(w2.axial_mm_s[i], 2 * w1.axial_mm_s[i], 1e-12 * std::fabs(w1.axial_mm_s[i]));
    }
    // r doble con phi fijo: la amplitud cae a la mitad (1/R); a phi = pi/2 solo hay P y el tiempo se corre r/Vp.
    const double r = 7.0, vp = kRock.p_velocity_m_s;
    std::vector<double> shifted(t);
    for (double& s : shifted) s += r / vp;
    const auto near = nf::heelan_element(kSource, kRock, kA, kL, r, 0.0, t);
    const auto far = nf::heelan_element(kSource, kRock, kA, kL, 2 * r, 0.0, shifted);
    for (std::size_t i = 0; i < t.size(); ++i)
        EXPECT_NEAR(far.radial_mm_s[i], 0.5 * near.radial_mm_s[i], 1e-9 * abs_peak(near.radial_mm_s));
    // y con phi = 3pi/4 el pico del vector también cae a la mitad
    const auto t2 = grid(0.0, 0.02, 1e-7);
    expect_rel(vector_peak(nf::heelan_element(kSource, kRock, kA, kL, 2 * r, 2 * r, t2)),
               0.5 * vector_peak(nf::heelan_element(kSource, kRock, kA, kL, r, r, t2)), 1e-3);
}

TEST(NearfieldHeelan, Domain) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> t{0.0, 1e-3};
    EXPECT_THROW(nf::g_function(1, 1.0), std::invalid_argument);
    EXPECT_THROW(nf::g_function(21, 1.0), std::invalid_argument);
    EXPECT_THROW(nf::g_function(6, nan), std::invalid_argument);
    EXPECT_THROW(nf::g_peak(1), std::invalid_argument);
    EXPECT_THROW(nf::mean_frequency_factor(21), std::invalid_argument);
    EXPECT_THROW(nf::pressure_function({0.0, 6, 1e4}, 1e-4), std::invalid_argument);
    EXPECT_THROW(nf::pressure_function({1e9, 6, -1e4}, 1e-4), std::invalid_argument);
    EXPECT_THROW(nf::pressure_function({1e9, 6, 1e4}, nan), std::invalid_argument);
    EXPECT_THROW(nf::scaled_frequency(kSource, {3000, 3000, 2700}, kA), std::invalid_argument);  // Vp = Vs
    EXPECT_THROW(nf::scaled_frequency(kSource, {5000, 3000, 0}, kA), std::invalid_argument);
    EXPECT_THROW(nf::scaled_frequency(kSource, kRock, 0.0), std::invalid_argument);
    EXPECT_THROW(nf::heelan_amplitude(kSource, kRock, kA, -1.0), std::invalid_argument);
    EXPECT_THROW(nf::heelan_element(kSource, kRock, kA, kL, 0.0, 1.0, t), std::invalid_argument);
    EXPECT_THROW(nf::heelan_element(kSource, kRock, kA, kL, 1.0, nan, t), std::invalid_argument);
    EXPECT_THROW(nf::heelan_element(kSource, kRock, kA, kL, 1.0, 1.0, {0.0, nan}), std::invalid_argument);
    EXPECT_THROW(nf::heelan_element({1e9, 0, 1e4}, kRock, kA, kL, 1.0, 1.0, t), std::invalid_argument);
}
