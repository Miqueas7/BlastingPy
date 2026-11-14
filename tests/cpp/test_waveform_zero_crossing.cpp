#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/waveform.hpp"

using blastingpy::instantel_zc_limit_hz;
using blastingpy::zero_crossing_frequency;
using blastingpy::ZcFrequency;
using blastingpy::ZcMethod;
using blastingpy::ZcStatus;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();
const double kPi = 3.14159265358979323846;

template <class F>
void expect_domain_error(F&& f, const std::string& name) {
    try {
        f();
        FAIL() << "expected std::invalid_argument for parameter " << name;
    } catch (const std::invalid_argument& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.rfind("Domain error [BlastingPy]: parameter '" + name + "'", 0), 0u) << msg;
    }
}

// x = 10 sin(2 pi f t + phi) con la fase tal que el pico queda exactamente en la muestra n/2 (medio del registro).
std::vector<double> sine_with_mid_peak(double f, double fs, std::size_t n = 2048) {
    const double t_mid = static_cast<double>(n / 2) / fs;
    const double phi = kPi / 2.0 - 2.0 * kPi * f * t_mid;
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 10.0 * std::sin(2.0 * kPi * f * static_cast<double>(i) / fs + phi);
    return x;
}

// El pico analítico es la muestra del medio (peak_abs tomaría el primero de varios picos iguales).
ZcFrequency zc_of(const std::vector<double>& x, double fs, ZcMethod m) {
    return zero_crossing_frequency(x, fs, x.size() / 2, m);
}
}  // namespace

TEST(InstantelZcLimit, IsLinearInSampleRate) {
    EXPECT_NEAR(instantel_zc_limit_hz(1024), 100.0, 1e-12);
    EXPECT_NEAR(instantel_zc_limit_hz(2048), 200.0, 1e-12);
    EXPECT_NEAR(instantel_zc_limit_hz(4096), 400.0, 1e-12);
    EXPECT_NEAR(instantel_zc_limit_hz(512), 50.0, 1e-12);
}

TEST(InstantelZcLimit, RejectsBadSampleRate) {
    expect_domain_error([] { instantel_zc_limit_hz(0.0); }, "sample_rate_hz");
    expect_domain_error([] { instantel_zc_limit_hz(NaN); }, "sample_rate_hz");
}

TEST(ZeroCrossing, InterpolatedRecoversFrequencyWithinHalfPercent) {
    for (double f : {5.0, 20.0, 40.0, 80.0}) {
        const auto r = zc_of(sine_with_mid_peak(f, 1024.0), 1024.0, ZcMethod::Interpolated);
        EXPECT_EQ(r.status, ZcStatus::Ok) << f;
        EXPECT_NEAR(r.frequency_hz, f, 0.005 * f) << f;
        EXPECT_NEAR(r.half_period_s, 0.5 / f, 0.005 * 0.5 / f) << f;
    }
    const auto r = zc_of(sine_with_mid_peak(150.0, 4096.0), 4096.0, ZcMethod::Interpolated);
    EXPECT_EQ(r.status, ZcStatus::Ok);
    EXPECT_NEAR(r.frequency_hz, 150.0, 0.005 * 150.0);
}

TEST(ZeroCrossing, WholeSamplesStaysWithinTheManualErrorTable) {
    struct Case { double f, tol; };
    for (const Case c : {Case{40.0, 5.0}, Case{60.0, 8.0}, Case{80.0, 18.0}}) {
        const auto r = zc_of(sine_with_mid_peak(c.f, 1024.0), 1024.0, ZcMethod::WholeSamples);
        EXPECT_EQ(r.status, ZcStatus::Ok) << c.f;
        EXPECT_NEAR(r.frequency_hz, c.f, c.tol) << c.f;
    }
}

TEST(ZeroCrossing, WholeSamplesAboveLimitStillReportsFrequencyInterpolatedHasNoLimit) {
    const auto x = sine_with_mid_peak(150.0, 1024.0);
    const auto w = zc_of(x, 1024.0, ZcMethod::WholeSamples);
    EXPECT_EQ(w.status, ZcStatus::AboveLimit);
    EXPECT_GT(w.frequency_hz, 100.0);
    EXPECT_TRUE(std::isfinite(w.half_period_s));
    const auto i = zc_of(x, 1024.0, ZcMethod::Interpolated);
    EXPECT_EQ(i.status, ZcStatus::Ok);
    EXPECT_NEAR(i.frequency_hz, 150.0, 0.02 * 150.0);  // 6.8 muestras por ciclo: la interpolación lineal da ~0.7 %
}

TEST(ZeroCrossing, PeakAtRecordEdgeIsIncomplete) {
    std::vector<double> x(512);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::cos(2.0 * kPi * 10.0 * static_cast<double>(i) / 1024.0);
    for (ZcMethod m : {ZcMethod::WholeSamples, ZcMethod::Interpolated}) {
        const auto r = zero_crossing_frequency(x, 1024.0, 0, m);
        EXPECT_EQ(r.status, ZcStatus::Incomplete);
        EXPECT_TRUE(std::isnan(r.frequency_hz));
        EXPECT_TRUE(std::isnan(r.half_period_s));
    }
    // Falta el cruce siguiente: el registro se corta dentro del semiciclo.
    const std::vector<double> y = {-1.0, 0.5, 2.0, 3.0};
    EXPECT_EQ(zero_crossing_frequency(y, 1024.0, 3, ZcMethod::Interpolated).status, ZcStatus::Incomplete);
}

TEST(ZeroCrossing, ZeroAtPeakIsNotComputable) {
    const std::vector<double> x(64, 0.0);
    const auto r = zero_crossing_frequency(x, 1024.0, 10, ZcMethod::WholeSamples);
    EXPECT_EQ(r.status, ZcStatus::NotComputable);
    EXPECT_TRUE(std::isnan(r.frequency_hz));
    EXPECT_TRUE(std::isnan(r.half_period_s));
}

TEST(ZeroCrossing, NegativePeakWorksLikePositive) {
    auto x = sine_with_mid_peak(20.0, 1024.0);
    for (double& v : x) v = -v;
    const auto r = zc_of(x, 1024.0, ZcMethod::Interpolated);
    EXPECT_EQ(r.status, ZcStatus::Ok);
    EXPECT_NEAR(r.frequency_hz, 20.0, 0.1);
}

TEST(ZeroCrossing, WholeSamplesCountsSamplesOfTheSameSign) {
    // Muestras +: índices 2..11 (10 muestras) -> T½ = 10/fs -> f = fs/20
    std::vector<double> x(14, 1.0);
    x[0] = x[1] = x[12] = x[13] = -1.0;
    const auto r = zero_crossing_frequency(x, 800.0, 5, ZcMethod::WholeSamples);
    EXPECT_EQ(r.status, ZcStatus::Ok);
    EXPECT_NEAR(r.half_period_s, 10.0 / 800.0, 1e-15);
    EXPECT_NEAR(r.frequency_hz, 40.0, 1e-9);
}

TEST(ZeroCrossing, RejectsBadArguments) {
    const std::vector<double> x = {-1.0, 1.0, -1.0};
    expect_domain_error([&] { zero_crossing_frequency(x, 1024.0, 3, ZcMethod::WholeSamples); }, "peak_index");
    expect_domain_error([&] { zero_crossing_frequency(x, 0.0, 1, ZcMethod::WholeSamples); }, "sample_rate_hz");
    expect_domain_error([&] { zero_crossing_frequency(x, -5.0, 1, ZcMethod::WholeSamples); }, "sample_rate_hz");
    expect_domain_error([&] { zero_crossing_frequency(x, NaN, 1, ZcMethod::WholeSamples); }, "sample_rate_hz");
    expect_domain_error([&] { zero_crossing_frequency(x, Inf, 1, ZcMethod::WholeSamples); }, "sample_rate_hz");
    expect_domain_error([] { zero_crossing_frequency({}, 1024.0, 0, ZcMethod::WholeSamples); }, "x");
    expect_domain_error([] { zero_crossing_frequency({1.0, NaN}, 1024.0, 0, ZcMethod::WholeSamples); }, "x");
}
