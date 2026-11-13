#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/waveform.hpp"

using blastingpy::DiffMethod;
using blastingpy::peak_acceleration;
using blastingpy::peak_displacement;
using blastingpy::standard_gravity_mm_s2;

namespace {
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

std::vector<double> sine(double amp, double f, double fs, std::size_t n, double phase = 0.0) {
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = amp * std::sin(2.0 * kPi * f * static_cast<double>(i) / fs + phase);
    return x;
}
}  // namespace

TEST(WaveformDerived, StandardGravityIsTheDefinedConstant) {
    EXPECT_DOUBLE_EQ(standard_gravity_mm_s2, 9806.65);
}

TEST(WaveformDerived, AccelerationOfSine) {
    const auto x = sine(10.0, 10.0, 4096.0, 4096);  // 10 ciclos completos
    const double expected = 10.0 * 2.0 * kPi * 10.0;  // 628.319 mm/s2
    EXPECT_NEAR(peak_acceleration(x, 4096.0, DiffMethod::Central).value, expected, 0.001 * expected);
    EXPECT_NEAR(peak_acceleration(x, 4096.0, DiffMethod::Forward).value, expected, 0.01 * expected);
    EXPECT_NEAR(peak_acceleration(x, 4096.0, DiffMethod::Central).value / standard_gravity_mm_s2, 0.064070, 1e-4);
}

TEST(WaveformDerived, AccelerationIndexAndEdges) {
    const std::vector<double> x{0.0, 1.0, 5.0, 5.0, 4.0};
    const auto fwd = peak_acceleration(x, 10.0, DiffMethod::Forward);
    EXPECT_DOUBLE_EQ(fwd.value, 40.0);  // (5-1)*10
    EXPECT_EQ(fwd.index, 1u);
    const auto cen = peak_acceleration(x, 10.0, DiffMethod::Central);
    EXPECT_DOUBLE_EQ(cen.value, 25.0);  // (5-0)*10/2 en i=1; i=2 da 20
    EXPECT_EQ(cen.index, 1u);
    // Dos muestras: Central cae a la diferencia unilateral.
    EXPECT_DOUBLE_EQ(peak_acceleration({1.0, 3.0}, 2.0, DiffMethod::Central).value, 4.0);
    EXPECT_DOUBLE_EQ(peak_acceleration({1.0, 3.0}, 2.0, DiffMethod::Forward).value, 4.0);
}

TEST(WaveformDerived, DisplacementOfSine) {
    const double expected = 10.0 / (2.0 * kPi * 10.0);  // 0.159155 mm
    const auto x = sine(10.0, 10.0, 4096.0, 4096, 0.7);  // fase arbitraria, ciclos cortados en los bordes
    const auto d = peak_displacement(x, 4096.0);
    EXPECT_NEAR(d.value, expected, 0.005 * expected);
    EXPECT_NEAR(std::fabs(x[d.index]), 10.0, 0.05);  // índice en el pico de velocidad del medio ciclo
}

TEST(WaveformDerived, DisplacementUsesRectangleSumOverHalfCycleSamples) {
    // Un solo medio ciclo completo: {+1,+3,+2} entre ceros; Σ(V·dt) = 6·0.5 = 3; /2 = 1.5; pico en índice 2.
    const auto d = peak_displacement({-1.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0}, 2.0);
    EXPECT_DOUBLE_EQ(d.value, 1.5);
    EXPECT_EQ(d.index, 3u);
}

TEST(WaveformDerived, DisplacementDropsHalfCyclesCutByTheEdges) {
    // La racha del inicio y la del final están cortadas; la del medio (negativa) es la única completa.
    const auto d = peak_displacement({5.0, 5.0, -2.0, -4.0, 5.0, 5.0}, 1.0);
    EXPECT_DOUBLE_EQ(d.value, 3.0);  // |-6| / 2
    EXPECT_EQ(d.index, 3u);
}

TEST(WaveformDerived, DisplacementWithoutCompleteHalfCycleIsNaN) {
    EXPECT_TRUE(std::isnan(peak_displacement({1.0, 2.0, 3.0, 2.0}, 100.0).value));
    EXPECT_TRUE(std::isnan(peak_displacement({0.0, 0.0, 0.0}, 100.0).value));
    // Menos de medio ciclo de seno: arranca en 0 (la racha sube y el registro termina antes del cruce).
    EXPECT_TRUE(std::isnan(peak_displacement(sine(10.0, 10.0, 4096.0, 100), 4096.0).value));
}

TEST(WaveformDerived, DomainErrors) {
    expect_domain_error([] { peak_acceleration({1.0}, 100.0, DiffMethod::Forward); }, "velocity_mm_s");
    expect_domain_error([] { peak_displacement({}, 100.0); }, "velocity_mm_s");
    expect_domain_error([] { peak_displacement({1.0}, 100.0); }, "velocity_mm_s");
    expect_domain_error([] { peak_displacement({1.0, std::nan("")}, 100.0); }, "velocity_mm_s");
    expect_domain_error([] { peak_acceleration({1.0, 2.0}, 0.0, DiffMethod::Central); }, "sample_rate_hz");
    expect_domain_error([] { peak_displacement({1.0, 2.0}, -5.0); }, "sample_rate_hz");
}
