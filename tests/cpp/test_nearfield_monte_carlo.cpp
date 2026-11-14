#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
double peak(const std::vector<double>& v) {
    double p = 0.0;
    for (double x : v) p = std::max(p, std::fabs(x));
    return p;
}
double mean(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
// Semilla triangular (unipolar) de 20 ms a 1 kHz: 21 muestras, pico 1 en el centro.
std::vector<double> triangle() {
    std::vector<double> s(21);
    for (int i = 0; i <= 20; ++i) s[i] = 1.0 - std::fabs(i - 10) / 10.0;
    return s;
}
const std::vector<double> kSeed{0.0, 0.6, 1.0, -0.4, 0.1};  // 5 muestras, bipolar
const std::vector<double> kArrivals{0.0, 0.002, 0.005};
const std::vector<double> kAmplitudes{1.0, 0.8, -0.5};  // el tercero con polaridad invertida
}  // namespace

TEST(NearfieldMonteCarlo, NoScatterEqualsSuperpose) {  // Review Focus 5
    const double expected = peak(nf::superpose(kSeed, 1000.0, kArrivals, kAmplitudes));
    const auto peaks = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, {0.0, 0.0, 0.0}, 0.0, 50, 99);
    ASSERT_EQ(peaks.size(), 50u);
    for (double p : peaks) EXPECT_NEAR(p, expected, 1e-12 * expected);
}

TEST(NearfieldMonteCarlo, SameSeedSameVectorOtherSeedOtherVector) {
    const std::vector<double> sigma(3, 1e-3);
    const auto a = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, sigma, 0.25, 200, 2026);
    const auto b = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, sigma, 0.25, 200, 2026);
    const auto c = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, sigma, 0.25, 200, 2027);
    EXPECT_EQ(a, b);  // misma máquina y misma semilla: idénticos bit a bit
    EXPECT_NE(a, c);
}

// Candado de portabilidad: los cuatro picos salen de una referencia aparte en Python puro (mt19937_64 escrito a mano y
// validado con el valor de control del estándar, más Box-Muller y la superposición). Se comparan a rel 1e-9 y no bit a
// bit: log, cos y sin no redondean igual en todas las bibliotecas (solo sqrt es exacta), así que los últimos bits
// pueden diferir entre MSVC, GCC y Clang.
TEST(NearfieldMonteCarlo, PinnedPeaksPortabilityLock) {
    const auto peaks = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, {1e-3, 1e-3, 1e-3}, 0.25, 4, 12345);
    const std::vector<double> expected{1.5646578308494763, 0.9396280591874882, 0.984018375475602, 1.391979964537508};
    ASSERT_EQ(peaks.size(), expected.size());
    for (std::size_t p = 0; p < expected.size(); ++p) EXPECT_NEAR(peaks[p], expected[p], 1e-9 * expected[p]);
}

TEST(NearfieldMonteCarlo, SingleHoleAmplitudeStatistics) {  // w·(1 + c·z): media w·pico y desviación relativa c
    const auto peaks = nf::monte_carlo_peaks(triangle(), 1000.0, {0.0}, {3.0}, {0.0}, 0.25, 20000, 7);
    const double m = mean(peaks);
    double ss = 0.0;
    for (double p : peaks) ss += (p - m) * (p - m);
    const double sd = std::sqrt(ss / (peaks.size() - 1));
    EXPECT_NEAR(m, 3.0, 0.01 * 3.0);
    EXPECT_NEAR(sd / m, 0.25, 0.03 * 0.25);
}

TEST(NearfieldMonteCarlo, DelayScatterRaisesTheMeanPeak) {  // R6: semilla unipolar, algunas llegadas se montan
    const auto s = triangle();
    const std::vector<double> arrivals{0.0, 0.020};  // retardo nominal = duración de la semilla: no se montan
    const double deterministic = peak(nf::superpose(s, 1000.0, arrivals, {1.0, 1.0}));
    EXPECT_NEAR(deterministic, 1.0, 1e-12);
    // El primer pozo nominal en t = 0 con σ > 0 saca tiempos negativos: se recortan a 0 (sin recorte, superpose lanza).
    const auto peaks = nf::monte_carlo_peaks(s, 1000.0, arrivals, {1.0, 1.0}, {0.010, 0.010}, 0.0, 2000, 7);
    EXPECT_GT(mean(peaks), 1.05 * deterministic);
}

TEST(NearfieldMonteCarlo, DelaySigmaDoesNotShiftTheAmplitudeStream) {
    // Semilla plana: el reparto de una llegada entre muestras no cambia el pico, así que solo queda la amplitud. Si las
    // dos normales no se sacaran siempre, cambiar σ correría la secuencia y los picos serían otros.
    const std::vector<double> flat{1.0, 1.0, 1.0};
    const auto still = nf::monte_carlo_peaks(flat, 1000.0, {0.050}, {2.0}, {0.0}, 0.25, 100, 5);
    const auto moved = nf::monte_carlo_peaks(flat, 1000.0, {0.050}, {2.0}, {0.004}, 0.25, 100, 5);
    for (std::size_t p = 0; p < still.size(); ++p) EXPECT_NEAR(moved[p], still[p], 1e-12 * 2.0);
}

TEST(NearfieldMonteCarlo, NegativeAmplitudeFactorIsClippedToZero) {
    // c = 10: el factor 1 + 10·z sale negativo con probabilidad Φ(−0,1) ≈ 0,46; recortado da pico exactamente 0
    // (sin recorte, |v| lo volvería positivo y no habría ceros).
    const auto peaks = nf::monte_carlo_peaks(triangle(), 1000.0, {0.0}, {-1.0}, {0.0}, 10.0, 4000, 11);
    const double zeros = static_cast<double>(std::count(peaks.begin(), peaks.end(), 0.0)) / peaks.size();
    EXPECT_NEAR(zeros, 0.4602, 0.03);
}

TEST(NearfieldMonteCarlo, Domain) {
    const std::vector<double> s{0.0, 1.0, 0.0};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0, 0.01}, {1.0, 1.0}, {0.0, -1e-3}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0, 0.01}, {1.0, 1.0}, {0.0, nan}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0, 0.01}, {1.0, 1.0}, {0.0}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0, 0.01}, {1.0}, {0.0, 0.0}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0, -0.01}, {1.0, 1.0}, {0.0, 0.0}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0}, {1.0}, {0.0}, -0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0}, {1.0}, {0.0}, 0.1, 0, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks({}, 1000.0, {0.0}, {1.0}, {0.0}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 0.0, {0.0}, {1.0}, {0.0}, 0.1, 10, 1), std::invalid_argument);
    EXPECT_THROW(nf::monte_carlo_peaks(s, 1000.0, {0.0}, {nan}, {0.0}, 0.1, 10, 1), std::invalid_argument);
}
