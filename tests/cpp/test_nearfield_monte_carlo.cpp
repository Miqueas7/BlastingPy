#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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

// Candado de portabilidad: los cuatro picos salen de una referencia aparte en Python puro (mt19937_64 escrito desde la
// recurrencia del estándar y validado con su valor de control, más Box-Muller y una superposición propia que no corre
// los tiempos: arma cada simulación desde la muestra de la primera llegada). En las cuatro simulaciones la primera
// llegada sale negativa, así que el candado ejerce el corrimiento de muestras enteras (sin recorte a 0). Se comparan a
// rel 1e-9 y no bit a bit: log, cos y sin no redondean igual en todas las bibliotecas (solo sqrt es exacta) y el
// compilador puede fundir a·b + c en una FMA, así que los últimos bits pueden diferir entre MSVC, GCC y Clang.
TEST(NearfieldMonteCarlo, PinnedPeaksPortabilityLock) {
    const auto peaks = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, {1e-3, 1e-3, 1e-3}, 0.25, 4, 12345);
    const std::vector<double> expected{0.9346395891593839, 0.9014958054814796, 0.7745597484280603, 1.0646254016189869};
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
    // El primer pozo nominal en t = 0 con σ > 0 saca tiempos negativos: la simulación entera se corre un número entero
    // de muestras, sin recortar ninguno a 0.
    const auto peaks = nf::monte_carlo_peaks(s, 1000.0, arrivals, {1.0, 1.0}, {0.010, 0.010}, 0.0, 2000, 7);
    EXPECT_GT(mean(peaks), 1.05 * deterministic);
    // Sin sesgo por el origen del tiempo: la misma voladura 1 s más tarde (ninguna llegada negativa) da la misma media.
    const auto later = nf::monte_carlo_peaks(s, 1000.0, {1.0, 1.020}, {1.0, 1.0}, {0.010, 0.010}, 0.0, 2000, 7);
    EXPECT_NEAR(mean(peaks), mean(later), 1e-9 * mean(later));
}

TEST(NearfieldMonteCarlo, TimeOriginDoesNotChangeThePeaks) {  // R18: los tiempos no se recortan a 0
    const std::vector<double> sigma(3, 1e-3), later{1.0, 1.002, 1.005};  // kArrivals + 1 s (1000 muestras enteras)
    const auto at_zero = nf::monte_carlo_peaks(kSeed, 1000.0, kArrivals, kAmplitudes, sigma, 0.25, 500, 12345);
    const auto shifted = nf::monte_carlo_peaks(kSeed, 1000.0, later, kAmplitudes, sigma, 0.25, 500, 12345);
    ASSERT_EQ(at_zero.size(), shifted.size());
    for (std::size_t p = 0; p < shifted.size(); ++p) EXPECT_NEAR(at_zero[p], shifted[p], 1e-9 * shifted[p]) << p;
}

TEST(NearfieldMonteCarlo, ZeroScatterDoesNotShiftTheStreamEitherWay) {
    // Un pozo y semilla {1}: el pico es máx(frac, 1 − frac)·(1 + c·z'), el reparto de la llegada entre dos muestras por
    // el factor de amplitud. Con la misma rng_seed, anular c o σ no debe correr los sorteos del otro.
    const std::vector<double> one{1.0};
    const auto both = nf::monte_carlo_peaks(one, 1000.0, {0.050}, {1.0}, {0.004}, 0.25, 200, 5);
    const auto only_delay = nf::monte_carlo_peaks(one, 1000.0, {0.050}, {1.0}, {0.004}, 0.0, 200, 5);
    const auto only_amplitude = nf::monte_carlo_peaks(one, 1000.0, {0.050}, {1.0}, {0.0}, 0.25, 200, 5);
    for (std::size_t p = 0; p < both.size(); ++p)
        EXPECT_NEAR(both[p], only_delay[p] * only_amplitude[p], 1e-12 * both[p]) << p;
    EXPECT_NE(only_delay[0], 1.0);      // de verdad hay dispersión en el retardo
    EXPECT_NE(only_amplitude[0], 1.0);  // y en la amplitud
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

TEST(NearfieldMonteCarlo, OutputLengthIsBoundedBeforeSimulating) {  // nominal + 10·σ, más de 1e8 muestras
    const std::vector<double> s{0.0, 1.0, 0.0};
    expect_domain([&] { nf::monte_carlo_peaks(s, 1000.0, {1e9}, {1.0}, {0.0}, 0.0, 1, 1); },
                  "parameter 'nominal_arrival_times_s'");
    expect_domain([&] { nf::monte_carlo_peaks(s, 1000.0, {0.0}, {1.0}, {1e9}, 0.0, 1, 1); },
                  "parameter 'nominal_arrival_times_s'");
    expect_domain([&] { nf::monte_carlo_peaks(s, 1000.0, {}, {}, {}, 0.0, 1, 1); },
                  "parameter 'nominal_arrival_times_s' must not be empty");
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
