#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/waveform.hpp"

using blastingpy::Peak;
using blastingpy::peak_abs;
using blastingpy::pseudo_vector_sum;
using blastingpy::vector_sum_peak;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();

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
}  // namespace

TEST(PeakAbs, TakesLargestAbsoluteValueRegardlessOfSign) {
    const Peak p = peak_abs({1.0, -7.0, 3.0});
    EXPECT_DOUBLE_EQ(p.value, 7.0);
    EXPECT_EQ(p.index, 1u);
}

TEST(PeakAbs, TiesReturnFirstOccurrence) {
    EXPECT_EQ(peak_abs({2.0, -5.0, 5.0, -5.0}).index, 1u);
    EXPECT_EQ(peak_abs({0.0, 0.0}).index, 0u);
}

TEST(PeakAbs, RejectsEmptyAndNonFinite) {
    expect_domain_error([] { peak_abs({}); }, "x");
    expect_domain_error([] { peak_abs({1.0, NaN}); }, "x");
    expect_domain_error([] { peak_abs({Inf, 1.0}); }, "x");
}

TEST(VectorSumPeak, ThreeFourZeroInSameSampleGivesFive) {
    const Peak p = vector_sum_peak({0.0, 3.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 0.0});
    EXPECT_DOUBLE_EQ(p.value, 5.0);
    EXPECT_EQ(p.index, 1u);
}

TEST(VectorSumPeak, PeaksInDifferentSamplesGiveRealSumBelowPseudoSum) {
    const Peak p = vector_sum_peak({5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0});
    EXPECT_DOUBLE_EQ(p.value, 5.0);
    EXPECT_EQ(p.index, 0u);
    EXPECT_NEAR(pseudo_vector_sum(5.0, 5.0, 5.0), 8.660254, 1e-6);
}

TEST(VectorSumPeak, RejectsBadInputs) {
    expect_domain_error([] { vector_sum_peak({}, {}, {}); }, "t");
    expect_domain_error([] { vector_sum_peak({1.0, 2.0}, {1.0}, {1.0, 2.0}); }, "v");
    expect_domain_error([] { vector_sum_peak({1.0, 2.0}, {1.0, 2.0}, {1.0}); }, "l");
    expect_domain_error([] { vector_sum_peak({NaN}, {1.0}, {1.0}); }, "t");
    expect_domain_error([] { vector_sum_peak({1.0}, {Inf}, {1.0}); }, "v");
    expect_domain_error([] { vector_sum_peak({1.0}, {1.0}, {-Inf}); }, "l");
}

TEST(PseudoVectorSum, RejectsNegativeAndNonFinitePeaks) {
    expect_domain_error([] { pseudo_vector_sum(-1.0, 1.0, 1.0); }, "peak_t");
    expect_domain_error([] { pseudo_vector_sum(1.0, NaN, 1.0); }, "peak_v");
    expect_domain_error([] { pseudo_vector_sum(1.0, 1.0, Inf); }, "peak_l");
    EXPECT_DOUBLE_EQ(pseudo_vector_sum(0.0, 0.0, 0.0), 0.0);
}

TEST(VectorSumPeak, RealSumBetweenLargestComponentPeakAndPseudoSum) {
    std::mt19937 gen(12345);
    std::normal_distribution<double> dist(0.0, 3.0);
    for (int trial = 0; trial < 200; ++trial) {
        const std::size_t n = 5 + static_cast<std::size_t>(trial % 40);
        std::vector<double> t(n), v(n), l(n);
        for (std::size_t i = 0; i < n; ++i) {
            t[i] = dist(gen);
            v[i] = dist(gen);
            l[i] = dist(gen);
        }
        const double pt = peak_abs(t).value, pv = peak_abs(v).value, pl = peak_abs(l).value;
        const double real = vector_sum_peak(t, v, l).value;
        const double pseudo = pseudo_vector_sum(pt, pv, pl);
        const double biggest = std::max({pt, pv, pl});
        EXPECT_LE(real, pseudo * (1.0 + 1e-12));
        EXPECT_GE(real, biggest * (1.0 - 1e-12));
    }
}
