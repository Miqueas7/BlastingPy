#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "blastingpy/nearfield.hpp"

namespace nf = blastingpy::nearfield;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }
}  // namespace

TEST(NearfieldHp, MeanDistanceNiosh) {  // HP1, HP3
    expect_rel(nf::hp_mean_distance(-1.5, 1.5, 2, 0), 2.174, 1e-4);
    expect_rel(nf::hp_mean_distance(-1.5, 1.5, 0, 0), 0.75, 1e-12);
    EXPECT_GE(nf::hp_mean_distance(-1.5, 1.5, 2, 0), 2.0);
}

TEST(NearfieldHp, NioshModified) {  // HP2, HP4
    expect_rel(nf::hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, 2, 0), 471, 1e-3);
    expect_rel(nf::hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, 0.5, 0), 1650, 0.01);
}

TEST(NearfieldHp, OriginalIntegral) {  // HP5
    expect_rel(nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 0.5, 0), 2230, 0.02);
}

TEST(NearfieldHp, ArctanTerm) {  // HP11-HP13
    EXPECT_NEAR(std::log(nf::hp_term(4.98, 0, 29, 17.3, 14.5)), -0.9129, 1e-3);
    EXPECT_NEAR(std::log(nf::hp_term(4.98, 0, 29, 0.047, 14.5)), 5.805081, 1e-3);
    EXPECT_NEAR(std::log(nf::hp_term(10.62, 0, 0.9, 2.04, 0.45)), 0.815842, 1e-3);
    EXPECT_NEAR(std::log(nf::hp_term(10.62, 0, 0.9, 0.092, 0.45)), 5.75628, 1e-3);
}

TEST(NearfieldHp, DamageExtent) {  // HP7, HP18
    const double q[3] = {1.62, 1.27, 1.00}, want[3] = {1.8, 1.1, 0.5};
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(nf::hp_damage_extent(nf::HpModel::Niosh, 171, 1.4, 1, q[i], -1.85, 1.85, 0, 1000), want[i], 0.06);
    EXPECT_NEAR(nf::hp_damage_extent(nf::HpModel::Original, 456, 1.12, 2.24, 4.98, 0, 29, 14.5, 848), 6.5, 0.1);
}

TEST(NearfieldHp, DamageExtentNanWhenLimitUnreachable) {
    // Niosh is finite on the axis: a limit above it has no solution.
    EXPECT_TRUE(std::isnan(nf::hp_damage_extent(nf::HpModel::Niosh, 171, 1.4, 1, 0.01, -1.85, 1.85, 0, 1e9)));
}

TEST(NearfieldHp, ArctanEqualsQuadrature) {  // HP20
    const double a = 0.9, b = 1.8;
    for (double r : {0.3, 2.0, 15.0}) {
        const double exact = nf::hp_original(300, a, b, 2.5, 0, 12, r, 4);
        const double quad = nf::hp_original(300, a, b * (1 + 1e-9), 2.5, 0, 12, r, 4);
        expect_rel(quad, exact, 1e-6);
        expect_rel(exact, 300 * std::pow(nf::hp_term(2.5, 0, 12, r, 4), a), 1e-12);
    }
}

TEST(NearfieldHp, NioshAndOriginalAgree) {
    const double n = nf::hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, 2, 0);
    const double o = nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 2, 0);
    EXPECT_LT(std::fabs(n - o) / o, 0.03);
    const double far_pt = nf::hp_point(700, 0.7, 1.5, 3.0, 500);
    expect_rel(nf::hp_niosh(700, 0.7, 1.5, 1, -1.5, 1.5, 500, 0), far_pt, 1e-3);
    expect_rel(nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 500, 0), far_pt, 1e-3);
}

TEST(NearfieldHp, CriticalPpv) {  // PC1-PC4
    expect_rel(nf::ppv_critical(4.7e6, 3716, 9.6e9), 1819, 1e-3);
    expect_rel(nf::ppv_critical(11.7e6, 5800, 80e9), 848, 1e-3);
    expect_rel(nf::ppv_critical(15.8e6, 5180, 67e9), 1221, 1e-3);
    expect_rel(nf::ppv_critical(19.3e6, 6103, 87e9), 1353, 1e-3);
}

TEST(NearfieldHp, AxisWithBetaBelowAlphaIsFinite) {  // R13, condition derived here
    const double k = 100, a = 1.0, b = 0.5, q = 2.0;
    const double e = 1 - b / a;
    const double analytic = k * q * (std::pow(3.0, e) + std::pow(1.0, e)) / e;  // z = 1 in [0, 4]
    expect_rel(nf::hp_original(k, a, b, q, 0, 4, 0, 1), analytic, 1e-12);
    expect_rel(nf::hp_original(k, a, b, q, 0, 4, 1e-6, 1), analytic, 1e-3);
    EXPECT_TRUE(std::isnan(nf::hp_damage_extent(nf::HpModel::Original, k, a, b, q, 0, 4, 1, 1e9)));
    EXPECT_THROW(nf::hp_original(k, 0.5, 1.0, q, 0, 4, 0, 1), std::invalid_argument);  // beta >= alpha
}

TEST(NearfieldHp, Domain) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // on the axis inside the charged span: the original model diverges
    EXPECT_THROW(nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 0, 0), std::invalid_argument);
    EXPECT_NO_THROW(nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, 0, 3));  // r = 0 outside the span
    EXPECT_THROW(nf::hp_original(700, 0.7, 1.5, 1, 1.5, -1.5, 1, 0), std::invalid_argument);
    EXPECT_THROW(nf::hp_original(700, 0.7, 1.5, 1, -1.5, 1.5, -1, 0), std::invalid_argument);
    EXPECT_THROW(nf::hp_original(0, 0.7, 1.5, 1, -1.5, 1.5, 1, 0), std::invalid_argument);
    EXPECT_THROW(nf::hp_niosh(700, nan, 1.5, 1, -1.5, 1.5, 1, 0), std::invalid_argument);
    EXPECT_THROW(nf::hp_point(700, 0.7, 1.5, 0, 5), std::invalid_argument);
    EXPECT_THROW(nf::hp_term(1, 0, 1, 0, 0.5), std::invalid_argument);
    EXPECT_THROW(nf::ppv_critical(0, 3000, 1e10), std::invalid_argument);
    EXPECT_THROW(nf::hp_damage_extent(nf::HpModel::Niosh, 171, 1.4, 1, 1, -1, 1, 0, -5), std::invalid_argument);
}
