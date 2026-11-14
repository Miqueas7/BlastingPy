#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "blastingpy/flyrock.hpp"

namespace fr = blastingpy::flyrock;

namespace {
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
}  // namespace

TEST(FlyrockBallistics, MaxRangeFeetExamples) {  // E23
    const double v2[] = {3720, 2019, 9377, 30115};
    const double expected[] = {115, 63, 291, 935};
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(fr::max_range_level(std::sqrt(v2[i]), 32.174), expected[i], 0.01 * expected[i]);
}

TEST(FlyrockBallistics, VelocityForRangeAndMaxHeight) {  // E11, E12
    EXPECT_NEAR(fr::velocity_for_range(95, 45, 9.8), 30.5, 0.002 * 30.5);
    EXPECT_NEAR(fr::velocity_for_range(95, 75, 9.8), 43.1, 0.002 * 43.1);
    EXPECT_NEAR(fr::max_height(30.51, 45, 9.8), 23.7, 0.01 * 23.7);
}

TEST(FlyrockBallistics, RangeToHeight) {  // E13, E24
    EXPECT_NEAR(fr::range_to_height(22.14, 45, 10, 9.8), 36, 0.36);
    EXPECT_NEAR(fr::range_to_height(31.30, 75, 10), 47.1, 0.005 * 47.1);
    EXPECT_NEAR(fr::range_to_height(31.30, 75, 20), 43.8, 0.005 * 43.8);
    EXPECT_NEAR(fr::range_to_height(std::sqrt(63 * 32.174), 45, -30, 32.174), 85, 0.85);
    EXPECT_NEAR(fr::range_to_height(std::sqrt(65 * 32.174), 45, -69, 32.174), 107, 1.07);
}

TEST(FlyrockBallistics, Properties) {
    const double level = fr::range_level(40, 30);
    EXPECT_NEAR(fr::range_to_height(40, 30, 0), level, level * 1e-12);  // dos cálculos distintos: tolerancia relativa
    double best = 0;
    int best_angle = 0;
    for (int a = 1; a < 90; ++a)
        if (fr::range_level(40, a) > best) best = fr::range_level(40, a), best_angle = a;
    EXPECT_EQ(best_angle, 45);
    EXPECT_NEAR(fr::range_level(40, 45), fr::max_range_level(40), 1e-12);
}

TEST(FlyrockRM, FaceBurstAndCratering) {  // E1, E2, E3, E7, E8
    EXPECT_NEAR(fr::rm_face_burst(27, 11, 2.6, 9.8), 140, 1.4);
    EXPECT_NEAR(fr::rm_face_burst(27, 8.1, 1.43, 9.8), 440, 8.8);
    EXPECT_NEAR(fr::rm_cratering(13.5, 21.4, 1.6, 9.8), 295, 2.95);
    EXPECT_NEAR(fr::rm_cratering(22.3, 24.6, 5.0, 9.8), 49.7, 0.497);
    EXPECT_NEAR(fr::rm_cratering(17.2, 24.6, 4.1, 9.8), 49.5, 0.495);
}

TEST(FlyrockRM, StemmingEjection) {  // E4, E5
    const double th[] = {5, 10, 15, 20, 25, 30, 40, 45};
    const double expected[] = {17, 33, 48, 62, 74, 84, 95, 96.5};
    for (int i = 0; i < 8; ++i) EXPECT_NEAR(fr::rm_stemming_ejection(27, 11, 3.0, th[i], 9.8), expected[i], 0.5);
    EXPECT_DOUBLE_EQ(fr::rm_stemming_ejection(27, 11, 3.0, 0, 9.8), 0.0);
}

TEST(FlyrockRM, MinConfinement) {  // E6, E9, E10
    EXPECT_NEAR(fr::rm_min_confinement(13.5, 38.8, 50, 9.8), 4.3, 0.05);
    EXPECT_NEAR(fr::rm_min_confinement(17.2, 24.6, 95, 9.8), 3.2, 0.05);
    EXPECT_NEAR(fr::rm_min_confinement(22.3, 24.6, 80, 9.8), 4.2, 0.05);
    const double m[] = {24.6, 19.7, 14.8, 12.3, 9.8, 7.4};
    const double expected[] = {4.6, 4.1, 3.6, 3.2, 2.9, 2.5};
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(fr::rm_min_confinement(20, m[i], 50, 9.8), expected[i], 0.05);
}

TEST(FlyrockRM, Consistency) {
    const double l = fr::rm_face_burst(20, 15, 3.3);
    EXPECT_NEAR(fr::rm_min_confinement(20, 15, l), 3.3, 3.3e-12);
    EXPECT_NEAR(l, fr::max_range_level(fr::rm_launch_velocity(20, 15, 3.3)), l * 1e-12);
}

TEST(FlyrockDomain, Errors) {
    EXPECT_THROW(fr::range_level(0, 45), std::invalid_argument);
    EXPECT_THROW(fr::range_level(-1, 45), std::invalid_argument);
    EXPECT_THROW(fr::range_level(kNan, 45), std::invalid_argument);
    EXPECT_THROW(fr::range_level(10, kInf), std::invalid_argument);
    EXPECT_THROW(fr::range_level(10, 0), std::invalid_argument);
    EXPECT_THROW(fr::range_level(10, 90), std::invalid_argument);
    EXPECT_THROW(fr::range_level(10, 45, 0), std::invalid_argument);
    EXPECT_THROW(fr::velocity_for_range(95, 90), std::invalid_argument);
    EXPECT_THROW(fr::velocity_for_range(0, 45), std::invalid_argument);
    EXPECT_THROW(fr::range_to_height(10, 45, 100), std::invalid_argument);  // unreachable height
    EXPECT_THROW(fr::range_to_height(10, 45, kNan), std::invalid_argument);
    EXPECT_THROW(fr::rm_launch_velocity(0, 10, 2), std::invalid_argument);
    EXPECT_THROW(fr::rm_face_burst(27, 0, 2), std::invalid_argument);
    EXPECT_THROW(fr::rm_cratering(27, 10, -2), std::invalid_argument);
    EXPECT_THROW(fr::rm_min_confinement(27, 10, 0), std::invalid_argument);
    EXPECT_THROW(fr::rm_stemming_ejection(27, 11, 3, 90), std::invalid_argument);
    EXPECT_THROW(fr::rm_stemming_ejection(27, 11, 3, -1), std::invalid_argument);
    for (double bad : {-1.0, 90.0, kNan}) {  // un solo chequeo, un solo mensaje
        try {
            fr::rm_stemming_ejection(27, 11, 3, bad);
            FAIL() << bad;
        } catch (const std::invalid_argument& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("'hole_angle_from_vertical_deg'"), std::string::npos) << what;
            if (!std::isnan(bad)) EXPECT_NE(what.find("[0, 90)"), std::string::npos) << what;
        }
    }
    try {
        fr::rm_face_burst(27, 11, -1);
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Domain error [BlastingPy]: parameter 'burden_m'"), std::string::npos);
    }
}
