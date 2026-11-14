#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "blastingpy/flyrock.hpp"

namespace fr = blastingpy::flyrock;
using blastingpy::Verification;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }
}  // namespace

TEST(FlyrockLundborg, PublishedExamples) {  // E16, E17, E18
    expect_rel(fr::lundborg_max_range(127), 761, 0.005);
    expect_rel(fr::lundborg_max_range(101.6), 655.2, 0.005);
    expect_rel(fr::lundborg_optimal_fragment(127), 0.293, 0.005);
    expect_rel(fr::lundborg_range(101.6, 1.35), 658, 0.005);
    expect_rel(fr::lundborg_range(203.2, 1.11), fr::lundborg_max_range(203.2), 0.005);
}

TEST(FlyrockLundborg, PowderFactorMustExceedPointTwo) {
    EXPECT_THROW(fr::lundborg_range(100, 0.2), std::invalid_argument);
    EXPECT_THROW(fr::lundborg_range(100, 0.1), std::invalid_argument);
}

TEST(FlyrockMcKenzie, CoherenceAndArithmetic) {  // E21
    expect_rel(fr::mckenzie_max_range(0.596, 127), fr::lundborg_max_range(127), 0.03);
    expect_rel(fr::mckenzie_max_range(1.0, 100), 10 * std::pow(100.0, 0.667), 1e-12);
}

TEST(FlyrockDiameterGuard, InchesOrMetresByMistake) {
    for (double d : {5.0, 0.127, 600.0}) {
        for (int i = 0; i < 4; ++i) {
            try {
                if (i == 0) fr::lundborg_max_range(d);
                if (i == 1) fr::lundborg_optimal_fragment(d);
                if (i == 2) fr::lundborg_range(d, 1.0);
                if (i == 3) fr::mckenzie_max_range(1.0, d);
                FAIL() << "no throw, case " << i << " d=" << d;
            } catch (const std::invalid_argument& e) {
                EXPECT_NE(std::string(e.what()).find("millimetres"), std::string::npos) << e.what();
            }
        }
    }
}

TEST(FlyrockRoth, GurneyAndRatio) {
    expect_rel(fr::roth_gurney_velocity(4273, fr::Explosive::Anfo), 1880, 0.001);
    expect_rel(fr::roth_gurney_velocity(6900, fr::Explosive::Other), 2300, 1e-12);
    expect_rel(fr::roth_charge_mass_ratio(10, 2700, 3, 90), 10.0 / (2700 * 9), 1e-12);
    expect_rel(fr::roth_launch_velocity(2300, 1e-4), 23.0, 1e-12);
}

TEST(FlyrockRoth, CalibrationsZeroAtBOverA) {  // E25
    const fr::RothCalibration cal[] = {fr::RothCalibration::Granite, fr::RothCalibration::GraniteAllData,
                                       fr::RothCalibration::DolomiteLimestone, fr::RothCalibration::SandstoneCrater};
    const double cm0[] = {1.67e-4, 1.42e-4, 0.67e-4, 2.10e-4};
    const double a[] = {3.487e6, 3.66e6, 3e6, 2.266e6};
    const double b[] = {584, 518, 200, 475};
    for (int i = 0; i < 4; ++i) {
        expect_rel(b[i] / a[i], cm0[i], 0.01);
        EXPECT_THROW(fr::roth_calibrated_velocity(cal[i], b[i] / a[i]), std::invalid_argument);
        EXPECT_THROW(fr::roth_calibrated_velocity(cal[i], 0.5 * b[i] / a[i]), std::invalid_argument);
        expect_rel(fr::roth_calibrated_velocity(cal[i], 5e-4), std::sqrt(a[i] * 5e-4 - b[i]), 1e-12);
    }
    try {
        fr::roth_calibrated_velocity(fr::RothCalibration::Granite, 1e-4);
        FAIL();
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("minimum"), std::string::npos) << e.what();
    }
}

TEST(FlyrockRoth, ReferenceGurneyVelocity) {
    EXPECT_EQ(fr::roth_reference_gurney_velocity(fr::RothCalibration::Granite), 2300);
    EXPECT_TRUE(std::isnan(fr::roth_reference_gurney_velocity(fr::RothCalibration::GraniteAllData)));
    EXPECT_EQ(fr::roth_reference_gurney_velocity(fr::RothCalibration::DolomiteLimestone), 1880);
    EXPECT_EQ(fr::roth_reference_gurney_velocity(fr::RothCalibration::SandstoneCrater), 1750);
}

TEST(FlyrockInfo, EveryModel) {
    const fr::Model all[] = {fr::Model::Ballistics,     fr::Model::RichardsMooreFaceBurst,
                             fr::Model::RichardsMooreCratering, fr::Model::RichardsMooreStemmingEjection,
                             fr::Model::LundborgMaxRange, fr::Model::LundborgRange,
                             fr::Model::McKenzie2018,   fr::Model::RothGurney, fr::Model::RothCalibrated};
    const char* ids[] = {"FLY-BALLISTIC",     "FLY-RM-FACE",       "FLY-RM-CRATER",
                         "FLY-RM-STEMMING",   "FLY-LUNDBORG-MAX",  "FLY-LUNDBORG-Q",
                         "FLY-MCKENZIE-2018", "FLY-ROTH-GURNEY",   "FLY-ROTH-CAL"};
    for (int i = 0; i < 9; ++i) {
        const auto m = fr::info(all[i]);
        EXPECT_EQ(m.method_id, ids[i]);
        EXPECT_FALSE(m.source.empty());
        const bool lund = all[i] == fr::Model::LundborgMaxRange || all[i] == fr::Model::LundborgRange;
        EXPECT_EQ(m.verification, lund ? Verification::Secondary : Verification::Primary);
    }
    EXPECT_NE(fr::info(fr::Model::McKenzie2018).source.find("version 2018, coefficient 10"), std::string::npos);
}
