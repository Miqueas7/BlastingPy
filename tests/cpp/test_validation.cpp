#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "blastingpy/validation.hpp"

namespace v = blastingpy::validation;

TEST(Validation, RejectsNanAndInf) {
    EXPECT_THROW(v::require_finite(std::nan(""), "x"), std::invalid_argument);
    EXPECT_THROW(v::require_finite(INFINITY, "x"), std::invalid_argument);
    EXPECT_NO_THROW(v::require_finite(-3.5, "x"));
}

TEST(Validation, Positive) {
    EXPECT_NO_THROW(v::require_positive(1e-9, "x"));
    EXPECT_THROW(v::require_positive(0.0, "x"), std::invalid_argument);
}

TEST(Validation, NonNegative) {
    EXPECT_NO_THROW(v::require_non_negative(0.0, "x"));
    EXPECT_THROW(v::require_non_negative(-1e-12, "x"), std::invalid_argument);
}

TEST(Validation, InRangeIsInclusive) {
    EXPECT_NO_THROW(v::require_in_range(0.0, 0.0, 100.0, "p"));
    EXPECT_NO_THROW(v::require_in_range(100.0, 0.0, 100.0, "p"));
    EXPECT_THROW(v::require_in_range(100.1, 0.0, 100.0, "p"), std::invalid_argument);
}

TEST(Validation, CustomCondition) {
    EXPECT_NO_THROW(v::require(true, "x", "must hold", 1.0));
    EXPECT_THROW(v::require(false, "x", "must hold", 1.0), std::invalid_argument);
}

TEST(Validation, MessageNamesParameter) {
    try {
        v::require_positive(0.0, "x");
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()).rfind("Domain error [BlastingPy]: parameter 'x'", 0), 0u);
    }
}
