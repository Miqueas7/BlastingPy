#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "blastingpy/least_squares.hpp"

using blastingpy::ols;

TEST(Ols, ExactLine) {
    const std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y;
    for (double xi : x) y.push_back(2.0 + 3.0 * xi);
    const auto r = ols({x}, y);
    ASSERT_EQ(r.coefficients.size(), 2u);
    EXPECT_NEAR(r.coefficients[0], 2.0, 1e-12);
    EXPECT_NEAR(r.coefficients[1], 3.0, 1e-12);
    EXPECT_LE(r.sse, 1e-20);
    EXPECT_EQ(r.dof, 3);
    EXPECT_EQ(r.r_squared, 1.0);
    EXPECT_EQ(r.residuals.size(), 5u);
}

TEST(Ols, TwoPredictorsOnAGrid) {
    std::vector<double> x1, x2, y;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            x1.push_back(i);
            x2.push_back(10.0 * j + 0.5 * i);
            y.push_back(1.0 + 2.0 * x1.back() - 0.5 * x2.back());
        }
    const auto r = ols({x1, x2}, y);
    ASSERT_EQ(r.coefficients.size(), 3u);
    EXPECT_NEAR(r.coefficients[0], 1.0, 1e-10);
    EXPECT_NEAR(r.coefficients[1], 2.0, 1e-10);
    EXPECT_NEAR(r.coefficients[2], -0.5, 1e-10);
    EXPECT_EQ(r.dof, 13);
}

TEST(Ols, NoPredictorsIsTheMean) {
    const auto r = ols({}, {1.0, 2.0, 6.0});
    ASSERT_EQ(r.coefficients.size(), 1u);
    EXPECT_NEAR(r.coefficients[0], 3.0, 1e-15);
    EXPECT_EQ(r.dof, 2);
    EXPECT_NEAR(r.r_squared, 0.0, 1e-15);
    EXPECT_NEAR(r.sigma, std::sqrt(14.0 / 2.0), 1e-12);
}

TEST(Ols, ConstantYIsAPerfectFit) {
    const auto r = ols({{1, 2, 3, 4}}, {5, 5, 5, 5});
    EXPECT_EQ(r.r_squared, 1.0);
    EXPECT_NEAR(r.coefficients[1], 0.0, 1e-12);
}

TEST(Ols, ConstantYWithInexactMeanIsStillAPerfectFit) {
    // 0.1 y 0.7 no son representables: la media calculada difiere de y en ~1e-17 y SST no da 0 exacto.
    EXPECT_EQ(ols({{1, 2, 3}}, {0.1, 0.1, 0.1}).r_squared, 1.0);
    EXPECT_EQ(ols({{1, 2, 3, 4, 5, 6}}, std::vector<double>(6, 0.7)).r_squared, 1.0);
    EXPECT_EQ(ols({{1, 2, 3}}, {0.2, 0.2, 0.2}).r_squared, 1.0);
}

TEST(Ols, RSquaredStaysInUnitInterval) {
    // Pendiente casi nula: el redondeo no debe dar r2 negativo.
    const auto r = ols({{1, 2, 3, 4, 5}}, {0.1, 0.3, 0.2, 0.3, 0.1});
    EXPECT_GE(r.r_squared, 0.0);
    EXPECT_LE(r.r_squared, 1.0);
}

TEST(Ols, NoisyLineHasKnownResiduals) {
    // x = 1..4, y = 1, 3, 2, 4: pendiente 0.8, ordenada 0.5, SSE 1.8 (SST 5 - 0.8*4)
    const auto r = ols({{1, 2, 3, 4}}, {1, 3, 2, 4});
    EXPECT_NEAR(r.coefficients[0], 0.5, 1e-12);
    EXPECT_NEAR(r.coefficients[1], 0.8, 1e-12);
    EXPECT_NEAR(r.sse, 1.8, 1e-12);
    EXPECT_NEAR(r.sigma, std::sqrt(0.9), 1e-12);
    EXPECT_NEAR(r.r_squared, 1.0 - 1.8 / 5.0, 1e-12);
}

TEST(Ols, RejectsInvalidInputs) {
    EXPECT_THROW(ols({{1, 2}}, {1, 2}), std::invalid_argument);            // n <= predictores + 1
    EXPECT_THROW(ols({}, {1.0}), std::invalid_argument);
    EXPECT_THROW(ols({{1, 2, 3}}, {1, 2}), std::invalid_argument);         // largos distintos
    EXPECT_THROW(ols({{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, {1, 2, 3}), std::invalid_argument);  // 3 columnas
    EXPECT_THROW(ols({{1, std::nan(""), 3, 4}}, {1, 2, 3, 4}), std::invalid_argument);
    EXPECT_THROW(ols({{1, 2, 3, 4}}, {1, INFINITY, 3, 4}), std::invalid_argument);
}

TEST(Ols, RejectsConstantAndCollinearPredictors) {
    auto message = [](const std::vector<std::vector<double>>& p) {
        try {
            ols(p, {1, 2, 3, 5, 4});
        } catch (const std::invalid_argument& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::vector<double> x = {1, 2, 3, 4, 5};
    EXPECT_NE(message({{7, 7, 7, 7, 7}}).find("collinear or constant"), std::string::npos);
    EXPECT_NE(message({{0.1, 0.1, 0.1, 0.1, 0.1}}).find("collinear or constant"), std::string::npos);
    EXPECT_NE(message({x, {2, 4, 6, 8, 10}}).find("collinear or constant"), std::string::npos);
    EXPECT_NE(message({x, {5, 4, 3, 2, 1}}).find("collinear or constant"), std::string::npos);
    EXPECT_EQ(message({x, {1, 3, 2, 5, 4}}), "");
    EXPECT_EQ(message({{7, 7, 7, 7, 7}}).rfind("Domain error [BlastingPy]: parameter 'predictors'", 0), 0u);
}
