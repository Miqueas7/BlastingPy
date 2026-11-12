#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "blastingpy/units.hpp"

namespace units = blastingpy::units;

TEST(Units, Constants) {
    EXPECT_DOUBLE_EQ(units::m_per_ft, 0.3048);
    EXPECT_DOUBLE_EQ(units::kg_per_lb, 0.45359237);
    EXPECT_DOUBLE_EQ(units::mm_per_in, 25.4);
}

TEST(ScaledDistance, SquareRoot) {
    EXPECT_DOUBLE_EQ(units::scaled_distance(100.0, 25.0, 0.5), 20.0);
}

TEST(ScaledDistance, CubeRoot) {
    EXPECT_NEAR(units::scaled_distance(100.0, 27.0, 1.0 / 3.0), 33.333333, 1e-6);
}

TEST(ScaledDistance, OneFootPerPoundIsSiEquivalent) {
    EXPECT_NEAR(units::scaled_distance(units::m_per_ft, units::kg_per_lb, 0.5), 0.452566, 1e-6);        // 1 ft/lb^1/2
    EXPECT_NEAR(units::scaled_distance(units::m_per_ft, units::kg_per_lb, 1.0 / 3.0), 0.396700, 1e-5);  // 1 ft/lb^1/3
}

// Caso publicado: USBM Bulletin 656 (Nicholls, Johnson y Duvall, 1971), pp. 70-71, da W = (D/SD)^2.
// Con D = 500 ft: SD 20 -> 625 lb y SD 50 -> 100 lb; con D = 1000 ft: 2500 lb y 400 lb.
// La distancia escalada es aritmética sin unidades, así que sirve con los valores en pies y libras.
TEST(ScaledDistance, UsbmBulletin656PublishedCases) {
    EXPECT_DOUBLE_EQ(units::scaled_distance(500.0, 625.0, 0.5), 20.0);
    EXPECT_DOUBLE_EQ(units::scaled_distance(500.0, 100.0, 0.5), 50.0);
    EXPECT_DOUBLE_EQ(units::scaled_distance(1000.0, 2500.0, 0.5), 20.0);
    EXPECT_DOUBLE_EQ(units::scaled_distance(1000.0, 400.0, 0.5), 50.0);
}

TEST(ScaledDistance, RejectsInvalidInputs) {
    EXPECT_THROW(units::scaled_distance(0.0, 25.0, 0.5), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(-1.0, 25.0, 0.5), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(100.0, 0.0, 0.5), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(100.0, std::nan(""), 0.5), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(100.0, 25.0, 0.0), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(100.0, 25.0, std::nan("")), std::invalid_argument);
    EXPECT_THROW(units::scaled_distance(100.0, 25.0, INFINITY), std::invalid_argument);
}

TEST(ConvertK, ImperialToSi) {
    EXPECT_NEAR(units::convert_k_to_si(160.0, 1.6, 0.5), 1143.0, 0.5);  // trampa 3 de la investigación
}

TEST(ConvertK, KeepsTheSameVelocity) {
    // v = K (D/W^e)^-beta debe dar la misma velocidad en ambos sistemas.
    const double k = 160.0, beta = 1.6, e = 0.5;
    const double d_ft = 500.0, w_lb = 100.0;
    const double v_imp_in_s = k * std::pow(d_ft / std::pow(w_lb, e), -beta);
    const double sd_si = units::scaled_distance(d_ft * units::m_per_ft, w_lb * units::kg_per_lb, e);
    const double v_si_mm_s = units::convert_k_to_si(k, beta, e) * std::pow(sd_si, -beta);
    EXPECT_NEAR(v_si_mm_s, v_imp_in_s * units::mm_per_in, 1e-9 * v_si_mm_s);
}

TEST(ConvertK, RejectsInvalidInputs) {
    EXPECT_THROW(units::convert_k_to_si(0.0, 1.6, 0.5), std::invalid_argument);
    EXPECT_THROW(units::convert_k_to_si(160.0, 0.0, 0.5), std::invalid_argument);
    EXPECT_THROW(units::convert_k_to_si(160.0, std::nan(""), 0.5), std::invalid_argument);
    EXPECT_THROW(units::convert_k_to_si(160.0, INFINITY, 0.5), std::invalid_argument);
    EXPECT_THROW(units::convert_k_to_si(160.0, 1.6, -0.5), std::invalid_argument);
}
