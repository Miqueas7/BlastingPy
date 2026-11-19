#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "blastingpy/pattern.hpp"
#include "blastingpy/units.hpp"

namespace pt = blastingpy::pattern;
using blastingpy::Verification;
using blastingpy::units::m_per_ft;
using blastingpy::units::mm_per_in;

namespace {
void expect_rel(double actual, double expected, double rel) { EXPECT_NEAR(actual, expected, rel * std::fabs(expected)); }
}  // namespace

// Ejemplo 6.1 (FHWA p. 110) usa la ec. 6.1, B2 = B1·De2/De1: sin cambiar roca ni explosivo, la ec. 6.2 escala igual.
TEST(PatternKonya, FhwaExample61) {
    const auto b6 = pt::konya(6 * mm_per_in, 10, 2400, 850, pt::Initiation::Delayed);
    const auto b4 = pt::konya(4 * mm_per_in, 10, 2400, 850, pt::Initiation::Delayed);
    expect_rel(15 * m_per_ft * b4.burden_m / b6.burden_m, 10 * m_per_ft, 0.01);
}

// Ejemplo 6.2 (FHWA p. 111): SGe 1,3, SGr 2,6, De 2,5 in -> B = 6,25 ft.
TEST(PatternKonya, FhwaExample62) {
    const auto d = pt::konya(2.5 * mm_per_in, 10, 2600, 1300, pt::Initiation::Delayed);
    expect_rel(d.burden_m, 6.25 * m_per_ft, 0.01);
}

// Ejemplos 7.1 a 7.4 (FHWA pp. 159-162), todos con B = 8 ft: De 4 in y SGe/SGr = 0,25 dan (0,5 + 1,5)·4 = 8 ft.
TEST(PatternKonya, SpacingFourCases) {
    struct Case { double bench_ft; pt::Initiation init; double spacing_ft; };
    const Case cases[] = {
        {15, pt::Initiation::Instantaneous, 10.33},  // 7.1: L/B 1,88, (L + 2B)/3
        {33, pt::Initiation::Instantaneous, 16.0},   // 7.2: L/B 4,12, 2B
        {12, pt::Initiation::Delayed, 8.5},          // 7.3: L/B 1,5, (L + 7B)/8
        {35, pt::Initiation::Delayed, 11.2},         // 7.4: L/B 4,38, 1,4B
    };
    for (const auto& c : cases) {
        const auto d = pt::konya(4 * mm_per_in, c.bench_ft * m_per_ft, 3200, 800, c.init);
        expect_rel(d.burden_m, 8 * m_per_ft, 1e-12);
        expect_rel(d.spacing_m, c.spacing_ft * m_per_ft, 0.01);
    }
}

TEST(PatternKonya, StemmingAndSubdrill) {
    const auto d = pt::konya(102, 12, 2700, 1200, pt::Initiation::Delayed);
    expect_rel(d.stemming_m, 0.7 * d.burden_m, 1e-12);
    expect_rel(d.subdrill_m, 0.3 * d.burden_m, 1e-12);
}

TEST(PatternKonya, CorrectionsMultiply) {
    const auto base = pt::konya(102, 12, 2700, 1200, pt::Initiation::Delayed);
    const auto corr = pt::konya(102, 12, 2700, 1200, pt::Initiation::Delayed, 1.1, 1.2, 0.9);
    expect_rel(corr.burden_m, base.burden_m * 1.188, 1e-12);
}

TEST(PatternKonya, InfoIsPrimary) {
    const auto d = pt::konya(102, 12, 2700, 1200, pt::Initiation::Delayed);
    EXPECT_EQ(d.info.verification, Verification::Primary);
    EXPECT_NE(d.info.source.find("FHWA-HI-92-001"), std::string::npos) << d.info.source;
}

TEST(PatternAsh, BurdenByClass) {  // instantáneo: burden_m es el B verdadero
    const double d6 = 6 * mm_per_in;
    const auto inst = pt::Initiation::Instantaneous;
    expect_rel(pt::ash(d6, pt::ExplosiveClass::Average, inst).burden_m, 30 * 6 / 12.0 * m_per_ft, 1e-9);
    expect_rel(pt::ash(d6, pt::ExplosiveClass::LowDensity, inst).burden_m, 25 * 6 / 12.0 * m_per_ft, 1e-9);
    expect_rel(pt::ash(d6, pt::ExplosiveClass::Dense, inst).burden_m, 35 * 6 / 12.0 * m_per_ft, 1e-9);
}

// Promedios de la Tabla 1 (IC 8550 p. 16): K_S 2 simultáneo, K_J 0,3, K_T 0,7.
TEST(PatternAsh, RatiosFromTable1) {
    const auto inst = pt::ash(150, pt::ExplosiveClass::Average, pt::Initiation::Instantaneous);
    expect_rel(inst.spacing_m, 2.0 * inst.burden_m, 1e-12);
    expect_rel(inst.subdrill_m, 0.3 * inst.burden_m, 1e-12);
    expect_rel(inst.stemming_m, 0.7 * inst.burden_m, 1e-12);
}

// Con retardos la fila va en el marco de la malla perforada (p. 15): b = 1,4B, s = K_S·b con K_S = 1 (Tabla 1,
// p. 16); taco y sobreperforación sobre el B verdadero = 30·6/12 = 15 ft.
TEST(PatternAsh, DelayedUsesDrilledFrame) {
    const auto d = pt::ash(6 * mm_per_in, pt::ExplosiveClass::Average, pt::Initiation::Delayed);
    expect_rel(d.burden_m, 1.4 * 15 * m_per_ft, 1e-9);
    expect_rel(d.spacing_m, 1.0 * d.burden_m, 1e-12);
    expect_rel(d.stemming_m, 0.7 * 15 * m_per_ft, 1e-9);
    expect_rel(d.subdrill_m, 0.3 * 15 * m_per_ft, 1e-9);
}

TEST(PatternAsh, InfoIsSecondary) {
    const auto d = pt::ash(150, pt::ExplosiveClass::Average, pt::Initiation::Delayed);
    EXPECT_EQ(d.info.verification, Verification::Secondary);
    EXPECT_NE(d.info.source.find("IC 8550"), std::string::npos) << d.info.source;
}

TEST(PatternBurden, RejectsNonPositiveOrNan) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(pt::konya(0, 10, 2600, 1300, pt::Initiation::Delayed), std::invalid_argument);
    EXPECT_THROW(pt::konya(100, 10, nan, 1300, pt::Initiation::Delayed), std::invalid_argument);
    EXPECT_THROW(pt::konya(100, -1, 2600, 1300, pt::Initiation::Delayed), std::invalid_argument);
    EXPECT_THROW(pt::konya(100, 10, 2600, 1300, pt::Initiation::Delayed, 1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(pt::ash(0, pt::ExplosiveClass::Average, pt::Initiation::Delayed), std::invalid_argument);
    EXPECT_THROW(pt::ash(nan, pt::ExplosiveClass::Average, pt::Initiation::Delayed), std::invalid_argument);
}
