#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/criteria_curve.hpp"

using blastingpy::CurveLimit;
using blastingpy::Segment;
using blastingpy::SegmentKind;
using blastingpy::intersection_frequency;
using blastingpy::limit_at;

namespace {
const double NaN = std::numeric_limits<double>::quiet_NaN();
const double Inf = std::numeric_limits<double>::infinity();

// Curva alternativa de USBM RI 8507, Apéndice B (en pulgadas): 0.030 in, 0.75 in/s, 0.008 in, 2.0 in/s.
std::vector<Segment> alternative_curve() {
    const double f1 = intersection_frequency(0.75, 0.030), f2 = intersection_frequency(0.75, 0.008),
                 f3 = intersection_frequency(2.0, 0.008);
    return {{SegmentKind::ConstantDisplacement, 1.0, f1, 0.030, NaN},
            {SegmentKind::ConstantVelocity, f1, f2, 0.75, NaN},
            {SegmentKind::ConstantDisplacement, f2, f3, 0.008, NaN},
            {SegmentKind::ConstantVelocity, f3, 100.0, 2.0, NaN}};
}
}  // namespace

TEST(IntersectionFrequency, AlternativeCurveBreakpoints) {
    EXPECT_NEAR(intersection_frequency(0.75, 0.030), 3.97887, 1e-4);
    EXPECT_NEAR(intersection_frequency(0.75, 0.008), 14.9208, 1e-3);
    EXPECT_NEAR(intersection_frequency(2.0, 0.008), 39.7887, 1e-3);
}

TEST(IntersectionFrequency, RejectsNonPositiveOrNonFinite) {
    EXPECT_THROW(intersection_frequency(0.0, 0.03), std::invalid_argument);
    EXPECT_THROW(intersection_frequency(0.75, -0.03), std::invalid_argument);
    EXPECT_THROW(intersection_frequency(NaN, 0.03), std::invalid_argument);
    EXPECT_THROW(intersection_frequency(0.75, Inf), std::invalid_argument);
}

TEST(LimitAt, FourSegmentCurve) {
    const auto c = alternative_curve();
    const double f2 = c[2].f_min;
    EXPECT_NEAR(limit_at(c, 2.0).limit, 0.376991, 1e-5);
    EXPECT_NEAR(limit_at(c, 10.0).limit, 0.75, 1e-12);
    EXPECT_NEAR(limit_at(c, 20.0).limit, 1.005310, 1e-5);
    EXPECT_NEAR(limit_at(c, 50.0).limit, 2.0, 1e-12);
    EXPECT_EQ(limit_at(c, f2).segment, 2);  // en el quiebre manda el tramo superior
    EXPECT_EQ(limit_at(c, 2.0).segment, 0);
    EXPECT_TRUE(limit_at(c, 100.0).defined);  // el último tramo incluye su f_max
    EXPECT_EQ(limit_at(c, 100.0).segment, 3);
    EXPECT_TRUE(limit_at(c, 1.0).defined);  // y el primero incluye su f_min
}

TEST(LimitAt, OutsideTheCurveIsUndefined) {
    const auto c = alternative_curve();
    for (double f : {0.5, 150.0}) {
        const CurveLimit r = limit_at(c, f);
        EXPECT_FALSE(r.defined);
        EXPECT_TRUE(std::isnan(r.limit));
        EXPECT_EQ(r.segment, -1);
    }
}

TEST(LimitAt, LinearAndLogLog) {
    EXPECT_NEAR(limit_at({{SegmentKind::Linear, 10.0, 50.0, 5.0, 15.0}}, 30.0).limit, 10.0, 1e-12);
    const std::vector<Segment> ll{{SegmentKind::LogLog, 4.0, 15.0, 15.0, 20.0}};
    EXPECT_NEAR(limit_at(ll, 4.0).limit, 15.0, 1e-12);
    EXPECT_NEAR(limit_at(ll, 8.0).limit, 17.443, 1e-3);
    EXPECT_NEAR(limit_at(ll, 15.0).limit, 20.0, 1e-9);
}

TEST(LimitAt, StepCurve) {
    const std::vector<Segment> s{{SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN},
                                 {SegmentKind::ConstantVelocity, 40.0, 100.0, 2.0, NaN}};
    EXPECT_DOUBLE_EQ(limit_at(s, 39.9).limit, 0.75);
    EXPECT_DOUBLE_EQ(limit_at(s, 40.0).limit, 2.0);
}

TEST(LimitAt, OpenEndedLastConstantVelocity) {
    const std::vector<Segment> s{{SegmentKind::ConstantVelocity, 0.0, Inf, 12.5, NaN}};
    EXPECT_DOUBLE_EQ(limit_at(s, 1e9).limit, 12.5);
}

TEST(LimitAt, BreakpointsWithinRelativeToleranceAreContiguous) {
    const std::vector<Segment> s{{SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN},
                                 {SegmentKind::ConstantVelocity, 40.0 * (1.0 + 1e-12), 100.0, 2.0, NaN}};
    EXPECT_DOUBLE_EQ(limit_at(s, 50.0).limit, 2.0);
}

// Caso publicado: USBM RI 8507 (Siskind et al., 1980), Apéndice A, p. 71: un desplazamiento constante de 0.008 in
// equivale a unos 0.5 in/s a 10 Hz y 0.25 in/s a 5 Hz (valores redondeados en la fuente; 2π·f·d da 0.503 y 0.251).
TEST(LimitAt, UsbmRi8507ConstantDisplacementPublishedCase) {
    const std::vector<Segment> s{{SegmentKind::ConstantDisplacement, 1.0, 100.0, 0.008, NaN}};
    EXPECT_NEAR(limit_at(s, 10.0).limit, 0.5, 0.005);
    EXPECT_NEAR(limit_at(s, 5.0).limit, 0.25, 0.005);
}

TEST(LimitAt, RejectsInvalidFrequency) {
    const auto c = alternative_curve();
    for (double f : {0.0, -1.0, NaN, Inf}) EXPECT_THROW(limit_at(c, f), std::invalid_argument);
}

TEST(LimitAt, RejectsInvalidCurves) {
    const auto bad = [](std::vector<Segment> s) { EXPECT_THROW(limit_at(s, 10.0), std::invalid_argument); };
    bad({});
    // solape
    bad({{SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN}, {SegmentKind::ConstantVelocity, 30.0, 100.0, 2.0, NaN}});
    // hueco
    bad({{SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN}, {SegmentKind::ConstantVelocity, 50.0, 100.0, 2.0, NaN}});
    // sin ordenar
    bad({{SegmentKind::ConstantVelocity, 40.0, 100.0, 2.0, NaN}, {SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN}});
    // f_min >= f_max, f_min negativa, no finita
    bad({{SegmentKind::ConstantVelocity, 5.0, 5.0, 1.0, NaN}});
    bad({{SegmentKind::ConstantVelocity, -1.0, 5.0, 1.0, NaN}});
    bad({{SegmentKind::ConstantVelocity, NaN, 5.0, 1.0, NaN}});
    bad({{SegmentKind::ConstantVelocity, 1.0, NaN, 1.0, NaN}});
    // velocidades no positivas o NaN
    bad({{SegmentKind::ConstantVelocity, 1.0, 50.0, 0.0, NaN}});
    bad({{SegmentKind::ConstantDisplacement, 1.0, 50.0, -0.008, NaN}});
    bad({{SegmentKind::Linear, 1.0, 50.0, 5.0, NaN}});
    bad({{SegmentKind::Linear, 1.0, 50.0, 5.0, 0.0}});
    bad({{SegmentKind::LogLog, 1.0, 50.0, NaN, 5.0}});
    // LogLog con f_min = 0
    bad({{SegmentKind::LogLog, 0.0, 50.0, 5.0, 10.0}});
    // f_max infinito: solo en el último tramo y solo con velocidad constante
    bad({{SegmentKind::ConstantVelocity, 1.0, Inf, 1.0, NaN}, {SegmentKind::ConstantVelocity, Inf, 100.0, 2.0, NaN}});
    bad({{SegmentKind::ConstantDisplacement, 1.0, Inf, 0.008, NaN}});
    bad({{SegmentKind::Linear, 1.0, Inf, 1.0, 2.0}});
}

TEST(LimitAt, ErrorNamesSegments) {
    try {
        limit_at({{SegmentKind::ConstantVelocity, 1.0, 40.0, 0.75, NaN},
                  {SegmentKind::ConstantVelocity, 50.0, 100.0, 2.0, NaN}},
                 10.0);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_EQ(std::string(e.what()).rfind("Domain error [BlastingPy]: parameter 'segments'", 0), 0u);
    }
}
