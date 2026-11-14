#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "blastingpy/airblast.hpp"
#include "blastingpy/units.hpp"

namespace ab = blastingpy::airblast;
namespace units = blastingpy::units;

TEST(Airblast, Constants) {
    EXPECT_DOUBLE_EQ(ab::reference_pressure_pa, 20e-6);
    EXPECT_DOUBLE_EQ(units::pa_per_psi, 6894.757);
}

TEST(PaToDb, PublishedPsiPairsWithinHalfDb) {
    // RI 8485 redondea al entero: ±0,5 dB.
    const double pairs[][2] = {{0.0145, 134}, {0.0205, 137}, {0.0115, 132}, {0.0183, 136}, {0.030, 140},
                               {0.100, 151},  {0.0015, 114}, {0.00133, 113}, {0.00037, 102}};
    for (const auto& p : pairs) {
        EXPECT_NEAR(ab::pa_to_db(p[0] * units::pa_per_psi), p[1], 0.5) << p[0] << " psi";
    }
}

TEST(PaToDb, PublishedPaPairs) {
    EXPECT_NEAR(ab::pa_to_db(1340.0), 157.0, 0.5);
    EXPECT_NEAR(ab::pa_to_db(6900.0), 171.0, 0.5);
}

TEST(PaToDb, ExactValues) {
    EXPECT_EQ(ab::pa_to_db(20e-6), 0.0);
    EXPECT_NEAR(ab::pa_to_db(100.0), 133.9794, 1e-4);
}

TEST(DbToPa, RoundTrip) {
    for (double p : {1e-3, 1.0, 100.0, 6900.0}) {
        EXPECT_NEAR(ab::db_to_pa(ab::pa_to_db(p)) / p, 1.0, 1e-12);
    }
}

TEST(Airblast, DomainErrors) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (double p : {0.0, -1.0, nan, inf}) EXPECT_THROW(ab::pa_to_db(p), std::invalid_argument) << p;
    for (double d : {nan, inf, -inf}) EXPECT_THROW(ab::db_to_pa(d), std::invalid_argument) << d;
    EXPECT_THROW(ab::convert_k_to_si(0.0, 0.794), std::invalid_argument);
    EXPECT_THROW(ab::convert_k_to_si(-0.1, 0.794), std::invalid_argument);
    EXPECT_THROW(ab::convert_k_to_si(0.162, 0.0), std::invalid_argument);
    EXPECT_THROW(ab::convert_k_to_si(0.162, -0.5), std::invalid_argument);
}

TEST(ConvertKToSi, ReproducesImperialLaw) {
    const double beta = 0.794;
    const double k_si = ab::convert_k_to_si(0.162, beta);
    const double d_m = 100.0 * units::m_per_ft;
    const double w_kg = 1.0 * units::kg_per_lb;
    const double got = k_si * std::pow(d_m / std::cbrt(w_kg), -beta);
    EXPECT_NEAR(got / (0.162 * std::pow(100.0, -beta) * units::pa_per_psi), 1.0, 1e-12);
    EXPECT_NEAR(got / (0.0041833 * units::pa_per_psi), 1.0, 1e-5);  // valor del brief, redondeado a 5 cifras
}
