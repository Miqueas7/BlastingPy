#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "blastingpy/pattern.hpp"

namespace pt = blastingpy::pattern;
using pt::Scheme;
using pt::StartSide;

namespace {
// Filas de n taladros a lo largo de along = 0, 1, 2, ...; ids por fila y luego along, como layout().
std::vector<pt::Hole> grid(const std::vector<int>& counts) {
    std::vector<pt::Hole> out;
    int id = 1;
    for (std::size_t r = 0; r < counts.size(); ++r)
        for (int k = 0; k < counts[r]; ++k) out.push_back({id++, int(r), double(k), -double(r), double(k), false});
    return out;
}
}  // namespace

TEST(PatternSequence, RowByRow) {
    const auto t = pt::firing_times_ms(grid({3, 3}), Scheme::Row, 17, 100);
    EXPECT_EQ(t, (std::vector<double>{0, 17, 34, 100, 117, 134}));
}

TEST(PatternSequence, RowStartRight) {
    const auto t = pt::firing_times_ms(grid({3}), Scheme::Row, 17, 100, StartSide::Right);
    EXPECT_EQ(t, (std::vector<double>{34, 17, 0}));
}

TEST(PatternSequence, VSymmetric) {
    const auto t = pt::firing_times_ms(grid({5}), Scheme::V, 25, 100);
    EXPECT_EQ(t, (std::vector<double>{50, 25, 0, 25, 50}));
}

TEST(PatternSequence, VEvenCountApexIsLowerCentral) {
    const auto t = pt::firing_times_ms(grid({4}), Scheme::V, 25, 100);
    EXPECT_EQ(t, (std::vector<double>{25, 0, 25, 50}));
}

TEST(PatternSequence, VApexMustBeFrontRow) {
    EXPECT_THROW(pt::firing_times_ms(grid({3, 3}), Scheme::V, 25, 100, StartSide::Left, 4), std::invalid_argument);
}

TEST(PatternSequence, VFollowsApexAcrossRowsFigure715) {
    // FHWA fig. 7.15 (retardos iguales): 4 3 2 1 2 3 4 / 5 4 3 2 3 4 5 / 6 5 4 3 4 5 6.
    const auto t = pt::firing_times_ms(grid({7, 7, 7}), Scheme::V, 1, 1);
    EXPECT_EQ(t, (std::vector<double>{3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 3, 4, 5}));
}

TEST(PatternSequence, VNarrowerRowAndTieGoesToLowerRank) {
    // Fila 1 con along 1.5 y 2.5 (apex en along 2): empate, gana el rango menor.
    std::vector<pt::Hole> h = grid({5});
    h.push_back({6, 1, 0, 0, 1.5, false});
    h.push_back({7, 1, 0, 0, 2.5, false});
    const auto t = pt::firing_times_ms(h, Scheme::V, 10, 100);
    EXPECT_EQ(t[5], 100);
    EXPECT_EQ(t[6], 110);
}

TEST(PatternSequence, RejectsBadDelays) {
    EXPECT_THROW(pt::firing_times_ms(grid({3}), Scheme::Row, -1, 0), std::invalid_argument);
    EXPECT_THROW(pt::firing_times_ms(grid({3}), Scheme::Row, 1, NAN), std::invalid_argument);
    EXPECT_THROW(pt::firing_times_ms(grid({3}), Scheme::Row, INFINITY, 1), std::invalid_argument);
}

TEST(PatternSequence, RejectsRowsNotConsecutiveFromZero) {
    // sequence() recibe tablas del usuario: sin la fila 0, o con un hueco, se leería fuera de rango.
    auto no_front = grid({3, 3});
    no_front.erase(no_front.begin(), no_front.begin() + 3);
    EXPECT_THROW(pt::firing_times_ms(no_front, Scheme::Row, 17, 100), std::invalid_argument);
    EXPECT_THROW(pt::firing_times_ms(no_front, Scheme::V, 17, 100), std::invalid_argument);
    auto gap = grid({3, 3});
    for (auto& h : gap) if (h.row == 1) h.row = 2;
    EXPECT_THROW(pt::firing_times_ms(gap, Scheme::Row, 17, 100), std::invalid_argument);
    auto negative = grid({3});
    negative[0].row = -1;
    EXPECT_THROW(pt::firing_times_ms(negative, Scheme::Row, 17, 100), std::invalid_argument);
}

TEST(PatternSequence, RejectsNonFiniteAlong) {
    auto h = grid({3});
    h[1].along_m = NAN;
    EXPECT_THROW(pt::firing_times_ms(h, Scheme::Row, 17, 100), std::invalid_argument);
    h[1].along_m = INFINITY;
    EXPECT_THROW(pt::firing_times_ms(h, Scheme::V, 17, 100), std::invalid_argument);
}
