#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/pattern.hpp"

namespace pt = blastingpy::pattern;
using pt::Arrangement;
using pt::Point;

namespace {
// Rectángulo de 0 a w en x y de 0 a h en y, antihorario y abierto.
std::vector<Point> rect(double w, double h) { return {{0, 0}, {w, 0}, {w, h}, {0, h}}; }

std::vector<pt::Hole> row_of(const pt::Layout& lay, int row) {
    std::vector<pt::Hole> out;
    for (const auto& h : lay.holes)
        if (h.row == row) out.push_back(h);
    return out;
}

std::vector<double> xs(const std::vector<pt::Hole>& holes, bool edge) {
    std::vector<double> out;
    for (const auto& h : holes)
        if (h.edge == edge) out.push_back(h.x);
    std::sort(out.begin(), out.end());
    return out;
}

void expect_same_holes(const pt::Layout& a, const pt::Layout& b, double tol = 1e-9) {
    ASSERT_EQ(a.holes.size(), b.holes.size());
    for (std::size_t i = 0; i < a.holes.size(); ++i) {
        EXPECT_EQ(a.holes[i].id, b.holes[i].id);
        EXPECT_EQ(a.holes[i].row, b.holes[i].row);
        EXPECT_NEAR(a.holes[i].x, b.holes[i].x, tol);
        EXPECT_NEAR(a.holes[i].y, b.holes[i].y, tol);
        EXPECT_EQ(a.holes[i].edge, b.holes[i].edge);
    }
}

// L: brazo inferior 0..30 × 0..10 y brazo izquierdo 0..10 × 0..30, antihorario y abierto.
std::vector<Point> ell() { return {{0, 0}, {30, 0}, {30, 10}, {10, 10}, {10, 30}, {0, 30}}; }
}  // namespace

// Filas en y = 15, 10 y 5 (y = 0 es borde: fuera); x = 5..35 más bordes en 2,5 y 37,5.
TEST(PatternLayout, SquareInRectangleExactCount) {
    const auto lay = pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 5, Arrangement::Square);
    ASSERT_EQ(lay.holes.size(), 27u);
    EXPECT_EQ(std::count_if(lay.holes.begin(), lay.holes.end(), [](const pt::Hole& h) { return h.edge; }), 6);
    const double row_y[] = {15, 10, 5};
    for (int r = 0; r < 3; ++r) {
        const auto row = row_of(lay, r);
        ASSERT_EQ(row.size(), 9u);
        for (const auto& h : row) EXPECT_NEAR(h.y, row_y[r], 1e-9);
        const auto grid = xs(row, false);
        ASSERT_EQ(grid.size(), 7u);
        for (int k = 0; k < 7; ++k) EXPECT_NEAR(grid[k], 5.0 * (k + 1), 1e-9);
        const auto edges = xs(row, true);
        ASSERT_EQ(edges.size(), 2u);
        EXPECT_NEAR(edges[0], 2.5, 1e-9);
        EXPECT_NEAR(edges[1], 37.5, 1e-9);
    }
    for (std::size_t i = 0; i < lay.holes.size(); ++i) EXPECT_EQ(lay.holes[i].id, static_cast<int>(i) + 1);
    EXPECT_TRUE(lay.front_off_ids.empty());
    EXPECT_NEAR(lay.rear_gap_m, 5.0, 1e-9);
}

// Origen en el extremo (40, 20): fila 0 en x = 40 − k·5,75; fila 1 desplazada S/2.
TEST(PatternLayout, StaggeredOffsetsOddRows) {
    const double s = 5.75;
    const auto lay = pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, s, Arrangement::Staggered);
    for (int r = 0; r < 3; ++r) {
        const double offset = r % 2 == 1 ? s / 2 : 0.0;
        std::vector<double> expected;
        for (int k = 0;; ++k) {
            const double x = 40 - offset - k * s;
            if (x <= 0) break;
            if (x < 40) expected.push_back(x);
        }
        std::sort(expected.begin(), expected.end());
        const auto grid = xs(row_of(lay, r), false);
        ASSERT_EQ(grid.size(), expected.size()) << "row " << r;
        for (std::size_t i = 0; i < grid.size(); ++i) EXPECT_NEAR(grid[i], expected[i], 1e-9);
    }
    const auto r0 = xs(row_of(lay, 0), false), r1 = xs(row_of(lay, 1), false);
    EXPECT_NEAR(r1.back() - r0.back(), s / 2, 1e-9);  // 37,125 frente a 34,25
}

// Ancho 32,45: el tramo libre del lado x = 0 es 2,45 = 0,49·S (sin borde); con 32,55 es 2,55 = 0,51·S (borde).
TEST(PatternLayout, EdgeRuleThreshold) {
    const double w_short = 30 + 0.49 * 5, w_long = 30 + 0.51 * 5;
    const auto no_edge = row_of(pt::layout(rect(w_short, 20), {{0, 20}, {w_short, 20}}, 5, 5, Arrangement::Square), 0);
    const auto edges_no = xs(no_edge, true);
    ASSERT_EQ(edges_no.size(), 1u);  // solo el del lado del origen (tramo 5)
    EXPECT_NEAR(edges_no[0], w_short - 2.5, 1e-9);
    EXPECT_NEAR(xs(no_edge, false).front(), 0.49 * 5, 1e-9);

    const auto with_edge =
        row_of(pt::layout(rect(w_long, 20), {{0, 20}, {w_long, 20}}, 5, 5, Arrangement::Square), 0);
    const auto edges_yes = xs(with_edge, true);
    ASSERT_EQ(edges_yes.size(), 2u);
    EXPECT_NEAR(edges_yes[0], 0.51 * 5 / 2, 1e-9);  // punto medio del tramo
}

// Cresta en x + y = 41, B·√2 = 5: las filas cortan las dos ramas de la L. La fila 0 (x + y = 36) tiene dos tramos,
// x en (26, 30) y en (6, 10), cada uno con un taladro de la grilla y un borde a cada lado (tramos 3/√2 y 5/√2 > S/2).
TEST(PatternLayout, NonConvexRowSplitsIntoTwoIntervals) {
    const double b = 5 / std::sqrt(2.0);
    const auto lay = pt::layout(ell(), {{41, 0}, {0, 41}}, b, b, Arrangement::Square);
    for (const auto& h : lay.holes) {
        const bool lower = h.x > 0 && h.x < 30 && h.y > 0 && h.y < 10;
        const bool left = h.x > 0 && h.x < 10 && h.y > 0 && h.y < 30;
        EXPECT_TRUE(lower || left) << "hole " << h.id << " at (" << h.x << ", " << h.y << ")";
    }
    const auto row0 = row_of(lay, 0);
    const double x0[] = {29.25, 28.5, 27.25, 9.25, 8.5, 7.25};
    const bool e0[] = {true, false, true, true, false, true};
    ASSERT_EQ(row0.size(), 6u);
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(row0[i].id, i + 1);
        EXPECT_NEAR(row0[i].x, x0[i], 1e-9);
        EXPECT_NEAR(row0[i].y, 36 - x0[i], 1e-9);
        EXPECT_EQ(row0[i].edge, e0[i]);
    }
}

TEST(PatternLayout, OrientationAndClosureDoNotMatter) {
    const double b = 5 / std::sqrt(2.0);
    const std::vector<Point> crest = {{41, 0}, {0, 41}};
    const auto ccw = ell();
    auto cw = ccw;
    std::reverse(cw.begin(), cw.end());
    auto ccw_closed = ccw, cw_closed = cw;
    ccw_closed.push_back(ccw.front());
    cw_closed.push_back(cw.front());

    const auto ref = pt::layout(ccw, crest, b, b, Arrangement::Square);
    ASSERT_FALSE(ref.holes.empty());
    for (const auto& poly : {cw, ccw_closed, cw_closed})
        expect_same_holes(ref, pt::layout(poly, crest, b, b, Arrangement::Square));
}

TEST(PatternLayout, UtmCoordinatesMatchLocal) {
    const double e0 = 500000, n0 = 8600000;
    std::vector<Point> poly = rect(40, 20);
    for (auto& p : poly) p = {p.x + e0, p.y + n0};
    const auto local = pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 5, Arrangement::Square);
    const auto utm = pt::layout(poly, {{e0, n0 + 20}, {e0 + 40, n0 + 20}}, 5, 5, Arrangement::Square);
    ASSERT_EQ(utm.holes.size(), local.holes.size());
    for (std::size_t i = 0; i < local.holes.size(); ++i) {
        EXPECT_EQ(utm.holes[i].id, local.holes[i].id);
        EXPECT_EQ(utm.holes[i].edge, local.holes[i].edge);
        EXPECT_NEAR(utm.holes[i].x - e0, local.holes[i].x, 1e-6);
        EXPECT_NEAR(utm.holes[i].y - n0, local.holes[i].y, 1e-6);
        EXPECT_NEAR(utm.holes[i].along_m, local.holes[i].along_m, 1e-6);
    }
}

// Tresbolillo con S = 5,75: la malla no es simétrica, así que el origen tiene que salir igual con la cresta al revés.
TEST(PatternLayout, ReversedCrestGivesSameHoles) {
    const auto fwd = pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 5.75, Arrangement::Staggered);
    const auto rev = pt::layout(rect(40, 20), {{40, 20}, {0, 20}}, 5, 5.75, Arrangement::Staggered);
    expect_same_holes(fwd, rev);
}

// Entrante de 2 m entre x = 15 y x = 25: frente a él el burden real es 3 m, 40 % menos que B = 5.
TEST(PatternLayout, IrregularCrestFlagsFrontHoles) {
    const std::vector<Point> crest = {{0, 20}, {15, 20}, {15, 18}, {25, 18}, {25, 20}, {40, 20}};
    const auto lay = pt::layout(rect(40, 20), crest, 5, 5, Arrangement::Square);
    std::vector<int> expected;
    for (const auto& h : row_of(lay, 0))
        if (h.x > 15 - 1e-9 && h.x < 25 + 1e-9) expected.push_back(h.id);
    EXPECT_EQ(expected, (std::vector<int>{4, 5, 6}));  // x = 25, 20 y 15, numerados desde el extremo x = 40
    auto got = lay.front_off_ids;
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, expected);
}

// Cresta 10 m fuera del contorno: las filas de la grilla en y = 25 (fuera) y y = 20 (sobre el borde) quedan vacías y
// se saltan. La primera fila informada es la de y = 15, a 15 m de la cresta: todos sus taladros van a front_off_ids.
TEST(PatternLayout, CrestOutsideBeyondBurden) {
    const auto lay = pt::layout(rect(40, 20), {{0, 30}, {40, 30}}, 5, 5, Arrangement::Square);
    ASSERT_FALSE(lay.holes.empty());
    EXPECT_EQ(lay.holes.front().row, 0);
    const auto row0 = row_of(lay, 0);
    ASSERT_EQ(row0.size(), 9u);
    std::vector<int> ids;
    for (const auto& h : row0) {
        EXPECT_NEAR(h.y, 15, 1e-9);
        ids.push_back(h.id);
    }
    auto got = lay.front_off_ids;
    std::sort(got.begin(), got.end());
    EXPECT_EQ(got, ids);
    EXPECT_NEAR(row_of(lay, 1).front().y, 10, 1e-9);
    EXPECT_NEAR(row_of(lay, 2).front().y, 5, 1e-9);
    EXPECT_EQ(lay.holes.size(), 27u);
}

// La fila de la grilla de y = 10 cae en un cuello x ∈ (21, 24) sin posiciones (x = 40 − 5k): se salta, y la fila de
// y = 5, detrás del cuello, sale como fila 1.
TEST(PatternLayout, NeckDoesNotStopRows) {
    const std::vector<Point> neck = {{0, 0},   {40, 0},  {40, 8},  {24, 8}, {24, 12}, {40, 12},
                                     {40, 20}, {0, 20},  {0, 12},  {21, 12}, {21, 8}, {0, 8}};
    const auto lay = pt::layout(neck, {{0, 20}, {40, 20}}, 5, 5, Arrangement::Square);
    const auto row0 = row_of(lay, 0), row1 = row_of(lay, 1);
    ASSERT_EQ(row0.size(), 9u);
    ASSERT_EQ(row1.size(), 9u);
    EXPECT_NEAR(row0.front().y, 15, 1e-9);
    EXPECT_NEAR(row1.front().y, 5, 1e-9);
    EXPECT_EQ(lay.holes.size(), 18u);
    for (std::size_t i = 0; i < lay.holes.size(); ++i) EXPECT_EQ(lay.holes[i].id, static_cast<int>(i) + 1);
}

// El desfase del tresbolillo sigue la fila de la grilla, no la informada: con la cresta en y = 25 la primera fila
// informada (y = 15) es la fila 1 de la grilla y lleva el desfase S/2, igual que la fila 1 con la cresta en y = 20.
TEST(PatternLayout, StaggeredParityFollowsGridRow) {
    const auto shifted = pt::layout(rect(40, 20), {{0, 25}, {40, 25}}, 5, 5.75, Arrangement::Staggered);
    const auto ref = pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 5.75, Arrangement::Staggered);
    const auto a = row_of(shifted, 0), b = row_of(ref, 1);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i].x, b[i].x, 1e-9);
        EXPECT_NEAR(a[i].y, 15, 1e-9);
        EXPECT_EQ(a[i].edge, b[i].edge);
    }
}

TEST(PatternLayout, NoHoleFitsThrows) {
    try {
        pt::layout(rect(40, 3), {{0, 3}, {40, 3}}, 5, 5, Arrangement::Square);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("no hole fits"), std::string::npos) << e.what();
    }
}

TEST(PatternLayout, SquareRequiresEqualSpacing) {
    EXPECT_THROW(pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 6, Arrangement::Square), std::invalid_argument);
    EXPECT_NO_THROW(pt::layout(rect(40, 20), {{0, 20}, {40, 20}}, 5, 6, Arrangement::Rectangular));
}

TEST(PatternLayout, RejectsBadInput) {
    const std::vector<Point> crest = {{0, 20}, {40, 20}};
    EXPECT_THROW(pt::layout({{0, 0}, {1, 0}}, crest, 5, 5, Arrangement::Square), std::invalid_argument);
    EXPECT_THROW(pt::layout({{0, 0}, {1, 0}, {2, 0}}, crest, 5, 5, Arrangement::Square), std::invalid_argument);
    EXPECT_THROW(pt::layout(rect(40, 20), {{0, 20}, {0, 20}}, 5, 5, Arrangement::Square), std::invalid_argument);
    EXPECT_THROW(pt::layout(rect(40, 20), {{0, 20}}, 5, 5, Arrangement::Square), std::invalid_argument);
    EXPECT_THROW(pt::layout(rect(40, 20), crest, std::nan(""), 5, Arrangement::Square), std::invalid_argument);
    EXPECT_THROW(pt::layout(rect(40, 20), crest, 5, 0, Arrangement::Rectangular), std::invalid_argument);
    EXPECT_THROW(pt::layout({{0, 0}, {40, 0}, {40, std::nan("")}}, crest, 5, 5, Arrangement::Square),
                 std::invalid_argument);
}
