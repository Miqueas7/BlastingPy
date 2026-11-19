/**
 * @file pattern.hpp
 * @brief Malla de banco: burden, espaciamiento, taco y sobreperforación por Konya (FHWA), Ash (IC 8550) y López
 *        Jimeno (UC).
 *
 * Konya y Walter (1991), FHWA-HI-92-001 [Primary]. Páginas del PDF (la impresa entre paréntesis):
 * - Burden B = (2·SGe/SGr + 1,5)·De, B en pies y De en pulgadas: ec. 6.2, p. 110 (91); ejemplo 6.2 (6,25 ft), p. 111.
 *   De es el diámetro del explosivo; aquí el del taladro (carga acoplada). SGe/SGr = ρe/ρr (ambas en kg/m³).
 * - Correcciones B·Kd·Ks·Kr (pp. 112-114): Kr por número de filas, Tabla 6.2 p. 113 (1,0 una o dos filas; 0,9 tercera
 *   y siguientes o voladura contra escombro); Kd por depósito, Tabla 6.3 p. 113; Ks por estructura, Tabla 6.4 p. 114.
 *   Por omisión valen 1: la fórmula publicada sin corregir.
 * - Taco T = 0,7·B: ec. 6.4, p. 115 (96). Sobreperforación J = 0,3·B: ec. 6.6, p. 118 (99).
 * - Espaciamiento por la relación de rigidez L/B (pp. 157-162): L/B < 4 es banco bajo, ≥ 4 alto (p. 158).
 *   Instantáneo: S = (L + 2B)/3 (ec. 7.1, ejemplo 7.1 p. 159) y S = 2B (ec. 7.2, ejemplo 7.2 p. 160).
 *   Con retardos: S = (L + 7B)/8 (ec. 7.3, ejemplo 7.3 p. 161) y S = 1,4B (ec. 7.4, ejemplo 7.4 p. 162).
 *   La fuente define el banco bajo con 1 < L/B < 4 (p. 158); con L/B ≤ 1 se calcula igual y la nota lo avisa.
 *
 * Ash, por Pugliese (1972), USBM IC 8550 [Secondary: no se leyó a Ash directamente]:
 * - B = K_B·De/12 (pies, De en pulgadas): ec. (1), p. 13; K_B = 30 promedio, 25 explosivo de baja densidad (ANFO),
 *   35 denso (hidrogeles, gelatinas), para roca de ~2,7 g/cm³: p. 14.
 * - Instantáneo: S = K_S·B, K_S = 1,8-2 (p. 14); se usa el promedio 2 de la Tabla 1 (p. 16).
 * - Con retardos: s = K_S·b con b = 1,4·B (malla cuadrada), K_S = 1-1,2 (p. 15); se usa el promedio 1 de la Tabla 1
 *   (p. 16). Cada fila va en el marco de la malla perforada, como Konya: con retardos burden_m = b = 1,4·B y
 *   spacing_m = s; taco y sobreperforación siguen sobre el burden verdadero B (pp. 15-16).
 * - J = K_J·B con K_J = 0,3 (preferido, p. 15; promedio de la Tabla 1, p. 16).
 * - T = K_T·B con K_T = 0,7 (p. 16; promedio de la Tabla 1, p. 16). No está en pp. 14-15.
 *
 * López Jimeno, por U. Cantabria OCW, Tema 3.1, p. 18 [Secondary: tabla adaptada de López Jimeno et al. (2003)]:
 * - Por clase de UCS, con D en metros: V = 39/37/35/33·D, E = 51/47/43/38·D, T = 35/34/32/30·D, Sp = 10/11/12/12·D
 *   (blanda < 70, media 70-120, dura 120-180, muy dura > 180 MPa).
 * - La UC no resuelve los bordes: 70, 120 y 180 MPa pasan a la clase más dura (decisión del plan).
 * - Rango declarado: diámetro medio, 65-165 mm (título de p. 18). Fuera se calcula igual y la nota lo avisa.
 *
 * Langefors-Kihlström: no se programa. La ecuación de UC p. 13 (V = (d/33)·√(ρf·Sf/(c·f·E/V)), c sin corregir)
 * no coincide con Tomalá y Guerrero (2026, UAzuay, p. 51 del PDF): Bmax = (D/33)·D·√(de·PRP/(k·f·S/B)), k = c + 0,05.
 * Difieren en la forma (factor D de más) y en la constante de roca. Queda pendiente del libro.
 *
 * Konya y Ash se programan en pies y pulgadas, como se publicaron, y se convierten a SI en el borde.
 * Las entradas no admiten NaN ni infinito; los errores son de dominio (ver validation.hpp).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "blastingpy/method_info.hpp"
#include "blastingpy/units.hpp"
#include "blastingpy/validation.hpp"

namespace blastingpy::pattern {

enum class Initiation { Delayed, Instantaneous };
enum class ExplosiveClass { LowDensity, Average, Dense };

struct BlastDesign {
    double burden_m, spacing_m, stemming_m, subdrill_m;
    MethodInfo info;
    std::string note;
};

inline BlastDesign konya(double diameter_mm, double bench_height_m, double rock_density_kg_m3,
                         double explosive_density_kg_m3, Initiation initiation, double kd = 1.0, double ks = 1.0,
                         double kr = 1.0) {
    validation::require_positive(diameter_mm, "diameter_mm");
    validation::require_positive(bench_height_m, "bench_height_m");
    validation::require_positive(rock_density_kg_m3, "rock_density_kg_m3");
    validation::require_positive(explosive_density_kg_m3, "explosive_density_kg_m3");
    validation::require_positive(kd, "kd");
    validation::require_positive(ks, "ks");
    validation::require_positive(kr, "kr");

    const double de_in = diameter_mm / units::mm_per_in;
    const double l_ft = bench_height_m / units::m_per_ft;
    const double b_ft = (2.0 * explosive_density_kg_m3 / rock_density_kg_m3 + 1.5) * de_in * kd * ks * kr;
    const double stiffness = l_ft / b_ft;
    double s_ft;
    if (initiation == Initiation::Instantaneous)
        s_ft = stiffness < 4.0 ? (l_ft + 2.0 * b_ft) / 3.0 : 2.0 * b_ft;
    else
        s_ft = stiffness < 4.0 ? (l_ft + 7.0 * b_ft) / 8.0 : 1.4 * b_ft;

    std::string note;
    if (kd == 1.0 && ks == 1.0 && kr == 1.0) note = "uncorrected burden (Kd = Ks = Kr = 1)";
    if (stiffness <= 1.0) {
        if (!note.empty()) note += "; ";
        note += "stiffness ratio L/B <= 1 is outside the FHWA spacing range (1 < L/B)";
    }
    return {b_ft * units::m_per_ft, s_ft * units::m_per_ft, 0.7 * b_ft * units::m_per_ft,
            0.3 * b_ft * units::m_per_ft,
            {"FHWA-HI-92-001", Verification::Primary,
             "Konya & Walter (1991), FHWA-HI-92-001: burden eq. 6.2 p. 110, corrections pp. 112-114, stemming eq. 6.4 "
             "p. 115, subdrill eq. 6.6 p. 118, spacing eqs. 7.1-7.4 pp. 158-162"},
            note};
}

inline BlastDesign ash(double diameter_mm, ExplosiveClass cls, Initiation initiation) {
    validation::require_positive(diameter_mm, "diameter_mm");
    const double kb = cls == ExplosiveClass::LowDensity ? 25.0 : cls == ExplosiveClass::Dense ? 35.0 : 30.0;
    const double b_ft = kb * (diameter_mm / units::mm_per_in) / 12.0;
    const bool delayed = initiation == Initiation::Delayed;
    // Marco de la malla perforada, como Konya: con retardos las filas se perforan a b = 1,4·B (p. 15).
    const double drilled_b_ft = delayed ? 1.4 * b_ft : b_ft;
    const double s_ft = delayed ? 1.0 * drilled_b_ft : 2.0 * b_ft;
    return {drilled_b_ft * units::m_per_ft, s_ft * units::m_per_ft, 0.7 * b_ft * units::m_per_ft,
            0.3 * b_ft * units::m_per_ft,
            {"USBM-IC8550", Verification::Secondary,
             "Ash's ratios in Pugliese (1972), USBM IC 8550: K_B p. 14, K_S pp. 14-15, K_J p. 15, K_T p. 16 "
             "(averages of Table 1, p. 16)"},
            delayed ? "delayed: burden_m is the drilled row distance b = 1.4*B and spacing_m = K_S*b (square pattern, "
                      "IC 8550 p. 15), while stemming and subdrill use the true burden B"
                    : ""};
}

inline BlastDesign lopez_jimeno(double diameter_mm, double ucs_mpa) {
    validation::require_positive(diameter_mm, "diameter_mm");
    validation::require_positive(ucs_mpa, "ucs_mpa");
    // Filas de UC p. 18: {V, E, T, Sp} en múltiplos de D [m]; los bordes van a la clase más dura.
    static constexpr double table[4][4] = {{39, 51, 35, 10}, {37, 47, 34, 11}, {35, 43, 32, 12}, {33, 38, 30, 12}};
    const int cls = ucs_mpa < 70.0 ? 0 : ucs_mpa < 120.0 ? 1 : ucs_mpa < 180.0 ? 2 : 3;
    const double* k = table[cls];
    const double d_m = diameter_mm / 1000.0;

    std::string note;
    if (ucs_mpa == 70.0 || ucs_mpa == 120.0 || ucs_mpa == 180.0)
        note = "UCS on a class boundary: taken as the harder class (UC p. 18 does not resolve boundaries)";
    if (diameter_mm < 65.0 || diameter_mm > 165.0) {
        if (!note.empty()) note += "; ";
        note += "diameter outside the 65-165 mm range of the table (UC p. 18)";
    }
    return {k[0] * d_m, k[1] * d_m, k[2] * d_m, k[3] * d_m,
            {"LopezJimeno2003", Verification::Secondary,
             "Lopez Jimeno et al. (2003) table by UCS, as adapted in U. Cantabria OCW, Tema 3.1 (Perez Alvarez), p. 18"},
            note};
}

/*
 * Malla de banco: layout(). Ninguna fuente da un algoritmo; son decisiones de diseño (diseño de la malla de banco,
 * §4.2, 19-nov-2026), no fórmulas:
 * - Filas paralelas a la recta primer-último punto de la cresta, a (r+1)·B hacia el lado del centroide del contorno.
 * - Una sola grilla: O + k·S·u, más S/2 en las filas impares en tresbolillo. O es el extremo de la cresta desde el
 *   que apunta u, y u se orienta para que el interior quede a la izquierda: la cresta al revés da la misma malla.
 * - Dentro estricto: un collar sobre el contorno queda fuera. Las filas siguen mientras alguna posición caiga dentro.
 * - En cada extremo de cada tramo de la fila, si el hueco hasta el contorno pasa de S/2, va un taladro de borde en
 *   el punto medio. Un tramo sin taladros de la grilla no lleva ninguno.
 * - front_off_ids: taladros de la fila 0 cuya distancia a la cresta real difiere de B en más de front_tolerance·B.
 * Todo se calcula en un marco local (t a lo largo de u, v sobre n, origen O) para no perder precisión con UTM.
 */
struct Point { double x, y; };
enum class Arrangement { Square, Rectangular, Staggered };
struct Hole { int id; int row; double x, y, along_m; bool edge; };
struct Layout { std::vector<Hole> holes; std::vector<int> front_off_ids; double rear_gap_m; };

namespace detail {
inline double segment_distance(double px, double py, double ax, double ay, double bx, double by) {
    const double dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
    const double s = len2 > 0 ? std::clamp(((px - ax) * dx + (py - ay) * dy) / len2, 0.0, 1.0) : 0.0;
    return std::hypot(px - ax - s * dx, py - ay - s * dy);
}
}  // namespace detail

inline Layout layout(const std::vector<Point>& outline, const std::vector<Point>& crest, double burden_m,
                     double spacing_m, Arrangement arrangement, double front_tolerance = 0.20) {
    validation::require_positive(burden_m, "burden_m");
    validation::require_positive(spacing_m, "spacing_m");
    validation::require_non_negative(front_tolerance, "front_tolerance");
    if (arrangement == Arrangement::Square)
        validation::require(spacing_m == burden_m, "spacing_m",
                            "must equal burden_m for a square pattern (use rectangular otherwise)", spacing_m);
    for (const auto& p : outline) {
        validation::require_finite(p.x, "outline x");
        validation::require_finite(p.y, "outline y");
    }
    for (const auto& p : crest) {
        validation::require_finite(p.x, "crest x");
        validation::require_finite(p.y, "crest y");
    }
    validation::require(crest.size() >= 2, "crest", "must have at least 2 points", double(crest.size()));

    std::vector<Point> poly = outline;
    if (poly.size() > 1 && poly.front().x == poly.back().x && poly.front().y == poly.back().y) poly.pop_back();
    validation::require(poly.size() >= 3, "outline", "must have at least 3 vertices", double(poly.size()));

    Point o = crest.front();
    const double len = std::hypot(crest.back().x - o.x, crest.back().y - o.y);
    validation::require(len > 0, "crest", "first and last points must differ (length of the crest line)", len);
    double ux = (crest.back().x - o.x) / len, uy = (crest.back().y - o.y) / len;

    // Centroide por la fórmula del área (shoelace), relativo a o: vale para los dos sentidos de giro.
    double a2 = 0, cx = 0, cy = 0;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        const auto& p = poly[i];
        const auto& q = poly[(i + 1) % poly.size()];
        const double px = p.x - o.x, py = p.y - o.y, qx = q.x - o.x, qy = q.y - o.y;
        const double c = px * qy - qx * py;
        a2 += c;
        cx += (px + qx) * c;
        cy += (py + qy) * c;
    }
    validation::require(a2 != 0, "outline", "must enclose a non-zero area (area in m2)", a2 / 2);
    if ((-uy * cx + ux * cy) / a2 < 0) {  // v del centroide (·3) del lado −n: se invierten u y n
        ux = -ux;
        uy = -uy;
        o = crest.back();
    }
    const double nx = -uy, ny = ux;

    // Contorno en el marco local (t, v).
    std::vector<Point> tv;
    double depth = -INFINITY;
    for (const auto& p : poly) {
        const double dx = p.x - o.x, dy = p.y - o.y;
        tv.push_back({dx * ux + dy * uy, dx * nx + dy * ny});
        depth = std::max(depth, tv.back().y);
    }
    const auto on_boundary = [&](double t, double v) {
        for (std::size_t i = 0; i < tv.size(); ++i) {
            const auto& a = tv[i];
            const auto& b = tv[(i + 1) % tv.size()];
            if (detail::segment_distance(t, v, a.x, a.y, b.x, b.y) <= 1e-9) return true;
        }
        return false;
    };

    Layout out{{}, {}, 0.0};
    const double s = spacing_m;
    double last_v = 0;
    for (int r = 0;; ++r) {
        const double v = (r + 1) * burden_m;
        if (v >= depth) break;  // más allá del vértice más lejano no hay interior
        std::vector<double> cuts;
        for (std::size_t i = 0; i < tv.size(); ++i) {
            const auto& a = tv[i];
            const auto& b = tv[(i + 1) % tv.size()];
            if ((a.y > v) != (b.y > v)) cuts.push_back(a.x + (v - a.y) * (b.x - a.x) / (b.y - a.y));
        }
        std::sort(cuts.begin(), cuts.end());
        const double offset = arrangement == Arrangement::Staggered && r % 2 == 1 ? s / 2 : 0.0;
        const auto add = [&](double t, bool edge) {
            out.holes.push_back({0, r, o.x + t * ux + v * nx, o.y + t * uy + v * ny, t, edge});
        };
        bool any = false;
        for (std::size_t i = 0; i + 1 < cuts.size(); i += 2) {
            const double lo = cuts[i], hi = cuts[i + 1];
            std::vector<double> ts;
            for (auto k = static_cast<long long>(std::ceil((lo - offset) / s)); offset + k * s < hi; ++k) {
                const double t = offset + k * s;
                if (t > lo && !on_boundary(t, v)) ts.push_back(t);
            }
            if (ts.empty()) continue;
            any = true;
            if (ts.front() - lo > s / 2) add((lo + ts.front()) / 2, true);
            for (double t : ts) add(t, false);
            if (hi - ts.back() > s / 2) add((ts.back() + hi) / 2, true);
        }
        if (!any) break;
        last_v = v;
    }
    if (out.holes.empty())
        throw std::invalid_argument("no hole fits: burden_m = " + validation::detail::num(burden_m) +
                                    " but the outline reaches only " + validation::detail::num(depth) +
                                    " m from the crest line on the inward side");

    for (std::size_t i = 0; i < out.holes.size(); ++i) {
        auto& h = out.holes[i];
        h.id = static_cast<int>(i) + 1;
        if (h.row != 0) continue;
        const double hx = h.along_m * ux + burden_m * nx, hy = h.along_m * uy + burden_m * ny;
        double d = INFINITY;
        for (std::size_t j = 0; j + 1 < crest.size(); ++j)
            d = std::min(d, detail::segment_distance(hx, hy, crest[j].x - o.x, crest[j].y - o.y,
                                                     crest[j + 1].x - o.x, crest[j + 1].y - o.y));
        if (std::fabs(d - burden_m) > front_tolerance * burden_m) out.front_off_ids.push_back(h.id);
    }
    out.rear_gap_m = depth - last_v;
    return out;
}

}  // namespace blastingpy::pattern
