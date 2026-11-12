/**
 * @file least_squares.hpp
 * @brief Mínimos cuadrados ordinarios con ordenada y 0, 1 o 2 predictores.
 *
 * Ecuaciones normales sobre predictores centrados y escalados (matriz de correlación, 2x2 como máximo)
 * resueltas por eliminación gaussiana con pivoteo. Sirve para la ley de atenuación del sitio
 * (ln v = ln K - beta ln SD, o con dos predictores ln D y ln W).
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "blastingpy/validation.hpp"

namespace blastingpy {

struct OlsResult {
    std::vector<double> coefficients;  ///< coefficients[0] es la ordenada; luego un coeficiente por predictor
    double sse;                        ///< suma de cuadrados de los residuos
    double sigma;                      ///< sqrt(sse / dof)
    int dof;                           ///< n - (predictores + 1)
    double r_squared;                  ///< 1 - SSE/SST (SST respecto de la media de y)
    std::vector<double> residuals;     ///< y - ajuste, en el orden de entrada
};

/// Ajusta y = b0 + b1 x1 [+ b2 x2]. `predictors` trae una columna por predictor (0, 1 o 2).
inline OlsResult ols(const std::vector<std::vector<double>>& predictors, const std::vector<double>& y) {
    namespace v = validation;
    const std::size_t k = predictors.size();
    const std::size_t n = y.size();

    v::require(k <= 2, "predictors", "must have at most 2 columns", static_cast<double>(k));
    for (const auto& col : predictors) {
        v::require(col.size() == n, "predictors", "columns must have the same length as y", static_cast<double>(col.size()));
        for (double x : col) v::require_finite(x, "predictors");
    }
    for (double yi : y) v::require_finite(yi, "y");
    v::require(n > k + 1, "y", "needs more observations than coefficients (n > predictors + 1)", static_cast<double>(n));

    const double nn = static_cast<double>(n);
    auto mean = [&](const std::vector<double>& a) {
        double s = 0.0;
        for (double x : a) s += x;
        return s / nn;
    };

    // Columnas centradas; c[j] = norma de la columna centrada.
    std::vector<std::vector<double>> xc(k, std::vector<double>(n));
    std::vector<double> xbar(k), c(k);
    for (std::size_t j = 0; j < k; ++j) {
        xbar[j] = mean(predictors[j]);
        double raw = 0.0, cen = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            xc[j][i] = predictors[j][i] - xbar[j];
            raw += predictors[j][i] * predictors[j][i];
            cen += xc[j][i] * xc[j][i];
        }
        c[j] = std::sqrt(cen);
        v::require(c[j] > 1e-12 * std::sqrt(raw) && c[j] > 0.0, "predictors", "columns must not be collinear or constant", c[j]);
    }
    const double ybar = mean(y);
    std::vector<double> yc(n);
    double sst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        yc[i] = y[i] - ybar;
        sst += yc[i] * yc[i];
    }

    // Sistema R g = z'yc con z = xc / c (diagonal de R = 1); eliminación gaussiana con pivoteo parcial.
    double a[2][3] = {};
    for (std::size_t p = 0; p < k; ++p) {
        for (std::size_t q = 0; q < k; ++q) {
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += xc[p][i] * xc[q][i];
            a[p][q] = s / (c[p] * c[q]);
        }
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += xc[p][i] * yc[i];
        a[p][k] = s / c[p];
    }
    for (std::size_t col = 0; col < k; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < k; ++r)
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
        v::require(std::fabs(a[piv][col]) > 1e-12, "predictors", "columns must not be collinear or constant",
                   std::fabs(a[piv][col]));
        if (piv != col) std::swap_ranges(a[col], a[col] + k + 1, a[piv]);
        for (std::size_t r = col + 1; r < k; ++r) {
            const double f = a[r][col] / a[col][col];
            for (std::size_t m = col; m <= k; ++m) a[r][m] -= f * a[col][m];
        }
    }
    double g[2] = {};
    for (std::size_t r = k; r-- > 0;) {
        double s = a[r][k];
        for (std::size_t m = r + 1; m < k; ++m) s -= a[r][m] * g[m];
        g[r] = s / a[r][r];
    }

    OlsResult out;
    out.coefficients.assign(k + 1, 0.0);
    out.coefficients[0] = ybar;
    for (std::size_t j = 0; j < k; ++j) {
        out.coefficients[j + 1] = g[j] / c[j];
        out.coefficients[0] -= out.coefficients[j + 1] * xbar[j];
    }
    out.residuals.resize(n);
    out.sse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double fit = 0.0;
        for (std::size_t j = 0; j < k; ++j) fit += out.coefficients[j + 1] * xc[j][i];
        out.residuals[i] = yc[i] - fit;
        out.sse += out.residuals[i] * out.residuals[i];
    }
    out.dof = static_cast<int>(n - k - 1);
    out.sigma = std::sqrt(out.sse / out.dof);
    out.r_squared = sst > 0.0 ? 1.0 - out.sse / sst : (out.sse == 0.0 ? 1.0 : 0.0);
    return out;
}

}  // namespace blastingpy
