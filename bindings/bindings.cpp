/**
 * @file bindings.cpp
 * @brief Único punto de exposición del núcleo C++ a Python (módulo blastingpy._blastingpy_core).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "blastingpy/attenuation.hpp"
#include "blastingpy/criteria_curve.hpp"
#include "blastingpy/least_squares.hpp"
#include "blastingpy/method_info.hpp"
#include "blastingpy/units.hpp"
#include "blastingpy/waveform.hpp"

namespace nb = nanobind;
using namespace blastingpy;

NB_MODULE(_blastingpy_core, m) {
    nb::set_leak_warnings(false);
    m.doc() = "BlastingPy core: blast vibration, airblast and flyrock analysis for mining.";

    nb::enum_<Verification>(m, "Verification", "How well the method's source was verified.")
        .value("Primary", Verification::Primary,
               "Verified in the original source (or a faithful reproduction of its equations by a cited author)")
        .value("PrimaryLaterAuthors", Verification::PrimaryLaterAuthors,
               "Verified in a later publication by the same authors")
        .value("Secondary", Verification::Secondary, "Secondary source only")
        .value("UserDefined", Verification::UserDefined, "Calibrated by the user on site data");

    nb::class_<MethodInfo>(m, "MethodInfo", "Method id, verification level and source of a result.")
        .def(nb::init<std::string, Verification, std::string>(), nb::arg("method_id"), nb::arg("verification"),
             nb::arg("source"))
        .def_ro("method_id", &MethodInfo::method_id)
        .def_ro("verification", &MethodInfo::verification)
        .def_ro("source", &MethodInfo::source);

    // Submódulo privado (el `import *` del paquete no lo arrastra); la API pública es blastingpy.units.
    nb::module_ u = m.def_submodule("_units", "Exact unit factors and scaled distance.");
    u.attr("m_per_ft") = units::m_per_ft;
    u.attr("kg_per_lb") = units::kg_per_lb;
    u.attr("mm_per_in") = units::mm_per_in;
    u.def("scaled_distance", &units::scaled_distance, nb::arg("distance_m"), nb::arg("charge_kg"),
          nb::arg("charge_exponent"), "Scaled distance D / W**e (distance in m, charge in kg).");
    u.def("convert_k_to_si", &units::convert_k_to_si, nb::arg("k_imperial"), nb::arg("beta"),
          nb::arg("charge_exponent"),
          "Convert K of v[in/s] = K (D[ft]/W[lb]**e)**-beta to the K of v[mm/s] with D in m and W in kg.");

    nb::module_ ls = m.def_submodule("_least_squares", "Ordinary least squares with 0, 1 or 2 predictors.");
    nb::class_<OlsResult>(ls, "OlsResult", "Coefficients (intercept first), SSE, sigma, dof, R2 and residuals of a fit.")
        .def_ro("coefficients", &OlsResult::coefficients)
        .def_ro("sse", &OlsResult::sse)
        .def_ro("sigma", &OlsResult::sigma)
        .def_ro("dof", &OlsResult::dof)
        .def_ro("r_squared", &OlsResult::r_squared)
        .def_ro("residuals", &OlsResult::residuals);
    ls.def("ols", &ols, nb::arg("predictors"), nb::arg("y"),
           "Fit y = b0 + b1 x1 [+ b2 x2]; `predictors` is a list of 0, 1 or 2 columns.");

    nb::module_ at = m.def_submodule("_attenuation", "Site attenuation law: fit and upper bound.");
    nb::enum_<Law>(at, "Law", "Form of the site attenuation law.")
        .value("SquareRoot", Law::SquareRoot, "v = K (D / W**0.5)**-beta")
        .value("CubeRoot", Law::CubeRoot, "v = K (D / W**(1/3))**-beta")
        .value("General", Law::General, "v = K W**a D**-beta");
    nb::class_<SiteLaw>(at, "SiteLaw",
                        "Fitted (or user-defined) site law: median K, beta, sigma in log10 and the upper bound "
                        "k_upper = k * 10**(z * sigma).")
        .def_ro("law", &SiteLaw::law)
        .def_ro("k", &SiteLaw::k)
        .def_ro("beta", &SiteLaw::beta)
        .def_ro("charge_exponent", &SiteLaw::charge_exponent)
        .def_ro("sigma", &SiteLaw::sigma)
        .def_ro("n", &SiteLaw::n)
        .def_ro("dof", &SiteLaw::dof)
        .def_ro("r_squared", &SiteLaw::r_squared)
        .def_ro("z", &SiteLaw::z)
        .def_ro("coverage", &SiteLaw::coverage)
        .def_ro("k_upper", &SiteLaw::k_upper)
        .def_ro("convention", &SiteLaw::convention)
        .def_ro("distance_min_m", &SiteLaw::distance_min_m)
        .def_ro("distance_max_m", &SiteLaw::distance_max_m)
        .def_ro("charge_min_kg", &SiteLaw::charge_min_kg)
        .def_ro("charge_max_kg", &SiteLaw::charge_max_kg)
        .def_ro("residuals", &SiteLaw::residuals)
        .def_ro("outliers", &SiteLaw::outliers)
        .def_ro("info", &SiteLaw::info)
        .def_ro("warnings", &SiteLaw::warnings);
    at.def("normal_quantile", &normal_quantile, nb::arg("p"),
           "Standard normal quantile, 0 < p < 1 (normal_quantile(0.95) = 1.645).");
    at.def("fit_site_law", &fit_site_law, nb::arg("distance_m"), nb::arg("charge_kg"), nb::arg("ppv_mm_s"),
           nb::arg("law"), nb::arg("z") = 2.0, nb::arg("outlier_z") = 3.0,
           "Fit the site law by least squares in log10; the upper bound is k * 10**(z * sigma).");
    at.def("make_site_law", &make_site_law, nb::arg("law"), nb::arg("k"), nb::arg("beta"),
           nb::arg("charge_exponent") = validation::unset, nb::arg("sigma") = 0.0, nb::arg("z") = 2.0,
           "User-defined site law (no fit); charge_exponent is required for Law.General.");

    nb::enum_<Level>(at, "Level", "Which constant of the law a prediction uses.")
        .value("Median", Level::Median, "k: the median of the fit")
        .value("Upper", Level::Upper, "k_upper = k * 10**(z * sigma)");
    nb::class_<Prediction>(at, "Prediction", "A value and its warnings (extrapolation; upper bound with no margin).")
        .def_ro("value", &Prediction::value)
        .def_ro("warnings", &Prediction::warnings);
    at.def("predict_ppv", &predict_ppv, nb::arg("law"), nb::arg("distance_m"), nb::arg("charge_kg"),
           nb::arg("level") = Level::Upper, "PPV (mm/s) at distance_m (m) with charge_kg (kg per delay).");
    at.def("max_charge", &max_charge, nb::arg("law"), nb::arg("distance_m"), nb::arg("ppv_limit_mm_s"),
           nb::arg("level") = Level::Upper,
           "Maximum charge per delay (kg) at distance_m (m) that keeps the PPV at or below ppv_limit_mm_s.");
    at.def("distance_for_ppv", &distance_for_ppv, nb::arg("law"), nb::arg("charge_kg"), nb::arg("ppv_limit_mm_s"),
           nb::arg("level") = Level::Upper,
           "Minimum distance (m) at which charge_kg per delay keeps the PPV at or below ppv_limit_mm_s.");

    nb::module_ cc = m.def_submodule("_criteria_curve", "Piecewise PPV-frequency limit curves.");
    nb::enum_<SegmentKind>(cc, "SegmentKind", "Shape of one segment of a limit curve.")
        .value("ConstantVelocity", SegmentKind::ConstantVelocity, "v = v1")
        .value("ConstantDisplacement", SegmentKind::ConstantDisplacement, "v = 2 pi f v1 (v1 is a displacement)")
        .value("Linear", SegmentKind::Linear, "straight line between (f_min, v1) and (f_max, v2), linear axes")
        .value("LogLog", SegmentKind::LogLog, "straight line between (f_min, v1) and (f_max, v2), log-log axes");
    nb::class_<Segment>(cc, "Segment", "One segment of a limit curve, covering [f_min, f_max) in Hz.")
        .def(nb::init<SegmentKind, double, double, double, double>(), nb::arg("kind"), nb::arg("f_min"),
             nb::arg("f_max"), nb::arg("v1"), nb::arg("v2") = validation::unset)
        .def_ro("kind", &Segment::kind)
        .def_ro("f_min", &Segment::f_min)
        .def_ro("f_max", &Segment::f_max)
        .def_ro("v1", &Segment::v1)
        .def_ro("v2", &Segment::v2);
    nb::class_<CurveLimit>(cc, "CurveLimit", "Limit at a frequency; `segment` is the 0-based index (-1 if undefined).")
        .def_ro("defined", &CurveLimit::defined)
        .def_ro("limit", &CurveLimit::limit)
        .def_ro("segment", &CurveLimit::segment);
    cc.def("limit_at", &limit_at, nb::arg("segments"), nb::arg("frequency_hz"),
           "Limit of the curve at frequency_hz (Hz), in the units of the segments; undefined outside the curve.");
    cc.def("intersection_frequency", &intersection_frequency, nb::arg("velocity"), nb::arg("displacement"),
           "Frequency (Hz) where constant velocity and constant displacement meet: velocity / (2 pi displacement).");

    nb::module_ wf = m.def_submodule("_waveform", "Waveform peaks and vector sum.");
    nb::class_<Peak>(wf, "Peak", "Largest |x| (value >= 0) and the index of its first occurrence.")
        .def_ro("value", &Peak::value)
        .def_ro("index", &Peak::index);
    wf.def("peak_abs", &peak_abs, nb::arg("x"), "Peak of one channel: max |x| and its first index.");
    wf.def("vector_sum_peak", &vector_sum_peak, nb::arg("t"), nb::arg("v"), nb::arg("l"),
           "Largest sqrt(t^2 + v^2 + l^2) computed sample by sample (true vector sum) and its index.");
    wf.def("pseudo_vector_sum", &pseudo_vector_sum, nb::arg("peak_t"), nb::arg("peak_v"), nb::arg("peak_l"),
           "Pseudo vector sum sqrt(peak_t^2 + peak_v^2 + peak_l^2); an upper bound of the true vector sum.");
}
