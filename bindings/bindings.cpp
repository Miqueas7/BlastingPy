/**
 * @file bindings.cpp
 * @brief Único punto de exposición del núcleo C++ a Python (módulo blastingpy._blastingpy_core).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "blastingpy/airblast.hpp"
#include "blastingpy/attenuation.hpp"
#include "blastingpy/criteria_curve.hpp"
#include "blastingpy/flyrock.hpp"
#include "blastingpy/least_squares.hpp"
#include "blastingpy/method_info.hpp"
#include "blastingpy/nearfield.hpp"
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
    u.attr("pa_per_psi") = units::pa_per_psi;
    u.def("scaled_distance", &units::scaled_distance, nb::arg("distance_m"), nb::arg("charge_kg"),
          nb::arg("charge_exponent"), "Scaled distance D / W**e (distance in m, charge in kg).");
    u.def("convert_k_to_si", &units::convert_k_to_si, nb::arg("k_imperial"), nb::arg("beta"),
          nb::arg("charge_exponent"),
          "Convert K of v[in/s] = K (D[ft]/W[lb]**e)**-beta to the K of v[mm/s] with D in m and W in kg.");

    nb::module_ ab = m.def_submodule("_airblast", "Airblast: Pa <-> dB and psi-ft-lb coefficient conversion.");
    ab.attr("reference_pressure_pa") = airblast::reference_pressure_pa;
    ab.def("pa_to_db", &airblast::pa_to_db, nb::arg("pressure_pa"), "Level in dB re 20 uPa: 20 log10(p/p0).");
    ab.def("db_to_pa", &airblast::db_to_pa, nb::arg("level_db"), "Pressure in Pa from a level in dB re 20 uPa.");
    ab.def("convert_k_to_si", &airblast::convert_k_to_si, nb::arg("k_psi"), nb::arg("beta"),
           "Convert K of AB[psi] = K (D[ft]/W[lb]**(1/3))**-beta to the K of AB[Pa] with D in m and W in kg.");

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
    nb::enum_<Quantity>(at, "Quantity", "Quantity a site law is fitted on (names, warnings and citation).")
        .value("Ppv", Quantity::Ppv, "peak particle velocity, mm/s")
        .value("Airblast", Quantity::Airblast, "peak overpressure, Pa (cube-root law of USBM RI 8485)");
    at.def("fit_site_law", &fit_site_law, nb::arg("distance_m"), nb::arg("charge_kg"), nb::arg("ppv_mm_s"),
           nb::arg("law"), nb::arg("z") = 2.0, nb::arg("outlier_z") = 3.0, nb::arg("quantity") = Quantity::Ppv,
           "Fit the site law by least squares in log10; the upper bound is k * 10**(z * sigma).");
    at.def("make_site_law", &make_site_law, nb::arg("law"), nb::arg("k"), nb::arg("beta"),
           nb::arg("charge_exponent") = validation::unset, nb::arg("sigma") = 0.0, nb::arg("z") = 2.0,
           nb::arg("info") = nb::none(),
           "User-defined site law (no fit); charge_exponent is required for Law.General. `info` replaces the "
           "citation (default: the law's own, marked user-defined).");

    nb::enum_<Level>(at, "Level", "Which constant of the law a prediction uses.")
        .value("Median", Level::Median, "k: the median of the fit")
        .value("Upper", Level::Upper, "k_upper = k * 10**(z * sigma)");
    nb::class_<Prediction>(at, "Prediction", "A value and its warnings (extrapolation; upper bound with no margin).")
        .def(nb::init<double, std::vector<std::string>>(), nb::arg("value"), nb::arg("warnings"))
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

    nb::module_ fl = m.def_submodule(
        "_flyrock", "Flyrock: drag-free ballistics, Richards & Moore (face burst, cratering, stemming ejection), "
                    "Lundborg, McKenzie 2018 and Roth (Gurney and calibrated equations).");
    fl.attr("standard_gravity") = flyrock::standard_gravity;
    const double g0 = flyrock::standard_gravity;
    fl.def("range_level", &flyrock::range_level, nb::arg("velocity_m_s"), nb::arg("launch_angle_deg"),
           nb::arg("g") = g0, "Range on level ground: v^2 sin(2 theta) / g (angle from the horizontal).");
    fl.def("max_range_level", &flyrock::max_range_level, nb::arg("velocity_m_s"), nb::arg("g") = g0,
           "Maximum range on level ground (45 degrees): v^2 / g.");
    fl.def("range_to_height", &flyrock::range_to_height, nb::arg("velocity_m_s"), nb::arg("launch_angle_deg"),
           nb::arg("height_m"), nb::arg("g") = g0,
           "Horizontal range to a point height_m above the launch point (negative = below).");
    fl.def("max_height", &flyrock::max_height, nb::arg("velocity_m_s"), nb::arg("launch_angle_deg"),
           nb::arg("g") = g0, "Peak height above the launch point: v^2 sin^2(theta) / (2 g).");
    fl.def("velocity_for_range", &flyrock::velocity_for_range, nb::arg("range_m"), nb::arg("launch_angle_deg"),
           nb::arg("g") = g0, "Launch velocity that reaches range_m on level ground.");
    fl.def("rm_launch_velocity", &flyrock::rm_launch_velocity, nb::arg("k"), nb::arg("charge_kg_m"),
           nb::arg("confinement_m"), "Richards & Moore face velocity: k (sqrt(m)/B)^1.3.");
    fl.def("rm_face_burst", &flyrock::rm_face_burst, nb::arg("k"), nb::arg("charge_kg_m"), nb::arg("burden_m"),
           nb::arg("g") = g0, "Richards & Moore face-burst maximum range (k^2/g)(sqrt(m)/B)^2.6.");
    fl.def("rm_cratering", &flyrock::rm_cratering, nb::arg("k"), nb::arg("charge_kg_m"), nb::arg("stemming_m"),
           nb::arg("g") = g0, "Richards & Moore cratering maximum range (k^2/g)(sqrt(m)/SH)^2.6.");
    fl.def("rm_stemming_ejection", &flyrock::rm_stemming_ejection, nb::arg("k"), nb::arg("charge_kg_m"),
           nb::arg("stemming_m"), nb::arg("hole_angle_from_vertical_deg"), nb::arg("g") = g0,
           "Richards & Moore stemming-ejection range: cratering * sin(2 theta); 0 for a vertical hole.");
    fl.def("rm_min_confinement", &flyrock::rm_min_confinement, nb::arg("k"), nb::arg("charge_kg_m"),
           nb::arg("range_m"), nb::arg("g") = g0, "Burden or stemming that limits the maximum range to range_m.");
    fl.def("lundborg_max_range", &flyrock::lundborg_max_range, nb::arg("diameter_mm"),
           "Lundborg maximum range 260 d^(2/3) in m (d in inches in the source; here mm, 20 to 500). Secondary source.");
    fl.def("lundborg_optimal_fragment", &flyrock::lundborg_optimal_fragment, nb::arg("diameter_mm"),
           "Lundborg optimal rock size 0.1 d^(2/3) in m (hole diameter in mm, 20 to 500). Secondary source.");
    fl.def("lundborg_range", &flyrock::lundborg_range, nb::arg("diameter_mm"), nb::arg("powder_factor_kg_m3"),
           "Lundborg range 143 d (q - 0.2) in m; q must exceed 0.2 kg/m3 (inferred: the printed condition is a typo).");
    fl.def("mckenzie_max_range", &flyrock::mckenzie_max_range, nb::arg("sdob_m_kg13"), nb::arg("diameter_mm"),
           "McKenzie (2018) maximum range 10 SDoB^-2.167 d^0.667 in m; SDoB (m/kg^(1/3)) is supplied by the user "
           "(its definition is not in any source read).");
    nb::enum_<flyrock::Explosive>(fl, "Explosive", "Explosive family for the Gurney velocity.")
        .value("Anfo", flyrock::Explosive::Anfo, "sqrt(2E') = 0.44 D")
        .value("Other", flyrock::Explosive::Other, "sqrt(2E') = D / 3");
    nb::enum_<flyrock::RothCalibration>(fl, "RothCalibration", "Roth (1979) calibrated velocity equations.")
        .value("Granite", flyrock::RothCalibration::Granite, "eq. (13): 17 data, normalized to 2300 m/s")
        .value("GraniteAllData", flyrock::RothCalibration::GraniteAllData,
               "eq. (13a): the line Roth calls 'based on all data', yet it leaves out the data of two references and "
               "two points of a third one in his Table 2 (eq. (13) omits three points); r = 0.971; the source does "
               "not say which Gurney velocity it was normalized to")
        .value("DolomiteLimestone", flyrock::RothCalibration::DolomiteLimestone,
               "eq. (14): 'at most an approximation', normalized to 1880 m/s")
        .value("SandstoneCrater", flyrock::RothCalibration::SandstoneCrater, "eq. (17): normalized to 1750 m/s");
    fl.def("roth_charge_mass_ratio", &flyrock::roth_charge_mass_ratio, nb::arg("linear_charge_kg_m"),
           nb::arg("rock_density_kg_m3"), nb::arg("burden_m"), nb::arg("breakout_angle_deg") = 90.0,
           "Roth eq. (7): c/m = (W/l) / (rho b^2 tan(alpha/2)).");
    fl.def("roth_gurney_velocity", &flyrock::roth_gurney_velocity, nb::arg("detonation_velocity_m_s"),
           nb::arg("explosive"), "Roth eqs. (9)-(10): 0.44 D (ANFO) or D / 3.");
    fl.def("roth_launch_velocity", &flyrock::roth_launch_velocity, nb::arg("gurney_velocity_m_s"),
           nb::arg("charge_mass_ratio"), "Roth eq. (8): sqrt(2E') sqrt(c/m).");
    fl.def("roth_calibrated_velocity", &flyrock::roth_calibrated_velocity, nb::arg("rock"),
           nb::arg("charge_mass_ratio"),
           "Roth calibrated velocity sqrt(a c/m - b); valid for roth_reference_gurney_velocity(rock). "
           "Raises if c/m is at or below b/a. Near b/a the fit predicts near-zero velocities while Roth's data show "
           "finite ones (p. 21): it underestimates there.");
    fl.def("roth_minimum_charge_mass_ratio", &flyrock::roth_minimum_charge_mass_ratio, nb::arg("rock"),
           "Minimum c/m of a Roth calibration, b/a, where the fitted velocity is zero.");
    fl.def("roth_reference_gurney_velocity", &flyrock::roth_reference_gurney_velocity, nb::arg("rock"),
           "Gurney velocity (m/s) each calibration was normalized with; NaN for GraniteAllData (not stated).");
    nb::enum_<flyrock::Model>(fl, "Model", "Flyrock model, for info().")
        .value("Ballistics", flyrock::Model::Ballistics)
        .value("RichardsMooreFaceBurst", flyrock::Model::RichardsMooreFaceBurst)
        .value("RichardsMooreCratering", flyrock::Model::RichardsMooreCratering)
        .value("RichardsMooreStemmingEjection", flyrock::Model::RichardsMooreStemmingEjection)
        .value("LundborgMaxRange", flyrock::Model::LundborgMaxRange)
        .value("LundborgRange", flyrock::Model::LundborgRange)
        .value("McKenzie2018", flyrock::Model::McKenzie2018)
        .value("RothGurney", flyrock::Model::RothGurney)
        .value("RothCalibrated", flyrock::Model::RothCalibrated);
    fl.def("info", &flyrock::info, nb::arg("model"), "Method id, verification mark and source of a flyrock model.");

    nb::module_ wf = m.def_submodule("_waveform", "Waveform peaks and vector sum.");
    nb::class_<Peak>(wf, "Peak", "Largest |x| (value >= 0) and the index of its first occurrence.")
        .def_ro("value", &Peak::value)
        .def_ro("index", &Peak::index);
    wf.def("peak_abs", &peak_abs, nb::arg("x"), "Peak of one channel: max |x| and its first index.");
    wf.def("vector_sum_peak", &vector_sum_peak, nb::arg("t"), nb::arg("v"), nb::arg("l"),
           "Largest sqrt(t^2 + v^2 + l^2) computed sample by sample (true vector sum) and its index.");
    wf.def("pseudo_vector_sum", &pseudo_vector_sum, nb::arg("peak_t"), nb::arg("peak_v"), nb::arg("peak_l"),
           "Pseudo vector sum sqrt(peak_t^2 + peak_v^2 + peak_l^2); an upper bound of the true vector sum.");
    nb::enum_<ZcMethod>(wf, "ZcMethod", "How the half period is measured.")
        .value("WholeSamples", ZcMethod::WholeSamples, "Count whole samples of the peak's sign (Instantel-like)")
        .value("Interpolated", ZcMethod::Interpolated, "Linearly interpolate both zero crossings (no limit)");
    nb::enum_<ZcStatus>(wf, "ZcStatus", "Outcome of the zero-crossing frequency.")
        .value("Ok", ZcStatus::Ok)
        .value("AboveLimit", ZcStatus::AboveLimit, "Whole-sample frequency above the Instantel limit")
        .value("Incomplete", ZcStatus::Incomplete, "A zero crossing is missing (record cut); values are NaN")
        .value("NotComputable", ZcStatus::NotComputable, "The peak sample is zero; values are NaN");
    nb::class_<ZcFrequency>(wf, "ZcFrequency", "Zero-crossing frequency (Hz), half period (s) and status.")
        .def_ro("frequency_hz", &ZcFrequency::frequency_hz)
        .def_ro("half_period_s", &ZcFrequency::half_period_s)
        .def_ro("status", &ZcFrequency::status);
    wf.def("instantel_zc_limit_hz", &instantel_zc_limit_hz, nb::arg("sample_rate_hz"),
           "Instantel display limit of the zero-crossing frequency: 100 * fs / 1024 Hz.");
    wf.def("zero_crossing_frequency", &zero_crossing_frequency, nb::arg("x"), nb::arg("sample_rate_hz"),
           nb::arg("peak_index"), nb::arg("method"),
           "Frequency 1 / (2 T) of the half cycle containing x[peak_index], between its zero crossings.");
    nb::enum_<DiffMethod>(wf, "DiffMethod", "How velocity is differentiated into acceleration.")
        .value("Forward", DiffMethod::Forward, "x[i+1] - x[i] over dt")
        .value("Central", DiffMethod::Central, "(x[i+1] - x[i-1]) over 2 dt; one-sided at the ends");
    wf.attr("standard_gravity_mm_s2") = standard_gravity_mm_s2;
    wf.def("peak_acceleration", &peak_acceleration, nb::arg("velocity_mm_s"), nb::arg("sample_rate_hz"),
           nb::arg("method"), "Peak |dV/dt| (mm/s^2) of a velocity record (mm/s) and its sample index.");
    wf.def("peak_displacement", &peak_displacement, nb::arg("velocity_mm_s"), nb::arg("sample_rate_hz"),
           "Peak displacement (mm): largest complete half-cycle sum(V dt), halved; value is NaN if none is complete.");

    nb::module_ nfd = m.def_submodule(
        "_nearfield", "Near field: Holmberg-Persson (original integral and NIOSH modified form), damage extent and "
                      "critical PPV. PPV in the unit K carries (mm/s with the published constants, kg and m).");
    nb::enum_<nearfield::HpModel>(nfd, "HpModel", "Holmberg-Persson form.")
        .value("Original", nearfield::HpModel::Original, "integral with alpha outside (NIOSH eq. 5)")
        .value("Niosh", nearfield::HpModel::Niosh, "K (q L)^alpha / mean distance^beta (NIOSH eq. 8)");
    nfd.def("hp_point", &nearfield::hp_point, nb::arg("k"), nb::arg("alpha"), nb::arg("beta"), nb::arg("charge_kg"),
            nb::arg("distance_m"), "K Q^alpha / R^beta (NIOSH eq. 1).");
    nfd.def("hp_term", &nearfield::hp_term, nb::arg("linear_charge_kg_m"), nb::arg("z_start"), nb::arg("z_end"),
            nb::arg("r"), nb::arg("z"),
            "Closed form (l/r)[atan((z_end - z)/r) + atan((z - z_start)/r)], valid for beta = 2 alpha.");
    nfd.def("hp_original", &nearfield::hp_original, nb::arg("k"), nb::arg("alpha"), nb::arg("beta"),
            nb::arg("linear_charge_kg_m"), nb::arg("z_start"), nb::arg("z_end"), nb::arg("r"), nb::arg("z"),
            "K [l int dz' / (r^2 + (z' - z)^2)^(beta / (2 alpha))]^alpha (NIOSH eq. 5). "
            "Raises on the axis inside the charged span.");
    nfd.def("hp_mean_distance", &nearfield::hp_mean_distance, nb::arg("z_start"), nb::arg("z_end"), nb::arg("r"),
            nb::arg("z"), "NIOSH mean distance (eq. 9).");
    nfd.def("hp_niosh", &nearfield::hp_niosh, nb::arg("k"), nb::arg("alpha"), nb::arg("beta"),
            nb::arg("linear_charge_kg_m"), nb::arg("z_start"), nb::arg("z_end"), nb::arg("r"), nb::arg("z"),
            "K (l L)^alpha / mean distance^beta (NIOSH eq. 8).");
    nfd.def("hp_damage_extent", &nearfield::hp_damage_extent, nb::arg("model"), nb::arg("k"), nb::arg("alpha"),
            nb::arg("beta"), nb::arg("linear_charge_kg_m"), nb::arg("z_start"), nb::arg("z_end"), nb::arg("z"),
            nb::arg("ppv_limit_mm_s"),
            "Radial distance where the PPV falls to the limit at height z; NaN if it never does.");
    nfd.def("ppv_critical", &nearfield::ppv_critical, nb::arg("tensile_strength_pa"), nb::arg("p_velocity_m_s"),
            nb::arg("youngs_modulus_pa"),
            "T Vp / E in mm/s, T and E in Pa, Vp in m/s [V2]: Onederra and Esen (2004), Fragblast 8(2), eq. 11, p. 69, "
            "citing Persson, Holmberg and Lee (1994), Rock Blasting and Explosives Engineering, CRC, p. 247, which "
            "was not read. Pass the static E (inferred: it reproduces the source's Table 1).");

    nb::class_<nearfield::Medium>(nfd, "Medium", "Elastic medium: P and S velocities (m/s), density (kg/m3); mu = rho Vs^2.")
        .def(nb::init<double, double, double>(), nb::arg("p_velocity_m_s"), nb::arg("s_velocity_m_s"),
             nb::arg("density_kg_m3"))
        .def_ro("p_velocity_m_s", &nearfield::Medium::p_velocity_m_s)
        .def_ro("s_velocity_m_s", &nearfield::Medium::s_velocity_m_s)
        .def_ro("density_kg_m3", &nearfield::Medium::density_kg_m3);
    nb::class_<nearfield::PressureSource>(
        nfd, "PressureSource",
        "Blair (2007) eq. 8 pressure s(t) = P_VN (e gamma / n)^n t^n exp(-gamma t): von Neumann pressure (Pa), "
        "integer n (2 to 20) and decay gamma (1/s).")
        .def(nb::init<double, int, double>(), nb::arg("vn_pressure_pa"), nb::arg("n"), nb::arg("gamma_per_s"))
        .def_ro("vn_pressure_pa", &nearfield::PressureSource::vn_pressure_pa)
        .def_ro("n", &nearfield::PressureSource::n)
        .def_ro("gamma_per_s", &nearfield::PressureSource::gamma_per_s);
    nb::class_<nearfield::Waveform>(nfd, "Waveform", "Particle velocity (mm/s) in the (radial, axial) frame of the hole.")
        .def(
            "__init__",
            [](nearfield::Waveform* self, std::vector<double> radial_mm_s, std::vector<double> axial_mm_s) {
                validation::require(radial_mm_s.size() == axial_mm_s.size(), "axial_mm_s",
                                    "must have the same length as radial_mm_s",
                                    static_cast<double>(axial_mm_s.size()));
                for (double v : radial_mm_s) validation::require_finite(v, "radial_mm_s");
                for (double v : axial_mm_s) validation::require_finite(v, "axial_mm_s");
                new (self) nearfield::Waveform{std::move(radial_mm_s), std::move(axial_mm_s)};
            },
            nb::arg("radial_mm_s"), nb::arg("axial_mm_s"),
            "A waveform of your own (mm/s), e.g. for vector_peak: two sequences of finite numbers of equal length.")
        .def_ro("radial_mm_s", &nearfield::Waveform::radial_mm_s)
        .def_ro("axial_mm_s", &nearfield::Waveform::axial_mm_s);
    nfd.def("pressure_function", &nearfield::pressure_function, nb::arg("source"), nb::arg("t_s"),
            "Blair (2007) eq. 8 pressure (Pa) at t_s (s); its maximum is P_VN at t = n / gamma; 0 for t < 0.");
    nfd.def("g_function", &nearfield::g_function, nb::arg("n"), nb::arg("tau"),
            "Blair (2007) eq. 15: second derivative of tau^n exp(-tau); 0 for tau < 0.");
    nfd.def("g_peak", &nearfield::g_peak, nb::arg("n"),
            "G_n, the largest POSITIVE value of g(tau). Not the peak of |g|: for n >= 7 the negative trough is larger "
            "in modulus (n = 8: -767.83 vs 680.17), so never use G_n a / r as a PPV; take the peak of the waveform.");
    nfd.def("mean_frequency_factor", &nearfield::mean_frequency_factor, nb::arg("n"),
            "c = omega_A / gamma (Blair 2007 eqs. 10-12; closed form inferred): 0.5, 0.375, 0.3125 for n = 4, 6, 8.");
    nfd.def("scaled_frequency", &nearfield::scaled_frequency, nb::arg("source"), nb::arg("medium"),
            nb::arg("hole_radius_m"), "Omega_A = c gamma a / Vp (Blair 2007); Heelan holds for Omega_A <~ 0.1.");
    nfd.def("heelan_amplitude", &nearfield::heelan_amplitude, nb::arg("source"), nb::arg("medium"),
            nb::arg("hole_radius_m"), nb::arg("half_length_m"),
            "F(gamma) = gamma^2 a L P_VN (e/n)^n / (2 mu Vp) in mm/s (Blair 2007 eq. 16); L is the half length.");
    nfd.def("heelan_element", &nearfield::heelan_element, nb::arg("source"), nb::arg("medium"),
            nb::arg("hole_radius_m"), nb::arg("half_length_m"), nb::arg("r"), nb::arg("z"), nb::arg("t_s"),
            "Velocity (mm/s) radiated by a short charge element centred at the origin (Heelan 1953 with the Blair "
            "2007 source): P along R from R / Vp, SV perpendicular from R / Vs; phi = arccos(-z / R). "
            "`hole_radius_m` is the hole radius (Blair); Heelan defines it as the equivalent cavity radius.");

    nb::class_<nearfield::Column>(
        nfd, "Column",
        "Charge column between z_bottom and z_top (m), split into `elements` equal elements (odd); the primer is "
        "element `primer_element` counted from the bottom (1 = bottom priming); VOD in m/s; charge exponent A in (0, 1].")
        .def(nb::init<double, double, double, int, int, double, double>(), nb::arg("hole_radius_m"),
             nb::arg("z_bottom"), nb::arg("z_top"), nb::arg("elements"), nb::arg("primer_element"), nb::arg("vod_m_s"),
             nb::arg("charge_exponent"))
        .def_ro("hole_radius_m", &nearfield::Column::hole_radius_m)
        .def_ro("z_bottom", &nearfield::Column::z_bottom)
        .def_ro("z_top", &nearfield::Column::z_top)
        .def_ro("elements", &nearfield::Column::elements)
        .def_ro("primer_element", &nearfield::Column::primer_element)
        .def_ro("vod_m_s", &nearfield::Column::vod_m_s)
        .def_ro("charge_exponent", &nearfield::Column::charge_exponent);
    nfd.def("column_weights", &nearfield::column_weights, nb::arg("elements"), nb::arg("primer_element"),
            nb::arg("charge_exponent"),
            "Blair (2008) non-linear superposition weights E_m w_e^(-A), bottom element first (eqs. 1-2 and 22); "
            "they sum to elements^A; A = 1 gives all ones. A must be within (0, 1].");
    nfd.def("heelan_column", &nearfield::heelan_column, nb::arg("source"), nb::arg("medium"), nb::arg("column"),
            nb::arg("r"), nb::arg("z"), nb::arg("t_s"),
            "Velocity (mm/s) of a charge column: Heelan elements initiated from the primer at the VOD, each scaled by "
            "its Blair (2008) weight. Not the Blair and Minchinton (2006) scaled Heelan model. r = 0 is not admitted "
            "(r must be > 0). The waveform exists only at the instants of t_s: if t_s ends before the last arrival the "
            "peak is truncated silently. Last arrival ~ max over the elements of |z_m - z_primer| / VOD + R_m / Vs "
            "(R_m: distance from element m to the point); add the length of the pulse, (n + 4 sqrt(n) + 2) / gamma, as "
            "nearfield.heelan_ppv does.");
    nfd.def("vector_peak", &nearfield::vector_peak, nb::arg("waveform"),
            "Peak vector velocity max sqrt(radial^2 + axial^2) of a waveform (mm/s).");
    nfd.def("superpose", &nearfield::superpose, nb::arg("seed"), nb::arg("sample_rate_hz"), nb::arg("arrival_times_s"),
            nb::arg("amplitudes"),
            "v(t) = sum_j w_j s(t - d_j) (Hinzen 1988 eqs. 3 and 7; Blair 1999 eq. 1a), sampled from t = 0; length = "
            "ceil(max arrival * fs) + len(seed) + 1; an arrival between samples is spread linearly over its neighbours, "
            "and one within 1e-9 samples of a whole sample falls on it (9 * 0.001 s and 0.009 s give the same waveform). "
            "The linear spread is a low-pass filter: an arrival mid-sample lowers the peak of a component of frequency "
            "f by up to sin^2(pi f / fs) (9.5 % at fs / 10, 48 % for 500 Hz at 2048 Hz), so a seed should be sampled "
            "at 10 or more times its dominant frequency (nearfield.simulate_blast warns otherwise). "
            "Empty vectors and a waveform of more than 1e8 samples are domain errors.");
    nfd.def("monte_carlo_peaks", &nearfield::monte_carlo_peaks, nb::arg("seed"), nb::arg("sample_rate_hz"),
            nb::arg("nominal_arrival_times_s"), nb::arg("amplitudes"), nb::arg("delay_sigma_s"), nb::arg("amplitude_cv"),
            nb::arg("simulations"), nb::arg("rng_seed"),
            "Peak |v| of each simulation of the superposed seed (Blair 1999 eq. 1b) with w_jp = w_j (1 + c z') (eq. 7) "
            "and d_jp = d_j + sigma_j z, z and z' standard normal. Eq. 9 only defines the nominal time d_j; the normal "
            "delay scatter is a design choice (the source publishes no delay distribution), supported by Blair (1999, "
            "sec. 5, eqs. 17-19), who models the time of each hole as normal(mu_j, sigma_j). Times are NOT clipped: when "
            "the earliest arrival of a simulation is negative, all its arrivals are shifted by ceil(-min * fs) whole "
            "samples, which leaves the peak unchanged, so the peaks do not depend on the time origin. A negative "
            "amplitude factor is clipped to 0. Domain error if the nominal times plus 10 sigma would need more than 1e8 "
            "samples. Same rng_seed -> same peaks on every platform to about 1e-9 relative, not bit for bit (log, cos "
            "and sin round differently between libraries, and the compiler may contract a * b + c into an FMA).");
    nfd.def("screening_factor", &nearfield::screening_factor, nb::arg("distance_m"), nb::arg("screening_count"),
            nb::arg("g") = 23337.0, nb::arg("floor") = 0.2,
            "S = max(floor, 1 / (1 + g h^-2 sqrt(N_S))) (Blair 1999 eqs. 10b-10c); g in m^2 (inferred). The floor of "
            "0.2 is the author's, for lack of near-field data.");
    nfd.def("screening_counts", &nearfield::screening_counts, nb::arg("x"), nb::arg("y"), nb::arg("firing_times_s"),
            nb::arg("monitor_x"), nb::arg("monitor_y"), nb::arg("spacing_m"),
            "N_S of each hole: holes fired earlier whose centre lies in the rectangle (width = spacing, centred on the "
            "line) between the hole and the monitor (Blair 1999 Fig. 7a). Results in input order. Fired earlier means "
            "t_s < t_n - 1e-9 s (the time tolerance): holes with the same nominal time do not count for each other. The "
            "edges of the rectangle count (a design choice), and a hole beyond the monitor does not (unlike in "
            "damage_weights). Domain error if the monitor coincides with a hole or two holes share a position.");
    nfd.def("charge_scaling_weights", &nearfield::charge_scaling_weights, nb::arg("charges_kg"),
            nb::arg("nearest_fired_distance_m"), nb::arg("spacing_m"), nb::arg("charge_exponent"),
            nb::arg("d_factor") = 1.0, nb::arg("p") = 4,
            "Blair (2008) eq. 10 weights C_n, vectors in initiation order; use inf for the first hole. P = 4 is a design "
            "choice. Holes fired together (within the time tolerance, 1e-9 s) go next to each other in any order, each "
            "with the distance to the nearest hole fired before the tie (inf if none), never to its tie partner.");
    nfd.def("damage_weights", &nearfield::damage_weights, nb::arg("x"), nb::arg("y"), nb::arg("charges_kg"),
            nb::arg("firing_times_s"), nb::arg("monitor_x"), nb::arg("monitor_y"), nb::arg("spacing_m"), nb::arg("eta"),
            nb::arg("g") = 23337.0,
            "Blair (2008) damage model weights E_n = S_n / D_n (eqs. 16 and 20), in plan view, in input order; g in m^2. "
            "Fired earlier means t_j < t_n - 1e-9 s (the time tolerance). Every hole fired earlier with a.b > 0 screens, "
            "also one beyond the monitor (literal from the source; screening_counts excludes it). S_n has no floor "
            "(eq. 20 has none, unlike screening_factor): at near-field distances g / d^2 is huge and one hole on the line "
            "drives the weight to almost 0, an extrapolation without data. Domain error if the monitor coincides with a "
            "hole or two holes share a position.");
    nfd.def("max_charge_per_window", &nearfield::max_charge_per_window, nb::arg("firing_times_s"), nb::arg("charges_kg"),
            nb::arg("window_s") = 0.008,
            "Largest charge initiated within any window [t, t + window - 1e-9 s) starting at a firing time (the 1e-9 s is "
            "the time tolerance: holes nominally one window apart are not added by rounding noise); 8 ms is Duvall et al. "
            "(1963) as reported by Blair (1999). Charges must be > 0 and the vectors not empty.");
    // Con guion bajo: no es un nombre público del núcleo (la API reexporta todos los públicos); lo lee
    // python/blastingpy/nearfield.py para aplicar la misma regla de «ya inició».
    nfd.attr("_time_tolerance_s") = nearfield::time_tolerance_s;
}
