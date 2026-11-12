/**
 * @file bindings.cpp
 * @brief Único punto de exposición del núcleo C++ a Python (módulo blastingpy._blastingpy_core).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blastingpy/method_info.hpp"
#include "blastingpy/units.hpp"

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
}
