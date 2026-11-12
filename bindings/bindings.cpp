/**
 * @file bindings.cpp
 * @brief Único punto de exposición del núcleo C++ a Python (módulo blastingpy._blastingpy_core).
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "blastingpy/method_info.hpp"

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
}
