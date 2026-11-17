cmake_minimum_required(VERSION 3.20)
project(blastingpy
    VERSION 0.1.1
    DESCRIPTION "Blast vibration, airblast and flyrock analysis for mining"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Núcleo header-only (interfaz)
add_library(blastingpy_core INTERFACE)
target_include_directories(blastingpy_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(blastingpy_core INTERFACE cxx_std_20)
# MSVC lee el código como UTF-8 (las cabeceras llevan α, ≤ y rayas en comentarios y textos)
target_compile_options(blastingpy_core INTERFACE $<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

# Bindings Python (nanobind vía scikit-build-core)
option(BLASTINGPY_BUILD_PYTHON "Build Python bindings" ON)

if(BLASTINGPY_BUILD_PYTHON)
    find_package(Python 3.10
        REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule
    )

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE NB_DIR
        RESULT_VARIABLE NB_RESULT
    )

    if(NB_RESULT EQUAL 0 AND EXISTS "${NB_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
        find_package(nanobind CONFIG REQUIRED)

        nanobind_add_module(_blastingpy_core
            STABLE_ABI
            NB_STATIC
            bindings/bindings.cpp
        )
        target_link_libraries(_blastingpy_core PRIVATE blastingpy_core)

        install(TARGETS _blastingpy_core
            LIBRARY DESTINATION blastingpy
            RUNTIME DESTINATION blastingpy
            ARCHIVE DESTINATION blastingpy
        )

        # Stubs .pyi para editores y mypy; se generan al instalar (la ruta es relativa al prefijo del wheel).
        # python/blastingpy/py.typed marca el paquete como tipado.
        nanobind_add_stub(_blastingpy_core_stub
            MODULE _blastingpy_core
            OUTPUT blastingpy/_blastingpy_core.pyi
            PYTHON_PATH blastingpy
            INSTALL_TIME
        )
    else()
        message(FATAL_ERROR
            "nanobind not found, so the Python package would be incomplete.\n"
            "Install with: pip install nanobind"
        )
    endif()
endif()

# Pruebas C++ (Google Test)
option(BLASTINGPY_BUILD_TESTS "Build C++ unit tests" OFF)

if(BLASTINGPY_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Cada tarea agrega aquí su archivo de pruebas
    add_executable(blastingpy_tests
        tests/cpp/test_validation.cpp
        tests/cpp/test_units.cpp
        tests/cpp/test_airblast.cpp
        tests/cpp/test_least_squares.cpp
        tests/cpp/test_attenuation_fit.cpp
        tests/cpp/test_attenuation_inverse.cpp
        tests/cpp/test_criteria_curve.cpp
        tests/cpp/test_waveform_peaks.cpp
        tests/cpp/test_waveform_zero_crossing.cpp
        tests/cpp/test_waveform_derived.cpp
        tests/cpp/test_flyrock_ballistics.cpp
        tests/cpp/test_flyrock_models.cpp
        tests/cpp/test_nearfield_hp.cpp
        tests/cpp/test_nearfield_heelan.cpp
        tests/cpp/test_nearfield_column.cpp
        tests/cpp/test_nearfield_superposition.cpp
        tests/cpp/test_nearfield_monte_carlo.cpp
    )
    target_link_libraries(blastingpy_tests PRIVATE
        blastingpy_core
        GTest::gtest_main
    )
    # Avisos del compilador solo en las pruebas (sin tratarlos como errores)
    target_compile_options(blastingpy_tests PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wextra>
    )

    include(GoogleTest)
    gtest_discover_tests(blastingpy_tests)
endif()
