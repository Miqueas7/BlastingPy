import blastingpy


def test_version_and_core():
    assert blastingpy.__version__ == "0.1.1"
    info = blastingpy.MethodInfo("X", blastingpy.Verification.Secondary, "src")
    assert info.method_id == "X" and info.verification == blastingpy.Verification.Secondary


def test_version_is_the_same_everywhere():
    # pyproject, CMake y CITATION.cff se cambian a mano al publicar; esta prueba evita que uno se quede atrás.
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    found = {
        "pyproject.toml": re.search(r'^version = "(.+)"', (root / "pyproject.toml").read_text("utf-8"), re.M),
        "CMakeLists.txt": re.search(r"project\(blastingpy\s+VERSION (\S+)", (root / "CMakeLists.txt").read_text("utf-8")),
        "CITATION.cff": re.search(r'^version: "(.+)"', (root / "CITATION.cff").read_text("utf-8"), re.M),
    }
    assert {k: m and m.group(1) for k, m in found.items()} == dict.fromkeys(found, blastingpy.__version__)


MODULES = ("units", "events", "attenuation", "criteria", "datasets", "plotting", "io", "waveform", "airblast",
           "flyrock", "nearfield", "pattern")


def test_every_module_lists_its_public_names():
    import importlib
    import inspect

    for name in MODULES:
        mod = importlib.import_module(f"blastingpy.{name}")
        public = getattr(mod, "__all__", None)
        assert public is not None, name
        assert [n for n in public if not hasattr(mod, n)] == [], name
        defined = [n for n, obj in vars(mod).items() if not n.startswith("_")
                   and (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == mod.__name__]
        assert [n for n in defined if n not in public] == [], name


def test_io_star_import_brings_only_the_public_names():
    from blastingpy import io

    assert set(io.__all__) == {"Record", "from_arrays", "infer_units", "read_instantel_ascii",
                               "read_instantel_histogram", "read_waveform_csv", "read_obspy", "read_dxf_outline"}
    assert {"Candidate", "Criterion", "candidates"} <= set(blastingpy.criteria.__all__)
