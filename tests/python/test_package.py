import blastingpy


def test_version_and_core():
    assert blastingpy.__version__ == "0.1.0"
    info = blastingpy.MethodInfo("X", blastingpy.Verification.Secondary, "src")
    assert info.method_id == "X" and info.verification == blastingpy.Verification.Secondary


MODULES = ("units", "events", "attenuation", "criteria", "datasets", "plotting", "io", "waveform", "airblast",
           "flyrock", "nearfield")


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
                               "read_instantel_histogram", "read_waveform_csv", "read_obspy"}
    assert {"Candidate", "Criterion", "candidates"} <= set(blastingpy.criteria.__all__)
