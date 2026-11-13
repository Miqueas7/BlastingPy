import dataclasses
from datetime import datetime

import numpy as np
import pytest

import blastingpy
from blastingpy.io import Record, _make_record, from_arrays, infer_units

FS = 1024.0
STEP_G = 0.00788   # resolución del geófono en mm/s
STEP_M = 0.0156    # resolución del micrófono en Pa
G = {"geophone": "mm/s"}
GM = {"geophone": "mm/s", "mic": "Pa"}


def _wave(step, n=16, seed=0, scale=1):
    # múltiplos del paso, con la muestra 0 exactamente en un paso (el menor |x| distinto de cero)
    rng = np.random.default_rng(seed)
    x = rng.integers(2, 8, n).astype(float) * rng.choice([-1, 1], n) * step * scale
    x[0], x[1] = step, 0.0
    return x


def test_exported_as_submodule():
    import io as stdlib_io
    assert blastingpy.io.Record is Record
    assert hasattr(stdlib_io, "BytesIO")


def test_aliases_and_canonical_names():
    x = np.arange(8.0)
    r = from_arrays({"Tran": x, "Vert": x, "Long": x, "MicL": x}, FS, units=GM)
    assert list(r.channels) == ["transverse", "vertical", "longitudinal", "mic"]
    r = from_arrays({" tran 1": x, "VERT1": x, "long1": x, "Mic1": x}, FS, units=GM)
    assert list(r.channels) == ["transverse", "vertical", "longitudinal", "mic"]
    r = from_arrays({"transverse": x, "mic": x}, FS, units=GM)
    assert list(r.channels) == ["transverse", "mic"]
    r = from_arrays({"MicL1": x}, FS, units={"mic": "Pa"})
    assert list(r.channels) == ["mic"]


def test_unit_conversion_applied_once():
    x = np.array([0.0, 1.0, -2.0, 4.0])
    r = from_arrays({"Tran": x, "MicL": x}, FS, units={"geophone": "in/s", "mic": "psi"})
    np.testing.assert_allclose(r.channels["transverse"], x * 25.4)
    np.testing.assert_allclose(r.channels["mic"], x * 6894.757)
    r = from_arrays({"Tran": x, "MicL": x}, FS, units={"geophone": "MM/S.", "mic": "pa."})
    np.testing.assert_array_equal(r.channels["transverse"], x)
    np.testing.assert_array_equal(r.channels["mic"], x)


def test_missing_key_is_inferred():
    g = _wave(STEP_G)
    m = _wave(STEP_M)
    r = from_arrays({"Tran": g, "MicL": m}, FS, units=G)
    assert any("mic" in w and "0.0156" in w for w in r.warnings)
    assert not any("geophone" in w and "inferred" in w for w in r.warnings)


def test_infer_units_mm_s_and_pa():
    units, warns = infer_units({"transverse": _wave(STEP_G), "vertical": _wave(STEP_G, seed=1, scale=3),
                                "mic": _wave(STEP_M, scale=7)})
    assert units == {"geophone": "mm/s", "mic": "Pa"}
    assert len(warns) == 2
    assert all("Micromate" in w and "Minimate Pro" in w for w in warns)
    assert any("0.00788" in w for w in warns) and any("0.0156" in w for w in warns)


def test_infer_units_in_s_and_tolerance():
    units, _ = infer_units({"vertical": _wave(0.00031, scale=2)})
    assert units == {"geophone": "in/s"}
    units, _ = infer_units({"vertical": _wave(STEP_G * 1.029)})
    assert units == {"geophone": "mm/s"}
    with pytest.raises(ValueError, match="units="):
        infer_units({"vertical": _wave(STEP_G * 1.05)})


def test_infer_units_pooled_geophones_use_smallest():
    units, _ = infer_units({"transverse": _wave(STEP_G, scale=5), "vertical": _wave(STEP_G)})
    assert units["geophone"] == "mm/s"


def test_infer_units_not_a_multiple_rule():
    # un paso de 0.0159 mm/s (Series IV) es casi 2 x 0.00788: no debe pasar como mm/s
    with pytest.raises(ValueError, match="units="):
        infer_units({"transverse": _wave(0.0159)})
    with pytest.raises(ValueError, match="units="):
        infer_units({"mic": _wave(STEP_M * 2)})


def test_infer_units_all_zero_or_no_match():
    with pytest.raises(ValueError, match="units="):
        infer_units({"transverse": np.zeros(8)})
    with pytest.raises(ValueError, match="units="):
        infer_units({"transverse": np.array([0.0, 0.5, 1.0, 1.5])})


def test_from_arrays_without_units_infers_and_warns():
    r = from_arrays({"Tran": _wave(STEP_G), "MicL": _wave(STEP_M)}, FS, pretrigger_s=0.25)
    assert len(r.warnings) == 2
    with pytest.raises(ValueError, match="units="):
        from_arrays({"Tran": np.array([0.0, 0.5, 1.0])}, FS)


def test_mica_dropped_with_warning():
    x = np.arange(8.0)
    r = from_arrays({"Tran": x, "MicA": x, "Mic A": x * 0}, FS, units=G)
    assert list(r.channels) == ["transverse"]
    assert sum("A-weighted" in w for w in r.warnings) == 2


def test_mic_db_dropped_with_warning():
    x = np.arange(8.0)
    r = from_arrays({"Tran": x, "MicL": x}, FS, units={"geophone": "mm/s", "mic": "dB"})
    assert list(r.channels) == ["transverse"]
    assert any("dB" in w for w in r.warnings)


def test_all_channels_dropped_is_error():
    with pytest.raises(ValueError, match="channels"):
        from_arrays({"MicA": np.arange(8.0)}, FS)


def test_second_sensor_rejected():
    x = np.arange(8.0)
    for name in ("Tran2", "Vert2", "Long2", "MicL2", "Mic2"):
        with pytest.raises(ValueError, match="second"):
            from_arrays({"Tran": x, name: x}, FS, units=GM)


def test_unknown_and_duplicate_names():
    x = np.arange(8.0)
    with pytest.raises(ValueError, match="Foo") as e:
        from_arrays({"Foo": x}, FS, units=G)
    assert "transverse" in str(e.value)
    with pytest.raises(ValueError, match="same channel"):
        from_arrays({"Tran": x, "transverse": x}, FS, units=G)


@pytest.mark.parametrize("bad", [
    {"Tran": np.arange(8.0), "Vert": np.arange(7.0)},
    {"Tran": np.array([1.0])},
    {"Tran": np.array([1.0, np.nan, 2.0])},
    {"Tran": np.array([1.0, np.inf, 2.0])},
    {},
])
def test_invalid_channels(bad):
    with pytest.raises(ValueError, match=r"^Domain error \[BlastingPy\]: parameter 'channels'"):
        from_arrays(bad, FS, units=G)


@pytest.mark.parametrize("fs", [0, -1.0, float("nan"), float("inf")])
def test_invalid_sample_rate(fs):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        from_arrays({"Tran": np.arange(8.0)}, fs, units=G)


def test_invalid_units_and_pretrigger():
    x = np.arange(8.0)
    with pytest.raises(ValueError, match="units"):
        from_arrays({"Tran": x}, FS, units={"geophone": "km/h"})
    with pytest.raises(ValueError, match="units"):
        from_arrays({"Tran": x}, FS, units={"mic": "Pa", "foo": "x"})
    with pytest.raises(ValueError, match="pretrigger_s"):
        from_arrays({"Tran": x}, FS, units=G, pretrigger_s=-0.1)


def test_time_axis_and_pretrigger_warning():
    x = np.arange(8.0)
    r = from_arrays({"Tran": x}, 4.0, units=G, pretrigger_s=0.5)
    np.testing.assert_allclose(r.time_s(), np.arange(8) / 4.0 - 0.5)
    assert r.n_samples == 8 and not any("time starts at 0" in w for w in r.warnings)
    r = from_arrays({"Tran": x}, 4.0, units=G)
    np.testing.assert_allclose(r.time_s(), np.arange(8) / 4.0)
    assert r.pretrigger_s is None and any("time starts at 0" in w for w in r.warnings)


def test_read_only_and_user_array_untouched():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    keep = x.copy()
    r = from_arrays({"Tran": x}, FS, units={"geophone": "in/s"})
    assert not r.channels["transverse"].flags.writeable
    with pytest.raises(ValueError):
        r.channels["transverse"][0] = 9.0
    with pytest.raises(TypeError):
        r.channels["mic"] = x
    np.testing.assert_array_equal(x, keep)
    assert x.flags.writeable
    x[0] = 99.0  # el arreglo del usuario sigue siendo suyo y no afecta al registro
    assert r.channels["transverse"][0] == pytest.approx(25.4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.sample_rate_hz = 1.0


def test_metadata_and_start():
    t0 = datetime(2026, 1, 2, 3, 4, 5)
    md = {"unit": "A"}
    r = from_arrays({"Tran": np.arange(8.0)}, FS, units=G, start=t0, metadata=md)
    md["unit"] = "B"
    assert r.metadata == {"unit": "A"} and r.start == t0
    assert r.source == "arrays" and r.format_evidence == "user"
    with pytest.raises(TypeError):
        r.metadata["x"] = "y"
    with pytest.raises(ValueError, match="start"):
        from_arrays({"Tran": np.arange(8.0)}, FS, units=G, start="2026-01-01")


def test_make_record_caller_warnings_first_and_vocab():
    r = _make_record({"Tran": np.arange(8.0)}, FS, units=G, pretrigger_s=None,
                     start=None, metadata=None, source="csv", format_evidence="manual-only",
                     warnings=["caller note"])
    assert r.warnings[0] == "caller note" and len(r.warnings) == 2
    assert r.source == "csv" and r.format_evidence == "manual-only"
    kw = dict(units=G, pretrigger_s=0.0, start=None, metadata=None, warnings=[])
    with pytest.raises(ValueError, match="source"):
        _make_record({"Tran": np.arange(8.0)}, FS, source="xls", format_evidence="user", **kw)
    with pytest.raises(ValueError, match="format_evidence"):
        _make_record({"Tran": np.arange(8.0)}, FS, source="csv", format_evidence="guess", **kw)
