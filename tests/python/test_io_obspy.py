import sys
from datetime import datetime

import numpy as np
import pytest

from blastingpy.io import read_obspy

obspy = pytest.importorskip("obspy")

FS = 500.0
T0 = obspy.UTCDateTime(2026, 3, 4, 10, 20, 30)
CODES = {"transverse": "HHT", "vertical": "HHZ", "longitudinal": "HHL"}


def _signals(n=256):
    t = np.arange(n) / FS
    return {"transverse": 10 * np.sin(2 * np.pi * 20 * t), "vertical": 5 * np.cos(2 * np.pi * 35 * t),
            "longitudinal": np.linspace(-3, 3, n)}


def _stream(data, fs=FS, starts=None):
    traces = []
    for name, arr in data.items():
        tr = obspy.Trace(data=arr.astype(np.float32))
        tr.stats.sampling_rate = fs
        tr.stats.starttime = T0 if starts is None else starts.get(name, T0)
        tr.stats.channel = CODES[name]
        tr.stats.network, tr.stats.station = "XX", "SYN"
        traces.append(tr)
    return obspy.Stream(traces)


def _write(tmp_path, fmt, data=None, **kw):
    data = _signals() if data is None else data
    path = tmp_path / ("rec.mseed" if fmt == "MSEED" else "rec.sac")
    st = _stream(data, **kw)
    if fmt == "SAC":  # SAC guarda un solo traza por archivo: se escriben tres y se lee por comodín
        for tr in st:
            obspy.Stream([tr]).write(str(tmp_path / f"rec_{tr.stats.channel}.sac"), format="SAC")
        return tmp_path / "rec_*.sac", data
    st.write(str(path), format=fmt, encoding="FLOAT32" if fmt == "MSEED" else None)
    return path, data


# ObsPy avisa del redondeo del paso de muestreo al leer SAC; es de la biblioteca, no del lector
@pytest.mark.filterwarnings("ignore:Sample spacing read from SAC file")
@pytest.mark.parametrize("fmt", ["MSEED", "SAC"])
def test_roundtrip(tmp_path, fmt):
    path, data = _write(tmp_path, fmt)
    rec = read_obspy(path, channels=CODES, units={"geophone": "mm/s"})
    assert set(rec.channels) == set(CODES)
    for name, arr in data.items():
        np.testing.assert_allclose(rec.channels[name], arr.astype(np.float32), rtol=1e-6, atol=0)
    assert rec.sample_rate_hz == pytest.approx(FS)
    assert rec.start == datetime(2026, 3, 4, 10, 20, 30) and rec.start.tzinfo is None
    assert rec.source == "obspy" and rec.format_evidence == "user"


def test_counts_per_unit_and_in_per_s(tmp_path):
    path, data = _write(tmp_path, "MSEED")
    rec = read_obspy(path, channels=CODES, units={"geophone": "in/s"},
                     counts_per_unit={"transverse": 2.0, "vertical": 4.0, "longitudinal": 1.0})
    np.testing.assert_allclose(rec.channels["transverse"], data["transverse"].astype(np.float32) / 2 * 25.4,
                               rtol=1e-6)
    np.testing.assert_allclose(rec.channels["vertical"], data["vertical"].astype(np.float32) / 4 * 25.4, rtol=1e-6)


def test_counts_per_unit_must_be_positive(tmp_path):
    path, _ = _write(tmp_path, "MSEED")
    with pytest.raises(ValueError, match="counts_per_unit"):
        read_obspy(path, channels=CODES, units={"geophone": "mm/s"}, counts_per_unit={"transverse": 0})
    with pytest.raises(ValueError, match="counts_per_unit"):
        read_obspy(path, channels=CODES, units={"geophone": "mm/s"}, counts_per_unit={"nope": 1.0})


def test_pretrigger_passed_through(tmp_path):
    path, _ = _write(tmp_path, "MSEED")
    rec = read_obspy(path, channels=CODES, units={"geophone": "mm/s"}, pretrigger_s=0.1)
    assert rec.pretrigger_s == 0.1


def test_different_sample_rate_is_error(tmp_path):
    st = _stream(_signals())
    st[1].stats.sampling_rate = 250.0
    path = tmp_path / "mixed.mseed"
    st.write(str(path), format="MSEED", encoding="FLOAT32")
    with pytest.raises(ValueError, match="sample rate"):
        read_obspy(path, channels=CODES, units={"geophone": "mm/s"})


def test_different_start_is_error(tmp_path):
    path, _ = _write(tmp_path, "MSEED", starts={"vertical": T0 + 1.0})
    with pytest.raises(ValueError, match="start"):
        read_obspy(path, channels=CODES, units={"geophone": "mm/s"})


def test_start_within_half_sample_is_accepted(tmp_path):
    path, _ = _write(tmp_path, "MSEED", starts={"vertical": T0 + 0.4 / FS})
    rec = read_obspy(path, channels=CODES, units={"geophone": "mm/s"})
    assert set(rec.channels) == set(CODES)


def test_different_length_is_error(tmp_path):
    data = _signals()
    data["vertical"] = data["vertical"][:200]
    path, _ = _write(tmp_path, "MSEED", data=data)
    with pytest.raises(ValueError, match="length"):
        read_obspy(path, channels=CODES, units={"geophone": "mm/s"})


def test_channel_code_absent_or_ambiguous(tmp_path):
    path, _ = _write(tmp_path, "MSEED")
    with pytest.raises(ValueError, match="channels"):
        read_obspy(path, channels={**CODES, "mic": "HDF"}, units={"geophone": "mm/s", "mic": "Pa"})
    st = _stream(_signals())
    st[2].stats.channel = "HHZ"
    dup = tmp_path / "dup.mseed"
    st.write(str(dup), format="MSEED", encoding="FLOAT32")
    with pytest.raises(ValueError, match="channels"):
        read_obspy(dup, channels=CODES, units={"geophone": "mm/s"})


def test_unknown_canonical_name(tmp_path):
    path, _ = _write(tmp_path, "MSEED")
    with pytest.raises(ValueError, match="channels"):
        read_obspy(path, channels={"east": "HHT"}, units={"geophone": "mm/s"})


def test_missing_obspy_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "obspy", None)
    with pytest.raises(ImportError, match=r"pip install blastingpy\[seismic\]"):
        read_obspy(tmp_path / "x.mseed", channels=CODES, units={"geophone": "mm/s"})


def test_import_blastingpy_without_obspy():
    import subprocess
    code = "import sys; sys.modules['obspy'] = None; import blastingpy; blastingpy.io.read_obspy"
    subprocess.run([sys.executable, "-c", code], check=True)
