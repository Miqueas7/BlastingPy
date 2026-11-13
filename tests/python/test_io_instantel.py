from datetime import datetime

import numpy as np
import pytest

from blastingpy.io import read_instantel_ascii

FS = 1024.0
N = 64
NAMES = ("Tran", "Vert", "Long", "MicL")
CANON = ("transverse", "vertical", "longitudinal", "mic")

_t = np.arange(N) / FS
# Señal sintética conocida, escrita con 5 decimales (no son datos reales).
SIGNAL = {
    "Tran": np.round(2.5 * np.sin(2 * np.pi * 20 * _t), 5),
    "Vert": np.round(-1.75 * np.sin(2 * np.pi * 35 * _t + 0.3), 5),
    "Long": np.round(0.9 * np.cos(2 * np.pi * 12 * _t), 5),
    "MicL": np.round(31.25 * np.sin(2 * np.pi * 8 * _t), 5),
}

THOR_HEADER = [
    ("EventType", "Full Waveform"),
    ("SampleRate", "1024 sps"),
    ("Units", "mm/s and pa"),
    ("EventTime", "10:20:30"),
    ("EventDate", "2026-01-15"),
    ("Pre-triggerLength", "0.25 sec"),
    ("RecordTime", "1.0 sec"),
]


def _write(tmp_path, text, name="evento.txt", bom=False, crlf=False):
    if crlf:
        text = text.replace("\n", "\r\n")
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _rows(names=NAMES, sep="\t", fmt="{:.5f}", signal=SIGNAL):
    lines = []
    for i in range(N):
        lines.append(sep.join(fmt.format(signal[n][i]) for n in names))
    return "\n".join(lines) + "\n"


def _thor(header=THOR_HEADER, quoted=True, marker=True, names=NAMES, extra=""):
    out = []
    for key, value in header:
        line = f"{key} : {value}"
        out.append(f'"{line}"' if quoted else line)
    if marker:
        out.append("Waveform Data Channels")
    out.append("\t".join(names))
    return "\n".join(out) + "\n" + _rows() + extra  # datos siempre de SIGNAL; `names` solo rotula


def _blastware_header(lines):
    return "\n".join(f'"{x}"' for x in lines) + "\n"


def _assert_signal(rec, names=NAMES, scale=None):
    scale = scale or {}
    for name, canon in zip(names, CANON):
        np.testing.assert_allclose(rec.channels[canon], SIGNAL[name] * scale.get(canon, 1.0), atol=1e-5)


# ---------------------------------------------------------------- THOR


def test_thor_reads_channels_and_header(tmp_path):
    rec = read_instantel_ascii(_write(tmp_path, _thor()))
    _assert_signal(rec)
    assert rec.sample_rate_hz == 1024.0
    assert rec.pretrigger_s == 0.25
    assert rec.start == datetime(2026, 1, 15, 10, 20, 30)
    assert rec.source == "instantel-thor"
    assert rec.format_evidence == "third-party-parser"
    assert rec.metadata["SampleRate"] == "1024 sps"
    assert rec.metadata["Pre-triggerLength"] == "0.25 sec"
    assert any("THOR" in w and "not validated" in w for w in rec.warnings)


def test_thor_tolerant_without_quotes_or_marker(tmp_path):
    header = [("Event Type", "Full Waveform"), ("Sample Rate", "1024 sps"), ("Units", "mm/s and pa"),
              ("Pre-trigger Length", "0.25 sec")]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header, quoted=False, marker=False)))
    _assert_signal(rec)
    assert rec.source == "instantel-thor"
    assert rec.sample_rate_hz == 1024.0
    assert rec.pretrigger_s == 0.25
    assert rec.metadata["Sample Rate"] == "1024 sps"


def test_thor_without_marker_detected_by_two_specific_keys(tmp_path):
    header = [("SampleRate", "1024 sps"), ("Units", "mm/s and pa"), ("Pre-triggerLength", "0.25 sec"),
              ("GeoRange", "254.000 mm/s")]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header, marker=False)))
    assert rec.source == "instantel-thor"
    assert rec.format_evidence == "third-party-parser"
    _assert_signal(rec)


def test_thor_header_numbers_with_comma(tmp_path):
    header = [(k, {"Pre-triggerLength": "0,25 sec", "SampleRate": "1,024 sps"}.get(k, v)) for k, v in THOR_HEADER]
    path = _write(tmp_path, _thor(header))
    with pytest.raises(ValueError, match="sample_rate_hz"):  # «1,024» no se lee como 24 ni como 1024
        read_instantel_ascii(path)
    rec = read_instantel_ascii(path, sample_rate_hz=1024)
    assert rec.sample_rate_hz == 1024.0
    assert rec.pretrigger_s is None  # «0,25» no se lee como 25
    assert any("SampleRate" in w and "comma" in w for w in rec.warnings)
    assert any("Pre-triggerLength" in w and "comma" in w for w in rec.warnings)


def test_thor_header_decimal_comma_read_with_argument(tmp_path):
    header = [(k, "0,25 sec" if k == "Pre-triggerLength" else v) for k, v in THOR_HEADER]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)), decimal=",")
    assert rec.pretrigger_s == 0.25
    _assert_signal(rec)


def test_thor_key_without_space_before_colon(tmp_path):
    text = '"SampleRate: 1024 sps"\n"Units: mm/s and pa"\nWaveform Data Channels\n' + "\t".join(NAMES) + "\n" + _rows()
    rec = read_instantel_ascii(_write(tmp_path, text))
    assert rec.sample_rate_hz == 1024.0
    _assert_signal(rec)


@pytest.mark.parametrize("text, expected", [
    ("03:14:01:250:125 PM", datetime(2026, 1, 15, 15, 14, 1, 250125)),
    ("12:05:09:007 AM", datetime(2026, 1, 15, 0, 5, 9, 7000)),
    ("12:05:09 PM", datetime(2026, 1, 15, 12, 5, 9)),
    ("23:59:58:001:002", datetime(2026, 1, 15, 23, 59, 58, 1002)),
])
def test_thor_time_with_fractions_and_12_hour_clock(tmp_path, text, expected):
    header = [(k, text if k == "EventTime" else v) for k, v in THOR_HEADER]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)))
    assert rec.start == expected


@pytest.mark.parametrize("date, time", [("15/01/2026", "10:20:30"), ("2026-01-15", "25:00:00"),
                                        ("2026-01-15", "13:00:00 PM")])
def test_thor_unparseable_date_gives_no_start_with_warning(tmp_path, date, time):
    header = [(k, {"EventDate": date, "EventTime": time}.get(k, v)) for k, v in THOR_HEADER]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)))
    assert rec.start is None
    assert any("start" in w for w in rec.warnings)


def test_thor_in_s_and_psi_converts(tmp_path):
    header = [(k, "in/s and psi" if k == "Units" else v) for k, v in THOR_HEADER]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)))
    _assert_signal(rec, scale={"transverse": 25.4, "vertical": 25.4, "longitudinal": 25.4, "mic": 6894.757})


def test_thor_db_drops_mic_with_warning(tmp_path):
    header = [(k, "mm/s and db" if k == "Units" else v) for k, v in THOR_HEADER]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)))
    assert list(rec.channels) == ["transverse", "vertical", "longitudinal"]
    assert any("dB" in w for w in rec.warnings)


def test_thor_usb_section_ignored_with_warning(tmp_path):
    extra = "Waveform Data USB Channels\nUSB1\tUSB2\n1.0\t2.0\n3.0\t4.0\n"
    rec = read_instantel_ascii(_write(tmp_path, _thor(extra=extra)))
    _assert_signal(rec)
    assert rec.n_samples == N
    assert any("USB" in w and "ignored" in w for w in rec.warnings)


def test_thor_sample_rate_conflict_shows_both(tmp_path):
    with pytest.raises(ValueError, match="sample_rate_hz") as err:
        read_instantel_ascii(_write(tmp_path, _thor()), sample_rate_hz=2048)
    assert "1024" in str(err.value) and "2048" in str(err.value)


def test_thor_units_and_pretrigger_conflicts_show_both(tmp_path):
    path = _write(tmp_path, _thor())
    with pytest.raises(ValueError, match="units") as err:
        read_instantel_ascii(path, units={"geophone": "in/s"})
    assert "in/s" in str(err.value) and "mm/s" in str(err.value)
    with pytest.raises(ValueError, match="pretrigger_s") as err:
        read_instantel_ascii(path, pretrigger_s=0.5)
    assert "0.5" in str(err.value) and "0.25" in str(err.value)


def test_explicit_arguments_equal_to_header_are_accepted(tmp_path):
    rec = read_instantel_ascii(_write(tmp_path, _thor()), sample_rate_hz=1024, pretrigger_s=0.25,
                               units={"geophone": "MM/S", "mic": "Pa"})
    assert rec.sample_rate_hz == 1024.0
    assert rec.pretrigger_s == 0.25
    _assert_signal(rec)


def test_argument_fills_what_the_file_lacks(tmp_path):
    header = [(k, v) for k, v in THOR_HEADER if k not in ("SampleRate", "Pre-triggerLength", "Units")]
    rec = read_instantel_ascii(_write(tmp_path, _thor(header)), sample_rate_hz=2048, pretrigger_s=0.1,
                               units={"geophone": "in/s", "mic": "psi"})
    assert rec.sample_rate_hz == 2048.0
    assert rec.pretrigger_s == 0.1
    assert rec.source == "instantel-thor"
    _assert_signal(rec, scale={"transverse": 25.4, "vertical": 25.4, "longitudinal": 25.4, "mic": 6894.757})


def test_unknown_channel_name_suggests_read_waveform_csv(tmp_path):
    with pytest.raises(ValueError, match="unknown channel name") as err:
        read_instantel_ascii(_write(tmp_path, _thor(names=("GeoT", "Vert", "Long", "MicL"))))
    assert "read_waveform_csv" in str(err.value)
    assert str(err.value).startswith("Domain error [BlastingPy]: parameter 'channels'")


# ---------------------------------------------------------------- Blastware con encabezado

BW_HEADER = ["Event Report", "Sample Rate 1024 sps", "Geophone mm/s   Microphone Pa"]


@pytest.mark.parametrize("sep, fmt, names_row", [
    ("\t", "{:.5f}", "\t".join(NAMES)),
    (",", "{:.5f}", ",".join(NAMES)),
    ("", "{:10.5f}", "".join(f"{n:>10}" for n in NAMES)),
])
def test_blastware_table(tmp_path, sep, fmt, names_row):
    text = _blastware_header(BW_HEADER) + names_row + "\n" + _rows(sep=sep, fmt=fmt)
    rec = read_instantel_ascii(_write(tmp_path, text, crlf=True))
    _assert_signal(rec)
    assert rec.sample_rate_hz == 1024.0
    assert rec.source == "instantel-blastware-table"
    assert rec.format_evidence == "manual-only"
    assert rec.metadata["header_1"] == "Event Report"
    assert rec.metadata["header_2"] == "Sample Rate 1024 sps"
    assert any("manual" in w and "not validated" in w for w in rec.warnings)
    assert not any("inferred" in w for w in rec.warnings)  # unidades del encabezado


def _blocks(names=NAMES):
    return "\n".join(",".join(f"{v:.5f}" for v in SIGNAL[n]) for n in names) + "\n"


def test_blastware_blocks_in_header_channel_order(tmp_path):
    order = ("Vert", "Tran", "MicL", "Long")
    header = BW_HEADER + ["Trigger source Vert", "Channels: " + " ".join(order)]
    rec = read_instantel_ascii(_write(tmp_path, _blastware_header(header) + _blocks(order), crlf=True))
    assert rec.source == "instantel-blastware-blocks"
    assert rec.format_evidence == "manual-only"
    _assert_signal(rec)
    assert not any("default order" in w for w in rec.warnings)


def test_blastware_blocks_without_names_use_default_order_with_warning(tmp_path):
    rec = read_instantel_ascii(_write(tmp_path, _blastware_header(BW_HEADER) + _blocks()))
    _assert_signal(rec)
    assert any("default order" in w and "Tran, Vert, Long, MicL" in w for w in rec.warnings)


def test_blastware_block_count_mismatch(tmp_path):
    text = _blastware_header(BW_HEADER) + _blocks(NAMES[:3])
    with pytest.raises(ValueError, match="3") as err:
        read_instantel_ascii(_write(tmp_path, text))
    assert "4" in str(err.value)


def test_blastware_two_geophone_units_give_no_header_unit(tmp_path):
    header = ["Sample Rate 1024 sps", "Geophone mm/s", "Range 10 in/s", "Microphone Pa"]
    path = _write(tmp_path, _blastware_header(header) + "\t".join(NAMES) + "\n" + _rows())
    rec = read_instantel_ascii(path, units={"geophone": "in/s"})  # no hay conflicto: el encabezado no decide
    assert any("geophone" in w and "mm/s" in w and "in/s" in w for w in rec.warnings)
    _assert_signal(rec, scale={"transverse": 25.4, "vertical": 25.4, "longitudinal": 25.4})


COLON_HEADER = ["Sample Rate: 1024 sps", "Units: mm/s and pa", "Event Date: 2026-01-15", "Event Time: 10:20:30"]


@pytest.mark.parametrize("layout", ["table", "blocks"])
def test_blastware_colon_header_is_not_thor(tmp_path, layout):
    body = "\t".join(NAMES) + "\n" + _rows() if layout == "table" else _blocks()
    rec = read_instantel_ascii(_write(tmp_path, _blastware_header(COLON_HEADER) + body))
    assert rec.source == f"instantel-blastware-{layout}"
    assert rec.format_evidence == "manual-only"
    assert rec.sample_rate_hz == 1024.0
    _assert_signal(rec)


def test_blastware_sample_rate_with_comma_is_ignored(tmp_path):
    text = _blastware_header(["Sample Rate 16,384 sps", "Geophone mm/s Microphone Pa"]) + _blocks()
    with pytest.raises(ValueError, match="sample_rate_hz"):
        read_instantel_ascii(_write(tmp_path, text))


def test_blastware_without_sample_rate_needs_argument(tmp_path):
    text = _blastware_header(["Event Report", "Geophone mm/s Microphone Pa"]) + _blocks()
    with pytest.raises(ValueError, match="sample_rate_hz"):
        read_instantel_ascii(_write(tmp_path, text))
    rec = read_instantel_ascii(_write(tmp_path, text), sample_rate_hz=2048)
    assert rec.sample_rate_hz == 2048.0


# ---------------------------------------------------------------- Mínimo


def _minimal():
    return "\t".join(NAMES) + "\n" + _rows()


UNITS = {"geophone": "mm/s", "mic": "Pa"}


@pytest.mark.parametrize("bom, crlf", [(False, False), (True, True)])
def test_minimal_reads(tmp_path, bom, crlf):
    rec = read_instantel_ascii(_write(tmp_path, _minimal(), bom=bom, crlf=crlf), sample_rate_hz=FS, units=UNITS,
                               pretrigger_s=0.25)
    _assert_signal(rec)
    assert rec.sample_rate_hz == FS
    assert rec.pretrigger_s == 0.25
    assert rec.start is None
    assert rec.source == "instantel-minimal"
    assert rec.format_evidence == "real-file"


def test_minimal_without_sample_rate_asks_for_it(tmp_path):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        read_instantel_ascii(_write(tmp_path, _minimal()), units=UNITS)


def test_minimal_infers_units_from_resolution(tmp_path):
    k = np.arange(N) % 9 - 4
    sig = {"Tran": np.round(k * 0.00788, 5), "Vert": np.round(-k * 0.00788, 5),
           "Long": np.round(2 * k * 0.00788, 5), "MicL": np.round(k * 0.0156, 5)}
    text = "\t".join(NAMES) + "\n" + _rows(signal=sig)
    rec = read_instantel_ascii(_write(tmp_path, text), sample_rate_hz=FS)
    assert any("inferred as mm/s" in w for w in rec.warnings)
    assert any("inferred as Pa" in w for w in rec.warnings)
    np.testing.assert_allclose(rec.channels["mic"], sig["MicL"], atol=1e-5)


# ---------------------------------------------------------------- Números y filas


def test_decimal_comma_without_argument_names_line_and_hint(tmp_path):
    text = _minimal().replace(".", ",")
    with pytest.raises(ValueError, match=r"line 2\b") as err:
        read_instantel_ascii(_write(tmp_path, text), sample_rate_hz=FS, units=UNITS)
    assert 'decimal=","' in str(err.value)


def test_decimal_comma_accepted_with_argument(tmp_path):
    text = _minimal().replace(".", ",")
    rec = read_instantel_ascii(_write(tmp_path, text), sample_rate_hz=FS, units=UNITS, decimal=",")
    _assert_signal(rec)


def test_decimal_comma_rejected_when_separator_is_comma(tmp_path):
    text = _blastware_header(BW_HEADER) + _blocks()
    with pytest.raises(ValueError, match="decimal"):
        read_instantel_ascii(_write(tmp_path, text), decimal=",")


def test_short_row_names_its_line(tmp_path):
    lines = _minimal().splitlines()
    lines[5] = "\t".join(lines[5].split("\t")[:3])
    with pytest.raises(ValueError, match=r"line 6\b"):
        read_instantel_ascii(_write(tmp_path, "\n".join(lines) + "\n"), sample_rate_hz=FS, units=UNITS)


def test_invalid_decimal_argument(tmp_path):
    with pytest.raises(ValueError, match="decimal"):
        read_instantel_ascii(_write(tmp_path, _minimal()), sample_rate_hz=FS, units=UNITS, decimal=";")
