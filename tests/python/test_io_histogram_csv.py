from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from blastingpy.io import read_instantel_histogram, read_waveform_csv

# --------------------------------------------------------------------------- histograma de Instantel

HEADER_TOP = ["", "Tran", "Tran", "Vert", "Vert", "Long", "Long", "Geo", "MicA", "MicA"]
HEADER_MID = ["", "PPV", "Freq", "PPV", "Freq", "PPV", "Freq", "PVS", "Lmax", "L10"]
HEADER_LOW = ["", "mm/s", "Hz", "mm/s", "Hz", "mm/s", "Hz", "mm/s", "dB(A)", "dB(A)"]

# Valores inventados; las marcas van en columnas Freq, como en el manual.
ROWS = [
    ["9:00:00", "1.5", "12", "0.8", "30", "0.4", "<1.0", "1.7", "80.5", "60"],
    ["9:00:02", "0.25", "47", "0.1", ">100", "0.3", "N/A", "0.3", "71", "55.5"],
    ["9:00:04", "3.15", "3.9", "2.2", "21", "1.9", "N/A", "3.9", "90", "70"],
    ["9:00:06", "0", "N/A", "0", "N/A", "0", "N/A", "0", "50", "45"],
    ["9:00:09", "0.5", "8", "0.4", "10", "0.2", "<1.0", "0.6", "65", "52"],
]


def _hist(rows=ROWS, date="15-Feb-13", sep="\t", time_line=2, extra_lines=None, crlf=False, bom=False):
    header = [list(HEADER_TOP), list(HEADER_MID), list(HEADER_LOW)]
    for k, line in enumerate(header):
        line[0] = "Time" if k == time_line else ""
    lines = [sep.join(h) for h in header]
    if date:
        lines.append(date)
    lines += [sep.join(r) for r in rows]
    if extra_lines:
        lines += extra_lines
    text = "\n".join(lines) + "\n"
    if crlf:
        text = text.replace("\n", "\r\n")
    return text.encode("utf-8-sig" if bom else "utf-8")


def _write(tmp_path, data, name="hist.txt"):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


def test_histogram_layout_columns_markers_and_units(tmp_path):
    df = read_instantel_histogram(_write(tmp_path, _hist()))
    assert len(df) == 5
    assert list(df.columns) == [
        "time", "tran_ppv", "tran_freq", "tran_freq_flag", "vert_ppv", "vert_freq", "vert_freq_flag",
        "long_ppv", "long_freq", "long_freq_flag", "geo_pvs", "mica_lmax", "mica_l10"]
    assert df["time"].iloc[0] == datetime(2013, 2, 15, 9, 0, 0)
    assert df["time"].iloc[4] == datetime(2013, 2, 15, 9, 0, 9)  # los intervalos no son iguales
    np.testing.assert_allclose(df["tran_ppv"], [1.5, 0.25, 3.15, 0.0, 0.5])
    # marcas: NaN en la columna numérica y el texto en <columna>_flag
    assert np.isnan(df["long_freq"].iloc[0]) and df["long_freq_flag"].iloc[0] == "<1.0"
    assert df["long_freq_flag"].iloc[2] == "N/A" and df["long_freq_flag"].iloc[1] == "N/A"
    assert df["long_freq_flag"].iloc[4] == "<1.0"
    assert df["vert_freq_flag"].iloc[1] == ">100" and np.isnan(df["vert_freq"].iloc[1])
    assert df["tran_freq_flag"].iloc[3] == "N/A"
    # las demás celdas de una columna flag son None; sin marcas no hay columna flag
    assert df["vert_freq_flag"].iloc[0] is None
    assert "tran_ppv_flag" not in df.columns and "geo_pvs_flag" not in df.columns
    assert df["tran_freq"].iloc[0] == 12.0
    assert df.attrs["units"] == {
        "tran_ppv": "mm/s", "tran_freq": "Hz", "vert_ppv": "mm/s", "vert_freq": "Hz", "long_ppv": "mm/s",
        "long_freq": "Hz", "geo_pvs": "mm/s", "mica_lmax": "dB(A)", "mica_l10": "dB(A)"}
    assert any("manual-only" in w for w in df.attrs["warnings"])


def test_histogram_flag_column_follows_its_column(tmp_path):
    df = read_instantel_histogram(_write(tmp_path, _hist()))
    cols = list(df.columns)
    assert cols[cols.index("long_freq") + 1] == "long_freq_flag"
    assert cols[0] == "time"


@pytest.mark.parametrize("time_line", [0, 1, 2])
def test_histogram_time_label_on_any_header_line(tmp_path, time_line):
    df = read_instantel_histogram(_write(tmp_path, _hist(time_line=time_line)))
    assert len(df) == 5 and "tran_ppv" in df.columns


def test_histogram_comma_separator_bom_and_crlf(tmp_path):
    df = read_instantel_histogram(_write(tmp_path, _hist(sep=",", bom=True, crlf=True)))
    assert len(df) == 5 and df["time"].iloc[1] == datetime(2013, 2, 15, 9, 0, 2)


def test_histogram_markers_with_spaces(tmp_path):
    rows = [list(r) for r in ROWS]
    rows[0][2], rows[1][4] = "< 1.0", "> 100"
    df = read_instantel_histogram(_write(tmp_path, _hist(rows)))
    assert df["tran_freq_flag"].iloc[0] == "< 1.0" and np.isnan(df["tran_freq"].iloc[0])
    assert df["vert_freq_flag"].iloc[1] == "> 100"


def test_histogram_midnight_date_change(tmp_path):
    rows = [["23:59:58", "1", "10", "1", "10", "1", "10", "1", "70", "60"],
            ["23:59:59", "1", "10", "1", "10", "1", "10", "1", "70", "60"]]
    data = _hist(rows, date="28-Feb-13", extra_lines=[
        "1-Mar-13", "0:00:01\t1\t10\t1\t10\t1\t10\t1\t70\t60"])
    df = read_instantel_histogram(_write(tmp_path, data))
    assert list(df["time"]) == [datetime(2013, 2, 28, 23, 59, 58), datetime(2013, 2, 28, 23, 59, 59),
                                datetime(2013, 3, 1, 0, 0, 1)]


def test_histogram_row_before_any_date_is_error(tmp_path):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter 'path'.*line 4.*date"):
        read_instantel_histogram(_write(tmp_path, _hist(date=None)))


def test_histogram_unknown_marker_text_is_error_with_line_and_column(tmp_path):
    rows = [list(r) for r in ROWS]
    rows[2][2] = "abc"
    with pytest.raises(ValueError, match=r"line 7.*tran_freq.*abc"):
        read_instantel_histogram(_write(tmp_path, _hist(rows)))
    rows[2][2] = "<x"
    with pytest.raises(ValueError, match="<x"):
        read_instantel_histogram(_write(tmp_path, _hist(rows)))


def test_histogram_imperial_units_are_converted_once(tmp_path):
    top = ["", "Tran", "Tran", "Mic A"]
    mid = ["", "PPV", "Freq", "PPV"]
    low = ["Time", "in/s", "Hz", "psi"]
    text = "\n".join("\t".join(x) for x in (top, mid, low)) + "\n7-Nov-12\n10:31:53\t0.5\t20\t0.001\n10:31:54\t1\t<1.0\t0.002\n"
    df = read_instantel_histogram(_write(tmp_path, text))
    np.testing.assert_allclose(df["tran_ppv"], [12.7, 25.4])
    np.testing.assert_allclose(df["mica_ppv"], [0.001 * 6894.757, 0.002 * 6894.757])
    assert df["tran_freq"].iloc[0] == 20.0  # Hz no se convierte
    assert df.attrs["units"] == {"tran_ppv": "mm/s", "tran_freq": "Hz", "mica_ppv": "Pa"}
    assert df["time"].iloc[1] == datetime(2012, 11, 7, 10, 31, 54)


def test_histogram_mic_a_with_space_and_pa_with_dot(tmp_path):
    text = "\tMic A\tMic A\n\tPPV\tFreq\nTime\tpa.\tHz\n7-Nov-12\n10:31:53\t0.25\t30\n10:32:02\t0.5\t<1.0\n"
    df = read_instantel_histogram(_write(tmp_path, text))
    assert list(df.columns) == ["time", "mica_ppv", "mica_freq", "mica_freq_flag"]
    assert df.attrs["units"] == {"mica_ppv": "Pa", "mica_freq": "Hz"}
    assert df["time"].iloc[1] == datetime(2012, 11, 7, 10, 32, 2)


def test_histogram_duplicate_column_names_are_error(tmp_path):
    text = "\tTran\tTran\n\tPPV\tPPV\nTime\tmm/s\tmm/s\n7-Nov-12\n10:31:53\t1\t2\n"
    with pytest.raises(ValueError, match="tran_ppv"):
        read_instantel_histogram(_write(tmp_path, text))


def test_histogram_unsupported_separator_is_error(tmp_path):
    text = "Time;Tran\n;PPV\n;mm/s\n7-Nov-12\n10:31:53;1\n"
    with pytest.raises(ValueError, match="separator"):
        read_instantel_histogram(_write(tmp_path, text))


def test_histogram_row_with_wrong_field_count_is_error(tmp_path):
    with pytest.raises(ValueError, match="fields at line 5"):
        read_instantel_histogram(_write(tmp_path, _hist(extra_lines=None, rows=[ROWS[0][:-1]])))


def test_histogram_without_rows_is_error(tmp_path):
    with pytest.raises(ValueError, match="no data rows"):
        read_instantel_histogram(_write(tmp_path, _hist(rows=[])))


def test_histogram_returns_dataframe(tmp_path):
    assert isinstance(read_instantel_histogram(_write(tmp_path, _hist())), pd.DataFrame)


# --------------------------------------------------------------------------- CSV genérico

FS = 1024.0
N = 32
_T = np.arange(N) / FS
SIG_X = np.round(2.5 * np.sin(2 * np.pi * 20 * _T), 5)
SIG_Y = np.round(-1.75 * np.sin(2 * np.pi * 35 * _T + 0.3), 5)


def _csv(time=None, sep=",", dec=".", names=("t", "X", "Y"), x=SIG_X, y=SIG_Y):
    def f(v):
        return repr(float(v)).replace(".", dec)
    rows = [sep.join(names if time is not None else names[1:])]
    for i in range(len(x)):
        cells = ([f(time[i])] if time is not None else []) + [f(x[i]), f(y[i])]
        rows.append(sep.join(cells))
    return "\n".join(rows) + "\n"


def _csv_file(tmp_path, text, name="onda.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


MAP = {"transverse": "X", "vertical": "Y"}
UNITS = {"geophone": "mm/s"}


def test_csv_negative_start_gives_pretrigger_and_fs(tmp_path):
    t = _T - 0.25
    rec = read_waveform_csv(_csv_file(tmp_path, _csv(time=t)), columns=MAP, units=UNITS, time_column="t")
    assert rec.sample_rate_hz == pytest.approx(FS, rel=1e-9)
    assert rec.pretrigger_s == pytest.approx(0.25)
    assert rec.source == "csv" and rec.format_evidence == "user"
    assert set(rec.channels) == {"transverse", "vertical"}
    np.testing.assert_allclose(rec.channels["transverse"], SIG_X)
    np.testing.assert_allclose(rec.time_s(), t, atol=1e-9)
    assert rec.start is None


def test_csv_start_at_zero_gives_zero_pretrigger_without_warning(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv(time=_T)), columns=MAP, units=UNITS, time_column="t")
    assert rec.pretrigger_s == 0.0
    assert not any("shifted" in w for w in rec.warnings)


def test_csv_positive_start_warns_and_zero_pretrigger(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv(time=_T + 0.5)), columns=MAP, units=UNITS, time_column="t")
    assert rec.pretrigger_s == 0.0
    assert any("first sample" in w and "shifted" in w for w in rec.warnings)


def test_csv_non_uniform_time_is_error(tmp_path):
    t = _T.copy()
    t[10:] += 0.0004
    with pytest.raises(ValueError, match="uniform"):
        read_waveform_csv(_csv_file(tmp_path, _csv(time=t)), columns=MAP, units=UNITS, time_column="t")


def test_csv_decreasing_time_is_error(tmp_path):
    with pytest.raises(ValueError, match="time"):
        read_waveform_csv(_csv_file(tmp_path, _csv(time=-_T)), columns=MAP, units=UNITS, time_column="t")


def test_csv_semicolon_with_decimal_comma(tmp_path):
    text = _csv(time=_T - 0.1, sep=";", dec=",")
    rec = read_waveform_csv(_csv_file(tmp_path, text), columns=MAP, units=UNITS, time_column="t")
    np.testing.assert_allclose(rec.channels["vertical"], SIG_Y)
    assert rec.sample_rate_hz == pytest.approx(FS, rel=1e-9)


def test_csv_semicolon_with_point_decimal_is_rejected(tmp_path):
    text = _csv(sep=";", dec=".")
    with pytest.raises(ValueError, match="decimal"):
        read_waveform_csv(_csv_file(tmp_path, text), columns=MAP, units=UNITS, sample_rate_hz=FS)


def test_csv_comma_separated_reads_point_decimal(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv()), columns=MAP, units=UNITS, sample_rate_hz=FS)
    np.testing.assert_allclose(rec.channels["transverse"], SIG_X)
    assert rec.pretrigger_s is None


def test_csv_tab_separated_ambiguous_number_is_rejected(tmp_path):
    text = "X\tY\n1,250\t2\n3\t4\n"
    with pytest.raises(ValueError, match="ambiguous"):
        read_waveform_csv(_csv_file(tmp_path, text), columns=MAP, units=UNITS, sample_rate_hz=FS)
    rec = read_waveform_csv(_csv_file(tmp_path, text), columns=MAP, units=UNITS, sample_rate_hz=FS, decimal=",")
    assert rec.channels["transverse"][0] == pytest.approx(1.25)


def test_csv_without_time_column_or_fs_is_error(tmp_path):
    with pytest.raises(ValueError, match="sample_rate_hz"):
        read_waveform_csv(_csv_file(tmp_path, _csv()), columns=MAP, units=UNITS)


def test_csv_explicit_pretrigger_without_time_column(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv()), columns=MAP, units=UNITS, sample_rate_hz=FS,
                            pretrigger_s=0.25)
    assert rec.pretrigger_s == 0.25


def test_csv_fs_conflicting_with_time_column_is_error(tmp_path):
    with pytest.raises(ValueError, match=r"2048.*1024|1024.*2048"):
        read_waveform_csv(_csv_file(tmp_path, _csv(time=_T)), columns=MAP, units=UNITS, time_column="t",
                          sample_rate_hz=2048.0)


def test_csv_fs_agreeing_with_time_column_is_accepted(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv(time=_T)), columns=MAP, units=UNITS, time_column="t",
                            sample_rate_hz=1024.0)
    assert rec.sample_rate_hz == pytest.approx(FS, rel=1e-9)


def test_csv_pretrigger_conflicting_with_time_column_is_error(tmp_path):
    with pytest.raises(ValueError, match=r"pretrigger_s.*0\.5.*0\.25"):
        read_waveform_csv(_csv_file(tmp_path, _csv(time=_T - 0.25)), columns=MAP, units=UNITS, time_column="t",
                          pretrigger_s=0.5)
    rec = read_waveform_csv(_csv_file(tmp_path, _csv(time=_T - 0.25)), columns=MAP, units=UNITS,
                            time_column="t", pretrigger_s=0.25)
    assert rec.pretrigger_s == pytest.approx(0.25)


def test_csv_missing_mapped_column_is_error(tmp_path):
    with pytest.raises(ValueError, match="'Z'"):
        read_waveform_csv(_csv_file(tmp_path, _csv()), columns={"transverse": "Z"}, units=UNITS,
                          sample_rate_hz=FS)


def test_csv_missing_time_column_is_error(tmp_path):
    with pytest.raises(ValueError, match="'tiempo'"):
        read_waveform_csv(_csv_file(tmp_path, _csv()), columns=MAP, units=UNITS, time_column="tiempo")


def test_csv_imperial_units_converted_and_renamed_channels(tmp_path):
    text = _csv(names=("t", "CanalA", "CanalB"))
    rec = read_waveform_csv(_csv_file(tmp_path, text), columns={"transverse": "CanalA", "vertical": "CanalB"},
                            units={"geophone": "in/s"}, sample_rate_hz=FS)
    np.testing.assert_allclose(rec.channels["transverse"], SIG_X * 25.4)


def test_csv_mic_in_psi(tmp_path):
    text = _csv(names=("t", "X", "P"), y=SIG_Y * 10)
    rec = read_waveform_csv(_csv_file(tmp_path, text), columns={"transverse": "X", "mic": "P"},
                            units={"geophone": "mm/s", "mic": "psi"}, sample_rate_hz=FS)
    np.testing.assert_allclose(rec.channels["mic"], SIG_Y * 10 * 6894.757)


def test_csv_non_numeric_cell_names_column_and_row(tmp_path):
    text = "X,Y\n1,2\nabc,4\n5,6\n"
    with pytest.raises(ValueError, match=r"'X'.*row 3.*abc"):
        read_waveform_csv(_csv_file(tmp_path, text), columns=MAP, units=UNITS, sample_rate_hz=FS)


def test_csv_trailing_blank_lines_are_ignored(tmp_path):
    rec = read_waveform_csv(_csv_file(tmp_path, _csv() + "\n\n"), columns=MAP, units=UNITS, sample_rate_hz=FS)
    assert rec.n_samples == N


def test_csv_invalid_decimal_argument(tmp_path):
    with pytest.raises(ValueError, match="decimal"):
        read_waveform_csv(_csv_file(tmp_path, _csv()), columns=MAP, units=UNITS, sample_rate_hz=FS, decimal=";")


def test_csv_bom_and_crlf(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + _csv(time=_T).replace("\n", "\r\n").encode())
    rec = read_waveform_csv(path, columns=MAP, units=UNITS, time_column="t")
    assert rec.n_samples == N
