"""The worked example runs end to end on synthetic data and writes its outputs."""

import pathlib
import runpy
import sys
import warnings

import pytest

pytest.importorskip("matplotlib")

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "01-site-law-and-compliance" / "site_law.py"


def test_site_law_example(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert (tmp_path / "site_law.png").is_file()
    assert (tmp_path / "compliance.png").is_file()
    assert any(line.startswith("Max charge per delay") for line in capsys.readouterr().out.splitlines())


EXAMPLE_2 = pathlib.Path(__file__).resolve().parents[2] / "examples" / "02-waveform-to-events" / "waveform_to_events.py"


def test_waveform_to_events_example(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE_2)])  # no output directory: the current one
    with pytest.warns(UserWarning, match="units inferred") as caught:  # one summary warning for the six records
        runpy.run_path(str(EXAMPLE_2), run_name="__main__")
    assert len(caught) == 1
    assert (tmp_path / "record.png").is_file()
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("PPV") for line in lines)
    assert any(line.startswith("Warnings carried by the records") for line in lines)
    assert any("geophone units inferred as mm/s" in line for line in lines)
    start = next(i for i, line in enumerate(lines) if line.startswith("Events table"))
    table = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        table.append(line)
    assert len(table) - 1 == 6  # the header plus one row per record


EXAMPLE_3 = pathlib.Path(__file__).resolve().parents[2] / "examples" / "03-airblast" / "airblast.py"


def test_airblast_example(capsys):
    runpy.run_path(str(EXAMPLE_3), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Airblast site law") for line in lines)
    assert any(line.startswith("Limit") for line in lines)
    start = next(i for i, line in enumerate(lines) if line.startswith(" distance_m"))
    table = []
    for line in lines[start:]:
        if not line.strip():
            break
        table.append(line)
    assert len(table) - 1 == 8  # the header plus one row per event
    assert any("mixes measuring systems" in line for line in lines)


EXAMPLE_4 = pathlib.Path(__file__).resolve().parents[2] / "examples" / "04-flyrock" / "flyrock.py"


def test_flyrock_example(capsys):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the example stays within the published ranges: no UserWarning
        runpy.run_path(str(EXAMPLE_4), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    flyrock = next(line for line in lines if line.startswith("Flyrock"))
    assert "governed by its worst hole" in flyrock and "engineer's decision" in flyrock
    assert any(line.startswith("Clearance") for line in lines)
    assert any("Richards & Moore only" in line for line in lines)  # the contrast line
    assert any("c/m = " in line and "e-0" in line for line in lines)
    assert "max_model_m" in next(line for line in lines if "rm_max_m" in line)
    start = next(i for i, line in enumerate(lines) if line.startswith("hole "))
    table = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        table.append(line)
    assert len(table) == 6  # one row per hole, after the header and the index-name line
