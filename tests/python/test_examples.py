"""The worked example runs end to end on synthetic data and writes its outputs."""

import pathlib
import runpy

import pytest

pytest.importorskip("matplotlib")

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "01-site-law-and-compliance" / "site_law.py"


def test_site_law_example(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert (tmp_path / "site_law.png").is_file()
    assert (tmp_path / "compliance.png").is_file()
    assert any(line.startswith("Max charge per delay") for line in capsys.readouterr().out.splitlines())
