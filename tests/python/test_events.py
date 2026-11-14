import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import blastingpy
from blastingpy.events import read_events, write_events_template

COLS = {"distance_m": "Dist", "charge_kg": "Carga", "ppv_mm_s": "PPV"}


def _csv(tmp_path, text, name="e.csv", encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return p


def _xlsx(tmp_path, rows, header=("Dist", "Carga", "PPV"), name="e.xlsx"):
    p = tmp_path / name
    pd.DataFrame(rows, columns=list(header)).to_excel(p, index=False)
    return p


def test_exported_at_top_level():
    assert blastingpy.read_events is read_events
    assert blastingpy.events.write_events_template is write_events_template


def test_semicolon_and_decimal_comma(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n120,5;45;3,2\n300;60;0,85\n")
    df = read_events(p, columns=COLS, ppv_kind="component_max")
    assert df["distance_m"].tolist() == [120.5, 300.0] and df["ppv_mm_s"].tolist() == [3.2, 0.85]
    assert df.columns.tolist() == ["distance_m", "charge_kg", "ppv_mm_s", "ppv_kind"]
    assert df["ppv_kind"].tolist() == ["component_max"] * 2
    assert df.index.tolist() == [0, 1]


def test_comma_separator_bom_and_quoted_decimal_comma(tmp_path):
    p = _csv(tmp_path, '\ufeffDist,Carga,PPV\n"120,5",45,"3,2"\n', encoding="utf-8")
    df = read_events(p, columns=COLS, ppv_kind="vertical", decimal=",")
    assert df["distance_m"].tolist() == [120.5] and df["ppv_mm_s"].tolist() == [3.2]


def test_cp1252_file_from_spanish_excel(tmp_path):
    p = _csv(tmp_path, "Distancia (m);Carga máx;PPV\n100;20;2,5\n", encoding="cp1252")
    cols = {"distance_m": "Distancia (m)", "charge_kg": "Carga máx", "ppv_mm_s": "PPV"}
    assert read_events(p, columns=cols, ppv_kind="vertical")["charge_kg"].tolist() == [20.0]


def test_explicit_sep_and_decimal_override_detection(tmp_path):
    p = _csv(tmp_path, "Dist|Carga|PPV\n100|20|2,5\n")
    with pytest.raises(ValueError, match="'PPV'.*row 2"):
        read_events(p, columns=COLS, ppv_kind="vertical", sep="|", decimal=".")
    df = read_events(p, columns=COLS, ppv_kind="vertical", sep="|", decimal=",")
    assert df["ppv_mm_s"].tolist() == [2.5]


def test_blank_ppv_rows_are_skipped_with_warning(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;2,5\n150;20;\n200;30;1\n")
    with pytest.warns(UserWarning, match=r"1 row skipped: empty ppv_mm_s"):
        df = read_events(p, columns=COLS, ppv_kind="vertical")
    assert len(df) == 2 and df["distance_m"].tolist() == [100.0, 200.0]
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;\n150;20;\n200;30;1\n", name="f.csv")
    with pytest.warns(UserWarning, match=r"2 rows skipped: empty ppv_mm_s"):
        read_events(p, columns=COLS, ppv_kind="vertical")


def test_entirely_empty_rows_are_skipped_silently(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;2\n\n;;\n200;30;1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = read_events(p, columns=COLS, ppv_kind="vertical")
    assert len(df) == 2


def test_excel_numbers_stored_as_text(tmp_path):
    p = _xlsx(tmp_path, [("12,5", "10", "1.5"), ("12.5", 10, 2)])
    df = read_events(p, columns=COLS, ppv_kind="vertical")
    assert df["distance_m"].tolist() == [12.5, 12.5] and df["ppv_mm_s"].tolist() == [1.5, 2.0]


def test_excel_sheet_selection(tmp_path):
    p = tmp_path / "s.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"Dist": [1], "Carga": [1], "PPV": [9]}).to_excel(w, sheet_name="uno", index=False)
        pd.DataFrame({"Dist": [2], "Carga": [2], "PPV": [8]}).to_excel(w, sheet_name="dos", index=False)
    assert read_events(p, columns=COLS, ppv_kind="vertical")["ppv_mm_s"].tolist() == [9.0]
    assert read_events(p, columns=COLS, ppv_kind="vertical", sheet="dos")["ppv_mm_s"].tolist() == [8.0]
    assert read_events(p, columns=COLS, ppv_kind="vertical", sheet=1)["ppv_mm_s"].tolist() == [8.0]


def test_non_numeric_ppv_names_column_and_file_row(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;2\n150;20;N/T\n")
    with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 3.*'N/T'"):
        read_events(p, columns=COLS, ppv_kind="vertical")
    with pytest.warns(UserWarning, match="1 row skipped"):
        df = read_events(p, columns=COLS, ppv_kind="vertical", na_values=("N/T",))
    assert len(df) == 1


def test_imperial_and_per_quantity_units(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;10;1\n")
    df = read_events(p, columns=COLS, units="imperial", ppv_kind="vertical")
    assert df.loc[0, "distance_m"] == pytest.approx(30.48)
    assert df.loc[0, "charge_kg"] == pytest.approx(4.5359237)
    assert df.loc[0, "ppv_mm_s"] == pytest.approx(25.4)
    df = read_events(p, columns=COLS, units={"distance": "ft"}, ppv_kind="vertical")
    assert df.loc[0, "distance_m"] == pytest.approx(30.48) and df.loc[0, "charge_kg"] == 10.0
    for bad in ({"altitude": "m"}, {"distance": "km"}, {"distance": "lb"}, "si", 3):
        with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter 'units'"):
            read_events(p, columns=COLS, units=bad, ppv_kind="vertical")


def test_column_mapping_errors(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;10;1\n")
    with pytest.raises(ValueError, match=r"parameter 'columns'.*'charge_kg'"):
        read_events(p, columns={"distance_m": "Dist", "ppv_mm_s": "PPV"}, ppv_kind="vertical")
    with pytest.raises(ValueError, match=r"parameter 'columns'.*'altitude'"):
        read_events(p, columns={**COLS, "altitude": "Dist"}, ppv_kind="vertical")
    with pytest.raises(ValueError, match=r"parameter 'columns'.*'ppv_mm_s'.*'Velocidad'"):
        read_events(p, columns={**COLS, "ppv_mm_s": "Velocidad"}, ppv_kind="vertical")


def test_ppv_kind_argument_and_column(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV;Tipo\n100;10;1;vertical\n200;10;1;vector_sum\n")
    with pytest.raises(ValueError, match=r"parameter 'ppv_kind'"):
        read_events(p, columns=COLS)
    with pytest.raises(ValueError, match=r"parameter 'ppv_kind'.*'peak'"):
        read_events(p, columns=COLS, ppv_kind="peak")
    df = read_events(p, columns={**COLS, "ppv_kind": "Tipo"})
    assert df["ppv_kind"].tolist() == ["vertical", "vector_sum"]
    p = _csv(tmp_path, "Dist;Carga;PPV;Tipo\n100;10;1;vertical\n200;10;1;otro\n", name="g.csv")
    with pytest.raises(ValueError, match=r"column 'Tipo' \(ppv_kind\), row 3"):
        read_events(p, columns={**COLS, "ppv_kind": "Tipo"})


@pytest.mark.parametrize("row,col,canon", [("0;10;1", "Dist", "distance_m"), ("100;-1;1", "Carga", "charge_kg"),
                                           ("100;10;0", "PPV", "ppv_mm_s")])
def test_non_positive_values(tmp_path, row, col, canon):
    p = _csv(tmp_path, f"Dist;Carga;PPV\n50;5;1\n{row}\n")
    with pytest.raises(ValueError, match=rf"column '{col}' \({canon}\), row 3.*positive"):
        read_events(p, columns=COLS, ppv_kind="vertical")


def test_empty_distance_or_charge_with_ppv_is_an_error(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n50;5;1\n;5;1\n")
    with pytest.raises(ValueError, match=r"column 'Dist' \(distance_m\), row 3.*empty"):
        read_events(p, columns=COLS, ppv_kind="vertical")


def test_dataframe_source_optional_columns_and_row_index(tmp_path):
    src = pd.DataFrame({
        "d": [100.0, 200.0, 300.0], "w": [10, 20, 30], "v": [1.0, np.nan, 3.0], "f": [20.0, np.nan, 5.0],
        "fecha": ["2026-01-02", "2026-01-03", "2026-01-04"], "pt": ["P1", "P2", "P3"], "otra": [1, 2, 3],
    }, index=[10, 11, 12])
    cols = {"point": "pt", "date": "fecha", "frequency_hz": "f", "distance_m": "d", "charge_kg": "w",
            "ppv_mm_s": "v"}
    with pytest.warns(UserWarning, match="1 row skipped"):
        df = read_events(src, columns=cols, ppv_kind="vertical")
    assert df.columns.tolist() == ["distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz", "date", "point"]
    assert df.index.tolist() == [0, 1] and df["point"].tolist() == ["P1", "P3"]
    assert df["date"].tolist() == ["2026-01-02", "2026-01-04"]
    src.loc[12, "d"] = -5.0
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match=r"column 'd' \(distance_m\), row index 2"):
            read_events(src, columns=cols, ppv_kind="vertical")


def test_optional_numeric_column_may_be_empty_and_is_not_converted(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV;Hz;Vt\n100;10;1;12,5;\n200;10;2;;0,5\n")
    cols = {**COLS, "frequency_hz": "Hz", "ppv_t_mm_s": "Vt"}
    df = read_events(p, columns=cols, units="imperial", ppv_kind="vertical")
    assert df["frequency_hz"].tolist()[0] == 12.5 and np.isnan(df["frequency_hz"][1])
    assert np.isnan(df["ppv_t_mm_s"][0]) and df["ppv_t_mm_s"][1] == pytest.approx(12.7)


def _airblast_events(tmp_path, units, text="Dist;Carga;PPV;AB\n100;10;1;0,0145\n200;10;2;\n"):
    p = _csv(tmp_path, text)
    return read_events(p, columns={**COLS, "airblast_pa": "AB"}, units=units, ppv_kind="vertical")


def test_airblast_column_in_psi_kpa_db_and_imperial(tmp_path):
    df = _airblast_events(tmp_path, {"pressure": "psi"})
    assert df["airblast_pa"][0] == pytest.approx(99.97, rel=1e-4) and np.isnan(df["airblast_pa"][1])
    assert df.columns.tolist()[-1] == "airblast_pa"
    kpa = _airblast_events(tmp_path, {"pressure": "kPa"}, "Dist;Carga;PPV;AB\n100;10;1;0,1\n")
    assert kpa["airblast_pa"][0] == pytest.approx(100.0)
    db = _airblast_events(tmp_path, {"pressure": "dB"}, "Dist;Carga;PPV;AB\n100;10;1;134\n200;10;2;-5\n")
    assert db["airblast_pa"][0] == pytest.approx(100.24, rel=1e-4) and db["airblast_pa"][1] > 0
    assert _airblast_events(tmp_path, "imperial")["airblast_pa"][0] == pytest.approx(99.97, rel=1e-4)
    assert _airblast_events(tmp_path, "metric", "Dist;Carga;PPV;AB\n100;10;1;100\n")["airblast_pa"][0] == 100.0


def test_airblast_non_positive_pa_psi_kpa_is_a_cell_error(tmp_path):
    for unit in ("Pa", "kPa", "psi"):
        with pytest.raises(ValueError, match=r"column 'AB' \(airblast_pa\), row 3: expected a positive value"):
            _airblast_events(tmp_path, {"pressure": unit}, "Dist;Carga;PPV;AB\n100;10;1;1\n200;10;2;0\n")


def test_unknown_pressure_unit(tmp_path):
    with pytest.raises(ValueError, match="unknown unit 'bar' for 'pressure'"):
        _airblast_events(tmp_path, {"pressure": "bar"})


def test_unknown_file_type_and_source(tmp_path):
    p = _csv(tmp_path, "x", name="e.txt")
    with pytest.raises(ValueError, match=r"parameter 'source'"):
        read_events(p, columns=COLS, ppv_kind="vertical")
    with pytest.raises(ValueError, match=r"parameter 'source'"):
        read_events(42, columns=COLS, ppv_kind="vertical")


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_template_round_trip(tmp_path, ext):
    p = tmp_path / f"plantilla{ext}"
    write_events_template(p)
    cols = {c: c for c in ("distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz", "date", "point")}
    df = read_events(p, columns=cols)
    assert len(df) == 2 and (df["distance_m"] > 0).all() and df["ppv_kind"].notna().all()
    # Las filas de ejemplo se pueden comparar con los criterios publicados (vertical no lo era con ninguno).
    from blastingpy.criteria import evaluate
    assert (evaluate(df, "USBM-RI8507-1980", "appendix-b-drywall").status != "not_comparable").all()
    with pytest.raises(ValueError, match=r"parameter 'path'"):
        write_events_template(tmp_path / "plantilla.txt")


def test_xlsx_without_openpyxl(tmp_path, monkeypatch):
    p = _xlsx(tmp_path, [(1, 1, 1)])
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError, match=r"pip install blastingpy\[io\]"):
        read_events(p, columns=COLS, ppv_kind="vertical")
    with pytest.raises(ImportError, match=r"pip install blastingpy\[io\]"):
        write_events_template(tmp_path / "t.xlsx")


def test_semicolon_file_uses_decimal_comma(tmp_path):
    # En un archivo con ";" el punto es marca de miles (1.250 = 1250), no decimal: se rechaza en vez de leerlo como 1.25.
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;1,250\n")
    assert read_events(p, columns=COLS, ppv_kind="vertical")["ppv_mm_s"].tolist() == [1.25]
    for bad in ("1.250", "1.250,5"):
        p = _csv(tmp_path, f"Dist;Carga;PPV\n100;20;{bad}\n", name="m.csv")
        with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 2.*decimal"):
            read_events(p, columns=COLS, ppv_kind="vertical")
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;12.5\n", name="n.csv")
    assert read_events(p, columns=COLS, ppv_kind="vertical", decimal=".")["ppv_mm_s"].tolist() == [12.5]
    p = _csv(tmp_path, "Dist,Carga,PPV\n100,20,0.850\n", name="o.csv")  # con "," el punto sí es decimal
    assert read_events(p, columns=COLS, ppv_kind="vertical")["ppv_mm_s"].tolist() == [0.85]


@pytest.mark.parametrize("sep", [",", ";"])
def test_rows_with_a_different_field_count_are_errors(tmp_path, sep):
    longer = _csv(tmp_path, f"Dist{sep}Carga{sep}PPV\n10{sep}1{sep}1\n120{sep}5{sep}45{sep}3.2\n", name="long.csv")
    with pytest.raises(ValueError, match=r"row 3.*4 fields.*header has 3.*sep"):
        read_events(longer, columns=COLS, ppv_kind="vertical", decimal=".")
    short = _csv(tmp_path, f"Dist{sep}Carga{sep}PPV\n10{sep}1{sep}1\n100{sep}20\n", name="short.csv")
    with pytest.raises(ValueError, match=r"row 3.*2 fields.*header has 3"):
        read_events(short, columns=COLS, ppv_kind="vertical")


def test_quoted_fields_count_as_one_field(tmp_path):
    p = _csv(tmp_path, 'Dist,Carga,PPV\n"120,5",45,"3,2"\n')
    assert read_events(p, columns=COLS, ppv_kind="vertical", decimal=",")["distance_m"].tolist() == [120.5]


def test_row_numbers_after_blank_empty_and_skipped_rows_csv(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV\n100;20;2\n\n;;\n150;20;\n200;20;x\n")
    with pytest.warns(UserWarning, match="1 row skipped"):
        with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 6"):
            read_events(p, columns=COLS, ppv_kind="vertical")


def test_row_numbers_after_empty_and_skipped_rows_xlsx(tmp_path):
    p = _xlsx(tmp_path, [(100, 20, 2), (None, None, None), (150, 20, None), (200, 20, "x")])
    with pytest.warns(UserWarning, match="1 row skipped"):
        with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 5"):
            read_events(p, columns=COLS, ppv_kind="vertical")


def test_dataframe_source_is_not_modified(tmp_path):
    src = pd.DataFrame({"Dist": [100.0, 200.0], "Carga": [10.0, 20.0], "PPV": [1.0, 2.0]})
    before = src.copy()
    first = read_events(src, columns=COLS, units="imperial", ppv_kind="vertical")
    assert src.equals(before)
    assert read_events(src, columns=COLS, units="imperial", ppv_kind="vertical").equals(first)


def test_comma_file_uses_the_decimal_point(tmp_path):
    # Con "," la coma es separador de campo y de miles: "1,250" entre comillas se leía 1.25 sin aviso.
    for bad in ('"1,250"', '"120,5"'):
        p = _csv(tmp_path, f"Dist,Carga,PPV\n100,20,{bad}\n", name="c.csv")
        with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 2.*decimal="):
            read_events(p, columns=COLS, ppv_kind="vertical")
        with pytest.raises(ValueError, match=r"column 'PPV' \(ppv_mm_s\), row 2.*decimal="):
            read_events(p, columns=COLS, ppv_kind="vertical", sep=",")
    p = _csv(tmp_path, 'Dist,Carga,PPV\n100,20,"1,250"\n', name="d.csv")
    assert read_events(p, columns=COLS, ppv_kind="vertical", decimal=",")["ppv_mm_s"].tolist() == [1.25]


@pytest.mark.parametrize("text, decimal, value", [("1,250", ",", 1.25), ("1.250", ".", 1.25),
                                                  ("12.500", ".", 12.5), ("100,000", ",", 100.0)])
def test_ambiguous_thousands_shape_is_rejected_where_both_marks_are_accepted(tmp_path, text, decimal, value):
    # Tabulador, celdas de texto de Excel y de un DataFrame aceptan las dos marcas, salvo la forma de miles.
    message = rf"column 'PPV' \(ppv_mm_s\), row 2: {text!r} is ambiguous.*decimal="
    p = _csv(tmp_path, f"Dist\tCarga\tPPV\n100\t20\t{text}\n", name="t.csv")
    with pytest.raises(ValueError, match=message):
        read_events(p, columns=COLS, ppv_kind="vertical")
    x = _xlsx(tmp_path, [(100, 20, text)])
    with pytest.raises(ValueError, match=message):
        read_events(x, columns=COLS, ppv_kind="vertical")
    src = pd.DataFrame({"Dist": [100.0], "Carga": [20.0], "PPV": [text]})
    with pytest.raises(ValueError, match=rf"column 'PPV' \(ppv_mm_s\), row index 0: {text!r} is ambiguous"):
        read_events(src, columns=COLS, ppv_kind="vertical")
    for source in (p, x, src):
        assert read_events(source, columns=COLS, ppv_kind="vertical", decimal=decimal)["ppv_mm_s"].tolist() == [value]


def test_unambiguous_marks_are_still_accepted_where_both_are(tmp_path):
    p = _csv(tmp_path, "Dist\tCarga\tPPV\n100\t20\t0,850\n120,5\t20\t12.5\n1250\t20\t1.2500\n", name="t.csv")
    df = read_events(p, columns=COLS, ppv_kind="vertical")
    assert df["ppv_mm_s"].tolist() == [0.85, 12.5, 1.25] and df["distance_m"].tolist() == [100.0, 120.5, 1250.0]


def test_ppv_kind_as_argument_and_column_is_an_error(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV;Tipo\n100;10;1;vertical\n")
    with pytest.raises(ValueError, match=r"parameter 'ppv_kind'.*'Tipo'"):
        read_events(p, columns={**COLS, "ppv_kind": "Tipo"}, ppv_kind="vector_sum")


def test_duplicated_header_is_a_domain_error(tmp_path):
    p = _csv(tmp_path, "Dist;Carga;PPV;PPV\n100;10;1;2\n")
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter 'source'.*'PPV'.*twice"):
        read_events(p, columns=COLS, ppv_kind="vertical")
    # Un encabezado repetido que no se mapea (p. ej. columnas vacías al final) no estorba.
    p = _csv(tmp_path, "Dist;Carga;PPV;;\n100;10;1;;\n", name="g.csv")
    assert read_events(p, columns=COLS, ppv_kind="vertical")["ppv_mm_s"].tolist() == [1.0]


def test_missing_openpyxl_is_not_a_domain_error(tmp_path, monkeypatch):
    p = _xlsx(tmp_path, [(1, 1, 1)])
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises(ImportError) as info:
        read_events(p, columns=COLS, ppv_kind="vertical")
    assert not str(info.value).startswith("Domain error") and "openpyxl" in str(info.value)
