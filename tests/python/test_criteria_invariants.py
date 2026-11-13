"""Invariantes del veredicto sobre una grilla de eventos sintéticos (una llamada a evaluate por perfil).

En `appendix-b-drywall`, 20 mm/s excede a 4 Hz (límite 19.05) y cumple a 50 Hz (50.8); 5 mm/s cumple en ambas.
"""

import itertools
import math

import pandas as pd
import pytest

from blastingpy.criteria import evaluate

PPV = ("ppv_mm_s", "ppv_t_mm_s", "ppv_v_mm_s", "ppv_l_mm_s")
FREQ = ("frequency_hz", "freq_t_hz", "freq_v_hz", "freq_l_hz")
COLUMNS = PPV + FREQ
GRID = {**{c: (None, 5.0, 20.0) for c in PPV}, **{c: (None, 4.0, 50.0) for c in FREQ}}
KINDS = ("component_max", "vertical", "vector_sum")
CURVE = [{"kind": "constant_displacement", "f_min": 1.0, "f_max": "intersection", "value": 0.030},
         {"kind": "constant_velocity", "f_max": "intersection", "value": 0.75},
         {"kind": "constant_displacement", "f_max": "intersection", "value": 0.008},
         {"kind": "constant_velocity", "f_max": 100.0, "value": 2.0}]
VERTICAL = {"profile_id": "U-VERTICAL", "quantity": "vertical", "units": {"velocity": "in/s", "displacement": "in"},
            "criteria": [{"id": "c", "kind": "frequency_curve", "segments": CURVE}]}
PROFILES = {"component_max": ("USBM-RI8507-1980", "appendix-b-drywall"), "vertical": (VERTICAL, "c")}


def _grid():
    return [(kind, *values) for kind in KINDS for values in itertools.product(*(GRID[c] for c in COLUMNS))]


@pytest.fixture(scope="module")
def statuses():
    keys = _grid()
    frame = pd.DataFrame([{"ppv_kind": k[0], **{c: (math.nan if v is None else v) for c, v in zip(COLUMNS, k[1:])}}
                          for k in keys])
    out = {}
    for name, (profile, criterion) in PROFILES.items():
        out[name] = dict(zip(keys, evaluate(frame, profile, criterion)["status"]))
    return out


@pytest.mark.parametrize("profile", list(PROFILES))
def test_filling_an_absent_column_never_turns_exceeds_into_complies(statuses, profile):
    status = statuses[profile]
    bad = []
    for key, s in status.items():
        if s != "exceeds":
            continue
        for i, column in enumerate(COLUMNS, start=1):
            if key[i] is not None:
                continue
            for value in GRID[column][1:]:
                filled = key[:i] + (value,) + key[i + 1:]
                if status[filled] == "complies":
                    bad.append((key, column, value))
    assert not bad, f"{len(bad)} cases, first: {bad[0]}"


def test_complies_needs_the_maximum_component(statuses):
    for key, s in statuses["component_max"].items():
        if s != "complies":
            continue
        kind, values = key[0], dict(zip(COLUMNS, key[1:]))
        vertical = values["ppv_v_mm_s"] is not None or (kind == "vertical" and values["ppv_mm_s"] is not None)
        three = values["ppv_t_mm_s"] is not None and vertical and values["ppv_l_mm_s"] is not None
        assert three or (kind == "component_max" and values["ppv_mm_s"] is not None), key


def test_fully_specified_compliant_events_comply(statuses):
    for kind in KINDS:
        for peak, f, ft, fv, fl in itertools.product((None, 5.0), (None, 4.0, 50.0), (4.0, 50.0), (4.0, 50.0),
                                                     (4.0, 50.0)):
            key = (kind, peak, 5.0, 5.0, 5.0, f, ft, fv, fl)
            for profile in PROFILES:
                assert statuses[profile][key] == "complies", (profile, key)
    for f in (4.0, 50.0):
        key = ("component_max", 5.0, None, None, None, f, None, None, None)
        assert statuses["component_max"][key] == "complies", key
