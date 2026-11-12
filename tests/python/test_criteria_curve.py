import math

import pytest

from blastingpy._blastingpy_core import _criteria_curve as cc

NAN = float("nan")


def _curve():
    # Curva alternativa de USBM RI 8507, Apéndice B (en pulgadas), armada desde Python.
    f1 = cc.intersection_frequency(0.75, 0.030)
    f2 = cc.intersection_frequency(0.75, 0.008)
    f3 = cc.intersection_frequency(2.0, 0.008)
    return [
        cc.Segment(cc.SegmentKind.ConstantDisplacement, 1.0, f1, 0.030),
        cc.Segment(cc.SegmentKind.ConstantVelocity, f1, f2, 0.75),
        cc.Segment(cc.SegmentKind.ConstantDisplacement, f2, f3, 0.008),
        cc.Segment(cc.SegmentKind.ConstantVelocity, f3, 100.0, 2.0),
    ]


def test_four_segment_curve_values():
    c = _curve()
    assert cc.limit_at(c, 2.0).limit == pytest.approx(0.376991, abs=1e-5)
    assert cc.limit_at(c, 20.0).limit == pytest.approx(1.005310, abs=1e-5)
    r = cc.limit_at(c, 50.0)
    assert (r.defined, r.limit, r.segment) == (True, 2.0, 3)


def test_segment_fields_are_read_only():
    s = _curve()[0]
    assert math.isnan(s.v2) and s.kind == cc.SegmentKind.ConstantDisplacement
    with pytest.raises(AttributeError):
        s.v1 = 1.0


def test_out_of_domain_is_undefined():
    r = cc.limit_at(_curve(), 150.0)
    assert not r.defined and math.isnan(r.limit) and r.segment == -1


def test_invalid_curve_raises_value_error():
    gap = [
        cc.Segment(cc.SegmentKind.ConstantVelocity, 1.0, 40.0, 0.75),
        cc.Segment(cc.SegmentKind.ConstantVelocity, 50.0, 100.0, 2.0),
    ]
    with pytest.raises(ValueError, match="parameter 'segments'"):
        cc.limit_at(gap, 10.0)
