import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from blastingpy._blastingpy_core import _least_squares as core

finite = st.floats(-50, 50, allow_nan=False, allow_infinity=False)


@settings(deadline=None)
@given(arrays(float, 12, elements=finite), arrays(float, 12, elements=finite))
def test_matches_numpy(x, y):
    assume(np.ptp(x) > 1e-3)
    r = core.ols([x.tolist()], y.tolist())
    expected = np.linalg.lstsq(np.column_stack([np.ones(12), x]), y, rcond=None)[0]
    assert r.coefficients == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_two_predictors_and_fields():
    x1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    x2 = [1.0, 0.0, 2.0, 1.0, 3.0, 2.0]
    y = [1 + 2 * a - 0.5 * b for a, b in zip(x1, x2)]
    r = core.ols([x1, x2], y)
    assert r.coefficients == pytest.approx([1.0, 2.0, -0.5], abs=1e-10)
    assert r.dof == 3
    assert r.r_squared == pytest.approx(1.0)
    assert len(r.residuals) == 6
    with pytest.raises(AttributeError):
        r.sse = 1.0


def test_domain_errors():
    with pytest.raises(ValueError, match="collinear or constant"):
        core.ols([[1.0, 1.0, 1.0, 1.0]], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Domain error"):
        core.ols([[1.0, 2.0]], [1.0, 2.0])
