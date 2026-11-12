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


def test_two_predictors_with_noise_match_numpy():
    rng = np.random.default_rng(7)
    n = 20
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 1.5 + 0.8 * x1 - 1.2 * x2 + rng.normal(0, 0.5, n)
    r = core.ols([x1.tolist(), x2.tolist()], y.tolist())
    design = np.column_stack([np.ones(n), x1, x2])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    assert r.coefficients == pytest.approx(beta, rel=1e-9)
    assert r.sse == pytest.approx(sse, rel=1e-9)
    assert r.sigma == pytest.approx(np.sqrt(sse / (n - 3)), rel=1e-9)
    assert r.r_squared == pytest.approx(1 - sse / sst, rel=1e-9)
    assert r.residuals == pytest.approx(resid, abs=1e-9)
