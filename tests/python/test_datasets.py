import pytest

from blastingpy import attenuation
from blastingpy.datasets import synthetic_events

COLUMNS = ["distance_m", "charge_kg", "ppv_mm_s", "ppv_kind", "frequency_hz"]


def test_same_seed_same_table_other_seed_differs():
    a, b, c = synthetic_events(seed=1), synthetic_events(seed=1), synthetic_events(seed=2)
    assert a.equals(b)
    assert not a.equals(c)


def test_canonical_columns_and_positive():
    ev = synthetic_events(n=25)
    assert list(ev.columns) == COLUMNS and len(ev) == 25
    assert (ev.ppv_kind == "component_max").all()
    assert (ev[["distance_m", "charge_kg", "ppv_mm_s", "frequency_hz"]] > 0).all().all()
    assert ev.distance_m.between(50, 2000).all() and ev.frequency_hz.between(5, 60).all()


@pytest.mark.parametrize("law", ["sqrt", "cbrt"])
def test_noiseless_fit_recovers_parameters(law):
    f = attenuation.fit(synthetic_events(law, sigma=0.0, k=800.0, beta=1.4), law)
    assert f.k == pytest.approx(800.0, rel=1e-6) and f.beta == pytest.approx(1.4, rel=1e-6)


@pytest.mark.parametrize("law", ["sqrt", "cbrt"])
def test_noisy_fit_recovers_beta_and_sigma(law):
    f = attenuation.fit(synthetic_events(law, n=400, sigma=0.15, seed=3), law)
    assert f.beta == pytest.approx(1.6, abs=0.05) and f.sigma == pytest.approx(0.15, abs=0.02)


@pytest.mark.parametrize("kwargs", [
    dict(law="general"), dict(n=0), dict(k=0.0), dict(beta=-1.0), dict(sigma=-0.1),
    dict(distance_range=(10.0, 10.0)), dict(charge_range=(0.0, 5.0)), dict(frequency_range=(60.0, 5.0)),
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError, match=r"Domain error \[BlastingPy\]: parameter"):
        synthetic_events(**kwargs)
