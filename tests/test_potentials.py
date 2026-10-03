import numpy as np
import pytest

from src import potentials as pot


X = np.linspace(-2.0, 2.0, 41)


def test_harmonic_force_matches_finite_difference():
    num = -pot.numerical_derivative(lambda x: pot.harmonic_potential(x, k=3.0), X)
    np.testing.assert_allclose(pot.harmonic_force(X, k=3.0), num, atol=1e-6)


@pytest.mark.parametrize("a, b", [(2.0, 0.5), (1.0, 0.0), (3.0, -0.8)])
def test_double_well_gradient_matches_finite_difference(a, b):
    num = pot.numerical_derivative(lambda x: pot.double_well_potential(x, a, b), X)
    np.testing.assert_allclose(pot.double_well_gradient(X, a, b), num, atol=1e-5)
    np.testing.assert_allclose(pot.double_well_force(X, a, b), -num, atol=1e-5)


def test_landmarks_are_stationary_and_ordered():
    lm = pot.double_well_landmarks(2.0, 0.5)
    for key in ("x_left", "x_barrier", "x_right"):
        assert abs(pot.double_well_gradient(lm[key], 2.0, 0.5)) < 1e-10
    assert lm["x_left"] < lm["x_barrier"] < lm["x_right"]
    assert lm["V_left"] < lm["V_right"] < lm["V_barrier"]  # b > 0 tilts the right well up


def test_single_well_is_rejected():
    with pytest.raises(ValueError):
        pot.double_well_landmarks(0.1, 2.0)


def test_true_free_energy_minimum_is_zero():
    x = np.linspace(-2, 2, 20001)
    F = pot.true_free_energy(x, 2.0, 0.5)
    assert F.min() == pytest.approx(0.0, abs=1e-6)
    assert np.all(F >= -1e-12)
