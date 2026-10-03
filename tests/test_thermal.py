import numpy as np
import pytest

from src import thermal


def test_maxwell_boltzmann_variance_per_mass():
    rng = np.random.default_rng(0)
    masses = np.r_[np.full(5000, 1.0), np.full(5000, 4.0)]
    v = thermal.maxwell_boltzmann_velocities(masses, 1.5, masses.size, 2, rng)
    # Var(v) = kT/m; 10000 components per group -> ~1.4% statistical error.
    assert np.var(v[:5000]) == pytest.approx(1.5 / 1.0, rel=0.05)
    assert np.var(v[5000:]) == pytest.approx(1.5 / 4.0, rel=0.05)


def test_com_removal_zeroes_total_momentum():
    rng = np.random.default_rng(1)
    masses = rng.uniform(0.5, 3.0, 20)
    v = thermal.maxwell_boltzmann_velocities(masses, 1.0, 20, 2, rng) + 0.7
    assert np.linalg.norm(thermal.total_momentum(v, masses)) > 1.0
    v0 = thermal.remove_com_velocity(v, masses)
    np.testing.assert_allclose(thermal.total_momentum(v0, masses), 0.0, atol=1e-12)


def test_rescaling_reaches_target_temperature():
    rng = np.random.default_rng(2)
    n, d = 16, 2
    dof = thermal.degrees_of_freedom(n, d)
    v = thermal.remove_com_velocity(thermal.maxwell_boltzmann_velocities(1.0, 2.0, n, d, rng), 1.0)
    v, lam = thermal.rescale_to_temperature(v, 1.0, 0.8, dof)
    assert thermal.instantaneous_temperature(v, 1.0, dof) == pytest.approx(0.8, rel=1e-12)
    np.testing.assert_allclose(thermal.total_momentum(v, 1.0), 0.0, atol=1e-12)


def test_degrees_of_freedom():
    assert thermal.degrees_of_freedom(16, 2) == 30
    assert thermal.degrees_of_freedom(16, 2, remove_com=False) == 32
    assert thermal.degrees_of_freedom(10, 3, n_constraints=5) == 22
