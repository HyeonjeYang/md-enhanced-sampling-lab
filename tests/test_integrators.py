import numpy as np
import pytest

from src import integrators as integ
from src.potentials import harmonic_force, harmonic_potential

K, M = 1.0, 1.0


def energy(x, v):
    return 0.5 * M * v**2 + harmonic_potential(x, K)


def force(x):
    return harmonic_force(x, K)


@pytest.mark.parametrize("method", ["velocity_verlet", "leapfrog"])
def test_symplectic_integrators_conserve_energy(method):
    dt, n = 0.01, 6300  # ~10 periods
    out = getattr(integ, method)(1.0, 0.0, force, M, dt, n)
    E = energy(out[0], out[1])
    assert np.max(np.abs(E - E[0])) / E[0] < 1e-4


def test_euler_energy_grows():
    x, v = integ.euler(1.0, 0.0, force, M, 0.05, 400)
    E = energy(x, v)
    # Each explicit Euler step multiplies the oscillator energy by (1 + w^2 dt^2).
    np.testing.assert_allclose(E[-1] / E[0], (1 + 0.05**2) ** 400, rtol=1e-10)


def test_leapfrog_positions_equal_velocity_verlet():
    x_vv, v_vv = integ.velocity_verlet(0.3, 0.7, force, M, 0.05, 500)
    x_lf, v_lf, _ = integ.leapfrog(0.3, 0.7, force, M, 0.05, 500)
    np.testing.assert_allclose(x_lf, x_vv, atol=1e-12)
    np.testing.assert_allclose(v_lf, v_vv, atol=1e-12)


def test_integrators_accept_arrays():
    x0 = np.array([[1.0, 0.0], [0.0, -1.0]])
    x, v = integ.velocity_verlet(x0, np.zeros_like(x0), force, M, 0.01, 10)
    assert x.shape == (11, 2, 2)


def test_unstable_timestep_raises():
    with pytest.raises(FloatingPointError), np.errstate(all="ignore"):
        integ.velocity_verlet(1.0, 0.0, force, M, 2.5, 5000)  # w dt > 2 is unstable
