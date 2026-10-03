import numpy as np
import pytest

from src import lj, thermal


def _state(seed=0, n_side=6, rho=0.5, T=1.0):
    rng = np.random.default_rng(seed)
    L = np.sqrt(n_side**2 / rho)
    x = lj.square_lattice(n_side, L) + rng.normal(0, 0.05, (n_side**2, 2))
    v = thermal.remove_com_velocity(thermal.maxwell_boltzmann_velocities(1.0, T, n_side**2, 2, rng), 1.0)
    return x, v, L


def test_forces_sum_to_zero_and_match_energy_gradient():
    x, _, L = _state()
    f, U, _ = lj.lj_forces(x, L)
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
    h = 1e-6
    for i, a in [(0, 0), (7, 1), (20, 0)]:
        xp, xm = x.copy(), x.copy()
        xp[i, a] += h
        xm[i, a] -= h
        num = -(lj.lj_forces(xp, L)[1] - lj.lj_forces(xm, L)[1]) / (2 * h)
        assert f[i, a] == pytest.approx(num, rel=1e-5, abs=1e-6)


def test_ideal_gas_pressure_without_interactions():
    L = 40.0
    x = lj.square_lattice(3, L)  # spacing >> r_cut: no pairs interact
    f, U, W = lj.lj_forces(x, L)
    assert U == 0.0 and W == 0.0
    assert lj.pressure(9, 1.3, W, L**2) == pytest.approx(9 * 1.3 / L**2)


def test_small_box_is_rejected():
    with pytest.raises(ValueError):
        lj.lj_forces(np.zeros((2, 2)) + [[0, 0], [1, 1]], 4.0, r_cut=2.5)


def test_nve_energy_is_approximately_conserved():
    x, v, L = _state()
    r = lj.run_lj(x, v, L, 1000, 0.002, sample_stride=10)
    drift = np.abs(r["total"] - r["total"][0]).max() / len(x)
    assert drift < 1e-3


def test_rescaling_holds_temperature_exactly():
    x, v, L = _state()
    r = lj.run_lj(x, v, L, 200, 0.005, thermostat="rescale", T_target=0.9, sample_stride=10)
    np.testing.assert_allclose(r["temperature"][1:], 0.9, rtol=1e-10)


def test_langevin_requires_rng():
    x, v, L = _state()
    with pytest.raises(ValueError):
        lj.run_lj(x, v, L, 10, 0.005, thermostat="langevin")
