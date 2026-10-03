import numpy as np
import pytest

from src import metadynamics as md
from src.potentials import double_well_force, numerical_derivative

GRID = np.linspace(-2.5, 2.5, 501)


def _run(seed=0, hill_height=0.1, n_steps=5000, stride=50):
    return md.run_wt_metadynamics(
        -1.0, lambda x: double_well_force(x), n_steps, 0.002, 0.4, np.random.default_rng(seed),
        hill_height=hill_height, hill_sigma=0.1, deposition_stride=stride, bias_factor=8.0,
        x_grid=GRID, traj_stride=1,
    )


def test_hill_attenuation():
    kdT = md.delta_T_from_bias_factor(0.4, 8.0)
    assert kdT == pytest.approx(7 * 0.4)
    assert md.wt_hill_height(0.0, 0.1, kdT) == pytest.approx(0.1)
    assert md.wt_hill_height(kdT * np.log(2.0), 0.1, kdT) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        md.delta_T_from_bias_factor(0.4, 1.0)


def test_gaussian_hill_derivative():
    x = np.linspace(-1.0, 1.5, 37)
    _, dg = md.gaussian_hill(x, 0.3, 0.2, 0.15)
    num = numerical_derivative(lambda y: md.gaussian_hill(y, 0.3, 0.2, 0.15)[0], x, h=1e-6)
    np.testing.assert_allclose(dg, num, atol=1e-8)
    assert md.gaussian_hill(np.array([0.3]), 0.3, 0.2, 0.15)[0][0] == pytest.approx(0.2)


def test_bias_arrays_are_finite_and_consistent():
    r = _run()
    assert np.all(np.isfinite(r["bias"])) and r["bias"].max() > 0
    # Stored bias and derivative equal the sums over all deposited hills.
    hills = [md.gaussian_hill(GRID, c, w, 0.1) for c, w in zip(r["hill_center"], r["hill_height"])]
    np.testing.assert_allclose(r["bias"], sum(h[0] for h in hills), atol=1e-12)
    np.testing.assert_allclose(r["bias_derivative"], sum(h[1] for h in hills), atol=1e-12)
    # Well-tempered heights never exceed w0 and decrease as bias accumulates.
    assert np.all(r["hill_height"] <= 0.1 + 1e-15)
    assert r["hill_height"][-1] < r["hill_height"][0]


def test_bias_force_enters_the_dynamics():
    """Same noise, with and without hills: identical until the first deposition only."""
    with_bias = _run(seed=4, hill_height=0.5)["x"]
    without = _run(seed=4, hill_height=0.0)["x"]
    np.testing.assert_array_equal(with_bias[:51], without[:51])  # first hill at step 50
    assert np.max(np.abs(with_bias[52:] - without[52:])) > 1e-3


def test_reconstruction_formula():
    bias = np.array([0.0, 1.0, 3.5, 2.0])
    F = md.metad_free_energy(bias, bias_factor=8.0)
    np.testing.assert_allclose(F, -8.0 / 7.0 * bias - (-8.0 / 7.0 * 3.5))
    assert F.min() == 0.0
