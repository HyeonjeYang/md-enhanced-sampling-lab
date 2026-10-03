import numpy as np
import pytest

from src import analysis
from src.langevin import brownian_dynamics
from src.potentials import harmonic_force


def test_brownian_dynamics_samples_harmonic_equilibrium():
    k, kT, dt = 2.0, 0.5, 0.005
    rng = np.random.default_rng(3)
    traj = brownian_dynamics(np.zeros(32), lambda x: harmonic_force(x, k), 40_000, dt, kT, rng, stride=20)
    samples = traj[200:].ravel()
    # <x^2> = kT / k (Euler-Maruyama bias at k dt = 0.01 is ~0.5%).
    assert np.mean(samples**2) == pytest.approx(kT / k, rel=0.05)
    assert abs(np.mean(samples)) < 0.02


def test_brownian_dynamics_is_reproducible():
    run = lambda seed: brownian_dynamics(0.0, lambda x: -x, 100, 0.01, 1.0, np.random.default_rng(seed))
    np.testing.assert_array_equal(run(5), run(5))


def test_divergence_is_reported():
    with pytest.raises(FloatingPointError), np.errstate(all="ignore"):
        brownian_dynamics(3.0, lambda x: -x**3, 200, 1.0, 0.1, np.random.default_rng(0))


def test_transition_counting_uses_core_states():
    traj = np.array([-1, -0.2, 0.1, -0.1, 0.2, 1.0, 0.0, 1.2, -1.1, -0.9])
    assert analysis.count_transitions(traj, -0.8, 0.8) == 2
    both = np.column_stack([traj, -traj])
    np.testing.assert_array_equal(analysis.count_transitions(both, -0.8, 0.8), [2, 2])


def test_histogram_free_energy_marks_empty_bins():
    samples = np.r_[np.full(100, 0.1), np.full(10, 0.9)]
    centers, F, counts = analysis.histogram_free_energy(samples, np.linspace(0, 1, 6), kT=1.0)
    assert np.isnan(F[1]) and counts[1] == 0
    assert F[0] == 0.0
    assert F[4] == pytest.approx(np.log(10.0))


def test_weighted_histogram_matches_unweighted_for_equal_weights():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    bins = np.linspace(-3, 3, 31)
    _, F1, _ = analysis.histogram_free_energy(s, bins, 1.0)
    _, F2, _ = analysis.histogram_free_energy(s, bins, 1.0, log_weights=np.full(s.size, 7.0))
    np.testing.assert_allclose(F1, F2, atol=1e-10)


def test_kish_ess():
    assert analysis.kish_effective_sample_size(np.zeros(50)) == pytest.approx(50.0)
    assert analysis.kish_effective_sample_size(np.r_[0.0, np.full(49, -1e3)]) == pytest.approx(1.0)
