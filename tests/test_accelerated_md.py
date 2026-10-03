import numpy as np
import pytest

from src import accelerated_md as amd
from src import analysis
from src.potentials import numerical_derivative

V = np.linspace(-1.0, 4.0, 501)
E = 2.0


@pytest.mark.parametrize("boost, args", [(amd.amd_boost, (0.5,)), (amd.gamd_boost, (0.3,))])
def test_boost_zero_above_positive_below_continuous_at_threshold(boost, args):
    dV = boost(V, E, *args)
    assert np.all(dV[V >= E] == 0.0)
    assert np.all(dV[V < E] > 0.0)
    assert boost(E - 1e-9, E, *args) == pytest.approx(0.0, abs=1e-8)


V_OFF = V[np.abs(V - E) > 1e-3]  # avoid straddling the threshold with the stencil


def test_amd_force_scale_is_derivative_of_modified_potential():
    num = numerical_derivative(lambda v: v + amd.amd_boost(v, E, 0.5), V_OFF, h=1e-6)
    np.testing.assert_allclose(amd.amd_force_scale(V_OFF, E, 0.5), num, atol=1e-7)
    assert amd.amd_force_scale(E, E, 0.5) == pytest.approx(1.0)  # smooth at threshold


def test_gamd_force_scale_is_derivative_of_modified_potential():
    num = numerical_derivative(lambda v: v + amd.gamd_boost(v, E, 0.3), V_OFF, h=1e-6)
    np.testing.assert_allclose(amd.gamd_force_scale(V_OFF, E, 0.3), num, atol=1e-7)
    assert amd.gamd_force_scale(E, E, 0.3) == pytest.approx(1.0)


@pytest.mark.parametrize("method, params", [("amd", dict(E=2.5, alpha=0.4)), ("gamd", dict(E=3.0, k=0.2))])
def test_modified_force_matches_finite_difference_in_all_coordinates(method, params):
    rng = np.random.default_rng(0)
    system = dict(a=2.0, b=0.5, k_bath=10.0)
    q = rng.normal(0.0, 0.4, (6, 4))  # x and three bath coordinates

    def v_star(qq):
        V0 = amd.total_potential(qq, **system)
        return V0 + amd.boost_and_scale(V0, method, **params)[0]

    _, scale = amd.boost_and_scale(amd.total_potential(q, **system), method, **params)
    analytic = scale[:, None] * amd.total_gradient(q, **system)
    h = 1e-6
    for d in range(q.shape[1]):
        dq = np.zeros_like(q)
        dq[:, d] = h
        num = (v_star(q + dq) - v_star(q - dq)) / (2 * h)
        np.testing.assert_allclose(analytic[:, d], num, rtol=1e-5, atol=1e-6)


def _stats(V_max=3.0, V_min=-0.5, V_avg=0.5, V_std=0.6):
    return dict(V_max=V_max, V_min=V_min, V_avg=V_avg, V_std=V_std)


@pytest.mark.parametrize("sigma0", [0.1, 0.3, 1.0, 10.0])
def test_gamd_lower_bound_parameters_satisfy_constraints(sigma0):
    s = _stats()
    p = amd.gamd_parameters(s, sigma0, "lower")
    assert p["E"] == s["V_max"]
    assert 0.0 < p["k0"] <= 1.0
    assert s["V_max"] <= p["E"] <= s["V_min"] + 1.0 / p["k"] + 1e-12
    assert p["sigma_dV_estimate"] <= sigma0 + 1e-12
    # Monotonic and smoother: 0 < dV*/dV <= 1 on [V_min, V_max].
    Vs = np.linspace(s["V_min"], s["V_max"], 200)
    slope = amd.gamd_force_scale(Vs, p["E"], p["k"])
    assert np.all(slope >= -1e-12) and np.all(slope <= 1.0)


def test_gamd_upper_bound_used_only_when_valid():
    s = _stats(V_std=0.6)
    p = amd.gamd_parameters(s, sigma0=0.5, threshold="upper")  # k0'' = (1 - 0.5/0.6) * 3.5 / 1.0
    assert p["threshold_used"] == "upper"
    assert p["E"] == pytest.approx(s["V_min"] + 1.0 / p["k"])
    assert p["sigma_dV_estimate"] == pytest.approx(0.5)  # sigma0 constraint is saturated
    # sigma0 > sigma_V gives k0'' < 0: fall back to the lower-bound rule.
    p2 = amd.gamd_parameters(s, sigma0=1.0, threshold="upper")
    assert p2["threshold_used"] == "lower" and p2["E"] == s["V_max"]


def test_invalid_gamd_parameters_are_rejected():
    s = _stats()
    with pytest.raises(ValueError):
        amd.check_gamd_constraints(E=2.0, k=0.1, stats=s)  # E < V_max
    with pytest.raises(ValueError):
        amd.check_gamd_constraints(E=3.0, k=1.0, stats=s)  # E > V_min + 1/k
    with pytest.raises(ValueError):
        amd.amd_boost(V, E, 0.0)


def test_boosted_dynamics_records_consistent_boost():
    r = amd.run_boosted_dynamics(np.full(2, -1.0), 2000, 0.002, 0.4, np.random.default_rng(1),
                                 method="gamd", E=3.0, k=0.2, n_bath=3, stride=10)
    np.testing.assert_allclose(r["dV"], amd.gamd_boost(r["V"], 3.0, 0.2))
    assert r["x"].shape == (201, 2)


def test_cumulant_reweighting_exact_for_gaussian_boost():
    """With Gaussian Delta V in a bin, ln<exp(beta dV)> = beta mu + beta^2 s^2 / 2 exactly."""
    rng = np.random.default_rng(5)
    kT, n = 0.4, 200_000
    x = np.r_[np.full(n, 0.25), np.full(n, 0.75)]
    dV = np.r_[rng.normal(1.0, 0.2, n), rng.normal(0.3, 0.4, n)]
    bins = np.array([0.0, 0.5, 1.0])
    _, F2, _ = amd.cumulant_free_energy(x, dV, bins, kT)
    _, Fexp, _ = analysis.histogram_free_energy(x, bins, kT, log_weights=dV / kT)
    expected = -(0.3 + 0.4**2 / (2 * kT)) + (1.0 + 0.2**2 / (2 * kT))  # F(bin 1) - F(bin 0)
    assert F2[1] - F2[0] == pytest.approx(expected, abs=1e-2)
    assert Fexp[1] - Fexp[0] == pytest.approx(expected, abs=2e-2)


def test_cumulant_reweighting_leaves_sparse_bins_empty():
    _, F, counts = amd.cumulant_free_energy(np.r_[np.zeros(50), [0.9]], np.zeros(51), [0, 0.5, 1.0], 1.0)
    assert counts[1] == 1 and np.isnan(F[1]) and F[0] == 0.0
