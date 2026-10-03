import numpy as np
import pytest

from src import replica_exchange as rx


def test_tremd_acceptance_known_values():
    # Hotter slot j holds the lower energy: moving it to the cold slot is always accepted.
    assert rx.tremd_log_acceptance(2.0, 1.0, V_i=1.0, V_j=0.0) == 0.0
    assert np.exp(rx.tremd_log_acceptance(2.0, 1.0, V_i=0.0, V_j=1.0)) == pytest.approx(np.exp(-1.0))
    assert rx.tremd_log_acceptance(1.5, 1.5, 3.0, -2.0) == 0.0  # equal temperatures


def test_tremd_detailed_balance():
    rng = np.random.default_rng(0)
    for _ in range(100):
        bi, bj = rng.uniform(0.2, 3.0, 2)
        Vi, Vj = rng.normal(0, 2, 2)
        forward = np.exp(-bi * Vi - bj * Vj + rx.tremd_log_acceptance(bi, bj, Vi, Vj))
        backward = np.exp(-bi * Vj - bj * Vi + rx.tremd_log_acceptance(bi, bj, Vj, Vi))
        assert forward == pytest.approx(backward, rel=1e-12)


def test_general_criterion_reduces_to_tremd_and_hremd():
    bi, bj, Vi, Vj = 2.5, 1.2, 0.3, 1.7
    np.testing.assert_allclose(
        rx.exchange_log_acceptance(bi, bj, Vi, Vj, Vi, Vj), rx.tremd_log_acceptance(bi, bj, Vi, Vj))
    beta, u = 2.5, (0.1, 0.9, 0.4, 0.2)  # U_i(x_i), U_i(x_j), U_j(x_i), U_j(x_j)
    np.testing.assert_allclose(
        rx.exchange_log_acceptance(beta, beta, *u), rx.hremd_log_acceptance(beta, *u))


def test_hremd_cross_hamiltonian_expression():
    beta, lam_i, lam_j = 2.5, 1.0, 0.4
    xi, xj = np.array([[-1.0]]), np.array([[0.1]])  # i near a minimum, j near the barrier
    U = lambda q, lam: float(rx.ladder_potential(q, lam)[0])
    delta = beta * (U(xj, lam_i) + U(xi, lam_j) - U(xi, lam_i) - U(xj, lam_j))
    # Only the scaled term differs: Delta = beta (lam_i - lam_j) [A(x_j) - A(x_i)], A = a (x^2 - 1)^2.
    A = lambda x: 2.0 * (x**2 - 1.0) ** 2
    assert delta == pytest.approx(beta * (lam_i - lam_j) * (A(0.1) - A(-1.0)))
    log_acc = rx.hremd_log_acceptance(beta, U(xi, lam_i), U(xj, lam_i), U(xi, lam_j), U(xj, lam_j))
    assert log_acc == pytest.approx(-delta)


def test_metropolis_accept_frequency():
    rng = np.random.default_rng(1)
    assert rx.metropolis_accept(0.0, rng)
    assert not rx.metropolis_accept(-np.inf, rng)
    freq = np.mean([rx.metropolis_accept(np.log(0.3), rng) for _ in range(20000)])
    assert freq == pytest.approx(0.3, abs=0.015)


def test_identical_replicas_always_exchange():
    r = rx.run_replica_exchange(-1.0, np.full(4, 0.5), 1.0, 2000, 20, 0.002, np.random.default_rng(2))
    assert np.all(r["accepts"] == r["attempts"]) and np.all(r["attempts"] > 0)
    for row in r["walker_slot"]:
        assert sorted(row) == [0, 1, 2, 3]


def test_walker_trajectories_are_continuous():
    r = rx.run_replica_exchange(-1.0, rx.geometric_temperatures(0.4, 1.6, 4), 1.0, 20000, 50, 0.002,
                                np.random.default_rng(3), stride=1)
    walkers = rx.walker_trajectories(r)
    # Per step a walker moves by ~sqrt(2 kT dt) <~ 0.08; slot traces jump on swaps.
    assert np.max(np.abs(np.diff(walkers, axis=0))) < 0.6
    np.testing.assert_allclose(np.sort(walkers, axis=1), np.sort(r["x_slot"], axis=1))


def test_round_trip_counting():
    assert rx.count_round_trips([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 3, 0], 4) == 2
    assert rx.count_round_trips([1, 2, 3, 0], 4) == 0  # must start from the bottom
    assert rx.count_round_trips([0, 1, 2, 1, 0], 4) == 0  # never reached the top
