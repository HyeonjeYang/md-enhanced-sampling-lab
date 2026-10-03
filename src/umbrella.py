"""Umbrella sampling with harmonic restraints U_i(x) = k_i (x - x_i)^2 / 2."""

import numpy as np

from .langevin import brownian_dynamics


def umbrella_bias(x, center, spring_constant):
    """U_i(x) = k_i (x - x_i)^2 / 2."""
    return 0.5 * spring_constant * (np.asarray(x) - center) ** 2


def umbrella_force(x, center, spring_constant):
    """-dU_i/dx = -k_i (x - x_i)."""
    return -spring_constant * (np.asarray(x) - center)


def run_umbrella_windows(
    centers,
    spring_constant,
    force,
    n_equil,
    n_prod,
    dt,
    kT,
    rng,
    friction=1.0,
    sample_stride=10,
):
    """Run every window as an independent overdamped walker started at its centre.

    All windows are integrated together as one vectorized array (they never
    interact). The first `n_equil` steps are discarded as burn-in.

    Returns a dict with `samples` of shape (n_samples, n_windows),
    `centers` and `spring_constants` (one per window).
    """
    centers = np.asarray(centers, dtype=float)
    k = np.broadcast_to(np.asarray(spring_constant, dtype=float), centers.shape).copy()

    def biased_force(x):
        return force(x) + umbrella_force(x, centers, k)

    equil = brownian_dynamics(centers, biased_force, n_equil, dt, kT, rng, friction, stride=max(n_equil, 1))
    prod = brownian_dynamics(equil[-1], biased_force, n_prod, dt, kT, rng, friction, stride=sample_stride)
    return {"samples": prod[1:], "centers": centers, "spring_constants": k}
