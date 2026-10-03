"""Overdamped Langevin (Brownian) dynamics.

    dx = -(1/zeta) dU/dx dt + sqrt(2 k_B T dt / zeta) * eta,   eta ~ N(0, 1)

integrated with the Euler-Maruyama scheme. `friction` is zeta; the name
`gamma` is avoided because it is used for the WT-MetaD bias factor.
"""

import numpy as np


def brownian_dynamics(x0, force, n_steps, dt, kT, rng, friction=1.0, stride=1):
    """Integrate independent overdamped walkers.

    Parameters
    ----------
    x0 : float or array
        Initial positions; an array runs independent walkers in parallel.
    force : callable
        force(x) = -dU/dx, vectorized over walkers.
    kT : float or array
        Thermal energy, broadcastable to x0 (e.g. one temperature per walker).
    stride : int
        Store every `stride`-th configuration (the initial one is included).

    Returns an array of shape (n_steps // stride + 1, *x0.shape).
    """
    if dt <= 0 or friction <= 0:
        raise ValueError("dt and friction must be positive.")
    x = np.array(x0, dtype=float)
    noise = np.sqrt(2.0 * np.asarray(kT, dtype=float) * dt / friction)
    traj = np.empty((n_steps // stride + 1,) + x.shape)
    traj[0] = x
    for step in range(1, n_steps + 1):
        x = x + force(x) / friction * dt + noise * rng.standard_normal(x.shape)
        if step % stride == 0:
            traj[step // stride] = x
    if not np.all(np.isfinite(traj)):
        raise FloatingPointError("Brownian dynamics diverged; reduce dt.")
    return traj
