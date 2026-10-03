"""Explicit integrators for Newtonian dynamics m x'' = F(x).

Each function works for scalar positions or NumPy arrays of any shape and
returns arrays with a leading time axis of length n_steps + 1.
"""

import numpy as np


def _check_finite(x, name):
    if not np.all(np.isfinite(x)):
        raise FloatingPointError(
            f"{name} trajectory diverged (non-finite values). Reduce the timestep."
        )


def euler(x0, v0, force, mass, dt, n_steps):
    """Explicit Euler: x_{n+1} = x_n + v_n dt,  v_{n+1} = v_n + a_n dt."""
    x = np.empty((n_steps + 1,) + np.shape(x0))
    v = np.empty_like(x)
    x[0], v[0] = x0, v0
    for n in range(n_steps):
        a = force(x[n]) / mass
        x[n + 1] = x[n] + v[n] * dt
        v[n + 1] = v[n] + a * dt
    _check_finite(x, "Euler")
    return x, v


def leapfrog(x0, v0, force, mass, dt, n_steps):
    """Leapfrog with velocities at half steps.

        v_{n+1/2} = v_{n-1/2} + a_n dt
        x_{n+1}   = x_n + v_{n+1/2} dt

    The scheme is started with v_{-1/2} = v_0 - a_0 dt / 2.

    Returns
    -------
    x : positions x_n
    v : on-step velocities v_n = (v_{n-1/2} + v_{n+1/2}) / 2, needed for energies
    v_half : half-step velocities, v_half[n] = v_{n+1/2}
    """
    x = np.empty((n_steps + 1,) + np.shape(x0))
    v_half = np.empty_like(x)
    x[0] = x0
    v_prev = v0 - 0.5 * dt * force(x0) / mass  # v_{-1/2}
    v_start = v_prev
    for n in range(n_steps + 1):
        v_half[n] = v_prev + force(x[n]) / mass * dt
        if n < n_steps:
            x[n + 1] = x[n] + v_half[n] * dt
        v_prev = v_half[n]
    v_minus = np.concatenate([np.asarray(v_start)[None], v_half[:-1]])
    v = 0.5 * (v_minus + v_half)
    _check_finite(x, "Leapfrog")
    return x, v, v_half


def velocity_verlet_step(x, v, f, force, mass, dt):
    """One Velocity Verlet step. Returns new (x, v, f)."""
    x_new = x + v * dt + 0.5 * (f / mass) * dt**2
    f_new = force(x_new)
    v_new = v + 0.5 * (f + f_new) / mass * dt
    return x_new, v_new, f_new


def velocity_verlet(x0, v0, force, mass, dt, n_steps):
    """Velocity Verlet:

        x_{n+1} = x_n + v_n dt + a_n dt^2 / 2
        v_{n+1} = v_n + (a_n + a_{n+1}) dt / 2
    """
    x = np.empty((n_steps + 1,) + np.shape(x0))
    v = np.empty_like(x)
    x[0], v[0] = x0, v0
    f = force(x[0])
    for n in range(n_steps):
        x[n + 1], v[n + 1], f = velocity_verlet_step(x[n], v[n], f, force, mass, dt)
    _check_finite(x, "Velocity Verlet")
    return x, v
