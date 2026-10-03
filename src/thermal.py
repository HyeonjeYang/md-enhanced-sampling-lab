"""Thermal velocity initialization and kinetic temperature (k_B = 1 by default).

Velocities are arrays of shape (N, d); masses are scalars or arrays of shape (N,).
"""

import numpy as np


def _masses(masses, n):
    return np.broadcast_to(np.asarray(masses, dtype=float), (n,))


def maxwell_boltzmann_velocities(masses, T, n_particles, dim, rng, kB=1.0):
    """Draw v_{i,alpha} ~ N(0, kB T / m_i) independently for every component."""
    m = _masses(masses, n_particles)
    sigma = np.sqrt(kB * T / m)
    return rng.standard_normal((n_particles, dim)) * sigma[:, None]


def total_momentum(v, masses):
    """P = sum_i m_i v_i (vector of length d)."""
    m = _masses(masses, len(v))
    return (m[:, None] * v).sum(axis=0)


def remove_com_velocity(v, masses):
    """Subtract V_COM = sum_i m_i v_i / sum_i m_i from every particle."""
    m = _masses(masses, len(v))
    v_com = total_momentum(v, m) / m.sum()
    return v - v_com


def kinetic_energy(v, masses):
    """K = sum_i m_i |v_i|^2 / 2."""
    m = _masses(masses, len(v))
    return 0.5 * np.sum(m[:, None] * v**2)


def degrees_of_freedom(n_particles, dim, remove_com=True, n_constraints=0):
    """f = dN - d (COM removed) - number of holonomic constraints."""
    f = dim * n_particles - (dim if remove_com else 0) - n_constraints
    if f <= 0:
        raise ValueError("Number of degrees of freedom must be positive.")
    return f


def instantaneous_temperature(v, masses, dof, kB=1.0):
    """T_inst = 2 K / (f kB)."""
    return 2.0 * kinetic_energy(v, masses) / (dof * kB)


def rescale_to_temperature(v, masses, T_target, dof, kB=1.0):
    """Scale all velocities by lambda = sqrt(T_target / T_inst).

    After this, T_inst equals T_target *exactly*, so the natural
    kinetic-energy fluctuation of the Maxwell-Boltzmann draw is removed.
    Intended as an initialization step, not as a canonical thermostat.

    Returns (scaled velocities, lambda).
    """
    T_inst = instantaneous_temperature(v, masses, dof, kB)
    if not T_inst > 0:
        raise ValueError("Cannot rescale velocities with zero kinetic energy.")
    lam = np.sqrt(T_target / T_inst)
    return lam * v, lam
