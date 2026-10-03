"""Analytical toy potentials and their derivatives (reduced units, k_B = 1)."""

import numpy as np


# --- System A: 1D harmonic oscillator ---------------------------------------

def harmonic_potential(x, k=1.0):
    """V(x) = k x^2 / 2."""
    return 0.5 * k * np.asarray(x) ** 2


def harmonic_force(x, k=1.0):
    """F(x) = -dV/dx = -k x."""
    return -k * np.asarray(x)


# --- System C: asymmetric 1D double well ------------------------------------

def double_well_potential(x, a=2.0, b=0.5):
    """V(x) = a (x^2 - 1)^2 + b x."""
    x = np.asarray(x)
    return a * (x**2 - 1.0) ** 2 + b * x


def double_well_gradient(x, a=2.0, b=0.5):
    """dV/dx = 4 a x (x^2 - 1) + b."""
    x = np.asarray(x)
    return 4.0 * a * x * (x**2 - 1.0) + b


def double_well_force(x, a=2.0, b=0.5):
    """F(x) = -dV/dx."""
    return -double_well_gradient(x, a, b)


def double_well_landmarks(a=2.0, b=0.5):
    """Locate the two minima and the barrier top of the double well.

    Solves dV/dx = 0 (a cubic) and classifies the roots by curvature.
    Returns a dict with positions, energies, and the barrier heights seen
    from each minimum. Raises ValueError if the potential has only one minimum.
    """
    roots = np.roots([4.0 * a, 0.0, -4.0 * a, b])
    roots = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
    if roots.size != 3:
        raise ValueError(f"a={a}, b={b} gives a single well, not a double well.")
    x_left, x_barrier, x_right = roots
    V = double_well_potential(roots, a, b)
    return {
        "x_left": x_left,
        "x_barrier": x_barrier,
        "x_right": x_right,
        "V_left": V[0],
        "V_barrier": V[1],
        "V_right": V[2],
        "V_min": V[[0, 2]].min(),
        "barrier_from_left": V[1] - V[0],
        "barrier_from_right": V[1] - V[2],
    }


def true_free_energy(x, a=2.0, b=0.5):
    """Reference free energy of the 1D toy model, F(x) = V(x) - min V.

    This identity holds only because x is the *only* coordinate (or is
    decoupled from every other one). For a collective variable of a
    many-body system, F(s) is not the potential energy.
    """
    return double_well_potential(x, a, b) - double_well_landmarks(a, b)["V_min"]


# --- Utilities --------------------------------------------------------------

def numerical_derivative(f, x, h=1e-5):
    """Central finite difference (f(x+h) - f(x-h)) / 2h."""
    x = np.asarray(x, dtype=float)
    return (f(x + h) - f(x - h)) / (2.0 * h)
