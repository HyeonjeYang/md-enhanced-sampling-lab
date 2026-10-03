"""Grid-based well-tempered metadynamics on a 1D coordinate (Barducci et al. 2008).

The bias and its analytical derivative live on a fixed grid. Each deposition
adds one Gaussian (vectorized over the grid); each dynamics step only
interpolates the stored derivative, so the cost does not grow with the
number of hills.
"""

import warnings

import numpy as np


def delta_T_from_bias_factor(T, bias_factor):
    """Delta T = (gamma_WT - 1) T, with gamma_WT = (T + Delta T) / T."""
    if bias_factor <= 1:
        raise ValueError("bias_factor must be > 1 for well-tempered metadynamics.")
    return (bias_factor - 1.0) * T


def wt_hill_height(bias_here, initial_height, kB_delta_T):
    """w = w0 exp(-V_bias(s(t), t) / (k_B Delta T))."""
    return initial_height * np.exp(-bias_here / kB_delta_T)


def gaussian_hill(x_grid, center, height, sigma):
    """Gaussian hill and its derivative d/dx evaluated on the grid."""
    dx = x_grid - center
    g = height * np.exp(-0.5 * dx**2 / sigma**2)
    return g, -dx / sigma**2 * g


def metad_free_energy(bias, bias_factor):
    """F(s) = -gamma/(gamma - 1) V_bias(s), shifted so that min F = 0.

    The estimate is meaningful only where the walker has actually deposited
    bias; outside that region it is a flat artefact.
    """
    F = -bias_factor / (bias_factor - 1.0) * np.asarray(bias)
    return F - F.min()


def run_wt_metadynamics(
    x0,
    force,
    n_steps,
    dt,
    kT,
    rng,
    *,
    hill_height,
    hill_sigma,
    deposition_stride,
    bias_factor,
    x_grid,
    friction=1.0,
    traj_stride=10,
    snapshot_stride=None,
):
    """Overdamped Langevin dynamics on V(x) + V_bias(x, t).

    `force` is the physical force -dV/dx. The bias force -dV_bias/dx is
    interpolated from the grid and added at every step, so the accumulated
    bias changes the dynamics. Outside the grid the bias force is zero and
    no hills are deposited (counted in `n_outside_grid`, with a warning).

    Returns a dict with the strided trajectory, final bias, hill history and
    (optionally) sparse bias snapshots for animation.
    """
    kB_delta_T = delta_T_from_bias_factor(kT, bias_factor)  # k_B = 1
    x_grid = np.asarray(x_grid, dtype=float)
    bias = np.zeros_like(x_grid)
    dbias = np.zeros_like(x_grid)
    lo, hi = x_grid[0], x_grid[-1]

    x = float(x0)
    noise = np.sqrt(2.0 * kT * dt / friction) * rng.standard_normal(n_steps)
    traj = np.empty(n_steps // traj_stride + 1)
    traj[0] = x
    hills_t, hills_x, hills_w = [], [], []
    snaps_t, snaps_x, snaps_bias = [0.0], [x], [bias.copy()]
    n_outside = 0

    for step in range(1, n_steps + 1):
        f_bias = -np.interp(x, x_grid, dbias, left=0.0, right=0.0)
        x += (float(force(x)) + f_bias) / friction * dt + noise[step - 1]

        if step % deposition_stride == 0:
            if lo <= x <= hi:
                w = wt_hill_height(np.interp(x, x_grid, bias), hill_height, kB_delta_T)
                g, dg = gaussian_hill(x_grid, x, w, hill_sigma)
                bias += g
                dbias += dg
                hills_t.append(step * dt)
                hills_x.append(x)
                hills_w.append(w)
            else:
                n_outside += 1
        if step % traj_stride == 0:
            traj[step // traj_stride] = x
        if snapshot_stride and step % snapshot_stride == 0:
            snaps_t.append(step * dt)
            snaps_x.append(x)
            snaps_bias.append(bias.copy())
        if not np.isfinite(x):
            raise FloatingPointError(f"MetaD walker diverged at step {step}; reduce dt.")

    if not np.all(np.isfinite(bias)):
        raise FloatingPointError("Non-finite metadynamics bias.")
    if n_outside:
        warnings.warn(f"{n_outside} depositions skipped because the walker was outside the "
                      "bias grid; widen x_grid.", RuntimeWarning, stacklevel=2)
    return {
        "time": np.arange(len(traj)) * traj_stride * dt,
        "x": traj,
        "x_grid": x_grid,
        "bias": bias,
        "bias_derivative": dbias,
        "hill_time": np.asarray(hills_t),
        "hill_center": np.asarray(hills_x),
        "hill_height": np.asarray(hills_w),
        "snapshot_time": np.asarray(snaps_t),
        "snapshot_x": np.asarray(snaps_x),
        "snapshot_bias": np.asarray(snaps_bias),
        "n_outside_grid": n_outside,
    }
