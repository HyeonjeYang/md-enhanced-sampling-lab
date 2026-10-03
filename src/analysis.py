"""Shared analysis helpers: histogram free energies, error metrics, transitions."""

import numpy as np
from scipy.special import logsumexp


def histogram_free_energy(samples, bins, kT, log_weights=None, min_count=1):
    """F(x) = -kT ln P(x) from (optionally weighted) samples.

    Bins with fewer than `min_count` raw samples are returned as NaN rather
    than being assigned an artificial large free energy. Weights are passed
    as log-weights and combined with logsumexp to avoid overflow.

    Returns (bin_centers, F aligned so that min F = 0, raw counts).
    """
    samples = np.ravel(samples)
    edges = np.asarray(bins, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(samples, edges) - 1
    inside = (idx >= 0) & (idx < len(centers))
    idx = idx[inside]
    counts = np.bincount(idx, minlength=len(centers))

    if log_weights is None:
        log_hist = np.full(len(centers), -np.inf)
        log_hist[counts > 0] = np.log(counts[counts > 0])
    else:
        lw = np.ravel(log_weights)[inside]
        log_hist = np.full(len(centers), -np.inf)
        for j in np.flatnonzero(counts):
            log_hist[j] = logsumexp(lw[idx == j])

    widths = np.diff(edges)
    F = -kT * (log_hist - np.log(widths))
    F[counts < min_count] = np.nan
    return centers, align_free_energy(F), counts


def align_free_energy(F, mask=None):
    """Shift a profile so that its minimum (over `mask`, ignoring NaN) is zero."""
    F = np.asarray(F, dtype=float)
    sel = np.isfinite(F) if mask is None else np.isfinite(F) & mask
    if not np.any(sel):
        raise ValueError("No finite free-energy values to align.")
    return F - np.min(F[sel])


def free_energy_rmse(F_est, F_true, mask):
    """RMSE over points that are inside `mask` *and* actually estimated.

    Returns (rmse, fraction of masked points that were estimated). The
    coverage is reported so that an RMSE over a tiny region is not mistaken
    for a good reconstruction.
    """
    F_est = np.asarray(F_est, dtype=float)
    used = mask & np.isfinite(F_est)
    if not np.any(used):
        return np.nan, 0.0
    err = F_est[used] - np.asarray(F_true)[used]
    return float(np.sqrt(np.mean(err**2))), float(used.sum() / mask.sum())


def count_transitions(traj, left, right):
    """Count committed basin-to-basin transitions with core-state hysteresis.

    A transition is counted only when a walker leaves the core x < left and
    reaches the core x > right (or vice versa); recrossings of the barrier
    top that do not reach the other core are ignored.

    traj has time on axis 0; extra axes are independent walkers. Returns an
    int for a single walker or an array of counts per walker.
    """
    traj = np.asarray(traj)
    flat = traj.reshape(len(traj), -1)
    core = np.where(flat < left, -1, np.where(flat > right, 1, 0))
    # Forward-fill: between cores a walker keeps the label of the last core visited.
    last = np.where(core != 0, np.arange(len(core))[:, None], 0)
    last = np.maximum.accumulate(last, axis=0)
    state = np.take_along_axis(core, last, axis=0)
    changes = (state[1:] != state[:-1]) & (state[:-1] != 0)
    counts = changes.sum(axis=0)
    return int(counts[0]) if traj.ndim == 1 else counts.reshape(traj.shape[1:])


def kish_effective_sample_size(log_weights):
    """ESS = (sum w)^2 / sum w^2, computed in log space."""
    lw = np.ravel(log_weights)
    return float(np.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw)))
