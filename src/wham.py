"""Iterative 1D weighted histogram analysis method (WHAM), in log space.

For windows i with bias energies U_i(x) and histogram counts n_ij in bins j,
the self-consistent WHAM equations (Kumar et al. 1992) are

    p_j = sum_i n_ij / sum_i N_i exp(f_i - beta U_i(x_j))
    f_i = -ln sum_j p_j exp(-beta U_i(x_j))

where f_i are dimensionless window free-energy offsets. Both lines are
evaluated with logsumexp. Bins that no window visited get p_j = 0 and an
undefined (NaN) free energy.
"""

import warnings

import numpy as np
from scipy.special import logsumexp


def neighbor_overlaps(counts):
    """Overlap sum_j min(h_ij, h_{i+1,j}) of normalized neighbouring histograms."""
    h = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    return np.minimum(h[:-1], h[1:]).sum(axis=1)


def wham_1d(
    samples,
    centers,
    spring_constants,
    kT,
    bins,
    tol=1e-8,
    max_iter=10000,
    min_overlap=0.03,
):
    """Combine harmonic umbrella windows into one unbiased profile.

    Parameters
    ----------
    samples : array (n_samples, n_windows) or list of 1D arrays, one per window
    centers, spring_constants : per-window restraint parameters
    bins : histogram bin edges
    tol : convergence threshold on max |f_i^new - f_i^old|
    min_overlap : warn if any neighbouring-window histogram overlap is below this

    Returns a dict with bin centres, normalized probability density,
    free energy (min = 0, NaN in empty bins), window offsets f_i * kT,
    convergence flag, iteration count, neighbour overlaps, and a `reliable`
    flag that is False whenever a warning was issued.
    """
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter must be >= 1 and tol must be positive.")
    if isinstance(samples, np.ndarray) and samples.ndim == 2:
        samples = [samples[:, i] for i in range(samples.shape[1])]
    centers = np.asarray(centers, dtype=float)
    k = np.broadcast_to(np.asarray(spring_constants, dtype=float), centers.shape)
    order = np.argsort(centers)
    samples = [np.asarray(samples[i]) for i in order]
    centers, k = centers[order], k[order]

    edges = np.asarray(bins, dtype=float)
    x = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    counts = np.array([np.histogram(s, edges)[0] for s in samples], dtype=float)
    issues = []

    n_outside = sum(s.size for s in samples) - counts.sum()
    if n_outside > 0:
        issues.append(f"{int(n_outside)} samples fell outside the histogram range and were ignored.")
    N = counts.sum(axis=1)
    if np.any(N == 0):
        raise ValueError("At least one window has no samples inside the histogram range.")

    beta = 1.0 / kT
    u = beta * 0.5 * k[:, None] * (x[None, :] - centers[:, None]) ** 2  # (windows, bins)
    total = counts.sum(axis=0)
    visited = total > 0
    log_num = np.where(visited, np.log(np.where(visited, total, 1.0)), -np.inf)
    log_N = np.log(N)

    f = np.zeros(len(centers))
    converged = False
    for n_iter in range(1, max_iter + 1):
        log_p = log_num - logsumexp(log_N[:, None] + f[:, None] - u, axis=0)
        f_new = -logsumexp(log_p[None, :] - u, axis=1)
        f_new -= f_new[0]
        if not np.all(np.isfinite(f_new)):
            raise FloatingPointError("WHAM produced non-finite window offsets.")
        delta = np.max(np.abs(f_new - f))
        f = f_new
        if delta < tol:
            converged = True
            break
    if not converged:
        issues.append(f"WHAM did not converge in {max_iter} iterations (last change {delta:.2e}).")

    log_p = log_p - logsumexp(log_p + np.log(width))  # normalize the density
    F = np.full_like(x, np.nan)
    F[visited] = -kT * log_p[visited]
    F -= np.nanmin(F)

    overlaps = neighbor_overlaps(counts)
    poor = np.flatnonzero(overlaps < min_overlap)
    if poor.size:
        pairs = ", ".join(f"({i},{i + 1}): {overlaps[i]:.3f}" for i in poor)
        issues.append(f"Poor overlap between neighbouring windows {pairs}; the profile is unreliable.")
    between = (x >= centers[0]) & (x <= centers[-1])
    if not np.all(visited[between]):
        issues.append("Empty bins between window centres: windows are disconnected and "
                      "relative offsets across the gap are undetermined.")
    for msg in issues:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    return {
        "x": x,
        "probability": np.where(visited, np.exp(log_p), 0.0),
        "free_energy": F,
        "counts": counts,
        "offsets": kT * f,
        "centers": centers,
        "converged": converged,
        "n_iter": n_iter,
        "overlaps": overlaps,
        "reliable": not issues,
        "issues": issues,
    }
