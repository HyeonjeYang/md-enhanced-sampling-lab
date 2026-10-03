"""Accelerated MD (Hamelberg et al. 2004) and Gaussian accelerated MD (Miao et al. 2015).

Both methods add a boost Delta V(V) that depends on the potential energy
only, so the modified force is the original force times a scalar factor:

    -dV*/dq = -(1 + dDeltaV/dV) dV/dq

Toy system: the double-well coordinate x plus `n_bath` hidden harmonic
coordinates y_k that do not interact with x,

    V(x, y) = a (x^2 - 1)^2 + b x + sum_k k_bath y_k^2 / 2.

The bath leaves F(x) unchanged (it is decoupled), but because the boost acts
on the *total* potential energy, Delta V now fluctuates at fixed x, as it does
in real molecules. With n_bath = 0 the model reduces to the pure 1D well.
"""

import numpy as np


# --- aMD ---------------------------------------------------------------------

def amd_boost(V, E, alpha):
    """Delta V = (E - V)^2 / (alpha + E - V) for V < E, else 0."""
    if alpha <= 0:
        raise ValueError("alpha must be positive (alpha = 0 gives a non-smooth potential).")
    u = np.maximum(E - np.asarray(V, dtype=float), 0.0)
    return u**2 / (alpha + u)


def amd_force_scale(V, E, alpha):
    """1 + dDeltaV/dV = alpha^2 / (alpha + E - V)^2 for V < E, else 1."""
    u = np.maximum(E - np.asarray(V, dtype=float), 0.0)
    return alpha**2 / (alpha + u) ** 2


# --- GaMD --------------------------------------------------------------------

def gamd_boost(V, E, k):
    """Delta V = k (E - V)^2 / 2 for V < E, else 0."""
    u = np.maximum(E - np.asarray(V, dtype=float), 0.0)
    return 0.5 * k * u**2


def gamd_force_scale(V, E, k):
    """1 + dDeltaV/dV = 1 - k (E - V) for V < E, else 1."""
    u = np.maximum(E - np.asarray(V, dtype=float), 0.0)
    return 1.0 - k * u


def potential_statistics(V):
    """V_max, V_min, V_avg and sigma_V of sampled potential energies."""
    V = np.ravel(V)
    return {"V_max": V.max(), "V_min": V.min(), "V_avg": V.mean(), "V_std": V.std()}


def gamd_parameters(stats, sigma0, threshold="lower"):
    """Threshold E and force constant k from the standard GaMD rules.

    With k = k0 / (V_max - V_min) and 0 < k0 <= 1 (Miao et al. 2015):

    lower bound, E = V_max:
        k0 = min(1, k0'),  k0' = (sigma0 / sigma_V) (V_max - V_min) / (V_max - V_avg)
    upper bound, E = V_min + 1/k:
        k0 = k0'' = (1 - sigma0 / sigma_V) (V_max - V_min) / (V_avg - V_min),
        used only if 0 < k0'' <= 1; otherwise the lower-bound rule is applied.

    These choices keep V* monotonic in V and smoother than V
    (V_max <= E <= V_min + 1/k) and aim for sigma_DeltaV <= sigma0.
    """
    Vmax, Vmin, Vavg, sV = (stats[key] for key in ("V_max", "V_min", "V_avg", "V_std"))
    if not (Vmax > Vavg > Vmin) or sV <= 0:
        raise ValueError("Potential statistics are degenerate; run a longer calibration.")
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive.")
    k0_lower = min(1.0, sigma0 / sV * (Vmax - Vmin) / (Vmax - Vavg))
    k0_upper = (1.0 - sigma0 / sV) * (Vmax - Vmin) / (Vavg - Vmin)

    if threshold == "upper" and 0.0 < k0_upper <= 1.0:
        k0, mode = k0_upper, "upper"
        k = k0 / (Vmax - Vmin)
        E = Vmin + 1.0 / k
    elif threshold in ("lower", "upper"):
        k0, mode = k0_lower, "lower"
        k = k0 / (Vmax - Vmin)
        E = Vmax
    else:
        raise ValueError("threshold must be 'lower' or 'upper'.")
    check_gamd_constraints(E, k, stats)
    return {"E": E, "k": k, "k0": k0, "threshold_used": mode,
            "sigma_dV_estimate": k * (E - Vavg) * sV}


def check_gamd_constraints(E, k, stats, rtol=1e-9):
    """Raise ValueError unless V_max <= E <= V_min + 1/k (and k > 0)."""
    Vmax, Vmin = stats["V_max"], stats["V_min"]
    tol = rtol * max(1.0, abs(Vmax), abs(Vmin))
    if k <= 0 or E < Vmax - tol or E > Vmin + 1.0 / k + tol:
        raise ValueError(
            f"GaMD constraint V_max <= E <= V_min + 1/k violated "
            f"(V_max={Vmax:.3f}, E={E:.3f}, V_min + 1/k={Vmin + 1.0 / k:.3f})."
        )


# --- Toy system and dynamics ---------------------------------------------

def total_potential(q, a=2.0, b=0.5, k_bath=10.0):
    """V(x, y) for q[..., 0] = x and q[..., 1:] = hidden bath coordinates."""
    x, y = q[..., 0], q[..., 1:]
    return a * (x**2 - 1.0) ** 2 + b * x + 0.5 * k_bath * np.sum(y**2, axis=-1)


def total_gradient(q, a=2.0, b=0.5, k_bath=10.0):
    """dV/dq with the same layout as q."""
    grad = k_bath * q
    x = q[..., 0]
    grad[..., 0] = 4.0 * a * x * (x**2 - 1.0) + b
    return grad


def boost_and_scale(V, method="none", E=None, alpha=None, k=None):
    """Return (Delta V, force scale) for method 'none', 'amd' or 'gamd'."""
    if method == "none":
        return np.zeros_like(V), np.ones_like(V)
    if method == "amd":
        return amd_boost(V, E, alpha), amd_force_scale(V, E, alpha)
    if method == "gamd":
        return gamd_boost(V, E, k), gamd_force_scale(V, E, k)
    raise ValueError(f"Unknown boost method {method!r}")


def run_boosted_dynamics(
    x0,
    n_steps,
    dt,
    kT,
    rng,
    *,
    method="none",
    E=None,
    alpha=None,
    k=None,
    n_bath=4,
    k_bath=10.0,
    a=2.0,
    b=0.5,
    friction=1.0,
    stride=10,
):
    """Overdamped Langevin dynamics on V*(q) = V(q) + Delta V(V(q)).

    x0 may be an array of independent walkers. Bath coordinates start from
    their thermal distribution. Returns strided x, unboosted total V, and
    Delta V, each of shape (n_saved, n_walkers).
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    q = np.empty((x0.size, 1 + n_bath))
    q[:, 0] = x0
    q[:, 1:] = rng.standard_normal((x0.size, n_bath)) * np.sqrt(kT / k_bath)
    noise = np.sqrt(2.0 * kT * dt / friction)
    n_saved = n_steps // stride + 1
    out = {key: np.empty((n_saved, x0.size)) for key in ("x", "V", "dV")}

    def record(i):
        V = total_potential(q, a, b, k_bath)
        out["x"][i], out["V"][i] = q[:, 0], V
        out["dV"][i] = boost_and_scale(V, method, E, alpha, k)[0]

    record(0)
    for step in range(1, n_steps + 1):
        V = total_potential(q, a, b, k_bath)
        _, scale = boost_and_scale(V, method, E, alpha, k)
        grad = total_gradient(q, a, b, k_bath)
        q += -scale[:, None] * grad / friction * dt + noise * rng.standard_normal(q.shape)
        if step % stride == 0:
            record(step // stride)
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("Boosted dynamics diverged; reduce dt.")
    out["time"] = np.arange(n_saved) * stride * dt
    return out


def calibrate_gamd(x0, n_cmd_steps, n_rounds, steps_per_round, dt, kT, rng, *,
                   sigma0, threshold="lower", stride=10, **system):
    """Simplified version of the standard GaMD preparation stages.

    1. conventional dynamics -> potential statistics -> (E, k)
    2. `n_rounds` short boosted runs; V_max / V_min are running extremes over
       all calibration data and V_avg / sigma_V come from the latest round,
       after which (E, k) are recomputed.

    The returned parameters are meant to be held fixed in production.
    """
    cmd = run_boosted_dynamics(x0, n_cmd_steps, dt, kT, rng, stride=stride, **system)
    stats = potential_statistics(cmd["V"][len(cmd["V"]) // 5:])
    params = gamd_parameters(stats, sigma0, threshold)
    history = [dict(params, **stats)]
    x_last = cmd["x"][-1]
    for _ in range(n_rounds):
        run = run_boosted_dynamics(x_last, steps_per_round, dt, kT, rng, method="gamd",
                                   E=params["E"], k=params["k"], stride=stride, **system)
        new = potential_statistics(run["V"])
        stats = {"V_max": max(stats["V_max"], new["V_max"]),
                 "V_min": min(stats["V_min"], new["V_min"]),
                 "V_avg": new["V_avg"], "V_std": new["V_std"]}
        params = gamd_parameters(stats, sigma0, threshold)
        history.append(dict(params, **stats))
        x_last = run["x"][-1]
    return params, history


# --- Reweighting ------------------------------------------------------------

def cumulant_free_energy(x, dV, bins, kT, order=2, min_count=20):
    """GaMD-style cumulant reweighting along x.

        F(x_j) = -kT ln p*(x_j) - sum_{n<=order} beta^(n-1)/n! C_n(j)

    with C_1 = <dV>_j and C_2 = Var(dV)_j over the frames in bin j. Exact only
    if Delta V is Gaussian within each bin (order 2) and sampling is adequate.
    Bins with fewer than `min_count` frames are NaN. Returns (centres, F, counts)
    with min F = 0.
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    x, dV = np.ravel(x), np.ravel(dV)
    edges = np.asarray(bins, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(x, edges) - 1
    beta = 1.0 / kT
    F = np.full(len(centers), np.nan)
    counts = np.bincount(idx[(idx >= 0) & (idx < len(centers))], minlength=len(centers))
    for j in np.flatnonzero(counts >= min_count):
        s = dV[idx == j]
        correction = s.mean() + (0.5 * beta * s.var() if order == 2 else 0.0)
        F[j] = -kT * np.log(counts[j] / np.diff(edges)[j]) - correction
    return centers, F - np.nanmin(F), counts
