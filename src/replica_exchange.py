"""Temperature and Hamiltonian replica exchange on the 1D double well.

Slot k of the ladder has temperature T_k and Hamiltonian

    U_k(x, y) = lambda_k a (x^2 - 1)^2 + b x + sum_m k_bath y_m^2 / 2.

T-REMD: all lambda_k = 1, temperatures differ.
H-REMD: all T_k equal, lambda_k < 1 softens the barrier in auxiliary slots.

The optional hidden harmonic coordinates y (n_bath > 0) do not interact with x.
They are only there to show how the size of the system enters the exchange
statistics; with n_bath = 0 the model is the pure 1D double well.

Neighbouring slots exchange configurations with the Metropolis criterion
(Sugita & Okamoto 1999); acceptances are evaluated in log space. Dynamics are
overdamped, so there are no velocities to rescale after a swap. Inertial MD
implementations must also treat momenta / thermostats consistently.
"""

import numpy as np


def exchange_log_acceptance(beta_i, beta_j, u_i_xi, u_i_xj, u_j_xi, u_j_xj):
    """General log acceptance for swapping x_i <-> x_j between slots i and j.

    ln P_acc = min(0, -[beta_i U_i(x_j) + beta_j U_j(x_i) - beta_i U_i(x_i) - beta_j U_j(x_j)])
    """
    delta = beta_i * u_i_xj + beta_j * u_j_xi - beta_i * u_i_xi - beta_j * u_j_xj
    return np.minimum(0.0, -delta)


def tremd_log_acceptance(beta_i, beta_j, V_i, V_j):
    """T-REMD: ln P_acc = min(0, (beta_i - beta_j) [V(x_i) - V(x_j)])."""
    return np.minimum(0.0, (beta_i - beta_j) * (V_i - V_j))


def hremd_log_acceptance(beta, u_i_xi, u_i_xj, u_j_xi, u_j_xj):
    """H-REMD: ln P_acc = min(0, -Delta), Delta = beta [U_i(x_j) + U_j(x_i) - U_i(x_i) - U_j(x_j)]."""
    return np.minimum(0.0, -beta * (u_i_xj + u_j_xi - u_i_xi - u_j_xj))


def metropolis_accept(log_acceptance, rng):
    """Accept with probability exp(log_acceptance) without exponentiating large numbers."""
    return np.log(rng.random()) < log_acceptance


def geometric_temperatures(T_min, T_max, n_replicas):
    """T_k = T_min (T_max / T_min)^(k / (R - 1))."""
    return T_min * (T_max / T_min) ** (np.arange(n_replicas) / max(n_replicas - 1, 1))


def ladder_potential(q, lam, a=2.0, b=0.5, k_bath=10.0):
    """U_lambda(q) with q[..., 0] = x and q[..., 1:] = bath coordinates."""
    q = np.asarray(q, dtype=float)
    x = q[..., 0]
    return lam * a * (x**2 - 1.0) ** 2 + b * x + 0.5 * k_bath * np.sum(q[..., 1:] ** 2, axis=-1)


def ladder_force(q, lam, a=2.0, b=0.5, k_bath=10.0):
    """-dU_lambda/dq, same layout as q; lam broadcasts over leading axes."""
    q = np.asarray(q, dtype=float)
    f = -k_bath * q
    x = q[..., 0]
    f[..., 0] = -(lam * 4.0 * a * x * (x**2 - 1.0) + b)
    return f


def run_replica_exchange(
    x0,
    temperatures,
    lambdas,
    n_steps,
    exchange_stride,
    dt,
    rng,
    *,
    a=2.0,
    b=0.5,
    n_bath=0,
    k_bath=10.0,
    friction=1.0,
    stride=10,
):
    """Replica exchange with overdamped Langevin dynamics in every slot.

    Exchanges between neighbours alternate between even pairs (0,1),(2,3),...
    and odd pairs (1,2),(3,4),...  Slot 0 is the physical reference
    (lowest temperature for T-REMD, lambda = 1 for H-REMD).

    Returns a dict with
      x_slot        (n_saved, R) coordinate x held by each slot
      walker_slot   (n_rounds + 1, R) slot index occupied by each walker
      exchange_time (n_rounds + 1,) time of each exchange round (0 first)
      attempts, accepts, acceptance  per neighbouring pair (k, k+1)
    """
    T = np.asarray(temperatures, dtype=float)
    lam = np.broadcast_to(np.asarray(lambdas, dtype=float), T.shape).copy()
    R = T.size
    beta = 1.0 / T
    q = np.zeros((R, 1 + n_bath))
    q[:, 0] = np.broadcast_to(np.asarray(x0, dtype=float), T.shape)
    q[:, 1:] = rng.standard_normal((R, n_bath)) * np.sqrt(T[:, None] / k_bath)
    walker_at_slot = np.arange(R)
    noise = np.sqrt(2.0 * T * dt / friction)[:, None]
    pot = dict(a=a, b=b, k_bath=k_bath)

    n_saved = n_steps // stride + 1
    x_slot = np.empty((n_saved, R))
    x_slot[0] = q[:, 0]
    walker_slot = [np.argsort(walker_at_slot)]
    exchange_time = [0.0]
    attempts = np.zeros(R - 1, dtype=int)
    accepts = np.zeros(R - 1, dtype=int)
    n_rounds = 0

    for step in range(1, n_steps + 1):
        q += ladder_force(q, lam, **pot) / friction * dt + noise * rng.standard_normal(q.shape)
        if step % exchange_stride == 0:
            for i in range(n_rounds % 2, R - 1, 2):
                j = i + 1
                log_acc = exchange_log_acceptance(
                    beta[i], beta[j],
                    ladder_potential(q[i], lam[i], **pot), ladder_potential(q[j], lam[i], **pot),
                    ladder_potential(q[i], lam[j], **pot), ladder_potential(q[j], lam[j], **pot),
                )
                attempts[i] += 1
                if metropolis_accept(log_acc, rng):
                    accepts[i] += 1
                    q[[i, j]] = q[[j, i]]
                    walker_at_slot[[i, j]] = walker_at_slot[[j, i]]
            n_rounds += 1
            walker_slot.append(np.argsort(walker_at_slot))
            exchange_time.append(step * dt)
        if step % stride == 0:
            x_slot[step // stride] = q[:, 0]
    if not np.all(np.isfinite(x_slot)):
        raise FloatingPointError("Replica dynamics diverged; reduce dt.")
    return {
        "time": np.arange(n_saved) * stride * dt,
        "x_slot": x_slot,
        "walker_slot": np.asarray(walker_slot),
        "exchange_time": np.asarray(exchange_time),
        "attempts": attempts,
        "accepts": accepts,
        "acceptance": accepts / np.maximum(attempts, 1),
        "temperatures": T,
        "lambdas": lam,
    }


def walker_trajectories(result):
    """x(t) of each walker (continuous dynamics), shape (n_saved, R).

    The slot trace x_slot[:, 0] jumps whenever a swap is accepted; following
    each walker instead gives physically continuous paths.
    """
    t, ex_t = result["time"], result["exchange_time"]
    rounds = np.searchsorted(ex_t, t, side="right") - 1
    slots = result["walker_slot"][rounds]  # (n_saved, R): slot of walker w
    return np.take_along_axis(result["x_slot"], slots, axis=1)


def count_round_trips(slot_trace, n_slots):
    """Count bottom -> top -> bottom round trips in one walker's slot trace."""
    trips, reached_top, started = 0, False, False
    for s in slot_trace:
        if s == 0:
            if started and reached_top:
                trips += 1
            started, reached_top = True, False
        elif s == n_slots - 1 and started:
            reached_top = True
    return trips
