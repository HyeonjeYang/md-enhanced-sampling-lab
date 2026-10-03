"""Small periodic 2D Lennard-Jones fluid for thermostat/barostat demonstrations.

Reduced units: epsilon = sigma = m = k_B = 1. The pair potential is truncated
at r_cut and shifted so that V(r_cut) = 0; no tail corrections are applied.
Forces are evaluated for all O(N^2) pairs with the minimum-image convention,
which is fine for a few dozen particles. This is a teaching model, not a
production LJ simulator.
"""

import numpy as np

from .thermal import degrees_of_freedom, kinetic_energy, remove_com_velocity

DIM = 2


def square_lattice(n_per_side, box_length):
    """n_per_side^2 particles on a square lattice filling the periodic box."""
    spacing = box_length / n_per_side
    grid = (np.arange(n_per_side) + 0.5) * spacing
    gx, gy = np.meshgrid(grid, grid)
    return np.column_stack([gx.ravel(), gy.ravel()])


def minimum_image(dr, box_length):
    """Map separation vectors to the nearest periodic image."""
    return dr - box_length * np.round(dr / box_length)


def lj_forces(positions, box_length, r_cut=2.5, epsilon=1.0, sigma=1.0):
    """Forces, potential energy and pair virial of the truncated-shifted LJ fluid.

    Returns
    -------
    forces : (N, 2) array
    potential : float, sum_{i<j} [V(r_ij) - V(r_cut)] for r_ij < r_cut
    virial : float, W = sum_{i<j} r_ij . F_ij (enters the pressure)
    """
    if box_length < 2.0 * r_cut:
        raise ValueError(
            f"Box length {box_length:.3f} < 2 r_cut = {2 * r_cut:.3f}: "
            "the minimum-image convention would miss interactions."
        )
    dr = minimum_image(positions[:, None, :] - positions[None, :, :], box_length)
    r2 = np.sum(dr**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    inside = r2 < r_cut**2

    sr2 = np.where(inside, sigma**2 / r2, 0.0)
    sr6 = sr2**3
    sr12 = sr6**2
    sc6 = (sigma / r_cut) ** 6
    shift = 4.0 * epsilon * (sc6**2 - sc6)

    pair_energy = np.where(inside, 4.0 * epsilon * (sr12 - sr6) - shift, 0.0)
    # F_ij = f_over_r * r_ij, with f_over_r = 24 eps (2 (s/r)^12 - (s/r)^6) / r^2
    f_over_r = np.where(inside, 24.0 * epsilon * (2.0 * sr12 - sr6) / r2, 0.0)

    forces = np.sum(f_over_r[:, :, None] * dr, axis=1)
    potential = 0.5 * pair_energy.sum()
    virial = 0.5 * np.sum(f_over_r * np.where(inside, r2, 0.0))
    return forces, potential, virial


def internal_temperature(v, mass=1.0):
    """Kinetic temperature in the centre-of-mass frame, f = dN - d."""
    n = len(v)
    v_int = remove_com_velocity(v, mass)
    return 2.0 * kinetic_energy(v_int, mass) / degrees_of_freedom(n, DIM)


def pressure(n_particles, temperature, virial, area):
    """Virial pressure in 2D: P = (N k_B T + W / d) / A."""
    return (n_particles * temperature + virial / DIM) / area


def run_lj(
    positions,
    velocities,
    box_length,
    n_steps,
    dt,
    *,
    thermostat="none",
    T_target=1.0,
    friction=1.0,
    rescale_stride=1,
    barostat=False,
    P_target=1.0,
    tau_P=2.0,
    compressibility=1.0,
    rng=None,
    sample_stride=10,
    mass=1.0,
    r_cut=2.5,
):
    """Run the 2D LJ fluid with an optional thermostat and barostat.

    thermostat
        "none"      : Velocity Verlet, NVE.
        "rescale"   : Velocity Verlet + exact velocity rescaling to T_target
                      every `rescale_stride` steps (pedagogical contrast only;
                      it does not generate the canonical ensemble).
        "langevin"  : BAOAB Langevin splitting with friction coefficient zeta
                      (`friction`); the velocity damping rate is zeta / m.
    barostat
        If True, isotropic Berendsen weak coupling after every step:
        mu = [1 - compressibility * dt / tau_P * (P_target - P)]^(1/d),
        positions and box length are scaled by mu. `P_target` may be a scalar
        or an array of length n_steps (to apply a step change).
        This does not generate the exact NPT ensemble.

    Returns a dict of sampled time series and the final state.
    """
    if thermostat not in ("none", "rescale", "langevin"):
        raise ValueError(f"Unknown thermostat {thermostat!r}")
    if thermostat == "langevin" and rng is None:
        raise ValueError("The Langevin thermostat needs an explicit rng.")
    x = positions.copy() % box_length
    v = velocities.copy()
    L = float(box_length)
    n = len(x)
    P_target = np.broadcast_to(np.asarray(P_target, dtype=float), (n_steps,))
    c1 = np.exp(-friction / mass * dt)
    c2 = np.sqrt((1.0 - c1**2) * T_target / mass)

    f, U, W = lj_forces(x, L, r_cut)
    keys = ("time", "temperature", "kinetic", "potential", "total", "pressure", "box_length")
    out = {k: [] for k in keys}

    def record(step):
        T = internal_temperature(v, mass)
        K = kinetic_energy(v, mass)
        out["time"].append(step * dt)
        out["temperature"].append(T)
        out["kinetic"].append(K)
        out["potential"].append(U)
        out["total"].append(K + U)
        out["pressure"].append(pressure(n, T, W, L**DIM))
        out["box_length"].append(L)

    record(0)
    for step in range(1, n_steps + 1):
        if thermostat == "langevin":  # B A O A B
            v += 0.5 * dt * f / mass
            x += 0.5 * dt * v
            v = c1 * v + c2 * rng.standard_normal(v.shape)
            x += 0.5 * dt * v
            x %= L
            f, U, W = lj_forces(x, L, r_cut)
            v += 0.5 * dt * f / mass
        else:  # Velocity Verlet
            v += 0.5 * dt * f / mass
            x += dt * v
            x %= L
            f, U, W = lj_forces(x, L, r_cut)
            v += 0.5 * dt * f / mass
            if thermostat == "rescale" and step % rescale_stride == 0:
                v *= np.sqrt(T_target / internal_temperature(v, mass))

        if barostat:
            P = pressure(n, internal_temperature(v, mass), W, L**DIM)
            arg = 1.0 - compressibility * dt / tau_P * (P_target[step - 1] - P)
            if arg <= 0:
                raise FloatingPointError("Berendsen scaling became non-positive; increase tau_P.")
            mu = arg ** (1.0 / DIM)
            x *= mu
            L *= mu
            f, U, W = lj_forces(x, L, r_cut)

        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"LJ run diverged at step {step}; reduce dt.")
        if step % sample_stride == 0:
            record(step)

    result = {k: np.asarray(val) for k, val in out.items()}
    result.update(positions=x, velocities=v, final_box_length=L)
    return result
