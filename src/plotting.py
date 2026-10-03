"""Small plotting helpers shared by the notebooks (matplotlib only)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Fixed categorical order; a colour always follows the same entity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REFERENCE = "#0b0b0b"  # exact / reference curves (drawn dashed)
MUTED = "#898781"
METHOD_COLORS = {
    "unbiased": PALETTE[0],
    "metad": PALETTE[1],
    "umbrella": PALETTE[2],
    "amd": PALETTE[3],
    "gamd": PALETTE[4],
    "tremd": PALETTE[5],
    "hremd": PALETTE[6],
}


def set_style():
    """Plain, readable defaults: thin lines, light grid, no top/right spines."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.figsize": (7.0, 3.6),
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "animation.html": "jshtml",
    })


def plot_reference(ax, x, F, label=r"$F_\mathrm{true}$"):
    """Draw the exact reference profile as a dashed dark line."""
    return ax.plot(x, F, ls="--", color=REFERENCE, lw=1.4, label=label, zorder=5)[0]


def free_energy_axes(ax, ylim=None, xlabel=r"$x$ (reduced units)"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$F$ (reduced energy units)")
    if ylim is not None:
        ax.set_ylim(*ylim)


def save_figure(fig, name, root=None):
    """Save to <repo>/figures/<name>.png (directory created if needed)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=110, bbox_inches="tight")
    return out / f"{name}.png"


def show_animation(anim):
    """Embed an animation as JavaScript/HTML (no ffmpeg needed)."""
    from IPython.display import HTML

    html = HTML(anim.to_jshtml())
    plt.close(anim._fig)
    return html


def animate_metadynamics(result, potential, F_true, bias_factor, ylim=(-0.2, 6.0), xlim=None, interval=120):
    """Replay stored WT-MetaD snapshots; no simulation happens in the callbacks.

    Top: V(x), V_bias(x, t), V(x) + V_bias(x, t) and the walker position.
    Bottom: reference F(x) and the current estimate -gamma/(gamma-1) V_bias.
    """
    x = result["x_grid"]
    snaps = result["snapshot_bias"]
    V0 = potential - potential.min()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)

    ax1.plot(x, V0, color=REFERENCE, lw=1.2, label=r"$V(x)$")
    l_bias, = ax1.plot([], [], color=PALETTE[1], label=r"$V_\mathrm{bias}$")
    l_eff, = ax1.plot([], [], color=PALETTE[0], label=r"$V + V_\mathrm{bias}$")
    dot, = ax1.plot([], [], "o", ms=8, color=PALETTE[0], mec="white", mew=1.5)
    ax1.set_ylim(*ylim)
    ax1.set_ylabel("energy")
    ax1.legend(loc="upper center", ncol=3)
    title = ax1.set_title("")

    plot_reference(ax2, x, F_true)
    l_F, = ax2.plot([], [], color=PALETTE[1], label=r"$F_\mathrm{MetaD}(t)$")
    ax2.set_ylim(*ylim)
    free_energy_axes(ax2)
    ax2.legend(loc="upper center", ncol=2)
    ax2.set_xlim(*(xlim if xlim is not None else (x[0], x[-1])))
    fig.tight_layout()

    def update(i):
        bias = snaps[i]
        x_i = result["snapshot_x"][i]
        l_bias.set_data(x, bias)
        l_eff.set_data(x, V0 + bias)
        dot.set_data([x_i], [np.interp(x_i, x, V0 + bias)])
        F = -bias_factor / (bias_factor - 1.0) * bias
        l_F.set_data(x, F - F.min())
        title.set_text(f"t = {result['snapshot_time'][i]:.0f}")
        return l_bias, l_eff, dot, l_F, title

    return FuncAnimation(fig, update, frames=len(snaps), interval=interval, blit=False)


def animate_replicas(result, potentials, labels, frames, interval=150):
    """Replay replica positions on stacked potentials.

    potentials : list of (x_grid, U_k(x_grid)) per slot
    frames : indices into result["x_slot"] to show
    Walkers keep their colour as they move between slots.
    """
    R = len(potentials)
    fig, axes = plt.subplots(R, 1, figsize=(5.6, 1.25 * R + 0.6), sharex=True)
    t = result["time"]
    rounds = np.searchsorted(result["exchange_time"], t, side="right") - 1
    walker_at = np.argsort(result["walker_slot"], axis=1)[rounds]  # (n_saved, R)
    dots = []
    for k, ax in enumerate(axes):
        xg, U = potentials[k]
        ax.plot(xg, U - U.min(), color=MUTED, lw=1.2)
        ax.set_ylim(-0.3, 4.0)
        ax.set_yticks([])
        ax.set_ylabel(labels[k], rotation=0, ha="right", va="center")
        dots.append(ax.plot([], [], "o", ms=9, mec="white", mew=1.5)[0])
    axes[-1].set_xlabel(r"$x$")
    title = axes[0].set_title("")
    fig.tight_layout()

    def update(n):
        i = frames[n]
        for k in range(R):
            xg, U = potentials[k]
            xi = result["x_slot"][i, k]
            dots[k].set_data([xi], [np.interp(xi, xg, U - U.min())])
            dots[k].set_color(PALETTE[walker_at[i, k] % len(PALETTE)])
        title.set_text(f"t = {t[i]:.1f}   (colour = walker identity)")
        return dots + [title]

    return FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
