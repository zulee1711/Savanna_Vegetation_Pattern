"""
plotting.py - Plotting functions for the extended Klausmeier model
"""
import logging

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

_FIELDS = ["w", "g", "s", "b"]
_LABELS = ["w  (water)", "g  (grass)", "s  (saplings)", "b  (trees)"]
_CMAPS  = ["Blues", "Greens", "YlGn", "inferno"]
_COLORS = ["steelblue", "forestgreen", "yellowgreen", "sienna"]


def plot_spacetime_1d(w, g, s, b, times, mesh, cfg, save_path=None):
    """
    Plots space-time diagrams for w, g, s, b on a 1D domain.

    Inputs:
        w, g, s, b  (np.ndarray): Snapshot arrays, shape (nSnapshots, nCells)
        times       (np.ndarray): Time values, shape (nSnapshots,)
        mesh        (Mesh1D):     Mesh object
        cfg         (dict):       Configuration dictionary
        save_path   (str):        Optional path to save figure
    """
    L      = cfg['mesh']['L']
    T      = times[-1]
    arrays = [w, g, s, b]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Extended Klausmeier - 1D space-time", fontsize=12)

    for ax, data, label, cmap in zip(axes, arrays, _LABELS, _CMAPS):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[0, L, 0, T],
            cmap=cmap,
            vmin=0,
            vmax=max(np.nanmax(data), 1e-6),
        )
        ax.set_xlabel("x")
        ax.set_ylabel("time")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    _save_or_show(fig, save_path, "spacetime_1d.png")


def plot_snapshot_1d(w, g, s, b, times, mesh, cfg, t_idx=-1, save_path=None):
    """
    Plots a snapshot of all fields at a given time index.

    Inputs:
        w, g, s, b  (np.ndarray): Snapshot arrays, shape (nSnapshots, nCells)
        times       (np.ndarray): Time values, shape (nSnapshots,)
        mesh        (Mesh1D):     Mesh object
        cfg         (dict):       Configuration dictionary
        t_idx       (int):        Time index to plot (-1 for last snapshot)
        save_path   (str):        Optional path to save figure
    """
    arrays = [w, g, s, b]
    t      = times[t_idx]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Extended Klausmeier - snapshot at t={t:.2f}", fontsize=12)

    for ax, data, label, color in zip(axes, arrays, _LABELS, _COLORS):
        ax.plot(mesh.x, data[t_idx], color=color, lw=1.8)
        ax.set_xlabel("x")
        ax.set_ylabel("concentration")
        ax.set_title(label)
        ax.set_xlim(0, cfg['mesh']['L'])
        ax.set_ylim(0, None)

    plt.tight_layout()
    _save_or_show(fig, save_path, "snapshot_1d.png")


def animate_1d(w, g, s, b, times, mesh, cfg, save_path=None):
    """
    Animates all fields evolving in time.

    Inputs:
        w, g, s, b  (np.ndarray): Snapshot arrays, shape (nSnapshots, nCells)
        times       (np.ndarray): Time values, shape (nSnapshots,)
        mesh        (Mesh1D):     Mesh object
        cfg         (dict):       Configuration dictionary
        save_path   (str):        Optional path to save gif
    """
    arrays = [w, g, s, b]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Extended Klausmeier - 1D animation", fontsize=12)

    lines = []
    for ax, data, label, color in zip(axes, arrays, _LABELS, _COLORS):
        line, = ax.plot(mesh.x, data[0], color=color, lw=1.8)
        ax.set_xlabel("x")
        ax.set_ylabel("concentration")
        ax.set_title(label)
        ax.set_xlim(0, cfg['mesh']['L'])
        ax.set_ylim(0, max(np.nanmax(data), 0.1) * 1.1)
        lines.append(line)

    title = fig.text(0.5, 0.9, "", ha="center", fontsize=10)

    def update(frame):
        for line, data in zip(lines, arrays):
            line.set_ydata(data[frame])
        title.set_text(f"t = {times[frame]:.2f}")
        return lines + [title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=40, blit=True
    )

    if save_path:
        ani.save(save_path, writer="pillow", fps=25)
        logger.info(f"Saved animation -> {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _save_or_show(fig, save_path, default_name):
    """Saves figure if path given, otherwise shows it."""
    if save_path:
        fig.savefig(Path(save_path), dpi=150)
        logger.info(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_snapshot_2d(w, g, s, b, times, mesh, cfg, t_idx=-1, save_path=None):
    """
    Plots a 2D spatial heatmap of all fields at a given time index.
    Overlays terrain if available.

    Inputs:
        w, g, s, b  (np.ndarray): Snapshot arrays, shape (nSnapshots, nCellsY, nCellsX)
        times       (np.ndarray): Time values, shape (nSnapshots,)
        mesh        (Mesh2D):     Mesh object (may contain terrain h)
        cfg         (dict):       Configuration dictionary
        t_idx       (int):        Time index to plot (-1 for last snapshot)
        save_path   (str):        Optional path to save figure
    """
    arrays = [w, g, s, b]
    t      = times[t_idx]
    Lx, Ly = cfg['mesh']['Lx'], cfg['mesh']['Ly']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Extended Klausmeier - 2D snapshot at t={t:.2f}", fontsize=12)

    # Normalize terrain if available
    h_norm = None
    if mesh.h is not None:
        h_norm = (mesh.h - mesh.h.min()) / (mesh.h.max() - mesh.h.min() + 1e-12)

    alpha = 0.3  # Terrain overlay transparency

    for ax, data, label, cmap in zip(axes, arrays, _LABELS, _CMAPS):
        im = ax.imshow(
            data[t_idx],
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap=cmap,
            vmin=0,
            vmax=max(np.nanmax(data[t_idx]), 1e-6),
        )

        # Overlay terrain if available
        if h_norm is not None:
            ax.imshow(
                h_norm,
                origin="lower",
                extent=[0, Lx, 0, Ly],
                cmap="Greys",
                alpha=alpha
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    _save_or_show(fig, save_path, "snapshot_2d.png")


def animate_2d(w, g, s, b, times, mesh, cfg, save_path=None):
    arrays = [w, g, s, b]
    Lx, Ly = cfg['mesh']['Lx'], cfg['mesh']['Ly']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Extended Klausmeier - 2D animation", fontsize=12)

    # Compute fixed color limits across all frames
    vmaxes = [max(np.nanmax(data), 1e-6) for data in arrays]

    # Normalize terrain if available
    h_norm = None
    if mesh.h is not None:
        h_norm = (mesh.h - mesh.h.min()) / (mesh.h.max() - mesh.h.min() + 1e-12)

    alpha = 0.3  # Terrain overlay transparency

    ims = []
    h_ims = []  # Store terrain overlay images
    for ax, data, label, cmap, vmax in zip(axes, arrays, _LABELS, _CMAPS, vmaxes):
        im = ax.imshow(
            data[0],
            origin="lower",
            extent=[0, Lx, 0, Ly],
            cmap=cmap,
            vmin=0,
            vmax=vmax,      # ← fixed for entire animation
            animated=True,
        )

        # Add terrain overlay if available
        h_im = None
        if h_norm is not None:
            h_im = ax.imshow(
                h_norm,
                origin="lower",
                extent=[0, Lx, 0, Ly],
                cmap="Greys",
                alpha=alpha,
                animated=True
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
        ims.append(im)
        h_ims.append(h_im)

    title = fig.text(0.5, 0.9, "", ha="center", fontsize=10)

    def update(frame):
        for im, data in zip(ims, arrays):
            im.set_data(data[frame])
            # No set_clim - colormap is fixed
        title.set_text(f"t = {times[frame]:.2f}")
        return ims + h_ims + [title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=40, blit=False
    )

    if save_path:
        ani.save(save_path, writer="pillow", fps=25)
        logger.info(f"Saved animation -> {save_path}")
    else:
        plt.show()
    plt.close(fig)