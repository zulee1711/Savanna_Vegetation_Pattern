import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
sys.stdout.reconfigure(encoding='utf-8')

# -- Parameters ---------------------------------------------------------------
Lx  = 250.0
Ly  = 250.0
Nx  = 250
Ny  = 250
dx  = Lx / Nx
dy  = Ly / Ny

a   = 2.0
m   = 0.45
vtot   = 182.5      # advection velocity for w in x-direction
flow_dir = 0 * np.pi
v = np.array([np.cos(flow_dir)*vtot, np.sin(flow_dir)*vtot])
vx = v[0]
vy = v[1]
vx = np.full((Ny, Nx), vx, dtype=np.float64)
vy = np.full((Ny, Nx), vy, dtype=np.float64)
D   = 1.0        # diffusion coefficient for n
dw = 0.1

T       = 1000.0
dt      = 0.003
n_steps = int(T / dt)

n_eq = (a + np.sqrt((a + 2.*m)*(a - 2.*m))) / (2.*m)
w_eq = a / (1.0 + n_eq**2)

# Stability checks
cfl     = abs(vtot) * dt / dx
diff_nu = D * dt / dx**2
print(f"CFL number (advection): {cfl:.3f}  {'[UNSTABLE]' if cfl > 1 else '[OK]'}")
print(f"Diffusion number:       {diff_nu:.3f}  {'[UNSTABLE]' if diff_nu > 0.5 else '[OK]'}")
print(f"n_eq = {n_eq:.4f},  w_eq = {w_eq:.4f}")

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)

# -- Initial conditions -------------------------------------------------------
rng = np.random.default_rng(42)

w = w_eq + (rng.random((Ny, Nx)) < 0.02).astype(float)
n = n_eq + (rng.random((Ny, Nx)) < 0.02).astype(float)
#w = a + (rng.random((Ny, Nx)) < 0.02).astype(float)#
#n = 0 + (rng.random((Ny, Nx)) < 0.02).astype(float)

# -- Periodic index arrays ----------------------------------------------------
ixf = np.arange(Nx, dtype=np.int64)
ixb = np.arange(Nx, dtype=np.int64)
iyf = np.arange(Ny, dtype=np.int64)
iyb = np.arange(Ny, dtype=np.int64)

ixf = (ixf + 1) % Nx
ixb = (ixb - 1) % Nx
iyf = (iyf + 1) % Ny
iyb = (iyb - 1) % Ny

# -- Time integration ---------------------------------------------------------
save_every = 500
snapshots_w = []
snapshots_n = []
times       = []

@njit
def step_pde(w, n, vx, vy, dx, dy, dt, a, m, dw, D,
             ixf, ixb, iyf, iyb):

    Ny, Nx = w.shape

    w_new = w.copy()
    n_new = n.copy()

    for j in range(Ny):
        for i in range(Nx):

            # periodic indices
            ip = ixf[i]
            im = ixb[i]
            jp = iyf[j]
            jm = iyb[j]

            # upwind derivatives for w
            if vx[j, i] >= 0:
                dw_dx = (w[j, i] - w[j, im]) / dx
            else:
                dw_dx = (w[j, ip] - w[j, i]) / dx

            if vy[j, i] >= 0:
                dw_dy = (w[j, i] - w[jm, i]) / dy
            else:
                dw_dy = (w[jp, i] - w[j, i]) / dy

            # Laplacian w
            lap_w = (
                (w[j, ip] - 2.0*w[j, i] + w[j, im]) / (dx*dx) +
                (w[jp, i] - 2.0*w[j, i] + w[jm, i]) / (dy*dy)
            )

            # Laplacian n
            lap_n = (
                (n[j, ip] - 2.0*n[j, i] + n[j, im]) / (dx*dx) +
                (n[jp, i] - 2.0*n[j, i] + n[jm, i]) / (dy*dy)
            )

            # PDEs
            w_new[j, i] = w[j, i] + dt * (
                a - w[j, i] - w[j, i]*n[j, i]**2
                - vx[j, i]*dw_dx - vy[j, i]*dw_dy
                + dw * lap_w
            )

            n_new[j, i] = n[j, i] + dt * (
                w[j, i]*n[j, i]**2 - m*n[j, i]
                + D * lap_n
            )

    return w_new, n_new

for step in range(n_steps + 1):
    if step % 500 == 0:
        print(str(step*dt) + "/" +str(T))

    if step % save_every == 0:
        snapshots_w.append(w.copy())
        snapshots_n.append(n.copy())
        times.append(step * dt)

    if step == n_steps:
        break

    w, n = step_pde(
        w, n, vx, vy,
        dx, dy, dt,
        a, m, dw, D,
        ixf, ixb, iyf, iyb
    )

    w = np.clip(w, 0, 1e6)
    n = np.clip(n, 0, 1e6)

print(f"\nSimulation complete. Saved {len(snapshots_w)} snapshots.")


sand_to_grass = LinearSegmentedColormap.from_list(
    "sand_to_grass",
    ["#e6d2a0", "#c2b280", "#6aa84f", "#2e7d32"]
)
plt.colormaps.register(sand_to_grass)
black_to_blue = LinearSegmentedColormap.from_list(
    "black_to_blue",
    ["#000000", "#1f77ff"]  
)
plt.colormaps.register(black_to_blue)


snapshots_w = np.array(snapshots_w)
snapshots_n = np.array(snapshots_n)
times       = np.array(times)
w_min, w_max = snapshots_w[15:].min(), snapshots_w[15:].max()
n_min, n_max = snapshots_n.min(), snapshots_n.max()

# -- Static plots: first, middle, last snapshots ------------------------------
frames_to_plot = [0, len(times)//2, -1]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("2D PDE simulation  —  w (top) and n (bottom)", fontsize=12)

for col, fi in enumerate(frames_to_plot):
    im0 = axes[0, col].imshow(snapshots_w[fi], origin="lower",
                               extent=[0, Lx, 0, Ly], cmap="black_to_blue",
                               vmin=w_min, vmax=w_max)
    axes[0, col].set_title(f"w,  t = {times[fi]:.1f}")
    axes[0, col].set_xlabel("x")
    axes[0, col].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0, col])

    im1 = axes[1, col].imshow(snapshots_n[fi], origin="lower",
                               extent=[0, Lx, 0, Ly], cmap="sand_to_grass", vmin = n_min, vmax = n_max)
    axes[1, col].set_title(f"n,  t = {times[fi]:.1f}")
    axes[1, col].set_xlabel("x")
    axes[1, col].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1, col])

plt.tight_layout()
plt.savefig("snapshots_2d.png", dpi=150)
print("Saved snapshots -> snapshots_2d.png")

# -- Animation ----------------------------------------------------------------
fig2, (ax_w, ax_n) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("2D simulation", fontsize=11)

im_w = ax_w.imshow(snapshots_w[0], origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="black_to_blue", vmin=w_min, vmax=w_max, animated=True)
ax_w.set_title("w  (activator)")
ax_w.set_xlabel("x"); ax_w.set_ylabel("y")
plt.colorbar(im_w, ax=ax_w)

im_n = ax_n.imshow(snapshots_n[0], origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="sand_to_grass", vmin=n_min, vmax= n_max, animated=True)
ax_n.set_title("n  (inhibitor)")
ax_n.set_xlabel("x"); ax_n.set_ylabel("y")
plt.colorbar(im_n, ax=ax_n)

title_w = ax_w.set_title("w  (activator), t = 0")
title_n = ax_n.set_title("n  (inhibitor), t = 0")
def update(frame):
    im_w.set_data(snapshots_w[frame])
    im_n.set_data(snapshots_n[frame])

    title_w.set_text(f"w  (water),  t = {times[frame]:.1f}")
    title_n.set_text(f"n  (biomass),  t = {times[frame]:.1f}")

    return im_w, im_n, title_w, title_n
ani = animation.FuncAnimation(fig2, update, frames=len(times), interval=60, blit=False)
ani.save("animation_2d.gif", writer="pillow", fps=20)
print("Saved animation -> animation_2d.gif")

plt.show()
