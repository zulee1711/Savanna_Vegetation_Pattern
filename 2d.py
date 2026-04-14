
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.stdout.reconfigure(encoding='utf-8')

# -- Parameters ---------------------------------------------------------------
Lx  = 200.0
Ly  = 200.0
Nx  = 2*128
Ny  = 2*128
dx  = Lx / Nx
dy  = Ly / Ny

a   = 2.0
m   = 0.45
vtot   = 182.5      # advection velocity for w in x-direction
flow_dir = 1/4 * np.pi
v = np.array([np.cos(flow_dir)*vtot, np.sin(flow_dir)*vtot])
vx = v[0]
vy = v[1]
D   = 1.0        # diffusion coefficient for n
dw = 0.1

T       = 100.0
dt      = 0.001
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
ixf = (np.arange(Nx) + 1) % Nx   # x forward
ixb = (np.arange(Nx) - 1) % Nx   # x backward
iyf = (np.arange(Ny) + 1) % Ny   # y forward
iyb = (np.arange(Ny) - 1) % Ny   # y backward

# -- Time integration ---------------------------------------------------------
save_every = 500
snapshots_w = []
snapshots_n = []
times       = []

for step in range(n_steps + 1):

    if step % save_every == 0:
        snapshots_w.append(w.copy())
        snapshots_n.append(n.copy())
        times.append(step * dt)
        print(f"  t = {step * dt:.2f} / {T}", end="\r")

    if step == n_steps:
        break

    # upwind dw/dx (advection in x only)
    if vx <= 0:
        dw_dx = (w - w[:, ixb]) / dx
    else:
        dw_dx = (w[:, ixf] - w) / dx
    
    if vy <= 0:
        dw_dy = (w - w[iyb, :]) / dy
    else:
        dw_dy = (w[iyf, :] - w) / dy
    
    
        
    #dw_dx = (w[:,ixf] - w[:, ixb])/ dx

    # 2D Laplacian with central differences
    laplacian_n = (
        (n[:, ixf] - 2*n + n[:, ixb]) / dx**2 +
        (n[iyf, :] - 2*n + n[iyb, :]) / dy**2
    )
    laplacian_w = (
        (w[:, ixf] - 2*w + w[:, ixb]) / dx**2 +
        (w[iyf, :] - 2*w + w[iyb, :]) / dy**2
    )

    dw_dt = a - w - w * n**2 + vx * dw_dx + vy *dw_dy #+ dw*laplacian_w
    dn_dt = w * n**2 - m * n + D * laplacian_n

    w = w + dt * dw_dt
    n = n + dt * dn_dt
    w = np.clip(w, 0, 1e6)
    n = np.clip(n, 0, 1e6)

    if np.any(np.isnan(w)) or np.any(np.isnan(n)):
        print(f"\nNaN detected at step {step} (t={step*dt:.3f}). Stopping.")
        break

print(f"\nSimulation complete. Saved {len(snapshots_w)} snapshots.")

snapshots_w = np.array(snapshots_w)
snapshots_n = np.array(snapshots_n)
times       = np.array(times)
w_min, w_max = snapshots_w[100:].min(), snapshots_w[100:].max()
n_min, n_max = snapshots_n.min(), snapshots_n.max()

# -- Static plots: first, middle, last snapshots ------------------------------
frames_to_plot = [0, len(times)//2, -1]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("2D PDE simulation  —  w (top) and n (bottom)", fontsize=12)

for col, fi in enumerate(frames_to_plot):
    im0 = axes[0, col].imshow(snapshots_w[fi], origin="lower",
                               extent=[0, Lx, 0, Ly], cmap="inferno",
                               vmin=w_min, vmax=w_max)
    axes[0, col].set_title(f"w,  t = {times[fi]:.1f}")
    axes[0, col].set_xlabel("x")
    axes[0, col].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0, col])

    im1 = axes[1, col].imshow(snapshots_n[fi], origin="lower",
                               extent=[0, Lx, 0, Ly], cmap="viridis", vmin = n_min, vmax = n_max)
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
                   cmap="inferno", vmin=w_min, vmax=w_max, animated=True)
ax_w.set_title("w  (activator)")
ax_w.set_xlabel("x"); ax_w.set_ylabel("y")
plt.colorbar(im_w, ax=ax_w)

im_n = ax_n.imshow(snapshots_n[0], origin="lower", extent=[0, Lx, 0, Ly],
                   cmap="viridis", vmin=n_min, vmax= n_max, animated=True)
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
