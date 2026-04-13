"""
Simulation of the PDE system on a 1D grid:
 
  dw/dt = a - w - w*n^2 + v * dw/dx
  dn/dt = w*n^2 - m*n + d2n/dx2
 
w: activator field (no diffusion, advection with velocity v)
n: inhibitor/signal field (diffusion in space)
 
Discretisation:
  - Spatial: finite differences on a uniform 1D grid with periodic BCs
  - dw/dx: upwind scheme (sign of v determines direction)
  - d2n/dx2: central differences
  - Time: explicit (forward) Euler
"""
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.stdout.reconfigure(encoding='utf-8')
 
# ── Parameters ──────────────────────────────────────────────────────────────
L   = 100.0      # domain length
N   = 256        # number of grid points
dx  = L / N      # grid spacing
 
a   = 2.0    # source term for w
m   = 0.45       # decay rate for n
v   = 182.5      # advection velocity for w (can be negative)
dw  = 1.0          # diffusion coefficient for w
D   = 1.0        # diffusion coefficient for n (multiplies the Laplacian)
 
T   = 100.0      # total simulation time
dt  = 0.0002      # time step  (must satisfy CFL and diffusion stability)
n_steps = int(T / dt)

n_eq = (a - np.sqrt((a + 2.*m)*(a -2.*m)))/(2.*m)
w_eq = a / (1.0 + (n_eq*n_eq))
 
# Stability checks (print warnings if violated)
cfl     = abs(v) * dt / dx
diff_nu = D * dt / dx**2
print(f"CFL number (advection): {cfl:.3f}  {'⚠ UNSTABLE' if cfl > 1 else '✓'}")
print(f"Diffusion number:       {diff_nu:.3f}  {'⚠ UNSTABLE' if diff_nu > 0.5 else '✓'}")
 
x = np.linspace(0, L, N, endpoint=False)
 
# ── Initial conditions ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
 
# Small random perturbations around a homogeneous steady state
# Rough steady state estimate: n_ss ~ sqrt((a-1)/1) if a>1, else small
#w = np.full(N, a) + 0.01 * rng.standard_normal(N)
#n = np.full(N, 0.1) + 0.01 * rng.standard_normal(N)

w = w_eq + ((rng.random(N))<0.02)
n = n_eq+ ((rng.random(N))<0.02)*1 
print("n_eq = ", n_eq)
print("w_eq = ", w_eq)
#w = np.clip(w, 0, None)
#n = np.clip(n, 0, None)
 
# ── Helper: periodic index arrays ────────────────────────────────────────────
idx_p = np.arange(N)          # i
idx_f = (idx_p + 1) % N      # i+1  (forward)
idx_b = (idx_p - 1) % N      # i-1  (backward)
 
# ── Time integration ─────────────────────────────────────────────────────────
save_every = 100               # save a snapshot every this many steps
snapshots_w = []
snapshots_n = []
times       = []
 
for step in range(n_steps + 1):
 
    if step % save_every == 0:
        snapshots_w.append(w.copy())
        snapshots_n.append(n.copy())
        times.append(step * dt)
 
    if step == n_steps:
        break
 
    # ── Spatial derivatives ──────────────────────────────────────────────────
 
    # Upwind scheme for dw/dx  (first-order upwind)
    if v >= 0:
        dw_dx = (w - w[idx_b]) / dx          # backward difference
    else:
        dw_dx = (w[idx_f] - w) / dx          # forward difference
 
    # Central difference for d²n/dx²
    d2n_dx2 = (n[idx_f] - 2*n + n[idx_b]) / dx**2
    d2w_dx2 = (w[idx_f] - 2*w + w[idx_b]) / dx**2
    
 
    # ── RHS ─────────────────────────────────────────────────────────────────
    dw_dt = a - w - w * n**2 - v * dw_dx #+ dw*d2w_dx2
    dn_dt = w * n**2 - m * n + D * d2n_dx2
 
    # ── Euler step ──────────────────────────────────────────────────────────
    w = w + dt * dw_dt
    n = n + dt * dn_dt
    w = np.clip(w, 0, 1e6)
    n = np.clip(n, 0, 1e6)
 
    if np.any(np.isnan(w)) or np.any(np.isnan(n)):
        print(f"NaN detected at step {step} (t={step*dt:.3f}). Stopping.")
        break
 
print(f"Simulation complete. Saved {len(snapshots_w)} snapshots.")
 
snapshots_w = np.array(snapshots_w)
snapshots_n = np.array(snapshots_n)
times       = np.array(times)
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("1D PDE simulation  (dw/dt = a − w − wn² + v ∂w/∂x,  dn/dt = wn² − mn + ∂²n/∂x²)",
             fontsize=11)

im0 = axes[0].imshow(snapshots_w, aspect="auto", origin="lower",
                     extent=[0, L, 0, T], cmap="inferno", vmin=0, vmax=10)
axes[0].set_xlabel("x")
axes[0].set_ylabel("time")
axes[0].set_title("w  (activator)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(snapshots_n, aspect="auto", origin="lower",
                     extent=[0, L, 0, T], cmap="viridis")
axes[1].set_xlabel("x")
axes[1].set_ylabel("time")
axes[1].set_title("n  (inhibitor / diffusing)")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(r"C:\Users\Jasper\Desktop\advanced modelling\spacetime_wn.png", dpi=150)
print("Saved space-time diagram -> spacetime_wn.png") 
plt.show()
print(w)
print(n)

# # ── Static plot: space-time diagram ─────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("1D PDE simulation  (dw/dt = a − w − wn² + v ∂w/∂x,  dn/dt = wn² − mn + ∂²n/∂x²)",
#              fontsize=11)
 
# for ax, data, label, cmap in zip(
#         axes,
#         [snapshots_w, snapshots_n],
#         ["w  (activator)", "n  (inhibitor / diffusing)"],
#         ["inferno", "viridis"]):
#     im = ax.imshow(
#         data,
#         aspect="auto",
#         origin="lower",
#         extent=[0, L, 0, T],
#         cmap=cmap,
#     )
#     ax.set_xlabel("x")
#     ax.set_ylabel("time")
#     ax.set_title(label)
#     plt.colorbar(im, ax=ax)
#     plt.clim(0,1)
 
# plt.tight_layout()
# plt.savefig(r"C:\Users\Jasper\Desktop\advanced modelling\spacetime_wn.png", dpi=150)
# print("Saved space-time diagram → spacetime_wn.png")
 
# # ── Animation: w and n profiles evolving in time ────────────────────────────
# fig2, ax2 = plt.subplots(figsize=(10, 4))
# line_w, = ax2.plot(x, snapshots_w[0], lw=1.8, label="w", color="tomato")
# line_n, = ax2.plot(x, snapshots_n[0], lw=1.8, label="n", color="steelblue")
# ax2.set_xlim(0, L)
# vmin = min(np.nanmin(snapshots_w), np.nanmin(snapshots_n))
# vmax = max(np.nanmax(snapshots_w), np.nanmax(snapshots_n))
# ax2.set_ylim(max(vmin - 0.1, -1), min(vmax + 0.1, 20))
# ax2.set_xlabel("x")
# ax2.set_ylabel("concentration")
# ax2.legend(loc="upper right")
# title = ax2.set_title("")
 
# def update(frame):
#     line_w.set_ydata(snapshots_w[frame])
#     line_n.set_ydata(snapshots_n[frame])
#     title.set_text(f"t = {times[frame]:.1f}")
#     return line_w, line_n, title
 
# ani = animation.FuncAnimation(fig2, update, frames=len(times), interval=40, blit=True)
# ani.save(r"C:\Users\Jasper\Desktop\advanced modelling\animation_wn.gif", writer="pillow", fps=25)
# print("Saved animation → animation_wn.gif")
 
# plt.show()
