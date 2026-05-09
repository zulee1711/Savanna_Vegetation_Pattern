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
 
# a   = 2.0    # source term for w
m   = 0.45       # decay rate for n
v   = 0 # 182.5      # advection velocity for w (can be negative)
dw  = 1.0          # diffusion coefficient for w
D   = 1.0        # diffusion coefficient for n (multiplies the Laplacian)
 
T   = 100.0      # total simulation time
dt  = 0.001      # time step  (must satisfy CFL and diffusion stability)
n_steps = int(T / dt)
a_max = 2.0
a_min = 0.0
# a_range = np.arange(a_min, a_max, 0.5)
a_range = np.linspace(a_min, a_max,100)
 
# Stability checks (print warnings if violated)
cfl     = abs(v) * dt / dx
diff_nu = D * dt / dx**2
print(f"CFL number (advection): {cfl:.3f}  {'⚠ UNSTABLE' if cfl > 1 else '✓'}")
print(f"Diffusion number:       {diff_nu:.3f}  {'⚠ UNSTABLE' if diff_nu > 0.5 else '✓'}")
 
x = np.linspace(0, L, N, endpoint=False)

# ── Helper: periodic index arrays ────────────────────────────────────────────
idx_p = np.arange(N)          # i
idx_f = (idx_p + 1) % N      # i+1  (forward)
idx_b = (idx_p - 1) % N      # i-1  (backward)
 
# ── Initial conditions ───────────────────────────────────────────────────────

n_average = np.zeros(len(a_range))
n_max = np.zeros(len(a_range))


initial_conditions = {"non-zero eq": {"n_average": np.zeros(len(a_range)), "n_max": np.zeros(len(a_range))}, 
                      "zero eq": {"n_average":  np.zeros(len(a_range)), "n_max": np.zeros(len(a_range))}}
plt.figure()
rng = np.random.default_rng(100)

for ic_name, ic_results in initial_conditions.items(): 

    for i, a in enumerate(a_range): 

        print("a = ", a)

        # Non-zero equilibirum 
        if a < 2*m: 
            n_eq = 2
            w_eq = 0.3
        else: 
            n_eq = (a + np.sqrt((a + 2.*m)*(a -2.*m)))/(2.*m)
            w_eq = a / (1.0 + (n_eq*n_eq))
        print("n_eq = ", n_eq)
        print("w_eq = ", w_eq)


        if ic_name == "zero eq": 
            w = a + ((rng.random(N))<0.005)
            n = 0 + ((rng.random(N))<0.005)
        else: 
            w = w_eq + ((rng.random(N))<0.005)
            n = n_eq+ ((rng.random(N))<0.005)

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
            dw_dt = a - w - w * n**2 - v * dw_dx # + dw*d2w_dx2
            dn_dt = w * n**2 - m * n + D * d2n_dx2
        
            # ── Euler step ──────────────────────────────────────────────────────────

            w_new = w + dt * dw_dt
            n_new = n + dt * dn_dt
            w_new = np.clip(w_new, 0, 1e6)
            n_new = np.clip(n_new, 0, 1e6)

            if step % 100 == 0:
                change = np.linalg.norm(n_new - n) + np.linalg.norm(w_new - w)
                if change < 1e-4:
                    break

            w = w_new 
            n = n_new 
        
            if np.any(np.isnan(w)) or np.any(np.isnan(n)):
                print(f"NaN detected at step {step} (t={step*dt:.3f}). Stopping.")
                break
        
        ic_results["n_average"][i] = np.average(n) 
        ic_results["n_max"][i] = np.max(n) 


    print("Complete")

    plt.scatter(a_range,  ic_results["n_average"], label = f"IC near {ic_name}")

plt.vlines(2*m, 0, 6, linestyles='dashed', label = "a = 2m")
plt.xlabel("a")
plt.title(f"Bifurcation Diagram, {m = }")
plt.ylabel("n")
plt.legend()
# plt.savefig("bifurcation_diagram.png", transparent = True)
plt.show()










