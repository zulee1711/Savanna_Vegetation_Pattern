"""
Simulation function 
"""
import numpy as np
from sympy.abc import alpha

from mesh import Mesh
from tqdm import tqdm

#%%
def _rhs_equations(
        p: dict,
        w: np.ndarray,
        g: np.ndarray,
        s: np.ndarray,
        b: np.ndarray,
        adv_w,
        lap_g,
        lap_s,
        lap_b = None,
):
    """
    :param adv_w: advection operator applied to w = v * dw/dx
    :param lap_g: ∇²g
    :param lap_s: ∇²s
    :param lap_b: ∇²b - Not used in current model, but kept in case added in future
    """
    a = p['a']
    alpha = p['alpha']
    b1, m1, d1 = p['b1'], p['m1'], p['d1']
    b2, m2, d2 = p['b2'], p['m2'], p['d2']
    b3, m3     = p['b3'], p['m3']
    gamma = p['gamma']
    phi = p['phi']

    n = g + s + b

    dw = a - w - (alpha * w * n**2) + adv_w
    dg = (b1 * w * n * g) - (m1 * g) + (d1 * lap_g)
    ds = (b2 * w * n * s) - (m2 * s) - (gamma * s) + (d2 * lap_s) + (phi * b)
    db = (b3 * w * n * b) - (m3 * b) + (gamma * s)

    return dw, dg, ds, db

#%%
def _spatial_ops_1d(cfg: dict, mesh: Mesh, w: np.ndarray, g: np.ndarray, s: np.ndarray, b: np.ndarray):
    """
    Compute all spatial operators needed for _rhs_equations for 1D
    """
    v = cfg['model']['v']

    if v >= 0:
        adv_w = v * (w[mesh.idxRightNeighb] - w[mesh.idx]) / mesh.dx     # forward difference
    else:
        adv_w = v * (w[mesh.idx] - w[mesh.idxLeftNeighb]) / mesh.dx      # backward difference

    def lap1d(u):
        lap = (u[mesh.idxRightNeighb] - 2 * u[mesh.idx] + u[mesh.idxLeftNeighb]) / mesh.dx**2
        return lap

    return adv_w, lap1d(g), lap1d(s), lap1d(b)

#%%
def _spatial_ops_2d(cfg: dict, mesh: Mesh, w: np.ndarray, g: np.ndarray, s: np.ndarray, b: np.ndarray):
    """
    Computes RHS 2D with spatially-varying velocity field (from terrain)
    Uses upwind advection for numerical stability
    """
    m = cfg['model']

    # Use spatially-varying velocity if available, otherwise fall back to constant v
    if mesh.vx is not None and mesh.vy is not None:
        vx = mesh.vx
        vy = mesh.vy
    else:
        # Fallback: constant velocity in x-direction
        v = m.get('v', 0.0)
        vx = np.full_like(w, v)
        vy = np.zeros_like(w)

    # Laplacian function using central differences
    def lap2d(u):
        d2x = (u[:, mesh.idxRightNeighb] - 2 * u + u[:, mesh.idxLeftNeighb]) / mesh.dx ** 2
        d2y = (u[mesh.idxTopNeighb, :] - 2 * u + u[mesh.idxBottomNeighb, :]) / mesh.dy ** 2
        return d2x + d2y

    # Upwind advection for water (w) using spatially-varying velocity
    dw_dx = np.where(
        vx >= 0,
        (w[:, mesh.idxRightNeighb] - w) / mesh.dx,
        (w - w[:, mesh.idxLeftNeighb]) / mesh.dx
    )
    dw_dy = np.where(
        vy >= 0,
        (w[mesh.idxTopNeighb, :] - w) / mesh.dy,
        (w - w[mesh.idxBottomNeighb, :]) / mesh.dy
    )
    adv_w = vx * dw_dx  # + vy * dw_dy

    return adv_w, lap2d(g), lap2d(s), lap2d(b)

#%%
def _calculate_rhs(cfg : dict, mesh : Mesh, w : np.ndarray, g : np.ndarray, s : np.ndarray, b : np.ndarray):
    """
    Dispatcher function - sends to 1D or 2D RHS calculator
    """
    p = cfg['model']
    dim = cfg['mesh']['dim']
    if dim == 1:
        adv_w, lap_g, lap_s, lap_b = _spatial_ops_1d(cfg, mesh, w, g, s, b)
    elif dim == 2:
        adv_w, lap_g, lap_s, lap_b = _spatial_ops_2d(cfg, mesh, w, g, s, b)
    else:
        raise ValueError(f"Cannot handle dimension {dim}.")
    return _rhs_equations(p, w, g, s, b, adv_w, lap_g, lap_s, lap_b)

#%%
def simulate(cfg : dict, mesh : Mesh, IC : np.ndarray): 
    """ 
    Simulation of the system 
    """

    # Simulation parameters 
    T = cfg['simulation'].get('totalTime', 100.0) 
    dt = cfg['simulation'].get('dt', 0.001)
    nSteps = int(T / dt) 
    save_every = cfg['simulation']['save_every'] 

    # Snapshots
    snapshots_w = []
    snapshots_g = []
    snapshots_s = []
    snapshots_b = []
    times       = []

    # Initial conditions
    w, g, s, b, = IC 

    for step in tqdm(range(nSteps + 1), desc= "Simulating"):
        # print(f"Step = {step} of {nSteps}")

        state = np.array([w, g, s, b])

        if step % save_every == 0:
            snapshots_w.append(w.copy())
            snapshots_g.append(g.copy())
            snapshots_s.append(s.copy())
            snapshots_b.append(b.copy())
            times.append(step * dt)

        # Computing RHS  
        dw, dg, ds, db = _calculate_rhs(cfg, mesh, w, g, s, b)

        # Euler stepping 
        w_new = np.clip(w + dw * dt, 0, None)
        g_new = np.clip(g + dg * dt, 0, None)
        s_new = np.clip(s + ds * dt, 0, None) 
        b_new = np.clip(b + db * dt, 0, None)
        state_new = np.array([w_new, g_new, s_new, b_new])


        # Stopping conditions 
        if (np.linalg.norm(state_new - state) < 1e-6): 
            print("Steady state reached at t = ", step * dt)
            break
        elif step == nSteps: 
            print("Reached max number of steps. Steady satte not reached.")
            break

        if np.any(np.isnan(state_new)): 
            print(f"NaN detected at step {step} (t={step*dt:.3f}). Stopping.")
            break
    
        w = w_new 
        g = g_new 
        b = b_new 
        s = s_new 

    return np.array(snapshots_w), np.array(snapshots_g), np.array(snapshots_s), np.array(snapshots_b), np.array(times)


