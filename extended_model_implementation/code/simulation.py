"""
Simulation function 
"""
import numpy as np
from mesh import Mesh 

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

    for step in range(nSteps + 1): 
        print(f"Step = {step} of {nSteps}")

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

def _calculate_rhs(cfg : dict, mesh : Mesh, w : np.ndarray, g : np.ndarray, s : np.ndarray, b : np.ndarray): 
    """ 
    Dispatcher function - sends to 1D or 2D RHS calculator
    """
    dim = cfg['mesh']['dim']
    if dim == 1: 
        return _rhs_1d(cfg, mesh, w, g, s, b)
    elif dim == 2:  
        return _rhs_2d(cfg, mesh, w, g, s, b)
    else: 
        raise ValueError(f"Cannot handle dimension {dim}.")

def _rhs_1d(cfg : dict, mesh : Mesh, w : np.ndarray, g : np.ndarray, s : np.ndarray, b : np.ndarray): 
    """ 
    Computes 1D RHS 
    """
    m = cfg['model']
    a = m['a']
    alpha = m['alpha']
    v = m['v']
    b1 = m['b1']
    m1 = m['m1']
    d1 = m['d1']
    b2 = m['b2']
    m2 = m['m2']
    gamma = m['gamma']
    d2 = m['d2']
    b3 = m['b3']
    m3 = m['m3']
    n = g + s + b

    # Calculating spatial derivatives 
    if v >= 0:
        dw_dx = (w[mesh.idxRightNeighb] - w[mesh.idx]) / mesh.dx     # forward difference
    else:
        dw_dx = (w[mesh.idx] - w[mesh.idxLeftNeighb]) / mesh.dx      # backward difference

    d2g_dx2 = (g[mesh.idxRightNeighb] - (2 * g[mesh.idx]) + g[mesh.idxLeftNeighb]) / (mesh.dx**2) 
    d2b_dx2 = (b[mesh.idxRightNeighb] - (2 * b[mesh.idx]) + b[mesh.idxLeftNeighb]) / (mesh.dx**2) 

    # Calculating RHS 
    dw = a - w - (alpha * w * n**2) + v * dw_dx
    dg = (b1 * w * n * g) - (m1 * g) + (d1 * d2g_dx2)
    ds = (b2 * w * n * s) - (m2 * s) - (gamma * s) + (d2 * d2b_dx2)
    db = (b3 * w * n * b) - (m3 * b) + (gamma * s) 

    return dw, dg, ds, db

def _rhs_2d(cfg : dict, mesh : Mesh, w : np.ndarray, g : np.ndarray, s : np.ndarray, b : np.ndarray): 
    """
    Computes RHS 2D 
    """
    m = cfg['model']
    a = m['a']
    alpha = m['alpha']
    v = m['v']
    b1 = m['b1']
    m1 = m['m1']
    d1 = m['d1']
    b2 = m['b2']
    m2 = m['m2']
    gamma = m['gamma']
    d2 = m['d2']
    b3 = m['b3']
    m3 = m['m3']
    n = g + s + b

    # Laplacian function using central differences
    def laplacian(u): 
        d2x = (u[mesh.idxRightNeighb, :] - (2 * u[mesh.idxX, :]) + u[mesh.idxLeftNeighb, :]) / (mesh.dx**2) 
        d2y = (u[:, mesh.idxTopNeighb] - (2 * u[:, mesh.idxY]) + u[:, mesh.idxBottomNeighb]) / (mesh.dy**2) 
        return d2x + d2y

    # Calculating spatial derivatives 
    if v >= 0:
        dw_dx = (w[:, mesh.idxRightNeighb] - w[:, mesh.idxX]) / mesh.dx
    else:
        dw_dx = (w[:, mesh.idxX] - w[:, mesh.idxLeftNeighb]) / mesh.dx

    # Calculating RHS 
    dw = a - w - (alpha * w * n**2) + v * dw_dx
    dg = (b1 * w * n * g) - (m1 * g) + (d1 * laplacian(g))
    ds = (b2 * w * n * s) - (m2 * s) - (gamma * s) + (d2 * laplacian(b))
    db = (b3 * w * n * b) - (m3 * b) + (gamma * s) 
    
    return dw, dg, ds, db
