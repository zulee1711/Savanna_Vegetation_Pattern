"""
Calculates equilibrium points and builds initial conditions 
"""
import numpy as np 
from scipy.optimize import fsolve

def calculate_eq(cfg: dict): 
    """
    Calculates equilibrium points (w*, g*, s*, b*) for the extended Klausmeier model.

    Input:
        cfg (dict): Configuration dictionary
    Returns:
        dict of equilibria, each a np.ndarray of shape (4,) = [w*, g*, s*, b*]
    """
    p = cfg['model']
    a      = p['a']
    alpha  = p['alpha']
    b1, m1 = p['b1'], p['m1']
    b2, m2 = p['b2'], p['m2']
    b3, m3 = p['b3'], p['m3']
    gamma  = p['gamma']

    equilibria = {} 

    # --- E0: trivial barren state -----------------------------
    
    equilibria['E0'] = np.array([a, 0.0, 0.0, 0.0])
    print(f"E0: w* = {a}")

    # --- E1+/E1-: Grass only, s* = b* = 0 -------------------------
    
    disc = (b1 * a)**2 - 4 * alpha * m1**2
    if disc >= 0: 
        g_plus = (b1 * a + np.sqrt(disc)) / (2 * alpha * m1)
        g_minus = (b1 * a - np.sqrt(disc)) / (2 * alpha * m1)
        w_plus = a / (1 + alpha * g_plus**2)
        w_minus = a / (1 + alpha * g_minus**2)
        equilibria['E1+'] = np.array([w_plus, g_plus, 0.0, 0.0])
        equilibria['E1-'] = np.array([w_minus, g_minus, 0.0, 0.0])
        print(f"E1+: w*={w_plus:.4f}, g*={g_plus:.4f}")
        print(f"E1-: w*={w_minus:.4f}, g*={g_minus:.4f}")

    else: 
        print(f"E1 does not exist: need b1*a >= 2*m1*sqrt(alpha)")
        equilibria['E1+'] = np.full(4, np.nan)
        equilibria['E1-'] = np.full(4, np.nan)

    # --- E2: Saplings + trees, g* = 0 -----------------------------
    
    def _eq3(vars):
        s, b, w = vars
        n = s + b
        eq_w = a - w - alpha * w * n**2
        eq_s = b2 * w * n * s - m2 * s - gamma * s
        eq_b = b3 * w * n * b - m3 * b + gamma * s
        return [eq_w, eq_s, eq_b]
      
    # Using scipy fsolve to find roots of coupled nonlinear equations 
    E2_found = False
    for s0, b0, w0 in [(1.0, 1.0, 0.5), (0.5, 2.0, 0.3), (2.0, 0.5, 0.2)]: # Trying multiple initial guess to find solution 
        try:
            sol = fsolve(_eq3, [s0, b0, w0], full_output=True)
            x, info, integer, _ = sol
            s, b, w = x
            residual = np.linalg.norm(info['fvec']) # Output 
            if integer == 1 and residual < 1e-10 and s > 0 and b > 0 and w > 0:
                equilibria['E2'] = np.array([w, 0.0, s, b])
                print(f"E2: w*={w:.4f}, s*={s:.4f}, b*={b:.4f}")
                E2_found = True
                break
        except:
            continue
    if not E2_found:
        print("E2 (only saplings and trees) not found for these parameters.")
        equilibria['E2'] = np.full(4, np.nan)
    
    # --- E2/E4: Checking compatibilty constraint -------------------------------

    compatibility = (b2 / b1) - ((m2 + gamma)/m1)
    if abs(compatibility) < 1e-8: 
        print(f"Compatibility = {compatibility}. Constraint satisfied. Coexistence possible.")
    else: 
        print(f"Compatibility = {compatibility}. Coexistence not possible.")

    return equilibria

def get_initial_conditions(cfg : dict, equilibria : dict):
    """
    Returns initial conditions near the equilibria of choice 

    Inputs; 
        cfg (dict): Dictionary of parameters 
        equilibria (np.ndarray): Array holding equilibria E0, E1+, E1-
    Returns: 
        w (np.ndarray): Initial condition - perturbation around w*
        g (np.ndarray): Initial condition - perturbation around g*
        s (np.ndarray): Initial condition - perturbation around s*
        b (np.ndarray): Initial condition - perturbation around b*
    """
    equilibrium = cfg['simulation']['eq_branch']
   
    if equilibrium == "E0": 
        eq = equilibria['E0']
        print("Using initial conditions near E0")
    elif equilibrium == "E1+": 
        eq = equilibria['E1+'] 
        print("Using initial conditions near E1+")
    elif equilibrium == "E1-": 
        eq = equilibria['E1-'] 
        print("Using initial conditions near E1-")
    elif equilibrium == "E2": 
        print("Using initial conditions near E2")
        eq = equilibria['E2'] 
    
    if np.any(np.isnan(eq)): 
        raise ValueError(f"Equilibrium {equilibrium} does not exist for these parameters")
    
    w_eq, g_eq, s_eq, b_eq = eq 

    dim = cfg['mesh']['dim']
    if dim == 1: 
        shape = (cfg['mesh']['nCells'], )
    else:  
        shape = (cfg['mesh']['nCellsX'], cfg['mesh']['nCellsY'])

    # Peturbing around equilibrium 
    perturb = cfg['simulation'].get('perturbation', 0.01)
    seed = cfg['simulation'].get('seed', 100) 
   
    rng = np.random.default_rng(seed)

    if cfg['model']['type'] == 'klausmeier': 
        w = w_eq + perturb * rng.standard_normal(shape) 
        g = g_eq + perturb * rng.standard_normal(shape) 
        s = np.zeros(shape) 
        b = np.zeros(shape) 
    elif cfg['model']['type'] == 'extended': 
        w = w_eq + perturb * rng.standard_normal(shape) 
        g = g_eq + perturb * rng.standard_normal(shape) 
        s = s_eq + perturb * rng.standard_normal(shape) 
        b = b_eq + perturb * rng.standard_normal(shape) 

    # Ensuring all non-negative
    w = np.clip(w, 0.0, None)
    g = np.clip(g, 0.0, None)
    s = np.clip(s, 0.0, None)
    b = np.clip(b, 0.0, None)

    return w, g, s, b # Python automatically packs as tuple 






















    # model_type = cfg['model']['type']

    # if model_type == 'klausmeier':
    #     return _eq_klausmeier(cfg)
    # elif model_type == 'extended':
    #     return _eq_extended(cfg)
    # raise ValueError(f"Unknown model type: {model_type}. Choose 'klausmeier' or 'extended'.")


# def _eq_klausmeier(cfg: dict) -> np.ndarray:
#     """Equilibria for the original Klausmeier model."""
#     m = cfg['model']['m_g']
#     a = cfg['model']['b_g']   # in Klausmeier, a is the rainfall parameter = b_g

#     E0 = np.array([1.0, 0.0, 0.0, 0.0])

#     discriminant = a**2 - 4 * m**2
#     if discriminant < 0:
#         print(f"No real equilibrium. Need a >= 2m. Got a={a}, 2m={2*m:.4f}")
#         return np.array([E0, np.full(4, np.nan), np.full(4, np.nan)])

#     n_plus  = (a + np.sqrt(discriminant)) / (2 * m)
#     n_minus = (a - np.sqrt(discriminant)) / (2 * m)
#     w_plus  = a / (1.0 + n_plus**2)
#     w_minus = a / (1.0 + n_minus**2)

#     print(f"E1+: w*={w_plus:.4f}, n*={n_plus:.4f}")
#     print(f"E1-: w*={w_minus:.4f}, n*={n_minus:.4f}")

#     return np.array([
#         E0,
#         np.array([w_plus,  n_plus,  0.0, 0.0]),
#         np.array([w_minus, n_minus, 0.0, 0.0]),
#     ])


# def _eq_extended(cfg: dict) -> np.ndarray:
#     """Equilibria for the extended Klausmeier model."""
#     m  = cfg['model']
#     b_g = m['b_g']
#     m_g = m['m_g']
#     a   = m['a']

#     E0 = np.array([1.0, 0.0, 0.0, 0.0])

#     discriminant = b_g**2 - 4 * a * m_g**2
#     if discriminant < 0:
#         print(
#             f"No real E1 equilibrium. Need b_g >= 2*m_g*sqrt(a).\n"
#             f"Got b_g={b_g}, 2*m_g*sqrt(a)={2*m_g*np.sqrt(a):.4f}"
#         )
#         return np.array([E0, np.full(4, np.nan), np.full(4, np.nan)])

#     g_plus  = (b_g + np.sqrt(discriminant)) / (2 * a * m_g)
#     g_minus = (b_g - np.sqrt(discriminant)) / (2 * a * m_g)
#     w_plus  = 1.0 / (1.0 + a * g_plus**2)
#     w_minus = 1.0 / (1.0 + a * g_minus**2)

#     print(f"E1+: w*={w_plus:.4f}, g*={g_plus:.4f}")
#     print(f"E1-: w*={w_minus:.4f}, g*={g_minus:.4f}")

#     return np.array([
#         E0,
#         np.array([w_plus,  g_plus,  0.0, 0.0]),
#         np.array([w_minus, g_minus, 0.0, 0.0]),
#     ])

