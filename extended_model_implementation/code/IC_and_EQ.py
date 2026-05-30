"""
Calculates equilibrium points and builds initial conditions 
"""
import numpy as np 
from scipy.optimize import fsolve

#%%
def _wn_quadratic(p_star, a, alpha):
    disc = a ** 2 - 4.0 * alpha * p_star ** 2
    if disc < 0:
        return np.nan, np.nan
    sq = np.sqrt(disc)
    pos_root = (a + sq) / (2.0 * alpha * p_star)
    neg_root = (a - sq) / (2.0 * alpha * p_star)
    return pos_root, neg_root

#%%
def _det_quadratic_roots(b2, b3, m2, m3, gamma, phi):
    A = b2 * b3
    B = -(b2 * m3 + b3 * (m2 + gamma))
    C = (m2 + gamma) * m3 - phi * gamma
    disc = B ** 2 - 4.0 * A * C
    if disc < 0:
        return np.nan, np.nan
    sq = np.sqrt(disc)
    pos_root = (-B + sq) / (2.0 * A)
    neg_root = (-B - sq) / (2.0 * A)
    return pos_root, neg_root

#%%
def _sb_ratio(p_star, b2, m2, gamma, phi):
    r = (m2 + gamma - b2 * p_star) / phi
    if r > 0:
        return r
    else:
        return np.nan

#%%
def _sb_from_ratio_and_n(r, n_star):
    s_star = n_star / (1.0 + r)
    b_star = r * n_star / (1.0 + r)
    return s_star, b_star

#%%
def _eq_E0(p: dict) -> np.ndarray:
    """
    E0: Trivial barren state (w* = a, g* = s* = b* = 0)
    Always exists.
    """
    a = p['a']
    print(f"E0: w* = {a}")
    return np.array([a, 0.0, 0.0, 0.0])

#%%
def _eq_E1(p: dict) -> tuple:
    """
    E1+/E1-: Grass only state (g* > 0, s* = b* = 0)
    Returns a tuple of (E1+, E1-).
    """
    a, alpha = p['a'], p['alpha']
    b1, m1 = p['b1'], p['m1']

    disc1 = (b1 * a) ** 2 - 4 * alpha * m1 ** 2
    if disc1 < 0:
        print(f"E1 does not exist: need b1 * a >= 2 * m1 * sqrt(alpha)")
        nan4 = np.full(4, np.nan)
        return nan4, nan4

    sq = np.sqrt(disc1)
    g_plus = (b1 * a + sq) / (2 * alpha * m1)
    g_minus = (b1 * a - sq) / (2 * alpha * m1)
    w_plus = a / (1 + alpha * g_plus ** 2)
    w_minus = a / (1 + alpha * g_minus ** 2)

    E1p = np.array([w_plus, g_plus, 0.0, 0.0])
    E1m = np.array([w_minus, g_minus, 0.0, 0.0])
    print(f"E1+: w*={w_plus:.4f}, g*={g_plus:.4f}")
    print(f"E1-: w*={w_minus:.4f}, g*={g_minus:.4f}")
    return E1p, E1m

#%%
def _eq_E2(p: dict) -> np.ndarray:
    """
    E2: Saplings + trees state (g* = 0, s* > 0, b* > 0)
    Calculated analytically. Returns the + branch representative.
    """
    a, alpha = p['a'], p['alpha']
    b2, m2 = p['b2'], p['m2']
    b3, m3 = p['b3'], p['m3']
    gamma = p['gamma']
    phi = p['phi']

    p_plus, p_minus = _det_quadratic_roots(b2, b3, m2, m3, gamma, phi)

    if np.isnan(p_plus):
        print("E2 does not exist: determinant quadratic has no real roots.")
        return np.full(4, np.nan)

    found = []
    for p_star, p_branch in [(p_plus, 'p+'), (p_minus, 'p-')]:
        if p_star <= 0:
            continue
        r = _sb_ratio(p_star, b2, m2, gamma, phi)
        if np.isnan(r):
            continue

        n_plus, n_minus = _wn_quadratic(p_star, a, alpha)
        for n_star, n_branch in [(n_plus, 'n+'), (n_minus, 'n-')]:
            if np.isnan(n_star) or n_star <= 0:
                continue
            s_star, b_star = _sb_from_ratio_and_n(r, n_star)
            if s_star <= 0 or b_star <= 0:
                continue
            w_star = a / (1.0 + alpha * n_star ** 2)
            label = f"{p_branch},{n_branch}"
            print(f"E2 ({label}): w*={w_star:.6g},  s*={s_star:.6g},  "
                  f"b*={b_star:.6g}  [p*={p_star:.4g}, n*={n_star:.4g}, r={r:.4g}]")
            found.append(np.array([w_star, 0.0, s_star, b_star]))

    if not found:
        print("E2 (saplings+trees) does not exist for these parameters.")
    return found

#%%
def _eq_E_all(p: dict) -> np.ndarray:
    """
    E_all: All species coexistence continuum (g* > 0, s* > 0, b* > 0)
    Returns a representative point exactly in the middle of the valid b* range.
    """
    a, alpha = p['a'], p['alpha']
    b1, m1 = p['b1'], p['m1']
    b2, m2 = p['b2'], p['m2']
    b3, m3 = p['b3'], p['m3']
    gamma = p['gamma']
    phi = p['phi']

    # p* fixed by grass
    p_star = m1 / b1

    # Compatibility condition
    det_val = (b2 * p_star - (m2 + gamma)) * (b3 * p_star - m3) - phi * gamma
    if abs(det_val) > 1e-8:
        print(f"E_all does not exist: compatibility not satisfied  "
              f"(det = {det_val:+.6g} != 0 at p* = m1/b1 = {p_star:.4g})")
        return np.full(4, np.nan)

    # Existence condition for r > 0: b_3 * p* < m3
    if b3 * p_star >= m3:
        print(f"E_all does not exist: need b_3 * p* < m3"
              f"(b_3 * p* = {b3 * p_star:.4g} >= m3 = {m3:.4g})")
        return np.full(4, np.nan)

    # Ratio r = b*/s* from row (S)
    r = _sb_ratio(p_star, b2, b3, m2, m3, gamma, phi)
    if np.isnan(r):
        print("E_all does not exist: r = b*/s* <= 0")
        return np.full(4, np.nan)

    # n* from water quadratic
    n_star, _ = _wn_quadratic(p_star, a, alpha)
    if np.isnan(n_star):
        print(f"E_all does not exist: a < 2p*√α  "
              f"(a = {a:.4g},  2p*√α = {2 * p_star * np.sqrt(alpha):.4g})")
        return np.full(4, np.nan)

    # Valid s range: 0 < s < s_max = n*/(1+r)
    s_max = n_star / (1.0 + r)

    # Representative point at midpoint of valid range
    s_rep = s_max / 2.0
    b_rep = r * s_rep
    g_rep = n_star - s_rep * (1.0 + r)  # = n* - s_max = s_max > 0 by construction
    w_star = a / (1.0 + alpha * n_star ** 2)

    print(f"E_all (all-species, CONTINUUM): w* = {w_star:.6g},  n* = {n_star:.6g},  "
          f"r = b/s = {r:.4g},  s ∈ (0, {s_max:.4g})")
    print(f"  Representative (s = s_max/2): "
          f"g* = {g_rep:.6g},  s* = {s_rep:.6g},  b* = {b_rep:.6g}")
    return np.array([w_star, g_rep, s_rep, b_rep])

#%%
def calculate_eq(cfg: dict) -> dict:
    """
    Orchestrates the calculation of all analytical equilibrium points.

    Input:
        cfg (dict): Configuration dictionary
    Returns:
        dict of equilibria, each a np.ndarray of shape (4,) = [w*, g*, s*, b*]
    """
    p = cfg['model']
    equilibria = {}

    equilibria['E0'] = _eq_E0(p)
    equilibria['E1+'], equilibria['E1-'] = _eq_E1(p)
    for i, eq in enumerate(_eq_E2(p), start=1):
        equilibria[f'E2_{i}'] = eq
    if not any(k.startswith('E2') for k in equilibria):
        equilibria['E2_1'] = np.full(4, np.nan)
    equilibria['E_all'] = _eq_E_all(p)

    return equilibria

#%%
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
    label = cfg['simulation']['eq_branch']
    eq = equilibria.get(label)

    if eq is None:
        raise KeyError(f"Unknown equilibrium branch '{label}'.")
    if np.any(np.isnan(eq)):
        raise ValueError(f"Equilibrium '{label}' does not exist for these parameters.")

    if label in ('E3', 'E_all'):
        raise Warning(f"{label} is a continuum of equilibria.")

    print(f"Using initial conditions near {label}: [w,g,s,b] = {eq}")

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

#%%
def make_terrain_and_velocity(cfg: dict):
    """
    Generate periodic rolling-hill terrain and derive spatially-varying velocity field.

    The velocity field is derived from terrain gradients (steepest descent) and capped
    at v_max to ensure numerical stability.

    Inputs:
        cfg (dict): Configuration dictionary containing mesh and velocity parameters

    Returns:
        h (np.ndarray): Terrain height field, shape (nCellsY, nCellsX) for 2D or (nCells,) for 1D
        vx (np.ndarray): x-component velocity field, same shape as h
        vy (np.ndarray): y-component velocity field, same shape as h (None for 1D)
    """
    mesh_cfg = cfg['mesh']
    terrain_cfg = cfg.get('terrain', {})
    vel_cfg = cfg.get('velocity', {})

    # Terrain parameters
    Nx = mesh_cfg['nCellsX'] if mesh_cfg['dim'] == 2 else mesh_cfg['nCells']
    Ny = mesh_cfg['nCellsY'] if mesh_cfg['dim'] == 2 else 1
    Lx = mesh_cfg['Lx'] if mesh_cfg['dim'] == 2 else mesh_cfg['L']
    Ly = mesh_cfg['Ly'] if mesh_cfg['dim'] == 2 else mesh_cfg['L']

    n_octaves = terrain_cfg.get('n_octaves', 5)
    amplitude_rolloff = terrain_cfg.get('amplitude_rolloff', 0.9)
    seed = terrain_cfg.get('seed', 42)
    v_max = vel_cfg.get('v_max', 1.0)

    # --- Generate periodic terrain via Fourier modes ---
    rng = np.random.default_rng(seed)
    h = np.zeros((Ny, Nx))

    for octave in range(n_octaves):
        freq = 2 ** octave
        amp = 1.0 / (freq ** amplitude_rolloff)

        # Random integer wavenumbers for this octave
        size = 3
        kx_int = rng.integers(-freq, freq + 1, size=size)
        ky_int = rng.integers(-freq, freq + 1, size=size)
        phases = rng.uniform(0, 2 * np.pi, size=size)

        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)

        for kx_i, ky_i, phi in zip(kx_int, ky_int, phases):
            h += amp * np.sin(2 * np.pi * kx_i * X / Lx
                              + 2 * np.pi * ky_i * Y / Ly
                              + phi)

    # Normalize h to [0, 1]
    h = (h - h.min()) / (h.max() - h.min() + 1e-12)

    # --- Derive velocity field from terrain gradients ---
    dx = Lx / Nx
    dy = Ly / Ny

    hx, hy = np.gradient(h, dx, dy)

    # Magnitude of slope
    slope = np.sqrt(hx ** 2 + hy ** 2)
    slope_median = np.median(slope) + 1e-8

    # Velocity proportional to slope (steepest descent)
    vx_raw = v_max * (hx / slope_median)
    vy_raw = v_max * (hy / slope_median)

    # Cap velocity magnitude at v_max
    speed = np.sqrt(vx_raw ** 2 + vy_raw ** 2)
    scale = np.minimum(1.0, v_max / (speed + 1e-8))

    vx = vx_raw * scale
    vy = vy_raw * scale

    return h, vx, vy