import numpy as np
import shutil
import yaml
import sys
from pathlib import Path
import datetime
# Written functions in external files 
from mesh import Mesh
from IC_and_EQ import calculate_eq, get_initial_conditions, make_terrain_and_velocity
from simulation import simulate
from plotting import plot_spacetime_1d, plot_snapshot_1d, animate_1d, animate_2d, plot_snapshot_2d

# Gets parent folder name
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#%% Loading configurations
def load_config(path : str):
    """
    Inputs: 
        path (str): Path name to config file 
    Outputs: 
        cfg (dict): Dictionary with parameters 
    """
    with open(path) as f: 
        cfg = yaml.safe_load(f)
    return cfg

#%%
def output_dir_from_config(cfg: dict, config_path: Path):
    output_cfg = cfg.get("output", {})

    base_dir = Path(str(output_cfg.get("dir", "outputs")).rstrip("\\/"))
    if not base_dir.is_absolute():
        base_dir = config_path.resolve().parents[1] / base_dir

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")

    run_dir = Path.joinpath(base_dir, date_str, time_str)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

#%%
if __name__ == "__main__":

    # Loading in configs 
    config_path = Path.joinpath(PROJECT_ROOT, "configs/extended_model.yaml")
    cfg = load_config(config_path)
    
    # Create output directory if it doesn't exist
    output_dir = output_dir_from_config(cfg, config_path)
    output_cfg  = cfg.get("output", {})

    mesh = Mesh.build_from_config(cfg)
    
    # Generate terrain and velocity field if enabled (2D only)
    if cfg['mesh']['dim'] == 2 and cfg['terrain'].get('enabled', False):
        print("Generating terrain-based velocity field...")
        h, vx, vy = make_terrain_and_velocity(cfg)
        mesh.set_terrain_and_velocity(h, vx, vy)
        print(f"Terrain attached to mesh")
        print(f"  Terrain shape: {h.shape}")
        print(f"  Terrain range: [{h.min():.4f}, {h.max():.4f}]")
        v_mag = np.sqrt(vx**2 + vy**2)
        print(f"  Velocity magnitude range: [{v_mag.min():.6f}, {v_mag.max():.6f}]")
        print(f"  Max velocity magnitude: {v_mag.max():.6f}")
    else:
        print(f"Terrain disabled (using constant velocity)")
        print(f"  terrain.enabled = {cfg['terrain'].get('enabled')}")
        print(f"  mesh.dim = {cfg['mesh']['dim']}")
    
    equilibria = calculate_eq(cfg)
    IC = get_initial_conditions(cfg, equilibria)

    w, g, s, b, times = simulate(cfg, mesh, IC)

    if output_cfg.get("save_config", True):
        shutil.copy2(str(config_path), str(Path.joinpath(output_dir, "config.yaml")))

    if output_cfg.get("save_plot", True):
        if cfg['mesh']['dim'] == 1:
            plot_spacetime_1d(w, g, s, b, times, mesh, cfg, save_path=output_dir / f"spacetime_1d.png")
            plot_snapshot_1d(w, g, s, b, times, mesh, cfg, save_path=output_dir / f"snapshot_1d.png")
        elif cfg['mesh']['dim'] == 2:
            plot_snapshot_2d(w, g, s, b, times, mesh, cfg, save_path=output_dir / f"snapshot_2d.png")

    if output_cfg.get("save_animation", True):
        if cfg['mesh']['dim'] == 1:
            animate_1d(w, g, s, b, times, mesh, cfg , save_path=output_dir / f"animation_1d.gif")
        elif cfg['mesh']['dim'] == 2:
            animate_2d(w, g, s, b, times, mesh, cfg, save_path=output_dir / f"animation_2d.gif")
