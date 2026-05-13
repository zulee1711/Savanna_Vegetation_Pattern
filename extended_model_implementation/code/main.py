import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
from pathlib import Path

# Written functions in external files 
from mesh import Mesh
from IC_and_EQ import calculate_eq, get_initial_conditions
from simulation import simulate
from plotting import plot_spacetime_1d, plot_snapshot_1d, animate_1d, animate_2d, plot_snapshot_2d

# Gets parent folder name
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Loading configurations
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

if __name__ == "__main__":

    # Loading in configs 
    config_path = Path.joinpath(PROJECT_ROOT, "configs/extended_model.yaml")
    cfg = load_config(config_path)
    
    # Create output directory if it doesn't exist
    output_dir = Path(PROJECT_ROOT) / cfg['simulation'].get('output_dir', 'outputs')
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = Mesh.build_from_config(cfg)
    equilibria = calculate_eq(cfg)
    IC = get_initial_conditions(cfg, equilibria)

    w, g, s, b, times = simulate(cfg, mesh, IC)

    if cfg['mesh']['dim'] == 1:
        plot_spacetime_1d(w, g, s, b, times, mesh, cfg) #, save_path=output_dir / "spacetime_1d.png")
        plot_snapshot_1d(w, g, s, b, times, mesh, cfg) #, save_path=output_dir / "snapshot_1d.png")
        animate_1d(w, g, s, b, times, mesh, cfg) # , save_path=output_dir / "animation_1d.gif")
    elif cfg['mesh']['dim'] == 2:
        plot_snapshot_2d(w, g, s, b, times, mesh, cfg) #, save_path=output_dir / "snapshot_2d.png")
        animate_2d(w, g, s, b, times, mesh, cfg) #, save_path=output_dir / "animation_2d.gif")