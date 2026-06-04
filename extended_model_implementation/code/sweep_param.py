"""
sweep.py — Parameter sweep for the improved Klausmeier model.

Runs the simulation for every combination of parameter values defined
in a sweep config, saving each run to its own directory and collecting
results into a summary CSV.

Output structure
----------------
outputs/
└── <YYYYMMDD>/
    └── sweep_<sweep_name>/
        ├── summary.csv                    ← one row per run
        ├── a=1.0_eq=E1+/                  ← one folder per combination
        │   ├── config.yaml
        │   ├── snapshot_1d.png
        │   └── spacetime_1d.png
        └── a=2.0_eq=E1+/
            └── ...

Sweep config (sweep_config.yaml)
---------------------------------
sweep:
  name: rainfall_sweep           # used in the output folder name

  # Each key is a dotted path into the base config.
  # Values is the list to sweep over.
  params:
    model.a:   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    model.phi: [0.01, 0.05, 0.1]

  # Optional: fix eq_branch per run based on which equilibria exist,
  # or use a fixed branch for all runs.
  eq_branch: E1+                 # override simulation.eq_branch for all runs
                                 # omit to use whatever is in the base config

Run
---
  python sweep.py --config ../configs/extended_model.yaml
                  --sweep  ../configs/sweep_config.yaml
"""

from __future__ import annotations

import copy
import csv
import shutil
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mesh import Mesh
from IC_and_EQ import calculate_eq, get_initial_conditions, make_terrain_and_velocity
from simulation import simulate
from plotting import (
    plot_spacetime_1d, plot_snapshot_1d, animate_1d,
    plot_snapshot_2d, animate_2d,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set cfg['model']['a'] from dotted key 'model.a'."""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def _get_nested(d: dict, dotted_key: str) -> Any:
    """Get cfg['model']['a'] from dotted key 'model.a'."""
    for k in dotted_key.split("."):
        d = d[k]
    return d


def apply_params(base_cfg: dict, param_values: dict[str, Any]) -> dict:
    """Return a deep copy of base_cfg with param_values applied."""
    cfg = copy.deepcopy(base_cfg)
    for key, val in param_values.items():
        _set_nested(cfg, key, val)
    return cfg


def run_label(param_values: dict[str, Any], sweep_cfg: dict) -> str:
    """
    Build a short human-readable folder name from the swept parameter values.
    Skips simulation.eq_branch if it was set at sweep level (constant across
    all runs, so it adds no information to the folder name).

    e.g. {'model.a': 1.5, 'model.phi': 0.05} -> 'a=1.5_phi=0.05'
    """
    sweep_eq = sweep_cfg.get("sweep", {}).get("eq_branch")
    parts = []
    for key, val in param_values.items():
        if key == "simulation.eq_branch" and sweep_eq is not None:
            continue                             # constant — skip from label
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}={val}")
    return "_".join(parts) if parts else "run"


# ─────────────────────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────────────────────

def sweep_root(base_cfg: dict, sweep_cfg: dict, config_path: Path) -> Path:
    """
    outputs/<YYYYMMDD>/sweep_<name>/
    """
    output_cfg  = base_cfg.get("output", {})
    sweep_name  = sweep_cfg.get("sweep", {}).get("name", "unnamed")

    base_dir = Path(str(output_cfg.get("dir", "outputs")).rstrip("/\\"))
    if not base_dir.is_absolute():
        base_dir = config_path.resolve().parents[1] / base_dir

    date_str = datetime.now().strftime("%Y%m%d")
    root = base_dir / date_str / f"sweep_{sweep_name}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg: dict, run_dir: Path) -> dict:
    """
    Run a single simulation with the given cfg, save outputs to run_dir.
    Returns a summary dict (one CSV row).
    """
    output_cfg = cfg.get("output", {})

    # ── mesh ──────────────────────────────────────────────────────────────────
    mesh = Mesh.build_from_config(cfg)
    if cfg["mesh"]["dim"] == 2 and cfg.get("terrain", {}).get("enabled", False):
        h, vx, vy = make_terrain_and_velocity(cfg)
        mesh.set_terrain_and_velocity(h, vx, vy)

    # ── equilibria & ICs ──────────────────────────────────────────────────────
    try:
        equilibria = calculate_eq(cfg)
        IC         = get_initial_conditions(cfg, equilibria)
    except (KeyError, ValueError) as e:
        print(f"    Skipped — IC error: {e}")
        return {"status": f"ic_error: {e}"}

    # ── simulate ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    w, g, s, b, times = simulate(cfg, mesh, IC)
    elapsed = time.perf_counter() - t0

    # ── save config ───────────────────────────────────────────────────────────
    if output_cfg.get("save_config", True):
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # ── plots ─────────────────────────────────────────────────────────────────
    if output_cfg.get("save_plot", True):
        if cfg["mesh"]["dim"] == 1:
            plot_spacetime_1d(w, g, s, b, times, mesh, cfg,
                              save_path=run_dir / "spacetime_1d.png")
            plot_snapshot_1d(w, g, s, b, times, mesh, cfg,
                             save_path=run_dir / "snapshot_1d.png")
        else:
            plot_snapshot_2d(w, g, s, b, times, mesh, cfg,
                             save_path=run_dir / "snapshot_2d.png")

    if output_cfg.get("save_animation", False):   # off by default for sweeps
        if cfg["mesh"]["dim"] == 1:
            animate_1d(w, g, s, b, times, mesh, cfg,
                       save_path=run_dir / "animation_1d.gif")
        else:
            animate_2d(w, g, s, b, times, mesh, cfg,
                       save_path=run_dir / "animation_2d.gif")

    # ── summary stats (final snapshot) ────────────────────────────────────────
    w_f, g_f, s_f, b_f = w[-1], g[-1], s[-1], b[-1]
    return {
        "status":       "ok",
        "final_time":   float(times[-1]),
        "elapsed_s":    round(elapsed, 2),
        "w_mean":       float(np.mean(w_f)),
        "g_mean":       float(np.mean(g_f)),
        "s_mean":       float(np.mean(s_f)),
        "b_mean":       float(np.mean(b_f)),
        "biomass_mean": float(np.mean(g_f + s_f + b_f)),
        "w_max":        float(np.max(w_f)),
        "g_max":        float(np.max(g_f)),
        "s_max":        float(np.max(s_f)),
        "b_max":        float(np.max(b_f)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    # Union of all keys so missing fields get empty strings
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in all_keys})
    print(f"\nSummary CSV → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(base_cfg: dict, sweep_cfg: dict, root: Path) -> None:
    sweep      = sweep_cfg.get("sweep", {})
    params     = sweep.get("params", {})         # {dotted_key: [values]}
    eq_branch  = sweep.get("eq_branch", None)    # optional global override

    if not params:
        raise ValueError("sweep.params is empty — nothing to sweep over.")

    # Build all combinations
    param_keys   = list(params.keys())
    param_values = list(params.values())
    combinations = list(product(*param_values))

    total = len(combinations)
    print(f"Sweep '{sweep.get('name', 'unnamed')}': "
          f"{total} run(s) over {param_keys}")
    print(f"Output root: {root}\n")

    summary_rows: list[dict] = []

    for i, combo in enumerate(combinations, start=1):
        param_dict = dict(zip(param_keys, combo))

        # Override eq_branch if specified at sweep level
        if eq_branch is not None:
            param_dict["simulation.eq_branch"] = eq_branch

        label   = run_label(param_dict, sweep_cfg)
        run_dir = root / label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{i:>{len(str(total))}}/{total}]  {label}")

        cfg     = apply_params(base_cfg, param_dict)
        results = run_one(cfg, run_dir)

        row = {"run": i, "label": label, **param_dict, **results}
        summary_rows.append(row)

        status = results.get("status", "?")
        if status == "ok":
            print(f"    done in {results['elapsed_s']:.1f}s  "
                  f"biomass_mean={results['biomass_mean']:.4f}")

    save_summary(summary_rows, root / "summary.csv")
    print(f"\nAll {total} runs complete → {root}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description=__doc__,
    #              formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("--config", default="configs/extended_model.yaml",
    #                     help="Base model config (yaml)")
    # parser.add_argument("--sweep",  default="configs/sweep_cfg.yaml",
    #                     help="Sweep definition (yaml)")
    # args = parser.parse_args()
    #
    # config_path = Path(args.config)
    # sweep_path  = Path(args.sweep)

    config = Path.joinpath(PROJECT_ROOT, "configs/extended_model.yaml")
    sweep = Path.joinpath(PROJECT_ROOT, "configs/sweep_cfg.yaml")

    config_path = config
    sweep_path  = sweep

    base_cfg  = load_yaml(config_path)
    sweep_cfg = load_yaml(sweep_path)

    root = sweep_root(base_cfg, sweep_cfg, config_path)
    run_sweep(base_cfg, sweep_cfg, root)