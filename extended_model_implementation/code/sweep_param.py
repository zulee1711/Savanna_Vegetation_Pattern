"""
sweep.py - Parameter sweep for the improved Klausmeier model.

Runs the simulation for every combination of parameter values defined
in a sweep config, saving each run to its own directory and collecting
results into a summary CSV.

Output structure
----------------
outputs/
└── <YYYYMMDD>/
    └── sweep_<sweep_name>/
        ├── summary.csv                    <- one row per run
        ├── a=1.0_eq=E1+/                  <- one folder per combination
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

import logging
from logging_utils import setup_logging

logger = logging.getLogger(__name__)

import copy
import csv
import os
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def apply_params(base_cfg: dict, param_values: dict[str, Any]) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key, val in param_values.items():
        _set_nested(cfg, key, val)
    return cfg


def run_label(param_values: dict[str, Any], sweep_cfg: dict) -> str:
    sweep_eq = sweep_cfg.get("sweep", {}).get("eq_branch")
    parts = []
    for key, val in param_values.items():
        if key == "simulation.eq_branch" and sweep_eq is not None:
            continue
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}={val}")
    return "_".join(parts) if parts else "run"


def sweep_root(base_cfg: dict, sweep_cfg: dict, config_path: Path) -> Path:
    output_cfg = base_cfg.get("output", {})
    sweep_name = sweep_cfg.get("sweep", {}).get("name", "unnamed")

    base_dir = Path(str(output_cfg.get("dir", "outputs")).rstrip("/\\"))
    if not base_dir.is_absolute():
        base_dir = config_path.resolve().parents[1] / base_dir

    date_str = datetime.now().strftime("%Y%m%d")
    root = base_dir / date_str / f"sweep_{sweep_name}"
    root.mkdir(parents=True, exist_ok=True)
    return root

def setup_worker_logging(log_path: Path) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg: dict, run_dir: Path) -> dict:
    output_cfg = cfg.get("output", {})

    mesh = Mesh.build_from_config(cfg)
    if cfg["mesh"]["dim"] == 2 and cfg.get("terrain", {}).get("enabled", False):
        h, vx, vy = make_terrain_and_velocity(cfg)
        mesh.set_terrain_and_velocity(h, vx, vy)

    try:
        equilibria = calculate_eq(cfg)
        IC = get_initial_conditions(cfg, equilibria)
    except (KeyError, ValueError) as e:
        return {"status": f"ic_error: {e}"}

    t0 = time.perf_counter()
    w, g, s, b, times = simulate(cfg, mesh, IC)
    elapsed = time.perf_counter() - t0

    if output_cfg.get("save_config", True):
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if output_cfg.get("save_plot", True):
        if cfg["mesh"]["dim"] == 1:
            plot_spacetime_1d(w, g, s, b, times, mesh, cfg,
                              save_path=run_dir / "spacetime_1d.png")
            plot_snapshot_1d(w, g, s, b, times, mesh, cfg,
                             save_path=run_dir / "snapshot_1d.png")
        else:
            plot_snapshot_2d(w, g, s, b, times, mesh, cfg,
                             save_path=run_dir / "snapshot_2d.png")

    if output_cfg.get("save_animation", False):
        if cfg["mesh"]["dim"] == 1:
            animate_1d(w, g, s, b, times, mesh, cfg,
                       save_path=run_dir / "animation_1d.gif")
        else:
            animate_2d(w, g, s, b, times, mesh, cfg,
                       save_path=run_dir / "animation_2d.gif")

    if len(times) == 0:
        return {"status": "error: no snapshots saved"}

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


def worker(job: tuple[int, dict[str, Any], dict, dict, str]) -> dict:
    """Run one parameter combination in a separate process."""
    run_i, param_dict, base_cfg, sweep_cfg, root_str = job

    root = Path(root_str)
    label = run_label(param_dict, sweep_cfg)
    run_dir = root / label
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    setup_worker_logging(log_path)

    try:
        import simulation as simulation_module
        simulation_module.tqdm = lambda iterable, *args, **kwargs: iterable
    except Exception:
        pass

    try:
        cfg = apply_params(base_cfg, param_dict)

        # Force expensive outputs off for sweeps.
        cfg.setdefault("output", {})
        cfg["output"]["save_animation"] = True
        cfg["output"]["save_plot"] = True

        # Capture any remaining print() calls from old code.
        with open(log_path, "a") as log_file, redirect_stdout(log_file), redirect_stderr(log_file):
            results = run_one(cfg, run_dir)

        return {"run": run_i, "label": label, **param_dict, **results}

    except Exception as e:
        logging.getLogger(__name__).exception("Worker failed for %s", label)
        return {"run": run_i, "label": label, **param_dict, "status": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r.get("run", 0))
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in all_keys})


# ─────────────────────────────────────────────────────────────────────────────
# Parallel sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_parallel(base_cfg: dict, sweep_cfg: dict, root: Path, workers: int, logger: logging.Logger) -> None:
    sweep = sweep_cfg.get("sweep", {})
    params = sweep.get("params", {})
    eq_branch = sweep.get("eq_branch", None)

    if not params:
        raise ValueError("sweep.params is empty - nothing to sweep over.")

    param_keys = list(params.keys())
    combinations = list(product(*params.values()))
    total = len(combinations)

    jobs = []
    for i, combo in enumerate(combinations, start=1):
        param_dict = dict(zip(param_keys, combo))
        if eq_branch is not None:
            param_dict["simulation.eq_branch"] = eq_branch
        jobs.append((i, param_dict, base_cfg, sweep_cfg, str(root)))

    logger.info("Sweep '%s': %s run(s) over %s", sweep.get("name", "unnamed"), total, param_keys)
    logger.info("Output root: %s", root)
    logger.info("Workers: %s", workers)

    rows: list[dict] = []
    completed = 0
    ok = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            completed += 1
            if row.get("status") == "ok":
                ok += 1
                logger.info("[%s/%s] ok | %s | biomass_mean=%.4f | elapsed_s=%s",
                            completed, total, row["label"], row.get("biomass_mean", float("nan")), row.get("elapsed_s"))
            else:
                logger.warning("[%s/%s] %s | %s", completed, total, row.get("status"), row["label"])

    summary_path = root / "summary.csv"
    save_summary(rows, summary_path)
    logger.info("Finished: %s/%s ok", ok, total)
    logger.info("Summary CSV: %s", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Parallel parameter sweep with logging.")
    # parser.add_argument("--config", default="configs/extended_model.yaml", help="Base model config YAML")
    # parser.add_argument("--sweep", default="configs/sweep_cfg.yaml", help="Sweep config YAML")
    # parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Number of worker processes")
    # parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    # args = parser.parse_args()
    #
    # config_path = Path(args.config)
    # sweep_path  = Path(args.sweep)

    log_path = setup_logging(run_date=None, log_name="sweep.log")

    config = Path.joinpath(PROJECT_ROOT, "configs/extended_model.yaml")
    sweep = Path.joinpath(PROJECT_ROOT, "configs/sweep_cfg.yaml")
    workers = 8

    config_path = config
    sweep_path  = sweep

    base_cfg  = load_yaml(config_path)
    sweep_cfg = load_yaml(sweep_path)

    root = sweep_root(base_cfg, sweep_cfg, config_path)
    run_sweep_parallel(base_cfg, sweep_cfg, root, workers, logger)