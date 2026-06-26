# Savanna Vegetation Pattern

Simulations of vegetation pattern formation in savanna ecosystems using reaction-advection-diffusion PDE models.

This repository includes:

* A 1D simplified Klausmeier model
* A 2D Klausmeier vegetation-water model
* Stability analysis tools for the 1D model
* An extended multi-species model with grass, saplings, and adult trees

## Overview

The models simulate water-vegetation interactions in dryland and savanna landscapes. Spatial vegetation patterns emerge from the interaction between:

* Water input, loss, redistribution, and advection
* Vegetation growth and mortality
* Spatial diffusion or spread of vegetation
* Species-specific water uptake and life-stage dynamics in the extended model

The original Klausmeier model describes the interaction between water and a single vegetation biomass variable. The extended model separates vegetation into grass, saplings, and adult trees, allowing the simulations to capture more realistic savanna dynamics.

## Repository Structure

```text
.
├── 1D_simple_model.py
├── 2D_klausmeier.py
├── stability_analysis_simple_1D/
├── extended_model_implementation/
│   ├── code/
│   │   ├── IC_and_EQ.py
│   │   ├── logging_utils.py
│   │   ├── main.py
│   │   ├── mesh.py
│   │   ├── plotting.py
│   │   ├── simulation.py
│   │   └── sweep_param.py
│   ├── configs/
│   │   ├── extended_model.yaml
│   │   └── sweep_cfg.yaml
│   └── outputs/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/zulee1711/Savanna_Vegetation_Pattern.git
cd Savanna_Vegetation_Pattern
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, use:

```bash
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Models

### 1D Simple Klausmeier Model

Run:

```bash
python 1D_simple_model.py
```

This script implements a simplified 1D vegetation-water model for testing, visualization, and learning.

A typical nondimensional Klausmeier-type 1D system has the form:

```text
dw/dt = a - w - w n² + v dw/dx
dn/dt = w n² - m n + d d²n/dx²
```

where:

* `w` is water
* `n` is vegetation biomass
* `a` is rainfall
* `m` is vegetation mortality
* `v` is water advection velocity
* `d` is vegetation diffusion

### 2D Klausmeier Model

Run:

```bash
python 2D_klausmeier.py
```

This script simulates the classical two-variable Klausmeier model in two spatial dimensions.

The nondimensional Klausmeier model describes water and vegetation biomass:

```text
∂w/∂t = a - w - w n² + v ∂w/∂x

∂n/∂t = w n² - m n + ∇²n
```

where:

* `w` is water
* `n` is vegetation biomass
* `a` is rainfall
* `m` is vegetation mortality
* `v` is downhill water advection velocity
* `∇²n` is vegetation diffusion

The script produces a live matplotlib visualization of the simulation.

## Extended Multi-Species Model

Run:

```bash
cd extended_model_implementation/code
python main.py
```

The extended model separates total plant biomass into three vegetation types:

* Grass
* Saplings
* Adult trees

The simulation is configured through:

```text
extended_model_implementation/configs/extended_model.yaml
```

The default configuration includes:

* 1D or 2D mesh settings
* Periodic boundary conditions
* Rainfall and water-advection parameters
* Species-specific growth, mortality, and diffusion parameters
* Sapling-to-tree maturation
* Tree-to-sapling recruitment
* Simulation time step and total runtime
* Output settings for plots, animations, and saved configuration files

## Extended Model Equations

The extended model uses four state variables:

* `w`: water
* `g`: grass
* `s`: saplings
* `b`: adult trees

Let

```text
n = g + s + b
```

be the total vegetation biomass. The nondimensional extended model is:

```text
∂w/∂t = a - w - w(g + s + b)² + v ∂w/∂x

∂g/∂t = b1 w(g + s + b)g - m1 g + d1 ∇²g

∂s/∂t = b2 w(g + s + b)s - m2 s - γs + d2 ∇²s + λb

∂b/∂t = b3 w(g + s + b)b - m3 b + γs
```

The water equation uses total vegetation biomass to model the combined effect of grass, saplings, and trees on water consumption and infiltration.

The vegetation equations represent:

* Grass growth, mortality, and diffusion
* Sapling growth, mortality, maturation into trees, diffusion, and recruitment from adult trees
* Adult tree growth, mortality, and gain from sapling maturation

Adult trees are assumed spatially immobile, so there is no tree diffusion term.

### Parameter Notation

In the project report, the vegetation growth parameters are written as:

```text
j1, j2, j3
```

In the code and configuration files, these same parameters are written as:

```text
b1, b2, b3
```

Thus:

```text
report: j1, j2, j3
code:   b1, b2, b3
```

The parameters are:

* `a`: nondimensional rainfall rate
* `v`: nondimensional downhill water advection speed
* `b1`: grass growth/yield parameter
* `b2`: sapling growth/yield parameter
* `b3`: adult tree growth/yield parameter
* `m1`: grass mortality rate
* `m2`: sapling mortality rate
* `m3`: adult tree mortality rate
* `d1`: grass diffusion coefficient
* `d2`: sapling diffusion coefficient
* `γ`: sapling-to-tree maturation rate
* `λ`: tree-to-sapling recruitment rate

Setting `s = b = 0` reduces the water-grass subsystem to the nondimensional Klausmeier model.

## Configuration

Edit the extended model configuration file:

```text
extended_model_implementation/configs/extended_model.yaml
```

Important sections include:

### `mesh`

Controls spatial dimension, domain size, grid resolution, and boundary conditions.

Example fields:

```yaml
mesh:
  periodic_bc: true
  dim: 2
  Lx: 100.0
  Ly: 100.0
  nCellsX: 100
  nCellsY: 100
```

### `model`

Controls ecological and PDE parameters.

Example fields:

```yaml
model:
  a: 2.0
  v: 182.5
  b1: 1.0
  b2: 1.0
  b3: 1.0
  m1: 0.45
  m2: 0.30
  m3: 0.50
  d1: 1.0
  d2: 0.5
  gamma: 0.15
  lambda: 0.21
  type: extended
```

Depending on the current code version, the tree-to-sapling recruitment parameter may be named `lambda`, `lam`, or another Python-safe variant. Use the name expected by the implementation in `extended_model_implementation/code`.

### `simulation`

Controls runtime, time step, saved output frequency, random seed, equilibrium branch, and perturbation size.

Example fields:

```yaml
simulation:
  totalTime: 100.0
  dt: 0.002
  save_every: 100
  seed: 42
  eq_branch: E_all
  perturbation: 0.001
```

### `output`

Controls whether plots, animations, and configuration copies are saved.

```yaml
output:
  dir: outputs
  save_plot: true
  save_config: true
  save_animation: true
```

## Equilibrium Branches

The extended model supports several spatially homogeneous ecological states:

### `E0`: Barren State

No vegetation is present:

```text
g = s = b = 0
```

Water remains at the rainfall-determined equilibrium.

### `E1+`: Grass-Only State

Grass persists while saplings and adult trees are absent:

```text
g > 0, s = 0, b = 0
```

This branch recovers the Klausmeier-type vegetation state when woody vegetation is excluded.

### `E2`: Sapling-and-Tree State

Grass is absent while saplings and adult trees persist:

```text
g = 0, s > 0, b > 0
```

This represents a woody vegetation state.

### `E3`: Coexistence State

Grass, saplings, and adult trees all persist:

```text
g > 0, s > 0, b > 0
```

This represents a mixed savanna state. In the report analysis, the coexistence state requires a compatibility condition between the sapling-tree parameters, so not all parameter combinations produce a stable coexistence equilibrium.

## Outputs

Extended model runs save results under:

```text
extended_model_implementation/outputs/YYYYMMDD/HHMMSS/
```

Depending on the configuration and mesh dimension, outputs may include:

* 1D space-time diagrams
* 1D or 2D snapshots
* GIF animations
* A copy of the configuration used for the run

## Parameter Sweeps

The extended model includes a parameter sweep script:

```bash
cd extended_model_implementation/code
python sweep_param.py
```

Sweep settings are controlled by:

```text
extended_model_implementation/configs/sweep_cfg.yaml
```

Use the sweep configuration to explore how equilibria and pattern formation change across parameter regimes.

## Notes

* The extended model supports both 1D and 2D simulations.
* Periodic boundary conditions can be controlled from the mesh configuration.
* The `seed` parameter makes random perturbations reproducible.
* Grass and saplings diffuse spatially.
* Adult trees are spatially immobile in the extended model.
* For faster testing, reduce `totalTime`, increase `dt` carefully, reduce grid resolution, or disable animation output.
* For high-resolution 2D simulations, memory usage can increase quickly.

## Citation

This project is based on Klausmeier-type dryland vegetation pattern models and an extended multi-species savanna vegetation model.

Klausmeier, C. A. (1999). Regular and irregular patterns in semiarid vegetation. *Science*, 284(5421), 1826–1828.