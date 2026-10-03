# interactive-metadynamics

Seoul National University (Fall 2026) Biophysics: Toy MD Simulation

본 리포지토리는 서울대학교 2026-2학기 생물물리학 수업의 MD 시뮬레이션 실습 및 개념 학습을 돕기 위해 제작되었습니다. 교육 목적의 toy model로 구성되어 있어 실제 분자 동역학 시뮬레이션과는 차이가 있을 수 있습니다.

This repository is designed to help students grasp the core concepts of molecular dynamics (MD) simulations in the SNU Biophysics course (Fall 2026). It contains simplified toy models for educational purposes only and may differ from production-grade MD simulations.

Hands-on Jupyter notebooks exploring metadynamics and free energy landscape reconstruction

## Quick Start

The scientific code depends only on `numpy`, `scipy` and `matplotlib`. Work through the notebooks in order:
`00 → 01 → 02 → 03 → 04 → 05 → 06`.

| # | Notebook | Colab | What it demonstrates |
|---|---|---|---|
| 00 | [00_md_fundamentals](notebooks/00_md_fundamentals.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/00_md_fundamentals.ipynb) | Euler, Leapfrog and Velocity Verlet on a harmonic oscillator; Maxwell–Boltzmann velocities, COM removal, kinetic temperature, initial rescaling |
| 01 | [01_thermostats_and_barostats](notebooks/01_thermostats_and_barostats.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/01_thermostats_and_barostats.ipynb) | 2D periodic Lennard-Jones fluid: NVE vs. velocity rescaling vs. Langevin (BAOAB); Berendsen barostat |
| 02 | [02_metastability_and_brownian_dynamics](notebooks/02_metastability_and_brownian_dynamics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/02_metastability_and_brownian_dynamics.ipynb) | asymmetric double well, overdamped Langevin dynamics, finite-time trapping |
| 03 | [03_well_tempered_metadynamics](notebooks/03_well_tempered_metadynamics.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/03_well_tempered_metadynamics.ipynb) | grid-based WT-MetaD, hill attenuation, free-energy reconstruction, animation |
| 04 | [04_umbrella_sampling_and_wham](notebooks/04_umbrella_sampling_and_wham.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/04_umbrella_sampling_and_wham.ipynb) | harmonic windows, overlap diagnostics, iterative WHAM, poor vs. reasonable overlap |
| 05 | [05_accelerated_and_gaussian_accelerated_md](notebooks/05_accelerated_and_gaussian_accelerated_md.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/05_accelerated_and_gaussian_accelerated_md.ipynb) | aMD boost and exponential reweighting; GaMD parameter rules, boost statistics, cumulant reweighting |
| 06 | [06_replica_exchange_and_method_comparison](notebooks/06_replica_exchange_and_method_comparison.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HyeonjeYang/interactive-metadynamics/blob/main/notebooks/06_replica_exchange_and_method_comparison.ipynb) | T-REMD and H-REMD, ladder spacing, method comparison against the exact 1D profile |

### Google Colab

Click a badge above. The first code cell of each notebook clones this repository so that `src/` can be imported.

### Local Jupyter

```bash
git clone https://github.com/HyeonjeYang/interactive-metadynamics.git
cd interactive-metadynamics
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scriptsctivate
python -m pip install -r requirements.txt
python -m pip install jupyterlab    # Jupyter itself is not a scientific dependency
jupyter lab
```

Optional checks: `python -m pip install pytest` and then `python -m pytest`.

### Layout

- `src/` — small NumPy modules: potentials, integrators, thermal initialization, LJ fluid, Brownian dynamics, WT-MetaD, umbrella sampling, WHAM, aMD/GaMD, replica exchange, analysis and plotting helpers
- `notebooks/` — the seven tutorials (stored without outputs; every simulation uses a fixed seed)
- `tests/` — physical and numerical checks (gradients, energy conservation, exchange criteria, WHAM, …)
- `figures/` — key figures written by the notebooks

All systems are toy models in reduced units ($k_B = 1$). They show how each method works; they do not predict how a method performs on biomolecules.
