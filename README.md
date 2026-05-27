# PIFNO-LAW
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![Paper](https://img.shields.io/badge/Paper-AMM-blue)](https://doi.org/10.1016/j.apm.2026.117079)


This repository contains the official implementation for the paper:

**Learned Adaptive Weighting for Physics-Informed Fourier Neural Operators:
Solving Discontinuous PDEs with Limited Data**

PIFNO-LAW is a physics-informed Fourier neural operator framework for solving
partial differential equations with discontinuous solutions in limited-data
settings. The method uses a dual-operator design: a solution operator predicts
the PDE solution, while an auxiliary Fourier neural operator learns a dynamic
spatial weight field for the physics loss. This learned adaptive weighting helps
stabilize training near shocks and other discontinuities, where large localized
residuals can otherwise dominate gradients and degrade the learned operator.

The codebase includes training and evaluation utilities for the discontinuous
PDE benchmarks used in the paper, including configurations for PIFNO-LAW,
unweighted physics-informed FNO baselines, static heuristic weighting baselines,
and data-driven FNO variants.




## Quick Start

We recommend using [uv](https://github.com/astral-sh/uv) (a fast Python package manager), but you can also use `pip` or `conda`.

### 1. Install Dependencies

#### Option A: Using uv
```bash
# Clone and enter the repository
git clone https://github.com/Gxinhu/PIFNO-LAW.git
cd PIFNO-LAW

# Install all dependencies and create a virtual environment (.venv)
uv sync
```

#### Option B: Using pip
```bash
git clone https://github.com/Gxinhu/PIFNO-LAW.git
cd PIFNO-LAW

# We recommend using a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project and dependencies
pip install -e .
```

#### Option C: Using conda
```bash
git clone https://github.com/Gxinhu/PIFNO-LAW.git
cd PIFNO-LAW

# Create and activate a conda environment
conda create -n pifno python=3.11
conda activate pifno

# Install the project and dependencies
pip install -e .
```

### 2. Configure the Dataset Directory

This project requires a `.env` file to define the system path to your dataset folders (so `Hydra` configurations can dynamically mount the data).

Create your `.env` file using the provided example:

```bash
# Copy the example template
cp .env.example .env
```

Open the newly created `.env` file and set the `DATASET_PATH` variable to point to the `data/` folder you will be creating in the next step:
```env
# Open .env and ensure it has this line:
DATASET_PATH="./data/"
```

### 3. Download the Datasets

The simulation data is hosted on Hugging Face at [huxinchn/PIFNO-LAW](https://huggingface.co/datasets/huxinchn/PIFNO-LAW).

Create a `data/` directory in the repository root and download the dataset from Hugging Face into it:

```bash
mkdir -p data/
cd data/

# Download the 1D Burgers dataset using the Hugging Face CLI:
uv run huggingface-cli download huxinchn/PIFNO-LAW --repo-type dataset --local-dir ./
cd ..
```

After downloading, your root directory structure should cleanly look like this:
```text
PIFNO-LAW/
├── configs/
├── data/
│   └── burgers.h5
├── script/
├── src/
└── pyproject.toml
```

### 4. Run Training Scripts

The application relies on `Hydra` for hierarchical configurations. You can run all foundational experiments and baselines using the provided bash script, which queues multiple network layouts iteratively over different data sample sizes:

```bash
# Make sure to run this from the project root!
bash script/experiment.sh
```

Alternatively, you can test a single specific experiment using your environment's python runner:

```bash
# If using uv (Recommended)
uv run src/train.py -m experiment=pino_shock_law data.n_train=50 seed=1 logger=csv

# If using pip/conda
python src/train.py -m experiment=pino_shock_law data.n_train=50 seed=1 logger=csv
```

## References
[PINO_Application](https://github.com/shawnrosofsky/PINO_Applications.git)

## Citation

If this repository is useful for your research, please cite our paper:

```bibtex
@article{HU2026117079,
  title = {Learned Adaptive Weighting for Physics-Informed Fourier Neural Operators: Solving Discontinuous PDEs with Limited Data},
  journal = {Applied Mathematical Modelling},
  pages = {117079},
  year = {2026},
  issn = {0307-904X},
  doi = {https://doi.org/10.1016/j.apm.2026.117079},
  url = {https://www.sciencedirect.com/science/article/pii/S0307904X26003409},
  author = {Xin Hu and Bo An and Yongke Guan and Liang Xu and Min Yu and Dong Li},
}
```
