
## 🚀 Quick Start

We recommend using [uv](https://github.com/astral-sh/uv) (a fast Python package manager), but you can also use `pip` or `conda`.

#### Option A: Using uv (Recommended)
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