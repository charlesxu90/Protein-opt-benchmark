# AlphaVariant Environment Setup

This document describes how to set up the environment for running AlphaVariant.

## Requirements

- Python 3.10
- Mamba/Conda package manager

## Quick Setup

### Option 1: Using the environment.yml file

```bash
cd /path/to/Benchmark/alphavariant
mamba env create -f environment.yml
mamba activate alphavariant-env
```

### Option 2: Manual Installation

```bash
# Create environment
mamba create -n alphavariant-env python=3.10 -y
mamba activate alphavariant-env

# Install conda packages
mamba install -c conda-forge -c bioconda \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn=1.5 \
    numba \
    tensorboardX \
    biopandas \
    biopython \
    bioconda::hmmer \
    jupyterlab \
    ipython \
    ipywidgets

# Install pip packages
pip install \
    easydict \
    attrdict \
    loguru \
    tqdm \
    aaindex \
    omegaconf \
    tensorboard \
    Levenshtein \
    openpyxl
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.10 | Runtime |
| numpy | latest | Numerical computing |
| scipy | latest | Scientific computing |
| pandas | latest | Data manipulation |
| scikit-learn | 1.5 | Machine learning utilities |
| numba | latest | JIT compilation |
| biopython | latest | Biological sequence analysis |
| biopandas | latest | PDB file handling with pandas |
| hmmer | latest | Hidden Markov Models for sequences |
| tensorboardX | latest | TensorBoard logging |
| omegaconf | latest | Configuration management |
| loguru | latest | Logging |
| Levenshtein | latest | Edit distance calculations |
| aaindex | latest | Amino acid index database |

## Environment Location

The environment can be installed either as:
- Named environment: `alphavariant-env`
- Or using the existing environment.yml file

## Activation

```bash
mamba activate alphavariant-env
```

## Verification

After setup, verify the installation:

```bash
python -c "import numpy, scipy, pandas, sklearn; print('Core dependencies OK')"
python -c "import Bio, biopandas; print('Bio dependencies OK')"
python -c "import omegaconf, loguru; print('Config/logging OK')"
```

## Notes

- HMMER is included for profile Hidden Markov Model searches
- BioPandas allows easy manipulation of PDB files as DataFrames
- Numba provides JIT compilation for performance-critical code
- This environment focuses on CPU-based operations; GPU support is optional
- TensorBoard is available for experiment visualization
