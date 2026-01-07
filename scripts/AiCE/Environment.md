# AiCE Environment Setup

This document describes how to set up the environment for running AiCE (AI-driven Combinatorial Evolution).

## Requirements

- Python 3.11
- CUDA 12.1 (for GPU support)
- Mamba/Conda package manager

## Quick Setup

### Option 1: Using the environment.yml file

```bash
cd /path/to/Benchmark/AiCE
mamba env create -f environment.yml
mamba activate AiCE
```

### Option 2: Manual Installation

```bash
# Create environment
mamba create -n AiCE python=3.11 -y
mamba activate AiCE

# Install conda packages
mamba install -c pytorch -c nvidia -c conda-forge \
    numpy=1.23.5 \
    scipy=1.12.0 \
    pandas=2.2.3 \
    matplotlib=3.10.0 \
    networkx=3.2.1 \
    biopython=1.79 \
    "joblib>=1.1.0" \
    pytorch=2.2.1 \
    pytorch-cuda=12.1 \
    dssp

# Install pip packages
pip install prody==2.4.1 ml-collections==0.1.1 dm-tree==0.1.8
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.11 | Runtime |
| pytorch | 2.2.1 | Deep learning framework |
| pytorch-cuda | 12.1 | CUDA support |
| numpy | 1.23.5 | Numerical computing |
| scipy | 1.12.0 | Scientific computing |
| pandas | 2.2.3 | Data manipulation |
| biopython | 1.79 | Biological sequence analysis |
| prody | 2.4.1 | Protein dynamics analysis |
| networkx | 3.2.1 | Graph operations |
| dssp | - | Secondary structure prediction |
| ml-collections | 0.1.1 | Configuration management |
| dm-tree | 0.1.8 | Tree data structures |

## Verification

After setup, verify the installation:

```bash
mamba activate AiCE
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import numpy, scipy, pandas, Bio, prody; print('All dependencies OK')"
```

## Notes

- DSSP is required for secondary structure calculations
- GPU with CUDA 12.1 support is recommended for optimal performance
- The environment uses PyTorch 2.2.1 with CUDA 12.1 bindings
