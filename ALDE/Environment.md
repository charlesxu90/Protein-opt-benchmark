# ALDE Environment Setup

This document describes how to set up the environment for running ALDE (Active Learning for Directed Evolution).

## Requirements

- Python 3.11
- CUDA 12.1 (for GPU support)
- Mamba/Conda package manager

## Quick Setup

### Create environment from scratch

```bash
# Create a prefix environment in the ALDE directory
mamba create -p ./env python=3.11 -y

# Activate the environment
mamba activate ./env

# Install conda packages
mamba install -c pytorch -c nvidia -c conda-forge \
    numpy=1.26.4 \
    scipy \
    pandas=1.5.3 \
    matplotlib \
    networkx=3.4 \
    pytorch=2.1.1 \
    pytorch-cuda=12.1 \
    scikit-learn \
    statsmodels \
    seaborn \
    logomaker \
    openpyxl \
    ipykernel \
    tqdm

# Install pip packages
pip install botorch==0.9.4 gpytorch==1.11 loguru xgboost
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.11 | Runtime |
| pytorch | 2.1.1 | Deep learning framework |
| pytorch-cuda | 12.1 | CUDA support |
| numpy | 1.26.4 | Numerical computing |
| scipy | latest | Scientific computing |
| pandas | 1.5.3 | Data manipulation |
| botorch | 0.9.4 | Bayesian optimization |
| gpytorch | 1.11 | Gaussian processes |
| scikit-learn | latest | Machine learning utilities |
| statsmodels | latest | Statistical modeling |
| xgboost | latest | Gradient boosting |
| logomaker | 0.8.6 | Sequence logo visualization |
| seaborn | latest | Statistical visualization |

## Environment Location

The environment is installed as a prefix environment at:
```
/home/xux/Desktop/AlphaVariant/Benchmark/ALDE/env
```

## Activation

```bash
mamba activate /home/xux/Desktop/AlphaVariant/Benchmark/ALDE/env
```

## Verification

After setup, verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import botorch, gpytorch; print('Bayesian optimization OK')"
python -c "import numpy, scipy, pandas, sklearn; print('All dependencies OK')"
```

## Notes

- Uses BoTorch for Bayesian optimization with Gaussian processes
- GPU with CUDA 12.1 support is recommended for optimal performance
- The environment includes tools for sequence logo generation (logomaker)
