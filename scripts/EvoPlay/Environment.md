# EvoPlay Environment Setup

This document describes how to set up the environment for running EvoPlay (Evolutionary Algorithm for Protein Design).

## Requirements

- Python 3.8
- CUDA 11.8 (for GPU support)
- Mamba/Conda package manager

## Quick Setup

### Create environment from scratch

```bash
# Create a prefix environment in the EvoPlay directory
mamba create -p ./env python=3.8 -y

# Activate the environment
mamba activate ./env

# Install conda packages
mamba install -c bioconda -c conda-forge \
    numpy=1.24.4 \
    scipy=1.10.1 \
    matplotlib=3.7.3 \
    biopython=1.83 \
    scikit-learn=1.3.2 \
    hhsuite=3.3.0 \
    kalign3 \
    openmm=7.5.1 \
    pdbfixer \
    cudatoolkit=11.8 \
    jupyterlab \
    ipykernel \
    seaborn \
    logomaker

# Install PyTorch with CUDA 11.8 support
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional pip packages
pip install jax==0.3.13 jaxlib==0.3.10+cuda11.cudnn82 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install dm-haiku==0.0.4 dm-tree==0.1.8 chex==0.0.7 ml-collections==0.1.0
pip install tape-proteins==0.5 openpyxl py3Dmol seaborn logomaker
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.8 | Runtime |
| pytorch | 2.4.1+cu118 | Deep learning framework |
| jax | 0.3.13 | Numerical computing (for structure prediction) |
| jaxlib | 0.3.10+cuda11 | JAX CUDA support |
| numpy | 1.24.4 | Numerical computing |
| scipy | 1.10.1 | Scientific computing |
| biopython | 1.83 | Biological sequence analysis |
| hhsuite | 3.3.0 | Homology search (HHblits) |
| kalign3 | 3.2.2 | Multiple sequence alignment |
| openmm | 7.5.1 | Molecular dynamics |
| pdbfixer | 1.7 | PDB file fixing |
| dm-haiku | 0.0.4 | Neural network library (JAX) |
| tape-proteins | 0.5 | Protein representation learning |

## Environment Location

The environment is installed as a prefix environment at:
```
/home/xux/Desktop/AlphaVariant/Benchmark/EvoPlay/env
```

## Activation

```bash
mamba activate /home/xux/Desktop/AlphaVariant/Benchmark/EvoPlay/env
```

## Verification

After setup, verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import jax; print(f'JAX: {jax.__version__}')"
python -c "import numpy, scipy, Bio; print('All dependencies OK')"
```

## Notes

- Uses both PyTorch and JAX for different components
- HHsuite (HHblits) is required for sequence homology search
- OpenMM is included for molecular dynamics simulations
- CUDA 11.8 is required for GPU acceleration
- Python 3.8 is required for compatibility with older JAX versions
