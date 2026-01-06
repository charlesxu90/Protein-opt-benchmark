# FLEXS (AdaLead) Environment Setup

This document describes how to set up the environment for running FLEXS/AdaLead (Adaptive Leading strand algorithm).

## Requirements

- Python 3.7
- Mamba/Conda package manager

## Quick Setup

### Create environment from scratch

```bash
# Create a prefix environment in the FLEXS directory
mamba create -p ./env python=3.7 -y

# Activate the environment
mamba activate ./env

# Install conda packages
mamba install -c bioconda -c conda-forge \
    viennarna=2.5.1 \
    perl

# Install pip packages
pip install lmdb==1.4.1
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.7 | Runtime |
| viennarna | 2.5.1 | RNA secondary structure prediction |
| perl | 5.32 | Required for some ViennaRNA tools |
| lmdb | 1.4.1 | Lightning Memory-Mapped Database |

## Environment Location

The environment is installed as a prefix environment at:
```
/home/xux/Desktop/AlphaVariant/Benchmark/FLEXS/env
```

## Activation

```bash
mamba activate /home/xux/Desktop/AlphaVariant/Benchmark/FLEXS/env
```

## Verification

After setup, verify the installation:

```bash
python -c "import lmdb; print('LMDB OK')"
RNAfold --version  # Should show ViennaRNA version
```

## Notes

- This is a lightweight environment focused on the FLEXS/AdaLead algorithm
- ViennaRNA is included for RNA secondary structure calculations
- Python 3.7 is required for compatibility with the FLEXS library
- This environment does not require GPU support
- For additional FLEXS functionality, you may need to install the `flexs` package separately
