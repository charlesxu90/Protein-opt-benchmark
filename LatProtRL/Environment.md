# LatProtRL Environment Setup

This document describes how to set up the environment for running LatProtRL (Latent Protein Reinforcement Learning).

## Requirements

- Python 3.9
- CUDA 11.8 (for GPU support)
- Mamba/Conda package manager

## Quick Setup

### Create environment from scratch

```bash
# Create the environment directory structure
mkdir -p env/latprotrl_env

# Create a prefix environment
mamba create -p ./env/latprotrl_env python=3.9 -y

# Activate the environment
mamba activate ./env/latprotrl_env

# Install PyTorch with CUDA 11.8 support
pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install pip packages
pip install \
    fair-esm==2.0.0 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    stable-baselines3==2.2.1 \
    wandb==0.16.1 \
    tqdm==4.66.1 \
    plotly==5.18.0 \
    kaleido==0.2.1 \
    rich==14.2.0
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.9 | Runtime |
| pytorch | 2.1.2+cu118 | Deep learning framework |
| fair-esm | 2.0.0 | ESM protein language models |
| stable-baselines3 | 2.2.1 | Reinforcement learning algorithms |
| pandas | 2.1.4 | Data manipulation |
| scikit-learn | 1.3.2 | Machine learning utilities |
| wandb | 0.16.1 | Experiment tracking |
| plotly | 5.18.0 | Interactive visualization |
| kaleido | 0.2.1 | Static image export for Plotly |
| rich | latest | Terminal formatting |
| tqdm | 4.66.1 | Progress bars |

## Environment Location

The environment is installed as a prefix environment at:
```
/home/xux/Desktop/AlphaVariant/Benchmark/LatProtRL/env/latprotrl_env
```

## Activation

```bash
mamba activate /home/xux/Desktop/AlphaVariant/Benchmark/LatProtRL/env/latprotrl_env
```

## Verification

After setup, verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import esm; print('ESM OK')"
python -c "import stable_baselines3; print('Stable Baselines3 OK')"
python -c "import pandas, sklearn, wandb; print('All dependencies OK')"
```

## Notes

- Uses ESM (Evolutionary Scale Modeling) for protein representations
- Stable Baselines3 provides reinforcement learning algorithms (PPO, A2C, etc.)
- Weights & Biases (wandb) is used for experiment tracking
- GPU with CUDA 11.8 support is required for ESM model inference
- Python 3.9 is required for compatibility with fair-esm
