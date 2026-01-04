# delta_cs (BioSeq-GFN-AL) Environment Setup

This document describes how to set up the environment for running delta-Conservative Search (BioSeq-GFN-AL).

## Requirements

- Python 3.7
- CUDA 11.6 (for GPU support)
- Mamba/Conda package manager

## Quick Setup

### Create environment from scratch

```bash
# Create the environment directory structure
mkdir -p env/delta_cs_env

# Create a prefix environment
mamba create -p ./env/delta_cs_env python=3.7 -y

# Activate the environment
mamba activate ./env/delta_cs_env

# Install conda packages
mamba install -c bioconda -c conda-forge -c pytorch \
    numpy=1.21.6 \
    pillow=9.2.0 \
    pytorch=1.12.0 \
    torchvision=0.13.0 \
    torchaudio=0.12.0 \
    cudatoolkit=11.6 \
    viennarna=2.5.1 \
    mkl

# Install pip packages
pip install \
    polyleven==0.8 \
    transformers==4.30.2 \
    tokenizers==0.13.3 \
    sentencepiece==0.2.0 \
    sequence-models==1.2.0 \
    wandb \
    tqdm \
    flexs==0.2.8 \
    safetensors==0.3.3
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python | 3.7 | Runtime |
| pytorch | 1.12.0 | Deep learning framework |
| cudatoolkit | 11.6 | CUDA support |
| numpy | 1.21.6 | Numerical computing |
| transformers | 4.30.2 | Hugging Face Transformers |
| tokenizers | 0.13.3 | Fast tokenization |
| sequence-models | 1.2.0 | Protein sequence models |
| polyleven | 0.8 | Fast Levenshtein distance |
| flexs | 0.2.8 | Fitness landscape exploration |
| viennarna | 2.5.1 | RNA secondary structure |
| wandb | latest | Experiment tracking |
| sentencepiece | 0.2.0 | Subword tokenization |

## Environment Location

The environment is installed as a prefix environment at:
```
/home/xux/Desktop/AlphaVariant/Benchmark/delta_cs/env/delta_cs_env
```

## Activation

```bash
mamba activate /home/xux/Desktop/AlphaVariant/Benchmark/delta_cs/env/delta_cs_env
```

## Verification

After setup, verify the installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import polyleven; print('Polyleven OK')"
python -c "import flexs; print('FLEXS OK')"
```

## Notes

- Uses Hugging Face Transformers for protein language models
- FLEXS library provides fitness landscape exploration utilities
- Polyleven is used for fast edit distance calculations
- ViennaRNA is included for RNA secondary structure prediction
- Python 3.7 is required for compatibility with older package versions
- GPU with CUDA 11.6 support is recommended
