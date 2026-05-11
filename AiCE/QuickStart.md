# AiCE Quick Start Guide

**AiCE** (AI-assisted protein evolution) is a high-fitness mutation prediction tool that nominates beneficial mutations by sampling inverse folding protein sequences using ProteinMPNN.

## Prerequisites

### Environment Setup

```bash
# Create mamba environment
mamba env create --prefix ./env -f environment.yml

# Activate environment
conda activate ./env
```

### Required External Tools

1. **plink v1.9** (for Linkage Disequilibrium calculation)
   ```bash
   wget -c https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20241022.zip
   unzip -d scripts/plink/ plink_linux_x86_64_20241022.zip
   ```

2. **mkdssp** (for secondary structure prediction) - included in `scripts/`
   ```bash
   chmod 755 scripts/mkdssp
   ```

## Quick Run

```bash
./run_script.sh
```

This runs the complete pipeline on the example SpCas9 structure.

---

## Core Algorithm Overview

### Pipeline Architecture

```
Input PDB/CIF → ProteinMPNN → MSA → Frequency Analysis → Mutation Filtering
                                ↓
                    LD Matrix ← DNA Conversion
                                ↓
                    SCA Matrix ← Statistical Coupling Analysis
                                ↓
                    Multi-mutation Nomination
```

---

## Step 1: Single Mutation Nomination

**Script:** `scripts/01.single_mut_prediction.sh`

### Core Algorithm

1. **Inverse Folding (ProteinMPNN)**
   - Generates 1000 sequences from protein structure using graph neural network
   - Parameters: `sampling_temp=0.5`, `batch_size=16`

2. **Residue Frequency Counting** (`scripts/count_residue_freq.py`)
   ```python
   # For each position in the MSA (excluding reference sequence):
   for position in range(protein_length):
       for amino_acid in "ACDEFGHIKLMNPQRSTVWY-X":
           counts[amino_acid][position] += 1
       highest_freq_aa[position] = argmax(counts[:, position])
   ```

3. **Secondary Structure Prediction** (`scripts/predict_dssp.py`)
   - Uses DSSP to classify each residue as: Helix (H), Sheet (E), or Coil (C)

4. **Mutation Filtering** (`scripts/predicted_single_HF_mutations.py`)
   ```python
   # Filter by frequency threshold based on secondary structure
   for pos, ref_aa, alt_aa, freq, ss in merged_data:
       if ref_aa != alt_aa:  # Must be a mutation
           if ss == 'C':     # Flexible coil region
               threshold = gamma  # default: 0.5
           else:             # Structured region (helix/sheet)
               threshold = beta   # default: 0.8

           if freq >= threshold:
               nominated_mutations.append((pos, ref_aa, alt_aa))
   ```

### Auto-threshold Prediction

AiCE can automatically predict optimal thresholds using pre-trained models:

```python
# Features: protein_size, flex_ratio (fraction of coil residues)
features = [protein_size, coil_count / protein_size]
features_scaled = scaler.transform(features)

beta = model_a.predict(features_scaled)  # Global threshold
gamma = model_b.predict(features_scaled) # Coil threshold
```

---

## Step 2: Linkage Disequilibrium (LD) Matrix

**Script:** `scripts/02.caculated_ld.py`

### Core Algorithm

1. **Protein → DNA Conversion**
   ```python
   # Convert amino acids to optimal codons
   OPTIMAL_CODON = {
       'A': 'GCC', 'C': 'TGC', 'D': 'GAC', ...
   }
   for aa in protein_sequence:
       dna_sequence += OPTIMAL_CODON[aa]
   ```

2. **Generate VCF Format**
   - First sequence = reference
   - Identify variants at each position across all sequences

3. **Calculate LD Matrix using plink**
   ```bash
   plink --vcf input.vcf --r square --out output
   ```

---

## Step 3: Statistical Coupling Analysis (SCA)

**Script:** `scripts/03.caculated_sca.sh`

### Core Algorithm

SCA identifies evolutionarily coupled positions using pySCA:

1. **MSA Processing** (`pySCA/scaProcessMSA.py`)
   - Remove non-standard residues
   - Trim gaps and low-quality sequences

2. **SCA Core Calculation** (`pySCA/scaCore.py`)
   - Computes positional conservation weights
   - Calculates coupling scores between position pairs
   - Performs 10 randomization trials for statistical significance

Output: `.sca_matrix.tsv` with pairwise coupling scores

---

## Step 4: Multi-Mutation Nomination

**Script:** `scripts/04.com_mut_prediction.sh`

### Core Algorithm (`scripts/com_mut_prediction.py`)

1. **Generate Mutation Combinations**
   ```python
   from itertools import combinations
   all_combos = list(combinations(nominated_positions, n))  # n = 2 for doubles
   ```

2. **Score Each Combination**
   ```python
   def calculate_mean_ld(ld_matrix, positions):
       """Calculate mean pairwise LD score for a set of positions"""
       submatrix = ld_matrix[np.ix_(positions, positions)]
       upper_tri = submatrix[np.triu_indices(n, 1)]
       return np.mean(upper_tri)
   ```

3. **Apply Thresholds**
   ```python
   # SCA: percentile-based (default: 90th percentile = top 10%)
   sca_threshold = pick_percentile_value(sorted_sca_scores, percentile=0.9)

   # LD: numeric threshold (default: 0.5)
   ld_threshold = 0.5

   # Flag recommended combinations
   if mean_score >= threshold:
       recommendation_flag = 1
   ```

---

## Output Files

| File | Description |
|------|-------------|
| `*.fa` | Generated sequences from ProteinMPNN |
| `*.ss` | Secondary structure predictions (H/E/C) |
| `*.freq` | Raw amino acid frequency counts |
| `*.mut` | Filtered high-fitness single mutations |
| `*.comb` | Combined data for all positions |
| `*.ld` | Linkage disequilibrium matrix |
| `*.vcf` | Variant call format file |
| `*.sca_matrix.tsv` | SCA coupling matrix |
| `*.sca.result` | Multi-mutation SCA scores |
| `*.ld.result` | Multi-mutation LD scores |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.8 | Frequency threshold for structured regions |
| `gamma` | 0.5 | Frequency threshold for flexible (coil) regions |
| `num_seq_per_target` | 1000 | Sequences generated by ProteinMPNN |
| `sampling_temp` | 0.5 | ProteinMPNN sampling temperature |
| `LD threshold` | 0.5 | Cutoff for LD-based multi-mutation selection |
| `SCA percentile` | 0.9 | Percentile for SCA-based selection (top 10%) |

---

## Example Usage

### Basic Pipeline
```bash
# Step 1: Single mutations
cd example/
bash ../scripts/01.single_mut_prediction.sh ../scripts ./ 0.8 0.5 ../output

# Step 2: LD matrix
python ../scripts/02.caculated_ld.py ../output/ ../output

# Step 3: SCA matrix (slow - ~1 hour for large proteins)
bash ../scripts/03.caculated_sca.sh ../scripts/pySCA/ ../output ../output

# Step 4: Multi-mutations
bash ../scripts/04.com_mut_prediction.sh ../scripts ../output/ 2 ../output
```

### Auto-threshold Mode
```bash
bash scripts/01.single_mut_Auto_prediction.sh ./example ../output
```

### Custom Mutation Combinations
```bash
# Create a file with specific position combinations (1-based indexing)
echo "4 56 789" > my_positions.txt
echo "10 20 30" >> my_positions.txt

# Run with custom positions
python scripts/com_mut_prediction.py -i output/protein.sca_matrix.tsv \
    -l my_positions.txt -o output/custom.result
```

---

## References

- Paper: "Advancing protein evolution with inverse folding models integrating structural and evolutionary constraints" - Cell (2025)
- ProteinMPNN: [github.com/dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- pySCA: Statistical Coupling Analysis toolkit
