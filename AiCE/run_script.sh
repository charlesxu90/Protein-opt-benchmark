#!/bin/bash
# =============================================================================
# AiCE Demo Pipeline - AI-assisted protein evolution
# =============================================================================
# This script runs the complete AiCE mutation prediction pipeline on the
# SpCas9 example structure.
#
# Pipeline Steps:
# 1. Single mutation nomination using inverse folding (ProteinMPNN)
# 2. Linkage Disequilibrium (LD) matrix calculation
# 3. Statistical Coupling Analysis (SCA) matrix calculation
# 4. Multi-mutation nomination combining SCA and LD scores
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
EXAMPLE_DIR="${SCRIPT_DIR}/example"
OUTPUT_DIR="${SCRIPT_DIR}/output"
SCRIPTS_DIR="${SCRIPT_DIR}/scripts"

# Thresholds for single mutation filtering
BETA=0.8    # Global occurrence threshold (recommended: 0.8)
GAMMA=0.5   # Flexible region (coil) threshold (recommended: 0.5)

# Multi-mutation parameters
NUM_MUTATIONS=2  # Number of mutations per combination (e.g., 2 for double mutations)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to print steps
print_step() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}STEP $1: $2${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check environment
if [ ! -d "$ENV_DIR" ]; then
    echo -e "${RED}[ERROR]${NC} Environment not found at $ENV_DIR"
    echo "Please run: mamba env create --prefix ./env -f environment.yml"
    exit 1
fi

# Activate the environment
print_info "Activating AiCE environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# =============================================================================
# STEP 1: Single Mutation Nomination
# =============================================================================
print_step 1 "Single Mutation Nomination"
print_info "Running inverse folding with ProteinMPNN..."
print_info "Input: ${EXAMPLE_DIR}/*.pdb"
print_info "Parameters: beta=$BETA, gamma=$GAMMA"

cd "$EXAMPLE_DIR"
bash "${SCRIPTS_DIR}/01.single_mut_prediction.sh" \
    "$SCRIPTS_DIR" \
    "$EXAMPLE_DIR" \
    "$BETA" \
    "$GAMMA" \
    "$OUTPUT_DIR"

print_info "Single mutation prediction complete!"
print_info "Output files:"
echo "  - ${OUTPUT_DIR}/*.fa   (Generated sequences)"
echo "  - ${OUTPUT_DIR}/*.ss   (Secondary structure)"
echo "  - ${OUTPUT_DIR}/*.freq (Residue frequencies)"
echo "  - ${OUTPUT_DIR}/*.mut  (High-fitness mutations)"
echo "  - ${OUTPUT_DIR}/*.comb (Combined data)"

# =============================================================================
# STEP 2: Linkage Disequilibrium (LD) Matrix Calculation
# =============================================================================
print_step 2 "Linkage Disequilibrium (LD) Matrix Calculation"
print_info "Calculating LD matrix using plink..."

cd "$SCRIPT_DIR"
python "${SCRIPTS_DIR}/02.caculated_ld.py" "$OUTPUT_DIR/" "$OUTPUT_DIR"

print_info "LD matrix calculation complete!"
print_info "Output files:"
echo "  - ${OUTPUT_DIR}/*.ld  (LD matrix)"
echo "  - ${OUTPUT_DIR}/*.vcf (VCF file)"

# =============================================================================
# STEP 3: Statistical Coupling Analysis (SCA) Matrix
# =============================================================================
print_step 3 "Statistical Coupling Analysis (SCA) Matrix Calculation"
print_info "This step may take significant time (~1 hour for large proteins)..."
print_info "Running SCA analysis with pySCA..."

cd "$SCRIPT_DIR"
bash "${SCRIPTS_DIR}/03.caculated_sca.sh" \
    "${SCRIPTS_DIR}/pySCA/" \
    "$OUTPUT_DIR" \
    "$OUTPUT_DIR"

print_info "SCA matrix calculation complete!"
print_info "Output files:"
echo "  - ${OUTPUT_DIR}/*.db           (SCA database)"
echo "  - ${OUTPUT_DIR}/*.sca_matrix.tsv (SCA matrix)"

# =============================================================================
# STEP 4: Multi-Mutation Nomination
# =============================================================================
print_step 4 "Multi-Mutation Nomination"
print_info "Generating mutation combinations and scoring..."
print_info "Number of mutations per combination: $NUM_MUTATIONS"

cd "$SCRIPT_DIR"
bash "${SCRIPTS_DIR}/04.com_mut_prediction.sh" \
    "$SCRIPTS_DIR" \
    "$OUTPUT_DIR/" \
    "$NUM_MUTATIONS" \
    "$OUTPUT_DIR"

print_info "Multi-mutation nomination complete!"
print_info "Output files:"
echo "  - ${OUTPUT_DIR}/*.sca.result (SCA-based recommendations)"
echo "  - ${OUTPUT_DIR}/*.ld.result  (LD-based recommendations)"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}===================================================${NC}"
echo ""
echo "Results are in: ${OUTPUT_DIR}"
echo ""
echo "Key output files:"
ls -la "${OUTPUT_DIR}"/*.mut 2>/dev/null || echo "  No .mut files found"
ls -la "${OUTPUT_DIR}"/*.sca.result 2>/dev/null || echo "  No .sca.result files found"
ls -la "${OUTPUT_DIR}"/*.ld.result 2>/dev/null || echo "  No .ld.result files found"
echo ""
echo "To view single mutations: cat ${OUTPUT_DIR}/<protein>.mut"
echo "To view multi-mutations:  cat ${OUTPUT_DIR}/<protein>.sca.result"
