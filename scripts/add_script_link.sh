#!/bin/bash
# Script to create symbolic links from repositories to centralized scripts

# Get the directory where this script is located (scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Get the parent directory (Benchmark/)
REPO_BASE="$(dirname "$SCRIPT_DIR")"

# Use absolute paths derived from script location
SCRIPT_BASE="$SCRIPT_DIR"

# Function to create links for a method
create_links() {
    local script_dir="$1"
    local repo_dir="$2"

    for script in "$script_dir"/*.py "$script_dir"/*.sh; do
        [ -e "$script" ] || continue
        filename=$(basename "$script")
        ln -sf "$script" "$repo_dir/$filename"
        echo "Linked: $script -> $repo_dir/$filename"
    done
}

# Create links for each method
create_links "$SCRIPT_BASE/AiCE" "$REPO_BASE/AiCE"
create_links "$SCRIPT_BASE/ALDE" "$REPO_BASE/ALDE"
create_links "$SCRIPT_BASE/alphavariant" "$REPO_BASE/alphavariant"
create_links "$SCRIPT_BASE/delta_cs/BioSeq-GFN-AL" "$REPO_BASE/delta_cs/BioSeq-GFN-AL"
create_links "$SCRIPT_BASE/EvoPlay" "$REPO_BASE/EvoPlay"
create_links "$SCRIPT_BASE/FLEXS" "$REPO_BASE/FLEXS"
create_links "$SCRIPT_BASE/LatProtRL" "$REPO_BASE/LatProtRL"
create_links "$SCRIPT_BASE/Random" "$REPO_BASE/Random"
create_links "$SCRIPT_BASE/GreedyWalk" "$REPO_BASE/GreedyWalk"

# Phase 2.1 baselines (scaffolding; see scripts/<method>/README_INTEGRATION.md)
mkdir -p "$REPO_BASE/EVOLVEpro" "$REPO_BASE/ftMLDE" "$REPO_BASE/MULTIevolve" "$REPO_BASE/Mu-Protein"
create_links "$SCRIPT_BASE/EVOLVEpro" "$REPO_BASE/EVOLVEpro"
create_links "$SCRIPT_BASE/ftMLDE" "$REPO_BASE/ftMLDE"
create_links "$SCRIPT_BASE/MULTIevolve" "$REPO_BASE/MULTIevolve"
create_links "$SCRIPT_BASE/Mu-Protein" "$REPO_BASE/Mu-Protein"
create_links "$SCRIPT_BASE/CLADE" "$REPO_BASE/CLADE"

# Handle AiCE nested scripts/ directory
mkdir -p "$REPO_BASE/AiCE/scripts"
create_links "$SCRIPT_BASE/AiCE/scripts" "$REPO_BASE/AiCE/scripts"

# delta_cs scripts live in delta_cs/BioSeq-GFN-AL/ — the launcher reads
# `run_subdir: BioSeq-GFN-AL` from method_resources.yaml and finds them
# there directly, so we don't mirror them at the top level (doing so breaks
# Python's __file__-relative imports because the script computes
# `os.path.dirname(__file__) + '/../..'` to find the benchmark utils).

echo "All symbolic links created successfully!"
