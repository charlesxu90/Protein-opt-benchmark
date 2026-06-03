#!/bin/bash
#SBATCH --job-name=av_gfp_test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
#SBATCH --constraint="a100"
#SBATCH --time=03:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-%j.out
#SBATCH --error=log-%x-%j.out

# SINGLE-SEED GFP run of AlphaVariant (Plan C) — doubles as (a) a SLURM environment /
# data / model readiness check (fail-fast pre-flight) and (b) a real estimate of whether
# AlphaVariant works on GFP + its per-seed wall time. Uses the EXACT final-benchmark
# config (500 steps), so the produced seed counts as a real benchmark seed: it writes to
# the real results_oracle/ and the full sweep will reuse it (idempotent skip), no recompute.

module purge
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/xux/miniconda3/envs/prot-gen-env

# ============================ CONFIG (edit) ============================
BENCH="${BENCH:-/home/xux/Desktop/AlphaVariant/Benchmark}"   # benchmark root on the cluster
DATASET="ms_GFP"
SEED="${SEED:-621}"
N_ROUNDS="${N_ROUNDS:-5}"
N_STEPS="${N_STEPS:-500}"       # final-benchmark config (GFP single seed ~110 min)
SIGMA="${SIGMA:-60}"
# =======================================================================

set -uo pipefail
echo "================ AlphaVariant GFP smoke test ================"
echo "host=$(hostname)  date=$(date)  BENCH=$BENCH"
echo "SEED=$SEED  N_ROUNDS=$N_ROUNDS  N_STEPS=$N_STEPS  SIGMA=$SIGMA"

fail() { echo "PREFLIGHT FAIL: $1" >&2; exit 1; }

# ---- 1. GPU visible ----
echo "--- GPU ---"; nvidia-smi -L || fail "nvidia-smi not available / no GPU"

# ---- 2. conda env + python imports ----
echo "--- python / imports ---"
which python
python - <<'PY' || fail "python import check failed (missing deps in prot-gen-env)"
import importlib, torch
print("python torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
for m in ["sklearn", "scipy", "numpy", "pandas", "loguru", "popgen", "popscorer"]:
    importlib.import_module(m); print("  import OK:", m)
PY

# ---- 3. staged files ----
echo "--- staged files ---"
for f in \
    "$BENCH/oracles/$DATASET/oracle.pt" \
    "$BENCH/alphavariant/priors/$DATASET/prior_model.pt" \
    "$BENCH/alphavariant/priors/$DATASET/prior_model.json" \
    "$BENCH/data/$DATASET/data.csv" \
    "$BENCH/data/$DATASET/wt.fasta" \
    "$BENCH/data/$DATASET/mutcompute.csv" \
    "$BENCH/utils/oracle_landscape.py" \
    "$BENCH/scripts/run_oracle_benchmark.py" ; do
    [ -f "$f" ] && echo "  OK  $f" || fail "missing $f"
done

# ---- 4. run one seed ----
# Final-benchmark config -> write to the real results_oracle/. The full sweep is
# idempotent and will SKIP this seed (no recompute). If you ever lower N_STEPS for a
# quick-only check, point --output_path at a throwaway dir instead so it isn't reused.
echo "--- running 1 GFP seed (Plan C oracle, final config) ---"
cd "$BENCH/alphavariant"
RES="$BENCH/results_oracle"
PRIOR="--prior_model_path priors/$DATASET/prior_model.pt"
python run_generic.py --dataset "$DATASET" --seed "$SEED" \
    --oracle --level uniform $PRIOR \
    --use_mutcompute --shap_prune_alphabet \
    --n_rounds "$N_ROUNDS" --n_steps_per_round "$N_STEPS" --sigma "$SIGMA" \
    --device cuda:0 \
    --output_path "$RES" --data_dir "$BENCH/data" || fail "run_generic.py exited non-zero"

# ---- 5. confirm output ----
OUT="$RES/$DATASET/AlphaVariant/seed${SEED}.json"
[ -f "$OUT" ] || fail "no result JSON written ($OUT)"
echo "--- result ---"
python - "$OUT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["metrics"]
print("  max_fitness_norm =", round(m["max_fitness_norm"], 4))
print("  top128_mean_norm =", round(m["top128_mean_norm"], 4))
PY
echo "================ SMOKE TEST PASSED ================"
