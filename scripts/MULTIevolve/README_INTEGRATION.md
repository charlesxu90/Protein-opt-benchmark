# MULTI-evolve Integration Notes

Repo: `https://github.com/ArcInstitute/MULTI-evolve` (cloned to `MULTIevolve/`)

## Setup

```bash
cd MULTIevolve
conda env create -f env.yml -p ./env
./env/bin/pip install -e .
```

WandB account required by default. To disable for benchmark runs, set the
`WANDB_MODE=offline` environment variable in `scripts/hpc/env_setup.sh` for
this method.

## Pipeline mapping

MULTI-evolve has three upstream scripts:
  - `p1_train.py` — train NN ensemble on single mutants
  - `p2_propose.py` — score and propose combinatorial variants
  - `p3_assembly_design.py` — wet-lab oligo design (skip for benchmark)

For our iterative benchmark (5 rounds × 96 variants):

- Round 0: query 96 random variants from the landscape; treat them as the
  single-mutant training set.
- Each subsequent round: re-train ensemble on accumulated data (p1), score all
  unqueried variants (p2), select top-96 by predicted fitness, query oracle.

The upstream `p1_train.py` uses CLI args; the wrapper should call into
`multievolve.predictors.train_predictor` directly.

## Per-seed run (TODO)

```python
from multievolve.predictors.train_predictor import train_predictor
from multievolve.proposers.proposers import VariantProposer

predictor = train_predictor(
    training_data=our_collected_variants,
    wt_seq=oracle.wildtype,
    seed=args.seed,
    mode="test",
)
proposals = VariantProposer(predictor).propose_top_k(
    candidate_pool=oracle.landscape.sequences,
    k=args.batch_size,
)
```

## Open work

- [ ] Verify `train_predictor` accepts seed for reproducibility
- [ ] Adapt to our `OracleHandle` interface
- [ ] Implement iterative loop (re-train each round)
- [ ] Parse output to `metrics_seed{seed}.json`
- [ ] Add per-dataset wrappers
- [ ] Verify WandB-offline mode produces identical results to online
