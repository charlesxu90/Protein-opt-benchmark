# ftMLDE / MLDE Integration Notes

Repo: `https://github.com/fhalab/MLDE` (cloned to `ftMLDE/`)

The "ft" in ftMLDE refers to *focused training*: instead of random initial
samples, ftMLDE selects an initial training set guided by zero-shot scores
(EVE / DeepSequence / ProteinMPNN, etc.).

## Compatibility

ftMLDE assumes a finite combinatorial design space (fixed sites, all variants
enumerable). Of our datasets, only the CombinGym ones qualify:

- ✅ GB1 (4 sites)
- ✅ PhoQ (4 sites)
- ✅ CR9114 (16 sites)
- ✅ CreiLOV (15 sites)
- ✅ eqFP611 (5 sites)
- ❌ AAV_med, AAV_hard, GFP_med, GFP_hard, ProteinGym substitutions
  (variable-length, not enumerable). Skip ftMLDE for these.

## Setup

```bash
cd ftMLDE
conda env create -f mlde.yml -p ./env
```

Pre-compute zero-shot scores for "focused" training (optional; ftMLDE without
focused training is the same as MLDE — still useful for our benchmark):

```bash
./env/bin/python predict_zero_shot.py --dataset GB1 ...
```

## Per-seed run (TODO)

ftMLDE's `simulate_mlde.py` runs M simulations × K cross-validation folds.
Our launcher runs one seed at a time, so the wrapper should set
`n_simulations=1` and pass our seed through.

```python
from code.run_mlde import run_mlde
results = run_mlde(
    default_models, sim_training_data,
    DESIGN_SPACE, COMBO_TO_IND,
    n_to_average=1,
    train_test_inds=cv_inds,
    progress_pos=None,
    _return_processed=True,
    _reshape_x=True,
    _seeds=[args.seed],
)
```

## Open work

- [ ] Write `build_design_space.py` to convert our `data/<combinatorial>/data.csv`
      to ftMLDE's design-space format (one-hot encoded combos).
- [ ] Adapt `run_mlde` invocation to single-seed iterative use.
- [ ] Parse output to `metrics_seed{seed}.json`.
- [ ] Add per-dataset wrappers for the 5 supported CombinGym datasets.
