# EVOLVEpro Integration Notes

Repo: `https://github.com/mat10d/EVOLVEpro` (cloned to `EVOLVEpro/`)

EVOLVEpro consumes pre-computed PLM embeddings and labels, then runs few-shot
active learning. Their `evolvepro.src.evolve.grid_search` is the main entry.

## Setup (one-time per machine)

```bash
cd EVOLVEpro
conda env create -f environment.yml -p ./env
conda env create -f plm_environment.yml -p ./plm_env  # only for embedding step
./env/bin/pip install -e .
```

## Per-dataset embedding step (one-time per dataset)

```bash
# Activate the PLM env (separate from the active-learning env)
conda activate ./plm_env

# Embed every variant in our landscape under ESM-2
python scripts/EVOLVEpro/embed_dataset.py --dataset GB1 --model esm2_650M
# Writes data/GB1/embeddings_esm2_650M.pt (~700MB for GB1)
```

`scripts/EVOLVEpro/embed_dataset.py` is **not yet written**. Stub plan:

1. Load `data/<dataset>/data.csv` (`seq, fitness`).
2. Use `evolvepro/plm/esm/extract.py` from upstream to produce per-variant embeddings.
3. Save tensor + matching labels CSV in EVOLVEpro's expected format.

## Per-seed run

Once embeddings exist, the wrapper should:

```python
from evolvepro.src.evolve import grid_search
results = grid_search(
    dataset_name=args.dataset,
    embeddings_path=embedding_path,
    labels_path=labels_path,
    num_simulations=1,
    num_iterations=[args.n_rounds],
    num_mutants_per_round=[args.batch_size],
    measured_var=["activity"],
    learning_strategies=["topn"],
    seed=args.seed,
)
# Parse `results` into our queried_indices + write metrics_seed*.json
```

## Open work

- [ ] Write `embed_dataset.py` for one-time per-dataset embedding
- [ ] Implement `grid_search` → metrics adapter in `run_generic.py`
- [ ] Write per-dataset wrappers (run_GB1.py, etc.) once generic works
- [ ] Add `EVOLVEpro/env` build to method_resources.yaml verification
