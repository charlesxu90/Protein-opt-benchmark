# Mu-Protein adapter — integration status

Tracking integration of the **Mu-Protein** baseline (µFormer fitness model +
µSearch RL search; Microsoft Research; <https://github.com/microsoft/Mu-Protein>)
into the 4-site combinatorial benchmark.

## Why this is non-trivial

Unlike the other baselines (EvoPlay, EVOLVEpro, MULTI-evolve), Mu-Protein
requires:

1. **A pretrained PMLM-650M checkpoint** (2.6 GB) saved by fairseq in a
   format that only loads under torch ≥ 1.6 (new ZIP-based serialization).
2. **fairseq 0.10.2** (Microsoft's pinned version) with custom task /
   model / criterion registrations from `Mu-Protein/mu-former/src/`.
3. **Sufficient labelled data to finetune µFormer**. The paper finetunes
   the decoder on hundreds-to-thousands of single-mutant fitness values.
   Our benchmark budget is 96 examples after round 1 (and 192/288/384
   after rounds 2-4). 96-shot µFormer finetuning is **out of distribution
   for the paper** and the released hyper-parameters are not designed
   for it.

## Upstream env state (as built by the user)

`Mu-Protein/env/` (Python 3.8.20) has:

| Package        | Version       | Status                                  |
|----------------|---------------|-----------------------------------------|
| torch          | 1.4.0         | **Too old** to load the released PMLM   |
| tape-proteins  | 0.4           | Compiled against torch 1.4 — fragile    |
| pyrosetta      | 2026.19       | Compiled against torch 1.4 — fragile    |
| biopython      | 1.78          | OK                                      |
| numpy          | 1.18.5        | OK                                      |
| scipy          | 1.5.2         | OK                                      |
| fairseq        | NOT installed | Need 0.10.2                             |

Upgrading torch in-place will break tape and pyrosetta. The non-destructive
path is a **fresh `muformer-env`** with torch 1.12 + fairseq 0.10.2.

The build is launched in the background; log at
`sweep_logs/_muformer_env_build.out`.

## What an honest Mu-Protein adapter would look like

```text
Round 1: random 96  (no labels yet)
Rounds 2-5:
  1. Build training set from cumulative (combo, fitness) pairs (96, 192, 288, 384 examples).
  2. Initialise µFormer from pretrained PMLM-650M (frozen encoder, trainable Siamese decoder).
  3. Finetune decoder for K epochs on the cumulative training set.
     ⚠ Hyperparameters (K, learning rate, batch size) are NOT defined for this regime
        in the upstream paper; need to pick defensible values.
  4. Score all uncollected variants with finetuned µFormer.
  5. Query top-96 by predicted fitness.
```

**Open design questions before adapter is defensible for publication:**

- Do we freeze the PMLM encoder (paper does for low-data) or finetune (paper does for high-data)?
- What learning rate, batch size, and epoch count for 96-384 training examples?
- Do we use the Siamese decoder (paper default) or a simpler regression head?
- How do we handle the PMLM-650M memory cost (16 GB on A100 with batch ≥ 16)?

Without resolving these, any "Mu-Protein" number we report would be
adapter-specific and not directly comparable to the paper.

## Recommended next steps (in order)

1. Wait for `muformer-env` build to finish (~20 min).
2. Write a smoke test that loads PMLM-650M and runs forward pass on one
   variant — verify checkpoint loads cleanly under torch 1.12.
3. Decide on the freeze/finetune regime (recommend freeze + Ridge head
   on PMLM features for 96-shot regime — defensible as "PMLM features
   via Mu-Protein checkpoint + active learning").
4. Implement `scripts/Mu-Protein/embed_dataset.py` (extracts PMLM
   features for all 140k variants — likely ~1 hour per dataset on A100).
5. Implement `scripts/Mu-Protein/run_generic.py` (Ridge on PMLM features,
   same iterative loop as EVOLVEpro).
6. Per-dataset wrappers + symlinks.
7. Run PhoQ × 5 seeds.

**Estimated total effort:** ~4-6 hours engineering + ~5 hours compute for
PhoQ embeddings + sweep.

## Why this isn't done yet (honest deferral)

The user explicitly directed "implement Mu-Protein if other tasks are
running" — meaning don't block on it. Other PhoQ × 5 sweeps
(EvoPlay, EVOLVEpro, MULTIevolve) are in progress. The env build is
the lightest cheap step that can run in parallel; the adapter work is
deferred until the env is ready and the design questions above are
resolved with the user.
