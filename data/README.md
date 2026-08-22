# Benchmark Datasets

Six datasets, split into the two benchmark panels. Every dataset lives at
`data/<name>/data.csv` with a **`seq`** column (not `sequence`) plus a
`fitness` column; most also carry `AACombo` and `n_muts`.

This directory is **git-ignored** — datasets are not distributed with the
repo. See [Re-generation](#re-generation) below.

## Panels

**Four-site, true landscape** (`4site_*`) — combinatorially-complete 20⁴
libraries. Methods select from the measured pool; `fitness` is the measured
value, so the landscape is exact and no oracle is involved.

**Multi-site, learned oracle** (`ms_*`) — a `BaseCNN` oracle trained on the
full table (`oracles/<name>/oracle.pt`) is the fitness function, and methods
generate over the varying positions rather than selecting from a pool.

## Summary

| Directory | n | seq len | Columns | Sites / varying positions | Fitness range |
|---|---:|---:|---|---|---:|
| `4site_GB1` | 149,361 | 56 | `seq, fitness, n_muts, AACombo` | V39, D40, G41, V54 | 0 – 8.761966 |
| `4site_PhoQ` | 140,517 | 486 | `seq, fitness, log_fitness, mutations, n_muts, AACombo` | A284, V285, S288, T289 | 0 – 133.594270 |
| `4site_TRPB` | 159,129 | 397 | `seq, fitness, AACombo` | V183, F184, S227, S228 | 0 – 1.0 |
| `ms_AAV` | 44,128 | 28 | `seq, fitness, AACombo, n_muts` | all 28 positions | 0 – 1.0 |
| `ms_CreiLOV` | 165,428 | 119 | `seq, fitness, AACombo, n_muts` | 15 of 119 | 620.276 – 15,686.305 |
| `ms_PAB1` | 36,522 | 75 | `seq, fitness, log_fitness, mutations, n_muts, AACombo` | all 75 positions | 0.002352 – 2.627943 |

All sequences within a dataset are fixed-length (`seq len` above is both the
min and max, and equals the `wt.fasta` length). `n_muts` ranges: GB1 0–4,
PhoQ 1–4, AAV 0–28, CreiLOV 0–15, PAB1 1–4. `4site_TRPB` ships no `n_muts`.

**Global maxima** — used to normalise raw `max_fitness` in the four-site panel:

| Dataset | max | combo |
|---|---:|---|
| `4site_GB1` | 8.761966 | `FWAA` |
| `4site_PhoQ` | 133.594270 | `TEMH` |
| `4site_TRPB` | 1.0 (already normalised) | `AIKG` |

Multi-site oracle outputs are normalised to [0,1] at training time;
`fit_min` / `fit_max` for de-normalisation live in
`oracles/<name>/oracle_meta.json` (CreiLOV 620.276–15686.305,
PAB1 0.002352–2.627943, AAV 0.0–1.0, i.e. already normalised).

## Per-dataset files

Beyond `data.csv` and `wt.fasta`, each directory carries the artifacts the
methods that use it need. `AACombo` is the design-space key: the 4-char combo
for the four-site panel, and the full varying-position string for `ms_*`
(28-char for AAV, 15-char for CreiLOV, 75-char for PAB1).

| File | Datasets | Consumed by |
|---|---|---|
| `data.csv` | all 6 | everything, via `utils/data.py` `FitnessLandscape` |
| `wt.fasta` | all 6 | all methods; also `EVPredictor` for the offset |
| `mutcompute.csv` | all 6 | AlphaVariant `--use_mutcompute` (`run_generic.py:1278`) |
| `<id>.pdb` | all 6 (`4site_TRPB` has two: `8VHH.pdb`, `8VHH-single.pdb`) | input to ProteinMPNN / MutCompute generation (not read at run time) |
| `embeddings_evolvepro.pt` + `.meta.json` + `labels_evolvepro.csv` | `4site_*` only | `scripts/EVOLVEpro/run_generic.py` |
| `varying_positions.txt` | `ms_*` only | `utils/candidate_generator.py`, `run_generic.py:588` |
| `aice_mpnn_freq.npz` | `ms_*` only | AiCE, via `scripts/run_oracle_benchmark.py:230` |
| `plmc/uniref100.model_params` (+ `.EC`) | `ms_*` only | `EVPredictor` under AlphaVariant `--features ev_onehot` |
| `prior_aligned.csv` | `ms_*` only | `scripts/alphavariant/train_ms_prior.py --aligned_csv` |
| `target_seqs.fasta` | `ms_*` only | `scripts/alphavariant/align_homologs.py` (regeneration only) |

`mutcompute.csv` is one row per residue of the PDB chain (GB1 56, CreiLOV 109,
PAB1 577, AAV 510 — the last two exceed the design length because the structure
covers a larger construct), with `pos`, `wtAA`, `pred_prob` and 20 `pr<AA3>`
columns. `aice_mpnn_freq.npz` holds `freq` (seq_len × 20), `covered`, `alphabet`,
`offset`, `identity`.

`varying_positions.txt` is a single comma-separated line of **0-indexed**
positions. For `ms_AAV` and `ms_PAB1` it lists every position (28 of 28,
75 of 75), so `AACombo == seq` and the design is full-length; only `ms_CreiLOV`
is a genuine subset — 0-indexed 2, 3, 4, 6, 28, 33, 46, 59, 60, 91, 95, 97,
106, 108, 112, which against the shipped `wt.fasta` reads
G3, L4, D5, S7, A29, T34, Q47, R60, Q61, C92, D96, D98, V107, V109, E113
(1-indexed). The `ms_CreiLOV` WT combo is therefore `GLDSATQRQCDDVVE`.

> Earlier revisions of this file labelled these sites G3, L4, R5, T7, A29, G34,
> Q47, R60, D61, I92, D96, R98, V107, V109, T113. That set does **not** match
> the shipped `wt.fasta` — 7 of 15 residues differ. The shipped files are
> internally consistent (`wt.fasta` is exactly the `n_muts == 0` row, and
> `n_muts` equals the Hamming distance to it at the varying positions), so the
> residues above are the ones the benchmark actually uses.

### Provenance

- **4site_GB1** — Protein G B1 domain IgG-binding. Near-complete 4-site scan at V39/D40/G41/V54 in the 56-aa domain. Wu *et al.* 2016, *eLife* 5:e16965, via CombinGym `GB1_clean.xlsx`.
- **4site_PhoQ** — *E. coli* PhoQ histidine kinase, sites A284/V285/S288/T289 in the 486-aa protein, FACS-seq signalling competence. Podgornaia & Laub 2015, *Science* 347:673.
- **4site_TRPB** — *T. maritima* Tm9D8\* TrpB, sites V183/F184/S227/S228 in the 397-aa enzyme, indole-condensation activity min-max normalised to [0,1]. Johnston *et al.* 2024, via ALDE `data/TrpB/fitness.csv`.
- **ms_AAV** — AAV VP1 capsid 28-mer insertion region; packaging score, min-max normalised. Bryant *et al.* 2021, *Nat. Biotechnol.* 39:691 (ProteinGym).
- **ms_CreiLOV** — *C. reinhardtii* FMN-based fluorescent protein, 15-site library over the 119-aa protein, ~90 % of 184,320 possible variants. `fitness` is raw mean fluorescence. Chen *et al.* 2023, *ACS Synth. Biol.* 12:1461, via CombinGym `CreiLOV_clean.xlsx`.
- **ms_PAB1** — Yeast poly(A)-binding protein RRM2 domain (75 aa), single- and multi-mutant DMS. `mutations` is HGVS-style, `log_fitness` mirrors `fitness` on a log scale. Melamed *et al.* 2013, *RNA* 19:1537 (ProteinGym).

## Re-generation

Neither prepare script currently emits the `4site_*` / `ms_*` directory names —
both write `data/<legacy_short_name>/data.csv`, so a rename is needed after:

```bash
# CombinGym: clones https://github.com/sitonglab/CombinGym into /tmp/combingym_cache/
python scripts/prepare_combingym.py --datasets GB1 CreiLOV
mv data/GB1 data/4site_GB1 && mv data/CreiLOV data/ms_CreiLOV

# ProteinGym
python scripts/prepare_proteingym.py --datasets PABP_YEAST
mv data/PABP_YEAST data/ms_PAB1
```

`prepare_combingym.py` also still exposes `CR9114`, `eqFP611`, `mTagBFP2`,
`SpCas9`, `SaCas9`, `RhlA`, `CR6261_H1/H9/CR6261`, and
`prepare_proteingym.py` 16 assays including `GFP_AEQVI`; none of these are in
the current panels. Pass `--local-repo` to `prepare_combingym.py` to reuse an
existing clone.

**Not covered by either script:** `4site_PhoQ`, `4site_TRPB`, `ms_AAV`.
`4site_TRPB` is built by embedding the 4-char `Combo` from
`ALDE/data/TrpB/fitness.csv` into the 397-aa wild type at 1-indexed positions
183, 184, 227, 228 (note the parent WT residues at those positions differ from
V/F/S/S — a different reference numbering than canonical Pf/Tm TrpB).

### Derived artifacts

```bash
python scripts/train_oracle.py --dataset ms_CreiLOV --smoke          # cheap check first
python scripts/train_oracle.py --dataset ms_CreiLOV --device cuda:0  # -> oracles/<ds>/oracle.pt
CUDA_VISIBLE_DEVICES=0 AiCE/env/bin/python scripts/compute_mpnn_freqs.py --dataset ms_CreiLOV
python scripts/EVOLVEpro/embed_dataset.py --dataset 4site_GB1
python scripts/alphavariant/align_homologs.py --dataset ms_CreiLOV     # -> prior_aligned.csv
python scripts/alphavariant/train_ms_prior.py --dataset ms_CreiLOV --device cuda:0
```

The `plmc/` couplings models are built out-of-band with
`alphavariant/fitness_model/ev_onehot/scripts/plmc.sh`, which needs an
`align/alignment.a2m` MSA (jackhmmer/EVcouplings output). Those MSA
directories are **not shipped** — the derived `plmc/` and `prior_aligned.csv`
are, so nothing in the benchmark needs them at run time.

### Checksums

```bash
python scripts/compute_dataset_checksums.py            # write data/CHECKSUMS.txt
python scripts/compute_dataset_checksums.py --verify   # or: sha256sum -c data/CHECKSUMS.txt
```

`CHECKSUMS.txt` covers the 6 `data.csv` files only — not `wt.fasta`, the
embeddings, `plmc/`, or any other artifact. Paths inside it are relative to the
benchmark root, so run `sha256sum -c` from there, not from `data/`.

Note `data/` is git-ignored, so `CHECKSUMS.txt` is not tracked either; it has to
travel with the datasets or be committed somewhere outside `data/` to be useful
to a reviewer.
