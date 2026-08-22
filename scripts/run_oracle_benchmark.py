#!/usr/bin/env python
"""
run_oracle_benchmark.py - Benchmark baseline methods against the learned ms_* oracles.

Pure-oracle, generative-proposal benchmark (see project memory multisite-oracle-benchmark):
all methods share one round loop and differ only in their candidate *selection rule*.

    Round 0:  random_init(96) -> oracle                       [96 calls]
    Rounds 1..4:
        train surrogate on all queried (varying-position one-hot -> oracle fitness)
        candidate pool = mutate(top elites, POOL)  (local search)
        score pool with surrogate + method selection rule -> top 96
        oracle(top 96)                                          [96 calls/round]
    Budget = 96 x 5 = 480 oracle calls.

Methods:
    Random      : no surrogate; 96 fresh random_init samples each round.
    GreedyWalk  : no surrogate; oracle-evaluate 96 single-mutant neighbors of best-so-far.
    ftMLDE      : zoo of regressors, CV-pick top-3, average -> top 96 by predicted mean.
    CLADE       : ensemble predict + MiniBatchKMeans diversity (best-per-cluster).
    ALDE        : RandomForest ensemble, Thompson sampling (mean + std*N(0,1)).

Usage:
    python scripts/run_oracle_benchmark.py --method ALDE --dataset ms_CreiLOV --seed 42
    python scripts/run_oracle_benchmark.py --method Random --dataset ms_CreiLOV --seeds 621 100 383
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.oracle_landscape import OracleLandscape
from utils.candidate_generator import CandidateGenerator
from utils.oracle_model import ALPHABET

warnings.filterwarnings("ignore")
AA_IDX = {a: i for i, a in enumerate(ALPHABET)}
METHODS = ["Random", "GreedyWalk", "ftMLDE", "CLADE", "ALDE",
           "AdaLead", "MULTIevolve", "EVOLVEpro", "AiCE"]


# ------------------------------------------------------------------ encoding
def encode_varying(seqs, var_positions):
    """One-hot of varying positions only -> (n, len(var)*20)."""
    n, L = len(seqs), len(var_positions)
    X = np.zeros((n, L * 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for k, p in enumerate(var_positions):
            X[i, k * 20 + AA_IDX.get(s[p], 0)] = 1.0
    return X


# ------------------------------------------------------------------ surrogates
def fit_ftmlde(Xtr, ytr):
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.model_selection import cross_val_score
    zoo = {
        "ridge": Ridge(alpha=1.0),
        "knn": KNeighborsRegressor(n_neighbors=min(5, len(Xtr))),
        "rf": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0),
        "et": ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=0),
    }
    scores = {}
    cv = min(3, max(2, len(Xtr) // 32))
    for name, m in zoo.items():
        try:
            scores[name] = cross_val_score(m, Xtr, ytr, cv=cv,
                                            scoring="r2", n_jobs=-1).mean()
        except Exception:
            scores[name] = -1e9
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    models = [zoo[n].fit(Xtr, ytr) for n in top3]
    return lambda X: np.mean([m.predict(X) for m in models], axis=0)


def fit_clade(Xtr, ytr):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    models = [
        Ridge(alpha=1.0).fit(Xtr, ytr),
        RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(Xtr, ytr),
        KNeighborsRegressor(n_neighbors=min(5, len(Xtr))).fit(Xtr, ytr),
    ]
    return lambda X: np.mean([m.predict(X) for m in models], axis=0)


def fit_alde(Xtr, ytr):
    """RandomForest; mean + per-tree std for Thompson sampling."""
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(Xtr, ytr)

    def predict_mean_std(X):
        per_tree = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
        return per_tree.mean(0), per_tree.std(0)
    return predict_mean_std


def fit_multievolve(Xtr, ytr, n_models=3, epochs=120, hidden=(256, 64), seed=0):
    """MULTI-evolve `Fcn`-style bootstrap ensemble of small fully-connected nets.

    Mirrors MULTI-evolve (Tran et al.) which trains an NN ensemble on the
    collected (variant, fitness) labels and predicts uncollected variants.
    Kept on CPU + tiny so it does not contend with the GPU oracle.
    """
    import torch
    import torch.nn as nn
    Xt = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
    yt = torch.tensor(np.asarray(ytr), dtype=torch.float32)
    in_dim = Xt.shape[1]
    nets = []
    for k in range(n_models):
        torch.manual_seed(seed + k)
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        net = nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        idx = np.random.RandomState(seed + k).choice(len(Xt), len(Xt), replace=True)
        xb, yb = Xt[idx], yt[idx]
        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((net(xb).squeeze(-1) - yb) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        nets.append(net)

    def predict(X):
        Xq = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            return np.mean([n(Xq).squeeze(-1).numpy() for n in nets], axis=0)
    return predict


def _recombine(a, b, rng):
    return "".join(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))


def adalead_propose(seqs, fn, gen, var, rng, batch, pool_size,
                    threshold=0.05, m_step=2):
    """AdaLead-style proposal (FLEXS): recombine high-fitness parents + mutate,
    surrogate-guided refinement, then take top-`batch` by the surrogate.

    Surrogate = MULTI-evolve-free regressor ensemble (CLADE-style avg) trained on
    the queried data; faithful to AdaLead's "model-guided evolutionary search".
    """
    fmax = fn.max()
    order = np.argsort(-fn)
    parents = [seqs[i] for i in order if fn[i] >= fmax * (1.0 - threshold)]
    if len(parents) < 2:
        parents = [seqs[i] for i in order[:max(2, batch // 4)]]
    surrogate = fit_clade(encode_varying(seqs, var), fn)

    def make(parent_pool, n):
        out = set()
        tries = 0
        while len(out) < n and tries < n * 40:
            tries += 1
            a = parent_pool[rng.randint(len(parent_pool))]
            if len(parent_pool) > 1 and rng.random() < 0.2:
                a = _recombine(a, parent_pool[rng.randint(len(parent_pool))], rng)
            out.add(gen.mutate([a], 1, rng, m_step=m_step)[0])
        return list(out)

    # wave 1 from elite parents, refine to new parents, wave 2 (rollout-like)
    pool1 = make(parents, pool_size // 2)
    p1 = surrogate(encode_varying(pool1, var))
    refined = [pool1[i] for i in np.argsort(-p1)[:max(8, batch // 4)]]
    pool2 = make(parents + refined, pool_size - len(pool1))
    cands = list(dict.fromkeys(pool1 + pool2))
    preds = surrogate(encode_varying(cands, var))
    return [cands[i] for i in np.argsort(-preds)[:batch]]


def make_esm_embedder(device="cuda:0", model_name="facebook/esm2_t12_35M_UR50D",
                      batch=64):
    """Lazy ESM-2 mean-pool embedder with a per-sequence cache (EVOLVEpro)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    cache = {}

    def embed(seqs):
        todo = [s for s in dict.fromkeys(seqs) if s not in cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                rep = model(**enc).last_hidden_state.mean(1).cpu().numpy()
            for s, e in zip(chunk, rep):
                cache[s] = e.astype(np.float32)
        return np.stack([cache[s] for s in seqs])
    return embed


# ------------------------------------------------------------------ AiCE
def _per_position_freq(seqs, seq_len, eps=1e-3):
    """Per-position AA frequency over a set of sequences -> (L, 20)."""
    f = np.full((seq_len, 20), eps, dtype=float)
    for s in seqs:
        for p in range(seq_len):
            j = AA_IDX.get(s[p])
            if j is not None:
                f[p, j] += 1.0
    return f / f.sum(1, keepdims=True)


def aice_combined_freq(dataset, data_dir, seqs_top, seq_len, w_obs):
    """AiCE frequency: ProteinMPNN structure freq blended with observed top-variant
    freq. Uncovered positions fall back to the observed frequency."""
    obs = _per_position_freq(seqs_top, seq_len) if seqs_top else \
        np.full((seq_len, 20), 1.0 / 20)
    path = os.path.join(data_dir, dataset, "aice_mpnn_freq.npz")
    if not os.path.exists(path):
        return obs  # no structure -> observed only
    d = np.load(path)
    mp, cov = d["freq"], d["covered"]
    comb = obs.copy()
    for p in range(seq_len):
        if cov[p]:
            comb[p] = (1.0 - w_obs) * mp[p] + w_obs * obs[p]
    comb = np.clip(comb, 1e-6, None)
    return comb / comb.sum(1, keepdims=True)


def aice_propose(seqs, fn, gen, var, dataset, data_dir, rng, batch, pool_size,
                 n_elites, m_step, round_frac):
    """Structure-guided AiCE proposal: sample mutations at varying positions from
    the (MPNN + observed) frequency, score by summed log-frequency, take top-batch."""
    seq_len = gen.seq_len
    top = [seqs[i] for i in np.argsort(-fn)[:n_elites]]
    w_obs = min(0.7, 0.2 + 0.5 * round_frac)  # lean on observations as rounds progress
    comb = aice_combined_freq(dataset, data_dir, top, seq_len, w_obs)
    logf = np.log(comb)
    aas = np.array(list(ALPHABET))

    def sample_aa(p):
        return aas[rng.choice(20, p=comb[p])]

    cands, seen = [], set()
    tries = 0
    while len(cands) < pool_size and tries < pool_size * 40:
        tries += 1
        parent = list(top[rng.randint(len(top))])
        for p in rng.choice(var, size=min(m_step, len(var)), replace=False):
            parent[int(p)] = sample_aa(int(p))
        c = "".join(parent)
        if c not in seen:
            seen.add(c)
            cands.append(c)
    score = np.array([sum(logf[p, AA_IDX[c[p]]] for p in var) for c in cands])
    return [cands[i] for i in np.argsort(-score)[:batch]]


# ------------------------------------------------------------------ selection
def select_batch(method, pool, surrogate, batch, rng):
    """Return indices into `pool` chosen by the method's rule."""
    if method in ("ftMLDE", "MULTIevolve"):
        preds = surrogate(pool)
        return np.argsort(-preds)[:batch]
    if method == "ALDE":
        mean, std = surrogate(pool)
        acq = mean + std * rng.randn(len(mean))   # Thompson sample
        return np.argsort(-acq)[:batch]
    if method == "CLADE":
        from sklearn.cluster import MiniBatchKMeans
        preds = surrogate(pool)
        k = min(batch, len(pool))
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3).fit(pool)
        labels = km.labels_
        chosen = []
        for c in range(k):
            idx = np.where(labels == c)[0]
            if len(idx):
                chosen.append(idx[np.argmax(preds[idx])])
        chosen = np.array(chosen)
        if len(chosen) < batch:  # fill with global top among unchosen
            rest = np.setdiff1d(np.argsort(-preds), chosen, assume_unique=False)
            chosen = np.concatenate([chosen, rest[:batch - len(chosen)]])
        return chosen[:batch]
    raise ValueError(method)


# ------------------------------------------------------------------ metrics
def hamming_varying(a, b, var):
    return sum(a[p] != b[p] for p in var)


def compute_metrics(seqs, fit_norm, fit_raw, wt, var, batch, n_rounds):
    order = np.argsort(-fit_norm)
    top = order[:128]
    top_seqs = [seqs[i] for i in top]
    # max-so-far per round
    traj = [float(fit_norm[:batch * r].max()) for r in range(1, n_rounds + 1)]
    # diversity: mean pairwise hamming over varying positions among top-128
    div = 0.0
    if len(top_seqs) > 1:
        ds = [hamming_varying(top_seqs[i], top_seqs[j], var)
              for i in range(len(top_seqs)) for j in range(i + 1, len(top_seqs))]
        div = float(np.mean(ds))
    nov = float(np.mean([hamming_varying(s, wt, var) for s in top_seqs]))
    best_i = int(order[0])
    return {
        "max_fitness_norm": float(fit_norm.max()),
        "max_fitness_raw": float(fit_raw.max()),
        "top128_mean_norm": float(fit_norm[top].mean()),
        "mean_all_norm": float(fit_norm.mean()),
        "diversity_top128": div,
        "novelty_top128_vs_wt": nov,
        "best_n_muts": hamming_varying(seqs[best_i], wt, var),
        "best_sequence": seqs[best_i],
        "fitness_trajectory": traj,
    }


# ------------------------------------------------------------------ main loop
def run_one(method, dataset, seed, device, batch=96, n_rounds=5,
            pool_size=8000, m_step=2, n_elites=24, out_dir=None):
    ls = OracleLandscape(dataset, device=device)
    gen = CandidateGenerator(dataset, alphabet_mode="full")
    var = gen.varying_positions
    wt = gen.wt
    rng = np.random.RandomState(seed)
    t0 = datetime.now()
    esm_embed = None  # lazily created for EVOLVEpro
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    seqs, fnorm = [], []

    def query(batch_seqs):
        f = ls.get_fitness_normalized(batch_seqs)
        seqs.extend(batch_seqs)
        fnorm.extend(f.tolist())

    # Round 0
    query(gen.random_init(batch, rng))

    for r in range(1, n_rounds):
        fn = np.array(fnorm)
        if method == "Random":
            query(gen.random_init(batch, rng))
            continue
        if method == "GreedyWalk":
            best = seqs[int(np.argmax(fn))]
            nbr = gen.neighbors(best)
            rng.shuffle(nbr)
            seen = set(seqs)
            nbr = [s for s in nbr if s not in seen][:batch]
            if len(nbr) < batch:
                nbr += gen.mutate([best], batch - len(nbr), rng, m_step=1)
            query(nbr)
            continue
        if method == "AdaLead":
            query(adalead_propose(seqs, fn, gen, var, rng, batch, pool_size,
                                  m_step=m_step))
            continue
        if method == "AiCE":
            query(aice_propose(seqs, fn, gen, var, dataset, data_dir, rng, batch,
                               pool_size, n_elites, m_step,
                               round_frac=r / max(1, n_rounds - 1)))
            continue
        if method == "EVOLVEpro":
            if esm_embed is None:
                esm_embed = make_esm_embedder(device=device)
            elites = [seqs[i] for i in np.argsort(-fn)[:n_elites]]
            pool = gen.mutate(elites, min(pool_size, 2000), rng, m_step=m_step)
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                       random_state=0).fit(esm_embed(seqs), fn)
            preds = rf.predict(esm_embed(pool))
            query([pool[i] for i in np.argsort(-preds)[:batch]])
            continue
        # surrogate-pool methods (generic mutate pool + predictor)
        elites = [seqs[i] for i in np.argsort(-fn)[:n_elites]]
        pool = gen.mutate(elites, pool_size, rng, m_step=m_step)
        Xtr = encode_varying(seqs, var)
        Xpool = encode_varying(pool, var)
        if method == "ftMLDE":
            surrogate = fit_ftmlde(Xtr, fn)
        elif method == "CLADE":
            surrogate = fit_clade(Xtr, fn)
        elif method == "ALDE":
            surrogate = fit_alde(Xtr, fn)
        elif method == "MULTIevolve":
            surrogate = fit_multievolve(Xtr, fn)
        idx = select_batch(method, Xpool, surrogate, batch, rng)
        query([pool[i] for i in idx])

    fnorm = np.array(fnorm)
    fraw = fnorm * ls.scale + ls.fit_min
    metrics = compute_metrics(seqs, fnorm, fraw, wt, var, batch, n_rounds)
    runtime = (datetime.now() - t0).total_seconds()

    result = {
        "method": method, "dataset": dataset, "seed": seed,
        "n_queries": len(seqs), "oracle_calls": ls.n_calls,
        "oracle_test_spearman": ls.test_spearman,
        "runtime_seconds": runtime, "metrics": metrics,
    }
    if out_dir:
        sub = os.path.join(out_dir, dataset, method)
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, f"seed{seed}.json"), "w") as f:
            json.dump(result, f, indent=2)
    print(f"  [{dataset}/{method}/seed{seed}] max_norm={metrics['max_fitness_norm']:.4f} "
          f"top128={metrics['top128_mean_norm']:.4f} best_muts={metrics['best_n_muts']} "
          f"div={metrics['diversity_top128']:.2f} {runtime:.1f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--seed", type=int)
    g.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pool_size", type=int, default=8000)
    ap.add_argument("--out_dir", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results_oracle")))
    args = ap.parse_args()
    seeds = args.seeds or ([args.seed] if args.seed is not None else [42])
    print(f"== {args.method} on {args.dataset}, seeds={seeds} ==")
    for s in seeds:
        run_one(args.method, args.dataset, s, args.device,
                pool_size=args.pool_size, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
