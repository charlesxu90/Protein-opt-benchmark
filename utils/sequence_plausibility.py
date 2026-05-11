"""
Sequence Plausibility via ESM-2

Computes pseudo-perplexity / pseudo-log-likelihood (PLL) of protein sequences
under ESM-2 as a model-independent plausibility signal. Used both as a quality
metric (PDFBench-style) and, optionally, as AlphaVariant's secondary reward.

Why pseudo-PPL: ESM-2 is a masked language model, so the standard left-to-right
PPL definition does not apply. We use the Salazar-2020 pseudo-log-likelihood:
mask each position one at a time and average the log-probability of the true
token.

Performance notes
-----------------
- Lazy heavy imports: torch / transformers are imported at first call so this
  module loads in environments where they aren't installed.
- Disk cache keyed by SHA-1 of the sequence at `~/.cache/alphavariant_ppl/`.
- Batched scoring across sequences; default batch size 8 sequences. Per-sequence
  cost is O(L) forward passes with naive PLL; we use the L-step batched form.

Models
------
Default `esm2_t33_650M_UR50D`. Pass `model_name="facebook/esm2_t12_35M_UR50D"`
for ~10x speedup at the cost of plausibility quality.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Union
import hashlib
import json
import os

import numpy as np


_DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/alphavariant_ppl")


# =============================================================================
# Cache
# =============================================================================

def _cache_path(seq: str, model_name: str, cache_dir: str) -> str:
    key = hashlib.sha1(f"{model_name}::{seq}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def _read_cache(seq: str, model_name: str, cache_dir: str) -> Optional[float]:
    path = _cache_path(seq, model_name, cache_dir)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return float(json.load(f)["pll"])
        except Exception:
            return None
    return None


def _write_cache(seq: str, model_name: str, cache_dir: str, pll: float) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(seq, model_name, cache_dir)
    with open(path, "w") as f:
        json.dump({"seq": seq, "model": model_name, "pll": float(pll)}, f)


# =============================================================================
# Lazy model loader (singleton)
# =============================================================================

_MODEL_CACHE: dict = {}


def _load_model(model_name: str, device: Optional[str] = None):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError as e:
        raise ImportError(
            "ESM-2 plausibility requires torch and transformers. "
            "Install with `pip install torch transformers`."
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _MODEL_CACHE[model_name] = (tokenizer, model, device, torch)
    return _MODEL_CACHE[model_name]


# =============================================================================
# Core scoring
# =============================================================================

def _pll_one(seq: str, tokenizer, model, device, torch_mod) -> float:
    """Pseudo-log-likelihood for one sequence (sum, not mean)."""
    enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)  # (1, L+2) with CLS/SEP
    attn = enc["attention_mask"].to(device)
    L = input_ids.shape[1]

    # Mask each non-special position one at a time, batched
    mask_token_id = tokenizer.mask_token_id
    # special tokens at positions 0 and L-1
    positions = list(range(1, L - 1))
    if not positions:
        return 0.0

    # Build a (P, L) batch where row k masks position positions[k]
    P = len(positions)
    masked = input_ids.repeat(P, 1).clone()
    targets = []
    for k, p in enumerate(positions):
        targets.append(int(masked[k, p].item()))
        masked[k, p] = mask_token_id
    attn_b = attn.repeat(P, 1)

    with torch_mod.no_grad():
        logits = model(masked, attention_mask=attn_b).logits  # (P, L, V)

    # Gather log-prob of the true token at each masked position
    log_probs = torch_mod.log_softmax(logits, dim=-1)
    pll = 0.0
    for k, p in enumerate(positions):
        pll += float(log_probs[k, p, targets[k]].item())
    return pll


def pll(
    sequences: Union[str, Sequence[str]],
    model_name: str = _DEFAULT_MODEL,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    normalize_by_length: bool = False,
) -> Union[float, np.ndarray]:
    """Pseudo-log-likelihood of one or more protein sequences under ESM-2.

    Args:
        sequences: a single sequence or a list of sequences.
        model_name: HuggingFace model id (default: ESM-2 t33 650M).
        cache_dir: directory for per-sequence JSON cache. Default
            ~/.cache/alphavariant_ppl.
        device: "cuda" / "cpu". Auto-detected if None.
        use_cache: if False, ignore and overwrite cache.
        normalize_by_length: if True, return PLL / sequence length.

    Returns:
        float (single sequence) or np.ndarray of shape (n,).
    """
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR

    single = isinstance(sequences, str)
    seqs = [sequences] if single else list(sequences)

    out = np.zeros(len(seqs), dtype=float)
    needed_indices: List[int] = []
    for i, s in enumerate(seqs):
        if use_cache:
            cached = _read_cache(s, model_name, cache_dir)
            if cached is not None:
                out[i] = cached / (len(s) if normalize_by_length else 1.0)
                continue
        needed_indices.append(i)

    if needed_indices:
        tokenizer, model, dev, torch_mod = _load_model(model_name, device)
        for i in needed_indices:
            score = _pll_one(seqs[i], tokenizer, model, dev, torch_mod)
            _write_cache(seqs[i], model_name, cache_dir, score)
            out[i] = score / (len(seqs[i]) if normalize_by_length else 1.0)

    return float(out[0]) if single else out


def perplexity(
    sequences: Union[str, Sequence[str]],
    model_name: str = _DEFAULT_MODEL,
    **kwargs,
) -> Union[float, np.ndarray]:
    """Pseudo-perplexity = exp(-PLL / L). Lower is more plausible.

    Convenience wrapper around `pll(..., normalize_by_length=True)`.
    """
    pll_per_residue = pll(sequences, model_name=model_name,
                          normalize_by_length=True, **kwargs)
    if isinstance(pll_per_residue, float):
        return float(np.exp(-pll_per_residue))
    return np.exp(-np.asarray(pll_per_residue))


# =============================================================================
# Aliases used in the refined benchmark plan
# =============================================================================

def esm2_ppl(sequences, **kwargs):
    """Alias: pseudo-perplexity under ESM-2 (default model)."""
    return perplexity(sequences, **kwargs)


def esm2_log_likelihood(sequences, **kwargs):
    """Alias: total pseudo-log-likelihood under ESM-2."""
    return pll(sequences, normalize_by_length=False, **kwargs)


__all__ = [
    "pll",
    "perplexity",
    "esm2_ppl",
    "esm2_log_likelihood",
]
