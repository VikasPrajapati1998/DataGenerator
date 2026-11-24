"""Semantic (near-duplicate) detection with embeddings.

Two embedders are available:

* :class:`HashingEmbedder` - character n-gram feature hashing (numpy only).
  Fast, deterministic across processes and needs no model; catches
  paraphrase-light near duplicates (reworded spacing, punctuation, small edits).
* :class:`OllamaEmbedder` - dense neural embeddings from an Ollama embedding
  model (e.g. ``nomic-embed-text`` or ``qwen3-embedding``); catches true
  paraphrases at a higher cost.

Within a batch, all pairs are compared exactly (matrix product). Across
batches, random-hyperplane SimHash signatures are stored per batch and
banded into LSH buckets, so only candidate pairs are compared.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.core.exceptions import StorageError
from app.utils.hashing import normalize_value

_DIM = 512
_NGRAM = 3
_SIG_BITS = 64
_BANDS = 8


class Embedder(Protocol):
    """Turns texts into L2-normalised vectors."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 matrix of unit vectors."""


def _bucket(token: str) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    value = int.from_bytes(digest, "big")
    return value % _DIM, 1.0 if value & 0x80000000 else -1.0


class HashingEmbedder:
    """Character n-gram + word feature-hashing embedder.

    Args:
        dim: Vector dimensionality.
    """

    def __init__(self, dim: int = _DIM) -> None:
        self.dim = dim
        self._cache: dict[str, tuple[int, float]] = {}

    def _features(self, text: str) -> list[str]:
        padded = f" {text} "
        grams = [padded[i:i + _NGRAM] for i in range(max(len(padded) - _NGRAM + 1, 1))]
        return grams + [f"w:{w}" for w in text.split()]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into unit vectors.

        Args:
            texts: Texts to embed.

        Returns:
            ``(len(texts), dim)`` float32 matrix.
        """
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                hit = self._cache.get(feature)
                if hit is None:
                    hit = _bucket(feature)
                    if len(self._cache) < 500_000:
                        self._cache[feature] = hit
                matrix[row, hit[0] % self.dim] += hit[1]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class OllamaEmbedder:
    """Dense embeddings through an Ollama embedding model.

    Args:
        adapter: An adapter whose ``embed`` method calls ``/api/embed``.
        batch: Texts per request.
    """

    def __init__(self, adapter: Any, batch: int = 64) -> None:
        self.adapter = adapter
        self.batch = batch

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` and L2-normalise the vectors."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch):
            vectors.extend(self.adapter.embed(list(texts[start:start + self.batch])))
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def record_text(record: dict[str, Any], fields: Sequence[str]) -> str:
    """Return the canonical text compared for semantic similarity.

    Args:
        record: The record.
        fields: Fields that participate.

    Returns:
        Normalised text joined with separators.
    """
    return " | ".join(normalize_value(record.get(f)) for f in fields if record.get(f) is not None)


def find_near_duplicates(vectors: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Return pairs ``(i, j, similarity)`` with ``i < j`` and similarity >= threshold.

    Args:
        vectors: ``(n, d)`` unit vectors.
        threshold: Cosine similarity threshold.

    Returns:
        Near-duplicate pairs; ``j`` is the later record (the one to reject).
    """
    if len(vectors) < 2:
        return []
    pairs: list[tuple[int, int, float]] = []
    block = 2048
    for start in range(0, len(vectors), block):
        sims = vectors[start:start + block] @ vectors.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i = start + r
            if i < c:
                pairs.append((i, c, float(sims[r, c])))
    return pairs


class SimHasher:
    """Random-hyperplane signatures for cross-batch LSH candidate search.

    Args:
        dim: Embedding dimensionality.
        seed: Hyperplane seed (fixed so signatures are comparable across tasks).
    """

    def __init__(self, dim: int = _DIM, seed: int = 1729) -> None:
        self.planes = np.random.default_rng(seed).standard_normal((dim, _SIG_BITS)).astype(np.float32)

    def signatures(self, vectors: np.ndarray) -> np.ndarray:
        """Return one unsigned 64-bit signature per vector."""
        bits = (vectors @ self.planes[: vectors.shape[1]]) > 0
        weights = (np.uint64(1) << np.arange(_SIG_BITS, dtype=np.uint64))
        return (bits.astype(np.uint64) * weights).sum(axis=1, dtype=np.uint64)


def save_signatures(path: Path, signatures: np.ndarray) -> None:
    """Persist signatures for later global LSH search (``.npy``), atomically.

    Raises:
        StorageError: If the file cannot be written.
    """
    tmp = path.with_suffix(".tmp.npy")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp, signatures.astype(np.uint64))
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise StorageError(f"Cannot write semantic signatures {path}: {exc}") from exc


def load_signatures(path: Path) -> np.ndarray:
    """Load signatures written by :func:`save_signatures`.

    Raises:
        StorageError: If the file is missing or corrupt.
    """
    try:
        return np.load(path)
    except (OSError, ValueError) as exc:
        raise StorageError(f"Cannot read semantic signatures {path}: {exc}") from exc


def lsh_candidates(signature_sets: list[tuple[int, np.ndarray]], max_hamming: int = 3
                   ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Find cross-batch record pairs with near-identical signatures.

    Args:
        signature_sets: ``(task_id, signatures)`` per batch.
        max_hamming: Maximum differing bits for a pair to count as near duplicate.

    Returns:
        Pairs of ``(task_id, row)`` locations; the second location is the one to repair.
    """
    band_bits = _SIG_BITS // _BANDS
    mask = np.uint64((1 << band_bits) - 1)
    buckets: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    found: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for task_id, sigs in signature_sets:
        for row, sig in enumerate(sigs.tolist()):
            sig_int = int(sig)
            for band in range(_BANDS):
                key = (band, (sig_int >> (band * band_bits)) & int(mask))
                bucket = buckets.setdefault(key, [])
                for other_task, other_row, other_sig in bucket:
                    if other_task != task_id and bin(other_sig ^ sig_int).count("1") <= max_hamming:
                        found.add(((other_task, other_row), (task_id, row)))
                if len(bucket) < 64:
                    bucket.append((task_id, row, sig_int))
    return sorted(found)
