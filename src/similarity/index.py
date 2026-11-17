"""Nearest-neighbour backends behind a common interface.

* :class:`ExactIndex` is brute-force ``X @ q``. It is exact, needs no build step, and costs
  O(N·d) per query (a few ms at 30k products). It is the ground truth that the ANN index is
  measured against.
* :class:`HnswIndex` uses FAISS ``IndexHNSWFlat``, the Hierarchical Navigable Small World
  graph of Malkov & Yashunin (arXiv:1603.09320, IEEE TPAMI 2020). It costs roughly O(log N)
  per query. See README "Part 3" for why we chose it.

**Inner product vs. HNSW.** Our score is a raw inner product, and vectors have unequal norms
(missing attributes give zero blocks). There are two standard ways to run HNSW on that:

* ``metric="ip"`` (default): build the graph directly on inner products (FAISS
  ``METRIC_INNER_PRODUCT``). Morozov & Babenko, "Non-metric Similarity Graphs for Maximum
  Inner Product Search" (NeurIPS 2018), show that this "ip-NSW" works well in practice.
* ``metric="l2"``: the MIPS-to-NN reduction of Bachrach et al. (RecSys 2014). Each item gets
  an extra coordinate ``sqrt(R² − ‖x‖²)`` and each query gets a 0, so the L2 nearest neighbour
  is exactly the maximum inner product and HNSW runs in a true metric space.

We implemented both and measured them on the real catalogue (``scripts/benchmark.py``). The
reduction is *exact* (flat-L2 recall 1.0), yet HNSW recall on the transformed vectors was
*lower*: 0.988 vs 0.996 at efSearch=16. Pushing every point onto a sphere of radius R
compresses distance contrast, which hurts greedy graph routing (Morozov & Babenko make the
same point). Our norms only vary over 0.69–1.0, so native IP is the default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import faiss
import numpy as np


class SimilarityIndex(Protocol):
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)``, each ``(n_queries, k)``, best first. Index -1 = no result."""
        ...


class ExactIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.atleast_2d(queries).astype(np.float32)
        k = min(k, len(self))
        scores = queries @ self.vectors.T
        # argpartition is O(N); fully sorting all N candidates is unnecessary.
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        return np.take_along_axis(top_scores, order, axis=1), np.take_along_axis(top, order, axis=1)


class HnswIndex:
    def __init__(self, index: faiss.Index, vectors: np.ndarray, metric: str = "ip"):
        self.index = index
        self.metric = metric
        self.vectors = vectors  # kept to report true inner-product scores
        self.max_norm = float(np.linalg.norm(vectors, axis=1).max()) if len(vectors) else 0.0

    def __len__(self) -> int:
        return self.index.ntotal

    @staticmethod
    def _augment(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        sq = np.square(vectors).sum(axis=1)
        extra = np.sqrt(np.maximum(max_norm**2 - sq, 0.0)).astype(np.float32)[:, None]
        return np.ascontiguousarray(np.hstack([vectors, extra]), dtype=np.float32)

    @classmethod
    def build(cls, vectors: np.ndarray, m: int = 32, ef_construction: int = 200,
              ef_search: int = 64, metric: Literal["ip", "l2"] = "ip") -> "HnswIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if metric == "ip":
            index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            payload = vectors
        else:
            index = faiss.IndexHNSWFlat(vectors.shape[1] + 1, m, faiss.METRIC_L2)
            payload = cls._augment(vectors, float(np.linalg.norm(vectors, axis=1).max()))
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(payload)
        return cls(index, vectors, metric)

    @property
    def ef_search(self) -> int:
        return self.index.hnsw.efSearch

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self.index.hnsw.efSearch = value

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(k, len(self))
        # HNSW returns at most efSearch results, so widen the beam for large k. This is passed
        # *per call* rather than written to the index: FastAPI serves sync endpoints from a
        # thread pool, and mutating shared state would race and permanently change latency.
        params = faiss.SearchParametersHNSW(efSearch=max(self.ef_search, k))
        if self.metric == "l2":
            queries_in = np.ascontiguousarray(np.hstack([queries, np.zeros((len(queries), 1), np.float32)]))
        else:
            queries_in = queries
        _, idx = self.index.search(queries_in, k, params=params)
        # Always report the true inner product (the ranking quantity), never the L2 surrogate.
        safe = np.where(idx >= 0, idx, 0)
        scores = np.einsum("qd,qkd->qk", queries, self.vectors[safe])
        scores[idx < 0] = -np.inf
        return scores.astype(np.float32), idx

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray, ef_search: int, metric: str = "ip") -> "HnswIndex":
        index = faiss.read_index(str(path))
        index.hnsw.efSearch = ef_search
        return cls(index, np.ascontiguousarray(vectors, dtype=np.float32), metric)
