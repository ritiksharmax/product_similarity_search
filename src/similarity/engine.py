"""Part 1: ``find_similar_products``, and the engine behind it.

The engine works in two phases:

* **offline** (``SimilarityEngine.build`` / ``scripts/build_artifacts.py``): clean the data,
  build the block matrix, build the HNSW graph, and write everything to ``artifacts/``. This
  runs during ``docker build``, so containers start in about a second.
* **online** (``SimilarityEngine.similar``): look up the product's vector, then search,
  filter, tie-break and cache.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from similarity.config import Settings, get_settings
from similarity.data import load_catalog
from similarity.features import BLOCK_ORDER, BlockMatrix, build_blocks
from similarity.index import ExactIndex, HnswIndex, SimilarityIndex

log = logging.getLogger(__name__)

# Extra candidates fetched beyond k, so ties at the cut-off are resolved by our deterministic
# tie-breaker instead of by whatever order the index happens to return.
_TIE_MARGIN = 16
_CATALOG_COLUMNS = ["uniq_id", "asin", "product_name", "brand", "category", "price", "rating", "review_count"]


class ProductNotFoundError(KeyError):
    """The requested ``product_id`` is not in the catalogue."""


@dataclass(frozen=True)
class Match:
    product_id: str
    score: float
    breakdown: dict[str, float] | None = None


class SimilarityEngine:
    def __init__(self, catalog: pd.DataFrame, blocks: BlockMatrix, settings: Settings,
                 hnsw: HnswIndex | None = None):
        self.settings = settings
        self.catalog = catalog.reset_index(drop=True)
        self.blocks = blocks
        self.ids: np.ndarray = self.catalog["uniq_id"].to_numpy()
        self._row: dict[str, int] = {pid: i for i, pid in enumerate(self.ids)}
        # The same ASIN crawled twice is the same product, so never recommend it to itself.
        self._same_product: dict[int, np.ndarray] = {}
        for rows in self.catalog.groupby("asin").indices.values():
            if len(rows) > 1:
                for r in rows:
                    self._same_product[int(r)] = rows
        self._rating = self.catalog["rating"].to_numpy(dtype=float)

        self.vectors = blocks.weighted(settings.weights)
        self.exact = ExactIndex(self.vectors)
        self.index: SimilarityIndex
        if settings.backend == "hnsw":
            self.index = hnsw or HnswIndex.build(
                self.vectors, settings.hnsw_m, settings.hnsw_ef_construction,
                settings.hnsw_ef_search, settings.hnsw_metric,
            )
        else:
            self.index = self.exact
        # Every query is searched to one fixed depth and cached once per product, then sliced
        # to the requested k. This makes top-k a prefix of top-(k+1) by construction, even under
        # HNSW (whose candidate pool otherwise grows with k), and any k hits the same cache entry.
        self._depth = max(1, settings.max_num_similar)
        self._cached_search = lru_cache(maxsize=settings.cache_size)(self._search_fixed_depth)

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, settings: Settings, catalog: pd.DataFrame | None = None) -> "SimilarityEngine":
        t0 = time.perf_counter()
        if catalog is None:
            catalog = load_catalog(settings.data_path, settings.rating_prior_m)
        blocks = build_blocks(catalog, settings)
        engine = cls(catalog, blocks, settings)
        log.info("built engine: %d products, dim=%d, encoder=%s, backend=%s in %.1fs",
                 len(catalog), blocks.dim, settings.encoder, settings.backend, time.perf_counter() - t0)
        return engine

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "blocks.npy", self.blocks.data)
        self.catalog[_CATALOG_COLUMNS].to_json(directory / "catalog.jsonl", orient="records", lines=True)
        if isinstance(self.index, HnswIndex):
            self.index.save(directory / "hnsw.faiss")
        manifest = {
            "encoder": self.settings.encoder,
            "weights": self.settings.weights,
            "slices": {k: [s.start, s.stop] for k, s in self.blocks.slices.items()},
            "n_products": len(self.ids),
            "dim": self.blocks.dim,
            "image": self.blocks.info.get("image"),
            "coverage": {k: round(v, 4) for k, v in self.blocks.coverage().items()},
            "hnsw": {"m": self.settings.hnsw_m, "ef_construction": self.settings.hnsw_ef_construction,
                     "metric": self.settings.hnsw_metric}
            if isinstance(self.index, HnswIndex) else None,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, settings: Settings) -> "SimilarityEngine":
        """Load prebuilt artifacts if they match the settings, otherwise build from the raw data."""
        directory = settings.artifacts_dir
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            log.warning("no artifacts in %s; building from %s", directory, settings.data_path)
            return cls.build(settings)
        manifest = json.loads(manifest_path.read_text())
        if set(manifest["slices"]) != set(BLOCK_ORDER):
            log.warning("artifacts have blocks %s, code expects %s; rebuilding",
                        sorted(manifest["slices"]), sorted(BLOCK_ORDER))
            return cls.build(settings)
        if manifest["encoder"] != settings.encoder:
            if not Path(settings.data_path).exists():
                # Serving images ship artifacts but not the raw data, so a rebuild is impossible.
                # Fail with the actual cause instead of a confusing "data not found".
                raise RuntimeError(
                    f"artifacts in {directory} were built with encoder={manifest['encoder']!r} but "
                    f"SIM_ENCODER={settings.encoder!r}, and the raw data needed to rebuild is absent. "
                    f"Unset SIM_ENCODER or rebuild the image with --build-arg ENCODER={settings.encoder}."
                )
            log.warning("artifacts use encoder=%s but settings ask for %s; rebuilding",
                        manifest["encoder"], settings.encoder)
            return cls.build(settings)

        blocks = BlockMatrix(
            data=np.load(directory / "blocks.npy"),
            slices={k: slice(a, b) for k, (a, b) in manifest["slices"].items()},
            info={"image": manifest.get("image")},
        )
        catalog = pd.read_json(directory / "catalog.jsonl", orient="records", lines=True,
                               dtype={"uniq_id": str, "asin": str})
        hnsw = None
        hnsw_path = directory / "hnsw.faiss"
        # The HNSW graph is built on *weighted* vectors, so it is only reusable when the
        # weights are unchanged. Otherwise the constructor rebuilds it (a few seconds at 30k).
        built_with = manifest.get("hnsw") or {}
        reusable = (
            manifest["weights"] == settings.weights
            and built_with.get("metric") == settings.hnsw_metric
            and built_with.get("m") == settings.hnsw_m
        )
        if settings.backend == "hnsw" and hnsw_path.exists() and reusable:
            hnsw = HnswIndex.load(hnsw_path, blocks.weighted(settings.weights),
                                  settings.hnsw_ef_search, settings.hnsw_metric)
        elif settings.backend == "hnsw":
            log.warning("prebuilt HNSW graph doesn't match current weights/params; rebuilding in memory")
        return cls(catalog, blocks, settings, hnsw=hnsw)

    # ------------------------------------------------------------------ querying

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, product_id: str) -> bool:
        return product_id in self._row

    def row_of(self, product_id: str) -> int:
        try:
            return self._row[product_id]
        except KeyError:
            raise ProductNotFoundError(product_id) from None

    def _search_fixed_depth(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        idx, scores = self._search(row, self._depth)
        idx.flags.writeable = scores.flags.writeable = False  # shared via the cache
        return idx, scores

    def _search(self, row: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        excluded = self._same_product.get(row, np.array([row]))
        available = len(self) - len(excluded)
        k = min(k, available)
        if k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        fetch = min(len(self), k + len(excluded) + _TIE_MARGIN)
        query = self.vectors[row : row + 1]

        scores, idx = self.index.search(query, fetch)
        scores, idx = scores[0], idx[0]
        keep = (idx >= 0) & ~np.isin(idx, excluded)
        if keep.sum() < k and self.index is not self.exact:
            # ANN graphs can (rarely) return too few results; fall back to exact search.
            scores, idx = self.exact.search(query, fetch)
            scores, idx = scores[0], idx[0]
            keep = ~np.isin(idx, excluded)
        scores, idx = scores[keep], idx[keep]

        # Deterministic ordering: score desc, then (shrunk) rating desc, then uniq_id asc.
        # Scores are rounded so float noise from the index doesn't decide ties.
        # np.lexsort sorts by its *last* key first.
        order = np.lexsort((self.ids[idx], -self._rating[idx], -np.round(scores, 6)))[:k]
        return idx[order].astype(np.int64), scores[order].astype(np.float32)

    def similar(self, product_id: str, num_similar: int, explain: bool = False) -> list[Match]:
        if num_similar < 1:
            raise ValueError("num_similar must be >= 1")
        row = self.row_of(product_id)
        k = int(num_similar)
        if k <= self._depth:
            rows, scores = self._cached_search(row)
            rows, scores = rows[:k], scores[:k]
        else:  # library callers may exceed the API cap; serve uncached
            rows, scores = self._search(row, k)
        breakdown = None
        if explain and len(rows):
            per_block = self.blocks.block_scores(row, rows)
            breakdown = [
                {name: round(float(v[j]), 4) for name, v in per_block.items()} for j in range(len(rows))
            ]
        return [
            Match(str(self.ids[r]), round(float(s), 6), breakdown[j] if breakdown else None)
            for j, (r, s) in enumerate(zip(rows, scores, strict=True))
        ]

    def find_similar_products(self, product_id: str, num_similar: int) -> List[str]:
        return [m.product_id for m in self.similar(product_id, num_similar)]

    def describe(self, product_id: str) -> dict:
        rec = self.catalog.iloc[self.row_of(product_id)][_CATALOG_COLUMNS].to_dict()
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}

    def cache_info(self) -> dict:
        return self._cached_search.cache_info()._asdict()


# ---------------------------------------------------------------------- module-level API

_engine: SimilarityEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> SimilarityEngine:
    """Process-wide engine, loaded lazily on first use (thread-safe)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = SimilarityEngine.load(get_settings())
    return _engine


def set_engine(engine: SimilarityEngine | None) -> None:
    """Inject an engine (used by the API's startup hook and by tests)."""
    global _engine
    _engine = engine


def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """Return the ``num_similar`` product IDs most similar to ``product_id``.

    Similarity is a weighted sum of per-attribute cosine similarities over product name
    (text), category, brand, colour, price, rating and weight (see ``features.py``).
    Results are sorted by descending score. Ties are broken by higher (review-shrunk)
    rating, then by ``uniq_id``. The product itself, and re-crawled copies with the same
    ASIN, are excluded.

    Raises:
        ProductNotFoundError: ``product_id`` is not in the dataset.
        ValueError: ``num_similar`` < 1.
    """
    return get_engine().find_similar_products(product_id, num_similar)


__all__ = ["Match", "ProductNotFoundError", "SimilarityEngine", "find_similar_products", "get_engine", "set_engine"]
