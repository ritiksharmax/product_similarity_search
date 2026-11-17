"""Product -> vector. The design is a *weighted block embedding*.

Each attribute becomes its own block, and every block is either unit-length or all-zero:

=========  ===================================================  ===============================
block      encoding                                             cosine between two products
=========  ===================================================  ===============================
text       TF-IDF+SVD (LSA) or MiniLM of ``product_name``       semantic similarity of names
image      CLIP (or ResNet50) image embedding, PCA to 128-d     visual similarity of photos
category   one-hot over Amazon child categories                 1 if same category else 0
brand      fixed random unit vector per brand (128-d)           1 if same brand else ~0
colour     multi-hot over a base palette                        overlap of colour sets
price      (cos θ, sin θ), θ = π · percentile(log price)        cos(π·|Δpercentile|)
rating     same arc encoding of the shrunk rating               cos(π·|Δpercentile|)
weight     same arc encoding of weight in grams                 cos(π·|Δpercentile|)
=========  ===================================================  ===============================

Scaling block ``b`` by ``sqrt(w_b)`` and concatenating gives, for products ``x`` and ``y``::

    <x, y> = Σ_b w_b · cos_b(x, y)

So one inner-product search computes an interpretable, configurable weighted sum of
per-attribute similarities. Exact search and FAISS HNSW then work on the same vectors.

Missing values give a zero block, which contributes 0 (neutral, "no evidence") rather than
an imputed guess. If the *query* lacks an attribute, that block adds the same 0 to every
candidate and drops out of the ranking. This is the fallback mechanism for incomplete data.

Why the arc encoding for numbers: an inner-product index can't compute ``|a - b|`` directly.
Mapping a percentile ``q`` to the point ``(cos πq, sin πq)`` on a half circle makes the dot
product a monotone function of the distance: identical gives 1, ten percentiles apart gives
0.95, opposite ends give -1. Percentiles (not raw values) make it robust to the heavy price
skew (₹39 to ₹9,988).

Why random vectors for brand: there are 6.4k brands, and a one-hot would give a mostly-empty
6.4k-d vector, which is too large for a dense ANN index. Random Gaussian unit vectors are
nearly orthogonal (Johnson-Lindenstrauss): same brand gives 1, different brands give
N(0, 1/d) cross-talk, about ±0.09 at d=128, which the brand weight scales down to roughly
±0.013 of total score. Categories (219 values) are few enough for an exact one-hot.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline

from similarity.config import DEFAULT_WEIGHTS, Settings
from similarity.data import BASE_COLOURS, percentile_rank

log = logging.getLogger(__name__)

BLOCK_ORDER: tuple[str, ...] = tuple(DEFAULT_WEIGHTS)


# --------------------------------------------------------------------------- helpers


def l2_normalise(m: np.ndarray) -> np.ndarray:
    """Row-normalise; all-zero rows stay zero (= missing)."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0).astype(np.float32)


def arc_encode(percentiles: np.ndarray) -> np.ndarray:
    """Percentile in [0, 1] -> unit vector on a half circle; NaN -> zero vector."""
    theta = np.pi * np.nan_to_num(percentiles, nan=0.0)
    out = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    out[np.isnan(percentiles)] = 0.0
    return out


def stable_random_vector(key: str, dim: int, seed: int) -> np.ndarray:
    """Deterministic pseudo-random unit vector for a categorical value.

    Seeded from a hash of the value, so the vector for "biba" is identical across processes
    and rebuilds. Python's ``hash()`` is randomised per process, so it isn't used here.
    """
    digest = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


# --------------------------------------------------------------------------- text encoders


class TextEncoder(Protocol):
    name: str

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...


class TfidfLsaEncoder:
    """The *efficient* encoder: TF-IDF followed by truncated SVD (Latent Semantic Analysis).

    Word 1-2 grams capture tokens like "t-shirt" and "slim fit". Character 3-5 grams (``char_wb``)
    make it robust to the spelling noise common in marketplace titles ("tshirt", "t shirt",
    "kurti"/"kurta"). SVD compresses ~100k sparse features into a dense ``dim``-vector suitable
    for ANN, and merges co-occurring terms (Deerwester et al., 1990). It fits in seconds on CPU
    and needs no model download.
    """

    name = "tfidf"

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        # Drop hapax terms on real catalogues; keep everything on tiny ones (tests).
        big = len(texts) >= 1000
        vectoriser = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2 if big else 1, sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w[\w-]*\b", max_features=100_000)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3 if big else 1,
                                         sublinear_tf=True, max_features=100_000)),
            ]
        )
        dim = max(1, min(self.dim, len(texts) - 1))
        pipeline = make_pipeline(vectoriser, TruncatedSVD(n_components=dim, random_state=self.seed))
        return l2_normalise(pipeline.fit_transform(texts).astype(np.float32))


class MiniLmEncoder:
    """The *accurate* encoder: a pretrained transformer sentence embedding.

    ``all-MiniLM-L6-v2`` (Reimers & Gurevych, Sentence-BERT, 2019; 384-d) knows that
    "kurti" ≈ "tunic" and "tee" ≈ "t-shirt" without seeing them co-occur in our catalogue.
    The cost is a torch dependency (~1 GB image) and slower encoding. Install with
    ``uv sync --extra minilm``.
    """

    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    @property
    def prefix(self) -> str:
        # E5 models were trained with role prefixes; for symmetric similarity both sides use
        # "query: " (Wang et al., 2022). MiniLM/BGE/GTE take raw text for symmetric tasks.
        return "query: " if "e5" in self.model_name.lower() else ""

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("encoder 'minilm' needs `uv sync --extra minilm`") from exc
        import torch

        # The faiss-cpu and torch wheels each bundle their own OpenMP runtime. Once faiss is
        # loaded, a multi-threaded torch forward pass deadlocks on macOS (reproduced; Linux
        # wheels carry the same risk). Encoding runs offline at build time, so we trade
        # torch's intra-op parallelism for safety: ~10s for the 30k catalogue on one thread.
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = SentenceTransformer(self.model_name, device="cpu")
            emb = model.encode([self.prefix + t for t in texts], batch_size=256,
                               normalize_embeddings=True, show_progress_bar=False)
        finally:
            torch.set_num_threads(previous)
        return np.asarray(emb, dtype=np.float32)


def make_text_encoder(settings: Settings) -> TextEncoder:
    if settings.encoder == "minilm":
        return MiniLmEncoder(settings.minilm_model)
    return TfidfLsaEncoder(settings.text_dim, settings.seed)


# --------------------------------------------------------------------------- image block


def load_image_block(catalog: pd.DataFrame, settings: Settings) -> tuple[np.ndarray, dict]:
    """Look up each product's precomputed image vector; missing -> zero row (fallback).

    Images are embedded offline (scripts/embed_images.py) because fetching and running a CNN
    over 30k photos does not belong in a request path or a Docker build. The file holds
    ``uniq_id``, unit-norm float16 ``vectors`` and a JSON ``meta`` describing model and PCA.
    """
    path = settings.image_embeddings_path
    if not path or not path.exists():
        log.warning("no image embeddings at %s; image block is empty (falls back to other attributes)", path)
        return np.zeros((len(catalog), settings.image_dim), dtype=np.float32), {"source": None}
    with np.load(path, allow_pickle=False) as npz:
        ids = npz["uniq_id"].astype(str)
        vectors = npz["vectors"].astype(np.float32)
        meta = json.loads(str(npz["meta"])) if "meta" in npz else {}
    row = {pid: i for i, pid in enumerate(ids)}
    block = np.zeros((len(catalog), vectors.shape[1]), dtype=np.float32)
    for i, pid in enumerate(catalog["uniq_id"]):
        j = row.get(pid)
        if j is not None:
            block[i] = vectors[j]
    return l2_normalise(block), {"source": path.name, **meta}


# --------------------------------------------------------------------------- block matrix


@dataclass
class BlockMatrix:
    """Unweighted unit blocks laid side by side, plus where each block lives."""

    data: np.ndarray  # (n_products, total_dim) float32, every block unit-norm or zero
    slices: dict[str, slice] = field(default_factory=dict)
    info: dict = field(default_factory=dict)  # provenance, e.g. which image model was used

    @property
    def dim(self) -> int:
        return self.data.shape[1]

    def column_scale(self, weights: dict[str, float]) -> np.ndarray:
        scale = np.zeros(self.dim, dtype=np.float32)
        for name, sl in self.slices.items():
            scale[sl] = np.sqrt(weights.get(name, 0.0))
        return scale

    def weighted(self, weights: dict[str, float]) -> np.ndarray:
        """Apply ``sqrt(w_b)`` per block, so the inner product is ``Σ w_b · cos_b``."""
        return np.ascontiguousarray(self.data * self.column_scale(weights), dtype=np.float32)

    def block_scores(self, i: int, js: np.ndarray) -> dict[str, np.ndarray]:
        """Unweighted per-block cosine between product ``i`` and products ``js`` (explainability)."""
        return {
            name: self.data[js, sl] @ self.data[i, sl]
            for name, sl in self.slices.items()
        }

    def coverage(self) -> dict[str, float]:
        """Fraction of products with a non-empty block (data-quality report)."""
        return {
            name: float((np.abs(self.data[:, sl]).sum(axis=1) > 0).mean())
            for name, sl in self.slices.items()
        }


def build_blocks(catalog: pd.DataFrame, settings: Settings, text_encoder: TextEncoder | None = None) -> BlockMatrix:
    encoder = text_encoder or make_text_encoder(settings)
    blocks: dict[str, np.ndarray] = {}

    blocks["text"] = encoder.fit_transform(catalog["product_name"].fillna("").tolist())
    blocks["image"], image_info = load_image_block(catalog, settings)

    categories = sorted(c for c in catalog["category"].dropna().unique())
    cat_index = {c: i for i, c in enumerate(categories)}
    cat = np.zeros((len(catalog), max(1, len(categories))), dtype=np.float32)
    for row, c in enumerate(catalog["category"]):
        if isinstance(c, str) and c in cat_index:
            cat[row, cat_index[c]] = 1.0
    blocks["category"] = cat

    brand_vecs: dict[str, np.ndarray] = {}
    brand = np.zeros((len(catalog), settings.brand_dim), dtype=np.float32)
    for row, b in enumerate(catalog["brand"]):
        if isinstance(b, str) and b:  # pandas stores missing as NaN, which is truthy
            if b not in brand_vecs:
                brand_vecs[b] = stable_random_vector(b, settings.brand_dim, settings.seed)
            brand[row] = brand_vecs[b]
    blocks["brand"] = brand

    colour_index = {c: i for i, c in enumerate(BASE_COLOURS)}
    colour = np.zeros((len(catalog), len(BASE_COLOURS)), dtype=np.float32)
    for row, cs in enumerate(catalog["colours"]):
        for c in cs:
            colour[row, colour_index[c]] = 1.0
    blocks["colour"] = l2_normalise(colour)

    blocks["price"] = arc_encode(percentile_rank(np.log1p(catalog["price"])))
    blocks["rating"] = arc_encode(percentile_rank(catalog["rating"]))
    blocks["weight"] = arc_encode(percentile_rank(catalog["weight_g"]))

    slices, start = {}, 0
    for name in BLOCK_ORDER:
        width = blocks[name].shape[1]
        slices[name] = slice(start, start + width)
        start += width
    data = np.ascontiguousarray(np.hstack([blocks[n] for n in BLOCK_ORDER]), dtype=np.float32)
    return BlockMatrix(data=data, slices=slices, info={"image": image_info})
