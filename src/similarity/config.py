"""Runtime configuration.

Every field can be overridden with an environment variable prefixed ``SIM_``
(e.g. ``SIM_BACKEND=exact``, ``SIM_WEIGHTS='{"text": 0.6, "brand": 0.1}'``),
which is how the Kubernetes ConfigMap tunes a deployment without a rebuild.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATA_FILE = (
    "data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)

# Relative importance of each attribute block. Every block is unit-normalised and scaled by
# sqrt(weight), so the final score is exactly sum(weight_b * cosine_b) (see features.py). Weights
# are normalised to sum to 1 (DEFAULT_WEIGHTS already does), so scores are a weighted average in [-1, 1].
#
# Chosen by grid search (scripts/tune_weights.py, maximising "also-bought hit@10" on a tuning
# sample) and reported on a disjoint held-out sample (scripts/evaluate.py). Two stages:
#   1. Text era: price band mattered more than guessed, and brand is partly redundant once text
#      is present (the brand usually appears in the product name). tfidf hit@10 0.440 -> 0.461.
#   2. Image block (CLIP, PCA-128): hit@10 rises on a broad plateau for image weights 0.3-0.7
#      (grid units, tuning split 0.559 -> 0.62-0.63). We take the low end of the plateau with
#      category kept at 0.20: category@10 stays ~0.97 instead of ~0.92, and 0.5% of products
#      have no photo, so leaning on images less is the more robust choice.
#   text      - product_name carries product *type* and usually the brand.
#   image     - CLIP photo embedding: style, pattern, silhouette (99.5% coverage).
#   category  - Amazon's curated child category (83% coverage).
#   price     - shoppers compare within a price band.
#   brand     - explicit brand match (72% coverage after removing "Generic").
#   colour    - useful but only ~20% coverage (the photo carries colour too).
#   rating    - weak: most products have 0-2 reviews (shrunk, see data.py).
#   weight    - 79% missing and mostly a shipping artefact, so a token weight.
HAND_SET_WEIGHTS: dict[str, float] = {
    "text": 0.45, "category": 0.20, "brand": 0.15, "price": 0.10, "colour": 0.05, "rating": 0.03, "weight": 0.02,
}
# Grid point text .45 / image .30 / category .20 / price .20 / brand .10 / colour .05 / rating .03 /
# weight .02, rescaled to sum to 1 and rounded.
DEFAULT_WEIGHTS: dict[str, float] = {
    "text": 0.33,
    "image": 0.22,
    "category": 0.15,
    "brand": 0.07,
    "price": 0.15,
    "colour": 0.04,
    "rating": 0.02,
    "weight": 0.02,
}
assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SIM_", protected_namespaces=())

    data_path: Path = Path(DEFAULT_DATA_FILE)
    artifacts_dir: Path = Path("artifacts")

    encoder: Literal["tfidf", "minilm"] = "tfidf"
    backend: Literal["exact", "hnsw"] = "hnsw"
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # Text / categorical embedding sizes.
    text_dim: int = 128
    brand_dim: int = 128
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Precomputed image embeddings (scripts/fetch_images.py + embed_images.py): uniq_id -> PCA'd
    # CLIP/ResNet vector. Products without one get a zero image block; if the file is absent the
    # whole block is zero and similarity falls back to the other attributes.
    image_embeddings_path: Path = Path("data/image_embeddings.npz")
    image_dim: int = 128  # block width used when the file is absent

    # Bayesian rating shrinkage strength: a product needs ~m reviews before its own
    # average outweighs the catalogue mean.
    rating_prior_m: float = 5.0

    # HNSW parameters (Malkov & Yashunin). M = graph degree, ef = beam width.
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64  # recall@10 = 0.998 vs exact on the 30k catalogue (benchmark.py)
    hnsw_metric: Literal["ip", "l2"] = "ip"  # see index.py for the ip-vs-L2-reduction experiment

    cache_size: int = 4096
    max_num_similar: int = 100
    seed: int = 42

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, w: dict[str, float]) -> dict[str, float]:
        unknown = set(w) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown weight blocks {sorted(unknown)}; valid: {sorted(DEFAULT_WEIGHTS)}")
        if any(v < 0 for v in w.values()):
            raise ValueError("weights must be non-negative")
        merged = {k: float(w.get(k, 0.0)) for k in DEFAULT_WEIGHTS}
        total = sum(merged.values())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        # Normalising doesn't change any ranking; it keeps scores comparable across configs.
        return {k: v / total for k, v in merged.items()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
