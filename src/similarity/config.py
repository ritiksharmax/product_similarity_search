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
# sqrt(weight), so the final score is exactly sum(weight_b * cosine_b) (see features.py), and
# weights are normalised to sum to 1, so scores are a weighted average in [-1, 1]. Values are
# relative: DEFAULT_WEIGHTS sums to 0.90 (grid-search units), which rescales text to 0.33.
#
# The values were chosen by a small grid search (scripts/tune_weights.py) that maximises
# "also-bought hit@10" on one query sample and is reported on a disjoint held-out sample
# (scripts/evaluate.py). This setting ranked top-2 for *both* text encoders. Held-out effect
# vs the initial hand-set weights (text .45 / category .20 / brand .15 / price .10):
#   tfidf: hit@10 0.440 -> 0.461, price drift 0.178 -> 0.135.
# Interpretation: price band matters more to shoppers than we guessed, and once text is
# present, brand is partly redundant (the brand usually appears in the product name).
#   text      - product_name carries product *type* and usually the brand.
#   category  - Amazon's curated child category (83% coverage).
#   price     - shoppers compare within a price band.
#   brand     - explicit brand match (72% coverage after removing "Generic").
#   colour    - useful but only ~20% coverage.
#   rating    - weak: most products have 0-2 reviews (shrunk, see data.py).
#   weight    - 79% missing and mostly a shipping artefact, so a token weight.
HAND_SET_WEIGHTS: dict[str, float] = {
    "text": 0.45, "category": 0.20, "brand": 0.15, "price": 0.10, "colour": 0.05, "rating": 0.03, "weight": 0.02,
}
DEFAULT_WEIGHTS: dict[str, float] = {
    "text": 0.30,
    "category": 0.20,
    "brand": 0.10,
    "price": 0.20,
    "colour": 0.05,
    "rating": 0.03,
    "weight": 0.02,
}


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
