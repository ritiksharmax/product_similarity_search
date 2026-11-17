"""Build the similarity artifacts (features + HNSW graph) from the raw dataset.

    uv run python scripts/build_artifacts.py            # uses SIM_* env / defaults
    SIM_ENCODER=minilm uv run python scripts/build_artifacts.py

Run during `docker build` so serving containers only *load* artifacts.
"""

from __future__ import annotations

import json
import logging
import time

from similarity.config import get_settings
from similarity.engine import SimilarityEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    t0 = time.perf_counter()
    engine = SimilarityEngine.build(settings)
    engine.save(settings.artifacts_dir)
    print(json.dumps({
        "products": len(engine),
        "dim": engine.blocks.dim,
        "encoder": settings.encoder,
        "backend": settings.backend,
        "block_coverage": {k: round(v, 3) for k, v in engine.blocks.coverage().items()},
        "seconds": round(time.perf_counter() - t0, 1),
        "artifacts": str(settings.artifacts_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
