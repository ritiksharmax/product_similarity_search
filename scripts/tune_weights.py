"""Small grid search over the main block weights (tuning split, seed 1).

Maximises also-bought hit@10 (see evaluate.py for the metric). Results are reported on the
disjoint seed-0 split by ``scripts/evaluate.py``, so the published numbers aren't tuned on.

    uv run python scripts/tune_weights.py --encoder tfidf
    uv run python scripts/tune_weights.py --encoder minilm --image   # image weight
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from evaluate import also_bought_targets, evaluate
from similarity.config import Settings
from similarity.data import load_catalog
from similarity.engine import SimilarityEngine
from similarity.features import build_blocks

GRID = {"text": [0.3, 0.45, 0.6], "category": [0.1, 0.2, 0.3], "brand": [0.1, 0.15, 0.25], "price": [0.05, 0.1, 0.2]}
FIXED = {"colour": 0.05, "rating": 0.03, "weight": 0.02, "image": 0.0}
# Second stage (--image): re-tune around the text-era optimum with image weight as a new axis.
IMAGE_GRID = {"image": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7], "text": [0.3, 0.45],
              "category": [0.1, 0.2], "price": [0.1, 0.2]}
IMAGE_FIXED = {"brand": 0.10, "colour": 0.05, "rating": 0.03, "weight": 0.02}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "minilm"])
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--image", action="store_true", help="tune the image weight (needs data/image_embeddings.npz)")
    args = ap.parse_args()
    grid, fixed = (IMAGE_GRID, IMAGE_FIXED) if args.image else (GRID, FIXED)

    base = Settings(backend="exact", encoder=args.encoder)
    catalog = load_catalog(base.data_path, base.rating_prior_m)
    targets = also_bought_targets(catalog)
    ids = catalog["uniq_id"].to_numpy()
    queries = np.random.default_rng(1).choice(len(catalog), args.queries, replace=False)
    blocks = build_blocks(catalog, base)

    rows = []
    for values in itertools.product(*grid.values()):
        weights = {**dict(zip(grid, values, strict=True)), **fixed}
        settings = base.model_copy(update={"weights": Settings(weights=weights).weights})
        engine = SimilarityEngine(catalog, blocks, settings)
        nn = {int(q): [engine.row_of(m.product_id) for m in engine.similar(str(ids[q]), 10)] for q in queries}
        rows.append({**weights, **evaluate(nn, catalog, targets)})
    df = pd.DataFrame(rows).sort_values(["also-bought hit@10", "also-bought recall@10"], ascending=False)
    print(df.head(10).to_markdown(index=False, floatfmt=".3f"))
    if args.image:  # the right baseline for the image gain: best config with the image block off
        print("\nbest without image:\n" + df[df["image"] == 0].head(1).to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
