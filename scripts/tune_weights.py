"""Small grid search over the main block weights (tuning split, seed 1).

Maximises also-bought hit@10 (see evaluate.py for the metric). Results are reported on the
disjoint seed-0 split by ``scripts/evaluate.py``, so the published numbers aren't tuned on.

    uv run python scripts/tune_weights.py --encoder tfidf
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
FIXED = {"colour": 0.05, "rating": 0.03, "weight": 0.02}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tfidf", choices=["tfidf", "minilm"])
    ap.add_argument("--queries", type=int, default=1000)
    args = ap.parse_args()

    base = Settings(backend="exact", encoder=args.encoder)
    catalog = load_catalog(base.data_path, base.rating_prior_m)
    targets = also_bought_targets(catalog)
    ids = catalog["uniq_id"].to_numpy()
    queries = np.random.default_rng(1).choice(len(catalog), args.queries, replace=False)
    blocks = build_blocks(catalog, base)

    rows = []
    for values in itertools.product(*GRID.values()):
        weights = {**dict(zip(GRID, values)), **FIXED}
        engine = SimilarityEngine(catalog, blocks, base.model_copy(update={"weights": Settings(weights=weights).weights}))
        nn = {int(q): [engine.row_of(m.product_id) for m in engine.similar(str(ids[q]), 10)] for q in queries}
        rows.append({**weights, **evaluate(nn, catalog, targets)})
    df = pd.DataFrame(rows).sort_values(["also-bought hit@10", "also-bought recall@10"], ascending=False)
    print(df.head(10).to_markdown(index=False, floatfmt=".3f"))


if __name__ == "__main__":
    main()
