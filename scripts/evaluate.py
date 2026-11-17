"""Offline quality evaluation: efficient (TF-IDF/LSA) vs accurate (MiniLM) encoders,
and the contribution of each attribute block.

There are no relevance labels, so we use proxy metrics on a fixed random sample of queries:

* also-bought hit@10 / recall@10: ``other_items_customers_buy`` lists product names that
  customers co-purchased. We map them back to catalogue ids by normalised exact name match.
  This is behavioural ground truth that is *not* an input feature. Targets with the same
  name as the query are dropped, so pure duplicate-name matching isn't rewarded.
* category@10 / brand@10: share of neighbours with the query's category / brand.
  This is partly circular when those blocks carry weight, so read it with the ablations.
* price drift: median |log(price_neighbour / price_query)|, where lower means a closer price band.

Queries are sampled with seed 0. ``scripts/tune_weights.py`` tunes on a disjoint seed-1
sample, so the "tuned" rows here are held-out numbers.

    uv run python scripts/evaluate.py --encoders tfidf minilm --queries 2000
"""

from __future__ import annotations

import argparse
import collections
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from similarity.config import DEFAULT_WEIGHTS, HAND_SET_WEIGHTS, Settings
from similarity.data import load_catalog
from similarity.engine import SimilarityEngine
from similarity.features import build_blocks

PRESETS: dict[str, dict[str, float]] = {
    "structured only (no text)": {**DEFAULT_WEIGHTS, "text": 0.0},
    "text only": {"text": 1.0},
    "blend (hand-set v0)": dict(HAND_SET_WEIGHTS),
    "blend (tuned, default)": dict(DEFAULT_WEIGHTS),
}


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def also_bought_targets(catalog: pd.DataFrame) -> dict[int, set[int]]:
    by_name: dict[str, set[int]] = collections.defaultdict(set)
    for i, n in enumerate(catalog["product_name"]):
        by_name[_norm(n)].add(i)
    out = {}
    for i, (name, ab) in enumerate(zip(catalog["product_name"], catalog["also_bought"])):
        own = _norm(name)
        targets = set().union(*(by_name.get(_norm(a), set()) for a in ab if _norm(a) != own)) - {i} if ab else set()
        if targets:
            out[i] = targets
    return out


def evaluate(neighbours: dict[int, list[int]], catalog: pd.DataFrame, targets: dict[int, set[int]]) -> dict[str, float]:
    cat = catalog["category"].to_numpy(dtype=object)
    brand = catalog["brand"].to_numpy(dtype=object)
    logp = np.log(catalog["price"].to_numpy(dtype=float))
    hits, recalls, cat_p, brand_p, drift = [], [], [], [], []
    for q, nn in neighbours.items():
        nn_set = set(nn)
        if q in targets:
            t = targets[q]
            hits.append(bool(t & nn_set))
            recalls.append(len(t & nn_set) / min(len(t), len(nn)))
        if isinstance(cat[q], str):
            cat_p.append(np.mean([cat[j] == cat[q] for j in nn]))
        if isinstance(brand[q], str):
            brand_p.append(np.mean([brand[j] == brand[q] for j in nn]))
        d = np.abs(logp[nn] - logp[q])
        if np.isfinite(d).any():
            drift.append(np.nanmedian(d))
    return {
        "also-bought hit@10": np.mean(hits),
        "also-bought recall@10": np.mean(recalls),
        "category@10": np.mean(cat_p),
        "brand@10": np.mean(brand_p),
        "price drift (|log ratio|)": float(np.nanmedian(drift)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["tfidf", "minilm"])
    ap.add_argument("--transformer-models", nargs="*", default=[],
                    help="extra sentence-transformer models to compare (encoder=minilm with this model)")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("reports/evaluation.md"))
    args = ap.parse_args()

    base = Settings()
    catalog = load_catalog(base.data_path, base.rating_prior_m)
    targets = also_bought_targets(catalog)
    rng = np.random.default_rng(0)
    queries = rng.choice(len(catalog), size=min(args.queries, len(catalog)), replace=False)
    ids = catalog["uniq_id"].to_numpy()
    print(f"{len(catalog)} products, {len(targets)} with also-bought ground truth, {len(queries)} sampled queries")

    rows = []
    # Baseline: random neighbours (sets the floor for every metric).
    rand = {int(q): [int(j) for j in rng.choice(len(catalog), args.k, replace=False) if j != q][: args.k] for q in queries}
    rows.append({"encoder": "-", "weights": "random baseline", "backend": "-", **evaluate(rand, catalog, targets)})

    runs = [(enc, base.model_copy(update={"encoder": enc})) for enc in args.encoders]
    runs += [(m.split("/")[-1], base.model_copy(update={"encoder": "minilm", "minilm_model": m}))
             for m in args.transformer_models]
    for enc, settings in runs:
        t0 = time.perf_counter()
        blocks = build_blocks(catalog, settings)
        encode_s = time.perf_counter() - t0
        print(f"[{enc}] features built in {encode_s:.1f}s (dim={blocks.dim})")
        for preset, weights in PRESETS.items():
            for backend in (["exact", "hnsw"] if "default" in preset else ["exact"]):
                s = settings.model_copy(update={"weights": Settings(weights=weights).weights, "backend": backend})
                engine = SimilarityEngine(catalog, blocks, s)
                t0 = time.perf_counter()
                nn = {int(q): [engine.row_of(m.product_id) for m in engine.similar(str(ids[q]), args.k)] for q in queries}
                ms = (time.perf_counter() - t0) / len(queries) * 1e3
                rows.append({"encoder": enc, "weights": preset, "backend": backend,
                             **evaluate(nn, catalog, targets), "ms/query": ms, "feature build (s)": encode_s})
                print(f"  {preset:28s} {backend:5s} done")

    df = pd.DataFrame(rows)
    table = df.to_markdown(index=False, floatfmt=".3f")
    print("\n" + table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        f"# Evaluation\n\n{len(queries)} random queries, k={args.k}, {len(targets)} products have "
        f"also-bought ground truth.\nGenerated by `scripts/evaluate.py`.\n\n{table}\n"
    )
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
