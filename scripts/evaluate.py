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
from similarity.features import BlockMatrix, build_blocks, load_image_block

PRESETS: dict[str, dict[str, float]] = {
    "structured only (no text/image)": {**DEFAULT_WEIGHTS, "text": 0.0, "image": 0.0},
    "text only": {"text": 1.0},
    "image only": {"image": 1.0},
    "blend (hand-set v0)": dict(HAND_SET_WEIGHTS),
    # The previous default (tuned before the image block existed): the "no image" baseline.
    "blend (tuned, no image)": {"text": 0.30, "category": 0.20, "brand": 0.10, "price": 0.20,
                                "colour": 0.05, "rating": 0.03, "weight": 0.02},
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


def with_image_model(catalog, settings, text_blocks, image_model):
    """Swap in ``data/image_embeddings_<model>.npz`` as the image block, reusing the (slow) text block."""
    if image_model is None:
        return text_blocks, (text_blocks.info.get("image") or {}).get("model") or "none"
    img, _ = load_image_block(
        catalog, settings.model_copy(update={"image_embeddings_path": Path(f"data/image_embeddings_{image_model}.npz")}))
    data = text_blocks.data.copy()
    data[:, text_blocks.slices["image"]] = img
    return BlockMatrix(data=data, slices=text_blocks.slices, info={"image": {"model": image_model}}), image_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["tfidf", "minilm"])
    ap.add_argument("--transformer-models", nargs="*", default=[],
                    help="extra sentence-transformer models to compare (encoder=minilm with this model)")
    ap.add_argument("--image-models", nargs="+", default=[None],
                    help="compare image embedding files data/image_embeddings_<model>.npz "
                         "(default: the shipped data/image_embeddings.npz)")
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
    rows.append({"encoder": "-", "image": "-", "weights": "random baseline", "backend": "-", **evaluate(rand, catalog, targets)})

    runs = [(enc, base.model_copy(update={"encoder": enc})) for enc in args.encoders]
    runs += [(m.split("/")[-1], base.model_copy(update={"encoder": "minilm", "minilm_model": m}))
             for m in args.transformer_models]
    image_only_presets = {"image only", "blend (tuned, default)"}
    for enc, settings in runs:
        t0 = time.perf_counter()
        text_blocks = build_blocks(catalog, settings)
        encode_s = time.perf_counter() - t0
        print(f"[{enc}] features built in {encode_s:.1f}s (dim={text_blocks.dim})")
        for n_img, image_model in enumerate(args.image_models):
            blocks, img_label = with_image_model(catalog, settings, text_blocks, image_model)
            for preset, weights in PRESETS.items():
                if n_img > 0 and preset not in image_only_presets:
                    continue  # presets without an image weight are identical across image models
                if preset == "image only" and img_label == "none":
                    continue  # no image embeddings available
                for backend in (["exact", "hnsw"] if "default" in preset else ["exact"]):
                    w = Settings(weights=weights).weights
                    s = settings.model_copy(update={"weights": w, "backend": backend})
                    engine = SimilarityEngine(catalog, blocks, s)
                    t0 = time.perf_counter()
                    nn = {int(q): [engine.row_of(m.product_id) for m in engine.similar(str(ids[q]), args.k)]
                          for q in queries}
                    ms = (time.perf_counter() - t0) / len(queries) * 1e3
                    rows.append({"encoder": enc, "image": img_label if w["image"] > 0 else "-", "weights": preset,
                                 "backend": backend, **evaluate(nn, catalog, targets), "ms/query": ms,
                                 "feature build (s)": encode_s})
                    print(f"  {img_label:9s} {preset:32s} {backend:5s} done")

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
