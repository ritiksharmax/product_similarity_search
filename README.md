# Product Similarity Search: Amazon Fashion 2020

The solution to the [SAP CX technical exercise](docs/EXERCISE.md). Given a product `uniq_id`, return the
`num_similar` most similar products across **brand, category, product name, product image, price, colour,
rating and weight**.

| Part | Status | Where |
|---|---|---|
| 1. `find_similar_products` | ✅ | [`src/similarity/engine.py`](src/similarity/engine.py) |
| 2. FastAPI `GET /find_similar_products`, Docker, Kubernetes | ✅ | [`app.py`](app.py), [`Dockerfile`](Dockerfile), [`k8s/`](k8s) |
| 3. Bonus: vector search at scale | ✅ FAISS HNSW | [`src/similarity/index.py`](src/similarity/index.py), [`reports/benchmark.md`](reports/benchmark.md) |
| Optional: multimodal (text + image) | ✅ CLIP vs ResNet50, measured | [`scripts/embed_images.py`](scripts/embed_images.py), [`reports/image_comparison.md`](reports/image_comparison.md) |

**Headline numbers** (held-out queries, see [Evaluation](#evaluation)):

- **Accuracy:** 60% of products have at least one product *that customers actually co-bought* in their top 10 (random: 0.2%). Adding product photos raised this from 53%.
- **Category match:** 97% of returned neighbours share the query's category.
- **Latency:** at 1M vectors HNSW answers in 0.21 ms vs 57 ms for brute force (recall@10 0.999).

---

## Contents
1. [Quickstart](#quickstart)
2. [Data findings that shaped the design](#data-findings-that-shaped-the-design)
3. [Part 1: the similarity function](#part-1-the-similarity-function)
4. [Evaluation: efficient vs accurate, and images](#evaluation)
5. [Part 2: microservice, Docker, Kubernetes](#part-2-microservice-docker-kubernetes)
6. [Part 3: vector search at scale (HNSW)](#part-3-vector-search-at-scale-hnsw)
7. [Trade-offs, limitations, next steps](#trade-offs-limitations-next-steps)

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11 is fetched automatically).

```bash
cd data && unzip archive.zip && cd ..
uv sync                                    # efficient encoder (TF-IDF); add `--extra minilm` for the accurate one
uv run python scripts/build_artifacts.py   # ~6s: features + HNSW graph -> artifacts/
uv run uvicorn app:app --port 8000

curl 'localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=5'
curl 'localhost:8000/find_similar_products?product_id=26d41bdc1495de290bc8e6062d927729&num_similar=3&explain=true'
```

Image embeddings ship precomputed in `data/image_embeddings.npz` (8 MB, CLIP, PCA-128), so the steps
above need no network. To regenerate them (~30 min download, ~5 min embedding on Apple MPS):

```bash
uv sync --extra images
uv run python scripts/fetch_images.py                                  # -> data/images/ (272 MB, gitignored)
uv run python scripts/embed_images.py --models clip resnet50 --ship clip
```

As a library (the exact signature from the exercise):

```python
from similarity import find_similar_products
find_similar_products("26d41bdc1495de290bc8e6062d927729", 5)   # -> list of 5 uniq_ids
```

Docker and Kubernetes:

```bash
docker build -t product-similarity:1.0.0 .                            # accurate (MiniLM), the default
docker build --build-arg ENCODER=tfidf -t product-similarity:tfidf .  # efficient: faster build, no model download
docker run --rm -p 8000:8000 product-similarity:1.0.0

kind create cluster --name sim && kind load docker-image product-similarity:1.0.0 --name sim
kubectl apply -k k8s/ && kubectl port-forward svc/product-similarity 8000:80
uv run python scripts/smoke_test.py http://localhost:8000
```

Tests and reports:

```bash
uv run pytest                                   # 64 tests: parsers, engine invariants, image block, API, ANN recall
uv run python scripts/smoke_test.py http://localhost:8000   # end-to-end checks against a running service
uv run python scripts/evaluate.py --image-models clip resnet50   # -> reports/evaluation.md (needs --extra minilm)
uv run python scripts/benchmark.py --scales 30000 300000 1000000   # -> reports/benchmark.md
```

---

## Data findings that shaped the design

I profiled all 30,000 rows before writing any similarity code. The dataset is noisy in ways
that would quietly break a naive "cosine over brand/price/weight/rating" approach:

| Field | What the data actually looks like | Decision |
|---|---|---|
| `weight` | **79% are the sentinel `"999999999"`**; the rest are strings (`"200 g"`, `"454 Grams"`, `"1.4 Kg"`, one row of marketing copy) | Parse to grams and treat the sentinel as *missing*. Taken literally, 999,999,999 g would dominate any distance. Weight gets a token weight (0.02) |
| `rating` | Always filled in, **even for products with no reviews** (2,015 unreviewed products show 5.0). Most rated products have 1–2 reviews | **Bayesian shrinkage** `(n·r + m·μ)/(n+m)` using the review count parsed from `Customer_Reviews`: one 5★ review becomes ~4.2, and unreviewed products collapse to the mean |
| `brand` | 27% missing, 6.4k distinct, 116 case variants (`MIZAGO`/`Mizago`), placeholders like `Generic` | Lower-case and trim. `Generic`/`Unbranded` count as missing: two unbranded items are not similar *because* both say "Generic" |
| `sales_price` | 10% missing; ₹39 to ₹9,988, heavily skewed | Compare percentiles of `log(price)`, not raw rupees |
| `colour` | Present for only 20%, free text `Black\|Navy Blue` | Map to a 22-colour base palette, multi-hot |
| category | `sales_rank_in_child_category` key: 219 values, 83% coverage, cleanly curated (`MensT_Shirts`, `WomensSarees`) | Strong product-type signal; one-hot |
| `product_name` | Always present, median 50 chars, carries product type **and usually the brand** | Main text signal |
| images (`medium`) | Every product lists photo URLs; 29,861 (99.5%) download successfully, 134 return 404 | First photo per product → CLIP → PCA-128 image block |
| `asin` | 451 ASINs appear twice (re-crawls with different `uniq_id`) | Never return a re-crawl of the query product as "similar" |
| `other_items_customers_buy` | Names of co-purchased products; 18k products map back to catalogue items | **Not used as a feature** but kept as held-out behavioural ground truth for evaluation |

---

## Part 1: the similarity function

### Weighted block embedding

Each product becomes one dense vector made of **blocks**, one per attribute. Every block is either
unit-length or all-zero:

| Block | Encoding | Cosine between two products |
|---|---|---|
| text | TF-IDF→SVD (LSA, 128-d) *or* MiniLM (384-d) over `product_name` | semantic name similarity |
| image | CLIP ViT-B/32 photo embedding, PCA to 128-d, precomputed offline | visual similarity: style, print, silhouette |
| category | one-hot over 219 child categories | 1 if same category, else 0 |
| brand | fixed pseudo-random unit vector per brand (128-d) | 1 if same brand, else ≈0 |
| colour | multi-hot base palette, L2-normalised | colour-set overlap |
| price / rating / weight | `(cos πq, sin πq)` where `q` = percentile | `cos(π·|Δq|)`: 1 when equal, 0.95 at 10 percentiles apart, −1 at opposite ends |

Scaling block *b* by √w<sub>b</sub> and concatenating gives the key property:

> **⟨x, y⟩ = Σ<sub>b</sub> w<sub>b</sub> · cos<sub>b</sub>(x, y)**

One inner-product search therefore computes an **interpretable, configurable weighted sum of per-attribute
similarities**, and the same vectors work unchanged with brute force and with FAISS HNSW.
`?explain=true` returns that per-attribute breakdown for each result, and a unit test checks that the
breakdown reconstructs the score.

Design choices in that table:

- **Why not a raw feature vector + cosine?** Cosine on concatenated raw features lets wide blocks
  (128 text dims) drown narrow ones (1 price dim), and makes "how much does brand matter?" unanswerable.
  Per-block normalisation turns the weights into the only knob, and an explicit one.
- **Why the arc encoding for numbers?** An inner-product index cannot compute |a − b|. Placing the
  percentile on a half circle makes the dot product a monotone function of distance. Percentiles make
  it robust to price skew and outliers.
- **Why random vectors for brand?** A 6.4k-wide one-hot would bloat a dense ANN index. Random Gaussian
  unit vectors are nearly orthogonal (Johnson–Lindenstrauss): cross-talk between different brands is
  N(0, 1/128), about ±0.09, which the brand weight scales to roughly ±0.01 of the total score.
  Categories (219) are few enough for an exact one-hot.
- **Missing values = zero block = "no evidence".** No imputation guesses. If the *query* lacks an
  attribute, that block adds the same 0 to every candidate and drops out of the ranking. This is the
  fallback mechanism for incomplete features.

### Weights

`text .33 · image .22 · category .15 · price .15 · brand .07 · colour .04 · rating .02 · weight .02`
(sums to 1), overridable at runtime via `SIM_WEIGHTS`. Custom weights are *relative*: they are
rescaled to sum to 1, which never changes a ranking, only the score scale.

These came from a grid search in two stages ([`scripts/tune_weights.py`](scripts/tune_weights.py)) on
one query sample, reported on a **disjoint held-out sample**. Stage 1 tuned the text-era weights
(the same setting ranked top-2 for both text encoders); stage 2 added the image weight, which sits
on a broad plateau (see [Images](#images-the-largest-single-gain)). Things it taught me that I would
not have guessed:

- **Price band matters more than I thought.** Raising the price weight from 0.10 to 0.20 cut price
  drift by a quarter *and* increased co-purchase hits.
- **Brand is partly redundant once text is in.** The brand usually appears in the product name.
- **Photos are worth about as much as names.** CLIP on photos alone finds co-purchases as well as
  TF-IDF on names alone, and the two are complementary.

### Ranking rules

- Sort by score descending. **Ties** go to higher (shrunk) rating, then `uniq_id`, so results are
  fully deterministic. Scores are rounded to 1e-6 first, so float noise from the index can't decide a tie.
- The query product and its **same-ASIN re-crawls are excluded**.
- If `num_similar` exceeds the number of other products, everything is returned. Unknown ids raise
  `ProductNotFoundError`, and `num_similar < 1` raises `ValueError`.
- Every query is searched to a **fixed depth** (`max_num_similar`, 100), and the result is **LRU-cached once
  per product** and sliced to `k`. Two consequences:
  - **Top-k is always a prefix of top-(k+1).** Before this change, HNSW's candidate pool grew with `k`, so
    for 0.4% of products the top 5 differed from the first 5 of the top 100. A client asking for 5 and then
    10 could see the ranking change. This is now 0 of 1,000, and HNSW-vs-exact overlap@5 rose to 0.9996.
  - **Any `k` hits the same cache entry.** The exercise's "caching for frequently accessed products" is
    cheap here because results are deterministic until the next build. In the end-to-end run the
    container served 409 requests with 406 cache hits. The cost is ~0.1 ms extra per uncached query.

### Offline / online split

`scripts/build_artifacts.py` does all the heavy work once: parsing, text encoding, joining the
precomputed image vectors, and HNSW construction. It writes `artifacts/` (`blocks.npy`, `catalog.jsonl`, `hnsw.faiss`, `manifest.json`).
The service only *loads* these. If the configured weights or HNSW parameters differ from the
manifest, the graph is rebuilt in memory from the stored blocks (seconds); the text encoder never
re-runs at serving time.

---

## Evaluation

There are no relevance labels, so "is this similar?" needs proxies. I used three, on 2,000 held-out
random queries (k = 10):

- **also-bought hit@10 / recall@10.** `other_items_customers_buy` lists products customers actually
  co-purchased. Mapped back to catalogue ids, this gives **behavioural ground truth that is never
  used as a feature** (targets with the query's exact name are dropped, so duplicate titles are not rewarded).
- **category@10 / brand@10.** These are partly circular when those blocks carry weight, so read them
  with the ablations.
- **price drift:** median |log(price<sub>neighbour</sub>/price<sub>query</sub>)|. Lower is better.

| encoder | weights | backend | also-bought hit@10 | recall@10 | category@10 | brand@10 | price drift | ms/query | feature build |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| – | random baseline | – | 0.002 | 0.001 | 0.047 | 0.001 | 0.613 | – | – |
| tfidf | structured only (no text/image) | exact | 0.382 | 0.190 | 0.987 | 0.420 | 0.080 | 1.06 | 3.7 s |
| tfidf | text only | exact | 0.305 | 0.180 | 0.688 | 0.271 | 0.316 | 1.05 | 3.7 s |
| tfidf | blend, hand-set v0 | exact | 0.440 | 0.254 | 0.953 | 0.408 | 0.178 | 1.05 | 3.7 s |
| tfidf | blend, tuned (no image) | exact | 0.461 | 0.255 | 0.972 | 0.383 | 0.135 | 1.05 | 3.7 s |
| tfidf | blend, tuned (no image) | **hnsw** | 0.460 | 0.254 | 0.971 | 0.383 | 0.135 | **0.16** | 3.7 s |
| minilm | text only | exact | 0.451 | 0.290 | 0.722 | 0.410 | 0.280 | 1.50 | 45 s |
| minilm | blend, hand-set v0 | exact | 0.526 | 0.312 | 0.961 | 0.471 | 0.162 | 1.49 | 45 s |
| minilm | blend, tuned (no image) | exact | 0.532 | 0.300 | 0.981 | 0.431 | 0.121 | 1.48 | 45 s |
| minilm | blend, tuned (no image) | **hnsw** | 0.531 | 0.299 | 0.981 | 0.430 | 0.121 | **0.17** | 45 s |
| – | CLIP image only | exact | 0.306 | 0.116 | 0.599 | 0.092 | 0.363 | 1.35 | offline |
| tfidf | blend + CLIP image | exact | 0.519 | 0.279 | 0.953 | 0.362 | 0.154 | 1.35 | 3.8 s |
| **minilm** | **blend + CLIP image (default)** | exact | **0.605** | **0.328** | 0.969 | 0.412 | 0.143 | 1.84 | 48 s |
| minilm | blend + CLIP image (default) | **hnsw** | 0.604 | 0.327 | 0.969 | 0.412 | 0.143 | **0.49** | 48 s |

What this says:

1. **Combining attributes beats any single signal.** Structured attributes alone beat TF-IDF text
   alone (0.38 vs 0.31). Blending both beats either by a clear margin.
2. **"Accurate" (MiniLM) beats "efficient" (TF-IDF) by about 15% relative** (0.532 vs 0.461 hit@10).
   It knows `tee ≈ t-shirt` and `kurti ≈ tunic` without seeing them co-occur.
3. **The accurate model costs nothing at serving time.** Queries are *by product id*, so all
   embeddings are precomputed. MiniLM only costs build time (45 s vs 4 s) and a torch install *in the
   Docker builder stage*. That's why the **Docker image defaults to MiniLM**, while the library default
   stays TF-IDF (no heavy dependency, works offline). `--build-arg ENCODER=tfidf` gives a faster build
   with no model download; the serving image is about the same size either way (1.07 vs 1.19 GB),
   because torch never ships in it.
4. **HNSW loses nothing measurable in quality** (≤ 0.001 on every metric) while cutting search time
   about 4–9×. (The HNSW rows with images search to a fixed depth of 100 for prefix-consistent
   results, see [Ranking rules](#ranking-rules), hence 0.49 ms rather than 0.17 ms.)
5. **Images give the largest single gain** (0.532 → 0.605), see the next section.

### Images: the largest single gain

The exercise lists *image* among the attributes, and every product has photo URLs, so the image is a
block like any other ([`reports/image_comparison.md`](reports/image_comparison.md)):

1. **Fetch** ([`scripts/fetch_images.py`](scripts/fetch_images.py)): the first photo per product at a
   256-px rendition (~10 KB instead of ~35 KB), deduplicated by URL, resumable, with retries.
   29,861 of 30,000 products (99.5%) have a photo; 134 URLs return 404.
2. **Embed** ([`scripts/embed_images.py`](scripts/embed_images.py)) with two models, to compare a
   classic CNN against a modern vision-language model:
   - **ResNet-50** (He et al., 2016; ImageNet weights), 2048-d pooled features, the suggested CNN baseline.
   - **CLIP ViT-B/32** (Radford et al., *Learning Transferable Visual Models From Natural Language
     Supervision*, 2021), 512-d. Trained on image-caption pairs, so it encodes *style*
     ("floral printed kurti") rather than ImageNet object classes.
3. **PCA to 128-d**, which costs nothing measurable (see [Dimensionality reduction](#dimensionality-reduction)),
   then ship as an 8 MB `data/image_embeddings.npz`. The Docker build reads that file, so it stays
   offline and reproducible; no vision model runs at serving time.
4. **Fallback:** a product without a usable photo gets a zero image block, so it is ranked on its
   other attributes, the same rule as every missing attribute.

Held-out results (same queries and metrics as above):

| text encoder | image | hit@10 | recall@10 | category@10 | price drift |
|---|---|---:|---:|---:|---:|
| MiniLM | none (previous default) | 0.532 | 0.300 | 0.981 | 0.121 |
| MiniLM | ResNet50 | 0.588 | 0.324 | 0.967 | 0.147 |
| **MiniLM** | **CLIP (default)** | **0.605** | **0.328** | 0.969 | 0.143 |
| TF-IDF | none | 0.461 | 0.255 | 0.972 | 0.135 |
| TF-IDF | CLIP | 0.519 | 0.279 | 0.953 | 0.154 |

- **Images help clearly:** +0.073 hit@10 with MiniLM, +0.058 with TF-IDF, about 4–5 standard errors.
  The small cost is a point of category@10 and some price drift, because a photo says nothing about price.
- **CLIP over ResNet50:** decisive on images alone (hit@10 0.306 vs 0.231, category@10 0.60 vs 0.53).
  In the blend the gap is about one standard error, and the weights were tuned with CLIP.
- **Weight:** on the tuning split hit@10 plateaus at 0.62–0.63 for image weights 0.3–0.7 (grid units).
  The default takes the low end with category kept higher: that keeps category@10 near 0.97 instead
  of 0.92, and leans less on a signal that 0.5% of products lack.

### Are newer embedding models better? Measured: not meaningfully

MiniLM (2021) isn't the current state of the art in sentence embeddings, so I tested three newer
models of the same size (384-d) that score higher on the MTEB benchmark. All used the same held-out
queries; e5 got its required `"query: "` prefix
([`reports/encoder_comparison.md`](reports/encoder_comparison.md)):

| text model | text only: hit@10 | blend: hit@10 | blend: recall@10 | blend: category@10 | blend: price drift | encode time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + SVD | 0.305 | 0.461 | 0.255 | 0.972 | 0.135 | 4 s |
| **all-MiniLM-L6-v2** (2021) | 0.451 | **0.532** | **0.300** | 0.981 | 0.121 | 47 s |
| bge-small-en-v1.5 (2023) | 0.455 | 0.529 | 0.284 | 0.987 | 0.106 | 99 s |
| e5-small-v2 (2022) | 0.463 | 0.519 | 0.271 | 0.987 | 0.097 | 107 s |
| gte-small (2023) | **0.464** | 0.522 | 0.268 | 0.987 | **0.096** | 712 s\* |

*"Blend" uses the text-era tuned weights, without images.* Only 1,173 of the 2,000 queries have co-purchase ground
truth, so hit@10 has a **standard error of about ±0.015**.

- **On text alone, the newer models are only marginally better** (+0.004 to +0.013 hit@10), well within noise.
- **In the full blend, none beats MiniLM.** All are within about one standard error. MiniLM nominally
  leads, but the blend weights were tuned with MiniLM, which slightly favours it.
- **The newer models stick more closely to category and price band** (category@10 0.987, price drift
  down to 0.096) at the cost of co-purchase recall. A different trade-off, not a clear win.
- Encoding takes 2× longer or more, though that only affects the offline build.

**Conclusion:** once text is combined with the structured attributes, the encoder generation stops
mattering much here, because the structured attributes carry much of the signal. MiniLM stays the
default as the fastest model with statistically equal quality. A larger jump would need a different
*kind* of signal (images, or weights learned from behaviour), not a newer text model. The image
block above confirms this: it adds +0.07, five times what any text-model swap moved.
\*gte-small's encode time is an outlier, likely thermal throttling late in the run on a laptop; it
has the same architecture size as the others.

---

## Part 2: microservice, Docker, Kubernetes

### API ([`app.py`](app.py))

| Endpoint | Behaviour |
|---|---|
| `GET /find_similar_products?product_id=…&num_similar=5` | `200` → `["id", …]` as in the spec. Add `explain=true` for `[{product_id, score, breakdown}]` |
| `GET /products/{id}` | catalogue record (handy for eyeballing results) |
| `GET /healthz` | liveness: the process is up |
| `GET /readyz` | readiness: index loaded; also reports backend, encoder and cache stats |

Error mapping, replacing the sample's `raise e`:

| Status | When |
|---|---|
| **404** | unknown `product_id` |
| **422** | `num_similar` missing, not an int, or outside 1–100 (FastAPI validation) |
| **503** | index not loaded yet |
| **500** | anything unexpected. The body is generic and the details go to the logs only |

The engine loads once in the FastAPI `lifespan` hook, never per request. Measured end-to-end over
HTTP: **p50 1.3 ms, p99 2.1 ms** for uncached products, measured against the container limited to 1 CPU
through Docker Desktop's port forwarding on macOS, which adds part of that. A single worker handled
~1,900 req/s with 32 concurrent clients on mostly-cached traffic.

### Docker ([`Dockerfile`](Dockerfile))

The image extends the provided sample. It keeps `uvicorn app:app` on port 8000 and adds:

- **Multi-stage build.** The *builder* installs torch (CPU-only wheel; the default PyPI wheel bundles
  ~2.5 GB of CUDA), unzips the data and runs `build_artifacts.py`. The *runtime* stage copies only
  the serving venv (no torch, no raw data) and the precomputed artifacts. Startup is about 1 s.
  Of the 1.19 GB image, the serving venv is ~480 MB (mostly SciPy, pandas, scikit-learn) and the
  artifacts ~160 MB. Dropping pandas/scikit-learn from the serving path (they are only needed to
  *build* features) would be the next size win.
- **Production hygiene.**
  - Non-root user and a `HEALTHCHECK` on `/readyz`.
  - The dependency layers come from `uv.lock` (`--frozen`), so builds are reproducible.
  - One uvicorn worker per container, scaling out by pods instead (predictable memory per pod).
  - Python 3.11, since 3.10 from the sample is near end-of-life.

### Kubernetes ([`k8s/`](k8s), `kubectl apply -k k8s/`)

- **Deployment.**
  - 2 replicas and a rolling update with `maxUnavailable: 0`.
  - Probes: startup and readiness on `/readyz`, liveness on `/healthz`.
  - Hardened security context: non-root, read-only root filesystem, all capabilities dropped.
  - Resources sized from **measured** RSS (~600 MiB with the MiniLM + CLIP index), not guessed.
- **ConfigMap.** `SIM_BACKEND`, `SIM_HNSW_EF_SEARCH`, cache size and weights are all tunable without
  rebuilding the image. The encoder is *not*: it is baked into the artifacts. Setting a different
  `SIM_ENCODER` fails fast at startup with an error that names the right `--build-arg`.
- **Service** (ClusterIP), **HPA** (CPU 70%, 2 to 10 pods) and a **PodDisruptionBudget**.
- **The index is baked into the image**, so pods are stateless and scaling is trivial. At much larger
  scale I'd move artifacts to object storage or a PVC and load them in an init container, so data
  refreshes don't need a new image.

**Deployed and tested on a local kind cluster** (kind v0.33, Kubernetes v1.37):

| Check | Result |
|---|---|
| `kubectl apply -k k8s/`, rollout | 2/2 pods Ready in ~6 s; HPA and PDB created |
| `scripts/smoke_test.py` through `kubectl port-forward svc/product-similarity` | all checks pass |
| Rolling restart under continuous in-cluster traffic (~75 req/s, client pod hitting the Service DNS) | **25 of 5,535 requests failed** without a `preStop` hook; **0 of 5,815** with it, and 0 of 5,341 again after redeploying the image-enabled build |

The failures are the standard Kubernetes shutdown race: endpoint removal and SIGTERM happen in
parallel, so kube-proxy briefly keeps routing to a pod that is already stopping. The Deployment now
has a 5 s `preStop` sleep, so a terminating pod keeps serving until it has left the Service.
The HPA shows `cpu: <unknown>` on kind, because kind ships without metrics-server; it works on
any cluster that has one.

---

## Part 3: vector search at scale (HNSW)

### Choice: FAISS `IndexHNSWFlat`

- **Algorithm:** Hierarchical Navigable Small World graphs. Yu. A. Malkov & D. A. Yashunin,
  *"Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World
  graphs"*, IEEE TPAMI 42(4), 2020. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- **Library:** FAISS. J. Johnson, M. Douze, H. Jégou, *"Billion-scale similarity search with GPUs"*,
  IEEE Trans. Big Data, 2019. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

**How it works, briefly.** HNSW builds a multi-layer proximity graph. Upper layers are sparse
"express lanes" and the bottom layer holds every point. A query greedily descends from the top,
then runs a beam search of width `efSearch` at the bottom. The cost grows about logarithmically
with N instead of linearly.

**Why HNSW over the alternatives:**

| Option | Why not (here) |
|---|---|
| Brute force | Exact and simple. It is the *right* choice at 30k (1 ms) and stays as the `exact` backend and ground truth, but cost grows linearly (57 ms p50 at 1M, d = 883) |
| Annoy (random-projection trees) | Lower recall per unit of latency than HNSW on ann-benchmarks; static index |
| IVF / IVF-PQ (FAISS) | Needs a k-means training step and `nprobe` tuning. PQ trades recall for memory, which we don't need below ~10⁸ vectors |
| ScaNN / DiskANN | Excellent, but heavier operationally for this size |

HNSW is consistently at or near the top of [ann-benchmarks](https://ann-benchmarks.com). It needs
**no training**, supports **incremental inserts** (new products can be added without a rebuild),
and exposes one intuitive recall/latency knob (`efSearch`). Costs: memory for the graph (~2·M·4 bytes
per vector, about 8 MB here) and no true deletes, so delisted products are filtered or handled by
periodic rebuilds.

### An experiment that changed my implementation: inner product vs. L2 reduction

Our score is a raw **inner product**, and vectors have unequal norms because missing attributes are
zero blocks. HNSW's routing assumes a metric, so I first implemented the textbook MIPS→NN reduction
(Bachrach et al., RecSys 2014). It appends `√(R² − ‖x‖²)` to every item, which makes L2 nearest
neighbour *exactly* equal to maximum inner product, and a unit test proves that under brute force.

Measured, though, the reduction gave **lower** recall than running HNSW directly on inner product:

| efSearch | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| native inner product (`ip`) | **0.990** | **0.997** | **0.998** | **0.999** |
| L2 reduction (`l2`) | 0.972 | 0.988 | 0.995 | 0.997 |

The reduction puts every point on a sphere of radius R, which compresses the distance contrast
that greedy routing relies on. This matches Morozov & Babenko, *"Non-metric Similarity Graphs for
Maximum Inner Product Search"*, NeurIPS 2018. Our norms only vary between 0.69 and 1.0, so native
inner product wins. It is the default, and the reduction stays available as `SIM_HNSW_METRIC=l2`.

### Benchmark ([`reports/benchmark.md`](reports/benchmark.md))

Single-query latency on an Apple-silicon laptop with the **shipped vectors** (MiniLM text + CLIP
image + structured blocks, d = 883), M = 32, efConstruction = 200, k = 10. Sizes above 30k are the
real vectors jittered with noise, which keeps the real cluster structure; i.i.d. random vectors
would make ANN look unrealistically bad.

| N | index | efSearch | recall@10 | p50 ms | p99 ms | build | memory |
|---:|---|---:|---:|---:|---:|---:|---:|
| 30k | exact | – | 1.000 | 1.69 | 2.15 | – | 101 MiB |
| 30k | HNSW | 64 | 0.998 | 0.24 | 0.43 | 2.8 s | 108 MiB |
| 300k | exact | – | 1.000 | 17.4 | 25.4 | – | 1.0 GiB |
| 300k | HNSW | 64 | 0.995 | 0.26 | 0.54 | 64 s | 1.1 GiB |
| 1M | exact | – | 1.000 | **56.9** | **69.7** | – | 3.3 GiB |
| 1M | HNSW | 64 | 0.999 | **0.21** | **0.48** | 249 s | 3.5 GiB |
| 1M | HNSW | 128 | 1.000 | 0.36 | 0.65 | 249 s | 3.5 GiB |

Brute force grows linearly (57 ms p50 at 1M); HNSW stays sub-millisecond and roughly flat in N,
about **265× faster at 1M** with recall@10 0.999. `efSearch = 64` is the default: at 30k it is within
0.001 of exact on every quality metric above. An earlier run with the 499-d TF-IDF vectors showed the
same shape (1M: 43.8 ms exact vs 0.12 ms HNSW); the higher dimension mainly costs memory and build
time, which is where the PQ compression below would come in.

### Dimensionality reduction

The README suggested PCA or t-SNE before similarity:

- **TF-IDF path:** already reduced. The sparse ~100k-term matrix becomes 128 dims via truncated SVD
  (LSA), the sparse-friendly equivalent of PCA.
- **Image path: PCA, measured.** CLIP (512-d) and ResNet50 (2048-d) image embeddings are reduced to
  128 dims with PCA before they become the image block. Image-only retrieval on the same held-out
  queries shows PCA-128 costs nothing measurable, while the file shrinks 4× (CLIP) / 16× (ResNet):

  | dims | CLIP hit@10 | CLIP category@10 | ResNet50 hit@10 | ResNet50 category@10 |
  |---:|---:|---:|---:|---:|
  | full (512 / 2048) | 0.299 | 0.599 | 0.235 | 0.530 |
  | 256 | 0.308 | 0.600 | 0.233 | 0.526 |
  | **128 (shipped)** | **0.306** | **0.599** | 0.230 | 0.526 |
  | 64 | 0.292 | 0.597 | 0.217 | 0.519 |
  | 32 | 0.255 | 0.589 | 0.209 | 0.507 |

  PCA-128 keeps 85% (CLIP) / 82% (ResNet50) of the variance. Quality only falls off below 64 dims.
- **t-SNE:** deliberately not used. It is a visualisation method: non-parametric (no transform for
  new products), it doesn't preserve global distances, and it is O(N²)-ish.
- **At 100M+ products:** the next step would be PCA to ~64 dims on the text block plus
  `IndexHNSWPQ` / `IndexIVFPQ` to compress vectors 8–16× (Jégou et al., *Product Quantization for
  Nearest Neighbor Search*, TPAMI 2011), validated with the same recall harness.

---

## Trade-offs, limitations, next steps

- **Image embeddings are a snapshot.** They are computed offline from one photo per product and
  shipped as a file. New products need `fetch_images.py` + `embed_images.py` (incremental runs reuse
  downloaded files). In production this would be an embedding job triggered on catalogue updates.
  Only the first photo is used; averaging all photos, or a fashion-tuned CLIP (e.g. FashionCLIP),
  are the obvious next experiments.
- **Variants cluster together.** Identical titles in different sizes or colours (e.g. several
  "Pepe Jeans Women's Solid T-Shirt") rank at the top, which is good for "more like this" and less
  good for discovery. A product page might want to collapse variants (group by normalised title) or
  apply MMR diversification.
- **Weights are hand-tuned on a proxy.** With real click or co-purchase logs, the natural next step
  is to *learn* the block weights, e.g. logistic regression over per-block cosines, which the
  `explain` breakdown already produces.
- **Brand cross-talk.** Random brand vectors add ±0.01 score noise between different brands. A wider
  brand block, or a sparse exact-match re-rank of the top 50, would remove it.
- **Catalogue updates.** HNSW supports adding vectors online, but TF-IDF's SVD basis is fit on the
  catalogue, so new products would need an occasional refit. MiniLM has no such issue, which is
  another point in its favour.
- **Memory.** Vectors are held three times (unweighted for `explain`, weighted, FAISS's copy). That's
  fine at 30k (~600 MiB RSS); at scale `explain` would derive from the weighted copy.

### Project layout

```
app.py                       FastAPI service (Part 2)
src/similarity/
  config.py                  settings (env SIM_*), weights and their rationale
  data.py                    loading + cleaning (sentinels, units, shrinkage, brands, colours)
  features.py                block embedding, TF-IDF/LSA + MiniLM encoders, image block
  index.py                   ExactIndex, HnswIndex (FAISS), ip-vs-L2 notes
  engine.py                  SimilarityEngine + find_similar_products (Part 1)
scripts/                     build_artifacts, fetch_images, embed_images, evaluate, tune_weights,
                             benchmark, smoke_test
tests/                       64 tests (fixture covers sentinel weight, dup ASIN, Generic brand, …)
reports/                     generated evaluation, benchmark, encoder and image comparisons
data/archive.zip             the dataset; data/image_embeddings.npz: precomputed CLIP vectors (PCA-128)
k8s/                         deployment, service, configmap, HPA + PDB, kustomization
docs/EXERCISE.md             the original exercise statement
```
