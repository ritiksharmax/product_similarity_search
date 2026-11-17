# Image similarity: CLIP vs ResNet50

Pipeline: `scripts/fetch_images.py` (first `medium` photo per product, 256-px rendition) →
`scripts/embed_images.py` (embed, L2-normalise, PCA to 128-d) → `data/image_embeddings.npz` →
`image` block in the weighted block embedding (`src/similarity/features.py`).

## Coverage

| | count |
|---|---:|
| products | 30,000 |
| distinct image URLs | 29,385 |
| downloaded | 29,251 (134 URLs return HTTP 404) |
| products with an image vector | **29,861 (99.5%)** |
| download size / time | 272 MB / 29 min (24 threads) |

Products without a photo get a zero image block and are ranked on their other attributes.

## Embedding cost (Apple-silicon laptop, MPS)

| model | raw dim | embed time (29,251 images) | PCA-128 explained variance | file size |
|---|---:|---:|---:|---:|
| CLIP ViT-B/32 (Radford et al., 2021) | 512 | 257 s | 0.852 | 7.8 MB |
| ResNet-50, ImageNet V2 weights (He et al., 2016) | 2048 | 296 s | 0.825 | 7.9 MB |

On CPU, ResNet50 measured 12.5 s per 64 images versus 0.57 s on MPS, so the script uses an
accelerator when one exists. This step is offline only; serving never runs a vision model.

## Image-only retrieval vs PCA dimension

2,000 held-out queries (seed 0), 1,173 with also-bought ground truth, k = 10, exact search,
same-ASIN re-crawls excluded.

| dims | CLIP hit@10 | CLIP category@10 | ResNet50 hit@10 | ResNet50 category@10 |
|---:|---:|---:|---:|---:|
| full (512 / 2048) | 0.299 | 0.599 | 0.235 | 0.530 |
| 256 | 0.308 | 0.600 | 0.233 | 0.526 |
| **128 (shipped)** | **0.306** | **0.599** | 0.230 | 0.526 |
| 64 | 0.292 | 0.597 | 0.217 | 0.519 |
| 32 | 0.255 | 0.589 | 0.209 | 0.507 |

PCA-128 is free in quality and 4× / 16× smaller. CLIP's photos alone are as predictive of
co-purchases as TF-IDF product names alone (0.306 vs 0.305).

## In the blend (held-out, `scripts/evaluate.py --image-models clip resnet50`)

| text encoder | image | weights | hit@10 | recall@10 | category@10 | brand@10 | price drift |
|---|---|---|---:|---:|---:|---:|---:|
| MiniLM | - | tuned, no image (previous default) | 0.532 | 0.300 | 0.981 | 0.431 | 0.121 |
| MiniLM | ResNet50 | tuned default | 0.588 | 0.324 | 0.967 | 0.403 | 0.147 |
| **MiniLM** | **CLIP** | **tuned default** | **0.605** | **0.328** | 0.969 | 0.412 | 0.143 |
| TF-IDF | - | tuned, no image | 0.461 | 0.255 | 0.972 | 0.383 | 0.135 |
| TF-IDF | ResNet50 | tuned default | 0.506 | 0.278 | 0.950 | 0.353 | 0.155 |
| TF-IDF | CLIP | tuned default | 0.519 | 0.279 | 0.953 | 0.362 | 0.154 |

Standard error of hit@10 is about ±0.015, so:

- **Images help clearly:** +0.073 (MiniLM) and +0.058 (TF-IDF) hit@10, about 4–5 standard errors.
  The cost is small: category@10 drops ~1 point and price drift rises slightly, because a photo
  says nothing about price.
- **CLIP over ResNet50:** decisive on images alone (0.306 vs 0.231 hit@10, 0.60 vs 0.53
  category@10). In the blend the gap is about one standard error, and the weights were tuned with
  CLIP, which slightly favours it. CLIP is trained on image-caption pairs, so it encodes style
  ("floral printed kurti") rather than ImageNet object classes, which fits fashion better.
- **Shipped:** CLIP, image weight 0.22 of the total.

## Weight tuning (seed-1 tuning split, MiniLM, CLIP)

Best configuration without the image block: hit@10 0.559. With it, hit@10 sits on a plateau of
0.62–0.63 for image weights 0.3–0.7 (grid units). The default takes the low end of the plateau with
category kept at 0.20, which keeps category@10 at ~0.97 instead of ~0.92.
