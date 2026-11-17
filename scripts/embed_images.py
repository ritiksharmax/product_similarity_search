"""Embed the downloaded product images, reduce with PCA, and write the image-block input file.

    uv sync --extra images
    uv run python scripts/fetch_images.py                       # once: data/images/
    uv run python scripts/embed_images.py --models clip resnet50 --ship clip

Per model this writes ``data/image_embeddings_<model>.npz`` (uniq_id + float16 PCA vectors + metadata).
``--ship`` copies the chosen one to ``data/image_embeddings.npz``, which the build (and Docker image)
reads. Raw full-dimension embeddings are cached in ``data/images/raw_<model>.npz`` (gitignored) so
the PCA trade-off can be measured.

Models:
* ``clip``     - CLIP ViT-B/32 image tower (Radford et al., 2021), 512-d. Trained contrastively on
                 image-caption pairs, so it encodes *style/semantics* ("floral kurti", "slim jeans").
* ``resnet50`` - ImageNet ResNet-50 (He et al., 2016), 2048-d global-average-pooled features.
                 The classic CNN baseline the exercise suggests; strong on shape/texture.

This script deliberately does **not** import faiss: with only torch's OpenMP runtime loaded, torch
can use all cores (see the deadlock note in ``features.MiniLmEncoder``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA

IMAGE_DIR = Path("data/images")
OUT_DIR = Path("data")


def load_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def embed_clip(paths: list[Path], batch: int, device: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("clip-ViT-B-32", device=device)
    out = []
    for i in range(0, len(paths), batch):
        imgs = [load_rgb(p) or Image.new("RGB", (224, 224)) for p in paths[i : i + batch]]
        out.append(model.encode(imgs, batch_size=batch, convert_to_numpy=True, show_progress_bar=False))
        if (i // batch) % 40 == 0:
            print(f"    clip {i + len(imgs)}/{len(paths)}", flush=True)
    return np.vstack(out).astype(np.float32)


def embed_resnet50(paths: list[Path], batch: int, device: str) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # keep the 2048-d pooled features, drop the classifier
    model.eval().to(device)
    preprocess = weights.transforms()
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = [load_rgb(p) or Image.new("RGB", (224, 224)) for p in paths[i : i + batch]]
            out.append(model(torch.stack([preprocess(im) for im in imgs]).to(device)).cpu().numpy())
            if (i // batch) % 40 == 0:
                print(f"    resnet50 {i + len(imgs)}/{len(paths)}", flush=True)
    return np.vstack(out).astype(np.float32)


EMBEDDERS = {"clip": embed_clip, "resnet50": embed_resnet50}


def pick_device(requested: str) -> str:
    """Offline step, so use an accelerator when there is one (Apple MPS: ResNet50 0.57 s vs 12.5 s
    per 64 images on CPU, measured on an M-series laptop). Results are the same up to float noise."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["clip"], choices=sorted(EMBEDDERS))
    ap.add_argument("--dim", type=int, default=128, help="PCA output dimension")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    ap.add_argument("--ship", choices=sorted(EMBEDDERS), help="copy this model's file to data/image_embeddings.npz")
    args = ap.parse_args()

    with open(IMAGE_DIR / "index.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["file"]]
    files = sorted({r["file"] for r in rows})
    # Drop files PIL cannot decode, so they become "missing" rather than a black-square embedding.
    readable = [f for f in files if load_rgb(IMAGE_DIR / f) is not None]
    print(f"{len(rows)} products with an image, {len(files)} distinct files, {len(readable)} decodable")
    file_row = {f: i for i, f in enumerate(readable)}
    product_ids = np.array([r["uniq_id"] for r in rows if r["file"] in file_row])
    product_file = np.array([file_row[r["file"]] for r in rows if r["file"] in file_row])

    for name in args.models:
        raw_path = IMAGE_DIR / f"raw_{name}.npz"
        t0 = time.perf_counter()
        if raw_path.exists() and list(np.load(raw_path)["files"]) == readable:
            raw = np.load(raw_path)["emb"]
            seconds = float(np.load(raw_path)["seconds"])
            print(f"{name}: reusing cached raw embeddings {raw.shape}")
        else:
            print(f"{name}: embedding {len(readable)} images")
            device = pick_device(args.device)
            print(f"{name}: device={device}")
            raw = EMBEDDERS[name]([IMAGE_DIR / f for f in readable], args.batch, device)
            seconds = time.perf_counter() - t0
            np.savez(raw_path, emb=raw, files=np.array(readable), seconds=seconds)

        # PCA is fitted on distinct images (re-crawled duplicates would otherwise get extra say),
        # on L2-normalised vectors so it preserves cosine geometry.
        unit = l2(raw)
        pca = PCA(n_components=args.dim, random_state=0).fit(unit)
        reduced = l2(pca.transform(unit)).astype(np.float16)
        variance = float(pca.explained_variance_ratio_.sum())
        meta = {
            "model": name, "raw_dim": int(raw.shape[1]), "pca_dim": args.dim,
            "pca_explained_variance": round(variance, 4), "images": len(readable),
            "products": int(len(product_ids)), "embed_seconds": round(seconds, 1),
        }
        out = OUT_DIR / f"image_embeddings_{name}.npz"
        np.savez_compressed(out, uniq_id=product_ids, vectors=reduced[product_file], meta=json.dumps(meta))
        print(f"{name}: {json.dumps(meta)} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if args.ship:
        shutil.copyfile(OUT_DIR / f"image_embeddings_{args.ship}.npz", OUT_DIR / "image_embeddings.npz")
        print(f"shipped {args.ship} -> data/image_embeddings.npz")


if __name__ == "__main__":
    main()
