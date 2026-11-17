"""Download one product image per catalogue item (offline step for the image block).

    uv run python scripts/fetch_images.py                  # -> data/images/*.jpg + data/images/index.csv

* Uses the first ``medium`` URL per product (``load_catalog`` -> ``image_url``).
* Asks Amazon's CDN for a 256-px rendition (``…/ID._SL256_.jpg``, ~10 KB instead of ~35 KB);
  falls back to the original URL if that fails. CLIP/ResNet resize to 224 px anyway.
* Deduplicates by URL (re-crawls and variants share images), is resumable (existing files are
  skipped), and retries transient failures. A product whose image cannot be fetched simply gets a
  zero image block later: the same "no evidence" fallback as every other missing attribute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from similarity.config import get_settings
from similarity.data import load_catalog

USER_AGENT = "Mozilla/5.0 (product-similarity research script)"


def resized(url: str, size: int = 256) -> str:
    stem, dot, ext = url.rpartition(".")
    return f"{stem}._SL{size}_.{ext}" if dot and "._" not in url.rsplit("/", 1)[-1] else url


def file_for(url: str, out_dir: Path) -> Path:
    return out_dir / (hashlib.blake2b(url.encode(), digest_size=10).hexdigest() + ".jpg")


def fetch(url: str, dest: Path, retries: int = 2) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    last = "error"
    for candidate in (resized(url), url):
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(candidate, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=15) as r:
                    body = r.read()
                if not body.startswith(b"\xff\xd8") and not body.startswith(b"\x89PNG"):
                    last = "not-an-image"
                    break
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(body)
                tmp.replace(dest)  # atomic: an interrupted run never leaves half a file
                return "ok"
            except urllib.error.HTTPError as e:
                last = f"http-{e.code}"
                if e.code in (403, 404, 410):
                    break  # permanent; try the next candidate
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = type(e).__name__
            time.sleep(0.5 * (attempt + 1))
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/images"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N products (smoke run)")
    args = ap.parse_args()

    settings = get_settings()
    catalog = load_catalog(settings.data_path, settings.rating_prior_m)
    if args.limit:
        catalog = catalog.head(args.limit)
    args.out.mkdir(parents=True, exist_ok=True)

    urls = sorted({u for u in catalog["image_url"] if isinstance(u, str) and u.startswith("http")})
    print(f"{len(catalog)} products, {len(urls)} distinct image URLs")
    t0 = time.perf_counter()
    status: dict[str, str] = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (url, st) in enumerate(zip(urls, pool.map(lambda u: fetch(u, file_for(u, args.out)), urls)), 1):
            status[url] = st
            if i % 2000 == 0:
                print(f"  {i}/{len(urls)}  {time.perf_counter() - t0:.0f}s", flush=True)

    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uniq_id", "file", "status"])
        for pid, url in zip(catalog["uniq_id"], catalog["image_url"]):
            st = status.get(url, "no-url") if isinstance(url, str) else "no-url"
            ok = st in ("ok", "cached")
            w.writerow([pid, file_for(url, args.out).name if ok else "", st])

    counts: dict[str, int] = {}
    for st in status.values():
        counts[st] = counts.get(st, 0) + 1
    print(f"done in {time.perf_counter() - t0:.0f}s: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
