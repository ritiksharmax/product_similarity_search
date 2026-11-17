"""End-to-end smoke test against a *running* service (local uvicorn, Docker, or k8s port-forward).

    uv run python scripts/smoke_test.py http://localhost:8000
    kubectl port-forward svc/product-similarity 8000:80 & python scripts/smoke_test.py http://localhost:8000

Standard library only, so it can also run from a bare ``python:3.11`` container. Exits non-zero
if any check fails. The product ids are real ``uniq_id`` values from the bundled dataset.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TSHIRT = "410c62298852e68f34c35560f2311e5a"      # men's slim-fit T-shirt, brand missing
DUP_ASIN = ("4f56df7404b88daa7c4469e70ebe25e4",   # same ASIN crawled twice:
            "f939828340e6250f76cda0c4f248de07")   # the twin must never be "similar"
SPARSE = "816b849b974b45b0caea83d02c7648e0"      # no brand, price or category (name-only fallback)

FAILURES: list[str] = []


def call(base: str, path: str, **params) -> tuple[int, object]:
    url = base.rstrip("/") + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -> {detail}" if not ok and detail != "" else ""))
    if not ok:
        FAILURES.append(name)


def main(base: str) -> int:
    print(f"smoke test against {base}")

    print("probes")
    s, body = call(base, "/healthz")
    check("healthz 200", s == 200, (s, body))
    s, body = call(base, "/readyz")
    check("readyz 200 with products", s == 200 and body.get("products", 0) > 1000, (s, body))

    print("happy path")
    s, ids = call(base, "/find_similar_products", product_id=TSHIRT)
    check("default num_similar=5", s == 200 and isinstance(ids, list) and len(ids) == 5, (s, ids))
    check("all ids are strings, unique, not self",
          all(isinstance(i, str) for i in ids) and len(set(ids)) == len(ids) and TSHIRT not in ids, ids)
    s, ids2 = call(base, "/find_similar_products", product_id=TSHIRT)
    check("deterministic across calls", ids == ids2, (ids, ids2))
    names = [call(base, f"/products/{i}")[1]["product_name"] for i in ids]
    check("T-shirt neighbours are T-shirts", sum("shirt" in n.lower() for n in names) >= 4, names)
    for k in (1, 100):
        s, r = call(base, "/find_similar_products", product_id=TSHIRT, num_similar=k)
        check(f"num_similar={k} returns exactly {k} unique", s == 200 and len(set(r)) == k, (s, len(r)))
    s, r = call(base, "/find_similar_products", product_id=f"  {TSHIRT} ")
    check("surrounding whitespace in id tolerated", s == 200 and r == ids, (s, r))

    print("data edge cases")
    s, r = call(base, "/find_similar_products", product_id=DUP_ASIN[0], num_similar=20)
    check("same-ASIN twin excluded", s == 200 and DUP_ASIN[1] not in r and DUP_ASIN[0] not in r, r)
    s, r = call(base, "/find_similar_products", product_id=SPARSE, num_similar=5)
    check("sparse product (no brand/price/category) still answers", s == 200 and len(r) == 5, (s, r))

    print("explain")
    s, r = call(base, "/find_similar_products", product_id=TSHIRT, num_similar=3, explain="true")
    ok = s == 200 and len(r) == 3 and all({"product_id", "score", "breakdown"} <= set(m) for m in r)
    check("explain returns score + breakdown", ok, r)
    if ok:
        check("explain ids match plain ids", [m["product_id"] for m in r] == ids[:3], r)
        scores = [m["score"] for m in r]
        check("scores non-increasing", scores == sorted(scores, reverse=True), scores)

    print("errors")
    s, _ = call(base, "/find_similar_products", product_id="does-not-exist")
    check("unknown id -> 404", s == 404, s)
    s, _ = call(base, "/products/does-not-exist")
    check("unknown product record -> 404", s == 404, s)
    for bad in (0, -1, 101, "abc", 2.5):
        s, _ = call(base, "/find_similar_products", product_id=TSHIRT, num_similar=bad)
        check(f"num_similar={bad!r} -> 422", s == 422, s)
    s, _ = call(base, "/find_similar_products")
    check("missing product_id -> 422", s == 422, s)
    s, _ = call(base, "/find_similar_products", product_id="")
    check("empty product_id -> 422", s == 422, s)
    s, _ = call(base, "/find_similar_products", product_id="   ")
    check("blank product_id -> 404/422 (not 500)", s in (404, 422), s)
    s, _ = call(base, "/find_similar_products", product_id="x" * 65)
    check("over-long product_id -> 422", s == 422, s)
    s, _ = call(base, "/find_similar_products", product_id="../../etc/passwd'; DROP--")
    check("hostile id -> 404", s == 404, s)

    print("concurrency")
    def one(k: int) -> list[str]:
        return call(base, "/find_similar_products", product_id=TSHIRT, num_similar=1 + k % 100)[1]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(32) as pool:
        results = list(pool.map(one, range(400)))
    dt = time.perf_counter() - t0
    reference = call(base, "/find_similar_products", product_id=TSHIRT, num_similar=100)[1]
    consistent = all(r == reference[: len(r)] for r in results)
    check(f"400 concurrent mixed-k requests consistent ({400 / dt:.0f} req/s)", consistent)

    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
