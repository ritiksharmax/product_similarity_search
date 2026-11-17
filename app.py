"""Part 2: FastAPI microservice exposing ``GET /find_similar_products``.

The engine is loaded once at startup (FastAPI lifespan) from artifacts baked into the
image, so requests only do an index lookup. Endpoints:

* ``GET /find_similar_products?product_id=...&num_similar=5`` -> ``["id1", "id2", ...]``
  (add ``explain=true`` for scores and a per-attribute breakdown)
* ``GET /products/{product_id}``                                -> catalogue record
* ``GET /healthz``  liveness  (the process is up)
* ``GET /readyz``   readiness (the index is loaded and the pod can take traffic)

Error mapping: unknown product -> 404, invalid params -> 422 (FastAPI validation),
engine not loaded yet -> 503, anything else -> 500 with a generic body (details only in logs).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from similarity.config import get_settings
from similarity.engine import ProductNotFoundError, SimilarityEngine, set_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("similarity.api")

settings = get_settings()


class SimilarProduct(BaseModel):
    product_id: str
    score: float
    breakdown: dict[str, float] | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    engine = SimilarityEngine.load(settings)
    app.state.engine = engine
    set_engine(engine)  # so the module-level find_similar_products() shares it
    log.info("engine ready: %d products, backend=%s, encoder=%s (%.2fs)",
             len(engine), settings.backend, settings.encoder, time.perf_counter() - t0)
    yield
    app.state.engine = None
    set_engine(None)


app = FastAPI(
    title="Product Similarity Service",
    version="1.0.0",
    description="Find similar Amazon Fashion products by brand, category, name, price, colour, rating and weight.",
    lifespan=lifespan,
)


def _engine(request: Request) -> SimilarityEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="similarity index is not loaded yet")
    return engine


@app.get(
    "/find_similar_products",
    response_model=Union[List[str], List[SimilarProduct]],
    responses={404: {"description": "Unknown product_id"}, 503: {"description": "Index not loaded"}},
)
def get_similar_products(
    request: Request,
    product_id: str = Query(..., min_length=1, max_length=64, description="uniq_id of the query product"),
    num_similar: int = Query(5, ge=1, le=settings.max_num_similar, description="How many similar products to return"),
    explain: bool = Query(False, description="Return scores and per-attribute similarity breakdown"),
):
    engine = _engine(request)
    try:
        matches = engine.similar(product_id.strip(), num_similar, explain=explain)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found") from None
    if explain:
        return [SimilarProduct(product_id=m.product_id, score=m.score, breakdown=m.breakdown) for m in matches]
    return [m.product_id for m in matches]


@app.get("/products/{product_id}")
def get_product(request: Request, product_id: str):
    try:
        return _engine(request).describe(product_id)
    except ProductNotFoundError:
        raise HTTPException(status_code=404, detail=f"product_id '{product_id}' not found") from None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz(request: Request):
    engine = _engine(request)
    return {"status": "ready", "products": len(engine), "backend": settings.backend,
            "encoder": settings.encoder, "cache": engine.cache_info()}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # pragma: no cover - safety net
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
