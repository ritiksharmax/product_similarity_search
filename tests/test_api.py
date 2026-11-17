import pytest
from fastapi.testclient import TestClient

import app as app_module
from similarity.engine import SimilarityEngine

from conftest import make_settings


@pytest.fixture(scope="module")
def client(tmp_path_factory, catalog):
    s = make_settings(tmp_path_factory.mktemp("api"), backend="hnsw")
    SimilarityEngine.build(s, catalog).save(s.artifacts_dir)
    app_module.settings = s
    with TestClient(app_module.app) as c:
        yield c


def test_similar_products_ok(client):
    r = client.get("/find_similar_products", params={"product_id": "ts1", "num_similar": 3})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 3 and "ts1" not in body


def test_explain(client):
    r = client.get("/find_similar_products", params={"product_id": "ts1", "num_similar": 2, "explain": True})
    assert r.status_code == 200
    assert {"product_id", "score", "breakdown"} <= set(r.json()[0])


def test_unknown_product_404(client):
    r = client.get("/find_similar_products", params={"product_id": "nope", "num_similar": 3})
    assert r.status_code == 404


@pytest.mark.parametrize("params", [
    {"product_id": "ts1", "num_similar": 0},
    {"product_id": "ts1", "num_similar": 101},
    {"product_id": "ts1", "num_similar": "abc"},
    {"num_similar": 3},
])
def test_validation_422(client, params):
    assert client.get("/find_similar_products", params=params).status_code == 422


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["products"] == 12


def test_product_lookup(client):
    assert client.get("/products/sa1").json()["category"] == "WomensSarees"
    assert client.get("/products/zzz").status_code == 404
