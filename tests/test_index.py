import numpy as np
import pytest

from similarity.index import ExactIndex, HnswIndex


def _recall(a, b):
    return np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)])


def test_exact_index_matches_numpy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 16)).astype(np.float32)
    s, i = ExactIndex(X).search(X[:3], 5)
    expected = np.argsort(-(X[:3] @ X.T), axis=1)[:, :5]
    assert (i == expected).all()
    assert (np.diff(s, axis=1) <= 0).all()


def _varying_norm_data(n=5000, d=32, seed=1):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((n, d)) * rng.uniform(0.2, 2.0, (n, 1))).astype(np.float32)
    return X, X[rng.choice(n, 200, replace=False)]


def test_hnsw_inner_product_high_recall_on_varying_norms():
    X, Q = _varying_norm_data()
    _, exact = ExactIndex(X).search(Q, 10)
    _, approx = HnswIndex.build(X, ef_search=64, metric="ip").search(Q, 10)
    assert _recall(exact, approx) >= 0.95


def test_l2_reduction_is_exact_under_brute_force():
    """Bachrach et al. transform: L2-NN on augmented vectors == max inner product."""
    import faiss

    X, Q = _varying_norm_data()
    _, exact = ExactIndex(X).search(Q, 10)
    flat = faiss.IndexFlatL2(X.shape[1] + 1)
    flat.add(HnswIndex._augment(X, float(np.linalg.norm(X, axis=1).max())))
    _, got = flat.search(np.hstack([Q, np.zeros((len(Q), 1), np.float32)]), 10)
    assert _recall(exact, got) == 1.0


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_hnsw_scores_are_true_inner_products(metric):
    X, Q = _varying_norm_data(n=1000)
    s, i = HnswIndex.build(X, metric=metric).search(Q[:5], 5)
    assert np.allclose(s, np.einsum("qd,qkd->qk", Q[:5], X[i]), atol=1e-5)


def test_hnsw_recall_on_real_catalogue(full_engine_exact):
    """Part 3 acceptance: HNSW must agree with brute force on the real 30k catalogue."""
    X = full_engine_exact.vectors
    rng = np.random.default_rng(7)
    Q = X[rng.choice(len(X), 500, replace=False)]
    _, exact = full_engine_exact.exact.search(Q, 10)
    _, approx = HnswIndex.build(X).search(Q, 10)
    assert _recall(exact, approx) >= 0.95


def test_large_k_does_not_mutate_shared_ef_search():
    X, Q = _varying_norm_data(n=2000)
    h = HnswIndex.build(X, ef_search=32)
    _, idx = h.search(Q[:1], 150)
    assert h.ef_search == 32           # shared index state untouched
    assert (idx >= 0).all() and len(set(idx[0])) == 150  # but the large request is fully served
