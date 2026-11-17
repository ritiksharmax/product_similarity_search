import numpy as np
import pytest

from similarity import engine as engine_mod
from similarity.engine import ProductNotFoundError, SimilarityEngine
from similarity.features import BLOCK_ORDER, arc_encode, build_blocks

from conftest import make_settings


def test_returns_exactly_k_and_excludes_self(engine):
    out = engine.find_similar_products("ts1", 4)
    assert len(out) == 4
    assert "ts1" not in out
    assert len(set(out)) == 4


def test_same_asin_recrawl_is_never_recommended(engine):
    assert "ts4b" not in engine.find_similar_products("ts4", len(engine))
    assert "ts4" not in engine.find_similar_products("ts4b", len(engine))


def test_k_larger_than_catalogue_returns_everything_else(engine):
    out = engine.find_similar_products("cap1", 1000)
    assert len(out) == len(engine) - 1


def test_scores_sorted_descending(engine):
    scores = [m.score for m in engine.similar("sa1", 10)]
    assert scores == sorted(scores, reverse=True)


def test_semantically_sensible_neighbours(engine):
    # Same brand + same type + similar price should win.
    assert engine.find_similar_products("ts1", 1) == ["ts5"]
    assert set(engine.find_similar_products("sa1", 2)) == {"sa2", "sa3"}
    assert set(engine.find_similar_products("ku1", 1)) == {"ku2"}


def test_deterministic(engine):
    assert engine.find_similar_products("ts2", 5) == engine.find_similar_products("ts2", 5)


def test_unknown_product_and_bad_k(engine):
    with pytest.raises(ProductNotFoundError):
        engine.find_similar_products("does-not-exist", 3)
    with pytest.raises(ValueError):
        engine.find_similar_products("ts1", 0)


def test_explain_breakdown_reconstructs_score(engine):
    w = engine.settings.weights
    for m in engine.similar("ts1", 3, explain=True):
        assert set(m.breakdown) == set(BLOCK_ORDER)
        assert m.score == pytest.approx(sum(w[b] * m.breakdown[b] for b in BLOCK_ORDER), abs=1e-3)


def test_inner_product_equals_weighted_sum_of_block_cosines(catalog):
    s = make_settings()
    blocks = build_blocks(catalog, s)
    X = blocks.weighted(s.weights)
    i, j = 0, 1
    per_block = blocks.block_scores(i, np.array([j]))
    assert X[i] @ X[j] == pytest.approx(sum(s.weights[b] * per_block[b][0] for b in BLOCK_ORDER), abs=1e-5)


def test_arc_encoding_is_monotone_in_distance():
    v = arc_encode(np.array([0.5, 0.55, 0.8, 1.0, np.nan]))
    sims = v[1:] @ v[0]
    assert sims[0] > sims[1] > sims[2]  # closer percentile -> higher similarity
    assert sims[3] == 0.0               # missing -> neutral


def test_weights_change_ranking(catalog):
    brand_only = SimilarityEngine.build(make_settings(backend="exact", weights={"brand": 1.0}), catalog)
    top = brand_only.find_similar_products("sa1", 1)
    assert top == ["sa2"]  # the only other "dhruvi trendz" product


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        make_settings(weights={"nonsense": 1.0})
    with pytest.raises(ValueError):
        make_settings(weights={"text": -1.0})


def test_artifact_roundtrip(tmp_path, catalog):
    s = make_settings(tmp_path, backend="hnsw")
    built = SimilarityEngine.build(s, catalog)
    built.save(s.artifacts_dir)
    loaded = SimilarityEngine.load(s)
    for pid in ["ts1", "sa1", "cap1"]:
        assert loaded.find_similar_products(pid, 5) == built.find_similar_products(pid, 5)


def test_encoder_mismatch_without_raw_data_fails_with_actionable_error(tmp_path, catalog):
    s = make_settings(tmp_path, backend="exact")
    SimilarityEngine.build(s, catalog).save(s.artifacts_dir)
    serving = make_settings(tmp_path, backend="exact", encoder="minilm", data_path=tmp_path / "absent.ldjson")
    with pytest.raises(RuntimeError, match="--build-arg ENCODER=minilm"):
        SimilarityEngine.load(serving)


def test_module_level_function_matches_spec_signature(engine):
    engine_mod.set_engine(engine)
    try:
        out = engine_mod.find_similar_products("ts1", 3)
        assert isinstance(out, list) and all(isinstance(x, str) for x in out) and len(out) == 3
    finally:
        engine_mod.set_engine(None)


def test_top_k_is_prefix_of_larger_k_and_shares_cache(catalog):
    eng = SimilarityEngine.build(make_settings(backend="hnsw", max_num_similar=6), catalog)
    full = eng.find_similar_products("ts1", 6)
    for k in range(1, 7):
        assert eng.find_similar_products("ts1", k) == full[:k]
    info = eng.cache_info()
    assert info["misses"] == 1 and info["hits"] == 6  # one search serves every k
    # Beyond the cached depth (library use only; the API caps k) results are still correct.
    assert eng.find_similar_products("ts1", 9)[:6] == full


def test_hnsw_prefix_consistency_on_full_catalogue():
    from conftest import FULL_DATA
    from similarity.config import Settings

    if not FULL_DATA.exists():
        pytest.skip("full dataset not unzipped")
    eng = SimilarityEngine.build(Settings(backend="hnsw", encoder="tfidf"))
    ids = eng.catalog["uniq_id"].sample(300, random_state=3).tolist()
    for pid in ids:
        top = eng.find_similar_products(pid, 100)
        assert eng.find_similar_products(pid, 5) == top[:5]
        assert eng.find_similar_products(pid, 37) == top[:37]


# ------------------------------------------------------------------ image block


def test_image_block_unit_or_zero_and_fallback(catalog):
    blocks = build_blocks(catalog, make_settings())
    img = blocks.data[:, blocks.slices["image"]]
    norms = np.linalg.norm(img, axis=1)
    has_image = set(catalog["uniq_id"][norms > 0.99])
    assert {"sa3", "cap1"}.isdisjoint(has_image)          # no photo -> zero block
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)
    assert blocks.info["image"]["model"] == "fixture"


def test_missing_image_file_gives_empty_block_and_engine_still_answers(tmp_path, catalog):
    s = make_settings(image_embeddings_path=tmp_path / "absent.npz")
    eng = SimilarityEngine.build(s, catalog)
    assert eng.blocks.coverage()["image"] == 0.0
    assert eng.find_similar_products("ts1", 1) == ["ts5"]


def test_image_weight_drives_ranking(catalog):
    eng = SimilarityEngine.build(make_settings(backend="exact", weights={"image": 1.0}), catalog)
    # ts1 shares its "look" with ts5 and ts2; image-only similarity must rank those first.
    assert set(eng.find_similar_products("ts1", 2)) == {"ts5", "ts2"}
    # A product without an image has nothing to compare: every score is 0, ties broken deterministically.
    assert all(m.score == 0 for m in eng.similar("cap1", 3))


def test_manifest_records_image_provenance_and_stale_blocks_rebuild(tmp_path, catalog):
    import json

    s = make_settings(tmp_path, backend="exact")
    SimilarityEngine.build(s, catalog).save(s.artifacts_dir)
    manifest = json.loads((s.artifacts_dir / "manifest.json").read_text())
    assert manifest["image"]["model"] == "fixture" and "image" in manifest["coverage"]
    # Artifacts from before the image block existed are detected and rebuilt, not half-loaded.
    del manifest["slices"]["image"]
    (s.artifacts_dir / "manifest.json").write_text(json.dumps(manifest))
    assert "image" in SimilarityEngine.load(s).blocks.slices
