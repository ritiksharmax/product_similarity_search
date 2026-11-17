from pathlib import Path

import pytest

from similarity.config import Settings
from similarity.data import load_catalog
from similarity.engine import SimilarityEngine

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ldjson"
FULL_DATA = Path(Settings().data_path)


@pytest.fixture(scope="session")
def catalog():
    return load_catalog(FIXTURE)


def make_settings(tmp_path: Path | None = None, **overrides) -> Settings:
    kw = dict(data_path=FIXTURE, text_dim=8, brand_dim=64, cache_size=64,
              image_embeddings_path=FIXTURE.with_name("image_embeddings.npz"))
    if tmp_path is not None:
        kw["artifacts_dir"] = tmp_path / "artifacts"
    kw.update(overrides)
    return Settings(**kw)


@pytest.fixture(scope="session", params=["exact", "hnsw"])
def engine(request, catalog):
    return SimilarityEngine.build(make_settings(backend=request.param), catalog)


@pytest.fixture(scope="session")
def full_engine_exact():
    if not FULL_DATA.exists():
        pytest.skip("full dataset not unzipped (cd data && unzip archive.zip)")
    return SimilarityEngine.build(Settings(backend="exact"))
