import math

import pandas as pd
import pytest

from similarity.data import (
    normalise_brand,
    parse_colours,
    parse_discount,
    parse_price,
    parse_review_count,
    parse_weight_grams,
    shrink_ratings,
)


@pytest.mark.parametrize("raw, grams", [
    ("200 g", 200.0), ("454 Grams", 454.0), ("1.4 Kg", 1400.0), ("2 Kilograms", 2000.0),
    ("99.8 g", 99.8), ("1,200 g", 1200.0),
])
def test_weight_parses_units(raw, grams):
    assert parse_weight_grams(raw) == pytest.approx(grams)


@pytest.mark.parametrize("raw", ["999999999", None, "", "The Slim N Lift slimming shirt", "0 g"])
def test_weight_sentinel_and_junk_are_missing(raw):
    assert math.isnan(parse_weight_grams(raw))


def test_price():
    assert parse_price("1,299.00") == 1299.0
    assert math.isnan(parse_price(None))
    assert math.isnan(parse_price("n/a"))


def test_review_count():
    assert parse_review_count({"product_details__k_v_pairs": {"Customer_Reviews": "4.5 out of 5 stars 1,234 customer reviews"}}) == 1234
    assert parse_review_count({"product_details__k_v_pairs": {"Customer_Reviews": "5.0 out of 5 stars 1 customer review"}}) == 1
    assert parse_review_count({"product_details__k_v_pairs": {"Customer_Reviews": "Be the first to review this item"}}) == 0
    assert parse_review_count({"no__of_reviews": "7"}) == 7
    assert parse_review_count({}) == 0


def test_brand_normalisation():
    assert normalise_brand("  MIZAGO ") == normalise_brand("Mizago") == "mizago"
    assert normalise_brand("Generic") is None
    assert normalise_brand("") is None and normalise_brand(None) is None


def test_colours_map_to_base_palette():
    assert parse_colours("Black|Navy Blue") == ["black", "navy"]
    assert parse_colours("Off White|Wine") == ["cream", "maroon"]
    assert parse_colours(None) == []


def test_discount():
    assert parse_discount("50%") == 0.5
    assert math.isnan(parse_discount(None))


def test_rating_shrinkage_pulls_low_evidence_to_mean():
    rating = pd.Series([5.0, 4.6, 5.0, 3.0])
    reviews = pd.Series([1, 200, 0, 50])
    shrunk = shrink_ratings(rating, reviews, m=5)
    mu = (5.0 + 4.6 + 3.0) / 3  # mean over *reviewed* items only
    assert shrunk[2] == pytest.approx(mu)            # unreviewed -> prior
    assert shrunk[0] < 4.6 and shrunk[0] > mu        # single 5-star review is heavily discounted
    assert shrunk[1] == pytest.approx(4.6, abs=0.01)  # lots of evidence -> barely moves


def test_catalog_shape(catalog):
    assert catalog["uniq_id"].is_unique
    assert catalog.loc[catalog.uniq_id == "ts3", "weight_g"].item() == 200.0
    assert catalog.loc[catalog.uniq_id == "ts4", "brand"].isna().item()  # "Generic"
