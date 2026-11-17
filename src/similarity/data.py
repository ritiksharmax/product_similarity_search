"""Loading and cleaning the Amazon Fashion ``.ldjson`` dump.

The raw data is messy in ways that matter for similarity (see README "Data findings"):

* ``weight`` is the sentinel ``"999999999"`` for 79% of rows, otherwise strings such as
  ``"200 g"``, ``"454 Grams"`` or ``"1.4 Kg"`` (and one row of marketing copy).
* ``rating`` is always filled in, even for products with *no reviews*, and most products have
  one or two reviews, so the raw value is mostly noise. We shrink it towards the catalogue
  mean using the review count parsed from ``product_details__k_v_pairs``.
* ``brand`` is missing for 27% of rows and has case variants (``MIZAGO`` / ``Mizago``).
* ``colour`` is present for ~20% of rows as pipe-separated free text (``Black|Navy Blue``).
* the same ASIN occurs twice for ~450 products (re-crawls), so they must not be
  recommended as "similar" to themselves.

We parse with the stdlib ``json`` module rather than ``pd.read_json`` so the nested dict
columns are handled explicitly and parsing stays O(rows) with no dtype guessing.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

WEIGHT_SENTINEL = "999999999"
_WEIGHT_RE = re.compile(r"^\s*([\d.,]+)\s*(g|gm|gms|grams?|kg|kgs|kilograms?)\s*$", re.IGNORECASE)
_REVIEWS_RE = re.compile(r"([\d,]+)\s+customer reviews?", re.IGNORECASE)

# Free-text colour tokens -> a small base palette. Order matters: more specific keys first
# ("navy" before "blue", "off white" before "white").
_COLOUR_KEYWORDS: list[tuple[str, str]] = [
    ("multi", "multi"), ("navy", "navy"), ("maroon", "maroon"), ("wine", "maroon"),
    ("burgundy", "maroon"), ("off white", "cream"), ("off-white", "cream"), ("cream", "cream"),
    ("ivory", "cream"), ("beige", "beige"), ("khaki", "beige"), ("nude", "beige"),
    ("black", "black"), ("white", "white"), ("grey", "grey"), ("gray", "grey"),
    ("charcoal", "grey"), ("silver", "silver"), ("gold", "gold"), ("teal", "teal"),
    ("turquoise", "teal"), ("aqua", "teal"), ("blue", "blue"), ("denim", "blue"),
    ("red", "red"), ("pink", "pink"), ("peach", "pink"), ("magenta", "pink"),
    ("rose", "pink"), ("green", "green"), ("olive", "green"), ("mint", "green"),
    ("yellow", "yellow"), ("mustard", "yellow"), ("orange", "orange"), ("rust", "orange"),
    ("coral", "orange"), ("purple", "purple"), ("violet", "purple"), ("lavender", "purple"),
    ("mauve", "purple"), ("brown", "brown"), ("tan", "brown"), ("coffee", "brown"),
]
BASE_COLOURS: list[str] = sorted({base for _, base in _COLOUR_KEYWORDS})


# --------------------------------------------------------------------------- field parsers


def parse_price(value: Any) -> float:
    """``"1,299.00"`` -> 1299.0; missing/garbage -> NaN."""
    if value is None:
        return math.nan
    try:
        price = float(str(value).replace(",", "").strip())
    except ValueError:
        return math.nan
    return price if price > 0 else math.nan


def parse_weight_grams(value: Any) -> float:
    """``"200 g"`` -> 200, ``"1.4 Kg"`` -> 1400; sentinel or free text -> NaN."""
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == WEIGHT_SENTINEL:
        return math.nan
    m = _WEIGHT_RE.match(text)
    if not m:
        return math.nan
    try:
        amount = float(m.group(1).replace(",", ""))
    except ValueError:
        return math.nan
    grams = amount * 1000 if m.group(2).lower().startswith("k") else amount
    return grams if grams > 0 else math.nan


def parse_rating(value: Any) -> float:
    try:
        r = float(value)
    except (TypeError, ValueError):
        return math.nan
    return r if 0 < r <= 5 else math.nan


def parse_review_count(record: dict[str, Any]) -> int:
    """Number of customer reviews. ``"Be the first to review"`` -> 0."""
    details = record.get("product_details__k_v_pairs")
    if isinstance(details, dict):
        text = details.get("Customer_Reviews") or ""
        m = _REVIEWS_RE.search(text)
        if m:
            return int(m.group(1).replace(",", ""))
        if text.lower().startswith("be the first"):
            return 0
    try:
        return int(str(record.get("no__of_reviews", "0")).replace(",", ""))
    except ValueError:
        return 0


def parse_discount(value: Any) -> float:
    """``"50%"`` -> 0.5."""
    if not value:
        return math.nan
    try:
        return float(str(value).strip().rstrip("%")) / 100.0
    except ValueError:
        return math.nan


# Placeholder "brands" carry no information: two unbranded items aren't similar *because*
# both say "Generic", so these are treated as missing.
_PLACEHOLDER_BRANDS = {"generic", "unbranded", "no brand", "none", "na", "n/a", "unknown", "others"}


def normalise_brand(value: Any) -> str | None:
    if not value or not str(value).strip():
        return None
    brand = re.sub(r"\s+", " ", str(value)).strip().lower()
    return None if brand in _PLACEHOLDER_BRANDS else brand


def parse_colours(value: Any) -> list[str]:
    """``"Black|Navy Blue"`` -> ``["black", "navy"]`` (base palette, de-duplicated, sorted)."""
    if not value:
        return []
    found: set[str] = set()
    for token in str(value).lower().split("|"):
        token = token.strip()
        for keyword, base in _COLOUR_KEYWORDS:
            if keyword in token:
                found.add(base)
                break
    return sorted(found)


def first_key(value: Any) -> str | None:
    """Category columns are dicts like ``{"MensT_Shirts": "#1793"}``; we want the key."""
    if isinstance(value, dict) and value:
        return next(iter(value))
    return None


def parse_also_bought(value: Any) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in str(value).split(" | ") if s.strip()]


# --------------------------------------------------------------------------- loading


def iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def clean_records(records: Iterable[dict[str, Any]], rating_prior_m: float = 5.0) -> pd.DataFrame:
    """Turn raw records into a tidy, typed catalogue DataFrame (one row per ``uniq_id``)."""
    rows = []
    for r in records:
        uid = r.get("uniq_id")
        if not uid:
            continue
        rows.append(
            {
                "uniq_id": str(uid),
                "asin": r.get("asin") or str(uid),
                "product_name": (r.get("product_name") or "").strip(),
                "brand": normalise_brand(r.get("brand")),
                "category": first_key(r.get("sales_rank_in_child_category")),
                "parent_category": first_key(r.get("sales_rank_in_parent_category")),
                "colours": parse_colours(r.get("colour")),
                "price": parse_price(r.get("sales_price")),
                "weight_g": parse_weight_grams(r.get("weight")),
                "rating_raw": parse_rating(r.get("rating")),
                "review_count": parse_review_count(r),
                "discount": parse_discount(r.get("discount_percentage")),
                "image_url": (r.get("medium") or "").split("|")[0] or None,
                "also_bought": parse_also_bought(r.get("other_items_customers_buy")),
            }
        )
    if not rows:
        raise ValueError("dataset is empty")
    df = pd.DataFrame(rows).drop_duplicates("uniq_id", keep="first").reset_index(drop=True)
    df["rating"] = shrink_ratings(df["rating_raw"], df["review_count"], rating_prior_m)
    return df


def shrink_ratings(rating: pd.Series, n_reviews: pd.Series, m: float) -> pd.Series:
    """Bayesian average: ``(n*r + m*mu) / (n + m)``.

    A 5.0 from one review becomes ~4.2 while a 4.6 from 200 reviews stays ~4.6, and
    unreviewed products (whose scraped rating is meaningless) collapse to the prior ``mu``.
    """
    reviewed = (n_reviews > 0) & rating.notna()
    mu = float(rating[reviewed].mean()) if reviewed.any() else 4.0
    n = n_reviews.where(reviewed, 0).astype(float)
    r = rating.fillna(mu)
    return (n * r + m * mu) / (n + m)


def load_catalog(path: str | Path, rating_prior_m: float = 5.0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `cd data && unzip archive.zip` first")
    return clean_records(iter_records(path), rating_prior_m)


def percentile_rank(values: pd.Series) -> np.ndarray:
    """Values -> percentile in [0, 1] (NaN preserved). Robust to skew and outliers."""
    return values.rank(pct=True, method="average").to_numpy(dtype=float)
