"""Writes tests/fixtures/sample.ldjson: a tiny hand-made catalogue covering the data's edge cases."""
import json
from pathlib import Path


def rec(uid, name, brand=None, price="499.00", weight="999999999", rating="4.0", reviews=None,
        child="MensT_Shirts", colour=None, asin=None):
    r = {"uniq_id": uid, "asin": asin or f"ASIN{uid}", "product_name": name, "weight": weight,
         "rating": rating, "medium": f"https://img/{uid}.jpg",
         "product_details__k_v_pairs": {"Customer_Reviews": reviews or "Be the first to review this item"}}
    if brand: r["brand"] = brand
    if price: r["sales_price"] = price
    if child: r["sales_rank_in_child_category"] = {child: "#1"}
    if colour: r["colour"] = colour
    return r

RECORDS = [
    rec("ts1", "Puma Men's Regular Fit Cotton T-Shirt Black", "Puma", "799.00", "200 g", "4.5", "4.5 out of 5 stars 120 customer reviews", colour="Black"),
    rec("ts2", "Puma Men's Slim Fit Cotton T-Shirt Navy", "PUMA", "849.00", "210 g", "4.2", "4.2 out of 5 stars 40 customer reviews", colour="Navy Blue"),
    rec("ts3", "Nike Men's Dri-Fit Cotton T-Shirt Black", "Nike", "999.00", "0.2 Kg", "4.4", "4.4 out of 5 stars 300 customer reviews", colour="Black|White"),
    rec("ts4", "Generic Men's Plain Cotton T-Shirt", "Generic", "199.00"),
    rec("ts4b", "Generic Men's Plain Cotton T-Shirt", "Generic", "199.00", asin="ASINts4"),  # re-crawl, same ASIN
    rec("ts5", "Puma Men's Regular Fit Cotton T-Shirt White", "Puma", "799.00", rating="5.0", reviews="5.0 out of 5 stars 1 customer review", colour="White"),
    rec("sa1", "Women's Soft Cotton Silk Saree with Blouse Piece Red", "Dhruvi Trendz", "355.00", child="WomensSarees", colour="Red"),
    rec("sa2", "Women's Banarasi Silk Saree with Blouse Piece Maroon", "Dhruvi Trendz", "405.00", child="WomensSarees", colour="Maroon"),
    rec("sa3", "Women's Georgette Printed Saree with Blouse Piece", None, None, child="WomensSarees"),
    rec("ku1", "Women's Cotton Printed Straight Kurti Blue", "BIBA", "699.00", child="WomensKurtasKurtis", colour="Blue"),
    rec("ku2", "Women's Rayon A-Line Kurta Pink", "BIBA", "749.00", child="WomensKurtasKurtis", colour="Pink"),
    rec("cap1", "Men's Embroidered Baseball Cap Black", "E4hats", "4329.00", "90 Grams", child=None, colour="Black"),
]

# Image "looks": products sharing a look have near-identical photos. sa3 and cap1 have no image,
# exercising the zero-block fallback.
IMAGE_LOOKS = {"ts1": "tee", "ts5": "tee", "ts2": "tee", "ts3": "sport", "ts4": "plain", "ts4b": "plain",
               "sa1": "saree", "sa2": "saree", "ku1": "kurti", "ku2": "kurti"}


def write_image_embeddings(path: Path, dim: int = 16) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    looks = {look: rng.standard_normal(dim) for look in sorted(set(IMAGE_LOOKS.values()))}
    ids = sorted(IMAGE_LOOKS)
    vecs = np.stack([looks[IMAGE_LOOKS[i]] + 0.05 * rng.standard_normal(dim) for i in ids])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.savez_compressed(path, uniq_id=np.array(ids), vectors=vecs.astype(np.float16),
                        meta=json.dumps({"model": "fixture", "pca_dim": dim}))


if __name__ == "__main__":
    out = Path(__file__).with_name("sample.ldjson")
    out.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n")
    write_image_embeddings(Path(__file__).with_name("image_embeddings.npz"))
    print(f"wrote {len(RECORDS)} records to {out} (+ image_embeddings.npz)")
