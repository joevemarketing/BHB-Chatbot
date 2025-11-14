from app.product_loader import get_all_products
from app.recommender import get_best_products
from app.models import UserConstraints


def test_recommender_strict_category_guard():
    products = get_all_products()
    c = UserConstraints(category="washer")
    top = get_best_products(products, c, top_n=5)
    assert all(p.category == "washer" for p in top)