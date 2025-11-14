import urllib.parse

from .models import Product


def get_bhb_link_or_search(product: Product) -> dict:
    """
    Return a link dict for BHB either as a direct product URL or a search URL.
    {"type": "direct"|"search", "url": "..."}
    """
    if product.bhb_product_url:
        return {"type": "direct", "url": product.bhb_product_url}
    base = "https://www.bhb.com.my/search?q="
    q = product.website_search_text or f"{product.brand} {product.model_name}"
    encoded = urllib.parse.quote(q)
    return {"type": "search", "url": base + encoded}