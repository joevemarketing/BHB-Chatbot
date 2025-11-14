#!/usr/bin/env python3
import argparse
import json
import re
import time
import random
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

HEADERS = {
    "User-Agent": "BHB-Scraper/1.0 (+https://www.bhb.com.my) contact=admin@example.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def can_fetch(base_url: str, path: str, ua: str = HEADERS["User-Agent"], ignore: bool = False) -> bool:
    if ignore:
        return True
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base_url, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        # If robots.txt is not reachable, be conservative unless ignore is True
        return False
    return rp.can_fetch(ua, urljoin(base_url, path))

def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")

def extract_price(text: str) -> float | None:
    # Expects RM prices like "RM 1,299.00" or "1,299"
    m = re.search(r"([0-9][0-9,\.]+)", text.replace("\xa0", " "))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None

def parse_product_page(session: requests.Session, url: str) -> dict:
    soup = get_soup(session, url)
    name_el = (
        soup.select_one("h1.product_title")
        or soup.select_one(".product_title.entry-title")
        or soup.select_one("h1.entry-title")
        or soup.select_one("h1.product__title")
        or soup.select_one(".product__title")
    )
    name = name_el.get_text(strip=True) if name_el else None
    if not name:
        ogt = soup.select_one('meta[property="og:title"]')
        if ogt and ogt.has_attr("content"):
            name = ogt["content"].strip()

    # Price
    price_el = (
        soup.select_one("p.price")
        or soup.select_one(".price")
        or soup.select_one(".price__regular")
        or soup.select_one("span.price-item.price-item--regular")
        or soup.select_one("div.product__price")
    )
    price_text = price_el.get_text(" ", strip=True) if price_el else ""
    price = extract_price(price_text)
    currency = "RM" if "rm" in price_text.lower() else "RM"  # default to RM

    # Brand (if present in attributes table)
    brand = None
    for row in soup.select("table.woocommerce-product-attributes tr"):
        th = row.select_one("th.woocommerce-product-attributes-item__label")
        td = row.select_one("td.woocommerce-product-attributes-item__value")
        if th and td and "brand" in th.get_text(strip=True).lower():
            brand = td.get_text(" ", strip=True)
            break

    # Categories from breadcrumbs
    categories = []
    for a in soup.select(".woocommerce-breadcrumb a, nav.woocommerce-breadcrumb a, nav.breadcrumb a, ol.breadcrumb a"):
        href = a.get("href") or ""
        text = a.get_text(strip=True)
        if ("/product-category/" in href or "/collections/" in href) and text:
            categories.append(text)

    # Features: gather bullet points from description tabs
    features = []
    for li in soup.select("div.woocommerce-Tabs-panel--description li, #tab-description li, .product-short-description li"):
        t = li.get_text(" ", strip=True)
        if t:
            features.append(t)

    # Image
    img_el = (
        soup.select_one("img.wp-post-image")
        or soup.select_one(".woocommerce-product-gallery__wrapper img")
        or soup.select_one("img.product__media")
        or soup.select_one("img#FeaturedImage-product-template")
    )
    image_url = None
    if img_el:
        if img_el.has_attr("src"):
            image_url = img_el.get("src")
        elif img_el.has_attr("data-src"):
            image_url = img_el.get("data-src")
    if not image_url:
        ogi = soup.select_one('meta[property="og:image"]')
        if ogi and ogi.has_attr("content"):
            image_url = ogi["content"].strip()

    # ID/slug
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if path else None

    return {
        "id": slug or url,
        "name": name or slug or "Unknown Product",
        "brand": brand,
        "category": categories[-1] if categories else None,
        "categories": categories,
        "price": price,
        "currency": currency,
        "features": features[:10],
        "image_url": image_url,
        "permalink": url,
        "_source": "bhb.com.my",
    }

def parse_listing_for_links(soup: BeautifulSoup, base_url: str) -> set[str]:
    links = set()
    # WooCommerce typical selectors
    for a in soup.select("a.woocommerce-LoopProduct-link, ul.products li.product a"):
        href = a.get("href")
        if href and "/product/" in href:
            links.add(href)
    # Shopify collections: anchors to /products/ paths
    for a in soup.select("a[href*='/products/'], a.grid-product__link, a.card-wrapper, a.full-unstyled-link"):
        href = a.get("href")
        if href:
            if "/products/" in href or "/product/" in href:
                links.add(href)
    # Normalize
    return {urljoin(base_url, href) for href in links}

def find_next_page(soup: BeautifulSoup, base_url: str) -> str | None:
    # Standard WooCommerce pagination patterns
    a = soup.select_one("a.next, a.page-numbers.next, a[rel='next']")
    if a and a.get("href"):
        return urljoin(base_url, a["href"])
    return None

def _parse_sitemap_xml(text: str) -> list[str]:
    urls: list[str] = []
    try:
        soup = BeautifulSoup(text, "xml")
        for loc in soup.find_all("loc"):
            if loc.text:
                urls.append(loc.text.strip())
    except Exception:
        try:
            root = ET.fromstring(text)
            ns = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
            for loc in root.iter(f"{ns}loc"):
                if loc.text:
                    urls.append(loc.text.strip())
        except Exception:
            pass
    return urls

def collect_sitemap_product_links(base_url: str, max_sitemaps: int = 50) -> set[str]:
    links: set[str] = set()
    session = requests.Session()
    # 1) WP core posts-product sitemaps
    for i in range(1, max_sitemaps + 1):
        url = f"{base_url.rstrip('/')}/wp-sitemap-posts-product-{i}.xml"
        try:
            resp = session.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 404:
                break
            if resp.status_code != 200:
                continue
            for u in _parse_sitemap_xml(resp.text):
                if "/product/" in u:
                    links.add(u)
        except Exception:
            break
    # 2) Yoast-style sitemap index
    try:
        idx = f"{base_url.rstrip('/')}/sitemap_index.xml"
        resp = session.get(idx, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            for sm in _parse_sitemap_xml(resp.text):
                if any(tok in sm.lower() for tok in ["product", "products"]):
                    try:
                        r = session.get(sm, headers=HEADERS, timeout=10)
                        if r.status_code == 200:
                            for u in _parse_sitemap_xml(r.text):
                                if "/product/" in u:
                                    links.add(u)
                    except Exception:
                        pass
    except Exception:
        pass
    # 3) Common plugin sitemap
    for candidate in [
        f"{base_url.rstrip('/')}/product-sitemap.xml",
        f"{base_url.rstrip('/')}/products-sitemap.xml",
    ]:
        try:
            r = session.get(candidate, headers=HEADERS, timeout=10)
            if r.status_code == 200:
                for u in _parse_sitemap_xml(r.text):
                    if "/product/" in u:
                        links.add(u)
        except Exception:
            pass
    return links

def fetch_sitemap_products(base_url: str, delay: float, ignore_robots: bool = False) -> list[dict]:
    products: list[dict] = []
    product_links = collect_sitemap_product_links(base_url)
    if not product_links:
        return products
    session = requests.Session()
    for link in sorted(product_links):
        if not can_fetch(base_url, urlparse(link).path, ignore=ignore_robots):
            continue
        try:
            print(f"  [Sitemap Product] {link}")
            item = parse_product_page(session, link)
            products.append(item)
            time.sleep(delay + random.uniform(0, delay / 2))
        except Exception as e:
            print(f"  Error parsing {link}: {e}")
    return products

def collect_collection_category_links(session: requests.Session, collections_url: str, base_url: str) -> set[str]:
    """Collect category/archive links from a collections-by-category page."""
    links: set[str] = set()
    soup = get_soup(session, collections_url)
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if not href:
            continue
        href_full = urljoin(base_url, href)
        hl = href_full.lower()
        # Target common category/archive patterns
        if "/product-category/" in hl or "/shop/" in hl or "/collections/" in hl:
            links.add(href_full)
    return links

def fetch_collection_products(collections_url: str, max_pages: int, delay: float, ignore_robots: bool = False) -> list[dict]:
    """Crawl categories discovered from a collections page and parse product pages."""
    base_url = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(collections_url))
    session = requests.Session()
    products: list[dict] = []
    seen_products: set[str] = set()

    cat_links = collect_collection_category_links(session, collections_url, base_url)
    if not cat_links:
        return products

    for cat in sorted(cat_links):
        if not can_fetch(base_url, urlparse(cat).path, ignore=ignore_robots):
            continue
        page_url = cat
        page_count = 0
        while page_url and page_count < max_pages:
            try:
                print(f"[Category Listing] {page_url}")
                soup = get_soup(session, page_url)
                product_links = parse_listing_for_links(soup, base_url)
                print(f"  Found {len(product_links)} product links")
                for link in product_links:
                    if link in seen_products:
                        continue
                    if not can_fetch(base_url, urlparse(link).path, ignore=ignore_robots):
                        continue
                    try:
                        print(f"  [Product] {link}")
                        item = parse_product_page(session, link)
                        products.append(item)
                        try:
                            nm = item.get("name")
                            print(f"    Added: {nm}")
                        except Exception:
                            pass
                        seen_products.add(link)
                        time.sleep(delay + random.uniform(0, delay / 2))
                    except Exception as e:
                        print(f"  Error parsing {link}: {e}")
                next_url = find_next_page(soup, base_url)
                page_url = next_url
                page_count += 1
                time.sleep(delay)
            except Exception as e:
                print(f"  Error category page {page_url}: {e}")
                break
    return products

def supports_store_api(base_url: str) -> bool:
    try:
        test = f"{base_url.rstrip('/')}/wp-json/wc/store/products?per_page=1"
        r = requests.get(test, timeout=10)
        if r.status_code != 200:
            return False
        try:
            js = r.json()
        except Exception:
            return False
        # Some sites return a dict wrapper; treat that as supported too
        return isinstance(js, list) or (isinstance(js, dict) and any(isinstance(v, list) for v in js.values()))
    except Exception:
        return False

def fetch_store_products(base_url: str, max_pages: int, per_page: int = 100) -> list[dict]:
    items = []
    for page in range(1, max_pages + 1):
        url = f"{base_url.rstrip('/')}/wp-json/wc/store/products"
        params = {"per_page": per_page, "page": page}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            break
        try:
            batch = resp.json()
        except Exception:
            break
        # Normalize payload into a list
        if isinstance(batch, list):
            products = batch
        elif isinstance(batch, dict):
            products = None
            for key in ("data", "results", "items", "products"):
                val = batch.get(key)
                if isinstance(val, list):
                    products = val
                    break
            if products is None:
                break
        else:
            break
        if not products:
            break
        for it in products:
            if not isinstance(it, dict):
                continue
            prices = it.get("prices", {}) or {}
            raw = prices.get("price") or prices.get("regular_price") or prices.get("sale_price")
            currency_symbol = prices.get("currency_symbol") or "$"
            amount = None
            if raw is not None:
                try:
                    amount = float(raw) / 100.0
                except Exception:
                    amount = None
            images = it.get("images") or []
            image_url = images[0]["src"] if images else None
            # Categories
            cats = it.get("categories") or []
            categories: list[str] = []
            if isinstance(cats, list):
                for c in cats:
                    if isinstance(c, dict):
                        nm = c.get("name") or c.get("slug")
                        if nm:
                            categories.append(str(nm))
                    elif isinstance(c, str):
                        categories.append(c)
            category = categories[-1] if categories else None
            # Attributes and tags as features
            features: list[str] = []
            attrs = it.get("attributes") or []
            for a in attrs:
                if isinstance(a, dict):
                    label = a.get("name") or a.get("label")
                    options = a.get("options") or []
                    if label:
                        features.append(str(label))
                    for opt in options:
                        features.append(str(opt))
            tags = it.get("tags") or []
            for t in tags:
                if isinstance(t, dict):
                    tn = t.get("name") or t.get("slug")
                    if tn:
                        features.append(str(tn))
                elif isinstance(t, str):
                    features.append(t)
            items.append({
                "id": it.get("id"),
                "name": it.get("name") or "Unknown Product",
                "brand": "",  # can be enriched from attributes if needed
                "category": category,
                "categories": categories,
                "price": amount,
                "currency": currency_symbol,
                "features": features,
                "image_url": image_url,
                "permalink": it.get("permalink") or it.get("permalink_url"),
                "_source": "bhb.com.my",
            })
    return items

def scrape_bhb(start_url: str, max_pages: int, delay: float, ignore_robots: bool = False) -> list[dict]:
    base_url = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(start_url))
    # Prefer Store API
    if supports_store_api(base_url):
        print("Using WooCommerce Store API")
        return fetch_store_products(base_url, max_pages)
    else:
        # Attempt Store API even if detection fails; fallback on empty
        print("Attempting WooCommerce Store API anyway")
        try:
            items = fetch_store_products(base_url, max_pages)
            if items:
                return items
        except Exception as e:
            print(f"Store API attempt failed: {e}")
    # Sitemap crawl fallback
    print("Attempting sitemap crawl for product URLs")
    sitemap_items = fetch_sitemap_products(base_url, delay, ignore_robots)
    if sitemap_items:
        return sitemap_items
    # Collections-by-category crawl fallback
    if "/pages/collections-by-category" in start_url:
        print("Attempting collections-by-category crawl")
        coll_items = fetch_collection_products(start_url, max_pages, delay, ignore_robots)
        if coll_items:
            return coll_items
    # HTML fallback
    # robots.txt check
    if not can_fetch(base_url, "/shop/", ignore=ignore_robots):
        print("robots.txt disallows fetching /shop/. Aborting.")
        return []

    session = requests.Session()
    collected = []
    seen_products = set()
    page_url = start_url
    page_count = 0

    while page_url and page_count < max_pages:
        print(f"[Listing] {page_url}")
        soup = get_soup(session, page_url)
        product_links = parse_listing_for_links(soup, base_url)
        print(f"  Found {len(product_links)} product links")

        for link in product_links:
            if link in seen_products:
                continue
            # robots for product page
            if not can_fetch(base_url, urlparse(link).path, ignore=ignore_robots):
                print(f"  Skipping disallowed by robots: {link}")
                continue
            try:
                print(f"  [Product] {link}")
                item = parse_product_page(session, link)
                collected.append(item)
                seen_products.add(link)
                time.sleep(delay + random.uniform(0, delay / 2))
            except Exception as e:
                print(f"  Error parsing {link}: {e}")

        next_url = find_next_page(soup, base_url)
        page_url = next_url
        page_count += 1
        time.sleep(delay)

    return collected

def main():
    parser = argparse.ArgumentParser(description="Scrape products from bhb.com.my")
    parser.add_argument("--start-url", default="https://www.bhb.com.my/shop/", help="Listing start URL")
    parser.add_argument("--max-pages", type=int, default=10, help="Max listing pages to crawl")
    parser.add_argument("--delay", type=float, default=0.75, help="Delay between requests (seconds)")
    parser.add_argument("--output", default=str(DATA_DIR / "products.json"), help="Output JSON path")
    parser.add_argument("--ignore-robots", action="store_true", help="Ignore robots.txt for local testing")
    args = parser.parse_args()

    products = scrape_bhb(args.start_url, args.max_pages, args.delay, args.ignore_robots)
    print(f"Collected {len(products)} products")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Backup existing
    if out_path.exists():
        backup = out_path.with_suffix(".bak.json")
        backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backed up existing to {backup}")

    out_path.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(products)} items to {out_path}")

if __name__ == "__main__":
    main()