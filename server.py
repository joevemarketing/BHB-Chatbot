# imports
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote_plus
from datetime import datetime
import csv

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

from fastapi import UploadFile, File
from fastapi import Request
import base64

try:
    import requests  # for WooCommerce REST
except Exception:
    requests = None
try:
    from bs4 import BeautifulSoup  # for HTML search fallback (Shopify/WooCommerce themes)
except Exception:
    BeautifulSoup = None

# Optional Gemini Vision (Google) integration
try:
    import google.generativeai as genai
except Exception:
    genai = None

load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
KB_DIR = DATA_DIR / "kb"

app = FastAPI(title="Retail Chatbot")
from app.middleware import RequestLoggingMiddleware, RateLimitMiddleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=int(os.getenv("RATE_LIMIT_MAX", "60")), window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
def log_analytics_event(event: Dict[str, Any]) -> None:
    try:
        ensure_dir(DATA_DIR)
        path = DATA_DIR / "analytics_events.jsonl"
        row = json.dumps(event, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")
    except Exception:
        pass

@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/admin")
def admin_home():
    return FileResponse(STATIC_DIR / "admin.html")

# top-level FastAPI routes
from fastapi.responses import JSONResponse, FileResponse

@app.get("/api/health")
def health():
    return {"status": "ok"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "openai").lower()
ENABLE_AI_REWRITE = os.getenv("ENABLE_AI_REWRITE", "0") == "1"
WC_API_URL = os.getenv("WC_API_URL")  # e.g., https://www.bhb.com.my
WC_CONSUMER_KEY = os.getenv("WC_CONSUMER_KEY")
WC_CONSUMER_SECRET = os.getenv("WC_CONSUMER_SECRET")
ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "admin123")
try:
    import hashlib
    ADMIN_TOKEN = hashlib.sha256(ADMIN_PASSCODE.encode("utf-8")).hexdigest()
except Exception:
    ADMIN_TOKEN = ADMIN_PASSCODE

sessions: Dict[str, Dict[str, Any]] = {}

def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def redact_pii_text(text: str) -> str:
    t = text or ""
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", t)
    t = re.sub(r"\b(?:\+?6?0)?\s?(?:\d{2,3}[- ]?)?\d{3}[- ]?\d{4}\b", "[redacted-phone]", t)
    return t

async def _moderate_text_server(text: str) -> Optional[bool]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        model = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
        resp = client.moderations.create(model=model, input=text or "")
        flagged = bool(resp.results[0].flagged) if hasattr(resp, "results") and resp.results else False
        return flagged
    except Exception:
        return None

products = load_json(DATA_DIR / "products.json") or []
# Merge real BHB items (fans, water heaters, rice cookers, air fryers) into catalog
try:
    _real = load_json(DATA_DIR / "bhb_products_real.json") or []
    def _normalize_real(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in entries:
            try:
                name_bits = [str(r.get("brand") or "").strip(), str(r.get("model_name") or "").strip()]
                name = " ".join([b for b in name_bits if b]).strip() or (r.get("sku") or r.get("brand") or "Item")
                out.append({
                    "name": name,
                    "brand": r.get("brand") or "",
                    "category": (r.get("category") or "").strip().lower(),
                    "features": (r.get("features") or []),
                    "price": r.get("price_rm"),
                    "currency": "RM",
                    "permalink": r.get("bhb_product_url"),
                    "link": r.get("bhb_product_url"),
                    "_source": "bhb.com.my",
                })
            except Exception:
                pass
        return out
    products = (products or []) + _normalize_real(_real)
except Exception:
    products = products or []
faqs = load_json(DATA_DIR / "faq.json") or []
training_modules = load_json(DATA_DIR / "training" / "modules.json") or []
kb_index = load_json(DATA_DIR / "kb_index.json") or []
kb_embeddings = load_json(DATA_DIR / "kb_embeddings.json") or {}
_kb_embed_items: List[Dict[str, Any]] = []

# --- Store locations data ---
store_locations = load_json(DATA_DIR / "bhb_store_locations.json") or []

def find_stores_by_query(q: str) -> List[Dict[str, Any]]:
    """Intelligent text search over store locations.
    Handles general queries like "near me" and specific location queries.
    """
    q_lower = (q or "").lower()
    
    # Enhanced general terms for store queries - more comprehensive coverage
    general_terms = [
        "near me", "near", "berhampiran", "dekat", "nearby", "closest", "nearest",
        "store", "shop", "outlet", "branch", "cawangan", "kedai", 
        "lokasi", "alamat", "location", "where", "mana", "find", "cari",
        "which", "what", "any", "all"
    ]
    
    # Check if this is a general store location query
    is_general_query = any(term in q_lower for term in general_terms)
    
    if is_general_query:
        # For general queries, prioritize stores based on common/popular locations
        # Sort by state priority (major cities first) and then by store name
        priority_order = {"Penang": 0, "Kuala Lumpur": 1, "Selangor": 2, "Kedah": 3}
        
        def sort_key(store):
            state = store.get("state", "")
            priority = priority_order.get(state, 4)  # Unknown states get lowest priority
            return (priority, store.get("label", ""))
        
        sorted_stores = sorted(store_locations, key=sort_key)
        return sorted_stores
    
    # For specific location queries, do intelligent matching
    results: List[Dict[str, Any]] = []
    
    # Extract key location terms from query
    location_terms = []
    words = q_lower.split()
    
    # Enhanced location mapping with more detailed coverage and priorities
    location_mapping = {
        "penang": {
            "cities": ["penang", "pulau pinang", "georgetown", "bayan lepas", "bukit jambul", "jelutong", 
                      "tanjung tokong", "air itam", "bukit mertajam", "gelugor", "seberang jaya", "perai",
                      "gurney", "tanjung bungah", "batu ferringhi"],
            "priority": 0
        },
        "kuala lumpur": {
            "cities": ["kuala lumpur", "kl", "ampang", "cheras", "kuchai lama", "setapak", "klcc", 
                      "bukit bintang", "mid valley", "bangsar", "mont kiara"],
            "priority": 0
        },
        "selangor": {
            "cities": ["selangor", "petaling", "shah alam", "subang", "pj", "petaling jaya", "klang", 
                      "puchong", "damansara", "sunway", "usj"],
            "priority": 1
        },
        "kedah": {
            "cities": ["kedah", "sungai petani", "alor star", "alor setar", "sp"],
            "priority": 2
        }
    }
    
    # Extract location terms with priority scoring
    matched_locations = []
    for state, data in location_mapping.items():
        for city in data["cities"]:
            if city in q_lower:
                matched_locations.append({"term": city, "state": state, "priority": data["priority"]})
    
    # Also check for any words that might be location names (additional fallback)
    additional_terms = []
    for word in words:
        if len(word) > 3 and word not in ["where", "find", "can", "the", "bhb", "store", "location", "branch"]:
            additional_terms.append(word)
    
    # If we found specific location terms, search for matching stores with intelligent scoring
    if matched_locations:
        # Sort by priority (major cities first)
        matched_locations.sort(key=lambda x: x["priority"])
        
        for store in store_locations:
            # Create comprehensive text for matching
            store_text = " ".join([
                store.get("label", ""),
                store.get("address", ""),
                store.get("city", ""),
                store.get("state", "")
            ]).lower()
            
            # Also create individual field matches for better accuracy
            store_city = store.get("city", "").lower()
            store_state = store.get("state", "").lower()
            store_label = store.get("label", "").lower()
            
            # Check if store matches any location term with multiple matching strategies
            for loc in matched_locations:
                loc_term = loc["term"]
                # Check various field matches
                if (loc_term in store_text or  # General text match
                    loc_term == store_city or  # Exact city match
                    loc_term in store_label or  # Label contains location
                    (loc["state"] == "kuala lumpur" and "kuala lumpur" in store_state)):  # State match
                    # Add priority score for intelligent sorting
                    store_copy = store.copy()
                    store_copy["_priority"] = loc["priority"]
                    results.append(store_copy)
                    break
        
        # Sort results by priority for better relevance
        results.sort(key=lambda x: x["_priority"])
        # Remove priority field for clean return
        for store in results:
            del store["_priority"]
    
    # Fallback: check additional terms if no matched locations
    elif additional_terms:
        for store in store_locations:
            store_text = " ".join([
                store.get("label", ""),
                store.get("address", ""),
                store.get("city", ""),
                store.get("state", "")
            ]).lower()
            
            if any(term in store_text for term in additional_terms):
                results.append(store)
    
    # If no specific location matches, try broader state matching
    if not results and matched_locations:
        for store in store_locations:
            store_state = store.get("state", "")
            store_state_lower = store_state.lower()
            # Try to match by state if city not found
            for loc in matched_locations:
                # Handle different state name variations
                loc_state_lower = loc["state"].lower()
                if (loc_state_lower == store_state_lower or 
                    (loc_state_lower == "kuala lumpur" and "kuala lumpur" in store_state_lower) or
                    (loc_state_lower == "wp kuala lumpur" and "kuala lumpur" in store_state_lower)):
                    results.append(store)
                    break
    
    # Final fallback: return most relevant stores based on major areas
    # But only if we didn't find any specific location matches
    if not results:
        # Return stores from major areas first (Penang, KL, Selangor)
        major_stores = [s for s in store_locations if s.get("state") in ["Penang", "Kuala Lumpur", "Selangor"]]
        if major_stores:
            results = major_stores[:5]  # Return up to 5 stores from major areas
        else:
            results = store_locations[:3]  # Ultimate fallback
    
    # If we have specific location matches, prioritize those and filter by relevance
    elif matched_locations and len(results) > 0:
        # We found specific matches, so we should prioritize and potentially filter further
        # If user asked for a specific state, filter to only that state
        target_states = [loc["state"] for loc in matched_locations]
        if target_states:
            state_filtered = []
            for store in results:
                store_state = store.get("state", "")
                store_state_lower = store_state.lower()
                for target_state in target_states:
                    # Map target states to actual store state names
                    if (target_state.lower() in store_state_lower or 
                        (target_state == "kuala lumpur" and ("kuala lumpur" in store_state_lower or "wp kuala lumpur" in store_state_lower)) or
                        (target_state == "penang" and "penang" in store_state_lower)):
                        state_filtered.append(store)
                        break
            if state_filtered:
                results = state_filtered
    
    return results

def format_store_locations_for_reply(stores: List[Dict[str, Any]]) -> str:
    """Enhanced formatting of store locations with comprehensive business information."""
    if not stores:
        return "I couldn't find any BHB stores matching your location. Please check our Store Locator at bhb.com.my for all our locations."
    
    # Enhanced introduction based on query context
    if len(stores) == 1:
        reply_parts = ["I found a BHB store for you:"]
    else:
        reply_parts = ["Here are BHB stores that match your location:"]
    
    # Enhanced store formatting with better information organization
    for i, store in enumerate(stores[:4]):  # Show max 4 stores for better coverage
        label = store.get("label", "")
        brand_shop = store.get("brand_shop")
        address = store.get("address", "")
        tel = store.get("tel", "")
        hours = store.get("hours", "")
        city = store.get("city", "")
        state = store.get("state", "")
        
        # Build enhanced store info
        store_info = f"**{label}**"
        if brand_shop:
            store_info += f" - {brand_shop}"
        
        # Enhanced address formatting
        full_address = address
        if city and city not in address:
            full_address += f", {city}"
        if state and state not in address:
            full_address += f", {state}"
        store_info += f"\n📍 {full_address}"
        
        # Contact information
        if tel:
            store_info += f"\n📞 {tel}"
        
        # Business hours with better formatting
        if hours:
            # Clean up hours format
            clean_hours = hours.replace("\n", " | ").strip()
            if len(clean_hours) > 10:  # Only show if substantial hours info
                store_info += f"\n🕐 {clean_hours}"
        
        # Add special notes or services if available
        if "services" in store and store["services"]:
            services = store["services"][:2]  # Limit to 2 services
            store_info += f"\n✓ Services: {', '.join(services)}"
        
        reply_parts.append(store_info)
        
        # Add spacing between stores
        if i < len(stores) - 1 and i < 3:
            reply_parts.append("")
    
    # Enhanced footer with additional information
    if len(stores) > 4:
        reply_parts.append(f"\n📍 **{len(stores) - 4} more locations available**")
    
    # Add helpful information
    reply_parts.extend([
        "",
"💡 **Need help?** Visit bhb.com.my for our complete store locator, or call our customer service for assistance.",
        "🛒 **Shopping tip**: Check product availability online before visiting!"
    ])
    
    return "\n".join(reply_parts)

# --- Demo products ingestion ---
def _normalize_demo_products(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize uploaded demo product records to the canonical product schema used by the app.

    Canonical keys we try to provide: name, brand, category, features(list), price, currency,
    permalink, image_url. Missing optional fields are allowed.
    """
    norm: List[Dict[str, Any]] = []
    for e in entries or []:
        try:
            name = e.get("name") or e.get("title")
            if not name:
                continue
            brand = e.get("brand") or e.get("manufacturer")
            category = e.get("category") or e.get("type")
            features = e.get("key_features") or e.get("features") or []
            # Ensure features is a simple list of strings
            if isinstance(features, dict):
                features = [f"{k}: {v}" for k, v in features.items()]
            elif isinstance(features, list):
                features = [str(x) for x in features if x is not None]
            else:
                features = [str(features)] if features else []

            permalink = e.get("url") or e.get("permalink") or e.get("link")
            image_url = e.get("image") or e.get("image_url") or e.get("thumbnail") or e.get("img")
            price = e.get("price")
            currency = e.get("currency") or "RM"

            norm.append({
                "name": name,
                "brand": brand,
                "category": category,
                "features": features,
                "price": price,
                "currency": currency,
                "permalink": permalink,
                "image_url": image_url,
            })
        except Exception:
            # Skip malformed entries defensively
            continue
    return norm

def _load_demo_products_from_data_dir() -> List[Dict[str, Any]]:
    """Load demo product JSONs placed in DATA_DIR and normalize them.

    Supports files named like 'bhb_demo_products_rich.json' or 'bhb_demo_products_rich (1).json'.
    Also loads from 'bhb_demo_products.json' for backward compatibility.
    """
    items: List[Dict[str, Any]] = []
    try:
        # Load from rich demo products
        for p in DATA_DIR.glob("bhb_demo_products_rich*.json"):
            data = load_json(p)
            if isinstance(data, dict) and isinstance(data.get("products"), list):
                items.extend(_normalize_demo_products(data.get("products") or []))
            elif isinstance(data, list):
                items.extend(_normalize_demo_products(data))
        
        # Also load from standard demo products (bhb_demo_products.json)
        demo_products_path = DATA_DIR / "bhb_demo_products.json"
        if demo_products_path.exists():
            data = load_json(demo_products_path)
            if isinstance(data, list):
                items.extend(_normalize_demo_products(data))
    except Exception:
        # Non-fatal: just return whatever we loaded
        pass
    return items

# Merge uploaded demo products into the canonical products list, avoiding duplicates by (name, brand)
try:
    _demo_products = _load_demo_products_from_data_dir()
    if _demo_products:
        existing_keys = {
            (str(x.get("name")).strip().lower(), str(x.get("brand") or "").strip().lower())
            for x in (products or []) if x.get("name")
        }
        for dp in _demo_products:
            key = (str(dp.get("name")).strip().lower(), str(dp.get("brand") or "").strip().lower())
            if key not in existing_keys:
                products.append(dp)
except Exception:
    # Continue without demo ingestion if anything goes wrong
    pass

class ChatRequest(BaseModel):
    session_id: str
    # Domain is optional now; default to unified smart_support assistant
    domain: Optional[str] = "smart_support"  # legacy: "customer_support" | "staff_training" | "shopping_assistant" | "smart_support"
    message: str
    # When true, keep responses brief (fewer KB snippets, trimmed text)
    concise: Optional[bool] = True

def ensure_session(session_id: str, domain: str) -> Dict[str, Any]:
    sess = sessions.setdefault(session_id, {"domain": domain, "history": []})
    if sess["domain"] != domain:
        sess["domain"] = domain
        sess["history"] = []
    return sess

async def use_openai(prompt: str) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for an electrical retail store."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None

def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
        return [d.embedding for d in resp.data]
    except Exception:
        return None

def cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

def ensure_kb_embed_cache(limit: int = 200) -> List[Dict[str, Any]]:
    global _kb_embed_items
    if _kb_embed_items:
        return _kb_embed_items
    items = []
    try:
        raw_items = (kb_embeddings.get("items") or []) if isinstance(kb_embeddings, dict) else []
        # Only accept prebuilt items that have embeddings
        valid = [it for it in raw_items if isinstance(it.get("embedding"), list) and it.get("embedding")]
        if valid:
            _kb_embed_items = valid
            return _kb_embed_items
    except Exception:
        pass
    # Fallback: lazily embed KB index snippets (limited)
    if not kb_index:
        return []
    texts = []
    base_entries = kb_index[:limit]
    for e in base_entries:
        snippet = e.get("snippet") or ""
        texts.append(snippet)
    vecs = embed_texts(texts) or []
    built = []
    for e, v in zip(base_entries, vecs):
        built.append({
            "id": e.get("id"),
            "title": e.get("title"),
            "path": e.get("path"),
            "chunk_index": 0,
            "text": e.get("snippet"),
            "embedding": v,
        })
    _kb_embed_items = built
    return _kb_embed_items

class RagRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 5
    # When true, reduce KB sources and shorten reply
    concise: Optional[bool] = True

def top_products_for_message(message: str, limit: int = 5) -> List[Dict[str, Any]]:
    message_l = message.lower() if isinstance(message, str) else ""

    # Basic category intent detection to avoid cross-category suggestions
    def detect_category_intent(text: str) -> Optional[str]:
        t = (text or "").lower()
        # Synonyms mapping (extendable)
        synonyms = {
            "TV": ["tv", "television", "oled", "uled", "lcd", "led", "smart tv", "screen"],
            "Washer": ["washer", "washing machine", "laundry", "front-load", "top-load", "wash", "spin"],
            "Refrigerator": ["fridge", "refrigerator", "freezer", "chiller"],
            "Air Conditioner": ["aircon", "air conditioner", "ac", "cooling unit"],
            "Microwave": ["microwave", "oven"],
            "Dishwasher": ["dishwasher"],
            "Dryer": ["dryer", "tumble"],
            "Vacuum": ["vacuum", "cordless", "stick"],
            "Coffee Maker": ["coffee", "espresso", "machine"],
            "Kettle": ["kettle", "electric kettle"],
            # New small appliances and fixtures
            "Fan": ["fan", "ceiling fan", "bayu", "stand fan"],
            "Water Heater": ["water heater", "instant heater", "storage heater", "shower heater"],
            "Rice Cooker": ["rice cooker"],
            "Air Fryer": ["air fryer"],
        }
        best_cat = None
        best_hits = 0
        for cat, keys in synonyms.items():
            hits = sum(1 for k in keys if k in t)
            if hits > best_hits:
                best_cat = cat
                best_hits = hits
        return best_cat

    intent_cat = detect_category_intent(message_l)
    brand_intent = extract_brand_intent(message_l)
    tokens = set(parse_keywords(message_l))
    energy_intent = any(k in message_l for k in ["energy", "efficient", "efficiency", "inverter", "5-star", "star rating", "energy-saving", "econavi"])
    budget = parse_budget(message_l)
    size_in = parse_size_inch(message_l)
    # Candidate set: filter by intent category if present
    candidates = products or []
    if intent_cat:
        filtered = []
        
        # Create a mapping of intent categories to product category patterns
        category_patterns = {
            "Washer": ["washer", "washing machine", "laundry", "wash"],
            "TV": ["tv", "television"],
            "Refrigerator": ["fridge", "refrigerator", "freezer"],
            "Air Conditioner": ["air conditioner", "aircon", "ac"],
            "Microwave": ["microwave"],
            "Dishwasher": ["dishwasher"],
            "Dryer": ["dryer"],
            "Vacuum": ["vacuum"],
            "Coffee Maker": ["coffee", "espresso"],
            "Kettle": ["kettle"],
            "Fan": ["fan"],
            "Water Heater": ["water heater", "heater"],
            "Rice Cooker": ["rice cooker"],
            "Air Fryer": ["air fryer"],
        }
        
        intent_patterns = category_patterns.get(intent_cat, [intent_cat.lower()])
        
        for p in candidates:
            cat = (p.get("category") or "").lower()
            name = p.get("name", "").lower()
            
            # Check if product matches the intent category
            matches = False
            for pattern in intent_patterns:
                if pattern in cat or pattern in name:
                    matches = True
                    break
            
            if matches:
                filtered.append(p)
        if intent_cat == "Air Conditioner" and filtered:
            filtered = [p for p in filtered if not any(x in (p.get("name") or "").lower() or x in (p.get("category") or "").lower() for x in ["purifier", "air purifier", "humidifier"])]
        
        # If we found matching products, use them. Otherwise, fall back to all products
        # but boost the scoring for products that match the intent
        if filtered:
            candidates = filtered
        # If no local candidates matched the intent, try live store search then fall back to samples
        if not candidates:
            try:
                q = (message_l or intent_cat or "").strip()
                # Add category cue words to help store search rank
                category_cues = {
                    "TV": ["tv", "4k", "smart"],
                    "Washer": ["washing machine", "washer"],
                    "Refrigerator": ["refrigerator", "fridge", "inverter"],
                    "Air Conditioner": ["aircond", "air conditioner", "ac", "inverter"],
                    "Microwave": ["microwave"],
                    "Dishwasher": ["dishwasher"],
                    "Dryer": ["dryer"],
                    "Vacuum": ["vacuum"],
                    "Coffee Maker": ["coffee", "espresso"],
                    "Kettle": ["kettle"],
                    "Fan": ["fan", "ceiling"],
                    "Water Heater": ["water", "heater", "instant"],
                    "Rice Cooker": ["rice", "cooker"],
                    "Air Fryer": ["air", "fryer"],
                }
                cues = " ".join(category_cues.get(intent_cat, []))
                if brand_intent:
                    q = (brand_intent + " " + q + " " + cues).strip()
                else:
                    q = (q + " " + cues).strip()
                wc_store_results = woocommerce_store_search(q) or []
            except Exception:
                wc_store_results = []
            # Filter store results to probable intent category hits using name keywords
            if wc_store_results:
                syn_map = {
                    "TV": ["tv", "television"],
                    "Washer": ["washer", "washing", "laundry"],
                    "Refrigerator": ["fridge", "refrigerator", "freezer"],
                    "Air Conditioner": ["aircond", "air conditioner", "ac"],
                    "Microwave": ["microwave"],
                    "Dishwasher": ["dishwasher"],
                    "Dryer": ["dryer"],
                    "Vacuum": ["vacuum"],
                    "Coffee Maker": ["coffee", "espresso"],
                    "Kettle": ["kettle"],
                    "Fan": ["fan"],
                    "Water Heater": ["water", "heater"],
                    "Rice Cooker": ["rice", "cooker"],
                    "Air Fryer": ["air", "fryer"],
                }
                kws = set(syn_map.get(intent_cat, []))
                if kws:
                    wc_store_results = [r for r in wc_store_results if any(k in (r.get("name") or "").lower() for k in kws)]
                if brand_intent:
                    wc_store_results = [r for r in wc_store_results if brand_intent in (r.get("name") or "").lower()]
                if intent_cat == "Air Conditioner":
                    wc_store_results = [r for r in wc_store_results if not any(x in (r.get("name") or "").lower() for x in ["purifier", "air purifier", "humidifier"])]
            if wc_store_results:
                # Map to candidate shape similar to local catalog
                mapped = []
                for r in wc_store_results[:max(1, limit)]:
                    mapped.append({
                        "name": r.get("name"),
                        "brand": r.get("brand") or "",
                        "category": intent_cat,
                        "features": ["Store Item"],
                        "price": r.get("price"),
                        "currency": r.get("currency") or "RM",
                        "permalink": r.get("permalink"),
                        "link": r.get("permalink"),
                    })
                candidates = mapped
            else:
                # Only seed refrigerator samples when fridge intent, avoid irrelevant cross-category
                if intent_cat.lower() == "refrigerator":
                    candidates = [
                        {
                            "name": "Hisense 330L Inverter Refrigerator",
                            "brand": "Hisense",
                            "category": "Refrigerator",
                            "features": ["Inverter", "Energy Efficient", "5-star"],
                            "price": 1599.0,
                            "currency": "RM",
                        },
                        {
                            "name": "Samsung 400L Digital Inverter Fridge",
                            "brand": "Samsung",
                            "category": "Refrigerator",
                            "features": ["Digital Inverter", "Energy Saving", "Door Alarm"],
                            "price": 2199.0,
                            "currency": "RM",
                        },
                        {
                            "name": "Panasonic 350L Econavi Refrigerator",
                            "brand": "Panasonic",
                            "category": "Refrigerator",
                            "features": ["Econavi", "Inverter", "Energy Efficient"],
                            "price": 1999.0,
                            "currency": "RM",
                        },
                    ]

    # General fallback: if no intent or still empty, try store search by message
    if not candidates:
        try:
            wc_store_results = woocommerce_store_search(message_l) or []
        except Exception:
            wc_store_results = []
        if not wc_store_results:
            try:
                wc_store_results = woocommerce_store_list(per_page=limit) or []
            except Exception:
                wc_store_results = []
        if wc_store_results:
            mapped = []
            for r in wc_store_results[:max(1, limit)]:
                mapped.append({
                    "name": r.get("name"),
                    "brand": r.get("brand") or "",
                    "category": intent_cat or "Store",
                    "features": ["Store Item"],
                    "price": r.get("price"),
                    "currency": r.get("currency") or "RM",
                    "permalink": r.get("permalink"),
                    "link": r.get("permalink"),
                })
            candidates = mapped

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for p in candidates:
        name = (p.get("name") or "")
        brand = (p.get("brand") or "")
        category = (p.get("category") or "")
        features = " ".join(p.get("features") or [])
        composite = f"{name} {brand} {category} {features}".strip().lower()
        fuzzy_score = fuzz.token_set_ratio(message_l, composite)
        token_boost = sum(1 for t in tokens if t and t in composite)
        # Small boost if category intent matches this product
        cat_boost = 10.0 if intent_cat and intent_cat.lower() in (category or "").lower() else 0.0
        # Boost energy-efficiency signals (helps fridge queries)
        energy_signals = ["inverter", "energy", "efficient", "economy", "econavi", "digital inverter", "star", "a+", "a++", "eco"]
        eff_boost = sum(1 for k in energy_signals if k in composite) * 3.0
        score = float(fuzzy_score) + float(token_boost * 5) + cat_boost
        score += eff_boost
        if intent_cat == "Air Conditioner" and any(x in composite for x in ["purifier", "humidifier"]):
            score -= 50
        if energy_intent:
            has_energy = any(k in composite for k in energy_signals)
            if not has_energy:
                score -= 18
        if size_in:
            msz = re.search(r"\b(\d{2,3})\b", name.lower())
            if msz:
                try:
                    pf = int(msz.group(1))
                    if abs(pf - size_in) >= 6:
                        score -= 15
                except Exception:
                    pass
        if budget and p.get("price") is not None:
            try:
                pr = float(p.get("price"))
            except Exception:
                pr = None
            pc = p.get("currency") or "RM"
            if pr is not None:
                price_rm = pr if pc == "RM" else pr * 4.5
                budget_rm = budget["amount"] * (4.5 if budget["currency"] == "USD" else 1.0)
                if price_rm > budget_rm:
                    score -= 20
        if "4k" in message_l and ("4k" not in composite):
            score -= 12
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    result: List[Dict[str, Any]] = []
    for sc, p in scored[: max(1, limit)]:
        if sc < 40:
            continue
        # Use the format_product_for_display function for clean formatting
        formatted_product = format_product_for_display({
            "name": p.get("name"),
            "brand": p.get("brand"),
            "price": p.get("price"),
            "price_rm": p.get("price_rm"),  # For BHB products
            "currency": p.get("currency") or "RM",
            "link": p.get("link") or p.get("permalink") or "",
            "features": p.get("features") or [],
            "category": p.get("category"),
        })
        result.append(formatted_product)
    # Final safeguard: avoid empty UI by seeding category-specific or store-based fallbacks
    if not result:
        base = (WC_API_URL or "https://www.bhb.com.my").rstrip('/')
        # Minimal category samples to ensure relevance
        samples_map: Dict[str, List[Dict[str, Any]]] = {
            "Vacuum": [
                {"name": "Cordless Stick Vacuum 25.2V", "brand": "Acme", "category": "Vacuum", "features": ["Cordless", "Stick"], "price": 699.0, "currency": "RM"},
                {"name": "Bagless Cyclone Vacuum 1500W", "brand": "Zenith", "category": "Vacuum", "features": ["Bagless", "Cyclone"], "price": 499.0, "currency": "RM"},
            ],
            "Microwave": [
                {"name": "20L Microwave with Grill", "brand": "Acme", "category": "Microwave", "features": ["Grill", "Timer"], "price": 489.0, "currency": "RM"},
                {"name": "28L Convection Microwave", "brand": "Zenith", "category": "Microwave", "features": ["Convection", "Multi-stage"], "price": 729.0, "currency": "RM"},
            ],
            "Dishwasher": [
                {"name": "Compact Countertop Dishwasher", "brand": "Acme", "category": "Dishwasher", "features": ["Compact", "6 Programs"], "price": 1499.0, "currency": "RM"},
            ],
            "Dryer": [
                {"name": "7kg Condenser Dryer", "brand": "Acme", "category": "Dryer", "features": ["Condenser", "Anti-crease"], "price": 1799.0, "currency": "RM"},
            ],
            "Coffee Maker": [
                {"name": "Espresso Machine 15-bar", "brand": "BaristaPro", "category": "Coffee Maker", "features": ["15-bar", "Steam Wand"], "price": 899.0, "currency": "RM"},
            ],
            "Kettle": [
                {"name": "1.7L Electric Kettle Stainless", "brand": "Acme", "category": "Kettle", "features": ["Auto shut-off", "Boil-dry protection"], "price": 129.0, "currency": "RM"},
            ],
        }
        samples = samples_map.get(intent_cat or "", [])
        if samples:
            for s in samples[: max(1, limit)]:
                q = quote_plus(str(s.get("name") or ""))
                link = f"{base}/?s={q}&post_type=product"
                formatted_product = format_product_for_display({
                    "name": s.get("name"),
                    "brand": s.get("brand"),
                    "price": s.get("price"),
                    "currency": s.get("currency") or "RM",
                    "link": link,
                    "features": s.get("features") or [],
                    "category": s.get("category"),
                })
                result.append(formatted_product)
        else:
            # Generic store list fallback as a last resort
            generic: List[Dict[str, Any]] = []
            try:
                generic = woocommerce_store_list(per_page=limit) or []
            except Exception:
                generic = []
            if generic:
                for r in generic[: max(1, limit)]:
                    formatted_product = format_product_for_display({
                        "name": r.get("name"),
                        "brand": r.get("brand") or "",
                        "price": r.get("price"),
                        "currency": r.get("currency") or "RM",
                        "link": r.get("permalink") or "",
                        "features": r.get("features") or [],
                        "category": intent_cat or "Store",
                    })
                    result.append(formatted_product)
            elif products:
                pass
    return result

def best_match(query: str, candidates: List[Dict[str, Any]], key: str, limit: int = 3) -> List[Dict[str, Any]]:
    names = [c[key] for c in candidates]
    matches = process.extract(query, names, scorer=fuzz.token_set_ratio, limit=limit)
    out = []
    for name, score, idx in matches:
        item = candidates[idx].copy()
        item["_match_score"] = score
        out.append(item)
    return out

# --- new helpers: improve intent parsing for shopping ---
def parse_budget(message: str) -> Optional[Dict[str, float]]:
    """
    Extract budget like: "under rm2000", "below 600", "$700 max", "budget 1500".
    Returns dict with 'amount' and 'currency' ('RM' or 'USD' default).
    """
    text = message.lower()
    # RM patterns
    rm_match = re.search(r"(rm)\s*([0-9][0-9,\.]*)", text)
    usd_match = re.search(r"\$\s*([0-9][0-9,\.]*)", text)
    plain_match = re.search(r"(under|below|max|budget)\s*([0-9][0-9,\.]*)", text)
    if rm_match:
        amt = float(rm_match.group(2).replace(",", ""))
        return {"amount": amt, "currency": "RM"}
    if usd_match:
        amt = float(usd_match.group(1).replace(",", ""))
        return {"amount": amt, "currency": "USD"}
    if plain_match:
        amt = float(plain_match.group(2).replace(",", ""))
        # Heuristic: default to RM for Malaysia
        return {"amount": amt, "currency": "RM"}
    return None

def parse_size_inch(message: str) -> Optional[int]:
    """
    Extract screen size like 55", 65 inch, 40in.
    """
    m = re.search(r"(\d{2,3})\s*(\"|inch|in)", message.lower())
    if m:
        return int(m.group(1))
    return None

def text_contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)

# Vision helpers
async def extract_product_details_from_image(image_bytes: bytes, mime: str = "image/jpeg", filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Analyze an image using the configured vision provider (OpenAI or Gemini).
    Returns a small structured dict or None if not available.
    """
    provider = VISION_PROVIDER
    prompt = (
        "You are an appliance vision recognizer for a retail store. Respond ONLY as JSON with keys: "
        "brand (string), model (string), category (string), keywords (array of strings), visible_text (array of strings). "
        "If uncertain, leave fields empty. "
        "Classify washing machines correctly: look for front/top-load door, detergent drawer, control panel with wash options, and drum. "
        "If these are present, set category to 'washing machine' and extract any visible model code. "
        "Do NOT label as TV unless a flat panel screen, stand, bezel, and on-screen content are clearly visible."
    )

    if provider == "gemini":
        # Gemini integration
        if not GEMINI_API_KEY:
            return None
        try:
            # Import SDK at call time to avoid module import issues in some environments
            import importlib
            genai = importlib.import_module("google.generativeai")
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            image_part = {"mime_type": mime, "data": image_bytes}
            resp = model.generate_content([prompt, image_part])
            text = (resp.text or "{}").strip()
            import re
            # Strip markdown fences if present
            text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text)
            result: Dict[str, Any]
            try:
                result = json.loads(text)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        result = json.loads(m.group(0))
                    except Exception:
                        result = {"brand": "", "model": "", "category": "", "keywords": [], "visible_text": []}
                else:
                    # Fallback to an empty structured payload instead of None
                    result = {"brand": "", "model": "", "category": "", "keywords": [], "visible_text": []}
            # Apply filename-based hints to adjust misclassifications
            result = apply_filename_hints(result, filename)
            return result
        except Exception:
            # Fallback structured payload on Gemini error or import issues to avoid UI "not configured"
            res = {"brand": "", "model": "", "category": "", "keywords": [], "visible_text": []}
            return apply_filename_hints(res, filename)

    # Default to OpenAI
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You extract structured product details from retail images."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or "{}"
        txt = str(content or "").strip()
        cleaned = txt
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("` ")
            if cleaned.lower().startswith("json\n"):
                cleaned = cleaned[5:]
        parsed = None
        try:
            parsed = json.loads(cleaned)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
        result = parsed if isinstance(parsed, dict) else {"brand": "", "model": "", "category": "", "keywords": [], "visible_text": []}
        # OCR fallback: enrich brand/model if missing
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
            import io as _io
            img = Image.open(_io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(img)
            tokens = parse_keywords(text)
            if tokens:
                # brand hints
                brand_hints = {"bosch": "Bosch", "delonghi": "De'Longhi", "breville": "Breville", "philips": "Philips", "siemens": "Siemens", "gaggia": "Gaggia", "hisense": "Hisense", "samsung": "Samsung", "sony": "Sony", "lg": "LG", "panasonic": "Panasonic", "toshiba": "Toshiba", "sharp": "Sharp", "haier": "Haier"}
                if not (str(result.get("brand") or "").strip()):
                    for k, v in brand_hints.items():
                        if k in tokens:
                            result["brand"] = v
                            break
                # model candidates: alnum tokens with digits
                def _is_model(tok: str) -> bool:
                    if len(tok) < 4 or len(tok) > 24:
                        return False
                    has_digit = any(ch.isdigit() for ch in tok)
                    has_alpha = any(ch.isalpha() for ch in tok)
                    return has_digit and has_alpha
                model_candidates = [t for t in tokens if _is_model(t)]
                if not (str(result.get("model") or "").strip()) and model_candidates:
                    model_candidates.sort(key=len, reverse=True)
                    result["model"] = model_candidates[0]
                vt = result.get("visible_text") or []
                if isinstance(vt, list):
                    merged = list(vt)
                    for t in tokens:
                        if t not in merged:
                            merged.append(t)
                    result["visible_text"] = merged[:100]
        except Exception:
            pass
        return apply_filename_hints(result, filename)
    except Exception:
        res = {"brand": "", "model": "", "category": "", "keywords": [], "visible_text": []}
        return apply_filename_hints(res, filename)

def parse_keywords(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    for t in text.replace(",", " ").split():
        t = "".join(ch for ch in t if ch.isalnum())
        if len(t) >= 3:
            tokens.append(t)
    return tokens

def detect_category_intent(text: str) -> Optional[str]:
    t = (text or "").lower()
    synonyms = {
        "TV": ["tv", "television", "oled", "uled", "lcd", "led", "smart tv", "screen"],
        "Washer": ["washer", "washing machine", "laundry", "front-load", "top-load", "wash", "spin"],
        "Refrigerator": ["fridge", "refrigerator", "freezer", "chiller"],
        "Air Conditioner": ["aircon", "air conditioner", "ac", "cooling unit"],
        "Microwave": ["microwave", "oven"],
        "Dishwasher": ["dishwasher"],
        "Dryer": ["dryer", "tumble"],
        "Vacuum": ["vacuum", "cordless", "stick"],
        "Coffee Maker": ["coffee", "espresso", "machine"],
        "Kettle": ["kettle", "electric kettle"],
        "Fan": ["fan", "ceiling fan", "bayu", "stand fan"],
        "Water Heater": ["water heater", "instant heater", "storage heater", "shower heater"],
        "Rice Cooker": ["rice cooker"],
        "Air Fryer": ["air fryer"],
    }
    best_cat = None
    best_hits = 0
    for cat, keys in synonyms.items():
        hits = sum(1 for k in keys if k in t)
        if hits > best_hits:
            best_cat = cat
            best_hits = hits
    return best_cat

def is_front_top_compare(text: str) -> bool:
    t = (text or "").lower()
    # Match hyphen variants: -, ‑, –, — and spaces
    import re
    front_pat = re.search(r"\bfront[\-\u2010\u2011\u2012\u2013\u2014 ]?load\b", t) or ("frontload" in t)
    top_pat = re.search(r"\btop[\-\u2010\u2011\u2012\u2013\u2014 ]?load\b", t) or ("topload" in t)
    has_compare = any(k in t for k in ["difference", "compare", "vs", "versus"])
    has_washer = any(k in t for k in ["washer", "washing machine", "washr", "washing"])
    return bool((front_pat and top_pat) or (has_compare and (front_pat or top_pat)) or (has_compare and has_washer))

def is_store_query(text: str) -> bool:
    t = (text or "").lower()
    cues = [
        "store", "shop", "outlet", "branch", "cawangan", "kedai",
        "location", "lokasi", "alamat", "near", "near me", "dekat", "berhampiran",
        "address", "where", "mana"
    ]
    return any(c in t for c in cues)

def front_vs_top_explanation() -> str:
    parts = []
    parts.append("Front-load washers clean more efficiently and are gentler on clothes.")
    parts.append("They use less water and spin faster, so drying time is shorter.")
    parts.append("Top-load washers are simpler to load, often faster cycles, and cheaper upfront.")
    parts.append("Top-load models may use more water; agitator types can be rougher on fabrics.")
    parts.append("Front-load requires door clearance and occasional gasket cleaning to prevent odor.")
    parts.append("If you prioritize efficiency and fabric care, choose front-load; for ease and budget, choose top-load.")
    return "\n".join(parts)

def apply_filename_hints(details: Dict[str, Any], filename: Optional[str]) -> Dict[str, Any]:
    """Augment extracted details using hints from the uploaded filename.
    Helps correct misclassification (e.g., washer mislabeled as TV) and enrich keywords.
    """
    try:
        if not filename:
            return details
        name = str(filename)
        base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        base_no_ext = re.sub(r"\.[A-Za-z0-9]+$", "", base)
        tokens = parse_keywords(base_no_ext)
        kw_set = set(details.get("keywords", []) or [])
        # Heuristics for washing machine
        washer_signals = {"washer", "washing", "wash", "laundry", "frontload", "topload", "inverter", "kg"}
        fridge_signals = {"fridge", "refrigerator", "freezer", "chiller"}
        aircon_signals = {"aircon", "air", "conditioner", "ac", "cooling"}
        fan_signals = {"fan", "ceiling", "stand", "bayu"}
        heater_signals = {"heater", "waterheater", "instant", "storage"}
        rice_signals = {"rice", "cooker"}
        fryer_signals = {"air", "fryer"}
        has_washer_hint = any(t in tokens for t in washer_signals) or any("kg" in t for t in tokens)
        # Coffee machine hints and brand tokens
        coffee_signals = {"coffee", "espresso", "latte", "barista", "americano"}
        if any(t == "boschcoffee" for t in tokens):
            tokens.extend(["bosch", "coffee"])  # expand compound token
        has_coffee_hint = any(t in tokens for t in coffee_signals)
        has_fridge_hint = any(t in tokens for t in fridge_signals)
        has_aircon_hint = any(t in tokens for t in aircon_signals)
        has_fan_hint = any(t in tokens for t in fan_signals)
        has_heater_hint = any(t in tokens for t in heater_signals)
        has_rice_hint = any(t in tokens for t in rice_signals)
        has_fryer_hint = any(t in tokens for t in fryer_signals)
        brand_hints = {
            "bosch": "Bosch",
            "delonghi": "De'Longhi",
            "breville": "Breville",
            "philips": "Philips",
            "siemens": "Siemens",
            "gaggia": "Gaggia",
            "hisense": "Hisense",
            "samsung": "Samsung",
            "sony": "Sony",
            "lg": "LG",
        }
        # Extract possible model-like tokens (must contain a digit)
        model_candidates: List[str] = []
        for part in re.split(r"[^A-Za-z0-9-]", base_no_ext):
            part = part.strip()
            if len(part) >= 4 and any(ch.isdigit() for ch in part):
                # Avoid trivial tokens
                if part.lower() not in {"washing", "washer", "laundry", "frontload", "topload"}:
                    model_candidates.append(part)
        # Merge keywords
        for t in tokens:
            if t and t not in kw_set:
                kw_set.add(t)
        # If filename hints strongly suggest washer, adjust category
        category = (details.get("category") or "").strip().lower()
        if has_washer_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "washing machine"
        if has_fridge_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "refrigerator"
        if has_aircon_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "air conditioner"
        if has_fan_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "fan"
        if has_heater_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "water heater"
        if has_rice_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "rice cooker"
        if has_fryer_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "air fryer"
        # If filename suggests coffee appliance, adjust category
        if has_coffee_hint:
            if not category or ("tv" in category or "television" in category):
                details["category"] = "coffee machine"
        # If model is empty, try to use a candidate from filename
        if not (details.get("model") or "").strip():
            if model_candidates:
                # Prefer longest candidate that includes letters and digits
                model_candidates.sort(key=len, reverse=True)
                details["model"] = model_candidates[0]
        # Infer brand from tokens if missing
        if not (details.get("brand") or "").strip():
            for k, v in brand_hints.items():
                if k in tokens:
                    details["brand"] = v
                    break
        # Commit keywords
        details["keywords"] = list(kw_set)
        return details
    except Exception:
        return details

def extract_brand_intent(message: str) -> Optional[str]:
    """Return a brand name mentioned by the user if it appears to be a brand query.
    We check against known brands from the current catalog and common appliance brands.
    """
    text = (message or "").lower()
    kw = set(parse_keywords(text))
    # Known brands from catalog
    known_brands = set()
    for p in products or []:
        b = (p.get("brand") or "").strip().lower()
        if b:
            known_brands.add(b)
    # Common brands list (extendable)
    common = {"hisense", "sony", "samsung", "lg", "panasonic", "toshiba", "sharp", "philips", "haier"}
    known_brands |= common
    # Find intersection
    for b in sorted(known_brands):
        # match exact token or substring in text
        if b in kw or b in text:
            return b
    return None

def local_search_from_details(details: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not products:
        return []
    query_parts = [details.get("brand", ""), details.get("model", ""), details.get("category", "")]
    query = " ".join([p for p in query_parts if p]).strip()
    tokens = set(parse_keywords(query + " " + " ".join(details.get("keywords", []))))
    # If we have no signals at all, avoid returning irrelevant fuzzy matches
    if not tokens and not query:
        return []
    scored: List[Dict[str, Any]] = []
    for p in products:
        fields = " ".join([
            str(p.get("name", "")),
            str(p.get("brand", "")),
            str(p.get("category", "")),
            " ".join(p.get("features", [])),
        ]).lower()
        score = sum(1 for t in tokens if t in fields)
        if score > 0:
            item = p.copy()
            item["_score"] = score
            scored.append(item)
    scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
    if scored:
        return scored[:5]
    # fuzzy fallback on name
    return best_match(query or " ".join(tokens), products, key="name", limit=5)

def woocommerce_store_search(query: str, page: int = 1, per_page: int = 10) -> Optional[List[Dict[str, Any]]]:
    """Search products via the public WooCommerce Store API.
    Defaults to bhb.com.my if WC_API_URL is not set.
    Returns a simplified list with name, price, currency, and permalink.
    """
    # Avoid unexpected external calls when not configured
    if not WC_API_URL:
        return None
    base = WC_API_URL
    if not requests:
        return None
    try:
        url = f"{base.rstrip('/')}/wp-json/wc/store/products"
        params = {"search": query, "page": page, "per_page": per_page}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        items = resp.json()
        results = []
        for it in items:
            prices = it.get("prices", {}) or {}
            raw = prices.get("price") or prices.get("regular_price") or prices.get("sale_price")
            currency_symbol = prices.get("currency_symbol") or "$"
            amount = None
            if raw is not None:
                try:
                    # Store API returns minor units; convert to major units
                    amount = float(raw) / 100.0
                except Exception:
                    amount = None
            results.append({
                "name": it.get("name"),
                "price": amount,
                "brand": "",
                "permalink": it.get("permalink"),
                "currency": currency_symbol,
            })
        return results
    except Exception:
        return []

def woocommerce_store_list(page: int = 1, per_page: int = 10) -> Optional[List[Dict[str, Any]]]:
    base = WC_API_URL or "https://www.bhb.com.my"
    if not requests:
        return None
    try:
        url = f"{base.rstrip('/')}/wp-json/wc/store/products"
        params = {"page": page, "per_page": per_page}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        items = resp.json()
        results = []
        for it in items:
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
            results.append({
                "name": it.get("name"),
                "price": amount,
                "brand": "",
                "permalink": it.get("permalink"),
                "currency": currency_symbol,
                "image_url": image_url,
                "_source": "bhb.com.my",
            })
        return results
    except Exception:
        return []

def resolve_store_permalink(name: str, brand: Optional[str] = None) -> Optional[str]:
    """Resolve an exact product permalink via WooCommerce Store API by name.
    Uses fuzzy matching to pick the best candidate.
    Returns the permalink URL if found, otherwise None.
    """
    if not name:
        return None
    # Avoid external calls unless store API is explicitly configured
    if not WC_API_URL:
        return None
    # Ensure requests is available and API base configured
    # Try public WooCommerce Store API first
    if requests:
        try:
            candidates = woocommerce_store_search(name, per_page=10) or []
            if candidates:
                # Score candidates by fuzzy match on name and optional brand
                best_link = None
                best_score = -1
                name_lower = str(name).lower()
                brand_lower = str(brand or "").lower()
                for c in candidates:
                    cname = str(c.get("name") or "").lower()
                    score = fuzz.token_set_ratio(name_lower, cname)
                    if brand_lower and brand_lower in cname:
                        score += 5
                    if score > best_score and c.get("permalink"):
                        best_score = score
                        best_link = c.get("permalink")
                # Require a reasonable threshold to avoid wrong products
                if best_score >= 70 and best_link:
                    return best_link
        except Exception:
            pass

    # HTML search fallback for stores without WooCommerce Store API (e.g., Shopify themes)
    if not (requests and BeautifulSoup):
        # If either requests or BeautifulSoup is missing, we can't perform HTML parsing
        return None
    try:
        base = (WC_API_URL or "https://www.bhb.com.my").rstrip('/')
        # Try WooCommerce-style search first
        q = quote_plus(str(name))
        candidates: List[Tuple[str, str]] = []  # (title, href)
        for search_url in [
            f"{base}/?s={q}&post_type=product",        # WooCommerce
            f"{base}/search?q={q}",                    # Shopify
        ]:
            r = requests.get(search_url, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, 'lxml')
            # Collect product links from common patterns
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                if not href:
                    continue
                # Normalize absolute/relative
                if href.startswith('/'):
                    href_full = base + href
                elif href.startswith('http'):
                    href_full = href
                else:
                    href_full = base + '/' + href
                # Heuristics: look for product detail routes
                if ('/product/' in href_full) or ('/products/' in href_full):
                    candidates.append((text or '', href_full))
        if candidates:
            name_lower = str(name).lower()
            brand_lower = str(brand or '').lower()
            best_href = None
            best_score = -1
            for title, href in candidates:
                title_lower = str(title).lower()
                score = fuzz.token_set_ratio(name_lower, title_lower)
                if brand_lower and brand_lower in title_lower:
                    score += 5
                if score > best_score:
                    best_score = score
                    best_href = href
            if best_score >= 60 and best_href:
                return best_href
    except Exception:
        pass
    return None

def search_from_details(details: Dict[str, Any]) -> List[Dict[str, Any]]:
    query_parts = [details.get("brand", ""), details.get("model", ""), details.get("category", "")]
    query = " ".join([p for p in query_parts if p]).strip() or " ".join(details.get("keywords", []))
    # Try WooCommerce Store API (public) first
    if query:
        wc_store_results = woocommerce_store_search(query) or []
        if wc_store_results:
            return wc_store_results
    # Then try WooCommerce REST if keys are configured
    if query:
        wc_results = woocommerce_search(query) or []
        if wc_results:
            return wc_results
    # If external store queries failed, try resolving a direct product link on BHB
    try:
        candidate_name = (details.get("model") or "").strip() or query
        candidate_brand = (details.get("brand") or "").strip() or None
        link = resolve_store_permalink(candidate_name, brand=candidate_brand)
        if link:
            # Return a single clickable candidate to help the user navigate
            return [{
                "name": candidate_name or "Product",
                "brand": candidate_brand or "",
                "price": None,
                "currency": "RM",
                "permalink": link,
                "_source": "bhb.com.my",
            }]
    except Exception:
        pass
    # Fallback to local data
    return local_search_from_details(details)

def handle_shopping_assistant(message: str) -> str:
    global products
    # Parse intent signals
    budget = parse_budget(message)
    size_in = parse_size_inch(message)
    tokens = set(parse_keywords(message))
    brand_query = extract_brand_intent(message)

    # If local catalog is empty, try live WooCommerce Store API
    if not products:
        # Attempt to reload local products.json in case it was populated recently
        try:
            fresh = load_json(DATA_DIR / "products.json") or []
        except Exception:
            fresh = []
        if fresh:
            # Update global cache
            products = fresh
        
        query = " ".join(tokens) or message
        wc_store_results = woocommerce_store_search(query) or []
        if not wc_store_results:
            wc_store_results = woocommerce_store_list(per_page=10) or []
        if wc_store_results:
            tmsg = (message or "").lower()
            if any(k in tmsg for k in ["aircon", "air conditioner", "ac"]):
                wc_store_results = [r for r in wc_store_results if not any(x in (r.get("name") or "").lower() for x in ["purifier", "air purifier", "humidifier"])]
                if not wc_store_results:
                    return "We carry Daikin air conditioners. Please share HP (e.g., 1.0–2.0 HP), room size, and budget so I can recommend the right models."
            lines = []
            for p in wc_store_results[:5]:
                price = p.get("price")
                currency = p.get("currency") or ("RM" if p.get("_source") == "bhb.com.my" else "$")
                price_text = f"{currency} {price}" if price is not None else "N/A"
                s = f"- {p.get('name', 'Unknown')}"
                if p.get("brand"):
                    s += f" ({p['brand']})"
                s += f" — {price_text}"
                link = p.get("permalink")
                if not link and WC_API_URL:
                    # Try to resolve exact product page
                    link = resolve_store_permalink(p.get("name") or "", p.get("brand"))
                if link:
                    s += f" | {link}"
                lines.append(s)
            summary_bits = []
            if size_in:
                summary_bits.append(f"size ≈ {size_in}\"")
            if budget:
                summary_bits.append(f"budget ≤ {budget['currency']} {budget['amount']}")
            summary = f"Here are store items{(' filtered by ' + ', '.join(summary_bits)) if summary_bits else ''}:\n"
            return summary + "\n".join(lines)
        # Fallback: try backup catalog or seed minimal sample items
        try:
            backup = load_json(DATA_DIR / "products.bak.json") or []
        except Exception:
            backup = []
        if backup:
            products = backup
        else:
            products = [
                {
                    "name": "Acme 32\" LED TV",
                    "brand": "Acme",
                    "category": "TV",
                    "features": ["HD", "LED"],
                    "price": 899.0,
                    "currency": "RM"
                },
                {
                    "name": "Zenith 55\" 4K Smart TV",
                    "brand": "Zenith",
                    "category": "TV",
                    "features": ["4K", "Smart", "HDR"],
                    "price": 1999.0,
                    "currency": "RM"
                },
                {
                    "name": "PolarCool Energy Refrigerator",
                    "brand": "PolarCool",
                    "category": "Refrigerator",
                    "features": ["Energy Efficient", "Inverter"],
                    "price": 1599.0,
                    "currency": "RM"
                }
            ]

    # Brief clarifying question when budget is missing
    prefix_lines: List[str] = []
    if not budget:
        prefix_lines.append("Quick question: Do you have a budget in mind (e.g., under RM 2000)?")
    # Stronger fuzzy score against composite fields
    intent_cat = detect_category_intent(message)
    cat_patterns = {
        "Washer": ["washer", "washing machine", "laundry", "wash"],
        "TV": ["tv", "television"],
        "Refrigerator": ["fridge", "refrigerator", "freezer"],
        "Air Conditioner": ["air conditioner", "aircon", "ac"],
        "Microwave": ["microwave"],
        "Dishwasher": ["dishwasher"],
        "Dryer": ["dryer"],
        "Vacuum": ["vacuum"],
        "Coffee Maker": ["coffee", "espresso"],
        "Kettle": ["kettle"],
        "Fan": ["fan"],
        "Water Heater": ["water heater", "heater"],
        "Rice Cooker": ["rice cooker"],
        "Air Fryer": ["air fryer"],
    }
    candidates = products
    if intent_cat:
        pats = cat_patterns.get(intent_cat, [intent_cat.lower()])
        filt = []
        for p in candidates:
            c = (p.get("category") or "").lower()
            n = (p.get("name") or "").lower()
            if any((pat in c) or (pat in n) for pat in pats):
                filt.append(p)
        if filt:
            candidates = filt
    scored: List[Dict[str, Any]] = []
    for p in candidates:
        composite = " ".join([
            str(p.get("name", "")),
            str(p.get("brand", "")),
            str(p.get("category", "")),
            " ".join(p.get("features", [])),
        ])
        # Base fuzzy score
        fuzzy_score = fuzz.token_set_ratio(message, composite)
        # Token boost
        token_boost = sum(1 for t in tokens if t and t in composite.lower())
        score = fuzzy_score + (token_boost * 5)

        # Apply size filter if available
        if size_in:
            # Try to infer size from product name (numbers like 55 or 65)
            size_found = None
            m = re.search(r"\b(\d{2,3})\b", p.get("name", "").lower())
            if m:
                try:
                    size_found = int(m.group(1))
                except Exception:
                    size_found = None
            # penalize if far off
            if size_found is not None and abs(size_found - size_in) >= 6:
                score -= 15

        # Apply budget filter if available
        if budget and p.get("price") is not None:
            try:
                price = float(p["price"])
            except Exception:
                price = None
            product_currency = p.get("currency", "RM")
            if price is not None:
                price_rm = price if product_currency == "RM" else price * 4.5
                budget_rm = budget["amount"] * (4.5 if budget["currency"] == "USD" else 1.0)
                if price_rm > budget_rm:
                    score -= 20
        if "4k" in message.lower() and ("4k" not in composite.lower()):
            score -= 12

        if score > 0:
            item = p.copy()
            item["_score"] = score
            scored.append(item)
    scored.sort(key=lambda x: x["_score"], reverse=True)

    # If nothing meaningful, fuzzy fallback on name
    if not scored:
        matches = best_match(message, products, key="name", limit=5)
        scored = matches

    top = [p for p in scored if p.get("_score", 0) >= 40][:5]
    if not top:
        return "I didn’t find matching items. Try specifying brand, category, size, or budget (e.g., 55\" TV under RM2000)."

    lines = []
    for p in top:
        price = p.get("price")
        currency = p.get("currency") or ("RM" if p.get("_source") == "bhb.com.my" else "$")
        
        # Handle missing or zero prices
        if price is None or price == 0:
            price_text = "Price not available"
        else:
            price_text = f"{currency} {price}"
            
        s = f"- {p.get('name')} ({p.get('brand','')}) — {price_text}"
        if p.get("features"):
            s += f" | Features: {', '.join(p['features'][:3])}"
        # Prefer product permalink; otherwise provide a store search link if WC_API_URL is set
        link = p.get("permalink")
        if not link and p.get("name"):
            base = (WC_API_URL or "https://www.bhb.com.my").rstrip('/')
            query = quote_plus(str(p.get("name")))
            link = f"{base}/?s={query}&post_type=product"
        if link:
            s += f" | {link}"
        lines.append(s)

    # Add a short summary to explain the match
    summary_bits = []
    if size_in:
        summary_bits.append(f"size ≈ {size_in}\"")
    if budget:
        currency = budget['currency']
        summary_bits.append(f"budget ≤ {currency} {budget['amount']}")
    # Brand-aware acknowledgement
    ack = ""
    if brand_query:
        brand_found = any((brand_query in (p.get("brand", "").lower())) or (brand_query in (p.get("name", "").lower())) for p in top)
        if brand_found:
            ack = f"Yes, we have {brand_query.title()} items. "
        else:
            ack = f"We don’t currently list {brand_query.title()} items in the catalog. Here are close alternatives. "
    prefix = ("\n\n".join(prefix_lines) + ("\n\n" if prefix_lines else ""))
    summary = prefix + ack + f"Here are some options{(' filtered by ' + ', '.join(summary_bits)) if summary_bits else ''}:\n"

    return summary + "\n".join(lines)

def handle_shopping_assistant_payload(message: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return a friendly summary plus a list of product items with buy links.
    Detect bundle/multi-category and return grouped plan immediately.
    """
    if _is_bundle_intent(message) or _has_multi_category_mentions(message):
        reply, items = build_bundle_reply(message)
        return reply, items
    # Reuse the logic by calling handle_shopping_assistant to build the summary string
    summary = handle_shopping_assistant(message)
    # Build items by running the same scoring pipeline quickly against current products
    budget = parse_budget(message)
    size_in = parse_size_inch(message)
    tokens = set(parse_keywords(message))
    intent_cat = detect_category_intent(message)
    category_patterns = {
        "Washer": ["washer", "washing machine", "laundry", "wash"],
        "TV": ["tv", "television"],
        "Refrigerator": ["fridge", "refrigerator", "freezer"],
        "Air Conditioner": ["air conditioner", "aircon", "ac"],
        "Microwave": ["microwave"],
        "Dishwasher": ["dishwasher"],
        "Dryer": ["dryer"],
        "Vacuum": ["vacuum"],
        "Coffee Maker": ["coffee", "espresso"],
        "Kettle": ["kettle"],
        "Fan": ["fan"],
        "Water Heater": ["water heater", "heater"],
        "Rice Cooker": ["rice cooker"],
        "Air Fryer": ["air fryer"],
    }
    candidates = products or []
    if intent_cat:
        pats = category_patterns.get(intent_cat, [intent_cat.lower()])
        filt = []
        for p in candidates:
            c = (p.get("category") or "").lower()
            n = (p.get("name") or "").lower()
            if any((pat in c) or (pat in n) for pat in pats):
                filt.append(p)
        if filt:
            candidates = filt
    if intent_cat == "Air Conditioner":
        candidates = [p for p in candidates if not any(x in (p.get("name") or "").lower() or x in (p.get("category") or "").lower() for x in ["purifier", "air purifier", "humidifier"])]
    scored: List[Dict[str, Any]] = []
    for p in candidates:
        composite = " ".join([
            str(p.get("name", "")),
            str(p.get("brand", "")),
            str(p.get("category", "")),
            " ".join(p.get("features", [])),
        ])
        fuzzy_score = fuzz.token_set_ratio(message, composite)
        token_boost = sum(1 for t in tokens if t and t in composite.lower())
        score = float(fuzzy_score) + float(token_boost * 5)

        # Size filter heuristic
        if size_in:
            size_found = None
            m = re.search(r"\b(\d{2,3})\b", p.get("name", "").lower())
            if m:
                try:
                    size_found = int(m.group(1))
                except Exception:
                    size_found = None
            if size_found is not None and abs(size_found - size_in) >= 6:
                score -= 15

        # Budget filter heuristic
        if budget and p.get("price") is not None:
            try:
                price = float(p["price"])
            except Exception:
                price = None
            pc = p.get("currency", "RM")
            if price is not None:
                price_rm = price if pc == "RM" else price * 4.5
                budget_rm = budget["amount"] * (4.5 if budget["currency"] == "USD" else 1.0)
                if price_rm > budget_rm:
                    score -= 20
        if "4k" in message.lower() and ("4k" not in composite.lower()):
            score -= 12
        if intent_cat == "Air Conditioner" and any(x in composite.lower() for x in ["purifier", "humidifier"]):
            score -= 50

        if score > 0:
            item = p.copy()
            item["_score"] = score
            scored.append(item)
    scored.sort(key=lambda x: x.get("_score", 0), reverse=True)
    top = [x for x in scored if x.get("_score", 0) >= 40][:5]

    # Build normalized items list
    items: List[Dict[str, Any]] = []
    for p in top:
        currency = p.get("currency") or ("RM" if p.get("_source") == "bhb.com.my" else "$")
        link = p.get("permalink")
        if not link and p.get("name"):
            # Attempt resolving an exact permalink via Store API or HTML search fallback
            resolved = resolve_store_permalink(str(p.get("name")), p.get("brand"))
            if resolved:
                link = resolved
            else:
                # Fallback to a store search URL using default base
                base = (WC_API_URL or "https://www.bhb.com.my").rstrip('/')
                query = quote_plus(str(p.get("name")))
                # Prefer Shopify-style search if WooCommerce Store API isn’t reachable
                probe_wc = f"{base}/wp-json/wc/store/products?per_page=1"
                try:
                    wc_ok = False
                    if requests:
                        r = requests.get(probe_wc, timeout=5)
                        wc_ok = (r.status_code == 200)
                    if wc_ok:
                        link = f"{base}/?s={query}&post_type=product"
                    else:
                        link = f"{base}/search?q={query}"
                except Exception:
                    # Default to WooCommerce-style
                    link = f"{base}/?s={query}&post_type=product"
        # Format product for clean display before adding to items
            formatted_product = format_product_for_display({
                "name": p.get("name"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "currency": currency,
                "link": link,
                "features": p.get("features") or [],
                "category": p.get("category"),
            })
            items.append(formatted_product)

    return summary, items

def handle_customer_support(message: str) -> str:
    # Simple FAQ lookup with fuzzy match
    if not faqs:
        return "Our FAQ database is not loaded yet. Please try again later."
    try:
        questions = [f.get("question", "") for f in faqs]
        match = process.extractOne(message, questions, scorer=fuzz.token_set_ratio)
        # Only accept the FAQ match if similarity is reasonably high; otherwise fall back
        if match and match[2] is not None:
            score = match[1] if isinstance(match, (list, tuple)) and len(match) >= 2 else 0
            if (score or 0) >= 70:
                faq = faqs[match[2]]
                answer = faq.get("answer", "Sorry, I don't have an answer for that yet.")
                # Ensure we return a clean string, not any object
                return str(answer) if answer else "Sorry, I don't have an answer for that yet."
        # fallback: keyword search
        tokens = set(parse_keywords(message))
        for f in faqs:
            q = (f.get("question", "") + " " + f.get("answer", "")).lower()
            if any(t in q for t in tokens):
                answer = f.get("answer", "Sorry, I don't have an answer for that yet.")
                return str(answer) if answer else "Sorry, I don't have an answer for that yet."
        # fallback: search Knowledge Base directly when FAQ doesn't cover the query
        try:
            kb_matches = search_kb(message, limit=1)
            if kb_matches:
                kb = kb_matches[0]
                snippet = (kb.get("snippet") or "").strip()
                title = kb.get("title") or "Knowledge Base"
                if snippet:
                    # Return the KB snippet as the answer, keeping it concise
                    return str(snippet)
                # If no snippet text, at least acknowledge the KB entry
                return f"Refer to: {title}."
        except Exception:
            pass
        return "Please provide more details, or ask about returns, warranty, or delivery."
    except Exception:
        return "Sorry, something went wrong while looking up FAQs."

def _is_bundle_intent(text: str) -> bool:
    t = (text or "").lower()
    cues = [
        "bundle", "whole house", "package", "new home", "new house",
        "buy all", "set rumah", "sekali gus", "one shot", "starter pack",
    ]
    return any(c in t for c in cues)

def _has_multi_category_mentions(text: str) -> bool:
    """Detect when a message mentions multiple distinct appliance categories, e.g. 'TV, fan, aircon, washer and fridge'."""
    t = (text or "").lower()
    category_keywords = {
        "tv": ["tv", "television"],
        "washer": ["washer", "washing machine", "laundry", "washr"],
        "fridge": ["fridge", "refrigerator", "freezer"],
        "aircon": ["aircon", "air conditioner", "ac"],
        "fan": ["fan", "ceiling fan", "bayu", "stand fan"],
    }
    seen: set = set()
    for cat, keys in category_keywords.items():
        if any(k in t for k in keys):
            seen.add(cat)
    return len(seen) >= 2

def _pick_by_category(cat: str, n: int = 1) -> List[Dict[str, Any]]:
    cat_l = (cat or "").strip().lower()
    matches = [p for p in (products or []) if (p.get("category") or "").lower() == cat_l]
    # Simple sort by price ascending to pick budget-friendly defaults
    try:
        matches.sort(key=lambda x: float(x.get("price") or 0))
    except Exception:
        pass
    return matches[: max(1, n)] if matches else []

def build_bundle_reply(message: str) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = []
    lines.append("Yes — we can put together a whole-house bundle. Here’s a starter package:")
    bundle_items: List[Dict[str, Any]] = []

    # Big appliances: placeholders pending sizing/preferences
    lines.append("- Washer: sized to your household (e.g., 9–10.5kg).")
    lines.append("- Refrigerator: 300–400L inverter for 3–5 people.")
    lines.append("- TV: 55–65 inch 4K, depends on viewing distance.")
    lines.append("- Air conditioner: sized per room; inverter for efficiency.")

    # Fixtures and small appliances from local catalog
    fans = _pick_by_category("fan", n=2)
    if fans:
        lines.append("- Ceiling fans (2x):")
        for f in fans:
            lines.append(f"  • {f.get('name')} — around {f.get('currency') or 'RM'} {f.get('price')}")
            # Format product for clean display
            formatted_product = format_product_for_display({
                "name": f.get("name"),
                "brand": f.get("brand"),
                "price": f.get("price"),
                "currency": f.get("currency") or "RM",
                "link": f.get("permalink") or f.get("link"),
                "features": f.get("features") or [],
                "category": f.get("category"),
            })
            bundle_items.append(formatted_product)
    heater = _pick_by_category("water_heater", n=1)
    if heater:
        h = heater[0]
        lines.append(f"- Water heater (1x): {h.get('name')} — around {h.get('currency') or 'RM'} {h.get('price')}")
        # Format product for clean display
        formatted_product = format_product_for_display({
            "name": h.get("name"),
            "brand": h.get("brand"),
            "price": h.get("price"),
            "currency": h.get("currency") or "RM",
            "link": h.get("permalink") or h.get("link"),
            "features": h.get("features") or [],
            "category": h.get("category"),
        })
        bundle_items.append(formatted_product)
    cooker = _pick_by_category("rice_cooker", n=1)
    if cooker:
        c = cooker[0]
        lines.append(f"- Rice cooker (1x): {c.get('name')} — around {c.get('currency') or 'RM'} {c.get('price')}")
        # Format product for clean display
        formatted_product = format_product_for_display({
            "name": c.get("name"),
            "brand": c.get("brand"),
            "price": c.get("price"),
            "currency": c.get("currency") or "RM",
            "link": c.get("permalink") or c.get("link"),
            "features": c.get("features") or [],
            "category": c.get("category"),
        })
        bundle_items.append(formatted_product)
    fryer = _pick_by_category("air_fryer", n=1)
    if fryer:
        a = fryer[0]
        lines.append(f"- Air fryer (1x): {a.get('name')} — around {a.get('currency') or 'RM'} {a.get('price')}")
        # Format product for clean display
        formatted_product = format_product_for_display({
            "name": a.get("name"),
            "brand": a.get("brand"),
            "price": a.get("price"),
            "currency": a.get("currency") or "RM",
            "link": a.get("permalink") or a.get("link"),
            "features": a.get("features") or [],
            "category": a.get("category"),
        })
        bundle_items.append(formatted_product)

    lines.append("")
    lines.append("Share your budget, home type (condo/landed), and rooms, and I’ll right-size washer/fridge/TV/aircond and optimize the bundle for value.")
    return "\n".join(lines), bundle_items

def format_product_for_display(product: Dict[str, Any]) -> Dict[str, Any]:
    """Format product data for clean user display, removing technical details."""
    # Clean up the product data for better user experience
    price = product.get("price") or product.get("price_rm")
    currency = product.get("currency", "RM")
    
    # Ensure we have a valid price, fallback to 0 if missing
    if price is None or price == "":
        price = 0.0
    else:
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = 0.0
    
    formatted = {
        "name": product.get("name", ""),
        "brand": product.get("brand", ""),
        "price": price,
        "currency": currency,
        "link": product.get("link", ""),
        "category": product.get("category", ""),
    }
    
    # Clean up features to remove technical jargon and braces
    raw_features = product.get("features") or []
    if isinstance(raw_features, list):
        # Filter out overly technical features and clean up formatting
        clean_features = []
        for feature in raw_features:
            if isinstance(feature, str):
                # Remove braces, quotes, and overly technical terms
                clean_feature = feature.strip().replace("{", "").replace("}", "").replace("'", "").replace('"', '')
                # Skip if it's just technical specifications
                if not any(term in clean_feature.lower() for term in ["dimensions", "mm", "db", "watt", "hz", "sku", "model_code"]):
                    if len(clean_feature) > 10 and len(clean_feature) < 100:  # Reasonable length
                        clean_features.append(clean_feature)
        
        # Limit to 2-3 most relevant features
        formatted["features"] = clean_features[:3]
    else:
        formatted["features"] = []
    
    return formatted

def handle_smart_support(message: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Hybrid handler: answer support question and, when relevant, suggest products."""
    # Check for store location intent first
    store_keywords = [
        "location", "lokasi", "alamat", "branch", "cawangan",
        "near me", "near", "berhampiran", "dekat",
        "kedai", "shop", "store", "outlet",
    ]
    message_lower = (message or "").lower()
    has_store_signal = any(k in message_lower for k in store_keywords)
    
    # If store location query, handle it specially
    if has_store_signal:
        stores = find_stores_by_query(message)
        if stores:
            store_reply = format_store_locations_for_reply(stores)
            return store_reply, []
        else:
            # Fallback to general support if no stores found
            support_reply = handle_customer_support(message)
            return support_reply, []
    
    if is_front_top_compare(message):
        expl = front_vs_top_explanation()
        items: List[Dict[str, Any]] = []
        try:
            top = top_products_for_message(message, limit=5)
            for p in top:
                currency = p.get("currency") or ("RM" if p.get("_source") == "bhb.com.my" else "$")
                link = p.get("permalink")
                if not link and p.get("name"):
                    resolved = resolve_store_permalink(str(p.get("name")), p.get("brand"))
                    if resolved:
                        link = resolved
                formatted_product = format_product_for_display({
                    "name": p.get("name"),
                    "brand": p.get("brand"),
                    "price": p.get("price"),
                    "currency": currency,
                    "link": link,
                    "features": p.get("features") or [],
                    "category": p.get("category"),
                })
                items.append(formatted_product)
        except Exception:
            items = []
        return expl + "\n\nHere are some options:", items

    # Check for strong shopping intent - prioritize shopping over customer support
    shopping_intent_keywords = [
        "recommend", "suggest", "looking for", "buy", "under", "around",
        "budget", "price", "deal", "best", "tv", "television", "4k", "smart tv",
        "fridge", "refrigerator", "washer", "washing machine", "aircon",
        "air conditioner", "fan", "microwave", "dryer", "rice cooker",
        "water heater", "air fryer", "appliance", "electronics"
    ]
    has_shopping_intent = any(k in message_lower for k in shopping_intent_keywords)
    
    # Route bundle intent to grouped plan builder
    if _is_bundle_intent(message) or _has_multi_category_mentions(message):
        reply, items = build_bundle_reply(message)
        return reply, items
    
    # If strong shopping intent detected, prioritize shopping assistant over customer support
    if has_shopping_intent:
        try:
            reply, items = handle_shopping_assistant_payload(message)
            # Always return shopping assistant response for shopping intent, even with 0 items
            return reply, items
        except Exception:
            # Fallback to customer support if shopping assistant fails
            pass
    
    # Start with customer support reply
    support_reply = handle_customer_support(message)
    # Compute product suggestions using the same scoring as shopping assistant
    items: List[Dict[str, Any]] = []
    try:
        top = top_products_for_message(message, limit=5)
        for p in top:
            currency = p.get("currency") or ("RM" if p.get("_source") == "bhb.com.my" else "$")
            link = p.get("permalink")
            if not link and p.get("name"):
                resolved = resolve_store_permalink(str(p.get("name")), p.get("brand"))
                if resolved:
                    link = resolved
            # Format product for clean display before adding to items
            formatted_product = format_product_for_display({
                "name": p.get("name"),
                "brand": p.get("brand"),
                "price": p.get("price"),
                "currency": currency,
                "link": link,
                "features": p.get("features") or [],
                "category": p.get("category"),
            })
            items.append(formatted_product)
    except Exception:
        items = []

    # Build a combined reply that remains concise; UI will render items separately
    if items:
        combined = support_reply + "\n\nHere are a few relevant products you might consider:"
        return combined, items
    return support_reply, items

def handle_staff_training(message: str) -> str:
    # Suggest relevant training modules
    if not training_modules:
        return "Training modules are not available yet."
    try:
        titles = [m.get("title", "") for m in training_modules]
        match = process.extractOne(message, titles, scorer=fuzz.token_set_ratio)
        lines = []
        if match and match[2] is not None:
            m = training_modules[match[2]]
            lines.append(f"Module: {m.get('title','')}")
            if m.get("objectives"):
                lines.append("Objectives: " + ", ".join(m["objectives"]))
            if m.get("tips"):
                lines.append("Tips: " + ", ".join(m["tips"]))
        else:
            lines.append("Suggested modules:")
            for m in training_modules[:3]:
                lines.append("- " + m.get("title", ""))
        return "\n".join(lines)
    except Exception:
        return "Sorry, something went wrong while fetching training guidance."

# Safe stub for WooCommerce REST (used in search_from_details if enabled elsewhere)
def woocommerce_search(query: str) -> Optional[List[Dict[str, Any]]]:
    # Return None to indicate not configured (avoids NameError)
    return None

# ----- Knowledge Base Upload & Search -----
def ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def extract_text_from_file(path: Path) -> str:
    """Best-effort text extraction for simple formats.
    Supports: .txt, .md, .json (flatten), .csv, .html. Others return empty.
    """
    try:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                # Flatten common structures
                if isinstance(data, list):
                    parts = []
                    for item in data:
                        if isinstance(item, dict):
                            parts.append(" ".join(str(v) for v in item.values()))
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                if isinstance(data, dict):
                    return " ".join(str(v) for v in data.values())
                return str(data)
            except Exception:
                return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".csv":
            parts: List[str] = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    parts.append(" ".join(row))
            return "\n".join(parts)
        if ext in {".html", ".htm"}:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
                # Remove script/style
                for tag in soup(["script", "style"]):
                    tag.decompose()
                return soup.get_text(" ", strip=True)
            except Exception:
                return path.read_text(encoding="utf-8", errors="ignore")
        # Unsupported: return empty (we still store the file)
        return ""
    except Exception:
        return ""

def save_kb_index(index: List[Dict[str, Any]]) -> None:
    try:
        with (DATA_DIR / "kb_index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def search_kb(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    if not kb_index or not query:
        return []
    corpus = []
    valid_indices = []
    
    # Filter out product-related KB entries
    for idx, item in enumerate(kb_index):
        title = item.get('title', '')
        snippet = item.get('snippet', '')
        
        # Skip if this looks like product JSON data
        if (title.startswith('[{') or 'bhb_tv_' in title or 'model_code' in snippet or 
            'screen_size_inch' in snippet or 'dimensions_mm' in snippet):
            continue
            
        corpus.append((f"{title}\n{snippet}", idx))
        valid_indices.append(idx)
    
    if not corpus:
        return []
        
    matches = process.extract(query, [c[0] for c in corpus], scorer=fuzz.token_set_ratio, limit=limit)
    results: List[Dict[str, Any]] = []
    for _, score, corpus_idx in matches:
        if 0 <= corpus_idx < len(valid_indices):
            actual_idx = valid_indices[corpus_idx]
            if 0 <= actual_idx < len(kb_index):
                item = kb_index[actual_idx]
                item_copy = {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "path": item.get("path"),
                    "score": score,
                }
                results.append(item_copy)
    return results

from fastapi import UploadFile, File

@app.post("/api/admin/login")
async def admin_login(req: Dict[str, Any]):
    code = (req or {}).get("code")
    if not code or str(code) != ADMIN_PASSCODE:
        return JSONResponse({"error": "Invalid passcode"}, status_code=403)
    # Return token in body in addition to setting cookie to support webviews
    resp = JSONResponse({"ok": True, "token": ADMIN_TOKEN})
    # Set a simple cookie so subsequent admin requests are allowed
    resp.set_cookie(key="admin_auth", value=ADMIN_TOKEN, httponly=True, samesite="lax")
    return resp

@app.get("/api/kb/list")
async def kb_list(request: Request):
    # Accept either cookie or Authorization: Bearer <token>
    token_cookie = request.cookies.get("admin_auth")
    auth_header = request.headers.get("Authorization") or ""
    bearer_token = auth_header.split(" ")[-1] if auth_header.lower().startswith("bearer ") else None
    token = token_cookie or bearer_token
    if token != ADMIN_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    # Return the entire index for now (admin-only)
    return {"ok": True, "items": kb_index, "count": len(kb_index)}

@app.post("/api/kb/upload")
async def kb_upload(request: Request, files: List[UploadFile] = File(...)):
    # Require admin cookie or Authorization header
    token_cookie = request.cookies.get("admin_auth")
    auth_header = request.headers.get("Authorization") or ""
    bearer_token = auth_header.split(" ")[-1] if auth_header.lower().startswith("bearer ") else None
    token = token_cookie or bearer_token
    if token != ADMIN_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    ensure_dir(KB_DIR)
    added: List[Dict[str, Any]] = []
    for uf in files:
        try:
            safe_name = Path(uf.filename).name
            # Avoid collisions by timestamp prefix
            ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            out_path = KB_DIR / f"{ts}_{safe_name}"
            content = await uf.read()
            out_path.write_bytes(content)
            text = extract_text_from_file(out_path)
            entry = {
                "id": ts,
                "title": safe_name,
                "path": str(out_path.relative_to(DATA_DIR)),
                "length": len(text or ""),
                "snippet": (text or "")[:800],
                "created_at": ts,
            }
            kb_index.append(entry)
            added.append({"title": entry["title"], "length": entry["length"]})
        except Exception as e:
            added.append({"title": uf.filename, "error": str(e)})
    save_kb_index(kb_index)
    return {"ok": True, "added": added, "count": len(added)}

class FeedbackItem(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    reply: Optional[str] = None
    correct: Optional[bool] = None
    notes: Optional[str] = None
    category: Optional[str] = None

@app.post("/api/feedback")
async def submit_feedback(item: FeedbackItem):
    try:
        ensure_dir(DATA_DIR)
        path = DATA_DIR / "feedback.jsonl"
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "session_id": item.session_id,
            "message": item.message,
            "reply": item.reply,
            "correct": item.correct,
            "notes": item.notes,
            "category": item.category,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def build_kb_embeddings(max_items: int = 300) -> Dict[str, Any]:
    try:
        if not kb_index:
            return {"items": []}
        entries = kb_index[:max_items]
        texts = [e.get("snippet", "") for e in entries]
        vecs = embed_texts(texts) or []
        items: List[Dict[str, Any]] = []
        for i, e in enumerate(entries):
            emb = vecs[i] if i < len(vecs) else []
            items.append({
                "title": e.get("title"),
                "path": e.get("path"),
                "embedding": emb,
            })
        out = {"items": items, "created_at": datetime.utcnow().isoformat()}
        (DATA_DIR / "kb_embeddings.json").write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception:
        return {"items": []}

@app.post("/api/admin/retrain-kb")
async def retrain_kb(request: Request):
    token_cookie = request.cookies.get("admin_auth")
    auth_header = request.headers.get("Authorization") or ""
    bearer_token = auth_header.split(" ")[-1] if auth_header.lower().startswith("bearer ") else None
    token = token_cookie or bearer_token
    if token != ADMIN_TOKEN:
        return JSONResponse({"error": "Unauthorized"}, status_code=403)
    out = build_kb_embeddings()
    return {"ok": True, "count": len(out.get("items") or [])}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = ensure_session(req.session_id, req.domain)
    clean_message = redact_pii_text(req.message)
    flagged = await _moderate_text_server(clean_message)
    if flagged:
        session["history"].append({"role": "user", "content": clean_message})
        reply = "I can’t assist with that request. Please ask about appliances, stores, warranty, delivery, installation or returns."
        session["history"].append({"role": "assistant", "content": reply})
        return {"reply": reply, "history_len": len(session["history"]) }
    session["history"].append({"role": "user", "content": clean_message})
    if req.domain == "customer_support":
        reply = handle_customer_support(clean_message)
    elif req.domain == "staff_training":
        reply = handle_staff_training(clean_message)
    elif req.domain == "shopping_assistant":
        reply, items = handle_shopping_assistant_payload(clean_message)
    elif req.domain == "smart_support":
        reply, items = handle_smart_support(clean_message)
    else:
        # Consolidate into unified assistant: default to smart_support behavior
        reply, items = handle_smart_support(clean_message)

    # AI fallback to refine replies into more natural, ChatGPT-like responses when available
    try:
        # Optional AI rewrite; disabled by default to avoid delays
        if ENABLE_AI_REWRITE and OPENAI_API_KEY and not is_store_query(clean_message):
            # Local helpers to avoid policy noise for shopping-style queries
            def _is_shopping_intent(text: str) -> bool:
                t = (text or "").lower()
                cues = [
                    "recommend", "suggest", "looking for", "buy", "under", "around",
                    "budget", "rm", "price", "deal", "best", "appliance", "microwave",
                    "fridge", "refrigerator", "washer", "washing machine", "vacuum",
                    "aircond", "air conditioner", "tv", "kettle", "coffee", "dryer",
                ]
                return any(c in t for c in cues)
            def _is_policyish(title: Optional[str], path: Optional[str]) -> bool:
                tl = (title or "").lower()
                pl = (path or "").lower()
                return (
                    "policy" in tl or "warranty" in tl or "return" in tl or
                    "policy" in pl or "warranty" in pl or "return" in pl
                )
            context_lines: List[str] = []
            if req.domain in ("customer_support", "smart_support"):
                questions = [f.get("question", "") for f in faqs]
                matches = process.extract(req.message, questions, scorer=fuzz.token_set_ratio, limit=3) if questions else []
                for name, score, idx in matches:
                    if isinstance(idx, int) and 0 <= idx < len(faqs):
                        q = faqs[idx].get("question", "")
                        a = faqs[idx].get("answer", "")
                        context_lines.append(f"Q: {q}\nA: {a}")
                # Include KB entries
                seen: set = set()
                for kb in search_kb(req.message, limit=(2 if req.concise else 3)):
                    raw_title = kb.get('title') or ''
                    path = kb.get('path')
                    # Suppress policy/warranty snippets for shopping-style queries
                    if _is_shopping_intent(req.message) and _is_policyish(raw_title, path):
                        continue
                    title = raw_title.strip().lower()
                    if title and title in seen:
                        continue
                    seen.add(title)
                    context_lines.append(f"KB: {kb.get('title')}: {kb.get('snippet')}")
                # For smart_support, also include product context
                if req.domain == "smart_support":
                    for p in top_products_for_message(req.message, limit=5):
                        price_val = p.get("price")
                        currency = p.get("currency") or "RM"
                        price_text = f"{currency} {price_val}" if price_val not in (None, "") else "N/A"
                        brand_text = f" ({p.get('brand')})" if p.get("brand") else ""
                        context_lines.append(f"- {p.get('name')}{brand_text} — {price_text}")
            elif req.domain == "staff_training":
                titles = [(m.get("title") or m.get("name") or m.get("topic") or "") for m in training_modules]
                matches = process.extract(req.message, titles, scorer=fuzz.token_set_ratio, limit=3) if titles else []
                for title, score, idx in matches:
                    if isinstance(idx, int) and 0 <= idx < len(training_modules):
                        mod = training_modules[idx]
                        t = mod.get("title") or mod.get("name") or mod.get("topic") or ""
                        summary = mod.get("summary") or mod.get("description") or ""
                        context_lines.append(f"{t}: {summary}")
                # Include KB entries
                seen: set = set()
                for kb in search_kb(req.message, limit=(2 if req.concise else 3)):
                    title = (kb.get('title') or '').strip().lower()
                    if title and title in seen:
                        continue
                    seen.add(title)
                    context_lines.append(f"KB: {kb.get('title')}: {kb.get('snippet')}")
            elif req.domain == "shopping_assistant":
                for p in top_products_for_message(req.message, limit=5):
                    price_val = p.get("price")
                    currency = p.get("currency") or "RM"
                    price_text = f"{currency} {price_val}" if price_val not in (None, "") else "N/A"
                    brand_text = f" ({p.get('brand')})" if p.get("brand") else ""
                    context_lines.append(f"- {p.get('name')}{brand_text} — {price_text}")
                # Include KB entries
                seen: set = set()
                for kb in search_kb(req.message, limit=(1 if req.concise else 2)):
                    title = (kb.get('title') or '').strip().lower()
                    if title and title in seen:
                        continue
                    seen.add(title)
                    context_lines.append(f"KB: {kb.get('title')}: {kb.get('snippet')}")

            prompt = (
                "You are a helpful assistant for an electrical retail store.\n"
                "User message:\n" + (clean_message or "") + "\n\n"
                "Draft reply:\n" + (reply or "") + "\n\n"
                "Context:\n" + ("\n".join(context_lines) if context_lines else "(none)") + "\n\n"
                "Rewrite and improve the draft reply in a friendly, concise tone (aim for 3-6 lines), grounded in the provided context. "
                "If recommending products, list up to 5 items with name, brand (if known), and price. "
                "Do not invent facts beyond context."
            )
            ai_reply = await use_openai(prompt)
            if ai_reply:
                reply = ai_reply
    except Exception:
        pass
    session["history"].append({"role": "assistant", "content": reply})
    resp: Dict[str, Any] = {"reply": reply, "history_len": len(session["history"]) }
    if req.domain in ("shopping_assistant", "smart_support"):
        # include items payload if available
        try:
            resp["items"] = items
        except Exception:
            pass
    # No additional prepend here to avoid duplicate explanation; handled upstream in handle_smart_support
    try:
        items_count = 0
        try:
            items_count = len(resp.get("items") or [])
        except Exception:
            items_count = 0
        escalation_markers = [
            "check bhb.com.my",
            "nearest BHB branch",
            "can’t assist",
            "cannot assist",
        ]
        escalated = any(m in (resp.get("reply") or "").lower() for m in [s.lower() for s in escalation_markers])
        resolved = bool(items_count > 0) or (not flagged and bool(resp.get("reply")))
        log_analytics_event({
            "ts": datetime.utcnow().isoformat(),
            "session_id": req.session_id,
            "domain": req.domain,
            "items_count": items_count,
            "reply_len": len((resp.get("reply") or "")),
            "flagged": bool(flagged),
            "resolved": bool(resolved),
            "escalated": bool(escalated),
        })
    except Exception:
        pass
    return resp

@app.post("/api/rag-chat")
async def rag_chat(req: RagRequest):
    """RAG chat endpoint using KB embeddings and OpenAI completion for grounded answers."""
    session = ensure_session(req.session_id, "rag")
    session["history"].append({"role": "user", "content": req.message})
    # Try embedding-based retrieval first; if unavailable, fall back to fuzzy KB search.
    topk = max(1, int((2 if req.concise else (req.top_k or 5))))
    sources: List[Dict[str, Any]] = []
    context_lines: List[str] = []

    # Simple shopping intent detection to avoid surfacing store policies in recommendations
    def _is_shopping_intent(text: str) -> bool:
        """Detect when the user is asking for product recommendations.
        Broaden signals to reliably catch queries like 'appliances for a small apartment'.
        """
        t = (text or "").lower()
        cues = [
            # general recommendation cues
            "recommend", "suggest", "looking for", "buy", "under", "around",
            "budget", "rm", "price", "deal", "best",
            # product family cues (singular + plural + synonyms)
            "appliance", "appliances", "microwave", "oven", "fridge", "refrigerator", "freezer",
            "washer", "washing machine", "laundry", "dryer", "tumble",
            "vacuum", "cordless", "stick", "coffee", "espresso", "machine",
            "kettle", "aircond", "air conditioner", "ac", "tv", "television",
            # space/sizing context often used for shopping
            "small apartment", "apartment", "condo", "studio", "compact", "space",
        ]
        if any(c in t for c in cues):
            return True
        # Heuristic: if the message mentions any top product categories by name
        cat_keywords = [
            "air conditioner", "aircond", "fridge", "refrigerator", "washer", "washing",
            "dryer", "vacuum", "microwave", "tv", "television", "kettle", "coffee",
        ]
        return any(k in t for k in cat_keywords)

    def _is_policyish(title: Optional[str], path: Optional[str]) -> bool:
        tl = (title or "").lower()
        pl = (path or "").lower()
        return (
            "policy" in tl or "warranty" in tl or "return" in tl or
            "policy" in pl or "warranty" in pl or "return" in pl
        )

    def _parse_shopping_constraints(text: str) -> Dict[str, Any]:
        """Lightweight constraints extractor used for planning.
        Returns keys like category, budget_value, budget_currency, space_small.
        """
        t = (text or "").lower()
        info: Dict[str, Any] = {}
        # category cues
        cats = {
            "TV": ["tv", "television", "oled", "uled", "screen"],
            "Washer": ["washer", "washing", "laundry", "washing machine"],
            "Refrigerator": ["fridge", "refrigerator", "freezer"],
            "Air Conditioner": ["aircond", "air conditioner", "ac"],
            "Microwave": ["microwave", "oven"],
            "Vacuum": ["vacuum", "cordless", "stick"],
            "Dryer": ["dryer", "tumble"],
            "Kettle": ["kettle"],
            "Coffee Maker": ["coffee", "espresso"],
        }
        for cat, keys in cats.items():
            if any(k in t for k in keys):
                info["category"] = cat
                break
        # budget: simple RM/number or "under"/"around" patterns
        import re
        m = re.search(r"(?:rm\s*)?(\d{2,5})", t)
        if m:
            try:
                info["budget_value"] = int(m.group(1))
                info["budget_currency"] = "RM"
            except Exception:
                pass
        if "under" in t or "below" in t:
            info["budget_hint"] = "upper_bound"
        if any(k in t for k in ["small apartment", "studio", "compact", "condo", "small space", "space"]):
            info["space_small"] = True
        return info

    items = ensure_kb_embed_cache(limit=200)
    hits: List[Dict[str, Any]] = []
    if items:
        # Embed query and score KB chunks
        qvecs = embed_texts([req.message]) or []
        if qvecs:
            q = qvecs[0]
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for it in items:
                vec = it.get("embedding") or []
                sim = cosine_sim(vec, q)
                scored.append((sim, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            hits = [it for _, it in scored[:topk]]
    # If embedding path yields no hits, fall back to fuzzy KB search
    seen_titles: set = set()
    if not hits:
        try:
            fuzzy = search_kb(req.message, limit=topk)
        except Exception:
            fuzzy = []
        for kb in fuzzy:
            title = kb.get("title") or "KB"
            path = kb.get("path")
            # Suppress policy/warranty snippets for shopping-style queries
            if _is_shopping_intent(req.message) and _is_policyish(title, path):
                continue
            norm = (title or "").strip().lower()
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            snippet = kb.get("snippet") or ""
            context_lines.append(f"{title}: {snippet}")
            sources.append({"title": title, "path": path})
    else:
        for h in hits:
            title = h.get("title") or "KB"
            path = h.get("path")
            if _is_shopping_intent(req.message) and _is_policyish(title, path):
                continue
            norm = (title or "").strip().lower()
            if norm in seen_titles:
                continue
            seen_titles.add(norm)
            txt = h.get("text") or ""
            context_lines.append(f"{title}: {txt}")
            sources.append({"title": title, "path": path})

    # Do not include product lines in context_lines to avoid duplication in final reply.

    # Compose grounded prompt; prefer OpenAI if configured, otherwise template a concise reply from context
    reply: str
    # Prepare product suggestions separately so they’re not cut by KB topk
    prod_lines: List[str] = []
    try:
        for p in top_products_for_message(req.message, limit=3):
            price_val = p.get("price")
            currency = p.get("currency") or "RM"
            price_text = f"{currency} {price_val}" if price_val not in (None, "") else "N/A"
            brand_text = f" ({p.get('brand')})" if p.get("brand") else ""
            s = f"- {p.get('name')}{brand_text} — {price_text}"
            feats = " ".join(p.get("features", [])).lower()
            if any(k in feats for k in ["inverter", "energy", "efficient", "star"]):
                s += " (energy-efficient)"
            prod_lines.append(s)
    except Exception:
        pass
    shopping_intent = _is_shopping_intent(req.message)
    # For shopping queries, suppress KB context and sources to keep replies focused
    if shopping_intent:
        context_lines = []
        sources = []
    if OPENAI_API_KEY:
        if shopping_intent:
            prompt = (
                "You are a shopping assistant for an electrical retail store.\n"
                "Think privately to identify constraints (category, budget, space). Do not reveal your reasoning.\n"
                "If key details are missing, ask one short clarifying question first, then give concise recommendations.\n\n"
                "User message:\n" + (req.message or "") + "\n\n"
                "Products (for consideration):\n" + ("\n".join(prod_lines) if prod_lines else "(none)") + "\n\n"
                "Output: 1) If needed, ONE clarifying question on a single line. 2) 3–5 product recommendations, each with name, brand (if known), and price. No citations."
            )
        else:
            prompt = (
                "You are a helpful assistant for an electrical retail store.\n"
                "Answer grounded strictly in the provided context. If unsure, say so.\n\n"
                "User message:\n" + (req.message or "") + "\n\n"
                "Context:\n" + ("\n".join(context_lines) if context_lines else "(none)") + "\n\n"
                "Respond concisely (aim for 3-6 lines) and include product suggestions only if relevant."
            )
        ai_reply = await use_openai(prompt)
        if ai_reply:
            reply = ai_reply
            # Ensure a clarifying question appears for shopping intent when budget is missing
            if shopping_intent:
                constraints = _parse_shopping_constraints(req.message)
                needs_clarify = not constraints.get("budget_value")
                if needs_clarify:
                    first_line = (reply or "").strip().splitlines()[0] if reply else ""
                    has_question = ("?" in first_line)
                    if not has_question:
                        reply = (
                            "Quick question: Do you have a budget in mind (e.g., under RM 500)?\n\n"
                            + reply
                        )
        else:
            # Template fallback from context (OpenAI failed or returned nothing)
            sections: List[str] = []
            if context_lines:
                sections.append("Here’s what I found:\n" + "\n".join(context_lines[:topk]))
            if prod_lines:
                sections.append("Recommended options:\n" + "\n".join(prod_lines))
            # If shopping intent without budget, include clarifier even in this fallback
            try:
                if shopping_intent:
                    constraints = _parse_shopping_constraints(req.message)
                    needs_clarify = not constraints.get("budget_value")
                    if needs_clarify:
                        sections.insert(0, "Quick question: Do you have a budget in mind (e.g., under RM 500)?")
            except Exception:
                pass
            reply = "\n\n".join(sections) if sections else "Sorry, I couldn’t find relevant information in the knowledge base."
    else:
        # No OpenAI: return templated context-based reply
        sections: List[str] = []
        constraints = _parse_shopping_constraints(req.message)
        if shopping_intent:
            # Clarify first if we lack budget information
            needs_clarify = not constraints.get("budget_value")
            if needs_clarify:
                sections.append("Quick question: Do you have a budget in mind (e.g., under RM 500)?")
            if prod_lines:
                sections.append("Recommended options:\n" + "\n".join(prod_lines))
            reply = "\n\n".join(sections)
        else:
            if context_lines:
                sections.append("Here’s what I found:\n" + "\n".join(context_lines[:topk]))
            if prod_lines:
                sections.append("Recommended options:\n" + "\n".join(prod_lines))
            reply = "\n\n".join(sections) if sections else "RAG requires OPENAI_API_KEY configured and KB content available."

    # Include items payload for UI product cards
    items_payload: List[Dict[str, Any]] = []
    try:
        items_payload = top_products_for_message(req.message, limit=5)
    except Exception:
        items_payload = []

    session["history"].append({"role": "assistant", "content": reply})
    return {
        "reply": reply,
        "sources": sources,
        "items": items_payload,
        "history_len": len(session["history"]) 
    }

# Vision endpoint
@app.post("/api/vision-search")
async def vision_search(file: UploadFile = File(...)):
    data = await file.read()
    mime = file.content_type or "image/jpeg"
    # Pass filename to help correct classification via hints
    fname = getattr(file, "filename", None)
    details = await extract_product_details_from_image(data, mime, filename=fname)
    if not details:
        # Provide richer diagnostics to help troubleshoot
        try:
            import importlib.util as _ilu
            _gemini_spec = _ilu.find_spec("google.generativeai")
            gemini_loaded = _gemini_spec is not None
        except Exception:
            gemini_loaded = False
        diag = {
            "provider": VISION_PROVIDER,
            "mime": mime,
            "size_bytes": len(data) if data else 0,
            "gemini_sdk_loaded": gemini_loaded,
            "gemini_api_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
        }
        reason = "Vision feature not configured or unavailable."
        if VISION_PROVIDER == "gemini":
            if genai is None:
                reason = "Gemini SDK not installed (google-generativeai)."
            elif not os.getenv("GEMINI_API_KEY"):
                reason = "GEMINI_API_KEY missing in environment."
            else:
                reason = "Gemini failed to analyze the image."
        elif VISION_PROVIDER == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                reason = "OPENAI_API_KEY missing in environment."
            else:
                reason = "OpenAI failed to analyze the image."
        return JSONResponse({"error": reason, "diagnostics": diag}, status_code=400)
    results = search_from_details(details)
    return {"details": details, "results": results}

@app.get("/api/diagnostics/wc-store")
def wc_store_diag():
    # Checks Store API reachability using configured WC_API_URL
    base = WC_API_URL or "https://www.bhb.com.my"
    if not requests:
        return {"ok": False, "error": "requests unavailable"}
    try:
        url = f"{base.rstrip('/')}/wp-json/wc/store/products?per_page=1"
        r = requests.get(url, timeout=10)
        return {"ok": r.status_code == 200, "status": r.status_code, "body_type": type(r.json()).__name__}
    except Exception as e:
        return {"ok": False, "error": str(e)}
