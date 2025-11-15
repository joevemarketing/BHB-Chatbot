import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .models import Product, ChatRequest, ChatResponse, ChatMessage
from .product_loader import get_all_products
from .recommender import build_constraints_from_text, get_best_products, infer_category_from_text, build_constraints_from_history
from .llm_client import generate_reply
from .store_locations import find_stores_by_query
from .utils import get_bhb_link_or_search
from .routes.vision import router as vision_router
from .service_faq import find_relevant_faq
from .house_planner import suggest_house_appliance_plan, flatten_plan_products


def detect_requested_categories(text: str) -> List[str]:
    t = (text or "").lower()
    cats: List[str] = []
    if any(k in t for k in ["washer", "washing machine", "mesin basuh"]):
        cats.append("washer")
    if any(k in t for k in ["fridge", "refrigerator", "peti sejuk", "peti ais"]):
        cats.append("fridge")
    if any(k in t for k in ["tv", "television"]):
        cats.append("tv")
    if any(k in t for k in ["aircond", "air con", "aircon", "air conditioner", "penghawa dingin"]):
        cats.append("aircond")
    if any(k in t for k in ["fan", "ceiling fan", "kipas"]):
        cats.append("fan")
    if any(k in t for k in ["water heater", "heater mandi", "shower heater"]):
        cats.append("water_heater")
    if any(k in t for k in ["vacuum", "vacuum cleaner"]):
        cats.append("vacuum")
    # dedupe while preserving order
    seen = set()
    out: List[str] = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def is_new_house_context(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in [
        "new house", "new home", "baru pindah", "baru pindah rumah",
        "new condo", "new apartment", "new place"
    ])


app = FastAPI(title="bhb-product-advisor")

from .middleware import RequestLoggingMiddleware, RateLimitMiddleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=int(os.environ.get("RATE_LIMIT_MAX", "60")), window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW", "60")))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web UI (served at /web)
try:
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")
except Exception:
    # web directory may not exist yet; ignore
    pass


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "bhb-product-advisor"}


@app.get("/api/products", response_model=List[Product])
def list_products(category: Optional[str] = None, brand: Optional[str] = None, max_price: Optional[float] = None):
    items = get_all_products()
    out: List[Product] = []
    for p in items:
        if category and p.category != category:
            continue
        if brand and p.brand.lower() != brand.lower():
            continue
        if max_price is not None and p.price_rm > max_price:
            continue
        out.append(p)
    return out


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Determine latest user message and full user history
    latest_user_message: str = ""
    full_user_history: List[ChatMessage] = []
    for m in (req.messages or []):
        if m.role == "user":
            latest_user_message = m.content
            full_user_history.append(m)
    if not latest_user_message and req.messages:
        latest_user_message = req.messages[-1].content

    # Build constraints from FULL user history (newer overrides older)
    constraints = build_constraints_from_history(full_user_history)

    # Simple intent detection using FAQ keywords, product keywords, and store/branch keywords
    faq_items = find_relevant_faq(latest_user_message)
    has_faq = bool(faq_items)
    cats = detect_requested_categories(latest_user_message)
    whole_house_flag = is_new_house_context(latest_user_message)
    has_product_kw = bool(infer_category_from_text(latest_user_message))
    shopping_keywords = [
        "recommend", "cadangan", "suggest", "nak beli", "want", "need", "beli", "buy", "looking for", "cari", "budget", "harga", "price",
    ]
    bundle_keywords = [
        "new house", "whole house", "moving into", "new condo", "buy all", "bundle", "package", "set rumah", "sekali gus", "one shot",
    ]
    has_shopping_signal = any(k in (latest_user_message or "").lower() for k in shopping_keywords)
    has_bundle_signal = any(k in (latest_user_message or "").lower() for k in bundle_keywords)
    store_keywords = [
        "location", "lokasi", "alamat", "branch", "cawangan",
        "near me", "near", "berhampiran", "dekat",
        "kedai", "shop", "store", "outlet",
    ]
    has_store_signal = any(k in (latest_user_message or "").lower() for k in store_keywords)

    # If user mentions an area and we can find matching stores, treat as store intent
    stores_prefetch = []
    if not has_store_signal and not has_product_kw and not has_shopping_signal:
        try:
            stores_prefetch = find_stores_by_query(latest_user_message)
        except Exception:
            stores_prefetch = []

    if has_store_signal or (stores_prefetch and not has_product_kw and not has_shopping_signal):
        intent = "store_location"
    elif has_faq and has_product_kw and has_shopping_signal:
        intent = "mixed"
    elif has_faq and not has_product_kw:
        intent = "service"
    elif has_faq and has_product_kw and not has_shopping_signal:
        # Service-only question mentioning a product category/brand (e.g., warranty for TV LG)
        intent = "service"
    elif has_bundle_signal or whole_house_flag or len(cats) >= 2:
        intent = "bundle"
    elif len(cats) == 1:
        intent = "product"
    else:
        intent = "general_enquiry"

    # Branch backend flow based on intent
    suggested_products: List[Product] = []
    reply_text: str = ""

    variant = assign_prompt_variant(latest_user_message)
    if intent == "store_location":
        # Store/branch location questions: skip recommender, pass store locations to LLM
        stores = stores_prefetch or find_stores_by_query(latest_user_message)
        store_context = {
            "store_locations": [
                {
                    "label": s.label,
                    "address": s.address,
                    "city": s.city,
                    "state": s.state,
                    "tel": s.tel,
                    "hours": s.hours,
                    "brand_shop": s.brand_shop,
                }
                for s in stores
            ]
        }
        # Prepare conversation: prior turns then latest user
        conv: List[dict] = []
        prior: List[ChatMessage] = []
        last_user_idx = -1
        for i, m in enumerate(req.messages or []):
            if m.role == "user":
                last_user_idx = i
        for i, m in enumerate(req.messages or []):
            if m.role in {"user", "assistant"} and i != last_user_idx:
                conv.append({"role": m.role, "content": m.content})
        if last_user_idx != -1:
            conv.append({"role": "user", "content": (req.messages[last_user_idx].content)})
        else:
            conv.append({"role": "user", "content": latest_user_message})

        reply_text = await generate_reply(
            user_message=latest_user_message,
            constraints=constraints.dict(),
            products=[],
            extra_context=store_context,
            conversation=conv,
            prompt_variant=variant,
        )
        suggested_products = []
    elif intent == "service":
        # Service-only: do not call recommender; pass FAQ answers to LLM
        service_context = {"faq_answers": [item.answer for item in faq_items]}
        conv: List[dict] = []
        last_user_idx = -1
        for i, m in enumerate(req.messages or []):
            if m.role == "user":
                last_user_idx = i
        for i, m in enumerate(req.messages or []):
            if m.role in {"user", "assistant"} and i != last_user_idx:
                conv.append({"role": m.role, "content": m.content})
        if last_user_idx != -1:
            conv.append({"role": "user", "content": (req.messages[last_user_idx].content)})
        else:
            conv.append({"role": "user", "content": latest_user_message})
        reply_text = await generate_reply(
            user_message=latest_user_message,
            constraints=constraints.dict(),
            products=[],
            extra_context=service_context,
            conversation=conv,
            prompt_variant=variant,
        )
        suggested_products = []
    else:
        # Product or Mixed: compute product recommendations
        all_products = get_all_products()

        if intent == "bundle":
            # Build whole-house plan and present grouped recommendations
            default_bundle_cats = ["fridge", "washer", "tv", "aircond", "fan", "water_heater"]
            requested = cats if len(cats) >= 2 else (default_bundle_cats if whole_house_flag else default_bundle_cats)
            plan = suggest_house_appliance_plan(all_products, constraints, requested_categories=requested)
            flattened = flatten_plan_products(plan)
            # Prepare LLM products (strip URLs)
            candidate_dicts = []
            for p in flattened:
                d = p.dict()
                d.pop("bhb_product_url", None)
                d.pop("website_search_text", None)
                candidate_dicts.append(d)
            # Full conversation for context
            conv: List[dict] = [
                {"role": m.role, "content": m.content}
                for m in (req.messages or [])
                if m.role in {"user", "assistant"}
            ]
            reply_text = await generate_reply(
                user_message=latest_user_message,
                constraints=constraints.dict(),
                products=candidate_dicts,
                extra_context={"house_plan": True, "scenario": "whole_house", "requested_categories": requested},
                conversation=conv,
                prompt_variant=variant,
            )
            # Attach links
            for p in flattened:
                p.bhb_link = get_bhb_link_or_search(p)
            suggested_products = flattened
        elif intent == "general_enquiry":
            # No clear category: ask for clarification without recommending products
            conv: List[dict] = [
                {"role": m.role, "content": m.content}
                for m in (req.messages or [])
                if m.role in {"user", "assistant"}
            ]
            reply_text = await generate_reply(
                user_message=latest_user_message,
                constraints=constraints.dict(),
                products=[],
                extra_context={"scenario": "general_enquiry"},
                conversation=conv,
                prompt_variant=variant,
            )
            suggested_products = []
        else:
            # Standard product/mixed flow
            products_for_scoring = (
                [p for p in all_products if p.category == constraints.category]
                if constraints.category is not None else all_products
            )

            candidates = get_best_products(products_for_scoring, constraints, top_n=5)
            # Pass product data to LLM without URL fields to prevent link leakage in chat text
            candidate_dicts = []
            for p in candidates:
                d = p.dict()
                d.pop("bhb_product_url", None)
                d.pop("website_search_text", None)
                candidate_dicts.append(d)

            extra_context = None
            if intent == "mixed" and faq_items:
                extra_context = {"faq_answers": [item.answer for item in faq_items]}

            # Send full conversation (all prior messages) to help the LLM maintain context
            conv: List[dict] = [
                {"role": m.role, "content": m.content}
                for m in (req.messages or [])
                if m.role in {"user", "assistant"}
            ]

            reply_text = await generate_reply(
                user_message=latest_user_message,
                constraints=constraints.dict(),
                products=candidate_dicts,
                extra_context=extra_context,
                conversation=conv,
                prompt_variant=variant,
            )

            # Failsafe filter and fallback before returning
            selected_products = candidates
            if constraints.category is not None:
                selected_products = [p for p in selected_products if p.category == constraints.category]
                if not selected_products and candidates:
                    selected_products = candidates

            # Final safety check on budget (optional mild filter)
            lo = constraints.budget_min_rm
            hi = constraints.budget_max_rm
            if lo is not None or hi is not None:
                def _ok_price(price: float) -> bool:
                    return (lo is None or price >= (lo * 0.9)) and (hi is None or price <= (hi * 1.1))
                selected_products = [p for p in selected_products if _ok_price(p.price_rm)]

            # Attach BHB link info (direct or search) for UI convenience
            for p in selected_products:
                p.bhb_link = get_bhb_link_or_search(p)

            suggested_products = selected_products

    try:
        from pathlib import Path
        import json
        def _log_event(ev):
            try:
                Path("data").mkdir(parents=True, exist_ok=True)
                with open("data/analytics_events.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception:
                pass
        resolved = bool(suggested_products)
        escalation_markers = [
            "check bhb.com.my",
            "nearest BHB branch",
            "can’t assist",
            "cannot assist",
        ]
        escalated = any(m in (reply_text or "").lower() for m in [s.lower() for s in escalation_markers])
        _log_event({
            "ts": __import__("datetime").datetime.utcnow().isoformat(),
            "intent": intent,
            "prompt_variant": variant,
            "items_count": len(suggested_products or []),
            "reply_len": len(reply_text or ""),
            "resolved": bool(resolved),
            "escalated": bool(escalated),
        })
    except Exception:
        pass
    return ChatResponse(reply=reply_text, suggested_products=suggested_products, constraints=constraints)


    


@app.get("/", response_class=HTMLResponse)
def index():
    # Redirect to the static web UI
    return RedirectResponse(url="/web/")


@app.get("/web")
def web_noslash_redirect():
    # Convenience redirect so /web (without trailing slash) works
    return RedirectResponse(url="/web/")

# Include routers with /api prefix
app.include_router(vision_router, prefix="/api")
def assign_prompt_variant(seed_text: str) -> str:
    t = (seed_text or "").strip().lower()
    try:
        h = abs(hash(t))
    except Exception:
        h = 0
    return "upsell" if (h % 2 == 0) else "default"
