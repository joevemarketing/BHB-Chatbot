from typing import Optional, List, Dict, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..models import Product, ChatResponse
from ..product_loader import get_all_products
from ..recommender import build_constraints_from_text, get_best_products
from ..llm_client import generate_reply
from ..utils import get_bhb_link_or_search
from ..vision_client import analyse_product_image, match_image_to_catalog, canonical_category


router = APIRouter()


@router.post("/vision-chat", response_model=ChatResponse)
async def vision_chat(image: UploadFile = File(...), message: Optional[str] = Form("")):
    """Photo-based advisor: analyse image, try match catalog, and respond.
    Validates image type, calls vision model, tries catalog match, and composes reply.
    """
    # Validate content type
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type: please upload an image.")

    # Read bytes and basic diagnostic
    img_bytes = await image.read()
    print(f"[vision] uploaded image bytes: {len(img_bytes)}")

    # Vision analysis
    vision_result = await analyse_product_image(img_bytes)
    print(f"[vision] vision_result: {vision_result}")

    # Load products and attempt match
    all_products: List[Product] = get_all_products()
    matched_product: Optional[Product] = match_image_to_catalog(vision_result, all_products)
    print(f"[vision] matched_product: {matched_product.model_name if matched_product else None}")

    # Build constraints based on vision
    constraints = build_constraints_from_text(message or "")
    category_hint = canonical_category(vision_result.get("category"))
    if category_hint in {"tv", "fridge", "aircond", "washer"}:
        constraints.category = category_hint
    brand_hint = (vision_result.get("brand") or "").strip()
    if brand_hint:
        constraints.brand_preferences = [brand_hint]

    # Decide active category
    # Prefer explicit constraint; else vision-derived canonical; else matched product category
    active_category = constraints.category or (category_hint if category_hint in {"tv", "fridge", "aircond", "washer"} else None)
    if active_category is None and matched_product is not None:
        active_category = matched_product.category

    # Select products (fallback to category recommendations if no match)
    selected_products: List[Product] = []
    candidates: List[Product] = []
    if matched_product is not None:
        selected_products = [matched_product]
    else:
        # Force category-based fallback if category is recognized
        if active_category is not None:
            products_for_scoring = [p for p in all_products if p.category == active_category]
            candidates = get_best_products(products_for_scoring, constraints, top_n=5)
            selected_products = candidates
        else:
            # Rare case: no category — keep original behaviour
            candidates = get_best_products(all_products, constraints, top_n=5)
            selected_products = candidates
    # Log selected products for quick verification during development
    print(f"[vision] selected_products count: {len(selected_products)}; categories: {[p.category for p in selected_products]}")

    # Compose advisor message
    advisor_message = (
        "The customer uploaded a product photo. Help explain what it is and advise. "
        f"Vision hints: brand={vision_result.get('brand')}, model_guess={vision_result.get('model_guess')}. "
        f"User message: {message or ''}"
    )

    # Call LLM. Only pass selected products; if empty, the LLM should explain honestly.
    # Strip URL/search fields before sending products to LLM
    llm_products = []
    for p in selected_products:
        d = p.dict()
        d.pop("bhb_product_url", None)
        d.pop("website_search_text", None)
        llm_products.append(d)

    reply_text = await generate_reply(
        user_message=advisor_message,
        constraints=constraints.dict(),
        products=llm_products,
        extra_context={"vision_result": vision_result},
        conversation=None,
    )

    # Failsafe filter and fallback before returning
    suggested_products: List[Product] = selected_products
    if active_category is not None:
        suggested_products = [p for p in suggested_products if p.category == active_category]
        if not suggested_products and candidates:
            suggested_products = candidates

    # Attach BHB link info (direct or search) for UI convenience
    for p in suggested_products:
        p.bhb_link = get_bhb_link_or_search(p)

    return ChatResponse(reply=reply_text, suggested_products=suggested_products, constraints=constraints)