"""
Quick self-test for the vision flow washer scenario.

This simulates a washer photo where vision_result only provides category="washer"
and no brand/model, so the system should fall back to category-based recommendations.

Run:
  python scripts/self_test_washer.py

Expected:
 - Prints vision_result
 - Prints selected_products with ONLY washer category items (up to 3)
 - Prints a sample reply (fallback text) grounded in selected products
"""

import asyncio
from typing import List, Dict, Any

from app.product_loader import get_all_products
from app.recommender import build_constraints_from_text, get_best_products
from app.llm_client import generate_reply


async def main():
    # Simulated vision result from a washer image with limited details
    vision_result: Dict[str, Any] = {
        "category": "washer",
        "brand": None,
        "model_guess": None,
        "capacity_guess": None,
        "confidence": 0.7,
        "notes": "Simulated washer photo; no visible brand/model."
    }

    print("[self-test] vision_result:", vision_result)

    # Load catalog and build constraints based on vision
    products = get_all_products()
    constraints = build_constraints_from_text("")
    constraints.category = vision_result["category"]

    # Fallback to category-based recommendations (top 3)
    selected_products: List = get_best_products(products, constraints, top_n=3)
    print("[self-test] selected_products (count):", len(selected_products))
    print("[self-test] selected_products categories:", [p.category for p in selected_products])
    assert all(p.category == "washer" for p in selected_products), "Selected products must be only washers"

    # Prepare LLM input and get a reply
    advisor_message = (
        "The customer uploaded a product photo. Help explain what it is and advise. "
        f"Vision hints: brand={vision_result.get('brand')}, model_guess={vision_result.get('model_guess')}"
    )
    reply_text = await generate_reply(
        user_message=advisor_message,
        constraints=constraints.dict(),
        products=[p.dict() for p in selected_products],
        extra_context={"vision_result": vision_result},
    )
    print("\n[self-test] sample reply:\n", reply_text)


if __name__ == "__main__":
    asyncio.run(main())