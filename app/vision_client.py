import base64
import json
import os
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def _extract_json(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON from model response, tolerant of code fences.
    Returns dict with keys: category, brand, model_guess, capacity_guess, confidence, notes.
    """
    if not text:
        return {
            "category": None,
            "brand": None,
            "model_guess": None,
            "capacity_guess": None,
            "confidence": 0,
            "notes": "No response text",
        }
    cleaned = text.strip()
    # Remove triple backticks if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("` ")
        # Sometimes models include a language hint like ```json
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:]
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {
                "category": data.get("category"),
                "brand": data.get("brand"),
                "model_guess": data.get("model_guess"),
                "capacity_guess": data.get("capacity_guess"),
                "confidence": data.get("confidence", 0) or 0,
                "notes": data.get("notes"),
            }
        return {
            "category": None,
            "brand": None,
            "model_guess": None,
            "capacity_guess": None,
            "confidence": 0,
            "notes": cleaned,
        }
    except Exception:
        # Fallback: attempt to heuristically extract brand/model_guess keys
        result: Dict[str, Any] = {
            "category": None,
            "brand": None,
            "model_guess": None,
            "capacity_guess": None,
            "confidence": 0,
            "notes": cleaned,
        }
        # crude extraction
        for key in ("brand", "model", "model_guess", "sku", "capacity"):
            idx = cleaned.lower().find(key.lower())
            if idx != -1:
                # take up to 40 chars following ':' or '='
                segment = cleaned[idx: idx + 80]
                sep_idx = segment.find(":")
                if sep_idx == -1:
                    sep_idx = segment.find("=")
                if sep_idx != -1:
                    value = segment[sep_idx + 1 :].strip().split("\n")[0]
                    if key in ("model", "model_guess"):
                        result["model_guess"] = value
                    elif key == "sku":
                        result["model_guess"] = value
                    elif key == "capacity":
                        result["capacity_guess"] = value
                    else:
                        result["brand"] = value
        return result


async def analyse_product_image(image_bytes: bytes) -> Dict[str, Optional[str]]:
    """Send the image to a vision-capable OpenAI-compatible endpoint and
    return a dict with keys: brand, model_guess, notes.
    If API key or client unavailable, return a best-effort notes-only result.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
    base_url = os.getenv("OPENAI_BASE_URL")

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    if not api_key or OpenAI is None:
        return {
            "brand": None,
            "model_guess": None,
            "notes": "Vision API not configured. Set OPENAI_API_KEY to enable.",
        }

    try:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        system_prompt = (
            "You are helping identify electrical products for a retailer. "
            "From this photo, decide which ONE of these categories best matches the main product: 'washer', 'fridge', 'tv', 'aircond'. "
            "Then read any visible BRAND and MODEL number or code. If you can see capacity (e.g. 8kg, 400L), include it. "
            "Respond ONLY with strict JSON: {"
            "\"category\": \"...\", \"brand\": \"...\", \"model_guess\": \"...\", \"capacity_guess\": \"...\", \"confidence\": 0-1, \"notes\": \"...\" }"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify product category, brand, model, capacity. Reply with ONLY the JSON."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            },
        ]
        resp = client.chat.completions.create(model=model_name, messages=messages, temperature=0)
        text = resp.choices[0].message.content if resp and resp.choices else ""
        parsed = _extract_json(text)
        # Ensure all keys exist
        return {
            "category": parsed.get("category"),
            "brand": parsed.get("brand"),
            "model_guess": parsed.get("model_guess"),
            "capacity_guess": parsed.get("capacity_guess"),
            "confidence": parsed.get("confidence", 0) or 0,
            "notes": parsed.get("notes"),
        }
    except Exception as e:
        return {
            "category": None,
            "brand": None,
            "model_guess": None,
            "capacity_guess": None,
            "confidence": 0,
            "notes": f"Vision API error: {e}",
        }


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def canonical_category(s: Optional[str]) -> Optional[str]:
    """Map various vision category strings to canonical values used in our catalog.
    Accepts common synonyms and capitalization variants.
    Returns one of: 'washer', 'fridge', 'tv', 'aircond', or None if unknown.
    """
    t = _norm(s)
    if not t:
        return None
    # direct matches
    if t in {"washer", "fridge", "tv", "aircond"}:
        return t
    # synonyms and variants
    if t in {"washing machine", "wash machine", "mesin basuh"}:
        return "washer"
    if t in {"refrigerator", "fridge/freezer", "peti ais"}:
        return "fridge"
    if t in {"air conditioner", "air-con", "air con", "airconditioner", "air conditioning"}:
        return "aircond"
    if t in {"television", "smart tv", "led tv", "oled tv"}:
        return "tv"
    return None


def match_image_to_catalog(vision_result: Dict[str, Optional[str]], products: List["Product"]):
    """Score products based on vision hints and return best match or None."""
    brand = _norm(vision_result.get("brand"))
    model_guess = _norm(vision_result.get("model_guess"))
    capacity_guess = _norm(vision_result.get("capacity_guess"))
    category_guess = canonical_category(vision_result.get("category")) or _norm(vision_result.get("category"))
    confidence = float(vision_result.get("confidence") or 0)

    if not products:
        return None

    # Filter by category first if provided
    filtered = products
    if canonical_category(category_guess) in {"tv", "fridge", "aircond", "washer"}:
        cat = canonical_category(category_guess)
        filtered = [p for p in products if getattr(p, "category", None) == cat]

    # parse capacity value
    cap_val: Optional[float] = None
    cap_unit: Optional[str] = None
    if capacity_guess:
        if "kg" in capacity_guess:
            try:
                cap_val = float(capacity_guess.replace("kg", "").strip())
                cap_unit = "kg"
            except Exception:
                pass
        elif "l" in capacity_guess or "liter" in capacity_guess:
            try:
                cap_val = float(capacity_guess.lower().replace("liter", "").replace("l", "").strip())
                cap_unit = "l"
            except Exception:
                pass

    best: Any = None
    best_score: float = -1

    for p in filtered:
        score = 0.0
        pb = _norm(getattr(p, "brand", None))
        pm = _norm(getattr(p, "model_name", None))
        psku = _norm(getattr(p, "sku", None))

        if brand and pb == brand:
            score += 5
        if model_guess and (model_guess in pm or model_guess in psku):
            score += 10

        # capacity rough match
        if cap_val is not None:
            if cap_unit == "kg" and getattr(p, "capacity_kg", None):
                if abs(float(p.capacity_kg) - cap_val) <= 1.0:
                    score += 3
            if cap_unit == "l" and getattr(p, "capacity_liters", None):
                if abs(float(p.capacity_liters) - cap_val) <= 50.0:
                    score += 3

        if confidence > 0.5:
            score += 2

        if score > best_score:
            best = p
            best_score = score

    # threshold for confident match
    if best_score >= 5:
        return best
    return None