import re
from typing import List, Tuple

from .models import Product, UserConstraints, ChatMessage

# Sentinel score to discard items (never recommend)
DISCARD_SCORE = -1e9


def infer_category_from_text(text: str):
    """Return a category string if positively detected from the text, else None.
    Recognizes English and Malay synonyms without guessing when absent.
    """
    t = (text or "").lower()
    if any(k in t for k in ["washer", "washing machine", "mesin basuh"]):
        return "washer"
    if any(k in t for k in ["fridge", "refrigerator", "peti ais", "peti sejuk"]):
        return "fridge"
    if any(k in t for k in ["aircond", "aircon", "air conditioner", "air con"]):
        return "aircond"
    if any(k in t for k in ["tv", "television"]):
        return "tv"
    # New categories
    if any(k in t for k in ["fan", "ceiling fan", "kipas"]):
        return "fan"
    if any(k in t for k in ["water heater", "shower heater", "heater mandi"]):
        return "water_heater"
    if any(k in t for k in ["rice cooker", "periuk nasi"]):
        return "rice_cooker"
    if "air fryer" in t:
        return "air_fryer"
    if any(k in t for k in ["vacuum", "vacuum cleaner"]):
        return "vacuum"
    return None


def build_constraints_from_text(text: str) -> UserConstraints:
    t = (text or "").lower()
    constraints = UserConstraints()

    # Category detection via explicit keywords only; leave None otherwise.
    constraints.category = infer_category_from_text(t)

    # Budget detection (RM values and ranges)
    # Range like "RM 1000-1500" or "1000 - 1500"
    m_range = re.search(r"(?:rm\s*)?(\d{2,6})\s*[-to]+\s*(?:rm\s*)?(\d{2,6})", t)
    if m_range:
        lo, hi = float(m_range.group(1)), float(m_range.group(2))
        constraints.budget_min_rm = min(lo, hi)
        constraints.budget_max_rm = max(lo, hi)
    else:
        # Single amount like "RM 1200" or "around 1500"
        m_one = re.search(r"(?:rm\s*)?(\d{2,6})", t)
        if m_one:
            val = float(m_one.group(1))
            # If user says under/below, treat as max
            if any(k in t for k in ["under", "below", "kurang", "<"]):
                constraints.budget_max_rm = val
            elif any(k in t for k in ["around", "about", "~", "lebih kurang"]):
                constraints.budget_min_rm = max(val * 0.8, 0)
                constraints.budget_max_rm = val * 1.2
            else:
                constraints.budget_max_rm = val

    # Household size detection
    m_family = re.search(r"family of (\d+)", t)
    if m_family:
        constraints.household_size = int(m_family.group(1))
    m_of_us = re.search(r"(\d+) of us", t)
    if m_of_us:
        constraints.household_size = int(m_of_us.group(1))

    # Home type
    for ht in ["condo", "kondo", "apartment", "studio", "terrace", "teres", "semi-d", "bungalow", "flat"]:
        if ht in t:
            constraints.home_type = ht
            break

    # Room size (sqm)
    m_room = re.search(r"(\d+(?:\.\d+)?)\s*(sqm|m2|m\^2)", t)
    if m_room:
        constraints.room_size_sqm = float(m_room.group(1))

    # Noise sensitivity
    if any(k in t for k in ["light sleeper", "noise sensitive", "quiet please", "senyap", "quiet"]):
        constraints.noise_sensitivity = "high"

    # Brand preferences (simple heuristic)
    m_brand_list = re.search(r"prefer(?:ence)?s?:?\s*([a-z0-9 ,]+)", t)
    brands: List[str] = []
    if m_brand_list:
        brands = [b.strip() for b in m_brand_list.group(1).split(",") if b.strip()]
    else:
        for b in ["samsung", "lg", "panasonic", "sharp", "toshiba", "midea", "bosch", "tefal"]:
            if f"prefer {b}" in t or f"like {b}" in t:
                brands.append(b)
    if brands:
        constraints.brand_preferences = brands

    # Priority
    prios: List[str] = []
    if "energy" in t or "jimat" in t:
        prios.append("energy_saving")
    if "capacity" in t or "large" in t or "besar" in t:
        prios.append("capacity")
    if "quiet" in t or "senyap" in t:
        prios.append("quiet")
    if prios:
        constraints.priority = prios

    return constraints


def build_constraints_from_history(user_messages: List[ChatMessage]) -> UserConstraints:
    """
    Merge constraints across the full user message history.
    Process oldest -> newest so that newer messages override earlier values.
    """
    constraints = UserConstraints()

    for msg in user_messages or []:
        if msg.role != "user":
            continue
        c = build_constraints_from_text(msg.content)
        # Category
        if c.category is not None:
            constraints.category = c.category
        # Budget
        if c.budget_min_rm is not None:
            constraints.budget_min_rm = c.budget_min_rm
        if c.budget_max_rm is not None:
            constraints.budget_max_rm = c.budget_max_rm
        # Household size
        if c.household_size is not None:
            constraints.household_size = c.household_size
        # Home type
        if c.home_type is not None:
            constraints.home_type = c.home_type
        # Room size
        if c.room_size_sqm is not None:
            constraints.room_size_sqm = c.room_size_sqm
        # Noise sensitivity
        if c.noise_sensitivity is not None:
            constraints.noise_sensitivity = c.noise_sensitivity
        # Brand preferences
        if c.brand_preferences is not None and len(c.brand_preferences) > 0:
            constraints.brand_preferences = c.brand_preferences
        # Priority list
        if c.priority is not None and len(c.priority) > 0:
            constraints.priority = c.priority

    return constraints


def _energy_bonus(product: Product, constraints: UserConstraints) -> float:
    if not constraints.priority or "energy_saving" not in constraints.priority:
        return 0.0
    label = (product.energy_label or "").lower()
    if any(s in label for s in ["5", "5-star", "★★★★★", "5 star"]):
        return 2.0
    if any(s in label for s in ["4", "4-star", "★★★★", "4 star"]):
        return 1.0
    return 0.2 if label else 0.0


def _noise_bonus(product: Product, constraints: UserConstraints) -> float:
    if constraints.noise_sensitivity != "high":
        return 0.0
    if product.noise_level_db is None:
        return 0.0
    return 1.0 if product.noise_level_db <= 50 else (-0.5)


def _capacity_bonus(product: Product, constraints: UserConstraints) -> float:
    # Very simple heuristics by category
    hh = constraints.household_size or 0
    if product.category == "fridge" and product.capacity_liters:
        target = max(hh, 1) * 100
        return 1.0 if product.capacity_liters >= target else -0.5
    if product.category == "washer" and product.capacity_kg:
        target = max(hh, 1) * 2.0
        return 1.0 if product.capacity_kg >= target else -0.5
    if product.category == "tv" and product.screen_size_inches and constraints.room_size_sqm:
        # Prefer larger screens for bigger rooms, but lightly
        if constraints.room_size_sqm >= 20 and product.screen_size_inches >= 55:
            return 0.8
        if constraints.room_size_sqm <= 12 and product.screen_size_inches <= 50:
            return 0.5
    if product.category == "aircond" and product.recommended_room_size_sqm and constraints.room_size_sqm:
        # Bonus if recommended size covers the room
        return 1.0 if product.recommended_room_size_sqm >= constraints.room_size_sqm else -0.5
    return 0.0


def score_product(product: Product, constraints: UserConstraints) -> float:
    # Discard if category mismatch
    if constraints.category and product.category != constraints.category:
        return DISCARD_SCORE

    score = 0.0

    # Price alignment
    if constraints.budget_min_rm and product.price_rm < constraints.budget_min_rm:
        score -= 2.0
    if constraints.budget_max_rm and product.price_rm > constraints.budget_max_rm:
        score -= 4.0
    if constraints.budget_min_rm or constraints.budget_max_rm:
        # small bonus for being inside the band
        if (constraints.budget_min_rm or 0) <= product.price_rm <= (constraints.budget_max_rm or float("inf")):
            score += 1.0

    # Brand preferences
    prefs = [p.lower() for p in (constraints.brand_preferences or [])]
    if prefs and product.brand.lower() in prefs:
        score += 2.0

    # Priority bonuses
    score += _energy_bonus(product, constraints)
    score += _noise_bonus(product, constraints)
    score += _capacity_bonus(product, constraints)

    # Tiny bias towards lower price
    score += max(0.0, 5000 - product.price_rm) / 10000.0

    return score


def get_best_products(all_products: List[Product], constraints: UserConstraints, top_n: int = 3) -> List[Product]:
    scored: List[Tuple[float, Product]] = []
    for p in all_products:
        s = score_product(p, constraints)
        if s > DISCARD_SCORE:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]