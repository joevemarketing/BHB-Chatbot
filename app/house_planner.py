from typing import List, Dict, Any, Optional

from .models import Product, UserConstraints


def _pick_n(items: List[Product], n: int) -> List[Product]:
    return sorted(items, key=lambda p: p.price_rm)[:max(n, 0)]


def suggest_house_appliance_plan(products: List[Product], constraints: UserConstraints, requested_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build a simple whole-house package across categories.
    Defaults:
      - Condo/apartment/studio: 1 fridge, 1 washer, 1 TV, 1 aircond, 1–2 fans, 1 water heater, 1 rice cooker, 1 air fryer.
      - Terrace/semi-d/bungalow: 1 fridge, 1 washer, 1 TV, 2 airconds, 3–4 fans, 2 water heaters, 1 rice cooker, 1 air fryer.
    """
    home_type = (constraints.home_type or "").lower()
    is_condo = any(k in home_type for k in ["condo", "apartment", "studio", "flat", "kondo"]) if home_type else False
    is_landed = any(k in home_type for k in ["terrace", "teres", "semi-d", "bungalow"]) if home_type else False

    # Default counts
    if is_condo:
        aircond_n = 1
        fan_n = 2
        heater_n = 1
    elif is_landed:
        aircond_n = 2
        fan_n = 4
        heater_n = 2
    else:
        aircond_n = 1
        fan_n = 2
        heater_n = 1

    fridges = [p for p in products if p.category == "fridge"]
    washers = [p for p in products if p.category == "washer"]
    tvs = [p for p in products if p.category == "tv"]
    airconds = [p for p in products if p.category == "aircond"]
    fans = [p for p in products if p.category == "fan"]
    heaters = [p for p in products if p.category == "water_heater"]
    rice_cookers = [p for p in products if p.category == "rice_cooker"]
    air_fryers = [p for p in products if p.category == "air_fryer"]

    plan: Dict[str, Any] = {
        "fridge": _pick_n(fridges, 1)[0] if fridges else None,
        "washer": _pick_n(washers, 1)[0] if washers else None,
        "tv": _pick_n(tvs, 1)[0] if tvs else None,
        "aircond": _pick_n(airconds, aircond_n),
        "fan": _pick_n(fans, fan_n),
        "water_heater": _pick_n(heaters, heater_n),
        "rice_cooker": _pick_n(rice_cookers, 1)[0] if rice_cookers else None,
        "air_fryer": _pick_n(air_fryers, 1)[0] if air_fryers else None,
    }

    # If requested categories provided, trim plan to only those keys
    if requested_categories and len(requested_categories) > 0:
        keep = set(requested_categories)
        plan = {k: v for k, v in plan.items() if k in keep}

    return plan


def flatten_plan_products(plan: Dict[str, Any]) -> List[Product]:
    out: List[Product] = []
    for key in ["fridge", "washer", "tv", "rice_cooker", "air_fryer"]:
        val = plan.get(key)
        if isinstance(val, Product):
            out.append(val)
    for key in ["aircond", "fan", "water_heater"]:
        lst = plan.get(key) or []
        for p in lst:
            if isinstance(p, Product):
                out.append(p)
    return out