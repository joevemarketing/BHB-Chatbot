import json
import os
import re
import string
from typing import List, Dict, Any, Optional

from .models import Product, UserConstraints


def _load_system_prompt() -> str:
    base = os.environ.get("PROMPT_BASE", "config/prompts")
    variant = os.environ.get("PROMPT_VARIANT", "default").strip().lower()
    names = [
        os.path.join(base, "system_product_advisor_" + variant + ".txt") if variant and variant != "default" else None,
        os.path.join(base, "system_product_advisor.txt"),
    ]
    for p in names:
        if not p:
            continue
        try:
            if os.path.exists(p):
                return open(p, "r", encoding="utf-8").read()
        except Exception:
            continue
    return (
        "You are ‘BHB Product Advisor’, a friendly Malaysian electrical shop consultant. "
        "You have TWO roles: (1) BHB Product Advisor – help choose TVs, fridges, washers, airconds, ceiling fans, instant water heaters, and small kitchen appliances such as rice cookers and air fryers. "
        "(2) BHB Customer Service Helper – explain basic info about warranty, delivery, installation, and returns using FAQ answers provided by the backend. "
        "Use ONLY the products provided to you when recommending items. Recommend 1–3 items that fit the user’s needs (budget, home type, family size, room size). "
        "Explain clearly."
        "\nLANGUAGE RULES\n"
        "- First detect the main language of the user’s latest message.\n"
        "- If the message is clearly in English, reply fully in English (natural Malaysian English is okay), and avoid switching into Malay unless the user does it first or explicitly asks.\n"
        "- If the message is mainly in Malay, you may reply in Malay or simple rojak.\n"
        "- If the user mixes both languages, you may reply in a similar style, but keep it clear and easy to read.\n"
        "\nOUTPUT RULES\n"
        "- Do NOT include any URLs or links in your reply.\n"
        "- Present product suggestions as a clean bullet list: ‘Brand Model — RM <price>: short reason’.\n"
        "- Keep the reply concise and scannable; avoid long paragraphs.\n"
        "\nBUDGET & CLARIFICATION RULES\n"
        "- Do NOT assume or invent a budget. If budget isn’t provided in conversation or constraints, ask a short clarifying question.\n"
        "- Summaries must only state facts given by the user or constraints; do not add a budget estimate unless it exists.\n"
        "\nStrict rules:\n"
        "- You are only allowed to mention specific products (brand + model) that appear in the provided JSON `products` list.\n"
        "- If you want to speak generally, you may say ‘a washer’, ‘a fridge’, etc., BUT you must NOT invent brand + model names that are not in the JSON.\n"
        "- If the JSON list is empty, you must say there is no suitable product in the catalog and ask the user to check bhb.com.my or contact the shop. Do not fabricate.\n"
        "- If `user_constraints.category` is set (e.g. 'washer'), you must keep your recommendations to that category only. Do NOT recommend products from other categories (e.g. TV) in this answer.\n"
        "- If `faq_answers` are provided in the context, use them as the source of truth for customer service questions. Do NOT invent detailed policies (fees, exact days) if not in the FAQ; instead say to check bhb.com.my or contact the nearest BHB branch.\n"
        "- If both FAQ answers and products are provided, answer customer service part first, then give product recommendations.\n"
        "\nSCENARIOS\n"
        "- WHOLE HOUSE / MULTI-CATEGORY REQUESTS:\n"
        "  - The backend may send: scenario='whole_house' and requested_categories=['tv','fan','aircond','washer','fridge', ...].\n"
        "  - You MUST reflect ALL requested categories in your understanding. Example: ‘You’re setting up a new house and need a TV, fan, air-conditioner, washer and fridge.’\n"
        "  - Group recommendations by category (TV, Fridge, Washer, Aircond, Fan, Water Heater, Rice Cooker, Air Fryer).\n"
        "  - If you have no product for a category, say so clearly and suggest the customer browse bhb.com.my for that category, while still recommending what you DO have from the catalog.\n"
        "- GENERAL ENQUIRY (no clear category):\n"
        "  - The backend may send: scenario='general_enquiry' with an empty products list.\n"
        "  - Do NOT guess a category. Ask a short clarification such as: ‘Are you thinking about a TV, fridge, washer, air-conditioner, fan, water heater, or all of them?’\n"
        "\nWHOLE HOUSE / BUNDLE\n"
        "- If the request is for a ‘whole new house’ or bundle, you may propose a simple package including: 1 fridge, 1 washer, 1 TV, 1–2 airconds, a few ceiling fans, water heaters, and 1–2 small appliances (rice cooker / air fryer).\n"
        "- Present the package clearly by category (e.g., Washer, Fridge, TV, Aircond, Fan, Water Heater, Rice Cooker, Air Fryer).\n"
        "- Keep the text clean; do NOT include raw URLs.\n"
        "\nREASONING & CLARITY\n"
        "- Carefully read the FULL conversation and the `user_constraints` object from the backend.\n"
        "- Before recommending, briefly summarise the customer’s request in 1–2 sentences using only stated facts (e.g. ‘You’re looking for a 9kg washer for a family of 4.’). Do not add budget unless provided.\n"
        "- If important info is missing or unclear (e.g. room size, budget range, category), ask up to 1–2 short clarification questions instead of guessing.\n"
        "- When recommending products, check each against the constraints: category must match, price close to budget if possible, capacity sensible for household size.\n"
        "- If no products match well, say so honestly and suggest closest alternatives; invite the user to adjust budget or requirements.\n"
        "You MUST base your recommendations on the `user_constraints` and `products` from BACKEND_CONTEXT. Do not ignore them.\n"
        "If products list is not empty, recommend ONLY from that list (do not invent others or prices). "
        "For each product, say why it suits them; do NOT include URLs or ‘click the link’. If you want to direct them to the site, say: ‘You can look for this model on the BHB website.’ Ask at most 1–3 short follow-up questions if key info is missing."
        "\nHONESTY & LIMITS\n"
        "- If the customer asks for a product category that is NOT in the provided products list (for example vacuum cleaner, microwave, built-in oven if not present), be honest: explain your current system is focused on the listed categories above and suggest checking bhb.com.my for those other items. Do NOT invent products that are not in JSON.\n"
        "\nSTORE LOCATION QUESTIONS\n"
        "- The backend may give you a list `store_locations` with label, address, city, state, phone and hours.\n"
        "- When the user asks about store / branch / location, use this list to answer.\n"
        "- If there are multiple relevant stores (e.g. several in Penang), show 1–3 options with: Store/branch name or label; Address; Phone number; Opening hours (short).\n"
        "- If there is no match for the area they mention, apologise and ask them to check the Store Locator on bhb.com.my.\n"
        "- Do NOT invent new branches that are not in the list.\n"
    )

SYSTEM_PROMPT = _load_system_prompt()

def _load_prompt_variant(variant: Optional[str]) -> str:
    v = (variant or "").strip().lower()
    if not v or v == "default":
        return SYSTEM_PROMPT
    try:
        base = os.environ.get("PROMPT_BASE", "config/prompts")
        p = os.path.join(base, f"system_product_advisor_{v}.txt")
        if os.path.exists(p):
            return open(p, "r", encoding="utf-8").read()
    except Exception:
        pass
    return SYSTEM_PROMPT

def redact_pii(text: str) -> str:
    t = text or ""
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", t)
    t = re.sub(r"\b(?:\+?6?0)?\s?(?:\d{2,3}[- ]?)?\d{3}[- ]?\d{4}\b", "[redacted-phone]", t)
    return t

async def moderate_text(text: str) -> Optional[bool]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
        model = os.environ.get("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
        resp = client.moderations.create(model=model, input=text or "")
        flagged = bool(resp.results[0].flagged) if hasattr(resp, "results") and resp.results else False
        return flagged
    except Exception:
        return None


def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _is_english(text: str) -> bool:
    """
    Lightweight heuristic: if strong Malay tokens are present, treat as Malay; otherwise default to English.
    This helps the fallback reply obey language rules without external libraries.
    """
    t = (text or "").lower()
    malay_keywords = [
        "bajet", "saiz", "rumah", "cawangan", "kedai", "tolong", "beritahu",
        "mesin basuh", "peti sejuk", "penghantaran", "warranty", "waranti", "pemasangan",
        "rojak", "harga", "belanjawan", "hantar", "pulang", "kediaman",
    ]
    if any(k in t for k in malay_keywords):
        return False
    # If message contains mostly ASCII and common English words, prefer English
    english_hints = ["washer", "family", "budget", "price", "room", "delivery", "return", "warranty", "store"]
    if any(w in t for w in english_hints):
        return True
    # Default to English to avoid accidental Malay mixing
    return True


async def generate_reply(
    user_message: str,
    constraints: dict,
    products: List[Dict[str, Any]],
    extra_context: Optional[Dict[str, Any]] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
    prompt_variant: Optional[str] = None,
) -> str:
    # If OpenAI-compatible environment available, try calling it; else fallback.
    if _has_openai():
        try:
            # Prefer OpenAI Python SDK v1 style
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
            model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
            payload_dict = {
                "user_constraints": constraints,
                "products": products,
                "vision_result": (extra_context or {}).get("vision_result"),
                "faq_answers": (extra_context or {}).get("faq_answers"),
                "store_locations": (extra_context or {}).get("store_locations"),
                "scenario": (extra_context or {}).get("scenario"),
                "requested_categories": (extra_context or {}).get("requested_categories"),
            }
            payload = json.dumps(payload_dict, ensure_ascii=False)
            clean_user = redact_pii(user_message)
            flagged = await moderate_text(clean_user)
            if flagged:
                return "I can’t assist with that request. Please ask about appliances, stores, warranty, delivery, installation or returns."
            system_prompt = _load_prompt_variant(prompt_variant)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": "BACKEND_CONTEXT: " + payload},
                {"role": "system", "content": "VISION_CONTEXT: " + json.dumps(extra_context or {}, ensure_ascii=False)},
                {"role": "system", "content": "STORE_CONTEXT: " + json.dumps({"store_locations": (extra_context or {}).get("store_locations")}, ensure_ascii=False)},
                {"role": "system", "content": "FAQ_CONTEXT: " + json.dumps({"faq_answers": (extra_context or {}).get("faq_answers")}, ensure_ascii=False)},
            ]
            # Include full conversation (user/assistant turns) if provided
            if conversation and len(conversation) > 0:
                for m in conversation:
                    role = m.get("role")
                    if role in {"user", "assistant"}:
                        content = m.get("content", "")
                        if role == "user":
                            content = redact_pii(content)
                        messages.append({"role": role, "content": content})
            else:
                messages.append({"role": "user", "content": clean_user})
            resp = client.chat.completions.create(model=model, messages=messages)
            txt = resp.choices[0].message.content or ""
            return txt.strip()
        except Exception:
            pass

    # Fallback: template a friendly response grounded in provided products
    faq_answers = (extra_context or {}).get("faq_answers") if extra_context else None
    store_locations = (extra_context or {}).get("store_locations") if extra_context else None
    # Decide language once for all fallback branches
    use_english = _is_english(user_message)
    scenario = (extra_context or {}).get("scenario") if extra_context else None
    if not products:
        if faq_answers:
            # Service-only fallback using provided FAQ answers
            lines: List[str] = [
                ("Here is service information based on the FAQ:" if use_english else "Berikut info perkhidmatan berdasarkan FAQ:"),
            ]
            for ans in faq_answers:
                lines.append(f"- {ans}")
            lines.append(
                (
                    "For exact details, please check bhb.com.my or contact your nearest BHB branch."
                    if use_english else
                    "Untuk maklumat tepat, sila rujuk bhb.com.my atau hubungi cawangan BHB terdekat."
                )
            )
            return "\n".join(lines)
        elif store_locations is not None:
            # Store-only fallback: list up to 3 locations or ask to check Store Locator
            if store_locations:
                lines: List[str] = [
                    ("Here are some nearby BHB branches:" if use_english else "Berikut beberapa cawangan BHB berdekatan:"),
                ]
                for s in store_locations[:3]:
                    label = s.get("label")
                    address = s.get("address")
                    tel = s.get("tel")
                    hours = s.get("hours")
                    brand = s.get("brand_shop")
                    brand_str = f" ({brand})" if brand else ""
                    lines.append(
                        (
                            f"- {label}{brand_str} — {address}. Tel: {tel}. Hours: {hours}"
                            if use_english else
                            f"- {label}{brand_str} — {address}. Tel: {tel}. Waktu operasi: {hours}"
                        )
                    )
                lines.append(
                    (
                        "If you need more options, please check the Store Locator on bhb.com.my."
                        if use_english else
                        "Jika perlukan pilihan lain, sila semak Store Locator di bhb.com.my."
                    )
                )
                return "\n".join(lines)
            else:
                return (
                    (
                        "Sorry, I couldn’t find a matching branch in that area. Please check the Store Locator on bhb.com.my for the latest locations."
                    )
                    if use_english else
                    (
                        "Maaf, tiada cawangan yang sepadan di kawasan itu. Sila semak Store Locator di bhb.com.my untuk lokasi terkini."
                    )
                )
        else:
            # General enquiry: ask for category clarification instead of defaulting
            if scenario == "general_enquiry":
                return (
                    (
                        "You’re setting up a new house and looking for electrical appliances. To help you better, are you thinking about a TV, fridge, washer, air-conditioner, fan, water heater, or all of them?"
                    )
                    if use_english else
                    (
                        "Anda sedang melengkapkan rumah baru dan perlukan peralatan elektrik. Untuk bantu lebih tepat, adakah anda fikir tentang TV, peti sejuk, mesin basuh, penghawa dingin, kipas, pemanas air, atau semuanya sekali?"
                    )
                )
            return (
                (
                    "Sorry, I don’t have products to recommend right now. If you can share the category (e.g., washer) or your budget, I can suggest suitable options."
                )
                if use_english else
                (
                    "Maaf, tiada produk untuk dicadangkan sekarang. Kalau boleh, beritahu kategori (contoh: washer) atau bajet supaya saya boleh beri cadangan yang tepat."
                )
            )

    # Compose natural summary and recommendations, obeying language rules
    cat = constraints.get("category")
    hh = constraints.get("household_size")
    home_type = constraints.get("home_type")
    bmin = constraints.get("budget_min_rm")
    bmax = constraints.get("budget_max_rm")

    use_english = _is_english(user_message)
    lines: List[str] = []

    # Natural summary sentence (no debug-like output)
    if use_english:
        if cat == "washer":
            if hh:
                lines.append(f"You’re looking for a washer suitable for a family of {hh}.")
            else:
                lines.append("You’re looking for a suitable washer.")
        elif cat == "fridge":
            lines.append("You’re looking for a suitable fridge.")
        elif cat == "tv":
            lines.append("You’re looking for a suitable TV.")
        elif cat == "aircond":
            lines.append("You’re looking for a suitable air conditioner.")
        elif cat in {"fan", "water_heater", "rice_cooker", "air_fryer"}:
            lines.append("You’re looking for suitable home appliances in that category.")
        else:
            lines.append("You’re looking for suitable home appliances.")
    else:
        if cat == "washer":
            if hh:
                lines.append(f"Anda mencari mesin basuh sesuai untuk keluarga {hh} orang.")
            else:
                lines.append("Anda mencari mesin basuh yang sesuai.")
        elif cat == "fridge":
            lines.append("Anda mencari peti sejuk yang sesuai.")
        elif cat == "tv":
            lines.append("Anda mencari TV yang sesuai.")
        elif cat == "aircond":
            lines.append("Anda mencari penghawa dingin yang sesuai.")
        elif cat in {"fan", "water_heater", "rice_cooker", "air_fryer"}:
            lines.append("Anda mencari peralatan yang sesuai dalam kategori itu.")
        else:
            lines.append("Anda mencari peralatan elektrik rumah yang sesuai.")

    # Recommendations: group by category when we have a whole-house plan context
    is_house_plan = bool((extra_context or {}).get("house_plan")) or scenario == "whole_house"
    requested_categories = (extra_context or {}).get("requested_categories") or []
    if is_house_plan and requested_categories:
        if use_english:
            lines.append("You’re setting up a new house and need: " + ", ".join(requested_categories) + ".")
        else:
            lines.append("Anda sedang melengkapkan rumah baru dan perlukan: " + ", ".join(requested_categories) + ".")
    if is_house_plan:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for p in products:
            grouped.setdefault(p.get("category"), []).append(p)
        section_order = ["washer", "fridge", "tv", "aircond", "fan", "water_heater", "rice_cooker", "air_fryer"]
        for sec in section_order:
            items = grouped.get(sec) or []
            if not items:
                continue
            header_en = {
                "washer": "Washer",
                "fridge": "Fridge",
                "tv": "TV",
                "aircond": "Air conditioner",
                "fan": "Fan",
                "water_heater": "Water heater",
                "rice_cooker": "Rice cooker",
                "air_fryer": "Air fryer",
            }[sec]
            header_ms = {
                "washer": "Mesin basuh",
                "fridge": "Peti sejuk",
                "tv": "TV",
                "aircond": "Penghawa dingin",
                "fan": "Kipas",
                "water_heater": "Pemanas air",
                "rice_cooker": "Periuk nasi",
                "air_fryer": "Air fryer",
            }[sec]
            lines.append((header_en + ":") if use_english else (header_ms + ":"))
            for p in items[:3]:
                name = f"{p.get('brand')} {p.get('model_name')}"
                price = p.get("price_rm")
                why_bits: List[str] = []
                if p.get("energy_label"):
                    why_bits.append("energy-efficient" if use_english else f"{p['energy_label']} (jimat elektrik)")
                if p.get("capacity_liters"):
                    why_bits.append(f"{p['capacity_liters']}L capacity" if use_english else f"kapasiti {p['capacity_liters']}L")
                if p.get("capacity_kg"):
                    why_bits.append(f"{p['capacity_kg']}kg capacity" if use_english else f"kapasiti {p['capacity_kg']}kg")
                if p.get("screen_size_inches"):
                    why_bits.append(f"{p['screen_size_inches']}\" screen" if use_english else f"skrin {p['screen_size_inches']}\"")
                if p.get("recommended_room_size_sqm"):
                    why_bits.append(f"for ~{p['recommended_room_size_sqm']} sqm room" if use_english else f"untuk bilik ~{p['recommended_room_size_sqm']} sqm")
                if p.get("fan_size_inch"):
                    why_bits.append(f"{p['fan_size_inch']}\" fan" if use_english else f"kipas {p['fan_size_inch']}")
                if p.get("heater_type"):
                    why_bits.append(p.get("heater_type"))
                if p.get("capacity_l"):
                    why_bits.append(f"{p['capacity_l']}L")
                if p.get("fryer_capacity_l"):
                    why_bits.append(f"{p['fryer_capacity_l']}L")
                why = ", ".join([w for w in why_bits if w]) or ("good value" if use_english else "nilai baik")
                line = f"- {name} — {'around RM ' + str(price) if use_english else 'RM ' + str(price)}: {why}."
                lines.append(line)
    else:
        # Simple list if no house plan
        lines.append("Here are a few options:" if use_english else "Berikut beberapa pilihan:")
        for p in products[:3]:
            name = f"{p.get('brand')} {p.get('model_name')}"
            price = p.get("price_rm")
            why_bits: List[str] = []
            if p.get("energy_label"):
                why_bits.append("energy-efficient" if use_english else f"{p['energy_label']} (jimat elektrik)")
            if p.get("capacity_liters"):
                why_bits.append(f"{p['capacity_liters']}L capacity" if use_english else f"kapasiti {p['capacity_liters']}L")
            if p.get("capacity_kg"):
                why_bits.append(f"{p['capacity_kg']}kg capacity" if use_english else f"kapasiti {p['capacity_kg']}kg")
            if p.get("screen_size_inches"):
                why_bits.append(f"{p['screen_size_inches']}\" screen" if use_english else f"skrin {p['screen_size_inches']}\"")
            if p.get("recommended_room_size_sqm"):
                why_bits.append(f"for ~{p['recommended_room_size_sqm']} sqm room" if use_english else f"untuk bilik ~{p['recommended_room_size_sqm']} sqm")
            why = ", ".join(why_bits) or "good value"
            line = f"- {name} — {'around RM ' + str(price) if use_english else 'RM ' + str(price)}: {why}."
            lines.append(line)

    # One polite follow-up at the end (budget preferred if missing)
    if use_english:
        if bmin is None and bmax is None:
            lines.append("If you share roughly your budget range, I can refine this list further.")
        elif cat == "washer" and hh is None:
            lines.append("How many people are in your household? That helps pick the right capacity.")
    else:
        if bmin is None and bmax is None:
            lines.append("Kalau boleh beritahu anggaran bajet anda, saya boleh refine senarai ini.")
        elif cat == "washer" and hh is None:
            lines.append("Berapa orang dalam rumah? Itu bantu pilih kapasiti yang sesuai.")
    return "\n".join(lines)
