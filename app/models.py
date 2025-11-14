from typing import List, Optional, Literal
from pydantic import BaseModel


Category = Literal[
    "tv",
    "fridge",
    "aircond",
    "washer",
    "fan",
    "water_heater",
    "rice_cooker",
    "air_fryer",
]


class DimensionsMM(BaseModel):
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None


class Product(BaseModel):
    # Common
    category: Category
    brand: str
    model_name: str
    sku: str
    price_rm: float
    energy_label: Optional[str] = None
    noise_level_db: Optional[float] = None
    features: Optional[List[str]] = None
    recommended_for: Optional[str] = None
    bhb_product_url: Optional[str] = None
    website_search_text: Optional[str] = None

    # Link info for UI convenience
    # {"type": "direct"|"search", "url": "..."}
    bhb_link: Optional[dict] = None

    # Fridge-specific
    capacity_liters: Optional[float] = None
    dimensions_mm: Optional[DimensionsMM] = None

    # Washer-specific
    capacity_kg: Optional[float] = None

    # Aircond-specific
    hp: Optional[float] = None
    recommended_room_size_sqm: Optional[float] = None

    # TV-specific
    screen_size_inches: Optional[float] = None
    resolution: Optional[str] = None
    panel_type: Optional[str] = None

    # Fan-specific
    fan_type: Optional[str] = None        # e.g., "ceiling", "wall"
    fan_size_inch: Optional[float] = None

    # Water heater-specific
    heater_type: Optional[str] = None     # e.g., "instant", "storage"
    pump: Optional[bool] = None           # built-in pump or non-pump
    power_watt: Optional[int] = None

    # Rice cooker-specific
    capacity_l: Optional[float] = None
    cooker_type: Optional[str] = None     # e.g., "microcomputer", "conventional", "fuzzy_logic"

    # Air fryer-specific
    fryer_capacity_l: Optional[float] = None
    fryer_power_watt: Optional[int] = None


class FAQItem(BaseModel):
    id: str
    category: str
    keywords: List[str]
    answer: str


class UserConstraints(BaseModel):
    category: Optional[Category] = None
    budget_min_rm: Optional[float] = None
    budget_max_rm: Optional[float] = None
    household_size: Optional[int] = None
    home_type: Optional[str] = None  # e.g. condo, terrace
    room_size_sqm: Optional[float] = None
    noise_sensitivity: Optional[str] = None  # low | medium | high
    brand_preferences: Optional[List[str]] = None
    priority: Optional[List[str]] = None  # e.g. ["energy_saving","capacity","quiet"]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    suggested_products: List[Product]
    constraints: UserConstraints