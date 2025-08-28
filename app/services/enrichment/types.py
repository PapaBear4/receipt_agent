from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class EnrichedItem:
    name: Optional[str]
    sku: Optional[str]
    brand: Optional[str]
    category: Optional[str]
    url: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    image: Optional[str]
    extra: Dict[str, Any]
