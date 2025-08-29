from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

class Payment(BaseModel):
    subtotal: float = 0.0
    tax: float = 0.0
    tip: float = 0.0
    discounts: float = 0.0
    fees: float = 0.0
    method: str = ""
    last4: str = ""

class Meta(BaseModel):
    date: str = ""
    payee: str = ""
    total: float = 0.0
    payment: Payment = Field(default_factory=Payment)

class Item(BaseModel):
    line_index: int = -1
    ocr_text: str = ""
    name: str = ""
    qty: float = 0.0
    unit_price: float = 0.0
    amount: float = 0.0
    confidence: float = 0.0

class PipelineContext(BaseModel):
    raw_text: str = ""
    ocr_lines: List[str] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)
    items: List[Item] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)

class PickResult(BaseModel):
    index: int = -1
    confidence: float = 0.0
    candidates: List[int] = Field(default_factory=list)

class ExtractResult(BaseModel):
    index: int = -1
    value: str | float = ""
    confidence: float = 0.0
