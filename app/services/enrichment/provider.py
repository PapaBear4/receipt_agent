import time
from typing import Iterable, Optional
from dataclasses import asdict
from app.config import settings
from .types import EnrichedItem

class Enricher:
    def enrich(self, vendor: str, query: str) -> Optional[EnrichedItem]:
        raise NotImplementedError

class NoopEnricher(Enricher):
    def enrich(self, vendor: str, query: str) -> Optional[EnrichedItem]:
        return None

# Minimal stub you can extend to call Bing Web Search, SerpAPI, etc.
class SearchEnricher(Enricher):
    def __init__(self) -> None:
        self.max_results = settings.ENRICH_MAX_RESULTS
        self.timeout = settings.ENRICH_TIMEOUT
        self.ua = settings.ENRICH_USER_AGENT
        self.api_key = settings.ENRICH_SEARCH_API_KEY
        self.provider = settings.ENRICH_PROVIDER

    def enrich(self, vendor: str, query: str) -> Optional[EnrichedItem]:
        # TODO: implement real provider calls. For now, return None.
        # Example signature kept for future expansion.
        return None

# Simple token bucket rate limiter
class RateLimiter:
    def __init__(self, rate_per_min: int) -> None:
        self.tokens = rate_per_min
        self.capacity = rate_per_min
        self.refill_time = 60.0
        self.last = time.monotonic()

    def acquire(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.capacity, self.tokens + elapsed * (self.capacity / self.refill_time))
        if self.tokens < 1:
            sleep_for = (1 - self.tokens) * (self.refill_time / self.capacity)
            time.sleep(max(0.0, sleep_for))
            self.tokens = 0
        self.tokens -= 1

_limiter = RateLimiter(settings.ENRICH_RATE_LIMIT_PER_MIN)

class LimitedEnricher(Enricher):
    def __init__(self, inner: Enricher) -> None:
        self.inner = inner

    def enrich(self, vendor: str, query: str) -> Optional[EnrichedItem]:
        _limiter.acquire()
        return self.inner.enrich(vendor, query)


def get_enricher() -> Enricher:
    if not settings.ENRICH_ENABLED:
        return NoopEnricher()
    # Right now: generic search-based enricher stub
    return LimitedEnricher(SearchEnricher())
