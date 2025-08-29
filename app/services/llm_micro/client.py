from __future__ import annotations
import time
import requests
from typing import Tuple
from app.config import settings

class LlmClient:
    def __init__(self, model: str | None = None, endpoint: str | None = None, timeout: int | None = None):
        self.model = model or settings.OLLAMA_MODEL
        self.endpoint = (endpoint or settings.OLLAMA_ENDPOINT).rstrip('/')
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def generate_json(self, prompt: str, temperature: float = 0.2, num_predict: int = 128) -> Tuple[str, dict]:
        url = f"{self.endpoint}/api/generate"
        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": float(temperature),
                "num_predict": int(num_predict),
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        data.setdefault("_client_wall_sec", max(0.0, time.perf_counter() - t0))
        return (str(data.get("response", "") or "").strip(), data)
