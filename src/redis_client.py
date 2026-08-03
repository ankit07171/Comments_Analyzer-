"""
redis_client.py
-----------------
Minimal wrapper around the Upstash Redis REST API.

Upstash is used here (instead of a self-hosted Redis + Docker) for:
  - de-duplicating alerts so the same negative comment doesn't spawn a
    new Jira ticket every time the dashboard re-analyzes a dataset
  - rolling counters (comments processed today, alerts raised today)
  - lightweight caching of the last analysis run

The client degrades gracefully: if UPSTASH_REDIS_REST_URL / TOKEN are not
configured, every method becomes a safe no-op so the rest of the app keeps
working with in-memory/session-only state.
"""

from __future__ import annotations

import os
import requests


class UpstashRedis:
    def __init__(self):
        self.base_url = os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/")
        self.token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

    def is_configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def _call(self, *segments) -> dict | None:
        if not self.is_configured():
            return None
        try:
            path = "/".join(str(s) for s in segments)
            resp = requests.get(f"{self.base_url}/{path}", headers=self._headers(), timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            return None
        return None

    # ---------------- basic ops ----------------

    def get(self, key: str):
        result = self._call("get", key)
        return result.get("result") if result else None

    def set(self, key: str, value, ex_seconds: int | None = None):
        if not self.is_configured():
            return False
        result = self._call("set", key, value)
        if ex_seconds:
            self._call("expire", key, ex_seconds)
        return bool(result)

    def setnx_with_ttl(self, key: str, value, ex_seconds: int) -> bool:
        """Set only if the key does not already exist (used for alert
        de-duplication). Returns True if this call created the key."""
        if not self.is_configured():
            return True  # no dedupe store available -> treat as "new" every time
        existing = self.get(key)
        if existing is not None:
            return False
        self.set(key, value, ex_seconds=ex_seconds)
        return True

    def incr(self, key: str) -> int | None:
        result = self._call("incr", key)
        return result.get("result") if result else None

    def exists(self, key: str) -> bool:
        result = self._call("exists", key)
        return bool(result and result.get("result"))


_client_singleton: UpstashRedis | None = None


def get_client() -> UpstashRedis:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = UpstashRedis()
    return _client_singleton
