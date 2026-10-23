"""A small, polite JSON client: spaces out requests and backs off when rate-limited."""

import time

import requests


class ApiClient:
    def __init__(self, min_interval: float = 2.1, retries: int = 3, timeout: float = 30, session=None):
        self.min_interval = min_interval  # GeckoTerminal's free tier allows about 30 calls a minute
        self.retries = retries
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": "trading-agent-research/0.1"})
        self._last_call = 0.0

    def get_json(self, url: str) -> dict | None:
        """Return parsed JSON, or None if the resource does not exist (404)."""
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                response = self.session.get(url, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout):
                # Often a connection the server closed while we were idle; a retry reconnects.
                if attempt == self.retries:
                    raise
                time.sleep(5 * (attempt + 1))
                continue
            if response.status_code == 404:
                return None
            if response.status_code == 429 and attempt < self.retries:
                time.sleep(10 * (attempt + 1))  # rate-limited: wait longer each time
                continue
            response.raise_for_status()
            return response.json()
        return None
