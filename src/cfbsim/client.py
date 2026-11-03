"""CollegeFootballData API client with a local cache and a monthly call budget.

The free tier allows 1,000 calls a month, so a cached response is never fetched
twice unless the caller asks for it. Cached files stay on this machine because
CFBD's terms do not allow raw API data to be committed to a public repo.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from cfbsim.config import API_BASE_URL, DEFAULT_MONTHLY_BUDGET, PROJECT_ROOT, RAW_DIR

TIMEOUT_SECONDS = 60


class CFBDError(RuntimeError):
    """The API could not supply the requested data."""


class MissingKeyError(CFBDError):
    """A request needed the network but no usable API key is configured."""


class BudgetExceededError(CFBDError):
    """A request would pass the local monthly call budget."""


def load_api_key() -> str | None:
    load_dotenv(PROJECT_ROOT / ".env")
    return os.environ.get("CFBD_API_KEY") or None


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


class CFBDClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path = RAW_DIR,
        monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
        session: Any = None,
        month: Callable[[], str] = _current_month,
        ledger_path: Path | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else load_api_key()
        self.cache_dir = Path(cache_dir)
        self.monthly_budget = monthly_budget
        self.session = session or requests.Session()
        self.month = month
        # A separate cache can still count against the project's shared call budget.
        self._ledger_override = Path(ledger_path) if ledger_path else None

    def cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        parts = [f"{key}={params[key]}" for key in sorted(params)]
        name = "__".join(parts) or "all"
        name = re.sub(r"[^A-Za-z0-9=_.-]", "-", name)
        return self.cache_dir / endpoint.strip("/").replace("/", "_") / f"{name}.json"

    def is_cached(self, endpoint: str, **params: Any) -> bool:
        return self.cache_path(endpoint, params).exists()

    def get(self, endpoint: str, *, refresh: bool = False, **params: Any) -> Any:
        """Return the JSON for a request, from cache when possible."""
        path = self.cache_path(endpoint, params)
        if path.exists() and not refresh:
            return json.loads(path.read_text())

        if not self.api_key:
            raise MissingKeyError(
                "No CFBD API key found. Get a free one at https://collegefootballdata.com/key "
                "and put it in a .env file (see .env.example)."
            )
        self._spend_call()
        url = f"{API_BASE_URL}/{endpoint.strip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as error:
            raise CFBDError(f"Request to {endpoint} failed: {type(error).__name__}") from None

        if response.status_code == 401:
            raise MissingKeyError("CFBD rejected the API key (HTTP 401). Check the value in .env.")
        if response.status_code == 429:
            raise BudgetExceededError("CFBD reports the monthly quota is used up (HTTP 429).")
        if response.status_code != 200:
            raise CFBDError(f"{endpoint} returned HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        temporary.replace(path)
        return data

    @property
    def _ledger_path(self) -> Path:
        return self._ledger_override or self.cache_dir / "_usage.json"

    def calls_this_month(self) -> int:
        if not self._ledger_path.exists():
            return 0
        return json.loads(self._ledger_path.read_text()).get(self.month(), 0)

    def _spend_call(self) -> None:
        """Count a call before making it, since failed requests may still use quota."""
        ledger = json.loads(self._ledger_path.read_text()) if self._ledger_path.exists() else {}
        used = ledger.get(self.month(), 0)
        if used >= self.monthly_budget:
            raise BudgetExceededError(
                f"Local budget of {self.monthly_budget} calls for {self.month()} is used up."
            )
        ledger[self.month()] = used + 1
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._ledger_path.write_text(json.dumps(ledger, indent=2))
