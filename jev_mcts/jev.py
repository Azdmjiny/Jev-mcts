"""Small typed client for TypeSafe's documented System One HTTP API."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cli import load_local_env
from pathlib import Path


PRICE_PER_MILLION_INPUT_TOKENS_USD = 0.042


class JevError(RuntimeError):
    """The real Jev service did not return a usable response."""


class BudgetExceeded(JevError):
    """The run reached its configured cost or time limit."""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    models: set[str] = field(default_factory=set)

    @property
    def estimated_usd(self) -> float:
        return self.input_tokens * PRICE_PER_MILLION_INPUT_TOKENS_USD / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_usd": self.estimated_usd,
            "models": sorted(self.models),
        }


class JevClient:
    """Use real HTTP responses only; no simulated response path exists."""

    def __init__(
        self,
        model: str = "jev-latest",
        *,
        max_usd: float = 5.0,
        deadline: float | None = None,
        timeout: float = 20.0,
        usage: Usage | None = None,
    ) -> None:
        load_local_env(Path(__file__).resolve().parent.parent)
        self.api_key = os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is missing")
        self.model = model
        self.max_usd = max_usd
        self.deadline = deadline
        self.timeout = timeout
        self.usage = usage or Usage()

    def check_budget(self) -> None:
        if self.usage.estimated_usd >= self.max_usd:
            raise BudgetExceeded(f"Jev estimate reached ${self.max_usd:.2f}")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise BudgetExceeded("experiment deadline reached")

    def ask_choice(
        self,
        state: Any,
        instructions: str,
        options: dict[str, str | None],
    ) -> dict[str, float]:
        if not 2 <= len(options) <= 255:
            raise ValueError("Jev Choice requires 2 to 255 options")
        self.check_budget()
        body = {
            "model": self.model,
            "state": state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": options,
                }
            },
        }
        request = Request(
            "https://api.typesafe.ai/v1/systemone",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(3):
            self.check_budget()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise JevError(f"Jev HTTP {exc.code}") from exc
            except URLError as exc:
                if attempt == 2:
                    raise JevError(f"Jev connection error: {exc.reason}") from exc
            time.sleep(2**attempt)
        try:
            answer = payload["answers"]["decision"]
            probabilities = answer["probabilities"]
            if answer["type"] != "choice" or set(probabilities) != set(options):
                raise ValueError("choice response does not match request")
            if any(not math.isfinite(float(v)) or float(v) < 0 for v in probabilities.values()):
                raise ValueError("choice probabilities are invalid")
            total = sum(float(v) for v in probabilities.values())
            if total <= 0:
                raise ValueError("choice probabilities sum to zero")
            usage = payload["usage"]
            self.usage.calls += 1
            self.usage.input_tokens += int(usage["input_tokens"])
            self.usage.output_tokens += int(usage["output_tokens"])
            self.usage.models.add(str(payload["model"]))
            return {key: float(probabilities[key]) / total for key in options}
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(f"invalid Jev Choice response: {exc}") from exc
