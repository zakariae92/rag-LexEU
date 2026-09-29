"""Load test: users asking golden-set questions (EN and FR) through the public API.

    LEXEU_API_KEY=lx_... uv run locust -f loadtests/locustfile.py --host http://127.0.0.1:8090

Run `loadtests/run.py` for the stepped, headless benchmark behind ADR 0009.
"""

import os
import random
from pathlib import Path

import yaml
from locust import FastHttpUser, between, task

GOLDEN = Path(__file__).parent.parent / "eval" / "golden" / "golden_v1.yaml"
QUESTIONS = [i["question"] for i in yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["items"]]


class Asker(FastHttpUser):
    wait_time = between(1, 3)  # a person reads the answer before asking again

    def on_start(self) -> None:
        self.headers = {"Authorization": f"Bearer {os.environ['LEXEU_API_KEY']}"}

    @task
    def ask(self) -> None:
        question = random.choice(QUESTIONS)  # noqa: S311 (sampling, not security)
        with self.client.post(
            "/v1/ask", json={"question": question}, headers=self.headers, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif resp.json().get("refused") and resp.json().get("refusal_reason") in (
                "invalid_output",
                "ungrounded",
            ):
                resp.failure("model glitch")
