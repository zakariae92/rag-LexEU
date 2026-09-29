"""Stepped load test: N concurrent users for a fixed time per step, then latency per stage.

    LEXEU_API_KEY=lx_... uv run python loadtests/run.py --users 1 5 10 20 --duration 60 \
        --label mock-1s --host http://127.0.0.1:8090

Client-side percentiles come from Locust; the per-stage breakdown (retrieval vs generation) comes
from the API's own histograms in Prometheus, over each step's window.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
PROMETHEUS = "http://127.0.0.1:9090"


def prom(query: str, at: float) -> dict[str, float]:
    url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": query, "time": at})
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 (fixed http URL)
        data = json.load(r)["data"]["result"]
    return {row["metric"].get("stage", "value"): float(row["value"][1]) for row in data}


def step(users: int, duration: int, host: str, label: str) -> dict[str, object]:
    prefix = RESULTS / f"{label}_u{users}"
    start = time.time()
    subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [
            sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--headless",
            "--host", host, "-u", str(users), "-r", str(users), "-t", f"{duration}s",
            "--csv", str(prefix), "--only-summary", "--loglevel", "WARNING",
        ],
        check=False,
    )  # fmt: skip
    time.sleep(20)  # let Prometheus scrape the last requests (15 s interval)
    end = time.time()
    window = f"{int(end - start)}s"

    with open(f"{prefix}_stats.csv", encoding="utf-8") as f:
        total = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    stages = prom(
        "histogram_quantile(0.95, sum by (le, stage) "
        f'(increase(lexeu_answer_stage_seconds_bucket{{cache="miss"}}[{window}])))',
        end,
    )
    return {
        "users": users,
        "requests": int(total["Request Count"]),
        "failures": int(total["Failure Count"]),
        "rps": round(float(total["Requests/s"]), 2),
        "p50_ms": float(total["50%"]),
        "p95_ms": float(total["95%"]),
        "p99_ms": float(total["99%"]),
        "p95_retrieval_ms": round(stages.get("retrieval", 0) * 1000),
        "p95_generation_ms": round(stages.get("generation", 0) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--host", default="http://127.0.0.1:8090")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    rows = []
    for users in args.users:
        row = step(users, args.duration, args.host, args.label)
        print(json.dumps(row), flush=True)
        rows.append(row)
    (RESULTS / f"{args.label}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        f"### {args.label}",
        "",
        "| Users | Requests | Failures | Req/s | p50 | p95 | p99 "
        "| p95 retrieval | p95 generation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['users']} | {r['requests']} | {r['failures']} | {r['rps']} | {r['p50_ms']:.0f} ms"
        f" | {r['p95_ms']:.0f} ms | {r['p99_ms']:.0f} ms | {r['p95_retrieval_ms']} ms"
        f" | {r['p95_generation_ms']} ms |"
        for r in rows
    ]
    (RESULTS / f"{args.label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
