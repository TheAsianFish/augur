"""Decode throughput of a local Ollama model, one request against several at once.

    python3 scripts/ollama_throughput.py [MODEL] [CONCURRENT]

Sends one short forecasting prompt alone, then CONCURRENT copies at once
(distinct seeds), and prints output tokens, wall seconds and tokens per
second for each. Ollama must run with OLLAMA_NUM_PARALLEL >= CONCURRENT for
the copies to be served together. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
PROMPT = (
    "You are a forecaster. Question: Will the S&P 500 close above 6000 on 2026-06-30? "
    "Think briefly (under 150 words), then end with 'Probability: NN%'."
)


def _one(model: str, seed: int) -> int:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "stream": False,
            "options": {"temperature": 0.7, "seed": seed, "num_ctx": 4096},
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310 - local Ollama
        f"{HOST}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310
        return int(json.loads(response.read()).get("eval_count") or 0)


def main(argv: list[str]) -> int:
    model = argv[0] if argv else "qwen2.5:7b"
    concurrent = int(argv[1]) if len(argv) > 1 else 5
    _one(model, 999)  # load the model; not timed
    for n in (1, concurrent):
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=n) as pool:
            tokens = sum(pool.map(lambda i: _one(model, i), range(n)))
        seconds = time.monotonic() - started
        print(
            f"{model}: {n} request(s) at once: {tokens} output tokens in {seconds:.1f} s, "
            f"{tokens / seconds:.1f} tokens/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
