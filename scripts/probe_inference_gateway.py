#!/usr/bin/env python3
"""Live, non-secret probes for the canonical LiteLLM paper transport."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.inference_transport import InferenceTransport
from core.strategy_registry import PAPER_TRANSPORT
from core.vllm_client import VLLMClient


async def probe(strategy: str, kind: str) -> dict[str, object]:
    os.environ["RAG_PAPER_MODE"] = "true"
    transport = InferenceTransport.resolve(strategy)
    client = VLLMClient(transport.generation_model)
    if kind == "chat":
        answer = await client.generate_response(
            [{"role": "user", "content": "Reply with exactly: gateway-ok"}],
            temperature=0.0,
            max_tokens=16,
        )
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LiteLLM chat probe returned an empty response")
        return {"status": "canary_passed", "schema_version": 1, "stage": "chat_probe", "answer": answer, "request_count": 1}

    if kind == "embedding":
        values = [f"gateway embedding probe {index}" for index in range(PAPER_TRANSPORT.embedding_batch_size)]
    else:
        # The gateway must reject this deliberately oversized aggregate with a
        # size-dependent error so the client demonstrates selective bisection.
        unit = "oversized-gateway-probe " * 8000
        values = [f"{index} {unit}" for index in range(PAPER_TRANSPORT.embedding_batch_size)]
    vectors = await client.get_embeddings(values, encoding_type="document")
    valid = (
        len(vectors) == len(values)
        and all(len(vector) == PAPER_TRANSPORT.embedding_dimensions for vector in vectors)
        and all(math.isfinite(float(value)) for vector in vectors for value in vector)
    )
    if not valid:
        raise RuntimeError("LiteLLM embedding probe violated count/dimension/finite contract")
    bisections = int(getattr(client, "_embedding_bisection_count", 0))
    if kind == "bisection" and bisections < 1:
        raise RuntimeError("oversized payload was accepted without exercising selective bisection")
    return {
        "status": "canary_passed",
        "schema_version": 1,
        "stage": f"{kind}_probe",
        "batch_size": transport.embedding_batch_size,
        "concurrency": transport.embedding_concurrency,
        "vectors": [{"index": i, "embedding": vector} for i, vector in enumerate(vectors)] if kind == "embedding" else [],
        "oversized_rejected": bisections > 0,
        "selective_bisection_passed": bisections > 0,
        "count": len(vectors),
        "dimension": PAPER_TRANSPORT.embedding_dimensions,
        "finite": True,
        "bisections": bisections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("prehop",), default="prehop")
    parser.add_argument("--probe", choices=("chat", "embedding", "bisection"), required=True)
    args = parser.parse_args()
    from core.paper_policy import configure_target_environment
    from scripts.check_paper_runtime import _load_runner_environment

    _load_runner_environment()
    configure_target_environment("prehop", "multihoprag", "gateway-probe")
    print(json.dumps(asyncio.run(probe(args.strategy, args.probe)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
