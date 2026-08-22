"""OpenAI Batch API collector for the benchmark LLM-as-a-judge call.

Enabled by default with an OpenAI ``EVAL_MODEL`` and ``OPENAI_API_KEY``. The
benchmark registers every judge prompt during its first pass, then submits a
single batch to the ``/v1/chat/completions`` endpoint. Batch creation or
resolution failures propagate; the benchmark never silently switches to the
more expensive synchronous path.

The OpenAI SDK calls are synchronous, so they run in worker threads via
``asyncio.to_thread`` to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from collections.abc import Callable

from utils.parsers import clean_and_unwrap_json

logger = logging.getLogger("Prehop")

# Terminal batch states per the OpenAI Batch API.
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenAIBatchJudge:
    """Collect judge prompts, run them as one OpenAI batch, map results back."""

    def __init__(
        self,
        model: str,
        api_key: str,
        poll_seconds: int = 15,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.poll_seconds = max(2, int(poll_seconds))
        self._requests: list[tuple[str, str]] = []

    def register(self, custom_id: str, prompt: str) -> None:
        self._requests.append((str(custom_id), prompt))

    @property
    def count(self) -> int:
        return len(self._requests)

    def _build_jsonl(self) -> bytes:
        buf = io.BytesIO()
        for custom_id, prompt in self._requests:
            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            }
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        return buf.getvalue()

    def _submit_sync(self, on_submitted: Callable[[str], None] | None = None) -> str | None:
        """Upload the JSONL and create the batch. Returns the batch id (no poll)."""
        from openai import OpenAI  # local import so the dep is only needed when used

        client = OpenAI(api_key=self.api_key)
        upload = client.files.create(
            file=("judge_batch.jsonl", self._build_jsonl()),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        if on_submitted is not None:
            on_submitted(batch.id)
        logger.info("Judge batch submitted: id=%s, %d requests", batch.id, self.count)
        return batch.id

    async def submit(self, on_submitted: Callable[[str], None] | None = None) -> str | None:
        """Submit the batch without waiting. Returns the batch id (or None when
        there is nothing to judge). Use `resolve_batches`/`poll_and_fetch` later
        to retrieve the results — the work runs asynchronously on OpenAI's side."""
        if not self._requests:
            return None
        return await asyncio.to_thread(self._submit_sync, on_submitted)


def _parse_batch_output(out_text: str, total: int | None = None) -> dict[str, dict | None]:
    """Parse a batch output JSONL into {custom_id: parsed_payload}."""
    results: dict[str, dict | None] = {}
    for raw in out_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            custom_id = row.get("custom_id")
            content = row["response"]["body"]["choices"][0]["message"]["content"]
            results[custom_id] = json.loads(clean_and_unwrap_json(content))
        except (AttributeError, IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Batch judge: could not parse output line: %s", exc)
    logger.info("Judge batch parsed: %d%s payloads", len(results), f"/{total}" if total else "")
    return results


def poll_and_fetch(
    api_key: str,
    batch_id: str,
    poll_seconds: int = 15,
    client=None,
) -> dict[str, dict | None]:
    """Block until `batch_id` reaches a terminal state, then download + parse.

    No client-side timeout — the OpenAI batch SLA is up to 24h. Raises if the
    batch ends in any state other than ``completed``.
    """
    from openai import OpenAI

    client = client or OpenAI(api_key=api_key)
    poll_seconds = max(2, int(poll_seconds))
    batch = client.batches.retrieve(batch_id)
    waited = 0
    while batch.status not in _TERMINAL:
        time.sleep(poll_seconds)
        waited += poll_seconds
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        logger.info(
            "Judge batch %s: status=%s, %ss elapsed%s",
            batch.id,
            batch.status,
            waited,
            f", {counts.completed}/{counts.total} done" if counts else "",
        )
    if batch.status != "completed":
        raise RuntimeError(f"Judge batch {batch_id} ended in state '{batch.status}'")
    out_text = client.files.content(batch.output_file_id).text
    return _parse_batch_output(out_text)


def resolve_batches(
    api_key: str,
    batch_ids: list[str],
    poll_seconds: int = 15,
) -> dict[str, dict[str, dict | None]]:
    """Poll several batches concurrently. Returns {batch_id: {custom_id: payload}}.

    Any failed/expired batch or download error propagates so the benchmark is
    visibly incomplete; the pending manifest remains available for diagnosis.
    """
    from concurrent.futures import ThreadPoolExecutor

    ids = [b for b in dict.fromkeys(batch_ids) if b]
    out: dict[str, dict[str, dict | None]] = {}
    if not ids:
        return out

    def _one(bid: str) -> tuple[str, dict[str, dict | None]]:
        return bid, poll_and_fetch(api_key, bid, poll_seconds)

    with ThreadPoolExecutor(max_workers=max(1, len(ids))) as pool:
        for bid, res in pool.map(_one, ids):
            out[bid] = res
    return out
