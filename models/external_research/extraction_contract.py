"""Adapter-owned extraction validation; never infer missing entities or relations."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

PROFILE = 'strict-extraction-v1'


class ExtractionAudit:
    def __init__(self, path: Path, *, profile=PROFILE):
        self.profile = profile
        self.path = path
        self.lock = threading.Lock()
        self.exhausted = 0

    def write(self, *, messages: Any, **record):
        prompt = json.dumps(messages, ensure_ascii=False, default=str)
        row = {'profile': self.profile, 'time': time.time(), 'prompt': messages,
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **record}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            if record.get('status') == 'exhausted':
                self.exhausted += 1


def audit_evidence(audit: ExtractionAudit) -> dict:
    """Bind the completed index prefix while later query calls may append records."""
    from collections import Counter

    raw = audit.path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    counts = dict(Counter(row['status'] for row in rows))
    return {'version': 1, 'path': str(audit.path.resolve()), 'prefix_bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(), 'counts': counts, 'profile': audit.profile}
