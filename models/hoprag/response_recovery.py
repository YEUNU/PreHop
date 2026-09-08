"""Adapter recovery for valid JSON rejected by the native cleaner and bad return arity."""
import json
import re
import threading
import time
from pathlib import Path

PROFILE = 'adapter-json-recovery-v1'


def install(tool, audit_path, attempts=3):
    if attempts < 1:
        raise ValueError('attempts must be positive')
    native_parser = tool.txt2obj
    native_completion = tool.get_chat_completion
    lock = threading.Lock()
    path = Path(audit_path)

    def parser(raw):
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
                text = re.sub(r'\s*```$', '', text)
            try:
                value = json.loads(text)
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                pass
        return native_parser(raw)

    def completion(*args, **kwargs):
        keys = kwargs.get('keys')
        expected = len(keys) + 1 if keys is not None else 2
        for attempt in range(1, attempts + 1):
            result = native_completion(*args, **kwargs)
            valid = isinstance(result, tuple) and len(result) == expected and isinstance(result[-1], list)
            if valid and keys:
                for key, value in zip(keys, result[:-1]):
                    valid = valid and (isinstance(value, str) if key == 'Title' else
                        isinstance(value, list) and all(isinstance(item, str) for item in value))
            row = dict(profile=PROFILE, time=time.time(), attempt=attempt,
                       status='accepted' if valid else 'retry' if attempt < attempts else 'exhausted',
                       keys=keys, result=result)
            with lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
            if valid:
                return result
        raise RuntimeError('HopRAG response exhausted adapter recovery; see '+str(path))

    tool.txt2obj = parser
    tool.get_chat_completion = completion
