"""Adapter recovery for valid JSON rejected by the native cleaner without changing native completion results."""
import json
import re
import threading
import time
from pathlib import Path

PROFILE = 'adapter-json-recovery-v2'


def install(tool, audit_path):
    native_parser = tool.txt2obj
    native_completion = tool.get_chat_completion
    lock = threading.Lock()
    path = Path(audit_path)

    def parser(raw):
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s*```$', '', text)
            try:
                value = json.loads(text)
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                pass
        return native_parser(raw)

    def completion(*args, **kwargs):
        keys = kwargs.get('keys', args[4] if len(args) > 4 else None)
        result = native_completion(*args, **kwargs)
        row = {'profile': PROFILE, 'time': time.time(), 'status': 'observed',
               'keys': keys, 'result': result}
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
        return result

    tool.txt2obj = parser
    tool.get_chat_completion = completion
