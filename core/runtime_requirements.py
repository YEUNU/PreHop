"""Machine-readable pinned runtime requirements for paper strategies."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

PATH = Path(__file__).resolve().parents[1] / "configs/paper_runtime_requirements.json"


def method_main_python(strategy: str, default: str | None = None) -> str:
    """Select the prepared coordinator runtime without resolving venv symlinks."""
    if strategy == "hoprag":
        from models.hoprag.native_runtime import RUNTIME_HOME
        executable = RUNTIME_HOME / "main-env/bin/python"
        return str(executable.absolute())
    return os.path.abspath(default or sys.executable)


def ensure_method_runtime(strategy: str) -> None:
    """Re-enter a model-specific CLI in its pinned interpreter before execution."""
    if strategy != "hoprag":
        return
    executable = method_main_python(strategy)
    prefix = Path(executable).parent.parent
    if Path(sys.prefix).resolve() == prefix.resolve():
        return
    environment = os.environ.copy()
    environment.update(PYTHON_BIN=executable, UV_PROJECT_ENVIRONMENT=str(prefix), PYTHONUNBUFFERED="1")
    os.execve(executable, [executable, *sys.argv], environment)


def load_runtime_requirements() -> dict[str, Any]:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    return payload


def runtime_requirement(strategy: str) -> dict[str, Any]:
    requirement = load_runtime_requirements().get(strategy)
    return requirement


def runtime_identity(strategy: str) -> dict[str, Any]:
    """Return current non-secret lock/constraint byte identities."""
    root = PATH.parents[1]
    if strategy == "hoprag":
        executable = method_main_python(strategy)
        prefix = Path(executable).parent.parent
        if Path(sys.prefix).resolve() != prefix.resolve():
            environment = os.environ.copy()
            environment.update(PYTHON_BIN=executable, UV_PROJECT_ENVIRONMENT=str(prefix))
            raw = subprocess.check_output([executable, "-c",
                "import json; from core.runtime_requirements import runtime_identity; print(json.dumps(runtime_identity('hoprag')))"],
                cwd=root, env=environment, text=True)
            return json.loads(raw)
    requirement = load_runtime_requirements().get(strategy, {})
    constraints = requirement.get("constraints_file") if isinstance(requirement, dict) else None
    constraints_path = (root / constraints).resolve() if isinstance(constraints, str) else None
    from core.strategy_registry import get_strategy

    if get_strategy(strategy).external:
        from models.official_baseline_runtime import official_root

        freeze_path = official_root(strategy).parent / "runtime.freeze.txt"
    else:
        freeze_path = root / "uv.lock"

    def identity(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        return {"path": label, "sha256": digest, "size": path.stat().st_size}

    installed = sorted((str(dist.metadata.get("Name", "")).lower(), dist.version)
                       for dist in metadata.distributions())
    main_runtime = {
        "prefix": str(Path(sys.prefix).absolute()),
        "python": str(Path(sys.executable).absolute()),
        "installed_sha256": hashlib.sha256(json.dumps(installed, separators=(",", ":")).encode()).hexdigest(),
    }
    if strategy == 'hoprag':
        main_runtime['pos_runtime_freeze'] = identity(root / 'data/runtime_envs/hoprag-paper-20260908/pos-env.freeze.txt')
        main_runtime['main_runtime_freeze'] = identity(root / 'data/runtime_envs/hoprag-paper-20260908/main-env.freeze.txt')
        main_runtime['upstream_revision'] = get_strategy(strategy).revision
    return {
        "main_runtime": main_runtime,
        "runtime_freeze": identity(freeze_path),
        "constraints": identity(constraints_path),
        "declared_constraints_sha256": (
            requirement.get("constraints_sha256") if isinstance(requirement, dict) else None
        ),
    }
