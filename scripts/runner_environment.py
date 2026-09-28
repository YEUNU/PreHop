"""Load project environment for command-line runners."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load_runner_environment() -> None:
    """Load .env below exports, then resolve defaults and the selected profile."""
    from dotenv import load_dotenv

    from core.execution_profile import resolved_execution_environment

    env_path = ROOT / ".env"
    if os.environ.get("RAG_SKIP_PROJECT_ENV") != "true" and env_path.is_file():
        load_dotenv(env_path, override=False)
    os.environ.update(resolved_execution_environment())


def export_runner_environment() -> None:
    """Emit shell-quoted changes; optional arguments select strategy/dataset/run."""
    import shlex

    from core.inference_transport import InferenceTransport, validate_public_inference_environment
    from core.paper_policy import configure_target_environment, preserve_method_environment

    before = os.environ.copy()
    _load_runner_environment()
    try:
        validate_public_inference_environment(os.environ)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    preserve_method_environment()
    os.environ['RAG_INFERENCE_BASE_URL'] = InferenceTransport.resolve('core').generation_base_url
    if sys.argv[1:]:
        configure_target_environment(*sys.argv[1:])
    for name, value in os.environ.items():
        if before.get(name) != value:
            print(f'export {name}={shlex.quote(value)}')


def selected_python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    prefix = Path(sys.prefix).resolve()
    executable = prefix / 'bin/python'
    environment.update(PYTHON_BIN=str(executable), UV_PROJECT_ENVIRONMENT=str(prefix))
    return environment


def safe_environment() -> dict[str, str]:
    """Only approved runtime selectors and canonical connection fields enter the unit."""
    from core.inference_transport import (
        _FORBIDDEN_AMBIENT_PROVIDER_KEYS,
        inference_environment_keys,
        preserve_provider_environment,
    )
    from core.paper_policy import method_environment_defaults, preserve_method_environment
    current = selected_python_environment()
    preserve_provider_environment(current)
    preserve_method_environment(current)
    allowed = {'RAG_MEASUREMENT_MAX_TARGETS', 'RAG_PREHOP_TRACE', 'RAG_PREHOP_TRACE_DIR', 'RAG_EXECUTION_PROFILE', 'PYTHON_BIN', 'UV_PROJECT_ENVIRONMENT', 'RAG_OFFICIAL_BASELINE_HOME',
        'NEO4J_URI', 'NEO4J_URL', 'NEO4J_USERNAME', 'NEO4J_USER',
        'NEO4J_PASSWORD', 'NEO4J_DATABASE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'PATH', 'HOME', 'LITELLM_MODE'}
    allowed.update(_FORBIDDEN_AMBIENT_PROVIDER_KEYS)
    allowed.update(method_environment_defaults())
    allowed.update(inference_environment_keys())
    env = {key: current[key] for key in allowed if key in current}
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true', RAG_PAPER_MODE='true')
    return env
