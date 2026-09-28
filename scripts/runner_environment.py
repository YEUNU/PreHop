"""Load project environment for command-line runners."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _load_runner_environment() -> None:
    """Load the same project file as shell runners without overriding exports."""
    from dotenv import load_dotenv

    if os.environ.get("RAG_SKIP_PROJECT_ENV") == "true":
        return
    env_path = ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def selected_python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    prefix = Path(sys.prefix).resolve()
    executable = prefix / 'bin/python'
    environment.update(PYTHON_BIN=str(executable), UV_PROJECT_ENVIRONMENT=str(prefix))
    return environment


def safe_environment() -> dict[str, str]:
    """Only approved runtime selectors and canonical connection fields enter the unit."""
    from core.inference_transport import _FORBIDDEN_AMBIENT_PROVIDER_KEYS, preserve_provider_environment
    from core.paper_policy import method_environment_defaults, preserve_method_environment
    from core.strategy_registry import paper_environment_defaults
    current = selected_python_environment()
    preserve_provider_environment(current)
    preserve_method_environment(current)
    allowed = {'RAG_MEASUREMENT_MAX_TARGETS', 'RAG_PREHOP_TRACE', 'RAG_PREHOP_TRACE_DIR', 'RAG_EXECUTION_PROFILE', 'PYTHON_BIN', 'UV_PROJECT_ENVIRONMENT', 'RAG_OFFICIAL_BASELINE_HOME',
        'RAG_INFERENCE_BASE_URL', 'RAG_INFERENCE_API_KEY', 'RAG_GENERATION_MODEL', 'RAG_EMBEDDING_MODEL',
        'RAG_GENERATION_REVISION', 'RAG_EMBEDDING_REVISION', 'NEO4J_URI', 'NEO4J_URL', 'NEO4J_USERNAME', 'NEO4J_USER',
        'NEO4J_PASSWORD', 'NEO4J_DATABASE', 'HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'PATH', 'HOME', 'LITELLM_MODE'}
    allowed.update(_FORBIDDEN_AMBIENT_PROVIDER_KEYS)
    allowed.update(method_environment_defaults())
    allowed.update(paper_environment_defaults())
    env = {key: current[key] for key in allowed if key in current}
    env.update(PYTHONDONTWRITEBYTECODE='1', RAG_SKIP_PROJECT_ENV='true', RAG_PAPER_MODE='true')
    return env
