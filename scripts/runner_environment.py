"""Load project environment for command-line runners."""
import os
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
