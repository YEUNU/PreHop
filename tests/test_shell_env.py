"""Shell entrypoint contracts affecting experiment reproducibility."""

import subprocess
from pathlib import Path


def test_project_env_preserves_exported_overrides(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_GRAPH_HOP_DEPTH=1\nRAG_HYPO_CHANNEL_VARIANT=full\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    command = (
        f". {project_root / 'scripts/lib.sh'}; "
        f"load_project_env {env_file}; "
        'printf "%s %s" "$RAG_GRAPH_HOP_DEPTH" "$RAG_HYPO_CHANNEL_VARIANT"'
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "RAG_GRAPH_HOP_DEPTH": "0",
        "RAG_HYPO_CHANNEL_VARIANT": "single_combined",
    }

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout == "0 single_combined"
