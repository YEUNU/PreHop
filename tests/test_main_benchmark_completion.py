from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["benchmark", "benchmark_all"])
@pytest.mark.parametrize("fails", [False, True])
async def test_benchmark_completion_and_failure_cleanup(monkeypatch, tmp_path, mode, fails):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    import main as entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["main.py", "--mode", mode])
    monkeypatch.setattr(entry.RAGConfig, "PREHOP_ABLATION_PROFILE", "")
    monkeypatch.setattr(entry.RAGConfig, "validate", lambda: None)
    monkeypatch.setattr(entry, "_ensure_run_id", lambda: "test-run")
    monkeypatch.setenv("RAG_BENCHMARK_TIMESTAMP", "test-run")
    monkeypatch.setattr(entry, "PRIMARY_STRATEGIES", ["naive", "prehop"])
    benchmark = AsyncMock(side_effect=[ValueError("query failed"), None] if fails else None)
    monkeypatch.setattr(entry, "run_benchmark_multi_seed", benchmark)
    neo_close, llm_close = AsyncMock(), AsyncMock()
    monkeypatch.setattr(entry.Neo4jService, "global_close", neo_close)
    monkeypatch.setattr(entry.VLLMClient, "global_close", llm_close)

    if fails:
        expected = RuntimeError if mode == "benchmark_all" else ValueError
        with pytest.raises(expected, match="query failed"):
            await entry.main()
    else:
        await entry.main()

    assert benchmark.await_count == (2 if mode == "benchmark_all" else 1)
    neo_close.assert_awaited_once()
    llm_close.assert_awaited_once()
