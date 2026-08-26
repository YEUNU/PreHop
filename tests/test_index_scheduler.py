import asyncio

import pytest

from cli.index import _collect_index_capacity, _reap_bounded_tasks, _submit_bounded_task


@pytest.mark.asyncio
async def test_bounded_scheduler_reuses_completed_slot_without_batch_barrier():
    slow_release = asyncio.Event()
    quick_finished = asyncio.Event()
    third_started = asyncio.Event()
    pending: dict[asyncio.Task[None], str] = {}

    async def slow_work():
        await slow_release.wait()

    async def quick_work():
        quick_finished.set()

    async def third_work():
        third_started.set()

    await _submit_bounded_task(pending, 2, "slow.txt", slow_work)
    await _submit_bounded_task(pending, 2, "quick.txt", quick_work)
    await quick_finished.wait()

    errors = await _submit_bounded_task(pending, 2, "third.txt", third_work)
    await third_started.wait()

    assert errors == []
    assert len(pending) == 2
    assert any(filename == "slow.txt" for filename in pending.values())

    slow_release.set()
    assert await _reap_bounded_tasks(pending, wait_for_one=False) == []
    assert pending == {}


@pytest.mark.asyncio
async def test_bounded_scheduler_reports_task_identity_on_failure():
    pending: dict[asyncio.Task[None], str] = {}

    async def fail():
        raise RuntimeError("broken")

    await _submit_bounded_task(pending, 1, "broken.txt", fail)
    errors = await _reap_bounded_tasks(pending, wait_for_one=False)

    assert len(errors) == 1
    assert errors[0][0] == "broken.txt"
    assert isinstance(errors[0][1], RuntimeError)
    assert pending == {}


@pytest.mark.asyncio
async def test_prehop_capacity_uses_fixed_logical_payload_formula():
    class FakeNeo4j:
        def __init__(self):
            self.queries = []
            self.rows = [
                {"floats": 10, "chars": 20, "list_items": 0, "records": 2},
                {"floats": 0, "chars": 5, "list_items": 3, "records": 4},
            ]

        async def execute_query(self, query):
            self.queries.append(query)
            return [self.rows[len(self.queries) - 1]]

    neo4j = FakeNeo4j()
    capacity = await _collect_index_capacity("prehop", "mu-sique", neo4j)

    assert capacity["measurement"] == "estimated_logical_property_payload"
    assert capacity["bytes"] == 10 * 8 + 25 + 3 * 8 + 6 * 8
    assert capacity["definition_version"] == 1
    assert all("PR_mu_sique" in query for query in neo4j.queries)


@pytest.mark.asyncio
async def test_ms_graphrag_capacity_excludes_nonretrieval_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "ms_graphrag_output" / "musique"
    (root / "artifacts").mkdir(parents=True)
    (root / "_cache").mkdir()
    (root / "_logs").mkdir()
    (root / "_input").mkdir()
    (root / "artifacts" / "retrieval.parquet").write_bytes(b"12345")
    (root / "root.bin").write_bytes(b"123")
    (root / "_cache" / "cache.bin").write_bytes(b"x" * 100)
    (root / "_logs" / "run.log").write_bytes(b"x" * 100)
    (root / "_input" / "docs.json").write_bytes(b"x" * 100)

    capacity = await _collect_index_capacity("ms_graphrag", "musique")

    assert capacity["measurement"] == "physical_retrieval_artifact_size"
    assert capacity["bytes"] == 8
    assert capacity["excluded_directories"] == ["_cache", "_input", "_logs"]
