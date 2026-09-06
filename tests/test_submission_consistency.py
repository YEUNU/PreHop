from scripts.verify_submission_consistency import _semantic_model_config_sha256


def _payload(*, revision: str, namespace: str, top_k: int = 12):
    return {
        "query_provenance": {"revision": revision, "source_tree_sha256": revision},
        "models": {
            "default": "generation-model",
            "generation_revision": "generation-v1",
            "embedding": "embedding-model",
            "embedding_revision": "embedding-v1",
            "llm_seed": 42,
        },
        "index_provenance": {
            "policy": {
                "index_namespace": namespace,
                "embedding_model": "embedding-model",
                "default_top_k": top_k,
            }
        },
        "ablation": {"q_minus": True, "q_plus": True},
    }


def test_semantic_config_ignores_commit_and_run_namespace():
    first = _payload(revision="commit-a", namespace="dataset_run_a")
    second = _payload(revision="commit-b", namespace="dataset_run_b")

    assert _semantic_model_config_sha256(first) == _semantic_model_config_sha256(second)


def test_semantic_config_detects_model_setting_change():
    first = _payload(revision="commit-a", namespace="dataset_run_a", top_k=12)
    second = _payload(revision="commit-a", namespace="dataset_run_a", top_k=8)

    assert _semantic_model_config_sha256(first) != _semantic_model_config_sha256(second)
