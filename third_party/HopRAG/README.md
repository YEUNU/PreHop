# HopRAG: Multi-hop Reasoning for Logic-Aware Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official repository for **HopRAG: Multi-hop Reasoning for Logic-Aware Retrieval-Augmented Generation**, accepted to ACL Findings 2025.

HopRAG is a novel Retrieval-Augmented Generation (RAG) framework that leverages graph databases to enhance multi-hop reasoning. Instead of treating documents as a flat collection, HopRAG models them as a graph of interconnected text chunks (nodes) within a **Neo4j** database. This structure allows for more sophisticated, logic-aware retrieval paths, enabling Large Language Models (LLMs) to answer complex questions that require synthesizing information from multiple sources.

We provide demonstration datasets from **HotpotQA** and **MuSiQue** to get you started quickly.

-----

## 🚀 Getting Started

Follow these steps to set up the HopRAG environment and prepare for your first run.

### Prerequisites

  * Python `3.10.10` or later
  * **Neo4j Community Edition** `5.26.0` installed and running locally.

### Installation and Configuration

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/LIU-Hao-2002/HopRAG.git
    cd HopRAG
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the Environment (`config.py`):**
    Before running any scripts, you must update `config.py` with your local setup details.

      * **Neo4j Connection:** Set your database credentials.

          * `neo4j_url`
          * `neo4j_user`
          * `neo4j_password`
          * `neo4j_dbname`

      * **LLM API:** Provide your API endpoint and key for generation, either openai api or local vllm api.

          * `personal_base`
          * `personal_key`
          * `default_gpt_model`
          * `local_base`
          * `local_key`
          * `local_model_name`

      * **Embedding Model:** In this vendored integration the model is served by
        the project's required external embedding endpoint and must be the same
        for graph construction and retrieval.

          * `embed_model`
          * `embed_model_dict`
          * `embed_dim`

      * **Generation:** `models/hoprag/official_indexer.py` and
        `models/hoprag/hoprag_adapter.py` inject the configured external
        endpoint. Local model and reranker loaders are not part of this fork.

-----

## ⚙️ Usage: A Step-by-Step Guide

Follow this pipeline to build the graph, run retrieval, and generate answers.

### Step 1: Prepare the Dataset

This project prepares its three supported corpora with
`scripts/datasets/prepare_multihoprag.py` and
`scripts/datasets/prepare_musique.py`. The unused upstream standalone preprocessor was
removed so there is one corpus contract.

`models/hoprag/official_indexer.py` stages that shared corpus contract into the
upstream problem-group format and installs all runtime configuration before
`HopBuilder` is imported.

### Step 2: Build Graph Nodes

This step chunks the documents from your doc pool and creates a node for each chunk in the Neo4j database. Run the `main_nodes` function in `HopBuilder.py`.

**Key Parameters:**

  * `docs_dir`: Path to the doc pool directory created in Step 1 (e.g., `quickstart_dataset/hotpot_example_docs`).
  * `cache_dir`: A directory to log progress. This allows the script to be resumed after an interruption.
  * `node_name`: A unique name (type) for your nodes in Neo4j (e.g., `hotpot_bgeen_qwen1b5`). Set this in `config.py`.

We recommend the **separate offline-online mode** for faster and more stable node creation.

#### Mode 1: Separate (Recommended)

1.  **Generate nodes offline:** This step processes the documents and saves the node data locally without connecting to Neo4j.
    ```python
    # In HopBuilder.py
    main_nodes(cache_dir='quickstart_dataset/cache_hotpot_offline',
               docs_dir="quickstart_dataset/hotpot_example_docs",
               label=node_name)
    ```
2.  **Push nodes to Neo4j:** This step uploads the locally cached nodes to your online database.
    ```python
    # In HopBuilder.py
    main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online',
               docs_dir="quickstart_dataset/hotpot_example_docs",
               label=node_name,
               original_cache_dir='quickstart_dataset/cache_hotpot_offline')
    ```

#### Mode 2: Hybrid (Alternative)

This mode processes and uploads nodes in a single step.

```python
# In HopBuilder.py
main_nodes(cache_dir='quickstart_dataset/cache_hotpot_online',
           docs_dir="quickstart_dataset/hotpot_example_docs",
           label=node_name,
           offline=False)
```

### Step 3: Build Edges and Index

Next, connect the nodes with edges and create the vector and keyword indices needed for efficient retrieval. Run the `main_edges_index` function in `HopBuilder.py`. Before running `HopBuilder.py`, please carefully examine the variables in `config.py`, especially `query_generator_model`, `embed_model`, `dataset_name`（`dataset_name` must contain only one of `hotpot`,`musique` or `wiki` to clearly specify which dataset）, `node_name`, `node_dense_index_name` etc.


  * **Specify Index Names:** Before running, define your index names in `config.py`:
      * `node_dense_index_name`
      * `edge_dense_index_name`
      * `node_sparse_index_name`
      * `edge_sparse_index_name`
  * **Run the script:** The `main_edges_index` function uses dataset-specific logic (e.g., `create_edges_hotpot` or `create_edges_musique`) to create edges based on the different data format.

After this step, your graph is fully built and indexed, ready for retrieval\!

### Step 4: Test Retrieval (Optional)

To verify that the graph and retrieval functions are working correctly, you can run a standalone search using the `search_docs` function in `HopRetriever.py`. This is a great way to debug or experiment with different retrieval hyperparameters, e.g. `max_hop`, `topk`, `traversal`, or `node_dense_index_name`/`edge_dense_index_name`/`node_sparse_index_name`/`edge_sparse_index_name` (the specific index names to retrieve from). HopRAG provides a lot of traversal strategies: `bfs_node`, `bfs_hop2` and so on. Feel free to test them here.

### Step 5: Retrieval-Augmented Generation

This repository calls `HopRetriever.search_docs` through
`models/hoprag/hoprag_adapter.py`, then applies the same external synthesis
prompt used by the in-repo comparison methods. The unused upstream standalone
generator and its optional reranking branch were removed.

### Step 6: Evaluation

The output files produced in the previous step are ready for evaluation. Use the corresponding official evaluation tools for your benchmark (e.g., the HotpotQA evaluation suite) to measure performance.

-----

## 📜 Citing HopRAG

If you find our work useful in your research, please cite our paper:

```bibtex
@article{liu2025hoprag,
  title={{HopRAG}: Multi-hop reasoning for logic-aware retrieval-augmented generation},
  author={Liu, Hao and Wang, Zhengren and Chen, Xi and Li, Zhiyu and Xiong, Feiyu and Yu, Qinhan and Zhang, Wentao},
  journal={arXiv preprint arXiv:2502.12442},
  year={2025}
}
```

Thank you for your interest in HopRAG\!
