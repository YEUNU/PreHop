#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
runtime_root=${RAG_OFFICIAL_BASELINE_HOME:-"$repo_root/data/official_baselines"}
mkdir -p "$runtime_root"

install_checkout() {
    strategy=$1
    repository=$2
    revision=$3
    target="$runtime_root/$strategy"
    source="$target/source"
    if [ ! -d "$source/.git" ]; then
        git clone --filter=blob:none --no-checkout "$repository" "$source"
    fi
    git -C "$source" fetch --depth 1 origin "$revision"
    git -C "$source" checkout --detach "$revision"
    actual=$(git -C "$source" rev-parse HEAD)
    [ "$actual" = "$revision" ] || { echo "$strategy revision verification failed" >&2; exit 1; }
    git -C "$source" diff --quiet && git -C "$source" diff --cached --quiet || {
        echo "$strategy official checkout has modified tracked files: $source" >&2
        exit 1
    }
    if [ ! -x "$target/venv/bin/python" ]; then
        uv venv --python 3.10 "$target/venv"
    fi
}

install_checkout browsenet https://github.com/bisect-group/BrowseNet.git ba82eeceb089104de2999d00b744cd02583fe8a4
uv pip install --python "$runtime_root/browsenet/venv/bin/python" -r "$repo_root/scripts/requirements-browsenet.txt"
checkpoint_dir="$runtime_root/browsenet/source/src/indexer/exp"
if [ ! -d "$checkpoint_dir/colbertv2.0" ]; then
    mkdir -p "$checkpoint_dir"
    archive=$(mktemp)
    curl -fL https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -o "$archive"
    tar -xzf "$archive" -C "$checkpoint_dir"
    rm -f "$archive"
fi

install_checkout proprag https://github.com/ReLink-Inc/PropRAG.git 3ec103488abd5589e569ee0fdd6e0c7067e5b783
uv pip install --python "$runtime_root/proprag/venv/bin/python" -r "$repo_root/scripts/requirements-proprag.txt"

echo "Official BrowseNet and PropRAG runtimes are ready under $runtime_root"
