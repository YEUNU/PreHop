#!/usr/bin/env bash
set -euo pipefail

mode=${1:---primary}
case "$mode" in
    --primary|--legacy|--all) ;;
    *) echo "Usage: $0 [--primary|--legacy|--all]" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "$0")/.." && pwd)
runtime_root=${RAG_OFFICIAL_BASELINE_HOME:-"$repo_root/data/official_baselines"}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$runtime_root"
exec 9>"$runtime_root/.runtime.lock"
flock -x 9
declare -A official_repo official_revision
installed_strategies=()
registry_rows=$(python3 "$repo_root/core/strategy_registry.py" --external-tsv) || {
    echo "strategy registry query failed" >&2
    exit 1
}
while IFS=$'\t' read -r name repository revision; do
    official_repo["$name"]=$repository
    official_revision["$name"]=$revision
done <<< "$registry_rows"

runtime_field() {
    python3 - "$repo_root/configs/paper_runtime_requirements.json" "$1" "$2" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
value = payload[sys.argv[2]][sys.argv[3]]
print(value)
PY
}

constraint_path() {
    python3 - "$repo_root" "$1" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads((root / "configs/paper_runtime_requirements.json").read_text(encoding="utf-8"))
requirement = payload[sys.argv[2]]
path = (root / requirement["constraints_file"]).resolve()
if root not in path.parents or not path.is_file():
    raise SystemExit(f"approved constraint file is missing: {path}")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != requirement["constraints_sha256"]:
    raise SystemExit(f"approved constraint digest differs: {path}")
print(path)
PY
}

write_snapshot_manifest() {
    strategy=$1
    subdir=$2
    model=$3
    revision=$4
    SNAPSHOT_DIR="$runtime_root/$strategy/artifacts/$subdir" SNAPSHOT_MODEL="$model" \
        SNAPSHOT_REVISION="$revision" "$runtime_root/$strategy/venv/bin/python" - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ["SNAPSHOT_DIR"]).resolve()
rows = []
for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
    relative = path.relative_to(root)
    if relative.parts[0] == ".cache" or relative.as_posix() == "artifact_manifest.json":
        continue
    rows.append({"path": relative.as_posix(), "size": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
if not rows:
    raise SystemExit(f"snapshot contains no model files: {root}")
tree = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
payload = {"schema_version": 1, "repository_id": os.environ["SNAPSHOT_MODEL"],
           "revision": os.environ["SNAPSHOT_REVISION"], "tree_sha256": tree,
           "file_count": len(rows), "total_bytes": sum(row["size"] for row in rows)}
(root / "artifact_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

install_checkout() {
    strategy=$1
    repository=$2
    revision=$3
    python_version=${4:-3.10}
    target="$runtime_root/$strategy"
    source="$target/source"
    if [ -d "$source/.git" ]; then
        [ -z "$(git -C "$source" status --porcelain --untracked-files=all --ignored)" ] || {
            echo "Refusing to checkout dirty existing upstream source: $strategy" >&2
            exit 1
        }
    else
        git clone --filter=blob:none --no-checkout "$repository" "$source"
    fi
    git -C "$source" fetch --depth 1 origin "$revision"
    git -C "$source" checkout --detach "$revision"
    actual=$(git -C "$source" rev-parse HEAD)
    [ "$actual" = "$revision" ] || { echo "$strategy revision verification failed" >&2; exit 1; }
    [ -z "$(git -C "$source" status --porcelain --untracked-files=all --ignored)" ] || {
        echo "$strategy official checkout is not clean: $source" >&2
        exit 1
    }
    if [ ! -x "$target/venv/bin/python" ]; then
        uv venv --python "$python_version" "$target/venv"
    else
        actual_python=$($target/venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        [ "$actual_python" = "$python_version" ] || {
            echo "$strategy runtime requires Python $python_version, found $actual_python at $target/venv" >&2
            exit 1
        }
    fi
    installed_strategies+=("$strategy")
}

install_source_package() {
    local strategy="$1" constraints="$2"
    shift 2
    python3 "$repo_root/scripts/build_official_package.py" \
        --source "$runtime_root/$strategy/source" \
        --revision "${official_revision[$strategy]}" \
        --artifacts "$runtime_root/$strategy/artifacts" \
        --python "$runtime_root/$strategy/venv/bin/python" \
        --constraint "$constraints" "$@"
}

if [ "$mode" = --legacy ] || [ "$mode" = --all ]; then
    install_checkout browsenet "${official_repo[browsenet]}" "${official_revision[browsenet]}"
    uv pip install --python "$runtime_root/browsenet/venv/bin/python" -r "$repo_root/scripts/requirements-browsenet.txt"
    browsenet_artifact_root="$runtime_root/browsenet/artifacts"
    browsenet_checkpoint="$browsenet_artifact_root/colbertv2.0"
    if [ ! -d "$browsenet_checkpoint" ]; then
        mkdir -p "$browsenet_artifact_root"
        archive=$(mktemp)
        curl -fL https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz -o "$archive"
        tar -xzf "$archive" -C "$browsenet_artifact_root"
        rm -f "$archive"
    fi

    install_checkout proprag "${official_repo[proprag]}" "${official_revision[proprag]}"
    uv pip install --python "$runtime_root/proprag/venv/bin/python" -r "$repo_root/scripts/requirements-proprag.txt"
fi

if [ "$mode" = --primary ] || [ "$mode" = --all ]; then

install_checkout lightrag "${official_repo[lightrag]}" "${official_revision[lightrag]}" "$(runtime_field lightrag python_version)"
lightrag_constraints=$(constraint_path lightrag)
install_source_package lightrag "$lightrag_constraints" --extra "openai==2.8.1"

install_checkout hipporag2 "${official_repo[hipporag2]}" "${official_revision[hipporag2]}" "$(runtime_field hipporag2 python_version)"
hipporag_constraints=$(constraint_path hipporag2)
install_source_package hipporag2 "$hipporag_constraints"

install_checkout gfm_rag "${official_repo[gfm_rag]}" "${official_revision[gfm_rag]}" "$(runtime_field gfm_rag python_version)"
gfm_constraints=$(constraint_path gfm_rag)
install_source_package gfm_rag "$gfm_constraints"
gfm_entity_model=$(runtime_field gfm_rag entity_linker_model)
gfm_entity_revision=$(runtime_field gfm_rag entity_linker_revision)
HF_MODEL="$gfm_entity_model" HF_REVISION="$gfm_entity_revision" HF_TARGET="$runtime_root/gfm_rag/artifacts/entity_linker" "$runtime_root/gfm_rag/venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_MODEL"], revision=os.environ["HF_REVISION"],
                  local_dir=os.environ["HF_TARGET"],
                  ignore_patterns=["model.onnx", "pytorch_model.bin"])
PY
write_snapshot_manifest gfm_rag entity_linker "$gfm_entity_model" "$gfm_entity_revision"
gfm_checkpoint_model=$(runtime_field gfm_rag checkpoint_model)
gfm_checkpoint_revision=$(runtime_field gfm_rag checkpoint_revision)
HF_MODEL="$gfm_checkpoint_model" HF_REVISION="$gfm_checkpoint_revision" HF_TARGET="$runtime_root/gfm_rag/artifacts/checkpoint" "$runtime_root/gfm_rag/venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_MODEL"], revision=os.environ["HF_REVISION"],
                  local_dir=os.environ["HF_TARGET"], allow_patterns=["config.json", "model.pth"])
PY
write_snapshot_manifest gfm_rag checkpoint "$gfm_checkpoint_model" "$gfm_checkpoint_revision"

# GPL-3.0 LinearRAG remains solely in its external checkout and isolated venv;
# no upstream source is copied into this MIT-licensed repository.
install_checkout linear_rag "${official_repo[linear_rag]}" "${official_revision[linear_rag]}" "$(runtime_field linear_rag python_version)"
linear_constraints=$(constraint_path linear_rag)
uv pip install --python "$runtime_root/linear_rag/venv/bin/python" \
    --constraint "$linear_constraints" -r "$runtime_root/linear_rag/source/requirements.txt"
uv pip install --python "$runtime_root/linear_rag/venv/bin/python" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.6.1/en_core_web_trf-3.6.1-py3-none-any.whl"
"$runtime_root/linear_rag/venv/bin/python" -c 'import spacy; spacy.load("en_core_web_trf")'
linear_model_row=$(python3 "$repo_root/core/strategy_registry.py" --local-model-tsv | awk -F '\t' '$1 == "linear_rag" {print $2 "\t" $3}')
IFS=$'\t' read -r linear_model linear_revision <<< "$linear_model_row"
HF_MODEL="$linear_model" HF_REVISION="$linear_revision" HF_TARGET="$runtime_root/linear_rag/artifacts/embedding" "$runtime_root/linear_rag/venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_MODEL"], revision=os.environ["HF_REVISION"],
                  local_dir=os.environ["HF_TARGET"],
                  ignore_patterns=["onnx/*", "openvino/*", "pytorch_model.bin"])
PY
write_snapshot_manifest linear_rag embedding "$linear_model" "$linear_revision"

# Youtu-GraphRAG is licensed for academic/research use only. Running its setup
# acknowledges that upstream restriction; it is not a production dependency.
install_checkout youtu_graphrag "${official_repo[youtu_graphrag]}" "${official_revision[youtu_graphrag]}" "$(runtime_field youtu_graphrag python_version)"
youtu_constraints=$(constraint_path youtu_graphrag)
uv pip install --python "$runtime_root/youtu_graphrag/venv/bin/python" \
    --constraint "$youtu_constraints" -r "$runtime_root/youtu_graphrag/source/requirements.txt"
uv pip install --python "$runtime_root/youtu_graphrag/venv/bin/python" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl"
"$runtime_root/youtu_graphrag/venv/bin/python" -c 'import spacy; spacy.load("en_core_web_lg")'
youtu_model_row=$(python3 "$repo_root/core/strategy_registry.py" --local-model-tsv | awk -F '\t' '$1 == "youtu_graphrag" {print $2 "\t" $3}')
IFS=$'\t' read -r youtu_model youtu_revision <<< "$youtu_model_row"
HF_MODEL="$youtu_model" HF_REVISION="$youtu_revision" HF_TARGET="$runtime_root/youtu_graphrag/artifacts/embedding" "$runtime_root/youtu_graphrag/venv/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["HF_MODEL"], revision=os.environ["HF_REVISION"],
                  local_dir=os.environ["HF_TARGET"],
                  ignore_patterns=["onnx/*", "openvino/*", "pytorch_model.bin", "tf_model.h5", "rust_model.ot"])
PY
write_snapshot_manifest youtu_graphrag embedding "$youtu_model" "$youtu_revision"
fi

# Freeze the complete resolved environment after every install. Preflight
# compares this lock with the installed environment, catching ambient package
# changes that source-revision checks cannot detect.
for strategy in "${installed_strategies[@]}"; do
    python3 "$repo_root/scripts/build_official_package.py" --check-only \
        --source "$runtime_root/$strategy/source" --revision "${official_revision[$strategy]}"
    python_path="$runtime_root/$strategy/venv/bin/python"
    [ -x "$python_path" ] || continue
    lock_path="$runtime_root/$strategy/runtime.freeze.txt"
    temporary_lock="$lock_path.tmp"
    uv pip freeze --python "$python_path" | LC_ALL=C sort > "$temporary_lock"
    mv "$temporary_lock" "$lock_path"
    uv pip check --python "$python_path"
done

echo "Pinned official baseline runtimes are ready under $runtime_root"
