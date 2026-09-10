"""Download pinned HotpotQA resources; validate the official archive checksum."""
import argparse
import json
import subprocess
from pathlib import Path

import requests

try:
    from scripts.datasets.prepare_hotpotqa import ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_URL, DEV_URL, digest_file
except ModuleNotFoundError:
    from prepare_hotpotqa import ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_URL, DEV_URL, digest_file

MIRROR_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
MIRROR_URL = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{MIRROR_REVISION}/fullwiki/validation-00000-of-00001.parquet"


def download(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_URL.rsplit("/", 1)[1]
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "15",
                        "--continue-at", "-", "--output", str(partial), ARCHIVE_URL], check=True)
        if partial.stat().st_size != ARCHIVE_BYTES or digest_file(partial, "md5") != ARCHIVE_MD5:
            raise ValueError("Official Wikipedia archive checksum mismatch")
        partial.rename(archive)
    if archive.stat().st_size != ARCHIVE_BYTES or digest_file(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("Existing Wikipedia archive checksum mismatch")
    gold = output / "hotpot_dev_fullwiki_v1.json"
    provenance = output / "download_provenance.json"
    if gold.exists():
        if not provenance.exists() or json.loads(provenance.read_text()).get("gold_sha256") != digest_file(gold):
            raise ValueError("Existing dev file lacks matching download provenance")
        return archive, gold
    try:
        response = requests.get(DEV_URL, timeout=(8, 90))
        response.raise_for_status()
        rows = response.json()
        gold.write_bytes(response.content)
        origin = {"gold_url": DEV_URL, "gold_format": "official-json"}
    except requests.RequestException:
        import pyarrow.parquet as pq
        response = requests.get(MIRROR_URL, timeout=(15, 120))
        response.raise_for_status()
        mirror = output / "validation.parquet"
        mirror.write_bytes(response.content)
        rows = pq.read_table(mirror).to_pylist()
        for row in rows:
            row["_id"] = row.pop("id")
            facts, context = row["supporting_facts"], row["context"]
            row["supporting_facts"] = [list(x) for x in zip(facts["title"], facts["sent_id"], strict=True)]
            row["context"] = [list(x) for x in zip(context["title"], context["sentences"], strict=True)]
        gold.write_text(json.dumps(rows, ensure_ascii=False))
        origin = {"gold_url": MIRROR_URL, "gold_format": "author-owned-parquet-converted-to-json",
                  "mirror_revision": MIRROR_REVISION, "mirror_sha256": digest_file(mirror)}
    if len(rows) != 7405:
        raise ValueError("Expected all 7,405 official fullwiki dev rows")
    provenance.write_text(json.dumps({**origin, "gold_sha256": digest_file(gold),
                         "archive_url": ARCHIVE_URL, "archive_md5": ARCHIVE_MD5,
                         "archive_sha256": digest_file(archive)}, indent=2))
    return archive, gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/hotpotqa_raw"))
    args = parser.parse_args()
    print(download(args.output))
