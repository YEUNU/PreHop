"""Create answer-only replay inputs from completed, saved retrieval result rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core.synthesis_replay import SynthesisInput, returned_passage_context
from utils.io import atomic_text_writer
from utils.official_results import dataset_key


def prepare_inputs(results: list[Path], output: Path) -> int:
    """Use every returned passage in its recorded order; do not run any model."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with atomic_text_writer(output) as stream:
        for path in results:
            raw = path.read_bytes()
            result = json.loads(raw)
            dataset = dataset_key(result.get('corpus_tag') or result['dataset'])
            method = result['strategy']
            provenance = {'kind': 'saved_returned_passages', 'source': str(path.resolve()),
                          'source_sha256': hashlib.sha256(raw).hexdigest(),
                          'field': 'details[].retrieved_sources', 'all_returned_passages': True,
                          'structured_graph_context': False}
            for row in result['details']:
                sources = row['retrieved_sources']
                entry = SynthesisInput(dataset, method, row['query_id'], row['query'],
                    returned_passage_context(sources), provenance,
                    {'original_query_id': row.get('original_query_id', row['query_id']),
                     'question_type': row['question_type'], 'returned_passages': len(sources),
                     'included_passages': len(sources)})
                stream.write(json.dumps(entry.as_record(), ensure_ascii=False) + '\n')
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, nargs='+', help='One saved benchmark result per dataset/system, in execution order')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(f'Wrote {prepare_inputs(args.results, args.output)} complete evidence contexts to {args.output}')


if __name__ == '__main__':
    main()
