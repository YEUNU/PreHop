"""Read-only comparison against locally pinned upstream parameter definitions."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import yaml

from core.strategy_registry import PRIMARY_STRATEGIES, get_strategy


def dataclass_defaults(path: Path, name: str) -> dict:
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    values = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        value = node.value
        if isinstance(value, ast.Call):
            value = next((k.value for k in value.keywords if k.arg == 'default'), None)
        try:
            values[node.target.id] = ast.literal_eval(value)
        except (TypeError, ValueError):
            pass
    return values


def audit(home: Path) -> dict:
    from graphrag_llm.config.model_config import ModelConfig

    checks = []
    def check(strategy, field, native, actual, source):
        checks.append({'strategy': strategy, 'field': field, 'native': native, 'adapter': actual,
                       'matched': native == actual, 'source': str(source)})
    def policy(strategy):
        return dict(get_strategy(strategy).paper_index_policy)
    assert ModelConfig.model_fields['call_args'].default_factory() == {}
    for name in ('extract_max_tokens', 'query_max_tokens', 'report_max_tokens'):
        check('ms_graphrag', name, None, policy('ms_graphrag')[name], 'graphrag_llm.config.ModelConfig.call_args={}')
    path = home/'linear_rag/source/src/config.py'
    defaults = dataclass_defaults(path, 'LinearRAGConfig')
    for field, registered in [('retrieval_top_k','retrieval_top_k'),('spacy_model','spacy_model'),('use_vectorized_retrieval','vectorized_retrieval')]:
        check('linear_rag', field, defaults[field], policy('linear_rag')[registered], path)
    path = home/'youtu_graphrag/source/config/base_config.yaml'
    defaults = yaml.safe_load(path.read_text())
    for field, native in [('query_mode',defaults['triggers']['mode']),('construction_mode',defaults['construction']['mode']),
                          ('retrieval_top_k',defaults['retrieval']['top_k']),('retrieval_top_k_filter',defaults['retrieval']['top_k_filter'])]:
        check('youtu_graphrag', field, native, policy('youtu_graphrag')[field], path)
    path = home/'gfm_rag/source/gfmrag/workflow/config/gfm_rag/qa_inference.yaml'
    defaults = yaml.safe_load(path.read_text())
    check('gfm_rag','retrieval_top_k',defaults['test']['top_k'],policy('gfm_rag')['retrieval_top_k'],path)
    path = home/'lightrag/source/lightrag/constants.py'
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, AttributeError, TypeError):
                pass
    for field, constant in [('retrieval_top_k','DEFAULT_TOP_K'),('chunk_top_k','DEFAULT_CHUNK_TOP_K')]:
        check('lightrag',field,constants[constant],policy('lightrag')[field],path)
    return {'status':'matched' if all(row['matched'] for row in checks) else 'mismatch',
            'strategies':list(PRIMARY_STRATEGIES),'checks':checks,
            'scope':'Selected method-defining defaults; not a claim that every provider setting is upstream-identical.',
            'owned_methods':['prehop','naive'],
            'declared_differences':[
                'Shared Gemma generation, remote Qwen embeddings where applicable, seed and execution concurrency are controlled experiment settings.',
                'GFM strict response schemas and Youtu structured extraction are adapter format controls; they are not upstream-default wire requests.',
                'Youtu parallel schema evolution uses an adapter lock and may depend on scheduling; original serial equivalence is not claimed.',
                'GFM uses the native single-pass qa.py workflow (top_k=5); the optional IRCOT workflow (top_k=10, max_steps=2) is a different variant.',
                'Native posthoc Youtu LLM evaluation is replaced by the common benchmark metrics after answer generation.',
                'Omitted MS output caps leave the effective server limit provider-controlled; omission is not unlimited output.'
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.home.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(result['checks'])}))
    return 0 if result['status']=='matched' else 1


if __name__ == '__main__':
    raise SystemExit(main())
