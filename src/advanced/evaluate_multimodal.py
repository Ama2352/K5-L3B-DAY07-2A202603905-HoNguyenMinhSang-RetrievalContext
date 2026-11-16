"""Run and freeze a source-only multimodal pipeline before independent advanced test.

The model receives only each question and retrieved source material. Benchmark
answers/evidence remain in the external evaluation branch and are never tool input.
Run ``python -m src.advanced.evaluate_multimodal --help`` for commands.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.advanced.selection_agent import run
from src.advanced.evidence_items import EvidenceItems
from src.advanced.multimodal_tools import MultimodalTools, SCHEMAS, dispatch
from src.common.secret_safety import redact
from src.common.embedding_worker import ProcessEmbedder


MODEL = 'qwen3:4b-instruct'
OPTIONS = {'temperature': 0, 'num_ctx': 2048, 'num_predict': 192, 'num_thread': 8, 'num_batch':128}
PIPELINE = {'model': MODEL, 'model_options': OPTIONS, 'context_budget_chars': 4000,
            'agent_mode':'schema_guided_function_calling_per_question_part', 'max_tool_calls': 2, 'max_page_images': 0, 'timeout_seconds': 300,
            'retrieval_mode':'bm25','excerpt_chars':1100,'candidate_items_per_part':3,
            'input_mode':'all_source_rows_visual_labels_and_paragraphs'}
from src.advanced.selection_agent import INSTRUCTIONS, SCHEMA as ACTIVE_SCHEMAS
BENCHMARKS = {'dev': Path('data/advanced/benchmark-dev.json'),
              'test': Path('data/advanced/benchmark-test.json')}
CODE_FILES = [Path(__file__), Path('src/advanced/multimodal_tools.py'), Path('src/advanced/selection_agent.py'),
              Path('src/advanced/evidence_items.py'),
              Path('src/advanced/vision_index.py'), Path('src/advanced/build_verified_catalog.py'), Path('src/agents/function_agent.py'),
              Path('src/retrieval/fusion.py'), Path('src/common/secret_safety.py'),Path('src/common/embedding_worker.py')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(redact(document), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temporary, path)


def fingerprint(data):
    """Hash exact source/catalog/code/config versions; never inspect test answers."""
    data = Path(data)
    with urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=15) as response:
        models = json.load(response).get('models', [])
    model = next((item for item in models if item['name'] == MODEL), None)
    if not model or not model.get('digest'):
        raise ValueError(f'Install and identify the exact Ollama model before evaluation: {MODEL}')
    return {'pipeline': PIPELINE, 'model_digest': model['digest'],
            'instructions': INSTRUCTIONS, 'schemas': ACTIVE_SCHEMAS,
            'code_sha256': {p.resolve().relative_to(Path.cwd().resolve()).as_posix(): digest(p) for p in CODE_FILES},
            'catalog_sha256': digest(data / 'multimodal_pages.json'),
            'vectors_sha256': digest(data / 'multimodal_vectors.json'),
            'dev_benchmark_sha256': digest(BENCHMARKS['dev']),
            'test_benchmark_sha256': digest(BENCHMARKS['test'])}


def check_catalog(data):
    document = json.loads((Path(data) / 'multimodal_pages.json').read_text(encoding='utf8'))
    pages = document.get('pages', [])
    if sorted(p['page'] for p in pages) != list(range(1, 128)):
        raise ValueError('Catalog must contain each of the 127 physical pages exactly once')
    if any(p.get('status') != 'ok' or p.get('truncated') for p in pages):
        raise ValueError('Resolve catalog errors/truncated descriptions before evaluation')
    for page in pages:
        image = Path(data) / page['page_image']
        if digest(image) != page['image_sha256']:
            raise ValueError(f'Original page image changed: {page["page"]}')
    return document


def image_provider(data, max_pages=4):
    supplied = set()

    def provide(results):
        images = []
        for row in results:
            page = row.get('page')
            if type(page) is not int or page in supplied or len(supplied) >= max_pages:
                continue
            path = Path(data) / 'rendered' / f'page-{page:03d}.png'
            images.append({'page': page, 'path': str(path),
                           'data': base64.b64encode(path.read_bytes()).decode('ascii')})
            supplied.add(page)
        return images
    return provide


def runtime_dispatch(tools, name, arguments):
    """Expose calculator output as visibly derived data, never a fabricated source."""
    output = dispatch(tools, name, arguments)
    if name == 'calculate':
        step = output['steps'][0]
        identifier = hashlib.sha256(arguments['expression'].encode()).hexdigest()[:16]
        output['results'] = [{
            'id': f'derived-calculation-{identifier}', 'source_id': 'derived_arithmetic',
            'source': 'derived:arithmetic (not a source document)', 'page': None,
            'text': f'DERIVED COMPUTATION ONLY: {step["expression"]} = {step["result"]}. '
                    'This verifies arithmetic only, not input accuracy. Cite original page labels for inputs.',
            'metadata': {'evidence_type': 'derived_computation', 'expression': step['expression'],
                         'result': step['result'], 'is_source_evidence': False}}]
    return output


def summarize(document):
    traces = document['traces']
    calls = [call for trace in traces for call in trace.get('tool_calls', [])]
    counts = Counter(call.get('name', 'unknown') for call in calls)
    return {'published_cases': len(document['query_ids']), 'completed_cases': len(traces),
            'status_counts': dict(Counter(t['status'] for t in traces)),
            'system_errors': sum(t['status'] == 'error' for t in traces),
            'cases_with_successful_retrieval': sum(any(c.get('status') == 'ok' and
                c.get('name') in ('search_evidence', 'inspect_page', 'select_evidence') for c in t.get('tool_calls', [])) for t in traces),
            'tool_call_count': len(calls), 'function_call_counts': dict(counts),
            'model_call_count': sum(t.get('model_call_count', 0) for t in traces),
            'elapsed_seconds_total': sum(t.get('elapsed_seconds', 0) for t in traces),
            'service_cost_usd': 0,
            'reviewed_cases': len(document.get('reviews', [])),
            'pass_rate': None, 'pass_rate_note': 'Requires manual source-backed per-case reviews; statuses are not grades.'}


def validate_cases(cases, split):
    if any(c['split'] != split for c in cases):
        raise ValueError('Benchmark contains the wrong split')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate benchmark IDs')
    groups = Counter(c['group'] for c in cases)
    answerable = len(cases) - groups['missing-evidence']
    if split == 'dev':
        if answerable < 15 or groups['missing-evidence'] < 3:
            raise ValueError('Dev requires at least 15 answerable and 3 missing-evidence cases')
    else:
        minima = {'table': 6, 'image': 5, 'chart': 5, 'multi-hop': 4,
                  'text': 4, 'missing-evidence': 6}
        if any(groups[group] < required for group, required in minima.items()):
            raise ValueError('Test group counts do not meet the published advanced requirements')
    if any(not c.get('question') for c in cases):
        raise ValueError('Each case needs a question')
    return ids


def freeze(output_root, data):
    dev_path = Path(output_root) / 'dev.json'
    dev = json.loads(dev_path.read_text(encoding='utf8'))
    signature = fingerprint(data)
    if dev['fingerprint'] != signature:
        raise ValueError('Dev result does not match current source/code/pipeline/benchmarks')
    expected = set(dev['query_ids'])
    if (len(expected) < 18 or len(dev['traces']) != len(expected)
            or {t['query_id'] for t in dev['traces']} != expected):
        raise ValueError('Complete the entire published advanced dev set before freezing')
    reviews = dev.get('reviews', [])
    if (len(reviews) != len(expected) or {r['query_id'] for r in reviews} != expected
            or any(type(r.get('passed')) is not bool or not r.get('reason') for r in reviews)):
        raise ValueError('Every dev case needs a pass/fail review with source-backed reason before freezing')
    selection = {'schema_version': 1, 'frozen': True, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                 'fingerprint': signature, 'dev_result_sha256': digest(dev_path),
                 'selection_rule': 'Source-reviewed full dev results and documented error analysis; fixed multimodal pipeline.',
                 'selected': 'multimodal_bm25_over_structured_source_rows_and_visual_catalog'}
    path = Path(output_root) / 'selection.json'
    if path.exists():
        raise FileExistsError('Selection already frozen; do not silently overwrite test provenance')
    save(path, selection)


def evaluate(args):
    check_catalog(args.data)
    root = args.day6.resolve()
    tools = EvidenceItems(ProcessEmbedder(root), args.data, retrieval_mode=PIPELINE['retrieval_mode'], excerpt_chars=PIPELINE['excerpt_chars'])
    signature = fingerprint(args.data)
    selection = None
    if args.split == 'test':
        selection = json.loads((args.output_root / 'selection.json').read_text(encoding='utf8'))
        if not selection.get('frozen') or selection['fingerprint'] != signature:
            raise ValueError('Freeze matching dev-selected pipeline and test SHA before test evaluation')
        if digest(args.output_root / 'dev.json') != selection['dev_result_sha256']:
            raise ValueError('Frozen dev result changed')
    # Test gold is read only here, after all selection checks, and never passed to tools/run().
    cases = json.loads(BENCHMARKS[args.split].read_text(encoding='utf8'))
    ids = validate_cases(cases, args.split)
    path = args.output_root / f'{args.split}.json'
    document = {'schema_version': 1, 'split': args.split, 'fingerprint': signature,
                'query_ids': ids, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'selection_sha256': digest(args.output_root / 'selection.json') if selection else None,
                'scope': 'All published cases; errors remain in denominator; model receives questions only.',
                'cost_note': 'Local API charge USD 0; excludes equipment and electricity.',
                'traces': [], 'reviews': [], 'summary': {}}
    if path.exists():
        if not args.resume:
            raise FileExistsError('Existing result: use --resume only with unchanged pipeline/questions')
        document = json.loads(path.read_text(encoding='utf8'))
        if document['fingerprint'] != signature or document['query_ids'] != ids:
            raise ValueError('Cannot resume changed pipeline/benchmark')
        recorded = [t['query_id'] for t in document['traces']]
        if len(set(recorded)) != len(recorded) or not set(recorded) <= set(ids):
            raise ValueError('Invalid existing trace IDs')
    for case in cases:
        if any(t['query_id'] == case['id'] for t in document['traces']):
            continue
        trace = run(case['question'], MODEL, tools, max_calls=PIPELINE['max_tool_calls'],
                    budget=PIPELINE['context_budget_chars'], timeout=PIPELINE['timeout_seconds'],
                    schemas=ACTIVE_SCHEMAS, dispatcher=runtime_dispatch,
                    image_provider=image_provider(args.data,PIPELINE['max_page_images']),
                    extra_instructions=INSTRUCTIONS, model_options=OPTIONS)
        trace.update(query_id=case['id'], split=args.split,
                     pipeline='multimodal_bm25_over_structured_source_rows_and_visual_catalog')
        document['traces'].append(trace)
        document['summary'] = summarize(document)
        save(path, document)
        print(case['id'], trace['status'], round(trace['elapsed_seconds'], 1), flush=True)
    document['finished_utc'] = datetime.now(timezone.utc).isoformat()
    document['summary'] = summarize(document)
    save(path, document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6', type=Path, help='Existing Day 6 project and embedding cache')
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--output-root', type=Path, default=Path('results/advanced'))
    parser.add_argument('--split', choices=['dev', 'test'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze', action='store_true', help='Freeze reviewed dev/config/test hashes before test')
    args = parser.parse_args()
    if args.freeze:
        if args.split:
            parser.error('--freeze and --split are separate operations')
        freeze(args.output_root, args.data)
    else:
        if not args.split or not args.day6:
            parser.error('Evaluation requires --split and --day6')
        evaluate(args)


if __name__ == '__main__':
    main()
