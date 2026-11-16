"""Run a compact, reproducible submission: native tool agents and query-original retrieval."""
import argparse
import copy
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.agents.function_agent import run
from src.retrieval.retrieval_tools import RetrievalTools, TOOL_SCHEMAS, execute_tool
from baseline import hit, pack_context
from src.common.secret_safety import database_url, redact

CONFIGS = {
    'dense_baseline': {'function': 'search_dense', 'top_k': 3, 'budget': 4000,
                       'hypothesis': 'Semantic dense baseline on the Day 6 vector store.'},
    'bm25': {'function': 'search_bm25', 'top_k': 3, 'budget': 4000,
             'hypothesis': 'Lexical ranking recovers exact technical terms missed by dense.'},
    'hybrid': {'function': 'search_hybrid', 'top_k': 5, 'budget': 6000,
               'hypothesis': 'Dense+BM25 RRF and larger context combine lexical and semantic evidence.'},
}
MODEL = 'qwen2.5:1.5b'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    os.replace(temporary, path)


def schema(config):
    result = [copy.deepcopy(s) for s in TOOL_SCHEMAS if s['function']['name'] == config['function']]
    result[0]['function']['parameters']['properties']['top_k'] = {
        'type': 'integer', 'enum': [config['top_k']], 'description': 'Fixed experiment cutoff'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-root', type=Path, default=Path('results/basic'))
    args = parser.parse_args()
    source = Path('data/benchmark.json' if args.split == 'dev' else 'data/benchmark-test.json')
    cases = [c for c in json.loads(source.read_text(encoding='utf8')) if c['split'] == args.split]
    config_names = list(CONFIGS)
    # All published cases are run; gold is never passed into run().
    agent_ids = {c['id'] for c in cases}
    if args.split == 'test':
        selection = json.loads((args.output_root/'selection.json').read_text(encoding='utf8'))
        if not selection.get('frozen') or selection['test_sha256'] != digest(source):
            raise ValueError('Freeze configuration and test before evaluating test')
        if selection['code_sha256'] != code_digest():
            raise ValueError('Agent code changed since selection')
        config_names = list(dict.fromkeys(['dense_baseline', selection['selected']]))
    output = args.output_root/f'{args.split}.json'
    document = {'schema_version': 1, 'split': args.split, 'model': MODEL,
        'benchmark_sha256': digest(source), 'code_sha256': code_digest(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rule': 'Highest dev grounded grade sum (errors=0), then correct abstentions, '
                          'then agent context grade sum, then lower total time.',
        'configs': {name: CONFIGS[name] for name in config_names},
        'agent_evaluation_ids': sorted(agent_ids),
        'agent_scope_note': 'All published dev/test cases, including provider errors. '
                            'Retrieval ranking metrics exclude missing-evidence cases.',
        'retrieval': [], 'agent': [], 'reviews': [], 'summary': {},
        'cost_note': 'Local services: USD 0 API charge; excludes hardware/electricity. '
                     'Times are single runs; retrieval excludes one-time initialization.'}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'{output} exists; use --resume or move it aside')
        document = json.loads(output.read_text(encoding='utf8'))
        if document['code_sha256'] != code_digest() or document['benchmark_sha256'] != digest(source):
            raise ValueError('Cannot resume with changed code or questions')
    tools = RetrievalTools(args.day6, database_url(args.day6))
    for name in config_names:
        config = CONFIGS[name]
        for case in cases:
            if case['answerable'] and not any(r['config'] == name and r['query_id'] == case['id'] for r in document['retrieval']):
                import time
                start = time.perf_counter()
                row = {'config': name, 'query_id': case['id'], 'query_original': case['question'],
                       'function': config['function'], 'arguments': {'query': case['question'], 'top_k': config['top_k']},
                       'tool_call_count': 1, 'filter': None, 'context_budget_chars': config['budget']}
                try:
                    result = execute_tool(tools, row['function'], row['arguments'])
                    ranked = result['results']
                    context, ids = pack_context(ranked, budget=config['budget'])
                    flags = [hit(c, case['gold']) for c in ranked]
                    row.update(status='ok', results=ranked, steps=result['steps'], context=context, context_ids=ids,
                               hit1=bool(flags and flags[0]), hit3=any(flags[:3]),
                               context_hit=any(flag and c['id'] in ids for flag, c in zip(flags, ranked)))
                except Exception as exc:
                    row.update(status='error', error=f'{type(exc).__name__}: {exc}', hit1=False, hit3=False, context_hit=False)
                row['elapsed_seconds'] = time.perf_counter() - start
                document['retrieval'].append(row)
                save(output, document)
            if case['id'] not in agent_ids or any(r['config'] == name and r['query_id'] == case['id'] for r in document['agent']):
                continue
            # Gold and expected answers are used only in the evaluation branch above, never sent to run().
            trace = run(case['question'], MODEL, tools, max_calls=3, budget=config['budget'],
                        schemas=schema(config), extra_instructions=f'Use top_k={config["top_k"]}. '
                        'Answer in at most two concise sentences, including evidence citations.',
                        model_options={'temperature':0,'num_predict':256,'num_ctx':4096})
            trace.update(query_id=case['id'], config=name, split=args.split)
            document['agent'].append(trace)
            save(output, document)
            print(name, case['id'], trace['status'], round(trace['elapsed_seconds'], 1), flush=True)
    document['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save(output, document)


def code_digest():
    files = ['agents/function_agent.py', 'retrieval/retrieval_tools.py', 'retrieval/fusion.py', 'evaluation/evaluate_lab.py', 'agents/run_agent.py', 'evaluation/retrieval_dev.py']
    return hashlib.sha256(b''.join((Path('src') / f).read_bytes() for f in files)).hexdigest()


if __name__ == '__main__':
    main()
