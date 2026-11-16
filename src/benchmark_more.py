"""Compare additional methods on the fixed dev split, never the exposed old test."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from retrieval_tools import RetrievalTools, execute_tool
from baseline import hit, pack_context
from secret_safety import database_url, redact, redact_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--day6', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    tools = RetrievalTools(args.day6, database_url(args.day6))
    questions = [q for q in json.loads(Path('data/benchmark.json').read_text(encoding='utf8'))
                 if q['split'] == 'dev' and q.get('answerable', True)]
    methods = ['search_dense', 'search_tfidf', 'search_bm25', 'search_hybrid', 'search_rerank', 'search_multistage']
    traces = []
    for q in questions:
        for method in methods:
            started = time.perf_counter()
            row = {'query_id': q['id'], 'split': 'dev', 'question': q['question'], 'method': method,
                   'query_original': q['question'], 'filter': None, 'filter_origin': 'No filter',
                   'top_k': 3, 'context_budget_chars': 4000, 'service_cost_usd': 0}
            try:
                result = execute_tool(tools, method, {'query': q['question'], 'top_k': 3})
                ranked = result['results']
                context, ids = pack_context(ranked, budget=4000)
                flags = [hit(c, q['gold']) for c in ranked]
                row.update(status='ok', results=ranked, steps=result['steps'], context=context,
                           context_ids=ids, hit1=bool(flags and flags[0]), hit3=any(flags[:3]),
                           context_hit=any(flag and c['id'] in ids for flag, c in zip(flags, ranked)))
                if method == 'search_multistage':
                    original = result['steps'][0]['results']
                    row['original_query_hit1'] = bool(original and hit(original[0], q['gold']))
                    row['original_query_hit3'] = any(hit(c, q['gold']) for c in original[:3])
            except Exception as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}',
                           hit1=False, hit3=False, context_hit=False)
            row['elapsed_ms'] = (time.perf_counter() - started) * 1000
            traces.append(row)
    summary = {}
    for method in methods:
        rows = [r for r in traces if r['method'] == method]
        summary[method] = {'n': len(rows), 'errors': sum(r['status'] == 'error' for r in rows),
            'hit1': sum(r['hit1'] for r in rows), 'hit3': sum(r['hit3'] for r in rows),
            'context_hit': sum(r['context_hit'] for r in rows),
            'mean_ms': statistics.mean(r['elapsed_ms'] for r in rows), 'service_cost_usd': 0}
        if method == 'search_multistage':
            summary[method]['original_query_hit1'] = sum(r.get('original_query_hit1', False) for r in rows)
            summary[method]['original_query_hit3'] = sum(r.get('original_query_hit3', False) for r in rows)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('traces.json', traces), ('summary.json', summary)]:
        (args.output / name).write_text(json.dumps(redact(value), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
