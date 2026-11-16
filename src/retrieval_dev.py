"""Compare dense, TF-IDF and reciprocal-rank fusion on answerable dev queries."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from baseline import TfidfIndex, hit, pack_context
from run_agent import DenseRetriever
from secret_safety import database_url, redact, redact_text


def rrf(dense, lexical, limit=3, k=60):
    scores, records = {}, {}
    for ranking in (dense, lexical):
        for rank, chunk in enumerate(ranking, 1):
            key = chunk['id']
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            records[key] = chunk
    return [{**records[key], 'score': scores[key]} for key in
            sorted(scores, key=lambda key: (-scores[key], key))[:limit]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    args = parser.parse_args()
    if args.split == 'test' and not Path('results/checkpoint3/selection.json').is_file():
        parser.error('Freeze the dev selection before running test')
    url = database_url(args.day6)
    chunks = json.loads(Path('data/day6_chunks.json').read_text(encoding='utf-8'))
    questions = [q for q in json.loads(Path('data/benchmark.json').read_text(encoding='utf-8'))
                 if q['split'] == args.split and q.get('answerable', True)]
    dense_index = DenseRetriever(args.day6.resolve(), url)
    lexical_index = TfidfIndex(chunks)
    traces = []
    for q in questions:
        start = time.perf_counter()
        dense = dense_index.search(q['question'], filters=None, top_k=20)
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lexical = lexical_index.search(q['question'], top_k=20)
        lexical_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        fused = rrf(dense, lexical, limit=5)
        fusion_ms = (time.perf_counter() - start) * 1000
        for name, results, budget, elapsed in [('dense', dense[:3], 4000, dense_ms), ('tfidf', lexical[:3], 4000, lexical_ms),
                                       ('hybrid_rrf', fused[:3], 4000, dense_ms + lexical_ms + fusion_ms),
                                       ('hybrid_top5', fused, 6000, dense_ms + lexical_ms + fusion_ms)]:
            context, ids = pack_context(results, budget=budget)
            flags = [hit(c, q['gold']) for c in results]
            traces.append({'query_id': q['id'], 'question': q['question'], 'method': name,
                           'results': results, 'context': context, 'context_ids': ids,
                           'budget_chars': budget, 'hit1': bool(flags and flags[0]), 'hit3': any(flags[:3]),
                           'context_hit': any(h and c['id'] in ids for h, c in zip(flags, results)),
                           'elapsed_ms': elapsed})
    summary = {}
    for name in ('dense', 'tfidf', 'hybrid_rrf', 'hybrid_top5'):
        rows = [x for x in traces if x['method'] == name]
        summary[name] = {'n': len(rows), 'hit1': sum(x['hit1'] for x in rows),
                         'hit3': sum(x['hit3'] for x in rows),
                         'context_hit': sum(x['context_hit'] for x in rows),
                         'mean_ms': sum(x['elapsed_ms'] for x in rows) / len(rows),
                         'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [('traces.json', traces), ('summary.json', summary)]:
        (args.output / name).write_text(json.dumps(redact(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
