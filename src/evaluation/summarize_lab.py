"""Aggregate source-reviewed labels; never invent labels or use a model judge."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from src.evaluation.evaluate_lab import CONFIGS, code_digest, digest, save


def summarize(document):
    rows = {}
    reviews = {(r['config'], r['query_id']): r for r in document['reviews']}
    for name in document['configs']:
        retrieval = [r for r in document['retrieval'] if r['config'] == name]
        agents = [r for r in document['agent'] if r['config'] == name]
        labels = []
        for trace in agents:
            key = (name, trace['query_id'])
            if key not in reviews:
                raise ValueError(f'Missing source review {key}')
            labels.append(reviews[key])
        answerable = [r for r in labels if r['answerable']]
        missing = [r for r in labels if not r['answerable']]
        successful = [r for r in agents if r['status'] != 'error']
        citation_labels = [c for r in labels for c in r['citation_reviews']]
        counts = Counter(r['grounded_grade'] for r in answerable)
        fraction = lambda n, N: {'n': n, 'N': N, 'percent': 100*n/N if N else None}
        rows[name] = {
            'retrieval': {metric: fraction(sum(bool(r[metric]) for r in retrieval), len(retrieval))
                          for metric in ['hit1', 'hit3', 'context_hit']},
            'retrieval_errors': sum(r['status'] == 'error' for r in retrieval),
            'retrieval_mean_ms': 1000*sum(r['elapsed_seconds'] for r in retrieval)/len(retrieval),
            'agent_cases': len(agents), 'agent_errors': len(agents)-len(successful),
            'tool_use': fraction(sum(any(c['status'] == 'ok' for c in t['tool_calls']) for t in successful),len(successful)),
            'tool_call_distribution': dict(Counter(t['tool_call_count'] for t in agents)),
            'grounded_grades_0_1_2': [counts[i] for i in range(3)],
            'grounded_sum': sum(r['grounded_grade'] for r in answerable),
            'grounded_mean_out_of_2': sum(r['grounded_grade'] for r in answerable)/len(answerable),
            'context_grades_0_1_2': [sum(r['context_grade'] == i for r in answerable) for i in range(3)],
            'agent_context_evidence': fraction(sum(r['context_grade'] == 2 for r in answerable),len(answerable)),
            'context_grade_sum': sum(r['context_grade'] for r in answerable),
            'citation_validity': fraction(sum(c['valid'] for c in citation_labels),len(citation_labels)),
            'answers_with_citation': fraction(sum(bool(t['citations']) for t in agents),len(agents)),
            'abstention': fraction(sum(r['abstention_correct'] for r in missing),len(missing)),
            'false_abstention': fraction(sum(r['false_abstention'] for r in answerable),len(answerable)),
            'agent_total_seconds': sum(t['elapsed_seconds'] for t in agents),
            'agent_mean_seconds': sum(t['elapsed_seconds'] for t in agents)/len(agents),
            'service_cost_usd': 0,
        }
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['dev','test'], required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--output-root', type=Path, default=Path('results/basic'))
    args = p.parse_args()
    path = args.output_root/f'{args.split}.json'
    doc = json.loads(path.read_text(encoding='utf8'))
    doc['reviews'] = json.loads(args.reviews.read_text(encoding='utf8'))
    doc['review_method'] = 'AI-assisted reading of original source and context; not independently adjudicated.'
    doc['summary'] = summarize(doc)
    save(path, doc)
    if args.freeze:
        if args.split != 'dev' or len(doc['agent']) != len(doc['configs'])*len(doc['agent_evaluation_ids']):
            raise ValueError('Complete all planned dev cases before freezing')
        selected = max(doc['summary'], key=lambda k: (doc['summary'][k]['grounded_sum'],
            doc['summary'][k]['abstention']['n'], doc['summary'][k]['context_grade_sum'],
            -doc['summary'][k]['agent_total_seconds']))
        save(args.output_root/'selection.json', {'frozen': True, 'selected': selected, 'configs': CONFIGS,
            'rule': doc['selection_rule'], 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
            'dev_sha256': digest(path), 'test_sha256': digest('data/benchmark-test.json'),
            'code_sha256': code_digest(), 'dev_summary': doc['summary'],
            'note': 'Test gold kept separate from implementation until dev selection was frozen.'})
        print('Frozen selection:', selected)
    print(json.dumps(doc['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
