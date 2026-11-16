"""Build a human-readable dev review packet; never assign quality labels."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traces', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cases = {q['id']: q for q in json.loads(Path('data/benchmark.json').read_text(encoding='utf8'))
             if q['split'] == 'dev'}
    sections = ['# Dev agent review packet', '', 'Fill grades and citation support by reading the source; '
                'retrieved ID membership alone does not establish support.', '']
    traces = (json.loads(args.traces.read_text(encoding='utf8')) if args.traces.is_file()
              else [json.loads(path.read_text(encoding='utf8')) for path in sorted(args.traces.glob('*.json'))])
    for trace in traces:
        case = cases[trace['query_id']]
        sections.extend([f"## {trace['method']} / {case['id']}", '',
                         f"Question: {case['question']}", '',
                         f"Expected: {case.get('expected_answer', 'Search and abstain')}", '',
                         f"Status: {trace['status']}; Answer: {trace.get('answer', '')}", '',
                         f"Gold: {json.dumps(case.get('gold', []), ensure_ascii=False)}", '',
                         f"Citations: {trace.get('citations', [])}", '', 'Context:', '', '```'])
        search = [e for e in trace.get('events', []) if e['event'] == 'search']
        sections.append(search[-1]['context'] if search else '(no search context)')
        sections.extend(['```', '', 'Grade / citation support / reasoning: TODO', ''])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(sections), encoding='utf8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
