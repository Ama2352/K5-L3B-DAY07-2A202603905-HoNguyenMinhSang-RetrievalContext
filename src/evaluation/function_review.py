"""Produce a readable dev review packet, without assigning correctness labels."""
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
    sections = ['# Native function agent: dev review packet', '',
                'Review each answer, citation and missing-evidence claim against the source. '
                'This file assigns no grades.', '']
    traces = (json.loads(args.traces.read_text(encoding='utf8')) if args.traces.is_file()
              else [json.loads(path.read_text(encoding='utf8')) for path in sorted(args.traces.glob('*.json'))])
    for trace in traces:
        case = cases[trace['query_id']]
        sections.extend([f"## {case['id']}", '', f"Question: {case['question']}", '',
                         f"Expected: {case.get('expected_answer', 'search and abstain')}", '',
                         f"Gold: {json.dumps(case.get('gold', []), ensure_ascii=False)}", '',
                         f"Status: {trace['status']}", '', f"Answer: {trace['display_answer']}", '',
                         f"Citations: {json.dumps(trace.get('citations', []), ensure_ascii=False)}", '',
                         'Context:', '', '```', trace.get('context', ''), '```', '',
                         'Grade, citation support and rationale: TODO', ''])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(sections), encoding='utf8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
