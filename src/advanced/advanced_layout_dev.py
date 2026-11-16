"""Incrementally evaluate the text/layout agent on advanced dev questions only."""
import argparse
import json
import os
import sys
from pathlib import Path
from src.advanced.advanced_layout_tools import AdvancedLayoutTools, ADVANCED_SCHEMAS, execute_advanced_tool
from src.agents.function_agent import run
from src.common.secret_safety import redact, redact_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cases = json.loads(Path('data/advanced/benchmark-dev-draft.json').read_text(encoding='utf8'))
    if any(q['split'] != 'dev' for q in cases):
        raise ValueError('This runner only accepts the advanced dev split')
    root = args.day6.resolve()
    sys.path.insert(0, str(root))
    from starter.src.embeddings import MyEmbedder
    tools = AdvancedLayoutTools(MyEmbedder(cache_dir=str(root / '.cache/models')), args.data)
    traces = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError('Refusing to overwrite existing traces')
        traces = json.loads(args.output.read_text(encoding='utf8'))
        allowed = {q['id']: q['question'] for q in cases}
        if len({x['query_id'] for x in traces}) != len(traces):
            raise ValueError('Duplicate query IDs in existing traces')
        if any(x['query_id'] not in allowed or x['model'] != args.model
               or x['question'] != allowed[x['query_id']] for x in traces):
            raise ValueError('Existing trace IDs, questions or model do not match this dev run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if any(t['query_id'] == case['id'] for t in traces):
            continue
        trace = run(case['question'], args.model, tools, max_calls=4, budget=8000,
                    schemas=ADVANCED_SCHEMAS, dispatcher=execute_advanced_tool)
        trace.update(query_id=case['id'], split='dev',
                     limitations=['Text/layout only; no images interpreted by the model.',
                                  'Candidate tables and citations require source review.'])
        traces.append(redact(trace))
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        os.replace(temporary, args.output)
        print(case['id'], trace['status'], round(trace['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
