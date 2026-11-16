"""Run the native function-calling agent on dev without passing gold to it."""
import argparse
import json
import os
import sys
from pathlib import Path
from src.agents.function_agent import run
from src.retrieval.retrieval_tools import RetrievalTools
from src.common.secret_safety import database_url, redact, redact_text
from src.common.trace_flow import add_reconstructed_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--resume', action='store_true', help='Resume matching dev traces from the JSON array')
    args = parser.parse_args()
    cases = [q for q in json.loads(Path('data/benchmark.json').read_text(encoding='utf8')) if q['split'] == 'dev']
    tools = RetrievalTools(args.day6, database_url(args.day6))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f'Refusing to overwrite existing trace: {args.output}')
        traces = json.loads(args.output.read_text(encoding='utf8'))
        allowed = {q['id']: q['question'] for q in cases}
        ids = set()
        for trace in traces:
            key = trace['query_id']
            if key in ids or key not in allowed or trace['question'] != allowed[key] or trace['model'] != args.model:
                raise ValueError('Existing traces do not match this dev set and model')
            ids.add(key)
    for case in cases:
        if any(t['query_id'] == case['id'] for t in traces):
            continue
        trace = run(case['question'], args.model, tools)
        trace.update(query_id=case['id'], split='dev')
        traces.append(add_reconstructed_flow(redact(trace)))
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
