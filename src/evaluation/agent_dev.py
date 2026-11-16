"""Run both real agent configurations on a specified split, retaining errors."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from src.agents.run_agent import DenseRetriever, HybridRetriever, OllamaController, run_agent
from src.common.tool_reporting import summarize_tools
from src.common.secret_safety import database_url, redact, redact_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    args = parser.parse_args()
    if args.split == 'test' and not Path('results/base-evaluation.json').is_file():
        parser.error('Freeze the dev selection before running test')
    url = database_url(args.day6)
    cases = [x for x in json.loads(Path('data/benchmark.json').read_text(encoding='utf8')) if x['split'] == args.split]
    indexes = {'dense': DenseRetriever(args.day6.resolve(), url),
               'hybrid': HybridRetriever(args.day6.resolve(), url)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing result array')
    traces = []
    for method, retriever in indexes.items():
        for case in cases:
            controller = OllamaController('llama3.2:3b')
            start = time.perf_counter()
            question = redact_text(case['question'])
            trace = {'query_id': case['id'], 'question': question, 'split': args.split,
                     'method': method, 'model': controller.model,
                     'top_k': 3 if method == 'dense' else 5,
                     'budget_chars': 4000 if method == 'dense' else 6000, 'service_cost_usd': 0}
            try:
                trace.update(run_agent(question, controller, retriever, max_searches=3,
                                       top_k=trace['top_k'], budget=trace['budget_chars']))
            except Exception as exc:
                trace.update(status='error', error=f'{type(exc).__name__}: {exc}')
            trace['model_calls'] = controller.calls
            trace['elapsed_seconds'] = time.perf_counter() - start
            trace['tools_used'] = summarize_tools(trace.get('events', []), method, trace['status'])
            trace = redact(trace)
            traces.append(trace)
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            os.replace(temporary, args.output)
            print(method, case['id'], trace['status'], round(trace['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
