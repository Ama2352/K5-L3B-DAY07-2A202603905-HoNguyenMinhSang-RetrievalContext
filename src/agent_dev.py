"""Run both real agent configurations on dev only, retaining per-case errors."""
import argparse
import json
import os
import time
from pathlib import Path
from run_agent import DenseRetriever, HybridRetriever, OllamaController, run_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    url = os.environ['DATABASE_URL']
    cases = [x for x in json.loads(Path('data/benchmark.json').read_text(encoding='utf8')) if x['split'] == 'dev']
    indexes = {'dense': DenseRetriever(args.day6.resolve(), url),
               'hybrid': HybridRetriever(args.day6.resolve(), url)}
    args.output.mkdir(parents=True, exist_ok=True)
    for method, retriever in indexes.items():
        for case in cases:
            path = args.output / f'{method}-{case["id"]}.json'
            if path.exists():
                continue
            controller = OllamaController('llama3.2:3b')
            start = time.perf_counter()
            trace = {'query_id': case['id'], 'question': case['question'], 'split': 'dev',
                     'method': method, 'model': controller.model,
                     'top_k': 3 if method == 'dense' else 5,
                     'budget_chars': 4000 if method == 'dense' else 6000, 'service_cost_usd': 0}
            try:
                trace.update(run_agent(case['question'], controller, retriever, max_searches=3,
                                       top_k=trace['top_k'], budget=trace['budget_chars']))
            except Exception as exc:
                trace.update(status='error', error=f'{type(exc).__name__}: {exc}')
            trace['model_calls'] = controller.calls
            trace['elapsed_seconds'] = time.perf_counter() - start
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
            print(method, case['id'], trace['status'], round(trace['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
