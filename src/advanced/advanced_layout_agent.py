"""Experimental native-function agent over extracted PDF layout, not a vision model."""
import argparse
import base64
import json
import sys
from pathlib import Path
from src.advanced.advanced_layout_tools import AdvancedLayoutTools, ADVANCED_SCHEMAS, execute_advanced_tool
from src.agents.function_agent import run
from src.common.secret_safety import redact, redact_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--question', required=True)
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--vision', action='store_true', help='Send up to three retrieved page images to a vision-capable model')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.day6.resolve()
    sys.path.insert(0, str(root))
    from starter.src.embeddings import MyEmbedder
    embedder = MyEmbedder(cache_dir=str(root / '.cache/models'))
    tools = AdvancedLayoutTools(embedder, args.data)
    seen = set()
    def images_for_results(results):
        images = []
        for result in results:
            page = result['page']
            if page in seen or len(seen) >= 3:
                continue
            path = args.data / 'rendered' / f'page-{page:03d}.png'
            images.append({'page': page, 'path': str(path), 'data': base64.b64encode(path.read_bytes()).decode('ascii')})
            seen.add(page)
        return images
    trace = run(args.question, args.model, tools, max_calls=4, budget=8000,
                schemas=ADVANCED_SCHEMAS, dispatcher=execute_advanced_tool,
                image_provider=images_for_results if args.vision else None)
    trace['limitations'] = (["Only retrieved page images are supplied; image-only retrieval remains incomplete."]
                            if args.vision else ['Text/layout only; images were not interpreted by the model.'])
    trace['limitations'].append('Candidate tables, visual claims and citations require source-page verification.')
    trace = redact(trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: trace.get(k) for k in ('status', 'display_answer', 'citations', 'tool_counts', 'error')}, ensure_ascii=False))
    if trace['status'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
