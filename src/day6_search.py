"""Read-only search against the existing Day 6 strategy B pgvector snapshot."""
import argparse
import json
import os
import sys
from pathlib import Path
from secret_safety import database_url, redact, redact_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    url = database_url(args.day6)
    root = args.day6.resolve()
    sys.path.insert(0, str(root))
    from starter.src.embeddings import MyEmbedder
    from starter.src.store import PgVectorStore
    store = PgVectorStore(url, 384)
    count = store.count('B')
    if count != 160:
        raise RuntimeError(f'Expected 160 strategy B records, found {count}; verify the Day 6 snapshot')
    embedder = MyEmbedder(cache_dir=str(root / '.cache/models'))
    results = store.search('B', embedder.embed_query(args.query), top_k=args.top_k)
    print(json.dumps(redact(results), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
