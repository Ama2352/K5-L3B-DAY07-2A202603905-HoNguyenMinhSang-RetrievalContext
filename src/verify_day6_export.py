"""Compare the reconstructed corpus against recorded Day 6 dev search hits."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', required=True, type=Path)
    parser.add_argument('--chunks', type=Path, default=Path('data/day6_chunks.json'))
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    by_id = {c['id']: c for c in chunks}
    if len(by_id) != len(chunks):
        raise ValueError('Duplicate chunk ID')
    traces = json.loads((args.day6 / 'results/traces_dev.json').read_text(encoding='utf-8'))
    hits = [h for t in traces if t['strategy'] == 'B' for h in t['hits']]
    for hit in hits:
        chunk = by_id.get(hit['chunk_id'])
        if chunk is None or chunk['text'] != hit['text']:
            raise ValueError(f"Missing or changed chunk: {hit['chunk_id']}")
        metadata = hit['metadata']
        if any(chunk[k] != metadata[v] for k, v in [('source_id', 'document_id'), ('source', 'source'), ('page', 'page')]):
            raise ValueError(f"Provenance mismatch: {hit['chunk_id']}")
        if any(metadata.get(k) != v for k, v in chunk['metadata'].items()):
            raise ValueError(f"Metadata mismatch: {hit['chunk_id']}")
    print(f'Validated {len(hits)} recorded B dev hits against {len(by_id)} reconstructed chunks; '
          f'{len({h["chunk_id"] for h in hits})} unique IDs. This does not validate vectors or the live store.')


if __name__ == '__main__':
    main()
