"""Build or query a local real-embedding text index for the bonus source.

This indexes extracted text only, and is not a multimodal solution.
"""
import argparse
import json
import math
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--query')
    args = parser.parse_args()
    root = args.day6.resolve()
    sys.path.insert(0, str(root))
    from starter.src.embeddings import MyEmbedder
    embedder = MyEmbedder(cache_dir=str(root / '.cache/models'))
    index_path = args.data / 'text_index.json'
    if args.query:
        data = json.loads(index_path.read_text(encoding='utf8'))
        query = embedder.embed_query(args.query)
        norm = math.sqrt(sum(x*x for x in query))
        scored = []
        for row in data:
            vector = row['embedding']
            denominator = norm * math.sqrt(sum(x*x for x in vector))
            score = sum(a*b for a,b in zip(query,vector)) / denominator
            scored.append({k:v for k,v in row.items() if k != 'embedding'} | {'score': score})
        print(json.dumps(sorted(scored, key=lambda x: -x['score'])[:3], ensure_ascii=False, indent=2))
        return
    pages = json.loads((args.data / 'pages.json').read_text(encoding='utf8'))
    rows = []
    for page in pages:
        text = page['text']
        for start in range(0, len(text), 900):
            fragment = text[start:start+1000]
            if fragment.strip():
                rows.append({'id': f"nasa-p{page['page']:03d}-{start:05d}", 'source': page['source'],
                             'page': page['page'], 'start': start, 'end': start+len(fragment), 'text': fragment})
    vectors = embedder.embed_documents([r['text'] for r in rows])
    for row, vector in zip(rows, vectors):
        row['embedding'] = vector
    index_path.write_text(json.dumps(rows, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'Indexed {len(rows)} text chunks on {len(pages)} pages with {embedder.model_name} / {embedder.dimensions} dimensions.')


if __name__ == '__main__':
    main()
