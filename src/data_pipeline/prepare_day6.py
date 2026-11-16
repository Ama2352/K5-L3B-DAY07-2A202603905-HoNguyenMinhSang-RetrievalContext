"""Recreate Day 6 strategy B chunks and import its previously used dev labels."""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    root = args.day6.resolve()
    sys.path.insert(0, str(root))
    from starter.src.chunking import by_paragraph
    from starter.src.ingestion import load_pages

    pages = load_pages(root / 'data/pages.json')
    chunks = [c for p in pages for c in by_paragraph(p, 1000)]
    rows = [{'id': c.chunk_id, 'text': c.text, 'source_id': c.document_id,
             'source': c.source, 'page': c.page, 'metadata': c.metadata} for c in chunks]
    questions = json.loads((root / 'benchmark/questions.json').read_text(encoding='utf-8'))
    dev = []
    for q in questions:
        if q['split'] != 'dev':
            continue
        dev.append({'id': q['id'], 'split': 'dev', 'question': q['question'],
                    'expected_answer': q['expected_answer'], 'answerable': True,
                    'gold': [{'source_id': q['gold_source_id'], 'page': q['gold_page'],
                              'evidence': q['evidence_quote']}], 'filter': q['filter']})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('day6_chunks.json', rows), ('day6_dev_answerable.json', dev)]:
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{len(pages)} pages, {len(rows)} chunks, {len(dev)} dev questions')


if __name__ == '__main__':
    main()
