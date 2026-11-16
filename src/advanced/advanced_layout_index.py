"""Index source-linked page geometry and visible text; table candidates need review."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from src.advanced.advanced_ingest import EXPECTED_SHA256, URL
from src.common.secret_safety import redact, redact_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day6', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    args = parser.parse_args()
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Unexpected PDF version')
    sys.path.insert(0, str((args.data / 'vendor').resolve()))
    sys.path.insert(0, str(args.day6.resolve()))
    import pymupdf
    from starter.src.embeddings import MyEmbedder
    pdf = pymupdf.open(args.pdf)
    pages, chunks, failures = [], [], []
    for number, page in enumerate(pdf, 1):
        entry = {'page': number, 'source': URL, 'page_image': f'rendered/page-{number:03d}.png',
                 'size_points': list(page.rect), 'blocks': [], 'tables': []}
        try:
            for i, block in enumerate(page.get_text('dict')['blocks']):
                bbox = list(block['bbox'])
                if block['type'] == 0:
                    text = ' '.join(span['text'] for line in block['lines'] for span in line['spans']).strip()
                    if text:
                        entry['blocks'].append({'type': 'text', 'bbox': bbox, 'text': text})
                elif block['type'] == 1:
                    entry['blocks'].append({'type': 'image', 'bbox': bbox})
            # Group nearby source-order text blocks, retaining each block's exact box.
            # This reduces embedding calls while preserving page and region provenance.
            group = []
            def flush():
                if group:
                    text = '\n'.join(b['text'] for b in group)
                    boxes = [b['bbox'] for b in group]
                    chunks.append({'id': f'layout-p{number:03d}-g{len(chunks):05d}',
                        'page': number, 'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                                 max(b[2] for b in boxes), max(b[3] for b in boxes)],
                        'block_boxes': boxes, 'source': URL, 'text': text,
                        'type': 'text_block_group', 'page_image': entry['page_image']})
                    group.clear()
            for block in entry['blocks']:
                if block['type'] != 'text':
                    flush()
                    continue
                for start in range(0, len(block['text']), 900):
                    part = {'text': block['text'][start:start + 900], 'bbox': block['bbox']}
                    if sum(len(b['text']) + 1 for b in group) + len(part['text']) > 1000:
                        flush()
                    group.append(part)
            flush()
            for table in page.find_tables().tables:
                if table.row_count >= 2 and table.col_count >= 2:
                    entry['tables'].append({'bbox': list(table.bbox), 'rows': table.extract(),
                                            'warning': 'Automatic candidate; row and column extraction may be wrong.'})
        except Exception as exc:
            failures.append({'page': number, 'error': f'{type(exc).__name__}: {exc}'})
        pages.append(entry)
    embedder = MyEmbedder(cache_dir=str(args.day6.resolve() / '.cache/models'))
    vectors = embedder.embed_documents([c['text'] for c in chunks])
    for row, vector in zip(chunks, vectors):
        row['embedding'] = vector
    (args.data / 'layout_pages.json').write_text(json.dumps(pages, ensure_ascii=False) + '\n', encoding='utf8')
    (args.data / 'layout_vectors.json').write_text(json.dumps(chunks, ensure_ascii=False) + '\n', encoding='utf8')
    summary = {'pages': len(pages), 'chunks': len(chunks), 'image_regions': sum(
        sum(b['type'] == 'image' for b in p['blocks']) for p in pages),
        'table_candidates': sum(len(p['tables']) for p in pages), 'errors': failures,
        'limitations': ['Table candidates are not verified cell-level transcriptions.',
                        'Images and vector charts are linked to full-page renders but are not semantically indexed.',
                        'Visible text extraction and page layouts require source verification.']}
    summary = redact(summary)
    (args.data / 'layout_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(redact_text(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(1)
