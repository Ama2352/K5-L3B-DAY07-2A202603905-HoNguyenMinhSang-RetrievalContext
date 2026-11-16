"""Render every bonus PDF page and save text/image geometry for source review."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from advanced_ingest import EXPECTED_SHA256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--dpi', type=int, default=110)
    args = parser.parse_args()
    if hashlib.sha256(args.pdf.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError('Unexpected source PDF hash')
    sys.path.insert(0, str(args.output / 'vendor'))
    import pymupdf
    doc = pymupdf.open(args.pdf)
    rendered = args.output / 'rendered'
    rendered.mkdir(parents=True, exist_ok=True)
    pages, errors = [], []
    for number, page in enumerate(doc, 1):
        try:
            path = rendered / f'page-{number:03d}.png'
            page.get_pixmap(dpi=args.dpi, alpha=False).save(path)
            blocks = []
            for b in page.get_text('dict')['blocks']:
                if b['type'] == 0:
                    content = ' '.join(s['text'] for line in b['lines'] for s in line['spans'])
                    blocks.append({'bbox': list(b['bbox']), 'text': content})
                else:
                    blocks.append({'bbox': list(b['bbox']), 'type': 'image'})
            pages.append({'page': number, 'render': f'rendered/{path.name}',
                          'page_size_points': list(page.rect), 'blocks': blocks,
                          'vector_paths': len(page.get_drawings())})
        except Exception as exc:
            errors.append({'page': number, 'error': f'{type(exc).__name__}: {exc}'})
    info = {'source_sha256': EXPECTED_SHA256, 'dpi': args.dpi, 'pages': pages, 'errors': errors}
    (args.output / 'layout.json').write_text(json.dumps(info, ensure_ascii=False) + '\n', encoding='utf8')
    print(f'Rendered {len(pages)}/{len(doc)} pages; {len(errors)} errors; '
          f'{sum(p["vector_paths"] > 0 for p in pages)} pages contain vector paths.')


if __name__ == '__main__':
    main()
