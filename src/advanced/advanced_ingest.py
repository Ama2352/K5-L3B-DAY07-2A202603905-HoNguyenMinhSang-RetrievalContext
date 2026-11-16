"""Prepare a traceable text/image inventory for the optional NASA corpus.

This does not describe images, render vector charts, or claim multimodal coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

URL = 'https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf'
EXPECTED_SHA256 = '65ee29adf5f3a92fcd74701696c25dae8f2d18f1ba2e7f4fba2bcfc059acbefc'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/advanced/generated'))
    args = parser.parse_args()
    digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('PDF hash differs from the frozen source; inspect the version before proceeding')
    reader = PdfReader(args.pdf)
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / 'images'
    images_dir.mkdir(exist_ok=True)
    pages, failures = [], []
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ''
        except Exception as exc:
            text = ''
            failures.append({'page': number, 'stage': 'text', 'error': f'{type(exc).__name__}: {exc}'})
        images = []
        try:
            for index, image in enumerate(page.images):
                suffix = Path(image.name).suffix.lower()
                if not suffix or not suffix[1:].isalnum():
                    suffix = '.bin'
                name = f'p{number:03d}-{index:03d}{suffix}'
                data = image.data
                (images_dir / name).write_bytes(data)
                images.append({'file': f'images/{name}', 'original_name': image.name,
                               'sha256': hashlib.sha256(data).hexdigest()})
        except Exception as exc:
            failures.append({'page': number, 'stage': 'images', 'error': f'{type(exc).__name__}: {exc}'})
        pages.append({'source_id': 'nasa_fy2024_afr', 'source': URL, 'page': number,
                      'text': text, 'embedded_images': images})
    (args.output / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    manifest = {'url': URL, 'sha256': digest, 'pages': len(pages),
                'text_characters': sum(len(p['text']) for p in pages),
                'pages_without_text': [p['page'] for p in pages if not p['text'].strip()],
                'embedded_images': sum(len(p['embedded_images']) for p in pages),
                'image_failures': failures,
                'limitations': ['Embedded image extraction does not capture vector charts.',
                                'Page text is not a verified description of embedded images.',
                                'Image positions and rendered regions are not yet indexed.',
                                'Some JPEG 2000 soft masks have dimensions that do not match their images; extracted appearance needs checking against the PDF.',
                                'No advanced benchmark or multimodal retrieval has been completed.']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
