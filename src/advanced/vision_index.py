"""Build corpus-wide page evidence from original renders, never benchmark answers."""
import argparse
import base64
import hashlib
import json
import time
import urllib.request
import sys
import re
from pathlib import Path
from src.common.secret_safety import redact
from src.advanced.advanced_ingest import EXPECTED_SHA256

MODEL = 'qwen3-vl:2b-instruct'
OPTIONS = {'temperature': 0, 'num_ctx': 8192, 'num_predict': 4096, 'num_thread': 4}
PROMPT = '''Catalog this original report page for evidence retrieval. Describe every informative
photograph, diagram, table and chart, including titles, labels, legends, axes, units, visible
numbers and relationships. Include a short summary of prose. Do not reproduce full prose or
every cell of long tables: name the table, its units, columns and key rows, retaining exact
printed numbers only when readable. Keep the entire response below 600 words. Do not invent facts or infer
unreadable values. Distinguish printed text from visual interpretation. Ignore decorative
logos and backgrounds. Return concise plain text: a short summary followed by each visual's
kind, location on the page, description, visible labels/units, and uncertainty. Say explicitly when unreadable.
No benchmark or question is supplied. Each entry refers to this entire original page.'''

# Source-only coverage audit of the geometry rule, independent of benchmark targets:
# two glossary tables and a full-page back-cover photograph missed by automatic region detection.
VISUAL_OVERRIDES = {124,125,127}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)


def chat_page(model, image, prompt, options=OPTIONS):
    payload = {'model': model, 'stream': False, 'options': options,
               'messages': [{'role': 'user', 'content': prompt,
                             'images': [base64.b64encode(image.read_bytes()).decode('ascii')]}]}
    req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.load(response)
    return result, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/advanced/generated'))
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--pages', type=int, nargs='*', help='Smoke subset only; omitted means full corpus')
    parser.add_argument('--pdf', type=Path, default=Path('data/advanced/nasa-fy-2024-afr.pdf'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    source_sha=hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if source_sha!=EXPECTED_SHA256:
        raise ValueError('Unexpected source PDF version')
    path = args.data/'multimodal_pages.json'
    pages = json.loads((args.data/'layout_pages.json').read_text(encoding='utf8'))
    sys.path.insert(0,str((args.data/'vendor').resolve()))
    import pymupdf
    pdf=pymupdf.open(args.pdf)
    if len(pdf)!=127 or sorted(p['page'] for p in pages)!=list(range(1,128)):
        raise ValueError('Expected all 127 source pages')
    with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=15) as response:
        installed=json.load(response)['models']
    model_digest=next(m['digest'] for m in installed if m['name']==args.model)
    selected = set(args.pages or [p['page'] for p in pages])
    index = {'model': args.model, 'model_digest':model_digest, 'options': OPTIONS, 'prompt': PROMPT, 'source_pdf_sha256':source_sha,
             'origin': 'Original page images + extracted source text, no benchmark/gold input', 'pages': []}
    if path.exists():
        if not args.resume:
            raise FileExistsError('Existing visual index: use --resume after checking model/prompt')
        index = json.loads(path.read_text(encoding='utf8'))
        if index['model'] != args.model or index['prompt'] != PROMPT or index['options'] != OPTIONS or index.get('model_digest')!=model_digest:
            raise ValueError('Cannot mix model/index configurations')
        if args.retry_errors:
            index.setdefault('previous_errors',[]).extend(r for r in index['pages'] if r['status']=='error')
            index['pages']=[r for r in index['pages'] if r['status']!='error']
    for page in pages:
        if page['page'] not in selected or any(p['page']==page['page'] for p in index['pages']):
            continue
        image = args.data/page['page_image']
        size=page['size_points']
        page_box=size if len(size)==4 else [0,0,*size]
        row = {'page': page['page'], 'source': page['source'], 'bbox_points': page_box,
               'page_image': page['page_image'], 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
               'text': '\n'.join(b['text'] for b in page['blocks'] if b['type']=='text'),
               'verification': 'Model extraction is unverified until compared with original image.'}
        try:
            region_area=lambda box:max(0,box[2]-box[0])*max(0,box[3]-box[1])
            area=region_area(page_box)
            large_images=[b for b in page['blocks'] if b['type']=='image' and region_area(b['bbox'])>area*.04]
            large_vectors=[list(d['rect']) for d in pdf[page['page']-1].get_drawings()
                           if area*.04<region_area(list(d['rect']))<area*.85]
            row['visual_detection']={'large_images':len(large_images),'large_vector_regions':len(large_vectors),
                'table_candidates':len(page['tables']),'area_threshold':.04,
                'source_visual_audit_override':page['page'] in VISUAL_OVERRIDES}
            # Content-based coverage rule, independent of benchmarks; small repeated logos are decorative.
            if large_images or large_vectors or page['tables'] or page['page'] in VISUAL_OVERRIDES:
                preview=args.data/'vision-rendered'/f'page-{page["page"]:03d}.png'
                preview.parent.mkdir(exist_ok=True)
                pdf[page['page']-1].get_pixmap(dpi=72,alpha=False).save(str(preview))
                row['vision_input_image']=str(preview.relative_to(args.data))
                row['vision_input_sha256']=hashlib.sha256(preview.read_bytes()).hexdigest()
                row['vision_input_dpi']=72
                result, elapsed = chat_page(args.model, preview, PROMPT)
                raw = result['message']['content']
                row.update(raw_response=raw, elapsed_seconds=elapsed, extraction_method='vision_and_source_text',
                           prompt_eval_count=result.get('prompt_eval_count'), eval_count=result.get('eval_count'))
                row.update(status='ok', visual={'summary':raw,'visuals':[]},
                           generation_stop=result.get('done_reason'),
                           truncated=result.get('done_reason')=='length')
                numbers=lambda text:set(re.findall(r'\d+(?:[,.]\d+)*',text))
                row['numeric_crosscheck']={'catalog_numbers_not_in_extracted_source':sorted(numbers(raw)-numbers(row['text'])),
                    'note':'Screening only; extracted source text may omit or reorder visual numbers. Verify original image.'}
            else:
                row.update(status='ok',visual={'summary':'Text page; no large image/vector/table region detected.',
                    'visuals':[]},extraction_method='source_text_and_layout',elapsed_seconds=0,
                    verification='Detection rule requires audit; small informative visuals could be missed.')
        except Exception as exc:
            row.update(status='error', error=f'{type(exc).__name__}: {exc}')
        index['pages'].append(row)
        index['coverage']={'indexed_pages': len(index['pages']), 'corpus_pages':len(pages),
                           'errors':sum(r['status']=='error' for r in index['pages'])}
        save(path,index)
        print('page',page['page'],row['status'],round(row.get('elapsed_seconds',0),1),flush=True)


if __name__=='__main__':
    main()
