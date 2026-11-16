"""Combine source-only visual annotations with all original PDF text/layout pages."""
import argparse
import hashlib
import json
from pathlib import Path
from src.advanced.advanced_ingest import EXPECTED_SHA256
from src.common.secret_safety import redact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts',type=Path,nargs='*',help='Merge source-only reviewed parts into published annotations')
    parser.add_argument('--annotations',type=Path,default=Path('data/advanced/visual-catalog.json'))
    parser.add_argument('--data',type=Path,default=Path('data/advanced/generated'))
    parser.add_argument('--pdf',type=Path,default=Path('data/advanced/nasa-fy-2024-afr.pdf'))
    args=parser.parse_args()
    if digest(args.pdf)!=EXPECTED_SHA256:raise ValueError('Unexpected PDF version')
    if args.parts:
        records=[]
        for path in args.parts:records.extend(json.loads(path.read_text(encoding='utf8')))
        if sorted(p['page'] for p in records)!=list(range(1,128)):
            raise ValueError('Require one actually reviewed annotation for each physical source page')
        annotations={'method':'Codex-assisted visual inspection of original renders; source-only corpus annotation',
                     'benchmark_input':False,'source_pdf_sha256':EXPECTED_SHA256,
                     'scope':'All pages viewed; informative visual regions described; not all table cells independently transcribed.',
                     'pages':sorted(records,key=lambda p:p['page'])}
        args.annotations.write_text(json.dumps(redact(annotations),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    annotations=json.loads(args.annotations.read_text(encoding='utf8'))
    if annotations['source_pdf_sha256']!=EXPECTED_SHA256:raise ValueError('Annotation source mismatch')
    notes={p['page']:p for p in annotations['pages']}
    source=json.loads((args.data/'layout_pages.json').read_text(encoding='utf8'))
    if sorted(notes)!=list(range(1,128)) or sorted(p['page'] for p in source)!=list(range(1,128)):
        raise ValueError('Incomplete source/annotation coverage')
    pages=[]
    for page in source:
        note=notes[page['page']]
        image=args.data/page['page_image']
        size=page['size_points']
        visual={k:v for k,v in note.items() if k!='page'}
        pages.append({'page':page['page'],'source':page['source'],'bbox_points':size if len(size)==4 else [0,0,*size],
                      'page_image':page['page_image'],'image_sha256':digest(image),
                      'text':'\n'.join(b['text'] for b in page['blocks'] if b['type']=='text'),
                      'visual':visual,'status':'ok','truncated':False,
                      'extraction_method':'original_PyMuPDF_text_and_source_only_reviewed_visual_catalog',
                      'verification':note.get('verification_notes','Review scope recorded in published annotations.')})
    document={'model':'Codex-assisted source visual review; no fictitious local-model responses',
              'source_pdf_sha256':EXPECTED_SHA256,'annotation_sha256':digest(args.annotations),
              'origin':annotations['method'],'benchmark_input':False,
              'coverage':{'indexed_pages':len(pages),'corpus_pages':127,'errors':0},
              'limitations':['Tables retain original text and rendered pages; not every cell is independently retyped.',
                             'Source page remains authoritative; every benchmark answer/citation needs original-page review.'],
              'pages':pages}
    (args.data/'multimodal_pages.json').write_text(json.dumps(redact(document),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Built source-only verified visual/text catalog:',len(pages),'pages')


if __name__=='__main__':main()
