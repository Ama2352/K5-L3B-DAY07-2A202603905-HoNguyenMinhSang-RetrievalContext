"""Export compact source/index provenance, not PDF images or model/cache files."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from src.common.secret_safety import redact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/advanced/generated'))
    p.add_argument('--annotations',type=Path,default=Path('data/advanced/visual-catalog.json'))
    p.add_argument('--output',type=Path,default=Path('results/advanced/index-manifest.json'))
    args=p.parse_args()
    catalog=json.loads((args.data/'multimodal_pages.json').read_text(encoding='utf8'))
    annotations=json.loads(args.annotations.read_text(encoding='utf8'))
    vectors=json.loads((args.data/'multimodal_vectors.json').read_text(encoding='utf8'))
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if len(catalog['pages'])!=127 or len(vectors['vectors'])!=127:
        raise ValueError('Require full source corpus/index')
    manifest={'source':{'title':'NASA FY2024 Agency Financial Report','physical_pages':127,
        'url':'https://www.nasa.gov/wp-content/uploads/2023/11/nasa-fy-2024-afr-1.pdf',
        'sha256':catalog['source_pdf_sha256'],'sharing':'Source and access recipe supplied; original PDF/third-party images not rehosted.'},
        'index':{'pages':127,'unprocessed_pages':[],'vector_count':len(vectors['vectors']),
        'dimensions':len(vectors['vectors'][0]),'embedding_model':'BAAI/bge-small-en-v1.5',
        'embedding_view':vectors['embedding_view'],'lexical_scope':'All original extracted text and source-reviewed visual descriptions',
        'catalog_method':annotations['method'],'benchmark_used_to_build_index':False,
        'visual_region_kinds':dict(Counter(r['kind'] for a in annotations['pages'] for r in a.get('regions',[]))),
        'original_text_characters':sum(len(page['text']) for page in catalog['pages'])},
        'hashes':{'annotations':digest(args.annotations),'catalog':digest(args.data/'multimodal_pages.json'),
                  'vectors':digest(args.data/'multimodal_vectors.json')},
        'verification_scope':annotations['scope'],
        'limitations':catalog['limitations']+['Dense embedding may truncate long inputs; BM25 indexes full page text/catalog.',
            'Descriptions were reviewed against all original pages, not every financial table cell independently retyped.',
            'Benchmark answer/citation checks use original pages, not catalog assertions alone.'],
        'pages':[{'page':page['page'],'image_sha256':page['image_sha256'],'bbox_points':page['bbox_points'],
                  'page_image':page['page_image'],'text_characters':len(page['text']),
                  'region_count':len(page['visual'].get('regions',[]))} for page in catalog['pages']]}
    from src.advanced.evidence_items import EvidenceItems
    items=EvidenceItems(None,args.data,retrieval_mode='bm25')
    manifest['index']['bm25_evidence_items']=len(items.items)
    manifest['index']['bm25_item_pages']=len({row['page'] for row in items.items})
    manifest['index']['bm25_item_kinds']=dict(Counter(row['metadata']['evidence_type'] for row in items.items))
    manifest['index']['query_constraints']='Only question-derived modality, unit and explicit financial-section conditions; no benchmark/gold input.'
    manifest['hashes']['itemizer_code']=digest(Path('src/advanced/evidence_items.py'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(redact(manifest),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Exported127-page index provenance:',args.output)


if __name__=='__main__':main()
