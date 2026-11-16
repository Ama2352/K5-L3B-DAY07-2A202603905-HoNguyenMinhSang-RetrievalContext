"""Aggregate complete, original-source reviews; never generate correctness labels."""
import argparse
import json
from collections import Counter
from pathlib import Path
from src.advanced.evaluate_multimodal import BENCHMARKS,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,default=Path('results/advanced'))
    args=parser.parse_args()
    if args.split=='test' and not (args.output_root/'selection.json').exists():
        raise ValueError('Freeze selection before opening test labels')
    path=args.output_root/f'{args.split}.json'
    doc=json.loads(path.read_text(encoding='utf8'))
    cases={c['id']:c for c in json.loads(BENCHMARKS[args.split].read_text(encoding='utf8'))}
    traces={t['query_id']:t for t in doc['traces']}
    reviews=json.loads(args.reviews.read_text(encoding='utf8'))
    ids=[r['query_id'] for r in reviews]
    if len(set(ids))!=len(cases) or len(ids)!=len(cases) or set(ids)!=set(cases) or set(traces)!=set(cases):
        raise ValueError('Every published case, including errors, needs one original-source review')
    for review in reviews:
        key=review['query_id'];case=cases[key];trace=traces[key]
        if type(review.get('passed')) is not bool or not review.get('reason'):
            raise ValueError(f'Incomplete manual judgment: {key}')
        if review.get('group')!=case['group']:
            raise ValueError(f'Group mismatch: {key}')
        if review['passed'] and trace['status']=='error':raise ValueError('System errors cannot pass')
        if review['passed'] and case['group']!='missing-evidence':
            evidence=review.get('source_evidence',[])
            if not evidence or any(type(e.get('page')) is not int or not 1<=e['page']<=127 or not e.get('region') for e in evidence):
                raise ValueError(f'Pass requires original page/region evidence: {key}')
            if case['group']=='multi-hop' and len({(e['page'],e['region']) for e in evidence})<2:
                raise ValueError('Multi-hop needs two original source items: pages, tables or figures')
        if review['passed'] and case['group']=='missing-evidence' and not review.get('absence_check'):
            raise ValueError('Missing-evidence pass requires a corpus absence check')
    n=sum(r['passed'] for r in reviews);N=len(cases)
    groups=Counter(c['group'] for c in cases.values())
    by_group={g:{'n':sum(r['passed'] and r['group']==g for r in reviews),'N':count} for g,count in groups.items()}
    for row in by_group.values():row['percent']=100*row['n']/row['N']
    doc['reviews']=reviews
    doc['review_method']='Codex-assisted inspection of original PDF regions, evidence and citations; not independent human adjudication.'
    doc['summary'].update(reviewed_cases=N,pass_rate=n/N,pass_fraction={'n':n,'N':N,'percent':100*n/N},by_group=by_group,
                          threshold_met=(args.split=='test' and N>=30 and 10*n>=9*N),
                          pass_rate_note='Strict pass/fail from complete source-backed reviews; errors retained in denominator.')
    save(path,doc)
    print(json.dumps(doc['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
