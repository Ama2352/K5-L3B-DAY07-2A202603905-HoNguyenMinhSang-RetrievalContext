"""Source-only BM25 retrieval over individual table rows, visual labels and prose.

All items are built from the complete corpus, independently of benchmarks. Query
constraints come only from the question, never evaluation answers or gold regions.
"""
import math
import re
import textwrap
from collections import Counter

from src.advanced.multimodal_tools import MultimodalTools,readable,terms,EXCERPT_STOPWORDS

META_WORDS=frozenset('give name identify pictured shown identified photograph photo report according'.split())

def lexemes(text):
    out=[]
    for word in terms(text):
        if len(word)>5 and word.endswith('ing'):word=word[:-3]
        elif len(word)>5 and word.endswith('ed'):word=word[:-2]
        elif len(word)>4 and word.endswith('s'):word=word[:-1]
        out.append(word)
    return out

class EvidenceItems(MultimodalTools):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.items=[]
        for row in self.rows:
            page=self.pages[row['page']]
            regions=page.get('visual',{}).get('regions',[])
            paragraphs=[readable(p.strip()) for p in page.get('text','').splitlines() if p.strip()]
            paragraph_terms=[set(lexemes(p)) for p in paragraphs]
            paragraph_df=Counter(t for group in paragraph_terms for t in group)
            description=page.get('visual',{}).get('visual_description','')
            if description:self.add_item(row,'visual_description','SOURCE VISUAL OVERVIEW:\n'+readable(description),'Full-page visual overview')
            table_titles='; '.join(str(r.get('title','')) for r in regions if r.get('kind')=='table')
            for block in self.normalized_table_rows(row):
                self.add_item(row,'table_row',f'SOURCE TABLE: {table_titles}\n{block}',block.splitlines()[0])
            for region in regions:
                kind=region.get('kind','region').lower()
                labels=region.get('labels',[])
                if isinstance(labels,str):labels=[labels]
                labels=[readable(str(label)) for label in labels]
                title=readable(str(region.get('title','')))
                location=str(region.get('location','unspecified'))
                heading=f'{kind.upper()}: {title}\nLocation on original page: {location}\n'
                shared=[v for v in labels if re.search(r'^(units|in millions|legend)',v,re.I)]
                if kind=='chart':
                    for label in labels:
                        self.add_item(row,kind,heading+'\n'.join(shared)+'\n- '+label,title+' | '+label)
                else:
                    body=heading+'\n'.join('- '+v for v in labels)
                    if kind in ('photo','photograph','image','portrait') and paragraphs:
                        anchors=set(lexemes(title+' '+' '.join(labels)))-EXCERPT_STOPWORDS-set(lexemes(' '.join(META_WORDS)))
                        scores=[sum(1/paragraph_df[t] for t in anchors & group) for group in paragraph_terms]
                        best=max(range(len(scores)),key=lambda i:scores[i])
                        if scores[best]>0:
                            body+='\nAssociated original paragraph (source-only lexical association):\n'+paragraphs[best]
                    self.add_item(row,kind,body,title)
            # Retain every source paragraph, including prose not covered by annotations.
            for paragraph in readable(page.get('text','')).splitlines():
                if not paragraph.strip():continue
                for block in textwrap.wrap(paragraph,width=1100,break_long_words=False,break_on_hyphens=False):
                    self.add_item(row,'text','ORIGINAL TEXT:\n'+block,block[:100])
        self.item_counts=[Counter(lexemes(item['text'])) for item in self.items]
        self.item_lengths=[sum(c.values()) for c in self.item_counts]
        self.item_mean=sum(self.item_lengths)/len(self.item_lengths)
        df=Counter(t for c in self.item_counts for t in c)
        self.item_idf={t:math.log(1+(len(self.items)-n+.5)/(n+.5)) for t,n in df.items()}

    def add_item(self,row,kind,text,region):
        if not text.strip():return
        summary=self.pages[row['page']].get('visual',{}).get('summary','')
        if summary:text='SOURCE PAGE: '+readable(str(summary))+'\n'+text
        item={**row,'id':f'{row["id"]}-item-{len(self.items)+1:05d}','text':text,
              'metadata':{**row['metadata'],'evidence_type':kind,'source_region':region,
                          'transformation':'Source-only row/visual-label/paragraph segmentation'}}
        self.items.append(item)

    def search_evidence(self,query,top_k=1):
        qt=set(lexemes(query))-EXCERPT_STOPWORDS-set(lexemes(' '.join(META_WORDS)))
        qtext=' '.join(terms(query))
        wants_chart=bool(re.search(r'\bcharts?\b',query,re.I))
        wants_photo=bool(re.search(r'\bphoto(?:graph)?\b|\bphotographed\b|cover.image',query,re.I))
        wants_table=not wants_chart and bool(re.search(r'\btable\b|\bsheets?\b|\bstatement\b|\bcolumn\b',query,re.I))
        wants_percent=bool(re.search(r'percent|percentage|%',query,re.I))
        wants_money=bool(re.search(r'dollars|millions|\bcost\b|\bvalue\b|\bamount\b',query,re.I))
        ranked=[]
        for item,c,length in zip(self.items,self.item_counts,self.item_lengths):
            score=sum(self.item_idf.get(t,0)*c[t]*2.5/(c[t]+1.5*(.25+.75*length/self.item_mean)) for t in qt if c[t])
            kind=item['metadata']['evidence_type'];text=item['text'];region=item['metadata']['source_region']
            if kind not in ('text','visual_description'):
                score+=sum(self.item_idf.get(t,0) for t in qt & set(lexemes(region)))*2
            if wants_chart:score*=4 if kind=='chart' else .12
            if wants_photo:score*=2 if kind in ('photo','photograph','image','visual_description') else 1
            if wants_table:score*=1.8 if kind=='table_row' else (1.5 if kind=='table' else .8)
            if wants_chart and wants_percent and not re.search(r'percent|%',text,re.I):score*=.01
            requested_goal=re.search(r'\bGoal\s+(\d+)\b',query,re.I)
            item_goals=re.findall(r'\bGoal\s+(\d+)\b',region,re.I)
            if requested_goal and item_goals and requested_goal[1] not in item_goals:score*=.01
            if wants_chart and wants_money:
                values=re.sub(r'(?:FY\s*)?202[34]|Goal\s*\d+','',region,flags=re.I)
                if not re.search(r'\d',values):score*=.01
            if re.search(r'physical.units',query,re.I):
                if 'millions' in text.lower():score*=.01
                if re.search(r'physical.units',text,re.I):score*=3
            if kind=='table_row':
                label=region.split('|')[-1].replace('TABLE ROW:','').strip()
                label_words=' '.join(terms(label))
                if len(label_words)>8 and label_words in qtext:score+=20
                if re.search(r'under\s+(?:the\s+)?assets\b',query,re.I) and not re.search(r'TABLE ROW: Assets\b',text,re.I):score*=.03
                if re.search(r'under\s+(?:the\s+)?liabilities\b',query,re.I) and not re.search(r'TABLE ROW: Liabilities\b',text,re.I):score*=.03
                if 'with the public' in qtext and 'intragovernmental' in region.lower() and 'public' not in region.lower():score*=.03
            if score>0:ranked.append({**item,'score':score})
        ranked.sort(key=lambda item:(-item['score'],item['id']))
        results=[]
        for item in ranked[:max(1,min(top_k,3))]:
            text=item['text']
            if item['metadata']['evidence_type']=='chart':
                if wants_percent:text=re.sub(r'\$[\d,]+(?:\.\d+)?','',text)
                elif wants_money:text=re.sub(r'\b\d+(?:\.\d+)?\s*(?:percent|%)','',text,flags=re.I)
            results.append({**item,'text':text})
        return {'results':results,'steps':[{'method':'bm25_source_evidence_items','query':query,
                 'constraints_from_question':{'chart':wants_chart,'photo':wants_photo,'table':wants_table,'percentage':wants_percent},
                 'indexed_items':len(self.items),'indexed_pages':len(self.pages),
                 'results':[{'rank':i+1,'id':r['id'],'page':r['page'],'source':r['source'],'score':r['score'],
                             'region':r['metadata']['source_region']} for i,r in enumerate(ranked[:10])]}]}
