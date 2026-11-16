"""Retrieve all indexed source pages using lexical and dense visual/text evidence."""
import ast
import json
import math
import operator
import textwrap
from collections import Counter
from pathlib import Path
from src.retrieval.fusion import rrf

EXCERPT_STOPWORDS=frozenset('what which how many much was were is are the a an of at in on for to and by did does do from as with its this that according s'.split())
SOURCE_TITLE_WEIGHT=3


def readable(text):
    import re
    text=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',text)
    return re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',text)


def terms(text):
    import re
    return re.findall(r'[a-z0-9]+',readable(text).lower())


class MultimodalTools:
    def __init__(self, embedder, data=Path('data/advanced/generated')):
        self.embedder=embedder
        self.data=data
        document=json.loads((data/'multimodal_pages.json').read_text(encoding='utf8'))
        self.pages={p['page']:p for p in document['pages'] if p['status']=='ok'}
        if len(self.pages)!=127:
            raise ValueError('Full 127-page visual/text catalog is required before benchmark')
        self.rows=[]
        for page in self.pages.values():
            visual=page['visual']
            text=page['text']+'\nVisual interpretation (verify in page image):\n'+json.dumps(visual,ensure_ascii=False)
            self.rows.append({'id':f'nasa-p{page["page"]:03d}', 'page':page['page'],'text':text,
                'source_id':'nasa_fy2024_afr','source':page['source'],
                'metadata':{'bbox_points':page['bbox_points'],'page_image':page['page_image'],
                            'evidence_type':'page_text_and_visual_catalog'}})
        self.counts=[Counter(terms(r['text'])) for r in self.rows]
        self.lengths=[sum(c.values()) for c in self.counts]
        self.mean_length=sum(self.lengths)/len(self.rows)
        df=Counter(t for c in self.counts for t in c)
        self.idf={t:math.log(1+(len(self.rows)-n+.5)/(n+.5)) for t,n in df.items()}
        vector_path=data/'multimodal_vectors.json'
        from hashlib import sha256
        corpus_hash=sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()
        if vector_path.exists():
            stored=json.loads(vector_path.read_text(encoding='utf8'))
            if stored['corpus_sha256']!=corpus_hash or stored.get('embedding_view')!='visual_catalog_then_original_text':
                raise ValueError('Rebuild stale multimodal_vectors.json for changed corpus')
            self.vectors=stored['vectors']
        else:
            self.vectors=embedder.embed_documents([json.dumps(self.pages[r['page']]['visual'],ensure_ascii=False)+'\n'+r['text'] for r in self.rows])
            vector_path.write_text(json.dumps({'corpus_sha256':corpus_hash,'embedding_view':'visual_catalog_then_original_text','vectors':self.vectors}),encoding='utf8')

    def evidence_blocks(self, row, include_original=True):
        """Keep each visual's title, units, legend and labels together for reading.

        This is source formatting, not query answering: no benchmark/gold input.
        Retrieval vectors continue to cover the unchanged full source corpus.
        """
        page=self.pages[row['page']]
        visual=page.get('visual',{})
        blocks=[]
        for region in visual.get('regions',[]):
            heading=(f"{region.get('kind','region').upper()}: {region.get('title','')}\n"
                     f"Location on original page: {region.get('location','unspecified')}")
            labels=region.get('labels',[])
            if isinstance(labels,str):labels=[labels]
            conditions=region.get('uncertainty','')
            body='\n'.join('- '+str(label) for label in labels)
            if conditions:body+='\nSource interpretation limits: '+str(conditions)
            # Repeat the region identity when a long region needs segmentation.
            for part in textwrap.wrap(readable(body),width=1400,break_long_words=False,
                                      break_on_hyphens=False,replace_whitespace=False):
                blocks.append(readable(heading)+'\n'+part)
        summary='\n'.join(str(visual[k]) for k in ('summary','visual_description') if visual.get(k))
        if summary:blocks.append('Page description: '+readable(summary))
        if not include_original:return blocks
        # Original text is retained, with neighboring rows rather than isolated cells.
        original=page.get('text',row['text'])
        lines=[line.strip() for line in readable(original).splitlines() if line.strip()]
        for i in range(0,len(lines),4):
            block='\n'.join(lines[max(0,i-1):i+5])
            blocks.extend(textwrap.wrap(block,width=1400,break_long_words=False,
                                         break_on_hyphens=False,replace_whitespace=False))
        return blocks

    def search_evidence(self,query,top_k=3):
        query_terms=set(terms(query))
        lexical=[]
        for row,c,length in zip(self.rows,self.counts,self.lengths):
            score=sum(self.idf.get(t,0)*c[t]*2.5/(c[t]+1.5*(.25+.75*length/self.mean_length)) for t in query_terms if c[t])
            lexical.append({**row,'score':score})
        lexical.sort(key=lambda r:(-r['score'],r['id']))
        q=self.embedder.embed_query(query)
        norm=lambda v:math.sqrt(sum(x*x for x in v))
        dense=[{**row,'score':sum(a*b for a,b in zip(q,v))/(norm(q)*norm(v))} for row,v in zip(self.rows,self.vectors)]
        dense.sort(key=lambda r:(-r['score'],r['id']))
        ranked=rrf(dense[:20],lexical[:20],limit=40)
        # Prefer a source modality only when it is explicitly named in the query.
        # Mixed-modality queries retain ordinary fusion, supporting multi-hop.
        modalities=[({'chart','charts','graph','graphs'},{'chart'}),
                    ({'photo','photos','photograph','photographs'},{'photo','photograph','portrait','photo_strip','image_grid'}),
                    ({'diagram','diagrams'},{'diagram','map','graphic'}),
                    ({'table','tables'},{'table'})]
        requested=[kinds for words,kinds in modalities if query_terms & words]
        preference=None
        if len(requested)==1:
            kinds=requested[0]
            def matches(row):
                return any(r.get('kind') in kinds for r in self.pages[row['page']].get('visual',{}).get('regions',[]))
            matching=[row for row in ranked if matches(row)]
            preference={'method':'query_modality_preference','preferred_kinds':sorted(kinds),
                        'origin':'Explicit words in the model search query; never benchmark labels',
                        'matching_candidate_count':len(matching),'policy':'Prefer matching candidates; retain others as fallback'}
            ranked=matching+[row for row in ranked if not matches(row)]
        ranked=ranked[:top_k]
        # Select whole source regions so column/legend conditions stay with values.
        excerpts=[]
        for row in ranked:
            blocks=self.evidence_blocks(row)
            content_terms=query_terms-EXCERPT_STOPWORDS
            def relevance(block):
                score=sum(self.idf.get(t,0) for t in content_terms & set(terms(block)))
                if '\nLocation on original page:' in block:
                    # A matching source heading identifies the requested table or
                    # figure better than repeated surrounding narrative does.
                    title=block.splitlines()[0].partition(':')[2]
                    score+=SOURCE_TITLE_WEIGHT*sum(self.idf.get(t,0) for t in content_terms & set(terms(title)))
                return score
            order=sorted(range(len(blocks)),key=lambda i:-relevance(blocks[i]))
            chosen=[];remaining=1800
            for i in order:
                if len(blocks[i])+2<=remaining:
                    chosen.append(i);remaining-=len(blocks[i])+2
            excerpt='\n\n'.join(blocks[i] for i in chosen)
            excerpts.append({**row,'text':excerpt,'metadata':{**row['metadata'],
                'excerpt_policy':'Content-term IDF overlap plus source-title weight 3 over complete visual regions and neighboring original rows; original image retained for verification'}})
        # Full selected snippets are above. Intermediate rankings need IDs/scores,
        # not 40 duplicate full pages in every function trace.
        def candidates(rows):
            return [{'rank':i+1,**{k:row[k] for k in ('id','page','source','score')}} for i,row in enumerate(rows)]
        steps=[{'method':'dense','query':query,'results':candidates(dense[:20])},
               {'method':'bm25','query':query,'results':candidates(lexical[:20])},{'method':'rrf','k':60}]
        if preference:steps.append(preference)
        return {'results':excerpts,'steps':steps}

    def inspect_page(self,page):
        row=next(r for r in self.rows if r['page']==page)
        text=('\n\n'.join(self.evidence_blocks(row,include_original=False))+
              '\n\nOriginal page text:\n'+readable(self.pages[page].get('text',row['text'])))
        parts=textwrap.wrap(readable(text),width=1800,break_long_words=False,break_on_hyphens=False)
        full=[{**row,'id':row['id']+'-full'+(f'-{i+1}' if i else ''),'text':part,
               'metadata':{**row['metadata'],'evidence_type':'full_original_page_text_and_visual_catalog',
                           'segment':i+1,'segments_total':len(parts)}} for i,part in enumerate(parts)]
        return {'results':full,'steps':[{'method':'original_page','page':page,'page_image':row['metadata']['page_image'],
                                        'segments_total':len(parts),'budget_policy':'Return all segments; model context retains those fitting its explicit budget.'}]}


def calculate(expression):
    operators={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left),evaluate(node.right))
        raise ValueError('Only numeric literals and +,-,*,/ are allowed')
    if not isinstance(expression,str) or len(expression)>200:raise ValueError('Invalid expression')
    value=evaluate(ast.parse(expression,mode='eval').body)
    if not math.isfinite(value):raise ValueError('Nonfinite result')
    return value


SCHEMAS=[{'type':'function','function':{'name':'search_evidence','description':'Search all NASA report pages, including visual descriptions, text and table labels.',
    'parameters':{'type':'object','properties':{'query':{'type':'string'},'top_k':{'type':'integer','minimum':1,'maximum':8}},'required':['query']}}},
    {'type':'function','function':{'name':'inspect_page','description':'Read a retrieved original physical PDF page for detailed evidence.',
    'parameters':{'type':'object','properties':{'page':{'type':'integer','minimum':1,'maximum':127}},'required':['page']}}},
    {'type':'function','function':{'name':'calculate','description':'Compute arithmetic from source numbers already retrieved. Cite the source pages, not this computation alone.',
    'parameters':{'type':'object','properties':{'expression':{'type':'string'}},'required':['expression']}}}]


def dispatch(tools,name,arguments):
    if not isinstance(arguments,dict):raise ValueError('Arguments must be an object')
    if name=='search_evidence' and not(set(arguments)-{'query','top_k'}):
        q=arguments.get('query');k=arguments.get('top_k',3)
        if isinstance(q,str) and q.strip() and type(k) is int and 1<=k<=8:return tools.search_evidence(q,k)
    if name=='inspect_page' and set(arguments)=={'page'} and type(arguments['page']) is int and arguments['page'] in tools.pages:
        return tools.inspect_page(arguments['page'])
    if name=='calculate' and set(arguments)=={'expression'}:
        result=calculate(arguments['expression'])
        return {'results':[],'steps':[{'method':'arithmetic','expression':arguments['expression'],'result':result}],
                'message':f'Arithmetic result: {result}. Cite original source evidence for input numbers.'}
    raise ValueError('Unknown function or invalid arguments')
