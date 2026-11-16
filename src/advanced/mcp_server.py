"""Read-only legacy MCP stdio tools over the public NASA source catalog.

Only source retrieval, original-page inspection and safe arithmetic are exposed.
No benchmark or credential file is opened by this server.
"""
import argparse
import base64
import json
import sys
import time
import copy
from pathlib import Path
from src.advanced.multimodal_tools import MultimodalTools,SCHEMAS
from src.advanced.evaluate_multimodal import runtime_dispatch
from src.common.embedding_worker import ProcessEmbedder
from src.common.secret_safety import redact

TOOL_SCHEMAS=copy.deepcopy(SCHEMAS)
TOOL_SCHEMAS[0]['function']['parameters']['properties']['top_k']['maximum']=8
TOOL_SCHEMAS[0]['function']['description']+=' Use top_k=3 by default, up to8 for broad multi-source queries.'


class SourceSession:
    def __init__(self,tools,data,trace,max_calls=8,budget=24000):
        self.tools=tools;self.data=Path(data);self.path=Path(trace)
        self.max_calls=max_calls;self.remaining=budget;self.labels={}
        self.trace={'protocol':'native MCP tools/call','tool_calls':[],'citation_map':{},'context':'','images':[]}

    def save(self):
        self.trace['tool_call_count']=len(self.trace['tool_calls'])
        self.trace['citation_map']={k:{'id':v['id'],'source':v['source'],'page':v['page'],'metadata':v.get('metadata',{})}
                                    for k,v in self.labels.items()}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(redact(self.trace),ensure_ascii=False,indent=2),encoding='utf8');temp.replace(self.path)

    def call(self,name,arguments,request_id):
        start=time.perf_counter()
        event={'name':name,'arguments':arguments,'call_id':f'mcp-{request_id}',
               'call_number':len(self.trace['tool_calls'])+1,'status':'error','results':[],'steps':[]}
        self.trace['tool_calls'].append(event)
        content=[]
        try:
            if len(self.trace['tool_calls'])>self.max_calls:raise ValueError('Maximum8toolcalls; answer or abstain from current evidence')
            if name=='search_evidence':
                if not isinstance(arguments,dict) or set(arguments)-{'query','top_k'}:
                    raise ValueError('Expected query and optional top_k')
                query=arguments.get('query');k=arguments.get('top_k',3)
                if not isinstance(query,str) or not query.strip() or type(k) is not int or not 1<=k<=8:
                    raise ValueError('query must be nonempty, top_k an integer1..8')
                output=self.tools.search_evidence(query,k)
            else:output=runtime_dispatch(self.tools,name,arguments)
            blocks=[];ids=[]
            for row in output['results']:
                old=next((label for label,value in self.labels.items() if value['id']==row['id']),None)
                label=old or f'C{len(self.labels)+1}'
                block=f'[{label}] Source: {row["source"]} | physical PDF page {row["page"]}\n{row["text"]}'
                if len(block)<=self.remaining:
                    self.labels[label]=row;blocks.append(block);ids.append(row['id']);self.remaining-=len(block)
            text='\n\n'.join(blocks) or 'No additional evidence fits the declared budget. Use existing source labels or abstain.'
            content.append({'type':'text','text':text})
            self.trace['context']+='\n\n'+text
            if name=='inspect_page':
                page=arguments['page'];path=self.data/'rendered'/f'page-{page:03d}.png'
                content.append({'type':'image','mimeType':'image/png','data':base64.b64encode(path.read_bytes()).decode('ascii')})
                self.trace['images'].append({'page':page,'path':str(path),'delivered_via':'native MCP image content'})
            event.update(status='ok',results=output['results'],steps=output['steps'],context=text,context_ids=ids)
        except Exception as exc:
            message=f'{type(exc).__name__}: {exc}'
            event['error']=message
            content=[{'type':'text','text':message+'; allowed functions: search_evidence(query,top_k1..8), inspect_page(page1..127), calculate(expression).'}]
        event['elapsed_seconds']=time.perf_counter()-start
        self.save()
        return {'content':redact(content),'isError':event['status']=='error'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--day6',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    args=p.parse_args()
    session=None
    for line in sys.stdin:
        request=json.loads(line);identifier=request.get('id');method=request.get('method');params=request.get('params',{})
        if identifier is None:continue
        try:
            if method=='initialize':
                result={'protocolVersion':'2025-06-18','capabilities':{'tools':{'listChanged':False}},
                        'serverInfo':{'name':'lab7-public-source-retrieval','version':'1.0.0'}}
            elif method=='ping':result={}
            elif method=='tools/list':
                result={'tools':[{'name':s['function']['name'],'description':s['function']['description'],
                    'inputSchema':s['function']['parameters'],'annotations':{'readOnlyHint':True,'destructiveHint':False,
                    'idempotentHint':True,'openWorldHint':False}} for s in TOOL_SCHEMAS]}
            elif method=='tools/call':
                if session is None:session=SourceSession(MultimodalTools(ProcessEmbedder(args.day6),args.data),args.data,args.trace)
                result=session.call(params['name'],params.get('arguments',{}),identifier)
            else:raise ValueError('Unsupported MCP method')
            response={'jsonrpc':'2.0','id':identifier,'result':result}
        except Exception as exc:
            response={'jsonrpc':'2.0','id':identifier,'error':{'code':-32603,'message':redact(f'{type(exc).__name__}: {exc}')}}
        print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
