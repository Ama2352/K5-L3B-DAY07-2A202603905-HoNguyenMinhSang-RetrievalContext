"""Regressions for merged PDF rows, note boundaries and complete two-hop output."""
import copy
import unittest
from unittest.mock import patch

from src.advanced.multimodal_tools import MultimodalTools
from src.advanced.evidence_items import EvidenceItems
from src.advanced.selection_agent import run,select_evidence


class AdvancedItemsTests(unittest.TestCase):
    def test_photo_keeps_original_person_agency_relationship(self):
        paragraph='Astronaut Ada Lin of the Solar Agency stands next to Eli Mar of the Lunar Agency.'
        def source(self,*args,**kwargs):
            self.rows=[{'id':'fixture-p1','page':1,'source':'fixture','text':paragraph,'metadata':{}}]
            self.pages={1:{'text':paragraph+'\nPhoto credit: Example Team',
              'visual':{'regions':[{'kind':'photo','title':'Astronaut crew','labels':['Ada Lin','Eli Mar']}]}}}
        with patch.object(MultimodalTools,'__init__',source):
            items=EvidenceItems(None)
        photo=next(r for r in items.items if r['metadata']['evidence_type']=='photo')
        self.assertIn(paragraph,photo['text'])

    def test_singular_currency_unit_does_not_confuse_million_cubic_feet(self):
        _,answer=select_evidence({'C1':{'text':'Total funding $913 million'}},['C1'],'913',False,'What was total funding?')
        self.assertEqual(answer,'913 million USD [C1]')
        _,answer=select_evidence({'C1':{'text':'TABLE\n- in millions\n- total net outlays 913'}},['C1'],'913',False,'What were total net outlays?')
        self.assertEqual(answer,'913 million USD [C1]')
        _,answer=select_evidence({'C1':{'text':'A balloon of 39 million cubic feet'}},['C1'],'39',False,'What was its cost?')
        self.assertNotIn('USD',answer)

    def test_rendering_uses_source_units_and_removes_local_citation_text(self):
        candidates={'C1':{'text':'Units: millions of dollars\nFY 2024 = 321; FY 2023 = 300'}}
        labels,answer=select_evidence(candidates,['C1'],'21',False,'How much did the cost increase?')
        self.assertEqual(answer,'21 million USD [C1]')
        labels,answer=select_evidence({'C1':{'text':'Three rovers in a laboratory'}},['C1'],'Three (C1)',False,'How many rovers?')
        self.assertEqual(answer,'Three [C1]')

    def parse(self,text):
        tools=MultimodalTools.__new__(MultimodalTools)
        tools.pages={1:{'text':text,'visual':{'regions':[{'kind':'table'}]}}}
        return tools.normalized_table_rows({'page':1,'text':text})

    def test_merged_rows_preserve_public_liability_scope(self):
        text=('IN MILLIONS\n2024 2023\nLiabilities:\nIntragovernmental: Accounts Payable 13 12 '
              'Total Intragovernmental 15 14\nWith the Public: Accounts Payable 47 43 '
              'Pensions and Benefits 21 19 Total with the Public 68 62')
        rows=self.parse(text)
        public=[r for r in rows if 'With the Public:' in r and 'original row Accounts Payable\n' in r]
        self.assertEqual(len(public),1)
        self.assertIn('FY 2024 = 47; FY 2023 = 43',public[0])
        self.assertNotIn('FY 2024 = 68',public[0])

    def test_receivable_columns_do_not_leak_into_next_note(self):
        text=('NOTE 4 Accounts Receivable\nIN MILLIONS\n2024 2023\nNET AMOUNT DUE\n'
              'Public 3 — 3 Public 2 — 2\nNOTE 5 Property Equipment\n2024\n'
              'Buildings 901 (500) 401')
        rows=self.parse(text)
        self.assertEqual(len(rows),1)
        self.assertIn('FY 2023: Accounts Receivable = 2',rows[0])
        self.assertNotIn('Buildings','\n'.join(rows))

    def test_two_hops_preserve_both_answers_and_remap_citations(self):
        def trace(answer,page):
            citation={'label':'C3','page':page,'id':f'p{page}','source':'fixture','metadata':{}}
            return {'status':'answered','answer':answer+' as shown in C3. [C3]','raw_answer':answer,
              'context':'[C3] '+answer,'candidate_context':'[C3] '+answer,
              'model_calls':[{'message':{'content':'fixture'}}],'model_call_count':1,'model_request_count':1,
              'tool_calls':[{'call_id':'json-function-1','name':'select_evidence','status':'ok'}],
              'citations':[citation],'citation_map':{'C3':{k:v for k,v in citation.items() if k!='label'}},'error':None}
        with patch('src.advanced.selection_agent._run_single',side_effect=[trace('Alpha',11),trace('Beta',12)]) as model:
            result=run('Give the department name and its project name.', 'fixture',object(),max_calls=2)
        self.assertEqual(model.call_count,2)
        self.assertEqual(result['answer'],'Alpha as shown in C3. [C3]\nBeta as shown in C6. [C6]')
        self.assertEqual([c['label'] for c in result['citations']],['C3','C6'])
        self.assertEqual(result['model_call_count'],2)
        self.assertEqual(result['tool_call_count'],2)
        self.assertEqual(result['tool_calls'][1]['local_to_global_citation_labels']['C3'],'C6')


if __name__=='__main__':unittest.main()
