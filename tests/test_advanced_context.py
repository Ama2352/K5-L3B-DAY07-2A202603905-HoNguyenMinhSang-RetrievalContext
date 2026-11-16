"""A precise table must survive a budget filled by rounded narrative."""
import unittest
from collections import Counter
from src.advanced.multimodal_tools import MultimodalTools,terms


class AdvancedContextTests(unittest.TestCase):
    def test_requested_table_survives_long_rounded_narrative(self):
        narrative=('The total outstanding purchase orders at the end of FY2024 were about 900 million. '+
                   'The annual financial report discusses the total outstanding purchase orders at the end of FY2024. ')*8
        visual={'regions':[{'kind':'table','title':'Outstanding Purchase Orders','location':'bottom',
            'labels':['Units: millions','2024 / 2023','Total: 912 / 891']+
                     ['Category description retained with its table values: items pending delivery.']*6}]}
        original='Annual report\n'+narrative+'\n'+narrative+'\nDetailed source table follows.'
        row={'id':'source-p001','page':1,'source':'fixture','text':original,'metadata':{}}
        tools=MultimodalTools.__new__(MultimodalTools)
        tools.pages={1:{'text':original,'visual':visual}};tools.rows=[row]
        tools.counts=[Counter(terms(original))];tools.lengths=[sum(tools.counts[0].values())]
        tools.mean_length=tools.lengths[0]
        tools.idf={t:1 for t in terms(original+' Outstanding Purchase Orders 2024')}
        tools.vectors=[[1.0]]
        tools.embedder=type('Embedder',(),{'embed_query':lambda self,query:[1.0]})()
        result=tools.search_evidence('What were total outstanding purchase orders at the end of FY2024?',1)
        text=result['results'][0]['text']
        self.assertIn('Total: 912 / 891',text)
        self.assertIn('Units: millions',text)
        self.assertLess(text.index('TABLE:'),text.index('about 900') if 'about 900' in text else len(text))


if __name__=='__main__':unittest.main()
