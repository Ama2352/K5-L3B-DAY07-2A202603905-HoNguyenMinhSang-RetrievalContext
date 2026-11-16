import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.advanced.evaluate_multimodal import freeze, image_provider, runtime_dispatch, summarize
from src.agents.function_agent import run
from src.advanced.multimodal_tools import MultimodalTools, SCHEMAS


class AdvancedEvaluationTests(unittest.TestCase):
    def test_arithmetic_reaches_actual_model_tool_message_as_derived_only(self):
        requests = []

        def model(model, messages, schemas, timeout, options=None):
            requests.append(json.loads(json.dumps(messages)))
            if len(requests) == 1:
                message = {'role': 'assistant', 'content': '', 'tool_calls': [
                    {'function': {'name': 'calculate', 'arguments': {'expression': '12-5'}}}]}
            else:
                message = {'role': 'assistant', 'content': 'Insufficient evidence for source inputs.'}
            return {'message': message}, 0.1

        with patch('src.agents.function_agent.call_model', side_effect=model):
            trace = run('Compute from source evidence', 'test', object(),
                        schemas=[{'type': 'function', 'function': {'name': 'calculate'}}],
                        dispatcher=runtime_dispatch)
        delivered = next(m['content'] for m in requests[1] if m['role'] == 'tool')
        self.assertIn('12-5 = 7', delivered)
        self.assertIn('DERIVED COMPUTATION ONLY', delivered)
        self.assertFalse(trace['tool_calls'][0]['results'][0]['metadata']['is_source_evidence'])
        self.assertEqual(trace['tool_call_count'], 1)

    def test_inspect_adds_full_evidence_after_a_search_excerpt(self):
        row={'id':'nasa-p001','page':1,'source':'public-report.pdf','text':'Complete source evidence with the value 7.',
             'metadata':{'page_image':'rendered/page-001.png'}}
        tools=MultimodalTools.__new__(MultimodalTools)
        tools.rows=[row];tools.pages={1:row}
        tools.search_evidence=lambda query,top_k=3:{'results':[{**row,'text':'Short excerpt without the value.'}],'steps':[]}
        messages=[
            ({'message':{'role':'assistant','tool_calls':[{'function':{'name':'search_evidence','arguments':{'query':'value'}}}]}},.1),
            ({'message':{'role':'assistant','tool_calls':[{'function':{'name':'inspect_page','arguments':{'page':1}}}]}},.1),
            ({'message':{'role':'assistant','content':'The value is 7 [C2].'}},.1)]
        with patch('src.agents.function_agent.call_model',side_effect=messages):
            trace=run('What value?','test',tools,schemas=SCHEMAS,dispatcher=runtime_dispatch)
        self.assertEqual(trace['status'],'answered')
        self.assertIn('Complete source evidence',trace['context'])
        self.assertEqual(trace['citation_map']['C2']['id'],'nasa-p001-full')
        self.assertEqual(trace['tool_call_count'],2)

    def test_images_skip_computation_and_limit_distinct_original_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'rendered').mkdir()
            for page in range(1, 7):
                (root / 'rendered' / f'page-{page:03d}.png').write_bytes(b'original-page')
            provide = image_provider(root)
            images = provide([{'page': None}] + [{'page': p} for p in range(1, 7)])
            self.assertEqual([i['page'] for i in images], [1, 2, 3, 4])
            self.assertEqual(provide([{'page': 1}, {'page': 5}]), [])

    def test_freeze_rejects_missing_reviews_then_locks_complete_dev(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = [f'dev-{i}' for i in range(18)]
            dev = {'fingerprint': {'version': 'fixed'}, 'query_ids': ids,
                   'traces': [{'query_id': i} for i in ids], 'reviews': []}
            path = root / 'dev.json'
            path.write_text(json.dumps(dev), encoding='utf8')
            with patch('src.advanced.evaluate_multimodal.fingerprint', return_value={'version': 'fixed'}):
                with self.assertRaises(ValueError):
                    freeze(root, root)
                dev['reviews'] = [{'query_id': i, 'passed': False, 'reason': 'Source checked; incorrect.'} for i in ids]
                path.write_text(json.dumps(dev), encoding='utf8')
                freeze(root, root)
                self.assertTrue(json.loads((root / 'selection.json').read_text())['frozen'])
                with self.assertRaises(FileExistsError):
                    freeze(root, root)

    def test_summary_never_equates_answered_status_with_pass(self):
        document = {'query_ids': ['q1'], 'traces': [{'status': 'answered', 'tool_calls': []}], 'reviews': []}
        self.assertIsNone(summarize(document)['pass_rate'])


if __name__ == '__main__':
    unittest.main()
