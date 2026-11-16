import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.function_agent import run, is_explicit_abstention
from src.retrieval.retrieval_tools import TOOL_SCHEMAS, RetrievalTools, execute_tool
from src.common.trace_flow import add_reconstructed_flow

CID = 'a' * 32
CHUNK = {'id': CID, 'text': 'Evidence.', 'source_id': 'doc', 'source': 'doc.pdf',
         'page': 4, 'metadata': {}, 'score': .5}


class FakeTools:
    def search_bm25(self, query, top_k):
        return [CHUNK]


class FunctionToolTests(unittest.TestCase):
    def test_plural_sources_refusal_is_abstention(self):
        self.assertTrue(is_explicit_abstention('The sources do not provide carbon-footprint details.'))
        self.assertFalse(is_explicit_abstention('The source provides a carbon-footprint figure.'))

    def test_bm25_and_multistage_use_real_corpus_content(self):
        other = {**CHUNK, 'id': 'b' * 32, 'text': 'Unrelated apples and oranges.'}
        relevant = {**CHUNK, 'text': 'Tokenizer vocabulary sentencepiece tokenizer vocabulary.'}
        class FakeDense:
            def __init__(self, *_): pass
            def search(self, query, *, filters, top_k):
                return [{**relevant, 'score': .8}, {**other, 'score': .2}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'chunks.json'
            path.write_text(json.dumps([other, relevant]), encoding='utf8')
            with patch('src.retrieval.retrieval_tools.DenseRetriever', FakeDense):
                tools = RetrievalTools(Path(folder), 'unused', path)
            self.assertEqual(tools.search_bm25('tokenizer vocabulary')[0]['id'], CID)
            result = execute_tool(tools, 'search_multistage', {'query': 'tokenizer'})
            self.assertEqual(result['results'][0]['id'], CID)
            self.assertEqual(result['steps'][0]['method'], 'bm25')
            self.assertEqual(result['steps'][1]['method'], 'dense')

    def test_schema_and_argument_validation(self):
        self.assertIn('search_bm25', [x['function']['name'] for x in TOOL_SCHEMAS])
        self.assertEqual(execute_tool(FakeTools(), 'search_bm25', {'query': 'evidence'})['results'], [CHUNK])
        for name, args in [('unknown', {'query': 'x'}), ('search_bm25', {'query': 'x', 'top_k': 100}),
                           ('search_bm25', {'query': 'x', 'gold': 'secret'})]:
            with self.assertRaises(ValueError):
                execute_tool(FakeTools(), name, args)

    def test_real_tool_call_protocol_and_visible_tool_name(self):
        messages = [({'message': {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_bm25', 'arguments': {'query': 'evidence', 'top_k': 3}}}]}}, .1),
            ({'message': {'role': 'assistant', 'content': 'Evidence [C1]'}}, .1)]
        with patch('src.agents.function_agent.call_model', side_effect=messages):
            trace = run('question', 'test-model', FakeTools())
        self.assertEqual(trace['status'], 'answered')
        self.assertEqual(trace['citations'][0]['page'], 4)
        self.assertIn('search_bm25', trace['display_answer'])
        self.assertEqual(trace['tool_calls'][0]['arguments']['query'], 'evidence')
        self.assertEqual(trace['tool_calls'][0]['returned_count'], 1)
        self.assertEqual(trace['tool_counts']['search_bm25'], {'attempted': 1, 'succeeded': 1, 'failed': 0})
        self.assertEqual([s['event'] for s in trace['flow']],
                         ['model_request', 'model_response', 'tool_call_started', 'tool_call_finished',
                          'model_request', 'model_response', 'final'])
        self.assertIn('[C1] Source:', trace['context'])
        self.assertEqual(trace['citations'][0]['id'], CID)

    def test_model_failure_preserves_prior_tool_results(self):
        first = ({'message': {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_bm25', 'arguments': {'query': 'evidence'}}}]}}, .1)
        with patch('src.agents.function_agent.call_model', side_effect=[first, TimeoutError('timeout')]):
            trace = run('question', 'test-model', FakeTools())
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['tool_calls'][0]['status'], 'ok')
        self.assertIn('[C1] Source:', trace['context'])
        self.assertEqual(trace['flow'][-2]['event'], 'error')
        self.assertEqual(trace['model_request_count'], 2)
        self.assertEqual(trace['model_call_count'], 1)
        self.assertEqual(trace['tool_calls'][0]['call_number'], 1)
        self.assertEqual(trace['tool_calls'][0]['call_id'], 'tool-1')

    def test_explicit_abstention_is_preserved_without_forcing_citation(self):
        messages = [({'message': {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_bm25', 'arguments': {'query': 'evidence'}}}]}}, .1),
            ({'message': {'role': 'assistant', 'content': 'Insufficient evidence to determine the cost.'}}, .1)]
        with patch('src.agents.function_agent.call_model', side_effect=messages) as model:
            trace = run('question', 'test-model', FakeTools())
        self.assertEqual(trace['status'], 'abstain')
        self.assertEqual(trace['answer'], trace['raw_answer'])
        self.assertEqual(model.call_count, 2)

    def test_unoffered_function_is_not_executed(self):
        messages = [({'message': {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_bm25', 'arguments': {'query': 'evidence'}}}]}}, .1),
            ({'message': {'role': 'assistant', 'content': 'Insufficient evidence.'}}, .1)]
        with patch('src.agents.function_agent.call_model', side_effect=messages):
            trace = run('question', 'test-model', FakeTools(), schemas=[])
        self.assertEqual(trace['status'], 'error')
        self.assertIn('not offered', trace['tool_calls'][0]['error'])

    def test_existing_trace_flow_counts_only_recorded_calls(self):
        trace = {'status': 'answered', 'answer': 'Evidence', 'citations': [],
                 'model_calls': [{'message': {'tool_calls': [{'function': {'name': 'search_bm25'}}]}}],
                 'tool_calls': [{'name': 'search_bm25', 'arguments': {'query': 'evidence'},
                                 'status': 'ok', 'results': [CHUNK]}]}
        result = add_reconstructed_flow(trace)
        self.assertEqual(result['flow'][1]['function'], 'search_bm25')
        self.assertEqual(result['tool_counts']['search_bm25']['attempted'], 1)
        self.assertIn('reconstructed', result['flow_origin'])

    def test_failed_only_tool_call_is_error_not_successful_abstention(self):
        messages = [({'message': {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'search_bm25', 'arguments': {'query': 'evidence', 'top_k': 50}}}]}}, .1),
            ({'message': {'role': 'assistant', 'content': 'Insufficient evidence.'}}, .1)]
        with patch('src.agents.function_agent.call_model', side_effect=messages):
            trace = run('question', 'test-model', FakeTools())
        self.assertEqual(trace['status'], 'error')
        self.assertEqual(trace['tool_counts']['search_bm25']['failed'], 1)

    def test_visual_context_is_sent_but_image_data_is_not_saved(self):
        calls = []
        def model(_model, messages, _tools, _timeout):
            calls.append(list(messages))
            if len(calls) == 1:
                return {'message': {'role': 'assistant', 'tool_calls': [{'function': {
                    'name': 'search_bm25', 'arguments': {'query': 'evidence'}}}]}}, .1
            return {'message': {'role': 'assistant', 'content': 'Evidence [C1]'}}, .1
        def provider(_results):
            return [{'page': 4, 'path': 'rendered/page.png', 'data': 'IMAGE_PAYLOAD_TEST'}]
        with patch('src.agents.function_agent.call_model', side_effect=model):
            trace = run('question', 'test-model', FakeTools(), image_provider=provider)
        self.assertEqual(trace['status'], 'answered')
        self.assertEqual(calls[1][-1]['images'], ['IMAGE_PAYLOAD_TEST'])
        self.assertNotIn('IMAGE_PAYLOAD_TEST', str(trace))
        self.assertEqual(trace['visual_context'][0]['page'], 4)


if __name__ == '__main__':
    unittest.main()
