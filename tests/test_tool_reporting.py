import sys
import unittest
from pathlib import Path

from src.common.tool_reporting import summarize_tools


class ToolReportingTests(unittest.TestCase):
    def test_only_observed_searches_are_reported(self):
        self.assertEqual(summarize_tools([], 'dense', 'abstain')['tools'], [])
        report = summarize_tools([{'event': 'search'}, {'event': 'answer'}], 'hybrid', 'answered')
        self.assertEqual(report['observed_searches'], 1)
        self.assertIn('TF-IDF lexical search', report['tools'])
        self.assertIn('Day 6 pgvector dense search', report['tools'])

    def test_errors_do_not_claim_the_tool_trace_is_complete(self):
        report = summarize_tools([], 'dense', 'error')
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['tools'], [])


if __name__ == '__main__':
    unittest.main()
