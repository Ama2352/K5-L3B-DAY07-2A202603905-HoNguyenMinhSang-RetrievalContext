import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from advanced_layout_tools import AdvancedLayoutTools, execute_advanced_tool


class Embedder:
    def embed_query(self, query): return [1.0, 0.0]


class AdvancedLayoutToolTests(unittest.TestCase):
    def test_search_and_page_keep_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            row = {'id': 'layout-1', 'text': 'A chart caption', 'page': 1, 'bbox': [1, 2, 3, 4],
                   'source': 'doc.pdf', 'type': 'text_block', 'page_image': 'rendered/page-001.png',
                   'embedding': [1.0, 0.0]}
            page = {'page': 1, 'source': 'doc.pdf', 'page_image': 'rendered/page-001.png',
                    'blocks': [{'type': 'text', 'bbox': [1, 2, 3, 4], 'text': 'A chart caption'},
                               {'type': 'image', 'bbox': [5, 6, 7, 8]}], 'tables': []}
            (data / 'layout_vectors.json').write_text(json.dumps([row]), encoding='utf8')
            (data / 'layout_pages.json').write_text(json.dumps([page]), encoding='utf8')
            tools = AdvancedLayoutTools(Embedder(), data)
            hit = execute_advanced_tool(tools, 'search_layout', {'query': 'chart'})['results'][0]
            self.assertEqual(hit['metadata']['bbox_points'], [1, 2, 3, 4])
            opened = execute_advanced_tool(tools, 'inspect_page', {'page': 1})
            self.assertEqual(opened['steps'][0]['image_regions'][0]['bbox'], [5, 6, 7, 8])
            with self.assertRaises(ValueError):
                execute_advanced_tool(tools, 'inspect_page', {'page': 2})


if __name__ == '__main__':
    unittest.main()
