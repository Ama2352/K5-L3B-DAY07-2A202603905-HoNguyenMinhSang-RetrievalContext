"""Source-linked layout search for the advanced PDF; images remain unverified."""
import json
import math
from pathlib import Path


class AdvancedLayoutTools:
    def __init__(self, embedder, data=Path('data/advanced/generated')):
        self.embedder = embedder
        self.data = data
        self.rows = json.loads((data / 'layout_vectors.json').read_text(encoding='utf8'))
        self.pages = {p['page']: p for p in json.loads((data / 'layout_pages.json').read_text(encoding='utf8'))}

    def search_layout(self, query, top_k=3):
        vector = self.embedder.embed_query(query)
        norm = math.sqrt(sum(x*x for x in vector))
        results = []
        for row in self.rows:
            v = row['embedding']
            score = sum(a*b for a, b in zip(vector, v)) / (norm * math.sqrt(sum(x*x for x in v)))
            results.append({'id': row['id'], 'text': row['text'], 'source_id': 'nasa_fy2024_afr',
                            'source': row['source'], 'page': row['page'], 'score': score,
                            'metadata': {'bbox_points': row['bbox'], 'block_boxes': row.get('block_boxes', []), 'page_image': row['page_image'],
                                         'type': row['type']}})
        results.sort(key=lambda x: (-x['score'], x['id']))
        return {'results': results[:top_k], 'steps': []}

    def inspect_page(self, page):
        record = self.pages[page]
        chunks = []
        for i, block in enumerate(record['blocks']):
            if block['type'] == 'text':
                chunks.append({'id': f'page-{page:03d}-block-{i:03d}', 'text': block['text'],
                    'source_id': 'nasa_fy2024_afr', 'source': record['source'], 'page': page,
                    'metadata': {'bbox_points': block['bbox'], 'page_image': record['page_image'],
                                 'type': 'text_block'}})
        return {'results': chunks, 'steps': [{'method': 'page_layout', 'page': page,
            'image_regions': [b for b in record['blocks'] if b['type'] == 'image'],
            'table_candidates': record['tables'], 'page_image': record['page_image'],
            'warning': 'Table cells and visual content are not automatically verified.'}]}


ADVANCED_SCHEMAS = [
    {'type': 'function', 'function': {'name': 'search_layout',
        'description': 'Search indexed PDF text blocks with physical-page and bounding-box provenance.',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'},
            'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 10}}, 'required': ['query']}}},
    {'type': 'function', 'function': {'name': 'inspect_page',
        'description': 'Inspect extracted text and visual-region metadata for one retrieved physical PDF page.',
        'parameters': {'type': 'object', 'properties': {'page': {'type': 'integer', 'minimum': 1, 'maximum': 127}},
                       'required': ['page']}}}]


def execute_advanced_tool(tools, name, arguments):
    if not isinstance(arguments, dict):
        raise ValueError('Arguments must be an object')
    if name == 'search_layout' and not (set(arguments) - {'query', 'top_k'}):
        query = arguments.get('query'); k = arguments.get('top_k', 3)
        if isinstance(query, str) and query.strip() and isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= 10:
            return tools.search_layout(query, k)
    if name == 'inspect_page' and set(arguments) == {'page'}:
        page = arguments['page']
        if isinstance(page, int) and not isinstance(page, bool) and page in tools.pages:
            return tools.inspect_page(page)
    raise ValueError('Unknown function or invalid arguments')
