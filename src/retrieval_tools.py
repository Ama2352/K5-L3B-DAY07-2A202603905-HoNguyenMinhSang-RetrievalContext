"""Real retrieval functions and Ollama function schemas for the Day 6 corpus."""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from baseline import TfidfIndex, tokenize
from run_agent import DenseRetriever
from retrieval_dev import rrf

STOP = set('a an and are as at be by can did do does for from how in into is it of on or the to was were what when where which why with'.split())


class RetrievalTools:
    def __init__(self, day6, database_url, chunks_path=Path('data/day6_chunks.json')):
        self.chunks = json.loads(chunks_path.read_text(encoding='utf8'))
        self.dense = DenseRetriever(day6.resolve(), database_url)
        self.tfidf = TfidfIndex(self.chunks)
        self.counts = [Counter(tokenize(c['text'])) for c in self.chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(t for counts in self.counts for t in counts)
        self.idf = {t: math.log(1 + (len(self.chunks) - n + .5) / (n + .5)) for t, n in df.items()}

    def search_dense(self, query, top_k=3):
        return self.dense.search(query, filters=None, top_k=top_k)

    def search_tfidf(self, query, top_k=3):
        return self.tfidf.search(query, top_k=top_k)

    def search_bm25(self, query, top_k=3):
        terms = set(tokenize(query)); rows = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            score = sum(self.idf.get(t, 0) * counts[t] * 2.5 /
                        (counts[t] + 1.5 * (0.25 + .75 * length / self.avg_len))
                        for t in terms if counts[t])
            if score > 0:
                rows.append({**chunk, 'score': score})
        return sorted(rows, key=lambda c: (-c['score'], c['id']))[:top_k]

    def search_hybrid(self, query, top_k=3):
        dense = self.search_dense(query, 20)
        lexical = self.search_bm25(query, 20)
        return {'results': rrf(dense, lexical, limit=top_k),
                'steps': [{'method': 'dense', 'query': query, 'top_k': 20, 'results': dense},
                          {'method': 'bm25', 'query': query, 'top_k': 20, 'results': lexical},
                          {'method': 'rrf', 'k': 60}]}

    def search_rerank(self, query, top_k=3):
        # Hybrid retrieves candidates; distinct-token coverage reranks those candidates.
        hybrid = self.search_hybrid(query, 20)
        candidates = hybrid['results']
        terms = set(tokenize(query)) - STOP
        if not terms:
            return {'results': candidates[:top_k], 'steps': hybrid['steps'] +
                    [{'method': 'term_coverage', 'query_terms': [], 'results': candidates}]}
        ranked = []
        for c in candidates:
            coverage = len(terms & set(tokenize(c['text']))) / len(terms)
            ranked.append({**c, 'candidate_score': c['score'], 'score': coverage})
        ranked.sort(key=lambda c: (-c['score'], -c['candidate_score'], c['id']))
        return {'results': ranked[:top_k], 'steps': hybrid['steps'] +
                [{'method': 'term_coverage', 'query_terms': sorted(terms), 'results': ranked}]}

    def search_multistage(self, query, top_k=3):
        first = self.search_bm25(query, 3)
        if not first:
            return {'results': [], 'steps': [{'query': query, 'results': []}]}
        original = set(tokenize(query))
        terms = Counter(tokenize(first[0]['text']))
        additions = sorted((t for t in terms if len(t) > 3 and t not in STOP and t not in original),
                           key=lambda t: (-terms[t] * self.idf.get(t, 0), t))[:3]
        expanded = ' '.join([query, *additions])
        second = self.search_dense(expanded, 20)
        results = rrf(first, second, limit=top_k)
        return {'results': results, 'steps': [{'query': query, 'method': 'bm25', 'results': first},
                                             {'query': expanded, 'added_terms': additions,
                                              'method': 'dense', 'top_k': 20, 'results': second},
                                             {'method': 'rrf', 'k': 60}]}


TOOL_DESCRIPTIONS = {
    'search_dense': 'Semantic vector search in the Day 6 pgvector store.',
    'search_tfidf': 'Search document chunks by TF-IDF lexical similarity.',
    'search_bm25': 'Search document chunks using BM25 lexical ranking.',
    'search_hybrid': 'Fuse dense and BM25 rankings using reciprocal rank fusion.',
    'search_rerank': 'Retrieve hybrid candidates and rerank by query-term coverage.',
    'search_multistage': 'Search BM25, expand the query from a top result, then search dense and fuse rankings.',
}

TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': name, 'description': desc,
    'parameters': {'type': 'object', 'properties': {
        'query': {'type': 'string', 'description': 'Search query based on the user question'},
        'top_k': {'type': 'integer', 'minimum': 1, 'maximum': 10}},
        'required': ['query']}}} for name, desc in TOOL_DESCRIPTIONS.items()]


def execute_tool(tools, name, arguments):
    if name not in TOOL_DESCRIPTIONS or not isinstance(arguments, dict) or set(arguments) - {'query', 'top_k'}:
        raise ValueError('Unknown tool or invalid arguments')
    query = arguments.get('query')
    top_k = arguments.get('top_k', 3)
    if not isinstance(query, str) or not query.strip() or not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
        raise ValueError('query must be nonempty and top_k must be an integer in 1..10')
    result = getattr(tools, name)(query, top_k)
    return result if isinstance(result, dict) else {'results': result, 'steps': []}
