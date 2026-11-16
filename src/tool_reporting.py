"""Describe observed retrieval calls without asking the model to invent tool use."""


def summarize_tools(events, method, status):
    searches = sum(event.get('event') == 'search' for event in events)
    descriptions = {
        'dense': ['BGE query embedding', 'Day 6 pgvector dense search'],
        'hybrid': ['BGE query embedding', 'Day 6 pgvector dense search',
                   'TF-IDF lexical search', 'reciprocal rank fusion'],
        'advanced': ['BGE query embedding', 'advanced text index search'],
    }
    if searches and method not in descriptions:
        raise ValueError(f'Unknown retrieval method: {method}')
    tools = descriptions[method] if searches else []
    # The starter loop can lose already-completed events if a later call raises.
    if status == 'error':
        return {'status': 'incomplete', 'observed_searches': searches, 'tools': tools,
                'note': 'The run failed; its recorded events may be incomplete.'}
    return {'status': 'complete', 'observed_searches': searches, 'tools': tools}
