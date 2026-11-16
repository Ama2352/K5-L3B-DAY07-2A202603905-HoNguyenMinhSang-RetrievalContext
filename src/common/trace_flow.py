"""Add a clearly marked chronological view to already-recorded function traces."""
from collections import Counter


def add_reconstructed_flow(trace):
    if 'flow' in trace:
        return trace
    flow = []
    calls = trace.get('tool_calls', [])
    next_tool = 0
    for number, model_call in enumerate(trace.get('model_calls', []), 1):
        message = model_call.get('message', {})
        requested = message.get('tool_calls') or []
        flow.append({'step': len(flow) + 1, 'event': 'model_response', 'model_call_number': number,
                     'requested_functions': [c.get('function', {}).get('name') for c in requested],
                     'content': message.get('content', ''), 'elapsed_seconds': model_call.get('elapsed_seconds')})
        for request in requested:
            name = request.get('function', {}).get('name')
            if next_tool < len(calls) and calls[next_tool].get('name') == name:
                item = calls[next_tool]
                flow.append({'step': len(flow) + 1, 'event': 'tool_call_finished',
                    'tool_call_number': next_tool + 1, 'function': name,
                    'arguments': item.get('arguments'), 'status': item.get('status'),
                    'result_count': len(item.get('results', [])), 'context_ids': item.get('context_ids', []),
                    'error': item.get('error'), 'elapsed_seconds': item.get('elapsed_seconds')})
                next_tool += 1
            else:
                flow.append({'step': len(flow) + 1, 'event': 'requested_tool_not_recorded', 'function': name})
    attempted = Counter(c.get('name') or 'unknown' for c in calls)
    success = Counter(c.get('name') or 'unknown' for c in calls if c.get('status') == 'ok')
    counts = {n: {'attempted': k, 'succeeded': success[n], 'failed': k-success[n]} for n, k in attempted.items()}
    if trace.get('status') == 'error':
        flow.append({'step': len(flow) + 1, 'event': 'error', 'error': trace.get('error')})
    flow.append({'step': len(flow) + 1, 'event': 'final', 'status': trace.get('status'),
                 'answer': trace.get('answer'), 'citations': trace.get('citations', []), 'tool_counts': counts})
    result = dict(trace)
    result['original_display_answer'] = trace.get('display_answer', '')
    result.update(flow=flow, flow_origin='reconstructed from recorded model responses and tool calls',
                  tool_counts=counts, tool_call_count=len(calls), filter=None, filter_origin='No filter')
    names = ', '.join(f'{n}: {k} call(s), {success[n]} succeeded' for n, k in attempted.items()) or 'none'
    result['display_answer'] = f"{result.get('answer', '')}\n\nTools used: {names}"
    return result
