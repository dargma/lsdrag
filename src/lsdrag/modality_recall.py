"""Explicitly named source-document proxies; no unsupported span/image claims."""


def source_recall(prediction, gold, chunks, figures, ks=(1, 3, 5, 10)):
    def sources(item):
        values = set(item.get('document_ids', []))
        if item.get('document_id'):
            values.add(item['document_id'])
        return values
    searches = []
    for step in prediction.get('trajectory', []):
        if step['tool_name'].endswith('search') and isinstance(step['result'], list):
            ranked = []
            for hit in step['result']:
                if not isinstance(hit, dict):
                    continue
                item = chunks.get(hit.get('chunk_id'), figures.get(hit.get('figure_id'), hit))
                ranked.append(sources(item))
            searches.append(ranked)
    first = searches[0] if searches else []
    retrieved = set().union(*(docs for hits in searches for docs in hits))
    read = set().union(*(sources(chunks.get(key, {})) for key in prediction.get('read_chunk_ids', [])))
    delivered = set().union(*(sources(figures.get(key, {})) for key in prediction.get('delivered_figure_ids', [])))
    result = {}
    for modality in ('text', 'table', 'image'):
        required = {item['doc_id'] for item in gold['supporting_context'] if item['doc_part'] == modality}
        recall = lambda observed: len(required & observed)/len(required) if required else None
        result[modality] = {'required_source_count': len(required),
            'first_search_source_recall_at_k': {str(k): recall(set().union(*first[:k])) for k in ks},
            'trajectory_retrieved_source_recall': recall(retrieved),
            'explicit_read_source_recall': recall(read),
            'image_delivered_source_recall': recall(delivered) if modality == 'image' else None}
    return result
